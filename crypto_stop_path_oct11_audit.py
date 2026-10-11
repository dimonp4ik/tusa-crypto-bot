"""Post-entry path audit for stops from the frozen OKX one-hour router.

The audit diagnoses whether a recorded stop was a close-confirmed reversal,
an intrabar wick that recovered, or a conservative same-bar ambiguity.  It
does not change entries, exits, or historical results.
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

import crypto_stop_cause_oct11_audit as causes
import crypto_okx_swap_1h_target_cascade_oct7_audit as cascade


OUT = causes.ROOT / "crypto_stop_path_oct11.json"
STOP_PCT = .01


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def touched_target(high: np.ndarray, low: np.ndarray, index: int,
                   target: float, direction: int) -> bool:
    return bool(high[index] >= target if direction > 0 else low[index] <= target)


def touched_entry(high: np.ndarray, low: np.ndarray, index: int,
                  entry: float, direction: int) -> bool:
    return bool(high[index] >= entry if direction > 0 else low[index] <= entry)


def inspect_stop(row: dict, candle: dict) -> dict:
    open_ = np.asarray(candle["open"], dtype=float)
    high = np.asarray(candle["high"], dtype=float)
    low = np.asarray(candle["low"], dtype=float)
    close = np.asarray(candle["close"], dtype=float)
    times = np.asarray(candle["time"], dtype=np.int64)
    first = int(row["entry_index"])
    raw_direction = row["direction"]
    direction = (1 if raw_direction == "LONG" else -1
                 if raw_direction == "SHORT" else int(raw_direction))
    target_pct = float(row["selected_target_pct"])
    hold = int(cascade.HOLD[target_pct])
    entry = float(open_[first])
    stop = entry * (1 - direction * STOP_PCT)
    target = entry * (1 + direction * target_pct / 100)
    original_end = min(len(times), first + hold)
    extended_end = min(len(times), first + 2 * hold)
    stop_index = None
    for index in range(first, original_end):
        stop_hit = low[index] <= stop if direction > 0 else high[index] >= stop
        if stop_hit:
            stop_index = index
            break
    if stop_index is None:
        raise ValueError(f"recorded stop not reproduced: {row['symbol']} {row['entry_time']}")
    recovered_close = close[stop_index] > stop if direction > 0 else close[stop_index] < stop
    same_bar_target = touched_target(high, low, stop_index, target, direction)
    later_original = any(touched_target(high, low, index, target, direction)
                         for index in range(stop_index + 1, original_end))
    later_extended = any(touched_target(high, low, index, target, direction)
                         for index in range(stop_index + 1, extended_end))
    entry_recovered = any(touched_entry(high, low, index, entry, direction)
                          for index in range(stop_index + 1, original_end))
    adverse_extreme = (float(np.min(low[stop_index:original_end])) if direction > 0
                       else float(np.max(high[stop_index:original_end])))
    adverse_pct = (100 * (adverse_extreme / entry - 1) * direction)
    return {
        "symbol": str(row["symbol"]),
        "entry_time": int(row["entry_time"]),
        "direction": "LONG" if direction > 0 else "SHORT",
        "selected_target_pct": target_pct,
        "stop_time": int(times[stop_index]),
        "stop_on_entry_bar": stop_index == first,
        "stop_close_recovered": bool(recovered_close),
        "same_bar_target_and_stop": bool(same_bar_target),
        "target_after_stop_original_hold": bool(later_original),
        "target_after_stop_double_hold": bool(later_extended),
        "entry_recovered_original_hold": bool(entry_recovered),
        "worst_signed_price_move_pct_through_original_hold": round(adverse_pct, 6),
        "opposing_level_before_target": bool(row["opposing_level_before_target"]),
        "opposing_order_block": bool(row["opposing_order_block"]),
        "failed_directional_breakout": bool(row["failed_directional_breakout"]),
    }


def summarize(rows: list[dict]) -> dict:
    n = len(rows)
    if not n:
        return {"stops": 0}
    fields = (
        "stop_on_entry_bar", "stop_close_recovered", "same_bar_target_and_stop",
        "target_after_stop_original_hold", "target_after_stop_double_hold",
        "entry_recovered_original_hold", "opposing_level_before_target",
        "opposing_order_block", "failed_directional_breakout",
    )
    result = {"stops": n}
    for field in fields:
        count = sum(bool(row[field]) for row in rows)
        result[field] = {"count": count, "pct": round(100 * count / n, 4)}
    hard = sum(
        not row["stop_close_recovered"]
        and not row["same_bar_target_and_stop"]
        and not row["target_after_stop_double_hold"]
        for row in rows)
    recoverable = sum(
        row["stop_close_recovered"]
        or row["same_bar_target_and_stop"]
        or row["target_after_stop_original_hold"]
        for row in rows)
    result["hard_reversal_proxy"] = {
        "count": hard, "pct": round(100 * hard / n, 4)}
    result["liquidity_sweep_or_ambiguity_proxy"] = {
        "count": recoverable, "pct": round(100 * recoverable / n, 4)}
    result["by_target"] = {
        str(target): sum(float(row["selected_target_pct"]) == target for row in rows)
        for target in cascade.TARGETS
    }
    result["by_symbol"] = dict(sorted(Counter(row["symbol"] for row in rows).items()))
    result["worst_move_pct"] = {
        "median": round(float(np.median([
            row["worst_signed_price_move_pct_through_original_hold"] for row in rows])), 6),
        "p10": round(float(np.quantile([
            row["worst_signed_price_move_pct_through_original_hold"] for row in rows], .1)), 6),
    }
    return result


def main() -> None:
    candles, sources = causes.load_candles()
    frame = causes.enrich(pd.read_pickle(causes.ROWS), candles)
    by_year = {}
    examples = []
    for year in range(2022, 2027):
        accepted = causes.accepted_frame(frame, (year,))
        stopped = accepted[accepted.exit_reason.eq(0)]
        rows = [inspect_stop(row, candles[str(row["symbol"])])
                for row in stopped.to_dict("records")]
        by_year[str(year)] = summarize(rows)
        examples.extend(rows[:10])
    all_rows = []
    for year in range(2022, 2027):
        accepted = causes.accepted_frame(frame, (year,))
        for row in accepted[accepted.exit_reason.eq(0)].to_dict("records"):
            all_rows.append(inspect_stop(row, candles[str(row["symbol"])]))
    report = {
        "status": "DIAGNOSTIC_ONLY",
        "ready_for_live_orders": False,
        "protocol": {
            "entries_and_outcomes": "unchanged frozen target-cascade portfolio",
            "years": [2022, 2023, 2024, 2025, 2026],
            "stop_pct": 1.0,
            "targets_pct": list(cascade.TARGETS),
            "same_bar_policy": "recorded result remains conservative stop-first",
            "market_entries_only": True,
        },
        "all_years": summarize(all_rows),
        "by_year": by_year,
        "examples_first_10_each_year": examples,
        "sources": {
            **sources,
            str(causes.ROWS): digest(causes.ROWS),
            str(Path(causes.__file__).name): digest(Path(causes.__file__)),
            str(Path(cascade.__file__).name): digest(Path(cascade.__file__)),
            str(Path(__file__)): digest(Path(__file__)),
        },
        "limitations": [
            "One-hour OHLCV cannot reveal the tick order when target and stop occur in the same candle.",
            "A post-stop recovery does not prove a live stop should be widened or removed.",
            "The audit diagnoses paths and does not estimate slippage or fees by user instruction.",
        ],
    }
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"status": report["status"],
                      "all_years": report["all_years"],
                      "by_year": report["by_year"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
