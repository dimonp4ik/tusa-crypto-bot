"""Causal one-hour confirmation audit for frozen OKX 1H pullback entries.

Pullback signals are accepted only after the next hourly candle closes.  The
entry is then moved to the following market open and the frozen target/stop is
resolved from that new open.  Other families keep their original entries.
Selection is 2022, validation is 2023, and later years remain sealed after a
failed validation gate.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

import crypto_okx_swap_1h_target_cascade_oct7_audit as cascade
import crypto_pullback_regime_transition_oct16_audit as transition
import crypto_stop_cause_oct11_audit as cause


OUT = transition.ROOT / "crypto_pullback_confirmation_oct16.json"
RULES = (
    "body_aligned", "body_025", "body_05", "strong_close",
    "break_signal_extreme", "body_volume", "break_volume",
    "rejection_body",
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def confirmation_features(frame: pd.DataFrame, candles: dict) -> pd.DataFrame:
    values = {name: np.zeros(len(frame), dtype=bool) for name in RULES}
    pullback = frame.family_pullback.fillna(0).eq(1).to_numpy(bool)
    for position in np.flatnonzero(pullback):
        row = frame.iloc[position]
        candle = candles[str(row.symbol)]
        index = int(row.entry_index)
        direction = int(row.direction)
        open_ = float(candle["open"][index])
        high = float(candle["high"][index])
        low = float(candle["low"][index])
        close = float(candle["close"][index])
        previous_high = float(candle["high"][index - 1])
        previous_low = float(candle["low"][index - 1])
        candle_range = max(high - low, 1e-15)
        body = direction * (close / open_ - 1)
        close_location = ((close - low) / candle_range if direction > 0
                          else (high - close) / candle_range)
        opposing_wick = ((min(open_, close) - low) / candle_range
                         if direction > 0 else
                         (high - max(open_, close)) / candle_range)
        prior_volume = np.asarray(
            candle["volume"][max(0, index - 168):index], dtype=float)
        median_volume = float(np.median(prior_volume)) if len(prior_volume) else 0
        volume_ratio = (float(candle["volume"][index]) / median_volume
                        if median_volume else 0)
        broke_extreme = (close > previous_high if direction > 0
                         else close < previous_low)
        values["body_aligned"][position] = body > 0
        values["body_025"][position] = body >= .0025
        values["body_05"][position] = body >= .005
        values["strong_close"][position] = body > 0 and close_location >= .70
        values["break_signal_extreme"][position] = broke_extreme
        values["body_volume"][position] = body > 0 and volume_ratio >= 1
        values["break_volume"][position] = broke_extreme and volume_ratio >= 1
        values["rejection_body"][position] = body > 0 and opposing_wick >= .25
    return pd.DataFrame(values, index=frame.index)


def delayed_pullbacks(frame: pd.DataFrame, candles: dict) -> pd.DataFrame:
    pullback = frame.family_pullback.fillna(0).eq(1)
    delayed = frame.loc[pullback].copy()
    delayed["entry_index"] = delayed.entry_index.astype(np.int64) + 1
    for symbol, indices in delayed.groupby("symbol").groups.items():
        candle = candles[symbol]
        times = np.asarray(candle["time"], dtype=np.int64)
        entry_indices = delayed.loc[indices, "entry_index"].to_numpy(np.int64)
        if np.any(entry_indices >= len(times)):
            raise ValueError(f"confirmation entry outside candle history: {symbol}")
        delayed.loc[indices, "entry_time"] = times[entry_indices]

    for target in cascade.TARGETS:
        target_mask = delayed.selected_target_pct.eq(target)
        for symbol, indices in delayed[target_mask].groupby("symbol").groups.items():
            positions = np.asarray(list(indices), dtype=np.int64)
            part = delayed.loc[positions]
            candle = candles[symbol]
            times = np.asarray(candle["time"], dtype=np.int64)
            values, exits, hits, reasons = cascade.resolve(
                *[np.asarray(candle[field], dtype=float)
                  for field in ("open", "high", "low", "close")],
                times, part.entry_index.to_numpy(np.int64),
                part.direction.to_numpy(np.int8), target, cascade.HOLD[target])
            delayed.loc[positions, "signed_return"] = values
            delayed.loc[positions, "exit_time"] = exits
            delayed.loc[positions, "target_hit"] = hits.astype(bool)
            delayed.loc[positions, "exit_reason"] = reasons

    timestamp = pd.to_datetime(delayed.entry_time, unit="s", utc=True)
    delayed["year"] = timestamp.dt.year.astype(int)
    delayed["month"] = timestamp.dt.strftime("%Y%m").astype(int)
    delayed["confirmation_delayed"] = True
    return delayed


def candidate_frame(frame: pd.DataFrame, delayed: pd.DataFrame,
                    passed: pd.Series) -> pd.DataFrame:
    pullback = frame.family_pullback.fillna(0).eq(1)
    original = frame.loc[~pullback].copy()
    original["confirmation_delayed"] = False
    confirmed = delayed.loc[passed.reindex(delayed.index, fill_value=False)].copy()
    combined = pd.concat([original, confirmed], ignore_index=True, sort=False)
    combined = combined.sort_values(
        ["entry_time", "confirmation_delayed", "quality", "symbol"],
        ascending=[True, True, False, True])
    return combined.drop_duplicates(["symbol", "entry_time"], keep="first")


def confirmation_stats(frame: pd.DataFrame, year: int) -> dict:
    rows = transition.accepted(frame, (year,))
    delayed = rows.confirmation_delayed.astype(bool)
    return {
        "accepted_n": len(rows),
        "accepted_delayed_pullbacks": int(delayed.sum()),
        "delayed_target_hits": int((delayed & rows.target_hit.astype(bool)).sum()),
        "delayed_stops": int((delayed & rows.exit_reason.eq(0)).sum()),
    }


def main() -> None:
    candles, candle_sources = cause.load_candles()
    frame = pd.read_pickle(transition.SOURCE).reset_index(drop=True)
    features = confirmation_features(frame, candles)
    delayed = delayed_pullbacks(frame, candles)
    baseline_selection = transition.metrics(frame, transition.SELECTION)
    frontier = []
    eligible = []
    cache = {}
    for rule in RULES:
        candidate = candidate_frame(frame, delayed, features[rule])
        cache[rule] = candidate
        score = transition.metrics(candidate, transition.SELECTION)
        passed = transition.dominates(score, baseline_selection, 700)
        item = {
            "confirmation": rule,
            "raw_delayed_pullbacks_2022": int(
                (features[rule] & frame.year.eq(2022)).sum()),
            "selection_confirmation_stats": confirmation_stats(candidate, 2022),
            "selection_2022": score,
            "selection_pass": passed,
        }
        frontier.append(item)
        if passed:
            rank = (
                baseline_selection["negative_months"] - score["negative_months"],
                score["minimum_month_pct"], score["return_pct"],
                score["literal_target_hit_rate"], -score["dd_pct"])
            eligible.append((rank, rule, item))

    frontier.sort(key=lambda item: (
        item["selection_pass"],
        -item["selection_2022"]["negative_months"],
        item["selection_2022"]["minimum_month_pct"],
        item["selection_2022"]["return_pct"]), reverse=True)

    frozen = validation = test = confirmation = None
    baseline_validation = baseline_test = None
    validation_passed = test_passed = None
    status = "REJECTED_SELECTION"
    if eligible:
        _, rule, selected = max(eligible, key=lambda item: item[0])
        candidate = cache[rule]
        frozen = {
            "confirmation": rule,
            "eligible_confirmations": len(eligible),
            "selection_confirmation_stats": selected["selection_confirmation_stats"],
            "selection_2022": selected["selection_2022"],
            "selection_baseline_2022": baseline_selection,
        }
        baseline_validation = transition.metrics(frame, transition.VALIDATION)
        validation = transition.metrics(candidate, transition.VALIDATION)
        validation_passed = transition.dominates(
            validation, baseline_validation, 500)
        status = "REJECTED_VALIDATION"
        if validation_passed:
            baseline_test = transition.metrics(frame, transition.TEST)
            test = transition.metrics(candidate, transition.TEST)
            test_passed = transition.dominates(test, baseline_test, 500)
            status = "REJECTED_TEST"
            if test_passed:
                confirmation = {
                    str(year): transition.metrics(candidate, (year,))
                    for year in transition.CONFIRMATION}
                status = ("PULLBACK_CONFIRMATION_RESEARCH_CANDIDATE"
                          if all(transition.objective_pass(value)
                                 for value in confirmation.values())
                          else "REJECTED_OBJECTIVE")

    report = {
        "status": status,
        "ready_for_live_orders": False,
        "protocol": {
            "input": "exact frozen OKX hourly target-cascade rows",
            "confirmation_candle": "original entry hour, closed before delayed entry",
            "entry": "following hourly market open",
            "candidate_count": len(frontier),
            "selection": list(transition.SELECTION),
            "validation": list(transition.VALIDATION),
            "conditional_test": list(transition.TEST),
            "published_confirmation_only": list(transition.CONFIRMATION),
            "targets_pct": [.5, 1.0, 1.5, 2.0],
            "stop_pct": 1.0,
            "money": "3% current-equity margin, fixed 10x gross",
            "market_orders_only": True,
            "no_fit_or_target_retuning": True,
        },
        "selection_baseline_2022": baseline_selection,
        "frontier": frontier,
        "frozen_before_validation": frozen,
        "validation_baseline_2023": baseline_validation,
        "validation_2023": validation,
        "validation_pass": validation_passed,
        "test_baseline_2024": baseline_test,
        "test_2024": test,
        "test_pass": test_passed,
        "confirmation_2025_2026": confirmation,
        "sources": {
            **candle_sources,
            str(transition.SOURCE): digest(transition.SOURCE),
            str(Path(cascade.__file__)): digest(Path(cascade.__file__)),
            str(Path(__file__)): digest(Path(__file__)),
        },
        "limitations": [
            "Confirmation uses direct hourly OHLCV, not historical order-book depth.",
            "Eight fixed interpretable rules limit multiple testing.",
            "A historical pass would still require fresh shadow outcomes before activation.",
        ],
    }
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({
        "status": status,
        "eligible": len(eligible),
        "frozen": frozen,
        "validation_baseline": baseline_validation,
        "validation": validation,
        "validation_pass": validation_passed,
        "test": test,
        "confirmation": confirmation,
        "top_frontier": frontier[:5],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
