"""Causal next-hour market-entry confirmations for the frozen crypto router.

The original signal is never entered immediately in this audit.  One complete
hour is observed, a fixed structure confirmation is evaluated, and an accepted
trade enters at the following hourly market open.  Targets remain the frozen
literal 0.5/1/1.5/2% cascade and the stop remains 1%; no limit orders or
post-entry label features are used.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

import crypto_okx_swap_1h_target_cascade_oct7_audit as cascade
import crypto_stop_cause_oct11_audit as causes


OUT = causes.ROOT / "crypto_structure_confirmation_oct13.json"
VARIANTS = (
    "delayed_all",
    "favorable_close",
    "body_ge_0_25atr",
    "body_ge_0_50atr",
    "structure_break_close",
    "favorable_and_structure_break",
    "no_stop_and_favorable",
    "adverse_halfpct_reclaim_entry",
    "stop_wick_reclaim_stop",
    "stop_wick_reclaim_entry",
    "body_ge_0_25atr_and_no_stop",
)
ECONOMIC_REQUIRED_PRICE_RETURN_PCT = .973137


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def delayed_frame(frame: pd.DataFrame, candles: dict[str, dict]) -> pd.DataFrame:
    pieces = []
    for symbol, part in frame.groupby("symbol", sort=False):
        candle = candles[symbol]
        open_ = np.asarray(candle["open"], dtype=float)
        high = np.asarray(candle["high"], dtype=float)
        low = np.asarray(candle["low"], dtype=float)
        close = np.asarray(candle["close"], dtype=float)
        times = np.asarray(candle["time"], dtype=np.int64)
        found = part.copy()
        original = found.entry_index.to_numpy(dtype=np.int64)
        delayed = original + 1
        direction = found.direction.to_numpy(dtype=np.int8)
        target = found.selected_target_pct.to_numpy(dtype=float)
        observable = np.asarray([
            delayed[index] + cascade.HOLD[float(target[index])] <= len(times)
            for index in range(len(found))], dtype=bool)
        found = found.loc[observable].copy()
        original = original[observable]
        delayed = delayed[observable]
        direction = direction[observable]
        target = target[observable]
        if not len(found):
            continue
        if np.any(times[original] != found.entry_time.to_numpy(dtype=np.int64)):
            raise ValueError(f"original entry mismatch: {symbol}")
        original_entry = open_[original]
        signed_body = direction * (close[original] / open_[original] - 1)
        adverse = np.where(direction > 0,
                           low[original] / original_entry - 1,
                           1 - high[original] / original_entry)
        original_stop = original_entry * (1 - direction * .01)
        stop_touch = np.where(direction > 0,
                              low[original] <= original_stop,
                              high[original] >= original_stop)
        recovered_stop = np.where(direction > 0,
                                   close[original] > original_stop,
                                   close[original] < original_stop)
        recovered_entry = direction * (close[original] / original_entry - 1) > 0
        signal = original - 1
        structure_break = np.where(
            direction > 0, close[original] > high[signal],
            close[original] < low[signal])
        atr_fraction = found.atr14_pct.to_numpy(dtype=float)
        found["original_entry_time"] = found.entry_time.astype(np.int64)
        found["original_entry_index"] = found.entry_index.astype(np.int64)
        found["confirmation_signed_body"] = signed_body
        found["confirmation_adverse_move"] = adverse
        found["confirmation_favorable_close"] = signed_body > 0
        found["confirmation_body_ge_0_25atr"] = signed_body >= .25 * atr_fraction
        found["confirmation_body_ge_0_50atr"] = signed_body >= .50 * atr_fraction
        found["confirmation_structure_break"] = structure_break
        found["confirmation_stop_touch"] = stop_touch
        found["confirmation_recovered_stop"] = recovered_stop
        found["confirmation_recovered_entry"] = recovered_entry
        found["entry_index"] = delayed
        found["entry_time"] = times[delayed]
        stamp = pd.to_datetime(found.entry_time, unit="s", utc=True)
        found["year"] = stamp.dt.year.astype(int)

        signed = np.empty(len(found), dtype=float)
        exits = np.empty(len(found), dtype=np.int64)
        hits = np.empty(len(found), dtype=np.int8)
        reasons = np.empty(len(found), dtype=np.int8)
        for target_value in cascade.TARGETS:
            mask = target == target_value
            if not np.any(mask):
                continue
            values = cascade.resolve(
                open_, high, low, close, times,
                delayed[mask], direction[mask], target_value,
                cascade.HOLD[target_value])
            signed[mask], exits[mask], hits[mask], reasons[mask] = values
        found["signed_return"] = signed
        found["exit_time"] = exits
        found["target_hit"] = hits.astype(bool)
        found["exit_reason"] = reasons.astype(int)
        pieces.append(found)
    return pd.concat(pieces, ignore_index=True)


def mask(frame: pd.DataFrame, name: str) -> pd.Series:
    favorable = frame.confirmation_favorable_close.astype(bool)
    quarter = frame.confirmation_body_ge_0_25atr.astype(bool)
    half = frame.confirmation_body_ge_0_50atr.astype(bool)
    structure = frame.confirmation_structure_break.astype(bool)
    stop = frame.confirmation_stop_touch.astype(bool)
    recovered_stop = frame.confirmation_recovered_stop.astype(bool)
    recovered_entry = frame.confirmation_recovered_entry.astype(bool)
    values = {
        "delayed_all": pd.Series(True, index=frame.index),
        "favorable_close": favorable,
        "body_ge_0_25atr": quarter,
        "body_ge_0_50atr": half,
        "structure_break_close": structure,
        "favorable_and_structure_break": favorable & structure,
        "no_stop_and_favorable": ~stop & favorable,
        "adverse_halfpct_reclaim_entry": (
            frame.confirmation_adverse_move.le(-.005) & recovered_entry),
        "stop_wick_reclaim_stop": stop & recovered_stop,
        "stop_wick_reclaim_entry": stop & recovered_entry,
        "body_ge_0_25atr_and_no_stop": quarter & ~stop,
    }
    return values[name]


def evaluate(frame: pd.DataFrame, years: tuple[int, ...]) -> dict:
    value = causes.metrics(frame, years)
    rows = causes.accepted_frame(frame, years).to_dict("records")
    n = len(rows)
    mean_price = (100 * sum(float(row["signed_return"]) for row in rows) / n
                  if n else 0.0)
    daily = float(value.get("trades_per_calendar_day", 0))
    per_trade_equity = .03 * 10 * mean_price / 100
    projected = (100 * ((1 + per_trade_equity) ** (30 * daily) - 1)
                 if n and 1 + per_trade_equity > 0 else -100.0)
    value.update({
        "mean_signed_price_return_pct": round(mean_price, 6),
        "required_mean_price_return_pct_at_3_per_day":
            ECONOMIC_REQUIRED_PRICE_RETURN_PCT,
        "projected_30d_return_at_observed_frequency_pct": round(projected, 4),
        "economic_30pct_gate": bool(
            daily >= 3 and mean_price >= ECONOMIC_REQUIRED_PRICE_RETURN_PCT),
    })
    return value


def dominates(candidate: dict, baseline: dict, minimum_n: int) -> bool:
    return bool(
        candidate["n"] >= minimum_n
        and candidate["trades_per_calendar_day"] >= 1.5
        and candidate["return_pct"] >= baseline["return_pct"]
        and candidate["literal_target_hit_rate"] >= baseline["literal_target_hit_rate"]
        and candidate["dd_pct"] <= baseline["dd_pct"]
        and candidate["negative_months"] <= baseline["negative_months"]
        and candidate["minimum_month_pct"] >= baseline["minimum_month_pct"]
        and candidate["max_loss_streak"] <= baseline["max_loss_streak"]
    )


def objective(value: dict | None) -> dict | None:
    if value is None:
        return None
    return {
        "literal_hit_rate_75_to_95": bool(
            75 <= value["literal_target_hit_rate"] <= 95),
        "at_least_3_trades_per_day": bool(
            value["trades_per_calendar_day"] >= 3),
        "no_negative_months": bool(value["negative_months"] == 0),
        "all_months_at_least_30pct": bool(
            value["months"] > 0
            and value["months_ge_30pct"] == value["months"]),
        "drawdown_at_most_5pct": bool(value["dd_pct"] <= 5),
        "economic_30pct_gate": bool(value["economic_30pct_gate"]),
    }


def main() -> None:
    candles, sources = causes.load_candles()
    original = causes.enrich(pd.read_pickle(causes.ROWS), candles)
    delayed = delayed_frame(original, candles)
    # Later periods stay sealed until the preceding gate passes.  The original
    # router has published later results elsewhere, but this new delayed-entry
    # mechanism must not inspect them after an early failure.
    baselines = {"selection_2022": evaluate(original, (2022,))}
    delayed_baselines = {"selection_2022": evaluate(delayed, (2022,))}

    frontier = []
    eligible = []
    for name in VARIANTS:
        candidate = delayed.loc[mask(delayed, name)].copy()
        selection = evaluate(candidate, (2022,))
        passed = dominates(selection, baselines["selection_2022"], 500)
        item = {"confirmation": name, "selection_2022": selection,
                "selection_pass": passed}
        frontier.append(item)
        if passed:
            rank = (baselines["selection_2022"]["negative_months"]
                    - selection["negative_months"],
                    selection["minimum_month_pct"], selection["return_pct"],
                    selection["literal_target_hit_rate"], -selection["dd_pct"])
            eligible.append((rank, name, candidate, selection))
    frontier.sort(key=lambda item: (
        item["selection_pass"], item["selection_2022"]["minimum_month_pct"],
        item["selection_2022"]["return_pct"]), reverse=True)

    frozen = validation = test = confirmation = None
    status = "REJECTED_SELECTION"
    if eligible:
        _, name, candidate, selection = max(eligible, key=lambda item: item[0])
        baselines["validation_2023"] = evaluate(original, (2023,))
        delayed_baselines["validation_2023"] = evaluate(delayed, (2023,))
        frozen = {"confirmation": name, "selection_2022": selection,
                  "selection_baseline_2022": baselines["selection_2022"],
                  "eligible_candidates": len(eligible)}
        validation = evaluate(candidate, (2023,))
        status = "REJECTED_VALIDATION"
        if dominates(validation, baselines["validation_2023"], 400):
            baselines["test_2024"] = evaluate(original, (2024,))
            delayed_baselines["test_2024"] = evaluate(delayed, (2024,))
            test = evaluate(candidate, (2024,))
            status = "REJECTED_TEST_2024"
            if dominates(test, baselines["test_2024"], 400):
                confirmation = {str(year): evaluate(candidate, (year,))
                                for year in (2025, 2026)}
                status = ("STRUCTURE_CONFIRMATION_RESEARCH_CANDIDATE"
                          if all(all(objective(value).values())
                                 for value in confirmation.values())
                          else "REJECTED_OBJECTIVES")

    report = {
        "status": status, "ready_for_live_orders": False,
        "protocol": {
            "router_and_targets": "unchanged frozen literal target cascade",
            "selection": 2022, "validation": 2023,
            "conditional_test": 2024, "confirmation_only": [2025, 2026],
            "confirmation_bar": "the original entry hour is fully closed before the decision",
            "entry": "following hourly market open; no limit orders",
            "stop_pct": 1.0, "targets_pct": list(cascade.TARGETS),
            "hold_hours_from_delayed_entry": cascade.HOLD,
            "money": "client-selected current-equity margin; evaluated at 3% and fixed 10x gross",
            "economic_gate": (
                ">=0.973137% mean signed price return at >=3 trades/day is required "
                "to compound to +30% over 30 days"),
        },
        "confirmation_definitions": {
            "favorable_close": "confirmation hour closes in trade direction",
            "body_ge_0_25atr": "directed confirmation body is at least 0.25 of pre-signal 14h ATR",
            "body_ge_0_50atr": "directed confirmation body is at least 0.50 of pre-signal 14h ATR",
            "structure_break_close": "confirmation closes beyond the signal candle high/low",
            "no_stop_and_favorable": "original 1% stop was not touched and confirmation closes favorable",
            "adverse_halfpct_reclaim_entry": "confirmation moves at least 0.5% adverse then closes beyond original entry",
            "stop_wick_reclaim_stop": "original 1% stop is wicked then the close reclaims the stop",
            "stop_wick_reclaim_entry": "original 1% stop is wicked then the close reclaims original entry",
        },
        "observable_delayed_rows": int(len(delayed)),
        "original_baseline_by_revealed_stage": baselines,
        "unfiltered_delayed_baseline_by_revealed_stage": delayed_baselines,
        "selection_frontier": frontier,
        "frozen_before_validation": frozen,
        "validation_2023": validation, "test_2024": test,
        "confirmation_2025_2026": confirmation,
        "objective_checks": {
            "validation": objective(validation), "test": objective(test),
            "confirmation": ({year: objective(value)
                              for year, value in confirmation.items()}
                             if confirmation else None),
        },
        "sources": {**sources, str(causes.ROWS): digest(causes.ROWS),
                    str(Path(causes.__file__).name): digest(Path(causes.__file__)),
                    str(Path(cascade.__file__).name): digest(Path(cascade.__file__)),
                    str(Path(__file__)): digest(Path(__file__))},
        "limitations": [
            "The hypothesis was motivated by already inspected stop paths, so even a pass would require new forward evidence.",
            "Hourly OHLCV cannot reveal tick order inside the confirmation candle.",
            "Waiting one hour can avoid a wick but can also enter after the favorable move and reduce opportunity.",
            "Fees are excluded by user instruction.",
        ],
    }
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({
        "status": status,
        "baseline_2022": baselines["selection_2022"],
        "delayed_baseline_2022": delayed_baselines["selection_2022"],
        "selection_frontier": frontier,
        "frozen": frozen, "validation": validation, "test": test,
        "confirmation": confirmation,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
