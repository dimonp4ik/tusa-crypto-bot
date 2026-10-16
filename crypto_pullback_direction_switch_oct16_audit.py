"""Audit switching stale 168h pullbacks into short-term continuation trades.

Entry timestamps and selected 0.5-2% targets remain frozen.  For each fixed
transition hypothesis, only pullback direction is reversed and the opposite
market path is resolved from the same next-hour open.  Selection is 2022;
2023 is opened only for the single frozen winner, with later years sealed on
validation failure.
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


OUT = transition.ROOT / "crypto_pullback_direction_switch_oct16.json"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def reverse_outcomes(frame: pd.DataFrame, candles: dict) -> pd.DataFrame:
    reversed_frame = frame.copy()
    reversed_frame["direction"] = -reversed_frame.direction.astype(np.int8)
    for target in cascade.TARGETS:
        target_mask = reversed_frame.selected_target_pct.eq(target)
        for symbol, indices in reversed_frame[target_mask].groupby("symbol").groups.items():
            positions = np.asarray(list(indices), dtype=np.int64)
            part = reversed_frame.loc[positions]
            candle = candles[symbol]
            times = np.asarray(candle["time"], dtype=np.int64)
            if not np.array_equal(
                    times[part.entry_index.to_numpy(np.int64)],
                    part.entry_time.to_numpy(np.int64)):
                raise ValueError(f"entry-time mismatch: {symbol}")
            values, exits, hits, reasons = cascade.resolve(
                *[np.asarray(candle[field], dtype=float)
                  for field in ("open", "high", "low", "close")],
                times, part.entry_index.to_numpy(np.int64),
                part.direction.to_numpy(np.int8), target, cascade.HOLD[target])
            reversed_frame.loc[positions, "signed_return"] = values
            reversed_frame.loc[positions, "exit_time"] = exits
            reversed_frame.loc[positions, "target_hit"] = hits.astype(bool)
            reversed_frame.loc[positions, "exit_reason"] = reasons
    return reversed_frame


def switched(frame: pd.DataFrame, reverse: pd.DataFrame,
             risk: pd.Series) -> pd.DataFrame:
    output = frame.copy()
    flip = (frame.pullback_entry & risk).to_numpy(bool)
    for field in ("direction", "signed_return", "exit_time",
                  "target_hit", "exit_reason"):
        output.loc[flip, field] = reverse.loc[flip, field].to_numpy()
    output["direction_switched"] = flip
    return output


def switch_stats(frame: pd.DataFrame, year: int) -> dict:
    rows = transition.accepted(frame, (year,))
    switched_rows = rows.direction_switched.astype(bool)
    return {
        "raw_switches": int(frame.loc[frame.year.eq(year),
                                  "direction_switched"].sum()),
        "accepted_switches": int(switched_rows.sum()),
        "accepted_switch_target_hits": int(
            (switched_rows & rows.target_hit.astype(bool)).sum()),
        "accepted_switch_stops": int(
            (switched_rows & rows.exit_reason.eq(0)).sum()),
    }


def main() -> None:
    candles, candle_sources = cause.load_candles()
    frame = transition.enrich(pd.read_pickle(transition.SOURCE)).reset_index(drop=True)
    reverse = reverse_outcomes(frame, candles).reset_index(drop=True)
    baseline_selection = transition.metrics(frame, transition.SELECTION)
    frontier = []
    eligible = []
    cache = {}
    for name, risk in transition.candidate_risks(frame):
        candidate = switched(frame, reverse, risk)
        cache[name] = candidate
        score = transition.metrics(candidate, transition.SELECTION)
        passed = transition.dominates(score, baseline_selection, 700)
        item = {
            "switch_when": name,
            "selection_switch_stats": switch_stats(candidate, 2022),
            "selection_2022": score,
            "selection_pass": passed,
        }
        frontier.append(item)
        if passed:
            rank = (
                baseline_selection["negative_months"] - score["negative_months"],
                score["minimum_month_pct"], score["return_pct"],
                score["literal_target_hit_rate"], -score["dd_pct"])
            eligible.append((rank, name, item))

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
        _, name, selected = max(eligible, key=lambda item: item[0])
        candidate = cache[name]
        frozen = {
            "switch_when": name,
            "eligible_switches": len(eligible),
            "selection_switch_stats": selected["selection_switch_stats"],
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
                status = ("PULLBACK_SWITCH_RESEARCH_CANDIDATE"
                          if all(transition.objective_pass(value)
                                 for value in confirmation.values())
                          else "REJECTED_OBJECTIVE")

    report = {
        "status": status,
        "ready_for_live_orders": False,
        "protocol": {
            "input": "exact frozen OKX hourly target-cascade rows",
            "action": "reverse only flagged pullback direction at the same market open",
            "candidate_count": len(frontier),
            "selection": list(transition.SELECTION),
            "validation": list(transition.VALIDATION),
            "conditional_test": list(transition.TEST),
            "published_confirmation_only": list(transition.CONFIRMATION),
            "targets_pct": [.5, 1.0, 1.5, 2.0],
            "stop_pct": 1.0,
            "money": "3% current-equity margin, fixed 10x gross",
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
            str(Path(transition.__file__)): digest(Path(transition.__file__)),
            str(Path(__file__)): digest(Path(__file__)),
        },
        "limitations": [
            "Direction switching uses frozen hourly OHLCV paths, not historical order-book depth.",
            "The same fixed candidate list as the block audit prevents an after-the-fact search.",
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
