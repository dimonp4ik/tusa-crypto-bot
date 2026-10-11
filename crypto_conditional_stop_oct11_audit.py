"""Chronological audit of regime-conditional stop geometry."""
from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

import crypto_stop_cause_oct11_audit as causes
import crypto_stop_geometry_oct11_audit as geometry


OUT = causes.ROOT / "crypto_conditional_stop_oct11.json"
STOP_LEVELS = (1.25, 1.5, 2.0)
MASKS = (
    "atr",
    "supportive_ob",
    "safe_path",
    "atr_safe_path",
    "atr_or_supportive",
    "aligned_atr_safe_path",
)

BASE_POLICY = {"name": "hard_1_00", "kind": "hard", "stop_pct": 1.0}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def mask_for(frame: pd.DataFrame, name: str) -> pd.Series:
    atr = frame.stop_inside_hourly_atr.astype(bool)
    supportive = frame.supportive_ob_flag.astype(bool)
    safe = ~(frame.opposing_level_before_target.astype(bool)
             | frame.opposing_order_block.astype(bool)
             | frame.failed_directional_breakout.astype(bool))
    aligned = ~(frame.higher_timeframe_conflict.astype(bool)
                | frame.btc_direction_conflict.astype(bool)
                | frame.market_breadth_conflict.astype(bool))
    values = {
        "atr": atr,
        "supportive_ob": supportive,
        "safe_path": safe,
        "atr_safe_path": atr & safe,
        "atr_or_supportive": atr | supportive,
        "aligned_atr_safe_path": aligned & atr & safe,
    }
    return values[name]


def route(frame: pd.DataFrame, candles: dict, mask_name: str,
          active_policy: dict) -> pd.DataFrame:
    found = frame.copy()
    mask = mask_for(found, mask_name)
    prepared = {
        symbol: {key: np.asarray(candle[key], dtype=(np.int64 if key == "time" else float))
                 for key in ("open", "high", "low", "close", "time")}
        for symbol, candle in candles.items()
    }
    signed, exits, hits, reasons = [], [], [], []
    for index, row in found.iterrows():
        policy = active_policy if bool(mask.at[index]) else BASE_POLICY
        value = geometry.resolve_row(
            row.to_dict(), prepared[str(row["symbol"])], policy)
        signed.append(value[0])
        exits.append(value[1])
        hits.append(value[2])
        reasons.append(value[3])
    found["signed_return"] = signed
    found["exit_time"] = exits
    found["target_hit"] = hits
    found["exit_reason"] = reasons
    return found


def selection_pass(candidate: dict, baseline: dict) -> bool:
    return bool(
        candidate["n"] >= 700
        and candidate["trades_per_calendar_day"] >= 2.0
        and candidate["return_pct"] >= baseline["return_pct"]
        and candidate["literal_target_hit_rate"] >= baseline["literal_target_hit_rate"]
        and candidate["dd_pct"] <= baseline["dd_pct"]
        and candidate["negative_months"] <= baseline["negative_months"]
        and candidate["minimum_month_pct"] >= baseline["minimum_month_pct"]
    )


def validation_pass(candidate: dict, baseline: dict) -> bool:
    return bool(
        candidate["n"] >= 500
        and candidate["trades_per_calendar_day"] >= 2.0
        and candidate["return_pct"] >= baseline["return_pct"]
        and candidate["literal_target_hit_rate"] >= baseline["literal_target_hit_rate"]
        and candidate["dd_pct"] <= baseline["dd_pct"]
        and candidate["negative_months"] <= baseline["negative_months"]
        and candidate["minimum_month_pct"] >= baseline["minimum_month_pct"]
    )


def objective_pass(value: dict) -> bool:
    return bool(
        75 <= value["literal_target_hit_rate"] <= 95
        and value["trades_per_calendar_day"] >= 3
        and value["negative_months"] == 0
        and value["months_ge_30pct"] == value["months"]
    )


def main() -> None:
    candles, sources = causes.load_candles()
    original = pd.read_pickle(causes.ROWS)
    frame = causes.enrich(original, candles)
    baseline_2022 = causes.metrics(frame, (2022,))
    frontier = []
    eligible = []
    for mask_name, stop_pct in itertools.product(MASKS, STOP_LEVELS):
        policy = {"name": f"conditional_hard_{stop_pct}",
                  "kind": "hard", "stop_pct": stop_pct}
        candidate = route(frame, candles, mask_name, policy)
        score = causes.metrics(candidate, (2022,))
        passed = selection_pass(score, baseline_2022)
        item = {"mask": mask_name, "active_policy": policy,
                "selection_2022": score, "selection_pass": passed}
        frontier.append(item)
        if passed:
            rank = (baseline_2022["negative_months"] - score["negative_months"],
                    score["minimum_month_pct"], score["return_pct"],
                    score["literal_target_hit_rate"], -score["dd_pct"])
            eligible.append((rank, mask_name, policy, candidate, score))

    frozen = validation = test_2024 = confirmation = None
    status = "REJECTED_SELECTION"
    if eligible:
        _, mask_name, policy, candidate, selection = max(
            eligible, key=lambda item: item[0])
        baseline_2023 = causes.metrics(frame, (2023,))
        validation = causes.metrics(candidate, (2023,))
        frozen = {
            "mask": mask_name,
            "active_policy": policy,
            "eligible_candidates": len(eligible),
            "selection_baseline_2022": baseline_2022,
            "selection_2022": selection,
        }
        status = "REJECTED_VALIDATION"
        if validation_pass(validation, baseline_2023):
            baseline_2024 = causes.metrics(frame, (2024,))
            test_2024 = causes.metrics(candidate, (2024,))
            status = "REJECTED_TEST_2024"
            if validation_pass(test_2024, baseline_2024):
                confirmation = {str(year): causes.metrics(candidate, (year,))
                                for year in (2025, 2026)}
                status = ("CONDITIONAL_STOP_RESEARCH_CANDIDATE"
                          if all(objective_pass(value)
                                 for value in confirmation.values())
                          else "REJECTED_OBJECTIVE")

    report = {
        "status": status,
        "ready_for_live_orders": False,
        "protocol": {
            "entry_and_target_router": "unchanged frozen target cascade",
            "default_stop_pct": 1.0,
            "conditional_stop_pct": list(STOP_LEVELS),
            "selection": 2022,
            "validation": 2023,
            "conditional_test": 2024,
            "published_confirmation_only": [2025, 2026],
            "market_entries_only": True,
            "money": "3% current equity margin, fixed 10x gross",
            "fees": "excluded by user instruction",
        },
        "mask_definitions": {
            "atr": "14h ATR is at least the default 1% stop",
            "supportive_ob": "valid same-direction order block lies within 1.2% behind entry",
            "safe_path": "no opposing level, opposing block, or failed breakout",
            "atr_safe_path": "ATR condition and safe path",
            "atr_or_supportive": "ATR condition or supportive order block",
            "aligned_atr_safe_path": "BTC, breadth and 168h trend align; ATR condition and safe path",
        },
        "selection_baseline_2022": baseline_2022,
        "selection_frontier": frontier,
        "frozen_before_validation": frozen,
        "validation_2023": validation,
        "test_2024": test_2024,
        "confirmation_2025_2026": confirmation,
        "sources": {
            **sources,
            str(causes.ROWS): digest(causes.ROWS),
            str(Path(causes.__file__).name): digest(Path(causes.__file__)),
            str(Path(geometry.__file__).name): digest(Path(geometry.__file__)),
            str(Path(geometry.cascade.__file__).name): digest(Path(geometry.cascade.__file__)),
            str(Path(__file__)): digest(Path(__file__)),
        },
        "limitations": [
            "Conditional widening still increases loss size when the larger stop is hit.",
            "OHLCV levels and order blocks are reproducible proxies, not full depth-of-book liquidity.",
            "No historical result is eligible for runtime use without independent forward evidence.",
        ],
    }
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({
        "status": status,
        "eligible_selection_candidates": len(eligible),
        "frontier": frontier,
        "frozen_before_validation": frozen,
        "validation_2023": validation,
        "test_2024": test_2024,
        "confirmation_2025_2026": confirmation,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
