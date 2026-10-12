"""Causal close-confirmed stops only for pre-entry sweep-prone crypto setups."""
from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

import crypto_conditional_stop_oct11_audit as conditional
import crypto_stop_cause_oct11_audit as causes
import crypto_stop_geometry_oct11_audit as geometry


OUT = causes.ROOT / "crypto_conditional_close_oct11.json"
MASKS = (
    "protective_swing",
    "supportive_ob",
    "atr_protective_swing",
    "atr_protective_or_ob",
    "aligned_atr_protective_or_ob",
    "aligned_safe_atr_protective_or_ob",
    "protective_swing_target_0_5",
    "supportive_ob_target_0_5",
    "atr_protective_swing_target_0_5",
    "atr_protective_or_ob_target_0_5",
    "aligned_atr_protective_or_ob_target_0_5",
    "aligned_safe_atr_protective_or_ob_target_0_5",
    "protective_swing_target_ge_1",
    "protective_swing_safe",
    "protective_swing_aligned",
    "protective_swing_aligned_safe",
    "protective_swing_normal_volume",
    "supportive_ob_safe",
    "supportive_ob_aligned",
    "supportive_ob_aligned_safe",
    "supportive_ob_normal_volume",
)
POLICIES = (
    {"name": "close_0_75_emergency_1_5", "kind": "close_confirm",
     "close_pct": .75, "stop_pct": 1.5},
    {"name": "close_1_emergency_1_5", "kind": "close_confirm",
     "close_pct": 1.0, "stop_pct": 1.5},
    {"name": "close_1_emergency_2", "kind": "close_confirm",
     "close_pct": 1.0, "stop_pct": 2.0},
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def mask_for(frame: pd.DataFrame, name: str) -> pd.Series:
    atr = frame.stop_inside_hourly_atr.astype(bool)
    protective = ~frame.no_protective_level_before_stop.astype(bool)
    supportive = frame.supportive_ob_flag.astype(bool)
    aligned = ~(frame.higher_timeframe_conflict.astype(bool)
                | frame.btc_direction_conflict.astype(bool)
                | frame.market_breadth_conflict.astype(bool))
    safe = ~(frame.opposing_level_before_target.astype(bool)
             | frame.opposing_order_block.astype(bool)
             | frame.failed_directional_breakout.astype(bool))
    normal_volume = ~frame.abnormal_quote_volume.astype(bool)
    half = frame.selected_target_pct.eq(.5)
    values = {
        "protective_swing": protective,
        "supportive_ob": supportive,
        "atr_protective_swing": atr & protective,
        "atr_protective_or_ob": atr & (protective | supportive),
        "aligned_atr_protective_or_ob": aligned & atr & (protective | supportive),
        "aligned_safe_atr_protective_or_ob": aligned & safe & atr & (protective | supportive),
        "protective_swing_target_0_5": protective & half,
        "supportive_ob_target_0_5": supportive & half,
        "atr_protective_swing_target_0_5": atr & protective & half,
        "atr_protective_or_ob_target_0_5": atr & (protective | supportive) & half,
        "aligned_atr_protective_or_ob_target_0_5": (
            aligned & atr & (protective | supportive) & half),
        "aligned_safe_atr_protective_or_ob_target_0_5": (
            aligned & safe & atr & (protective | supportive) & half),
        "protective_swing_target_ge_1": protective & ~half,
        "protective_swing_safe": protective & safe,
        "protective_swing_aligned": protective & aligned,
        "protective_swing_aligned_safe": protective & aligned & safe,
        "protective_swing_normal_volume": protective & normal_volume,
        "supportive_ob_safe": supportive & safe,
        "supportive_ob_aligned": supportive & aligned,
        "supportive_ob_aligned_safe": supportive & aligned & safe,
        "supportive_ob_normal_volume": supportive & normal_volume,
    }
    return values[name]


def routed(frame: pd.DataFrame, candles: dict, mask_name: str,
           policy: dict) -> pd.DataFrame:
    found = frame.copy()
    active = mask_for(found, mask_name)
    prepared = {
        symbol: {key: np.asarray(candle[key], dtype=(np.int64 if key == "time" else float))
                 for key in ("open", "high", "low", "close", "time")}
        for symbol, candle in candles.items()
    }
    signed, exits, hits, reasons = [], [], [], []
    for index, row in found.iterrows():
        selected = policy if bool(active.at[index]) else conditional.BASE_POLICY
        value = geometry.resolve_row(row.to_dict(), prepared[str(row["symbol"])], selected)
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
    return bool(candidate["n"] >= 700
                and candidate["trades_per_calendar_day"] >= 2.0
                and candidate["return_pct"] >= baseline["return_pct"]
                and candidate["literal_target_hit_rate"] >= baseline["literal_target_hit_rate"]
                and candidate["dd_pct"] <= baseline["dd_pct"]
                and candidate["negative_months"] <= baseline["negative_months"]
                and candidate["minimum_month_pct"] >= baseline["minimum_month_pct"])


def validation_pass(candidate: dict, baseline: dict) -> bool:
    return bool(candidate["n"] >= 500
                and candidate["trades_per_calendar_day"] >= 2.0
                and candidate["return_pct"] >= baseline["return_pct"]
                and candidate["literal_target_hit_rate"] >= baseline["literal_target_hit_rate"]
                and candidate["dd_pct"] <= baseline["dd_pct"]
                and candidate["negative_months"] <= baseline["negative_months"]
                and candidate["minimum_month_pct"] >= baseline["minimum_month_pct"])


def main() -> None:
    candles, sources = causes.load_candles()
    frame = causes.enrich(pd.read_pickle(causes.ROWS), candles)
    baseline = {year: causes.metrics(frame, (year,)) for year in range(2022, 2027)}
    frontier = []
    eligible = []
    for mask_name, policy in itertools.product(MASKS, POLICIES):
        candidate = routed(frame, candles, mask_name, policy)
        score = causes.metrics(candidate, (2022,))
        passed = selection_pass(score, baseline[2022])
        item = {"mask": mask_name, "policy": policy,
                "selection_2022": score, "selection_pass": passed}
        frontier.append(item)
        if passed:
            rank = (baseline[2022]["negative_months"]-score["negative_months"],
                    score["minimum_month_pct"], score["return_pct"],
                    score["literal_target_hit_rate"], -score["dd_pct"])
            eligible.append((rank, mask_name, policy, candidate, score))

    frozen = validation = test = confirmation = None
    status = "REJECTED_SELECTION"
    if eligible:
        _, mask_name, policy, candidate, selection = max(eligible, key=lambda item: item[0])
        validation = causes.metrics(candidate, (2023,))
        frozen = {"mask": mask_name, "policy": policy,
                  "selection_2022": selection,
                  "selection_baseline_2022": baseline[2022],
                  "eligible_candidates": len(eligible)}
        status = "REJECTED_VALIDATION"
        if validation_pass(validation, baseline[2023]):
            test = causes.metrics(candidate, (2024,))
            status = "REJECTED_TEST_2024"
            if validation_pass(test, baseline[2024]):
                confirmation = {str(year): causes.metrics(candidate, (year,))
                                for year in (2025, 2026)}
                status = ("CONDITIONAL_CLOSE_RESEARCH_CANDIDATE"
                          if all(validation_pass(confirmation[str(year)], baseline[year])
                                 for year in (2025, 2026))
                          else "REJECTED_CONFIRMATION")
    report = {
        "status": status, "ready_for_live_orders": False,
        "protocol": {
            "entry_and_target_router": "unchanged frozen target cascade",
            "default_stop": "hard 1% intrabar",
            "conditional_exit": "close confirmation with hard emergency stop",
            "selection": 2022, "validation": 2023, "test": 2024,
            "confirmation_only": [2025, 2026],
            "entry_and_exit_orders": "market proxies only",
            "money": "3% current-equity margin, fixed 10x gross",
        },
        "mask_definitions": {
            "protective_swing": "prior 24h directional support/resistance lies within 1.5% behind entry",
            "supportive_ob": "valid same-direction displacement block lies within 1.2% behind entry",
            "atr": "14h ATR is at least the default 1% stop",
            "aligned": "168h trend, BTC 24h return and market breadth do not conflict",
            "safe": "no opposing level/block or failed directional breakout in target path",
            "target_0_5": "conditional close stop is restricted to the 0.5% target branch",
            "target_ge_1": "conditional close stop is restricted to target branches at or above 1%",
        },
        "baseline": {str(year): value for year, value in baseline.items()},
        "selection_frontier": frontier,
        "frozen_before_validation": frozen,
        "validation_2023": validation, "test_2024": test,
        "confirmation_2025_2026": confirmation,
        "sources": {**sources, str(causes.ROWS): digest(causes.ROWS),
                    str(Path(causes.__file__).name): digest(Path(causes.__file__)),
                    str(Path(conditional.__file__).name): digest(Path(conditional.__file__)),
                    str(Path(geometry.__file__).name): digest(Path(geometry.__file__)),
                    str(Path(__file__)): digest(Path(__file__))},
        "limitations": [
            "Protective levels and order blocks are OHLCV proxies rather than historical book depth.",
            "Close confirmation can realize a larger gap loss than the close threshold.",
            "No candidate may change runtime before genuinely new forward confirmation.",
        ],
    }
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"status": status, "eligible": len(eligible),
                      "frozen": frozen, "validation": validation,
                      "test": test, "confirmation": confirmation,
                      "frontier": frontier}, indent=2), flush=True)


if __name__ == "__main__":
    main()
