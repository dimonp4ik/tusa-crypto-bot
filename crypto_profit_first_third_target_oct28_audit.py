"""Third-target audit for the frozen crypto Profit-First 2R shadow route.

The entry model, 0.5R first bank, threshold and portfolio rails remain frozen.
Candidates bank another part at 2R and keep only a protected remainder for 3R
or 4R.  Selection ends in 2023; 2024-2026 and current X-Perp are holdouts.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

import crypto_05_2r_precision_barbell_oct5_audit as legacy
import crypto_high_target_tree_oct5_audit as base
import crypto_profit_first_material_barbell_oct18_audit as joiner
import crypto_profit_first_material_target_oct18_audit as frozen
from crypto_runtime_shadow_portfolio_oct3_audit import runtime_gate


ROOT = Path("reports/audit_2026_09_08")
SOURCE_REPORT = ROOT / "crypto_profit_first_material_target_oct18.json"
NEAR_DIRECT = ROOT / "crypto_xperp_raw_router_audit_0_5_oct28_material_raw_raw_families.csv"
TWO_DIRECT = ROOT / "crypto_xperp_raw_router_audit_2_0_oct28_material_raw_raw_families.csv"
THIRD_DIRECT = {
    3.0: ROOT / "crypto_xperp_raw_router_audit_3_0_oct28_material_raw_raw_families.csv",
    4.0: ROOT / "crypto_xperp_raw_router_audit_4_0_oct28_material_raw_raw_families.csv",
}
OUT = ROOT / "crypto_profit_first_third_target_oct28.json"
ROWS_OUT = ROOT / "crypto_profit_first_third_target_oct28.pkl"
TARGETS = (3.0, 4.0)
MIDDLE_FRACTIONS = (.45, .65, .75, .85, .90)
NEAR_FRACTION = .05
HIST_KEYS = ["period", "symbol", "entry_time", "direction", "family"]
DIRECT_KEYS = ["symbol", "entry_time", "direction", "family"]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def attach_third(two: pd.DataFrame, third: pd.DataFrame,
                 keys: list[str]) -> pd.DataFrame:
    if two.duplicated(keys).any() or third.duplicated(keys).any():
        raise ValueError("duplicate signal identity")
    extra = third[keys + ["outcome", "exit_time"]].rename(columns={
        "outcome": "third_outcome", "exit_time": "third_exit_time"})
    joined = two.merge(extra, on=keys, how="inner", validate="one_to_one")
    if len(joined) != len(two):
        raise ValueError("third-target replay is missing 2R identities")
    return joined


def variant(frame: pd.DataFrame, target: float, middle_fraction: float,
            module: str) -> list[dict]:
    result = frame.copy()
    near_hit = result.near_outcome.eq("TP2")
    two_hit = result.outcome.eq("TP2")
    third_hit = result.third_outcome.eq("TP2")
    last_fraction = 1.0 - NEAR_FRACTION - middle_fraction
    if last_fraction <= 0:
        raise ValueError("third-target fraction must be positive")
    result["net_r"] = np.where(
        ~near_hit, -1.0,
        np.where(~two_hit, NEAR_FRACTION * .5,
                 NEAR_FRACTION * .5 + middle_fraction * 2.0
                 + np.where(third_hit, last_fraction * target, 0.0)))
    result["exit_time"] = np.where(
        two_hit, result.third_exit_time, result.exit_time)
    result["third_target_r"] = target
    result["middle_fraction"] = middle_fraction
    result["third_fraction"] = last_fraction
    return runtime_gate(base.standardized(result, module))


def baseline(frame: pd.DataFrame) -> list[dict]:
    return runtime_gate(base.standardized(
        frozen.barbell_frame(frame, NEAR_FRACTION, 2.0),
        "profit_first_material_2r"))


def stages(frame: pd.DataFrame, target: float, middle: float) -> dict:
    periods = {
        "selection_2022h1": frozen.SELECT,
        "confirmation_2022h2_2023": frozen.CONFIRM,
        "validation_2024": frozen.VALIDATION,
        "later_2025_2026": frozen.LATER,
    }
    output = {}
    for name, values in periods.items():
        part = base.stage(frame, values)
        output[name] = {
            "baseline": base.evaluate(baseline(part)),
            "candidate": base.evaluate(variant(
                part, target, middle, f"profit_first_third_{target:g}r")),
        }
    return output


def compact(value: dict) -> dict:
    keys = ("n", "return_pct", "dd_pct", "wr", "negative_months",
            "max_loss_streak", "trades_per_calendar_day")
    return {key: value[key] for key in keys}


def main() -> None:
    source = json.loads(SOURCE_REPORT.read_text(encoding="utf-8"))
    policy = source["frozen_before_2024"]
    if (float(policy["runner_target_r"]) != 2.0
            or float(policy["near_fraction"]) != NEAR_FRACTION):
        raise ValueError("source policy changed")
    threshold = float(policy["threshold"])
    two = frozen.load_joined(2.0)
    fit = base.stage(two, frozen.FIT)
    schema = base.design(fit)
    model = HistGradientBoostingClassifier(
        **frozen.MODEL, max_iter=150, l2_regularization=1.0,
        early_stopping=False, random_state=20260908)
    model.fit(base.matrix(fit, schema), fit.outcome.eq("TP2").astype(int))
    scored_two = legacy.scored(two, model, schema)
    chosen_two = scored_two[scored_two.model_probability.ge(threshold)].copy()

    historical = {}
    for target in TARGETS:
        third = base.load_target(target)
        historical[target] = attach_third(chosen_two, third, HIST_KEYS)

    candidates = []
    for target in TARGETS:
        for middle in MIDDLE_FRACTIONS:
            review = stages(historical[target], target, middle)
            development_pass = all(
                review[name]["candidate"]["return_pct"] >
                review[name]["baseline"]["return_pct"]
                and review[name]["candidate"]["dd_pct"] <=
                review[name]["baseline"]["dd_pct"] + .5
                and review[name]["candidate"]["negative_months"] <=
                review[name]["baseline"]["negative_months"]
                for name in ("selection_2022h1", "confirmation_2022h2_2023"))
            minimum_gain = min(
                review[name]["candidate"]["return_pct"]
                - review[name]["baseline"]["return_pct"]
                for name in ("selection_2022h1", "confirmation_2022h2_2023"))
            candidates.append({
                "third_target_r": target, "near_fraction": NEAR_FRACTION,
                "middle_fraction": middle,
                "third_fraction": 1-NEAR_FRACTION-middle,
                "development_pass": development_pass,
                "development_min_return_gain": minimum_gain,
                "stages": {name: {side: compact(value)
                                  for side, value in pair.items()}
                           for name, pair in review.items()},
            })
    qualified = [item for item in candidates if item["development_pass"]]
    qualified.sort(key=lambda item: item["development_min_return_gain"], reverse=True)
    selected = qualified[0] if qualified else None
    historical_holdout_pass = bool(selected and all(
        selected["stages"][name]["candidate"]["return_pct"] >=
        selected["stages"][name]["baseline"]["return_pct"]
        and selected["stages"][name]["candidate"]["dd_pct"] <=
        selected["stages"][name]["baseline"]["dd_pct"] + .5
        and selected["stages"][name]["candidate"]["negative_months"] <=
        selected["stages"][name]["baseline"]["negative_months"]
        for name in ("validation_2024", "later_2025_2026")))

    direct_two = joiner.join_direct_outcomes(TWO_DIRECT, NEAR_DIRECT)
    eligible = direct_two[
        direct_two.symbol.isin(base.ROBUST_SYMBOLS)
        & direct_two.risk_pct.between(legacy.RISK_PCT_MIN, legacy.RISK_PCT_MAX,
                                      inclusive="both")].copy()
    direct_scored = legacy.scored(eligible, model, schema)
    direct_chosen = direct_scored[
        direct_scored.model_probability.ge(threshold)].copy()
    direct_baseline_rows = baseline(direct_chosen)
    direct_baseline = base.evaluate(direct_baseline_rows)
    direct_candidate = None
    direct_holdout_pass = False
    if selected:
        target = float(selected["third_target_r"])
        third = pd.read_csv(THIRD_DIRECT[target])
        joined = attach_third(direct_chosen, third, DIRECT_KEYS)
        rows = variant(joined, target, float(selected["middle_fraction"]),
                       f"profit_first_third_{target:g}r")
        pd.DataFrame(rows).to_pickle(ROWS_OUT)
        direct_candidate = base.evaluate(rows)
        direct_holdout_pass = bool(
            direct_candidate["return_pct"] >= direct_baseline["return_pct"]
            and direct_candidate["dd_pct"] <= direct_baseline["dd_pct"] + .5
            and direct_candidate["negative_months"] <= direct_baseline["negative_months"])
    accepted = bool(historical_holdout_pass and direct_holdout_pass)
    report = {
        "status": ("RESEARCH_CANDIDATE_AWAIT_FRESH_FORWARD" if accepted
                   else "REJECTED_HOLDOUT" if selected
                   else "REJECTED_DEVELOPMENT"),
        "ready_for_live_orders": False, "orders_possible": False,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "protocol": {
            "entry_model": "unchanged frozen Profit-First 2R HGB",
            "near": "bank 5% at 0.5R",
            "candidate": "bank middle fraction at 2R; protected remainder to 3R/4R",
            "selection": ["2022h1", "2022h2-2023"],
            "holdouts": ["2024", "2025-2026", "direct X-Perp through Oct27"],
            "money": "3% current-equity margin; fixed 10x; market only",
        },
        "grid": {"candidates": len(candidates), "targets_r": list(TARGETS),
                 "middle_fractions": list(MIDDLE_FRACTIONS)},
        "selected_before_holdouts": selected,
        "historical_holdout_pass": historical_holdout_pass,
        "direct_xperp": {"baseline": compact(direct_baseline),
                         "candidate": (None if direct_candidate is None
                                       else compact(direct_candidate)),
                         "pass": direct_holdout_pass},
        "top_development": sorted(
            candidates, key=lambda x: x["development_min_return_gain"],
            reverse=True)[:10],
        "sources": {
            str(SOURCE_REPORT): digest(SOURCE_REPORT),
            str(NEAR_DIRECT): digest(NEAR_DIRECT), str(TWO_DIRECT): digest(TWO_DIRECT),
            **{str(path): digest(path) for path in THIRD_DIRECT.values()},
            str(Path(__file__).resolve()): digest(Path(__file__).resolve()),
        },
        "limitations": [
            "Current direct X-Perp history begins in April 2026.",
            "A missed TP3 remainder is conservatively valued at breakeven.",
            "Public OHLC cannot reproduce order-book slippage or latency.",
            "Gross results exclude fees at user request.",
        ],
    }
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({key: report[key] for key in (
        "status", "grid", "selected_before_holdouts",
        "historical_holdout_pass", "direct_xperp")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
