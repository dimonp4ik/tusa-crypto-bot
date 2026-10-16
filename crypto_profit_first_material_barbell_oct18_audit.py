"""Profit-first audit of material 0.5R/2R crypto barbell exits.

Only entries with a one-to-two-percent structural stop are considered, so the
first realized target is a 0.5%-1% price move and the runner is a 2%-4% move.
Models fit 2020-2021.  Model, cutoff and partial fraction use 2022-2023, then
freeze before 2024-2026 and the direct X-Perp transfer are evaluated.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

import crypto_05_2r_precision_barbell_oct5_audit as legacy
import crypto_high_target_tree_oct5_audit as base
from crypto_runtime_shadow_portfolio_oct3_audit import runtime_gate


ROOT = Path("reports/audit_2026_09_08")
OUT = ROOT / "crypto_profit_first_material_barbell_oct18.json"
ROWS_OUT = ROOT / "crypto_profit_first_material_barbell_oct18.pkl"
DIRECT_ROWS_OUT = ROOT / "crypto_profit_first_material_barbell_xperp_oct18.pkl"
FRESH_DIRECT_ROWS_OUT = ROOT / "crypto_profit_first_material_barbell_xperp_oct17_fresh.pkl"
FRESH_NEAR_PATH = ROOT / "crypto_xperp_raw_router_audit_0_5_oct17_material_raw_raw_families.csv"
FRESH_FAR_PATH = ROOT / "crypto_xperp_raw_router_audit_2_0_oct17_material_raw_raw_families.csv"
FIT = legacy.FIT
SELECT = legacy.SELECT
CONFIRM = legacy.CONFIRM
VALIDATION = legacy.VALIDATION
LATER = legacy.HIDDEN
FRACTIONS = (.00, .05, .10, .20, .30, .50, .70)
QUANTILES = (.75, .80, .825, .85, .875, .90, .925, .95, .96, .97, .98)
CONFIGS = legacy.CONFIGS
MIN_DEVELOPMENT_TRADES_PER_DAY = 1.0
DIRECT_KEYS = ("symbol", "entry_time", "direction", "family")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def monthly_equivalent(value: dict) -> float:
    months = max(1, len(value["all"].get("monthly_return_pct", {})))
    factor = 1 + float(value["all"]["return_pct"]) / 100
    return round(100 * math.expm1(math.log(factor) / months), 6) if factor > 0 else -100.0


def development_ok(value: dict) -> bool:
    all_ = value["all"]
    periods = list(value["periods"].values())
    return bool(all_["n"] >= 100 and all_["return_pct"] > 0
                and all_["dd_pct"] <= 10
                and all_["trades_per_calendar_day"] >= MIN_DEVELOPMENT_TRADES_PER_DAY
                and all(item["return_pct"] > 0 for item in periods))


def development_score(selection: dict, confirmation: dict) -> float:
    monthly = min(monthly_equivalent(selection), monthly_equivalent(confirmation))
    drawdown = max(selection["all"]["dd_pct"], confirmation["all"]["dd_pct"])
    negative = (selection["all"]["negative_months"]
                + confirmation["all"]["negative_months"])
    return round(monthly - .08 * drawdown - .10 * negative, 8)


def objective_ok(value: dict) -> bool:
    all_ = value["all"]
    months = list(all_.get("monthly_return_pct", {}).values())
    return bool(months and all(month >= 30 for month in months)
                and all_["dd_pct"] <= 10
                and all_["trades_per_calendar_day"] >= 3)


def compact(item: dict | None) -> dict | None:
    if item is None:
        return None
    return {key: value for key, value in item.items() if not key.startswith("_")}


def accepted(frame: pd.DataFrame, periods: tuple[str, ...], threshold: float,
             fraction: float) -> pd.DataFrame:
    chosen = base.stage(frame, periods)
    chosen = chosen[chosen.model_probability >= threshold].copy()
    chosen = legacy.barbell_frame(chosen, fraction)
    return pd.DataFrame(runtime_gate(base.standardized(
        chosen, "profit_first_material_barbell")))


def coverage(frame: pd.DataFrame, periods: tuple[str, ...]) -> dict:
    output = {}
    for period in periods:
        part = base.stage(frame, (period,))
        stamp = pd.to_datetime(part.entry_time, unit="s", utc=True)
        output[period] = {
            "rows": len(part),
            "first_entry": stamp.min().isoformat() if len(stamp) else None,
            "last_entry": stamp.max().isoformat() if len(stamp) else None,
            "calendar_months_present": sorted(stamp.dt.strftime("%Y-%m").unique().tolist()),
        }
    return output


def join_direct_outcomes(far_path: Path, near_path: Path) -> pd.DataFrame:
    """Join two target replays by immutable signal identity, never row order."""
    far = pd.read_csv(far_path)
    near = pd.read_csv(near_path)
    keys = list(DIRECT_KEYS)
    if far.duplicated(keys).any() or near.duplicated(keys).any():
        raise ValueError("direct target replay contains duplicate signal identities")
    near = near[keys + ["outcome", "exit_time"]].rename(columns={
        "outcome": "near_outcome", "exit_time": "near_exit_time"})
    joined = far.merge(near, on=keys, how="inner", validate="one_to_one")
    if len(joined) != len(far) or len(joined) != len(near):
        raise ValueError("direct target replays do not contain identical signals")
    return joined


def direct_evaluation(frame: pd.DataFrame, frozen: dict, threshold: float,
                      fraction: float, rows_out: Path) -> tuple[dict, dict]:
    eligible = frame[
        frame.symbol.isin(base.ROBUST_SYMBOLS)
        & frame.risk_pct.between(
            legacy.RISK_PCT_MIN, legacy.RISK_PCT_MAX,
            inclusive="both")].copy()
    scored = legacy.scored(eligible, frozen["_model"], frozen["_schema"])
    chosen = scored[scored.model_probability >= threshold].copy()
    chosen = legacy.barbell_frame(chosen, fraction)
    rows = pd.DataFrame(runtime_gate(base.standardized(
        chosen, "profit_first_material_barbell")))
    rows.to_pickle(rows_out)
    metrics = base.evaluate(rows.to_dict("records"))
    stamp = pd.to_datetime(eligible.entry_time, unit="s", utc=True)
    coverage_ = {
        "rows_before_score_filter": len(eligible),
        "selected_before_runtime_gate": len(chosen),
        "first_entry": stamp.min().isoformat() if len(stamp) else None,
        "last_entry": stamp.max().isoformat() if len(stamp) else None,
        "calendar_months_present": sorted(stamp.dt.strftime("%Y-%m").unique().tolist()),
    }
    return metrics, coverage_


def main() -> None:
    frame = legacy.load_joined()
    fit = base.stage(frame, FIT)
    schema = base.design(fit)
    x_fit = base.matrix(fit, schema)
    y_fit = fit.outcome.eq("TP2").astype(int).to_numpy()
    candidates = []
    model_reports = []
    for leaves, leaf, rate in CONFIGS:
        estimator = HistGradientBoostingClassifier(
            learning_rate=rate, max_iter=150, max_leaf_nodes=leaves,
            min_samples_leaf=leaf, l2_regularization=1.0,
            early_stopping=False, random_state=20260908)
        estimator.fit(x_fit, y_fit)
        scored = legacy.scored(frame, estimator, schema)
        selection_scores = base.stage(scored, SELECT).model_probability
        thresholds = sorted({float(selection_scores.quantile(q))
                             for q in QUANTILES}, reverse=True)
        model_candidates = 0
        for fraction in FRACTIONS:
            for quantile, threshold in zip(reversed(QUANTILES), thresholds):
                selection = legacy.metrics(scored, SELECT, threshold, fraction)
                if not development_ok(selection):
                    continue
                confirmation = legacy.metrics(scored, CONFIRM, threshold, fraction)
                if not development_ok(confirmation):
                    continue
                item = {
                    "max_leaf_nodes": leaves,
                    "min_samples_leaf": leaf,
                    "learning_rate": rate,
                    "near_fraction": fraction,
                    "selection_quantile": quantile,
                    "threshold": threshold,
                    "selection_2022h1": selection,
                    "confirmation_2022h2_2023": confirmation,
                    "score": development_score(selection, confirmation),
                    "_model": estimator, "_schema": schema, "_frame": scored,
                }
                candidates.append(item)
                model_candidates += 1
        model_reports.append({
            "max_leaf_nodes": leaves, "min_samples_leaf": leaf,
            "learning_rate": rate, "development_candidates": model_candidates,
        })
        print(json.dumps(model_reports[-1]), flush=True)
    candidates.sort(key=lambda item: item["score"], reverse=True)
    frozen = candidates[0] if candidates else None
    validation = later = direct = checks = None
    direct_coverage = fresh_direct = fresh_direct_coverage = None
    fresh_forward = fresh_forward_coverage = None
    status = "REJECTED_DEVELOPMENT"
    if frozen:
        threshold = float(frozen["threshold"])
        fraction = float(frozen["near_fraction"])
        scored = frozen["_frame"]
        validation = legacy.metrics(scored, VALIDATION, threshold, fraction)
        later = legacy.metrics(scored, LATER, threshold, fraction)
        historical = {
            "selection_2022h1": frozen["selection_2022h1"],
            "confirmation_2022h2_2023": frozen["confirmation_2022h2_2023"],
            "validation_2024": validation, "later_2025_2026": later,
        }
        checks = {name: objective_ok(value) for name, value in historical.items()}
        accepted(scored, (*SELECT, *CONFIRM, *VALIDATION, *LATER),
                 threshold, fraction).to_pickle(ROWS_OUT)

        far_path, near_path = base.direct_path(2.0), base.direct_path(.5)
        direct_frame = join_direct_outcomes(far_path, near_path)
        direct, direct_coverage = direct_evaluation(
            direct_frame, frozen, threshold, fraction, DIRECT_ROWS_OUT)

        fresh_frame = join_direct_outcomes(FRESH_FAR_PATH, FRESH_NEAR_PATH)
        fresh_direct, fresh_direct_coverage = direct_evaluation(
            fresh_frame, frozen, threshold, fraction, FRESH_DIRECT_ROWS_OUT)
        prior_end = float(direct_frame.entry_time.max())
        forward_frame = fresh_frame[fresh_frame.entry_time > prior_end].copy()
        fresh_forward, fresh_forward_coverage = direct_evaluation(
            forward_frame, frozen, threshold, fraction,
            ROOT / "crypto_profit_first_material_barbell_xperp_oct17_forward.pkl")
        status = ("MATERIAL_BARBELL_RESEARCH_CANDIDATE"
                  if all(checks.values()) and objective_ok({"all": direct,
                                                            "periods": {}})
                  else "REJECTED_OBJECTIVE")
    report = {
        "status": status, "ready_for_live_orders": False,
        "protocol": {
            "priority": "account return first; win rate diagnostic only",
            "role": "material-move supplement; combined bot must exceed three trades/day",
            "fit": list(FIT), "selection": list(SELECT),
            "pre_freeze_confirmation": list(CONFIRM),
            "validation": list(VALIDATION), "later": list(LATER),
            "entry_risk_pct_range": [legacy.RISK_PCT_MIN, legacy.RISK_PCT_MAX],
            "near_target_r": .5, "runner_target_r": 2.0,
            "near_price_move_range_pct": [.5, 1.0],
            "runner_price_move_range_pct": [2.0, 4.0],
            "entry": "next 15-minute market open", "orders": "market only",
            "money": "3% current-equity margin, fixed 10x gross",
            "objective": ">=30% account return in every month; >=3 trades/day; DD <=10%",
        },
        "coverage": coverage(frame, (*FIT, *SELECT, *CONFIRM, *VALIDATION, *LATER)),
        "model_reports": model_reports, "candidate_count": len(candidates),
        "development_top20": [compact(item) for item in candidates[:20]],
        "frozen_before_2024": compact(frozen),
        "validation_2024": validation, "later_2025_2026": later,
        "objective_checks": checks,
        "direct_xperp": direct, "direct_xperp_coverage": direct_coverage,
        "fresh_direct_xperp_through_oct17": fresh_direct,
        "fresh_direct_xperp_coverage": fresh_direct_coverage,
        "untouched_forward_after_prior_file": fresh_forward,
        "untouched_forward_coverage": fresh_forward_coverage,
        "sources": {
            **{str(base.cache_path(target, period)): digest(base.cache_path(target, period))
               for target in (.5, 2.0) for period in base.PERIODS},
            str(base.direct_path(.5)): digest(base.direct_path(.5)),
            str(base.direct_path(2.0)): digest(base.direct_path(2.0)),
            str(FRESH_NEAR_PATH): digest(FRESH_NEAR_PATH),
            str(FRESH_FAR_PATH): digest(FRESH_FAR_PATH),
            str(Path(__file__)): digest(Path(__file__)),
        },
        "limitations": [
            "Each named annual Binance period contains only the available six-month research window.",
            "The direct X-Perp file is a short transfer window, not a full market cycle.",
            "The post-October-1 untouched forward is only about two weeks and cannot establish a monthly expectation.",
            "Gross returns exclude fees per user instruction.",
        ],
    }
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"status": status, "frozen": compact(frozen),
                      "validation_2024": validation,
                      "later_2025_2026": later,
                      "direct_xperp": direct,
                      "direct_xperp_coverage": direct_coverage,
                      "fresh_direct_xperp_through_oct17": fresh_direct,
                      "untouched_forward_after_prior_file": fresh_forward,
                      "untouched_forward_coverage": fresh_forward_coverage,
                      "objective_checks": checks}, indent=2), flush=True)


if __name__ == "__main__":
    main()
