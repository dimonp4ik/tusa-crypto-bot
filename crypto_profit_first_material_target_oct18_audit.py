"""Profit-first comparison of 2R, 3R and 4R material crypto runners.

The model architecture is fixed from the earlier 2R development audit. Each
target model fits 2020-2021. Target, probability cutoff and near fraction use
2022-2023 only, then freeze before 2024-2026 and direct X-Perp evaluation.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

import crypto_05_2r_precision_barbell_oct5_audit as legacy
import crypto_high_target_tree_oct5_audit as base
import crypto_profit_first_material_barbell_oct18_audit as prior
from crypto_runtime_shadow_portfolio_oct3_audit import runtime_gate


ROOT = Path("reports/audit_2026_09_08")
OUT = ROOT / "crypto_profit_first_material_target_oct18.json"
ROWS_OUT = ROOT / "crypto_profit_first_material_target_oct18.pkl"
DIRECT_OUT = ROOT / "crypto_profit_first_material_target_xperp_oct18.pkl"
FRESH_OUT = ROOT / "crypto_profit_first_material_target_xperp_oct17_fresh.pkl"
FORWARD_OUT = ROOT / "crypto_profit_first_material_target_xperp_oct17_forward.pkl"
TARGETS = (2.0, 3.0, 4.0)
FRACTIONS = prior.FRACTIONS
QUANTILES = prior.QUANTILES
FIT, SELECT, CONFIRM = prior.FIT, prior.SELECT, prior.CONFIRM
VALIDATION, LATER = prior.VALIDATION, prior.LATER
MODEL = {"max_leaf_nodes": 7, "min_samples_leaf": 300,
         "learning_rate": .05}
JOIN_KEYS = ("period", "symbol", "entry_time", "direction", "family")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_joined(target: float) -> pd.DataFrame:
    far = base.load_target(target).copy()
    near = base.load_target(.5).copy()
    keys = list(JOIN_KEYS)
    if far.duplicated(keys).any() or near.duplicated(keys).any():
        raise ValueError("historical target replay contains duplicate signal identities")
    near = near[keys + ["outcome", "exit_time"]].rename(columns={
        "outcome": "near_outcome", "exit_time": "near_exit_time"})
    joined = far.merge(near, on=keys, how="inner", validate="one_to_one")
    unmatched = (len(far) - len(joined)) + (len(near) - len(joined))
    # A target replay may omit a handful of outcomes at a cache boundary. Use
    # only exact signal matches and fail if the discrepancy is material.
    if unmatched > max(5, int(.0001 * max(len(far), len(near)))):
        raise ValueError("historical target replays differ materially")
    result = joined[
        joined.symbol.isin(base.ROBUST_SYMBOLS)
        & joined.risk_pct.between(legacy.RISK_PCT_MIN, legacy.RISK_PCT_MAX,
                                  inclusive="both")].reset_index(drop=True)
    result.attrs["join_integrity"] = {
        "far_rows": len(far), "near_rows": len(near),
        "matched_rows": len(joined), "unmatched_rows": unmatched,
    }
    return result


def barbell_frame(frame: pd.DataFrame, fraction: float,
                  runner_target_r: float) -> pd.DataFrame:
    result = frame.copy()
    near_hit = result.near_outcome.eq("TP2")
    far_hit = result.outcome.eq("TP2")
    pnl = np.where(
        ~near_hit, -1.0,
        np.where(far_hit,
                 fraction * .5 + (1 - fraction) * runner_target_r,
                 fraction * .5))
    result["net_r"] = pnl.astype(float)
    result["barbell_outcome"] = np.where(
        ~near_hit, "SL", np.where(far_hit, "TP_RUNNER", "TP_BE"))
    result["runner_target_r"] = runner_target_r
    return result


def evaluate(frame: pd.DataFrame, periods: tuple[str, ...], threshold: float,
             fraction: float, target: float) -> dict:
    chosen = base.stage(frame, periods)
    chosen = chosen[chosen.model_probability >= threshold].copy()
    chosen = barbell_frame(chosen, fraction, target)
    rows = runtime_gate(base.standardized(
        chosen, f"profit_first_material_{target:g}r"))
    return {"all": base.evaluate(rows), "periods": {
        period: base.evaluate(runtime_gate(base.standardized(
            barbell_frame(base.stage(frame, (period,))[
                base.stage(frame, (period,)).model_probability >= threshold],
                fraction, target), f"profit_first_material_{target:g}r")))
        for period in periods}}


def accepted(frame: pd.DataFrame, periods: tuple[str, ...], threshold: float,
             fraction: float, target: float) -> pd.DataFrame:
    chosen = base.stage(frame, periods)
    chosen = chosen[chosen.model_probability >= threshold].copy()
    chosen = barbell_frame(chosen, fraction, target)
    return pd.DataFrame(runtime_gate(base.standardized(
        chosen, f"profit_first_material_{target:g}r")))


def direct_paths(target: float, fresh: bool) -> tuple[Path, Path]:
    suffix = str(target).replace(".", "_")
    far = (ROOT / f"crypto_xperp_raw_router_audit_{suffix}_oct17_material_raw_raw_families.csv"
           if fresh else base.direct_path(target))
    near = (ROOT / "crypto_xperp_raw_router_audit_0_5_oct17_material_raw_raw_families.csv"
            if fresh else base.direct_path(.5))
    return far, near


def score_direct(frame: pd.DataFrame, model, schema: dict, threshold: float,
                 fraction: float, target: float,
                 rows_out: Path | None = None) -> tuple[dict, dict]:
    eligible = frame[
        frame.symbol.isin(base.ROBUST_SYMBOLS)
        & frame.risk_pct.between(legacy.RISK_PCT_MIN, legacy.RISK_PCT_MAX,
                                 inclusive="both")].copy()
    scored = legacy.scored(eligible, model, schema)
    chosen = scored[scored.model_probability >= threshold].copy()
    rows = accepted(chosen.assign(period="direct"), ("direct",),
                    -np.inf, fraction, target)
    if rows_out is not None:
        rows.to_pickle(rows_out)
    stamp = pd.to_datetime(eligible.entry_time, unit="s", utc=True)
    coverage = {
        "rows_before_score_filter": len(eligible),
        "selected_before_runtime_gate": len(chosen),
        "first_entry": stamp.min().isoformat() if len(stamp) else None,
        "last_entry": stamp.max().isoformat() if len(stamp) else None,
        "calendar_months_present": sorted(stamp.dt.strftime("%Y-%m").unique().tolist()),
    }
    return base.evaluate(rows.to_dict("records")), coverage


def main() -> None:
    candidates = []
    model_reports = []
    frames = {}
    for target in TARGETS:
        frame = load_joined(target)
        join_integrity = frame.attrs["join_integrity"]
        fit = base.stage(frame, FIT)
        schema = base.design(fit)
        model = HistGradientBoostingClassifier(
            **MODEL, max_iter=150, l2_regularization=1.0,
            early_stopping=False, random_state=20260908)
        model.fit(base.matrix(fit, schema), fit.outcome.eq("TP2").astype(int))
        scored = legacy.scored(frame, model, schema)
        frames[target] = scored
        selection_scores = base.stage(scored, SELECT).model_probability
        thresholds = {q: float(selection_scores.quantile(q)) for q in QUANTILES}
        qualified = 0
        for fraction in FRACTIONS:
            for quantile, threshold in thresholds.items():
                selection = evaluate(scored, SELECT, threshold, fraction, target)
                if not prior.development_ok(selection):
                    continue
                confirmation = evaluate(scored, CONFIRM, threshold, fraction, target)
                if not prior.development_ok(confirmation):
                    continue
                candidates.append({
                    "runner_target_r": target, "near_fraction": fraction,
                    "selection_quantile": quantile, "threshold": threshold,
                    "selection_2022h1": selection,
                    "confirmation_2022h2_2023": confirmation,
                    "score": prior.development_score(selection, confirmation),
                    "_model": model, "_schema": schema, "_frame": scored,
                })
                qualified += 1
        model_reports.append({"runner_target_r": target,
                              "join_integrity": join_integrity,
                              "development_candidates": qualified})
        print(json.dumps(model_reports[-1]), flush=True)
    candidates.sort(key=lambda item: item["score"], reverse=True)
    frozen = candidates[0] if candidates else None
    target_reviews = {}
    for target in TARGETS:
        pool = [item for item in candidates if item["runner_target_r"] == target]
        if not pool:
            target_reviews[str(target)] = None
            continue
        candidate = pool[0]
        threshold = float(candidate["threshold"])
        fraction = float(candidate["near_fraction"])
        validation_ = evaluate(candidate["_frame"], VALIDATION, threshold,
                               fraction, target)
        later_ = evaluate(candidate["_frame"], LATER, threshold,
                          fraction, target)
        old_far, old_near = direct_paths(target, False)
        old_frame = prior.join_direct_outcomes(old_far, old_near)
        old_, old_coverage_ = score_direct(
            old_frame, candidate["_model"], candidate["_schema"], threshold,
            fraction, target)
        fresh_far, fresh_near = direct_paths(target, True)
        fresh_frame = prior.join_direct_outcomes(fresh_far, fresh_near)
        fresh_, fresh_coverage_ = score_direct(
            fresh_frame, candidate["_model"], candidate["_schema"], threshold,
            fraction, target)
        prior_end = float(old_frame.entry_time.max())
        forward_frame = fresh_frame[fresh_frame.entry_time > prior_end].copy()
        forward_, forward_coverage_ = score_direct(
            forward_frame, candidate["_model"], candidate["_schema"], threshold,
            fraction, target)
        target_reviews[str(target)] = {
            "frozen_before_2024": prior.compact(candidate),
            "validation_2024": validation_, "later_2025_2026": later_,
            "old_direct_xperp": old_, "old_direct_coverage": old_coverage_,
            "fresh_direct_xperp": fresh_, "fresh_direct_coverage": fresh_coverage_,
            "untouched_forward": forward_,
            "untouched_forward_coverage": forward_coverage_,
        }
    validation = later = old_direct = fresh_direct = forward = None
    old_coverage = fresh_coverage = forward_coverage = checks = None
    status = "REJECTED_DEVELOPMENT"
    if frozen:
        target = float(frozen["runner_target_r"])
        threshold = float(frozen["threshold"])
        fraction = float(frozen["near_fraction"])
        scored = frozen["_frame"]
        validation = evaluate(scored, VALIDATION, threshold, fraction, target)
        later = evaluate(scored, LATER, threshold, fraction, target)
        accepted(scored, (*SELECT, *CONFIRM, *VALIDATION, *LATER),
                 threshold, fraction, target).to_pickle(ROWS_OUT)
        old_far, old_near = direct_paths(target, False)
        old_frame = prior.join_direct_outcomes(old_far, old_near)
        old_direct, old_coverage = score_direct(
            old_frame, frozen["_model"], frozen["_schema"], threshold,
            fraction, target, DIRECT_OUT)
        fresh_far, fresh_near = direct_paths(target, True)
        fresh_frame = prior.join_direct_outcomes(fresh_far, fresh_near)
        fresh_direct, fresh_coverage = score_direct(
            fresh_frame, frozen["_model"], frozen["_schema"], threshold,
            fraction, target, FRESH_OUT)
        prior_end = float(old_frame.entry_time.max())
        forward_frame = fresh_frame[fresh_frame.entry_time > prior_end].copy()
        forward, forward_coverage = score_direct(
            forward_frame, frozen["_model"], frozen["_schema"], threshold,
            fraction, target, FORWARD_OUT)
        historical = {"selection": frozen["selection_2022h1"],
                      "confirmation": frozen["confirmation_2022h2_2023"],
                      "validation": validation, "later": later}
        checks = {name: prior.objective_ok(value)
                  for name, value in historical.items()}
        status = ("MATERIAL_TARGET_RESEARCH_CANDIDATE"
                  if all(checks.values()) else "REJECTED_OBJECTIVE")
    report = {
        "status": status, "ready_for_live_orders": False,
        "protocol": {
            "priority": "account return first; win rate diagnostic only",
            "fit": list(FIT), "development": [*SELECT, *CONFIRM],
            "frozen_validation": list(VALIDATION), "later": list(LATER),
            "runner_targets_r": list(TARGETS),
            "entry_risk_pct_range": [legacy.RISK_PCT_MIN, legacy.RISK_PCT_MAX],
            "near_target_r": .5, "near_fractions": list(FRACTIONS),
            "entry": "next 15-minute market open", "orders": "market only",
            "money": "3% current-equity margin, fixed 10x gross",
            "objective": ">=30% account return every month; >=3 trades/day; DD <=10%",
        },
        "model": MODEL, "model_reports": model_reports,
        "candidate_count": len(candidates),
        "development_top20": [prior.compact(item) for item in candidates[:20]],
        "best_by_runner_target": target_reviews,
        "frozen_before_2024": prior.compact(frozen),
        "validation_2024": validation, "later_2025_2026": later,
        "old_direct_xperp": old_direct, "old_direct_coverage": old_coverage,
        "fresh_direct_xperp": fresh_direct, "fresh_direct_coverage": fresh_coverage,
        "untouched_forward": forward, "untouched_forward_coverage": forward_coverage,
        "objective_checks": checks,
        "sources": {
            **{str(base.cache_path(target, period)): digest(base.cache_path(target, period))
               for target in (.5, *TARGETS) for period in base.PERIODS},
            **{str(path): digest(path) for target in TARGETS
               for fresh in (False, True) for path in direct_paths(target, fresh)},
            str(Path(__file__)): digest(Path(__file__)),
        },
        "limitations": [
            "Historical annual labels contain the available six-month research windows.",
            "Direct X-Perp history is short and the post-prior-file forward is about two weeks.",
            "Gross returns exclude fees at user request.",
        ],
    }
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"status": status, "frozen": prior.compact(frozen),
                      "validation_2024": validation, "later_2025_2026": later,
                      "fresh_direct_xperp": fresh_direct,
                      "untouched_forward": forward,
                      "untouched_forward_coverage": forward_coverage}, indent=2), flush=True)


if __name__ == "__main__":
    main()
