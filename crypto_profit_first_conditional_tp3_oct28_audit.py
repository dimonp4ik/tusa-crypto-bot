"""Conditional 3R remainder for the frozen crypto Profit-First route.

An independent model learns, before 2022, which entries that reach 2R continue
to 3R.  Only selected 2R winners retain a small protected remainder.  All
other entries and exits stay exactly on the frozen policy.
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
import crypto_profit_first_third_target_oct28_audit as prior
from crypto_runtime_shadow_portfolio_oct3_audit import runtime_gate


ROOT = Path("reports/audit_2026_09_08")
OUT = ROOT / "crypto_profit_first_conditional_tp3_oct28.json"
ROWS_OUT = ROOT / "crypto_profit_first_conditional_tp3_oct28.pkl"
SOURCE_REPORT = ROOT / "crypto_profit_first_material_target_oct18.json"
THRESHOLD_QUANTILES = (.50, .60, .70, .80, .90)
REMAINDERS = (.05, .10, .20)
MODEL = {"max_leaf_nodes": 7, "min_samples_leaf": 200,
         "learning_rate": .05}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def policy_rows(frame: pd.DataFrame, score_threshold: float | None,
                remainder: float) -> pd.DataFrame:
    result = frame.copy()
    near_hit = result.near_outcome.eq("TP2")
    two_hit = result.outcome.eq("TP2")
    third_hit = result.third_outcome.eq("TP2")
    result["net_r"] = np.where(
        ~near_hit, -1.0,
        np.where(two_hit, .05*.5 + .95*2.0, .05*.5))
    result["conditional_tp3"] = False
    if score_threshold is not None:
        selected = two_hit & result.continuation_score.ge(score_threshold)
        result.loc[selected, "conditional_tp3"] = True
        result.loc[selected, "net_r"] += np.where(
            third_hit[selected], remainder, -2*remainder)
        result.loc[selected, "exit_time"] = result.loc[selected, "third_exit_time"]
    return pd.DataFrame(runtime_gate(base.standardized(
        result, "profit_first_conditional_tp3")))


def metric(rows: pd.DataFrame) -> dict:
    value = base.evaluate(rows.to_dict("records"))
    selected = int(rows.conditional_tp3.sum()) if len(rows) else 0
    value.update({
        "conditional_tp3": selected,
        "conditional_share_pct": round(100*selected/len(rows), 4) if len(rows) else 0.0,
    })
    return value


def compact(value: dict) -> dict:
    keys = ("n", "return_pct", "dd_pct", "wr", "negative_months",
            "max_loss_streak", "trades_per_calendar_day",
            "conditional_tp3", "conditional_share_pct")
    return {key: value[key] for key in keys}


def main() -> None:
    source = json.loads(SOURCE_REPORT.read_text(encoding="utf-8"))
    entry_threshold = float(source["frozen_before_2024"]["threshold"])
    two = frozen.load_joined(2.0)
    third = base.load_target(3.0)
    joined = prior.attach_third(two, third, prior.HIST_KEYS)

    entry_fit = base.stage(two, frozen.FIT)
    entry_schema = base.design(entry_fit)
    entry_model = HistGradientBoostingClassifier(
        **frozen.MODEL, max_iter=150, l2_regularization=1.0,
        early_stopping=False, random_state=20260908)
    entry_model.fit(base.matrix(entry_fit, entry_schema),
                    entry_fit.outcome.eq("TP2").astype(int))
    joined = legacy.scored(joined, entry_model, entry_schema)
    joined = joined[joined.model_probability.ge(entry_threshold)].copy()

    continuation_fit = base.stage(joined, frozen.FIT)
    continuation_fit = continuation_fit[continuation_fit.outcome.eq("TP2")].copy()
    continuation_schema = base.design(continuation_fit)
    continuation_model = HistGradientBoostingClassifier(
        **MODEL, max_iter=150, l2_regularization=2.0,
        early_stopping=False, random_state=20261028)
    continuation_model.fit(
        base.matrix(continuation_fit, continuation_schema),
        continuation_fit.third_outcome.eq("TP2").astype(int))
    joined["continuation_score"] = continuation_model.predict_proba(
        base.matrix(joined, continuation_schema))[:, 1]
    selection_scores = base.stage(joined, frozen.SELECT)
    selection_scores = selection_scores[selection_scores.outcome.eq("TP2")]
    thresholds = [(q, float(selection_scores.continuation_score.quantile(q)))
                  for q in THRESHOLD_QUANTILES]
    stages = {
        "selection_2022h1": frozen.SELECT,
        "confirmation_2022h2_2023": frozen.CONFIRM,
        "validation_2024": frozen.VALIDATION,
        "later_2025_2026": frozen.LATER,
    }
    baseline = {name: metric(policy_rows(base.stage(joined, periods), None, 0.0))
                for name, periods in stages.items()}
    candidates = []
    for quantile, threshold in thresholds:
        for remainder in REMAINDERS:
            values = {name: metric(policy_rows(
                        base.stage(joined, periods), threshold, remainder))
                      for name, periods in stages.items()}
            development_pass = all(
                values[name]["return_pct"] > baseline[name]["return_pct"]
                and values[name]["dd_pct"] <= baseline[name]["dd_pct"] + .35
                and values[name]["negative_months"] <= baseline[name]["negative_months"]
                and values[name]["trades_per_calendar_day"] >= 3
                and values[name]["conditional_tp3"] >= 20
                for name in ("selection_2022h1", "confirmation_2022h2_2023"))
            candidates.append({
                "selection_quantile": quantile,
                "continuation_threshold": threshold,
                "tp3_remainder_fraction": remainder,
                "development_pass": development_pass,
                "development_min_return_gain": min(
                    values[n]["return_pct"]-baseline[n]["return_pct"]
                    for n in ("selection_2022h1", "confirmation_2022h2_2023")),
                "stages": {name: compact(value) for name, value in values.items()},
            })
    qualified = [item for item in candidates if item["development_pass"]]
    qualified.sort(key=lambda item: item["development_min_return_gain"], reverse=True)
    selected = qualified[0] if qualified else None
    historical_pass = bool(selected and all(
        selected["stages"][name]["return_pct"] >= baseline[name]["return_pct"]
        and selected["stages"][name]["dd_pct"] <= baseline[name]["dd_pct"] + .35
        and selected["stages"][name]["negative_months"] <= baseline[name]["negative_months"]
        for name in ("validation_2024", "later_2025_2026")))

    direct_two = joiner.join_direct_outcomes(prior.TWO_DIRECT, prior.NEAR_DIRECT)
    direct_third = pd.read_csv(prior.THIRD_DIRECT[3.0])
    direct = prior.attach_third(direct_two, direct_third, prior.DIRECT_KEYS)
    eligible = direct[
        direct.symbol.isin(base.ROBUST_SYMBOLS)
        & direct.risk_pct.between(legacy.RISK_PCT_MIN, legacy.RISK_PCT_MAX,
                                  inclusive="both")].copy()
    eligible = legacy.scored(eligible, entry_model, entry_schema)
    eligible = eligible[eligible.model_probability.ge(entry_threshold)].copy()
    eligible["continuation_score"] = continuation_model.predict_proba(
        base.matrix(eligible, continuation_schema))[:, 1]
    direct_baseline = metric(policy_rows(eligible, None, 0.0))
    direct_candidate = None
    direct_pass = False
    if selected:
        rows = policy_rows(
            eligible, float(selected["continuation_threshold"]),
            float(selected["tp3_remainder_fraction"]))
        rows.to_pickle(ROWS_OUT)
        direct_candidate = metric(rows)
        direct_pass = bool(
            direct_candidate["return_pct"] >= direct_baseline["return_pct"]
            and direct_candidate["dd_pct"] <= direct_baseline["dd_pct"] + .35
            and direct_candidate["negative_months"] <= direct_baseline["negative_months"]
            and direct_candidate["trades_per_calendar_day"] >= 3)
    accepted = bool(historical_pass and direct_pass)
    report = {
        "status": ("RESEARCH_CANDIDATE_AWAIT_FRESH_FORWARD" if accepted
                   else "REJECTED_HOLDOUT" if selected
                   else "REJECTED_DEVELOPMENT"),
        "ready_for_live_orders": False, "orders_possible": False,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "protocol": {
            "entry_and_base_exit": "unchanged frozen 0.5R/2R Profit-First route",
            "continuation_fit": list(frozen.FIT),
            "selection": ["2022h1", "2022h2-2023"],
            "holdouts": ["2024", "2025-2026", "direct X-Perp through Oct27"],
            "candidate": "retain 5/10/20% after 2R only when causal continuation score passes",
            "money": "3% current-equity margin; fixed 10x; market only",
        },
        "model": MODEL,
        "fit_2r_winners": len(continuation_fit),
        "fit_3r_rate_pct": round(100*float(
            continuation_fit.third_outcome.eq("TP2").mean()), 4),
        "grid": {"candidates": len(candidates),
                 "threshold_quantiles": list(THRESHOLD_QUANTILES),
                 "remainder_fractions": list(REMAINDERS)},
        "baseline": {name: compact(value) for name, value in baseline.items()},
        "selected_before_holdouts": selected,
        "historical_holdout_pass": historical_pass,
        "direct_xperp": {"baseline": compact(direct_baseline),
                         "candidate": (None if direct_candidate is None
                                       else compact(direct_candidate)),
                         "pass": direct_pass},
        "top_development": sorted(
            candidates, key=lambda x: x["development_min_return_gain"],
            reverse=True)[:10],
        "sources": {
            str(SOURCE_REPORT): digest(SOURCE_REPORT),
            str(prior.NEAR_DIRECT): digest(prior.NEAR_DIRECT),
            str(prior.TWO_DIRECT): digest(prior.TWO_DIRECT),
            str(prior.THIRD_DIRECT[3.0]): digest(prior.THIRD_DIRECT[3.0]),
            str(Path(__file__).resolve()): digest(Path(__file__).resolve()),
        },
        "limitations": [
            "The continuation decision uses entry-time features only.",
            "Current direct X-Perp history begins in April 2026.",
            "Public OHLC cannot reproduce order-book slippage or latency.",
            "Gross results exclude fees at user request.",
        ],
    }
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({key: report[key] for key in (
        "status", "fit_2r_winners", "fit_3r_rate_pct", "grid",
        "selected_before_holdouts", "historical_holdout_pass", "direct_xperp")},
        indent=2), flush=True)


if __name__ == "__main__":
    main()
