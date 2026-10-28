"""Higher-quality stop-guard audit for the frozen crypto Profit-First route."""
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


ROOT = Path("reports/audit_2026_09_08")
SOURCE_REPORT = ROOT / "crypto_profit_first_material_target_oct18.json"
NEAR_DIRECT = ROOT / "crypto_xperp_raw_router_audit_0_5_oct28_material_raw_raw_families.csv"
FAR_DIRECT = ROOT / "crypto_xperp_raw_router_audit_2_0_oct28_material_raw_raw_families.csv"
OUT = ROOT / "crypto_profit_first_stop_guard_oct28.json"
ROWS_OUT = ROOT / "crypto_profit_first_stop_guard_oct28.pkl"
QUANTILES = (.80, .825, .85, .875, .90, .925, .95)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def metrics(rows: pd.DataFrame) -> dict:
    value = base.evaluate(rows.to_dict("records"))
    stops = int(rows.barbell_outcome.eq("SL").sum()) if len(rows) else 0
    value.update({
        "stops": stops,
        "stop_rate_pct": round(100*stops/len(rows), 4) if len(rows) else 0.0,
    })
    return value


def accepted(scored: pd.DataFrame, periods: tuple[str, ...],
             threshold: float) -> pd.DataFrame:
    return frozen.accepted(scored, periods, threshold, .05, 2.0)


def compact(value: dict) -> dict:
    keys = ("n", "return_pct", "dd_pct", "wr", "negative_months",
            "max_loss_streak", "trades_per_calendar_day", "stops", "stop_rate_pct")
    return {key: value[key] for key in keys}


def main() -> None:
    source = json.loads(SOURCE_REPORT.read_text(encoding="utf-8"))
    policy = source["frozen_before_2024"]
    baseline_threshold = float(policy["threshold"])
    frame = frozen.load_joined(2.0)
    fit = base.stage(frame, frozen.FIT)
    schema = base.design(fit)
    model = HistGradientBoostingClassifier(
        **frozen.MODEL, max_iter=150, l2_regularization=1.0,
        early_stopping=False, random_state=20260908)
    model.fit(base.matrix(fit, schema), fit.outcome.eq("TP2").astype(int))
    scored = legacy.scored(frame, model, schema)
    selection_scores = base.stage(scored, frozen.SELECT).model_probability
    thresholds = [(q, float(selection_scores.quantile(q))) for q in QUANTILES]
    stages = {
        "selection_2022h1": frozen.SELECT,
        "confirmation_2022h2_2023": frozen.CONFIRM,
        "validation_2024": frozen.VALIDATION,
        "later_2025_2026": frozen.LATER,
    }
    baseline = {name: metrics(accepted(scored, values, baseline_threshold))
                for name, values in stages.items()}
    candidates = []
    for quantile, threshold in thresholds:
        values = {name: metrics(accepted(scored, periods, threshold))
                  for name, periods in stages.items()}
        development_pass = all(
            values[name]["return_pct"] > baseline[name]["return_pct"]
            and values[name]["stop_rate_pct"] < baseline[name]["stop_rate_pct"]
            and values[name]["dd_pct"] <= baseline[name]["dd_pct"] + .5
            and values[name]["trades_per_calendar_day"] >= 3
            for name in ("selection_2022h1", "confirmation_2022h2_2023"))
        candidates.append({
            "selection_quantile": quantile, "threshold": threshold,
            "development_pass": development_pass,
            "development_min_return_gain": min(
                values[name]["return_pct"]-baseline[name]["return_pct"]
                for name in ("selection_2022h1", "confirmation_2022h2_2023")),
            "stages": {name: compact(value) for name, value in values.items()},
        })
    qualified = [item for item in candidates if item["development_pass"]]
    qualified.sort(key=lambda item: item["development_min_return_gain"], reverse=True)
    selected = qualified[0] if qualified else None
    historical_holdout_pass = bool(selected and all(
        selected["stages"][name]["return_pct"] >= baseline[name]["return_pct"]
        and selected["stages"][name]["stop_rate_pct"] < baseline[name]["stop_rate_pct"]
        and selected["stages"][name]["dd_pct"] <= baseline[name]["dd_pct"] + .5
        for name in ("validation_2024", "later_2025_2026")))

    direct = joiner.join_direct_outcomes(FAR_DIRECT, NEAR_DIRECT)
    eligible = direct[
        direct.symbol.isin(base.ROBUST_SYMBOLS)
        & direct.risk_pct.between(legacy.RISK_PCT_MIN, legacy.RISK_PCT_MAX,
                                  inclusive="both")].copy()
    direct_scored = legacy.scored(eligible, model, schema).assign(period="direct")
    direct_baseline = metrics(accepted(direct_scored, ("direct",), baseline_threshold))
    direct_candidate = None
    direct_pass = False
    if selected:
        rows = accepted(direct_scored, ("direct",), float(selected["threshold"]))
        rows.to_pickle(ROWS_OUT)
        direct_candidate = metrics(rows)
        direct_pass = bool(
            direct_candidate["return_pct"] >= direct_baseline["return_pct"]
            and direct_candidate["stop_rate_pct"] < direct_baseline["stop_rate_pct"]
            and direct_candidate["dd_pct"] <= direct_baseline["dd_pct"] + .5
            and direct_candidate["trades_per_calendar_day"] >= 3)
    accepted_all = bool(historical_holdout_pass and direct_pass)
    report = {
        "status": ("RESEARCH_CANDIDATE_AWAIT_FRESH_FORWARD" if accepted_all
                   else "REJECTED_HOLDOUT" if selected
                   else "REJECTED_DEVELOPMENT"),
        "ready_for_live_orders": False, "orders_possible": False,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "protocol": {
            "change": "raise only the frozen HGB probability threshold",
            "unchanged": "entries, 0.5R/2R exit, 5% first bank, portfolio rails",
            "selection": ["2022h1", "2022h2-2023"],
            "holdouts": ["2024", "2025-2026", "direct X-Perp through Oct27"],
            "money": "3% current-equity margin; fixed 10x; market only",
        },
        "grid": {"candidate_thresholds": len(candidates),
                 "selection_quantiles": list(QUANTILES)},
        "baseline_threshold": baseline_threshold,
        "baseline": {name: compact(value) for name, value in baseline.items()},
        "selected_before_holdouts": selected,
        "historical_holdout_pass": historical_holdout_pass,
        "direct_xperp": {"baseline": compact(direct_baseline),
                         "candidate": (None if direct_candidate is None
                                       else compact(direct_candidate)),
                         "pass": direct_pass},
        "top_development": sorted(
            candidates, key=lambda x: x["development_min_return_gain"],
            reverse=True),
        "sources": {str(SOURCE_REPORT): digest(SOURCE_REPORT),
                    str(NEAR_DIRECT): digest(NEAR_DIRECT),
                    str(FAR_DIRECT): digest(FAR_DIRECT),
                    str(Path(__file__).resolve()): digest(Path(__file__).resolve())},
        "limitations": [
            "Current direct X-Perp history begins in April 2026.",
            "A threshold grid still creates multiple-testing risk.",
            "Public OHLC cannot reproduce order-book slippage or latency.",
            "Gross results exclude fees at user request.",
        ],
    }
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({key: report[key] for key in (
        "status", "grid", "baseline_threshold", "selected_before_holdouts",
        "historical_holdout_pass", "direct_xperp")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
