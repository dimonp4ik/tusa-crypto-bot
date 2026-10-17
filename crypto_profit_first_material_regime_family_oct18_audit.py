"""Profit-first material crypto router by BTC regime and signal family."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

import crypto_05_2r_precision_barbell_oct5_audit as legacy
import crypto_high_target_tree_oct5_audit as base
import crypto_profit_first_material_barbell_oct18_audit as prior
import crypto_profit_first_material_regime_oct18_audit as regime_audit
import crypto_profit_first_material_target_oct18_audit as target_audit
from crypto_runtime_shadow_portfolio_oct3_audit import runtime_gate


ROOT = Path("reports/audit_2026_09_08")
OUT = ROOT / "crypto_profit_first_material_regime_family_oct18.json"
ROWS_OUT = ROOT / "crypto_profit_first_material_regime_family_oct18.pkl"
FRESH_OUT = ROOT / "crypto_profit_first_material_regime_family_xperp_oct17_fresh.pkl"
FORWARD_OUT = ROOT / "crypto_profit_first_material_regime_family_xperp_oct17_forward.pkl"
TARGET_R = 2.0
FRACTIONS = regime_audit.FRACTIONS
QUANTILES = regime_audit.QUANTILES
MODEL = target_audit.MODEL
FIT, SELECT, CONFIRM = prior.FIT, prior.SELECT, prior.CONFIRM
VALIDATION, LATER = prior.VALIDATION, prior.LATER
RAILS = {"loss_pause": 3, "hourly_cap": 2}


def key(regime: str, family: str) -> str:
    return f"{regime}|{family}"


def route_rows(frame: pd.DataFrame, periods: tuple[str, ...],
               routes: dict[str, dict]) -> pd.DataFrame:
    staged = base.stage(frame, periods)
    parts = []
    for route in routes.values():
        part = staged[
            staged.btc_regime.eq(route["btc_regime"])
            & staged.family.eq(route["family"])
            & (staged.model_probability >= float(route["threshold"]))].copy()
        if len(part):
            parts.append(target_audit.barbell_frame(
                part, float(route["near_fraction"]), TARGET_R))
    if not parts:
        return staged.iloc[0:0].copy()
    chosen = pd.concat(parts, ignore_index=True)
    return pd.DataFrame(runtime_gate(base.standardized(
        chosen, "profit_first_material_regime_family"), **RAILS))


def detailed(frame: pd.DataFrame, periods: tuple[str, ...],
             routes: dict[str, dict]) -> dict:
    return {
        "all": base.evaluate(route_rows(frame, periods, routes).to_dict("records")),
        "periods": {period: base.evaluate(
            route_rows(frame, (period,), routes).to_dict("records"))
            for period in periods},
    }


def route_ok(selection: dict, confirmation: dict) -> bool:
    periods = [selection["all"], *confirmation["periods"].values()]
    return bool(selection["all"]["n"] >= 20
                and confirmation["all"]["n"] >= 30
                and all(item["return_pct"] > 0 for item in periods)
                and max(item["dd_pct"] for item in periods) <= 10)


def select_routes(scored: pd.DataFrame) -> tuple[dict, dict]:
    development = base.stage(scored, (*SELECT, *CONFIRM))
    groups = sorted(set(zip(development.btc_regime, development.family)))
    routes, frontiers = {}, {}
    for btc_regime, family in groups:
        group = scored[
            scored.btc_regime.eq(btc_regime) & scored.family.eq(family)].copy()
        scores = base.stage(group, SELECT).model_probability
        candidates = []
        for quantile in QUANTILES:
            threshold = float(scores.quantile(quantile))
            for fraction in FRACTIONS:
                route = {key(btc_regime, family): {
                    "btc_regime": btc_regime, "family": family,
                    "threshold": threshold, "near_fraction": fraction}}
                selection = detailed(group, SELECT, route)
                confirmation = detailed(group, CONFIRM, route)
                if not route_ok(selection, confirmation):
                    continue
                candidates.append({
                    "btc_regime": btc_regime, "family": family,
                    "selection_quantile": quantile, "threshold": threshold,
                    "near_fraction": fraction,
                    "selection_2022h1": selection,
                    "confirmation_2022h2_2023": confirmation,
                    "score": prior.development_score(selection, confirmation),
                })
        candidates.sort(key=lambda item: item["score"], reverse=True)
        route_key = key(btc_regime, family)
        frontiers[route_key] = candidates[:10]
        if candidates:
            best = candidates[0]
            routes[route_key] = {field: best[field] for field in (
                "btc_regime", "family", "selection_quantile", "threshold",
                "near_fraction", "score")}
    return routes, frontiers


def score_direct(frame: pd.DataFrame, model, schema: dict,
                 routes: dict[str, dict], rows_out: Path | None = None) -> tuple[dict, dict]:
    eligible = frame[
        frame.symbol.isin(base.ROBUST_SYMBOLS)
        & frame.risk_pct.between(legacy.RISK_PCT_MIN, legacy.RISK_PCT_MAX,
                                 inclusive="both")].copy()
    scored = legacy.scored(eligible, model, schema)
    scored["period"] = "direct"
    rows = route_rows(scored, ("direct",), routes)
    if rows_out is not None:
        rows.to_pickle(rows_out)
    stamp = pd.to_datetime(eligible.entry_time, unit="s", utc=True)
    return base.evaluate(rows.to_dict("records")), {
        "rows_before_score_filter": len(eligible), "accepted": len(rows),
        "first_entry": stamp.min().isoformat() if len(stamp) else None,
        "last_entry": stamp.max().isoformat() if len(stamp) else None,
        "calendar_months_present": sorted(stamp.dt.strftime("%Y-%m").unique().tolist()),
    }


def main() -> None:
    frame = target_audit.load_joined(TARGET_R)
    fit = base.stage(frame, FIT)
    schema = base.design(fit)
    model = HistGradientBoostingClassifier(
        **MODEL, max_iter=150, l2_regularization=1.0,
        early_stopping=False, random_state=20260908)
    model.fit(base.matrix(fit, schema), fit.outcome.eq("TP2").astype(int))
    scored = legacy.scored(frame, model, schema)
    routes, frontiers = select_routes(scored)
    selection = detailed(scored, SELECT, routes)
    confirmation = detailed(scored, CONFIRM, routes)
    validation = detailed(scored, VALIDATION, routes)
    later = detailed(scored, LATER, routes)
    route_rows(scored, (*SELECT, *CONFIRM, *VALIDATION, *LATER),
               routes).to_pickle(ROWS_OUT)
    old_far, old_near = target_audit.direct_paths(TARGET_R, False)
    old_frame = prior.join_direct_outcomes(old_far, old_near)
    old_direct, old_coverage = score_direct(old_frame, model, schema, routes)
    fresh_far, fresh_near = target_audit.direct_paths(TARGET_R, True)
    fresh_frame = prior.join_direct_outcomes(fresh_far, fresh_near)
    fresh_direct, fresh_coverage = score_direct(
        fresh_frame, model, schema, routes, FRESH_OUT)
    prior_end = float(old_frame.entry_time.max())
    forward_frame = fresh_frame[fresh_frame.entry_time > prior_end].copy()
    forward, forward_coverage = score_direct(
        forward_frame, model, schema, routes, FORWARD_OUT)
    historical = {"selection": selection, "confirmation": confirmation,
                  "validation": validation, "later": later}
    checks = {name: prior.objective_ok(value)
              for name, value in historical.items()}
    status = ("MATERIAL_REGIME_FAMILY_RESEARCH_CANDIDATE"
              if all(checks.values()) else "REJECTED_OBJECTIVE")
    report = {
        "status": status, "ready_for_live_orders": False,
        "protocol": {
            "priority": "account return first; win rate diagnostic only",
            "fit": list(FIT), "route_selection": [*SELECT, *CONFIRM],
            "frozen_validation": list(VALIDATION), "later": list(LATER),
            "router": "separate cutoff and near fraction by BTC regime and signal family",
            "runner_target_r": TARGET_R, "runtime_rails": RAILS,
            "entry": "next 15-minute market open", "orders": "market only",
            "money": "3% current-equity margin, fixed 10x gross",
            "objective": ">=30% account return every month; >=3 trades/day; DD <=10%",
        },
        "model": MODEL, "routes_frozen_before_2024": routes,
        "route_count": len(routes), "development_frontiers": frontiers,
        "selection_2022h1": selection,
        "confirmation_2022h2_2023": confirmation,
        "validation_2024": validation, "later_2025_2026": later,
        "old_direct_xperp": old_direct, "old_direct_coverage": old_coverage,
        "fresh_direct_xperp": fresh_direct, "fresh_direct_coverage": fresh_coverage,
        "untouched_forward": forward, "untouched_forward_coverage": forward_coverage,
        "objective_checks": checks,
        "sources": {
            **json.loads(target_audit.OUT.read_text())["sources"],
            str(Path(__file__)): target_audit.digest(Path(__file__)),
        },
        "limitations": [
            "Regime-family routing increases development choices and overfit risk.",
            "Historical annual labels contain the available six-month research windows.",
            "The post-prior-file direct forward is only about two weeks.",
            "Gross returns exclude fees at user request.",
        ],
    }
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"status": status, "route_count": len(routes),
                      "routes": routes, "selection": selection["all"],
                      "confirmation": confirmation["all"],
                      "validation": validation["all"], "later": later["all"],
                      "fresh_direct": fresh_direct,
                      "untouched_forward": forward}, indent=2), flush=True)


if __name__ == "__main__":
    main()
