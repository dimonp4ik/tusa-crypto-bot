"""Regime-routed profit-first 0.5R/2R material crypto audit.

One classifier fits 2020-2021. Each causal BTC regime selects its own score
cutoff and near fraction on 2022-2023. The combined router then freezes before
2024-2026 and direct X-Perp evaluation.
"""
from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

import crypto_05_2r_precision_barbell_oct5_audit as legacy
import crypto_high_target_tree_oct5_audit as base
import crypto_profit_first_material_barbell_oct18_audit as prior
import crypto_profit_first_material_target_oct18_audit as target_audit
from crypto_runtime_shadow_portfolio_oct3_audit import runtime_gate


ROOT = Path("reports/audit_2026_09_08")
OUT = ROOT / "crypto_profit_first_material_regime_oct18.json"
ROWS_OUT = ROOT / "crypto_profit_first_material_regime_oct18.pkl"
FRESH_OUT = ROOT / "crypto_profit_first_material_regime_xperp_oct17_fresh.pkl"
FORWARD_OUT = ROOT / "crypto_profit_first_material_regime_xperp_oct17_forward.pkl"
TARGET_R = 2.0
FRACTIONS = (0.0, .05, .10, .20)
QUANTILES = (.50, .60, .70, .75, .80, .85, .90, .925)
MODEL = target_audit.MODEL
FIT, SELECT, CONFIRM = prior.FIT, prior.SELECT, prior.CONFIRM
VALIDATION, LATER = prior.VALIDATION, prior.LATER


def route_rows(frame: pd.DataFrame, periods: tuple[str, ...],
               routes: dict[str, dict], *, loss_pause: int = 3,
               hourly_cap: int = 2) -> pd.DataFrame:
    staged = base.stage(frame, periods)
    parts = []
    for regime, route in routes.items():
        part = staged[
            staged.btc_regime.eq(regime)
            & (staged.model_probability >= float(route["threshold"]))].copy()
        if len(part):
            parts.append(target_audit.barbell_frame(
                part, float(route["near_fraction"]), TARGET_R))
    if not parts:
        return staged.iloc[0:0].copy()
    chosen = pd.concat(parts, ignore_index=True)
    return pd.DataFrame(runtime_gate(base.standardized(
        chosen, "profit_first_material_regime"),
        loss_pause=loss_pause, hourly_cap=hourly_cap))


def detailed(frame: pd.DataFrame, periods: tuple[str, ...],
             routes: dict[str, dict], *, loss_pause: int = 3,
             hourly_cap: int = 2) -> dict:
    return {
        "all": base.evaluate(route_rows(
            frame, periods, routes, loss_pause=loss_pause,
            hourly_cap=hourly_cap).to_dict("records")),
        "periods": {
            period: base.evaluate(
                route_rows(frame, (period,), routes, loss_pause=loss_pause,
                           hourly_cap=hourly_cap).to_dict("records"))
            for period in periods
        },
    }


def route_development_ok(selection: dict, confirmation: dict) -> bool:
    periods = [selection["all"], *confirmation["periods"].values()]
    return bool(selection["all"]["n"] >= 30
                and confirmation["all"]["n"] >= 50
                and all(item["return_pct"] > 0 for item in periods)
                and max(item["dd_pct"] for item in periods) <= 10)


def route_score(selection: dict, confirmation: dict) -> float:
    monthly = min(prior.monthly_equivalent(selection),
                  prior.monthly_equivalent(confirmation))
    drawdown = max(selection["all"]["dd_pct"], confirmation["all"]["dd_pct"])
    negative = selection["all"]["negative_months"] + confirmation["all"]["negative_months"]
    return round(monthly - .08 * drawdown - .10 * negative, 8)


def select_routes(scored: pd.DataFrame) -> tuple[dict, dict]:
    routes = {}
    frontiers = {}
    regimes = sorted(base.stage(scored, (*SELECT, *CONFIRM)).btc_regime.unique())
    for regime in regimes:
        regime_frame = scored[scored.btc_regime.eq(regime)].copy()
        selection_scores = base.stage(regime_frame, SELECT).model_probability
        candidates = []
        for quantile in QUANTILES:
            threshold = float(selection_scores.quantile(quantile))
            for fraction in FRACTIONS:
                route = {regime: {"threshold": threshold,
                                  "near_fraction": fraction}}
                selection = detailed(regime_frame, SELECT, route)
                confirmation = detailed(regime_frame, CONFIRM, route)
                if not route_development_ok(selection, confirmation):
                    continue
                candidates.append({
                    "btc_regime": regime, "selection_quantile": quantile,
                    "threshold": threshold, "near_fraction": fraction,
                    "selection_2022h1": selection,
                    "confirmation_2022h2_2023": confirmation,
                    "score": route_score(selection, confirmation),
                })
        candidates.sort(key=lambda item: item["score"], reverse=True)
        frontiers[regime] = candidates[:10]
        if candidates:
            routes[regime] = {key: candidates[0][key] for key in (
                "selection_quantile", "threshold", "near_fraction", "score")}
    return routes, frontiers


def select_route_subset(scored: pd.DataFrame,
                        routes: dict[str, dict]) -> tuple[dict, list[dict]]:
    frontier = []
    names = sorted(routes)
    for size in range(1, len(names) + 1):
        for selected in itertools.combinations(names, size):
            subset = {name: routes[name] for name in selected}
            selection = detailed(scored, SELECT, subset)
            confirmation = detailed(scored, CONFIRM, subset)
            selection_all, confirmation_all = selection["all"], confirmation["all"]
            if not (selection_all["n"] >= 100 and confirmation_all["n"] >= 100
                    and selection_all["return_pct"] > 0
                    and confirmation_all["return_pct"] > 0
                    and selection_all["dd_pct"] <= 10
                    and confirmation_all["dd_pct"] <= 10
                    and selection_all["trades_per_calendar_day"] >= 3
                    and confirmation_all["trades_per_calendar_day"] >= 2
                    and all(item["return_pct"] > 0
                            for item in confirmation["periods"].values())):
                continue
            frontier.append({
                "regimes": list(selected), "selection_2022h1": selection,
                "confirmation_2022h2_2023": confirmation,
                "score": prior.development_score(selection, confirmation),
            })
    frontier.sort(key=lambda item: item["score"], reverse=True)
    if not frontier:
        return routes, []
    chosen = frontier[0]["regimes"]
    return {name: routes[name] for name in chosen}, frontier


def select_runtime_rails(scored: pd.DataFrame,
                         routes: dict[str, dict]) -> tuple[dict, list[dict]]:
    frontier = []
    for loss_pause, hourly_cap in itertools.product((1, 2, 3), (1, 2)):
        selection = detailed(scored, SELECT, routes, loss_pause=loss_pause,
                             hourly_cap=hourly_cap)
        confirmation = detailed(scored, CONFIRM, routes, loss_pause=loss_pause,
                                hourly_cap=hourly_cap)
        a, b = selection["all"], confirmation["all"]
        if not (a["return_pct"] > 0 and b["return_pct"] > 0
                and a["dd_pct"] <= 10 and b["dd_pct"] <= 10
                and a["trades_per_calendar_day"] >= 3
                and b["trades_per_calendar_day"] >= 2
                and all(item["return_pct"] > 0
                        for item in confirmation["periods"].values())):
            continue
        frontier.append({
            "loss_pause": loss_pause, "hourly_cap": hourly_cap,
            "selection_2022h1": selection,
            "confirmation_2022h2_2023": confirmation,
            "score": prior.development_score(selection, confirmation),
        })
    frontier.sort(key=lambda item: item["score"], reverse=True)
    chosen = frontier[0] if frontier else {"loss_pause": 3, "hourly_cap": 2}
    return {"loss_pause": chosen["loss_pause"],
            "hourly_cap": chosen["hourly_cap"]}, frontier


def score_direct(frame: pd.DataFrame, model, schema: dict,
                 routes: dict[str, dict], rails: dict,
                 rows_out: Path | None = None) -> tuple[dict, dict]:
    eligible = frame[
        frame.symbol.isin(base.ROBUST_SYMBOLS)
        & frame.risk_pct.between(legacy.RISK_PCT_MIN, legacy.RISK_PCT_MAX,
                                 inclusive="both")].copy()
    scored = legacy.scored(eligible, model, schema)
    scored["period"] = "direct"
    rows = route_rows(scored, ("direct",), routes, **rails)
    if rows_out is not None:
        rows.to_pickle(rows_out)
    stamp = pd.to_datetime(eligible.entry_time, unit="s", utc=True)
    coverage = {
        "rows_before_score_filter": len(eligible), "accepted": len(rows),
        "first_entry": stamp.min().isoformat() if len(stamp) else None,
        "last_entry": stamp.max().isoformat() if len(stamp) else None,
        "calendar_months_present": sorted(stamp.dt.strftime("%Y-%m").unique().tolist()),
    }
    return base.evaluate(rows.to_dict("records")), coverage


def main() -> None:
    frame = target_audit.load_joined(TARGET_R)
    fit = base.stage(frame, FIT)
    schema = base.design(fit)
    model = HistGradientBoostingClassifier(
        **MODEL, max_iter=150, l2_regularization=1.0,
        early_stopping=False, random_state=20260908)
    model.fit(base.matrix(fit, schema), fit.outcome.eq("TP2").astype(int))
    scored = legacy.scored(frame, model, schema)
    regime_routes, frontiers = select_routes(scored)
    routes, subset_frontier = select_route_subset(scored, regime_routes)
    rails, rails_frontier = select_runtime_rails(scored, routes)
    selection = detailed(scored, SELECT, routes, **rails)
    confirmation = detailed(scored, CONFIRM, routes, **rails)
    validation = detailed(scored, VALIDATION, routes, **rails)
    later = detailed(scored, LATER, routes, **rails)
    route_rows(scored, (*SELECT, *CONFIRM, *VALIDATION, *LATER),
               routes, **rails).to_pickle(ROWS_OUT)

    old_far, old_near = target_audit.direct_paths(TARGET_R, False)
    old_frame = prior.join_direct_outcomes(old_far, old_near)
    old_direct, old_coverage = score_direct(old_frame, model, schema, routes, rails)
    fresh_far, fresh_near = target_audit.direct_paths(TARGET_R, True)
    fresh_frame = prior.join_direct_outcomes(fresh_far, fresh_near)
    fresh_direct, fresh_coverage = score_direct(
        fresh_frame, model, schema, routes, rails, FRESH_OUT)
    prior_end = float(old_frame.entry_time.max())
    forward_frame = fresh_frame[fresh_frame.entry_time > prior_end].copy()
    forward, forward_coverage = score_direct(
        forward_frame, model, schema, routes, rails, FORWARD_OUT)
    historical = {"selection": selection, "confirmation": confirmation,
                  "validation": validation, "later": later}
    checks = {name: prior.objective_ok(value)
              for name, value in historical.items()}
    status = ("MATERIAL_REGIME_RESEARCH_CANDIDATE"
              if all(checks.values()) else "REJECTED_OBJECTIVE")
    report = {
        "status": status, "ready_for_live_orders": False,
        "protocol": {
            "priority": "account return first; win rate diagnostic only",
            "fit": list(FIT), "route_selection": [*SELECT, *CONFIRM],
            "frozen_validation": list(VALIDATION), "later": list(LATER),
            "router": "separate probability cutoff and near fraction by causal BTC regime",
            "runner_target_r": TARGET_R,
            "entry_risk_pct_range": [legacy.RISK_PCT_MIN, legacy.RISK_PCT_MAX],
            "entry": "next 15-minute market open", "orders": "market only",
            "money": "3% current-equity margin, fixed 10x gross",
            "objective": ">=30% account return every month; >=3 trades/day; DD <=10%",
        },
        "model": MODEL, "routes_frozen_before_2024": routes,
        "per_regime_routes_before_subset": regime_routes,
        "development_frontiers": frontiers,
        "route_subset_frontier": subset_frontier,
        "runtime_rails_frozen_before_2024": rails,
        "runtime_rails_frontier": rails_frontier,
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
            "Regime routing adds four development choices and can still overfit 2022-2023.",
            "Historical annual labels contain the available six-month research windows.",
            "The post-prior-file direct forward is only about two weeks.",
            "Gross returns exclude fees at user request.",
        ],
    }
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"status": status, "routes": routes,
                      "selection": selection["all"],
                      "confirmation": confirmation["all"],
                      "validation": validation["all"], "later": later["all"],
                      "fresh_direct": fresh_direct,
                      "untouched_forward": forward}, indent=2), flush=True)


if __name__ == "__main__":
    main()
