"""Chronological ensemble audit for the two material crypto routers."""
from __future__ import annotations

import json
from pathlib import Path

from sklearn.ensemble import HistGradientBoostingClassifier

import crypto_05_2r_precision_barbell_oct5_audit as legacy
import crypto_high_target_tree_oct5_audit as base
import crypto_profit_first_material_barbell_oct18_audit as prior
import crypto_profit_first_material_regime_family_oct18_audit as family_audit
import crypto_profit_first_material_target_oct18_audit as target_audit


ROOT = Path("reports/audit_2026_09_08")
OUT = ROOT / "crypto_profit_first_material_ensemble_oct18.json"
ROWS_OUT = ROOT / "crypto_profit_first_material_ensemble_oct18.pkl"
FRESH_OUT = ROOT / "crypto_profit_first_material_ensemble_xperp_oct17_fresh.pkl"
FORWARD_OUT = ROOT / "crypto_profit_first_material_ensemble_xperp_oct17_forward.pkl"
BROAD_REPORT = ROOT / "crypto_profit_first_material_regime_oct18.json"
FAMILY_REPORT = ROOT / "crypto_profit_first_material_regime_family_oct18.json"
MODEL = target_audit.MODEL
FIT, SELECT, CONFIRM = prior.FIT, prior.SELECT, prior.CONFIRM
VALIDATION, LATER = prior.VALIDATION, prior.LATER
TARGET_R = 2.0


def normalized(route: dict, regime: str, family: str) -> dict:
    return {"btc_regime": regime, "family": family,
            "threshold": float(route["threshold"]),
            "near_fraction": float(route["near_fraction"])}


def policy_maps(frame, broad: dict, family: dict) -> dict[str, dict]:
    families = sorted(frame.family.unique())
    broad_map = {
        family_audit.key(regime, signal_family): normalized(
            route, regime, signal_family)
        for regime, route in broad.items() for signal_family in families
    }
    family_map = {
        name: normalized(route, route["btc_regime"], route["family"])
        for name, route in family.items()
    }
    union, intersection = {}, {}
    for name in sorted(set(broad_map) | set(family_map)):
        wide, exact = broad_map.get(name), family_map.get(name)
        if wide and exact:
            union[name] = dict(wide if wide["threshold"] <= exact["threshold"] else exact)
            intersection[name] = dict(
                wide if wide["threshold"] >= exact["threshold"] else exact)
        else:
            union[name] = dict(wide or exact)
    return {"broad_regime": broad_map, "regime_family": family_map,
            "union": union, "intersection": intersection}


def development_ok(selection: dict, confirmation: dict) -> bool:
    a, b = selection["all"], confirmation["all"]
    return bool(a["n"] >= 100 and b["n"] >= 100
                and a["return_pct"] > 0 and b["return_pct"] > 0
                and a["dd_pct"] <= 10 and b["dd_pct"] <= 10
                and a["trades_per_calendar_day"] >= 3
                and b["trades_per_calendar_day"] >= 2
                and all(item["return_pct"] > 0
                        for item in confirmation["periods"].values()))


def main() -> None:
    broad_report = json.loads(BROAD_REPORT.read_text())
    family_report = json.loads(FAMILY_REPORT.read_text())
    frame = target_audit.load_joined(TARGET_R)
    fit = base.stage(frame, FIT)
    schema = base.design(fit)
    model = HistGradientBoostingClassifier(
        **MODEL, max_iter=150, l2_regularization=1.0,
        early_stopping=False, random_state=20260908)
    model.fit(base.matrix(fit, schema), fit.outcome.eq("TP2").astype(int))
    scored = legacy.scored(frame, model, schema)
    policies = policy_maps(
        scored, broad_report["routes_frozen_before_2024"],
        family_report["routes_frozen_before_2024"])
    candidates = []
    for name, routes in policies.items():
        selection = family_audit.detailed(scored, SELECT, routes)
        confirmation = family_audit.detailed(scored, CONFIRM, routes)
        if not development_ok(selection, confirmation):
            continue
        candidates.append({
            "policy": name, "routes": routes,
            "selection_2022h1": selection,
            "confirmation_2022h2_2023": confirmation,
            "score": prior.development_score(selection, confirmation),
        })
    candidates.sort(key=lambda item: item["score"], reverse=True)
    frozen = candidates[0] if candidates else None
    validation = later = old_direct = fresh_direct = forward = None
    old_coverage = fresh_coverage = forward_coverage = checks = None
    status = "REJECTED_DEVELOPMENT"
    if frozen:
        routes = frozen["routes"]
        validation = family_audit.detailed(scored, VALIDATION, routes)
        later = family_audit.detailed(scored, LATER, routes)
        family_audit.route_rows(
            scored, (*SELECT, *CONFIRM, *VALIDATION, *LATER), routes).to_pickle(ROWS_OUT)
        old_far, old_near = target_audit.direct_paths(TARGET_R, False)
        old_frame = prior.join_direct_outcomes(old_far, old_near)
        old_direct, old_coverage = family_audit.score_direct(
            old_frame, model, schema, routes)
        fresh_far, fresh_near = target_audit.direct_paths(TARGET_R, True)
        fresh_frame = prior.join_direct_outcomes(fresh_far, fresh_near)
        fresh_direct, fresh_coverage = family_audit.score_direct(
            fresh_frame, model, schema, routes, FRESH_OUT)
        prior_end = float(old_frame.entry_time.max())
        forward_frame = fresh_frame[fresh_frame.entry_time > prior_end].copy()
        forward, forward_coverage = family_audit.score_direct(
            forward_frame, model, schema, routes, FORWARD_OUT)
        historical = {"selection": frozen["selection_2022h1"],
                      "confirmation": frozen["confirmation_2022h2_2023"],
                      "validation": validation, "later": later}
        checks = {name: prior.objective_ok(value)
                  for name, value in historical.items()}
        status = ("MATERIAL_ENSEMBLE_RESEARCH_CANDIDATE"
                  if all(checks.values()) else "REJECTED_OBJECTIVE")
    report = {
        "status": status, "ready_for_live_orders": False,
        "protocol": {
            "priority": "account return first; win rate diagnostic only",
            "fit": list(FIT), "policy_selection": [*SELECT, *CONFIRM],
            "frozen_validation": list(VALIDATION), "later": list(LATER),
            "policies": list(policies),
            "union": "lower development-frozen cutoff per regime-family",
            "intersection": "higher development-frozen cutoff; missing family routes excluded",
            "entry": "next 15-minute market open", "orders": "market only",
            "money": "3% current-equity margin, fixed 10x gross",
            "objective": ">=30% account return every month; >=3 trades/day; DD <=10%",
        },
        "development_candidates": [{key: value for key, value in item.items()
                                    if key != "routes"} for item in candidates],
        "frozen_before_2024": frozen,
        "validation_2024": validation, "later_2025_2026": later,
        "old_direct_xperp": old_direct, "old_direct_coverage": old_coverage,
        "fresh_direct_xperp": fresh_direct, "fresh_direct_coverage": fresh_coverage,
        "untouched_forward": forward, "untouched_forward_coverage": forward_coverage,
        "objective_checks": checks,
        "sources": {
            str(BROAD_REPORT): target_audit.digest(BROAD_REPORT),
            str(FAMILY_REPORT): target_audit.digest(FAMILY_REPORT),
            str(Path(__file__)): target_audit.digest(Path(__file__)),
        },
        "limitations": [
            "The ensemble chooses among four fixed combinations on 2022-2023.",
            "The direct post-prior-file window is only about two weeks.",
            "Gross returns exclude fees at user request.",
        ],
    }
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"status": status,
                      "development": [{"policy": item["policy"],
                                       "score": item["score"]}
                                      for item in candidates],
                      "frozen": frozen, "validation": validation,
                      "later": later, "fresh_direct": fresh_direct,
                      "untouched_forward": forward}, indent=2), flush=True)


if __name__ == "__main__":
    main()
