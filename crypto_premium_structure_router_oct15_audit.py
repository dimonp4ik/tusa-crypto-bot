"""Chronological audit of regime-specific structure/liquidity interactions."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

import crypto_5m_regime_router_literal_oct6_audit as price
import crypto_futures_funding_regime_oct6_audit as funding
import crypto_premium_dislocation_oct15_audit as premium
import crypto_premium_exit_geometry_oct15_audit as geometry
import crypto_premium_stop_causes_oct15_audit as causes


ROOT = Path("reports/audit_2026_09_08")
PARENT = ROOT / "crypto_premium_dislocation_oct15.json"
GEOMETRY = ROOT / "crypto_premium_exit_geometry_oct15.json"
OUT = ROOT / "crypto_premium_structure_router_oct15.json"
QUANTILES = (.50, .60, .70, .75, .80, .85, .90)
CONFIGS = (
    {"max_leaf_nodes": 15, "min_samples_leaf": 300, "l2": 3.0},
    {"max_leaf_nodes": 31, "min_samples_leaf": 500, "l2": 5.0},
    {"max_leaf_nodes": 31, "min_samples_leaf": 300, "l2": 8.0},
)
STRUCTURE_FEATURES = (
    "supportive_sweep", "adverse_rejection", "target_blocked",
    "directed_range_position", "low_volume", "shock_bar",
    "premium_accelerating_against", "premium_extreme",
    "price_chasing_edge", "against_btc_6h", "crowded_timestamp",
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def structure_matrix(frame, base_matrix: np.ndarray) -> np.ndarray:
    extra = np.column_stack([
        frame[name].astype(float).to_numpy() for name in STRUCTURE_FEATURES
    ]).astype(np.float32)
    return np.hstack((base_matrix, extra))


def gate(value: dict, validation: bool = False) -> bool:
    return bool(value["n"] >= (300 if validation else 500)
                and value["return_pct"] > 0
                and value["dd_pct"] <= 8
                and value["trades_per_calendar_day"] >= (2.0 if validation else 2.5))


def public(item: dict) -> dict:
    return {key: value for key, value in item.items()
            if key not in ("_frame", "_model")}


def main() -> None:
    parent_raw = PARENT.read_bytes()
    geometry_raw = GEOMETRY.read_bytes()
    parent = json.loads(parent_raw)
    geometry_report = json.loads(geometry_raw)
    chosen_geometry = geometry_report["frozen_geometry"]["config"]
    candles, _, candle_sources = funding.load()
    premium_data, premium_sources = premium.load_premium()
    raw = price.build_features(candles)
    enriched = premium.attach(raw, premium_data)
    proposals = premium.proposals(enriched)
    frame = geometry.add_outcomes(
        proposals, candles, chosen_geometry["target_pct"],
        chosen_geometry["stop_pct"], chosen_geometry["max_bars"])
    frame = causes.causal_structure(frame, candles)
    fit = frame.year.isin((2020, 2021)).to_numpy()
    selection_mask = frame.year.eq(2022).to_numpy()
    schema = price.columns(frame.loc[fit])
    _, base_matrix = premium.matrices(frame, schema)
    matrices = {
        "premium_price_control": base_matrix,
        "premium_price_structure": structure_matrix(frame, base_matrix),
    }
    target = frame.target_hit.astype(int).to_numpy()
    frontier, eligible = [], []
    for kind, matrix in matrices.items():
        for config in CONFIGS:
            model = HistGradientBoostingClassifier(
                max_iter=180, learning_rate=.06,
                max_leaf_nodes=config["max_leaf_nodes"],
                min_samples_leaf=config["min_samples_leaf"],
                l2_regularization=config["l2"], random_state=20261015)
            model.fit(matrix[fit], target[fit])
            scored = frame.copy()
            scored["quality"] = model.predict_proba(matrix)[:, 1]
            for quantile in QUANTILES:
                threshold = float(scored.loc[selection_mask, "quality"].quantile(quantile))
                item = {
                    "model_kind": kind, "model": config, "quantile": quantile,
                    "threshold": threshold,
                    "selection_2022": price.evaluate(scored, (2022,), threshold),
                }
                frontier.append(item)
                if not gate(item["selection_2022"]):
                    continue
                item["validation_2023"] = price.evaluate(scored, (2023,), threshold)
                item["validation_2024"] = price.evaluate(scored, (2024,), threshold)
                if not (gate(item["validation_2023"], True)
                        and gate(item["validation_2024"], True)):
                    continue
                item["score"] = round(min(
                    item[name]["return_pct"]-item[name]["dd_pct"]
                    for name in ("selection_2022", "validation_2023", "validation_2024")), 8)
                item["_frame"], item["_model"] = scored, model
                eligible.append(item)
    eligible.sort(key=lambda item: (
        item["score"],
        min(item[name]["literal_target_hit_rate"] for name in (
            "selection_2022", "validation_2023", "validation_2024"))), reverse=True)
    frozen_by_kind = {}
    for kind in matrices:
        matches = [item for item in eligible if item["model_kind"] == kind]
        if matches:
            frozen_by_kind[kind] = matches[0]
    results = {}
    for kind, item in frozen_by_kind.items():
        results[kind] = {str(year): price.evaluate(
            item["_frame"], (year,), item["threshold"])
            for year in (2022, 2023, 2024, 2025, 2026)}
    structure_result = results.get("premium_price_structure")
    control_result = results.get("premium_price_control")
    confirmation_better = bool(structure_result and control_result and all(
        structure_result[str(year)]["return_pct"] > control_result[str(year)]["return_pct"]
        and structure_result[str(year)]["dd_pct"] <= control_result[str(year)]["dd_pct"]
        for year in (2025, 2026)))
    objective_checks = ({str(year): price.objective_ok(value)
                         for year, value in structure_result.items()}
                        if structure_result else None)
    status = ("OBJECTIVE_PASS" if objective_checks and all(objective_checks.values())
              else "RESEARCH_CANDIDATE" if confirmation_better
              else "REJECTED_CONFIRMATION")
    report = {
        "status": status, "ready_for_live_orders": False,
        "decision": "No runtime change unless chronological objectives pass.",
        "protocol": {
            "fit": [2020, 2021], "selection": 2022,
            "validation": [2023, 2024], "confirmation_only": [2025, 2026],
            "entry": "next 5m market open", "geometry": chosen_geometry,
            "ambiguity": "stop first", "features": STRUCTURE_FEATURES,
            "position": "3% equity margin, fixed 10x",
        },
        "rows": len(frame), "fit_target_prevalence_pct": round(100*target[fit].mean(), 4),
        "qualified_before_confirmation": len(eligible),
        "frozen_before_confirmation": {
            kind: public(item) for kind, item in frozen_by_kind.items()},
        "results": results, "structure_beats_control_2025_2026": confirmation_better,
        "objective_checks": objective_checks,
        "selection_frontier": [public(item) for item in sorted(
            frontier, key=lambda x: x["selection_2022"]["return_pct"], reverse=True)[:24]],
        "sources": {
            str(PARENT): hashlib.sha256(parent_raw).hexdigest(),
            str(GEOMETRY): hashlib.sha256(geometry_raw).hexdigest(),
            **candle_sources, **premium_sources,
            str(Path(__file__)): digest(Path(__file__)),
        },
        "limitations": [
            "Historical order-book depth is unavailable; candle volume is a liquidity proxy.",
            "Confirmation years were observed by earlier research and are robustness evidence.",
            "Binance futures remain a proxy for OKX X-Perp execution.",
        ],
    }
    OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(json.dumps({
        "status": status, "qualified": len(eligible),
        "structure_beats_control": confirmation_better,
        "results": results,
    }, indent=2))


if __name__ == "__main__":
    main()
