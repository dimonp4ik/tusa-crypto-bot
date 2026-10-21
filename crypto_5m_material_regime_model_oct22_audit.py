"""Material 1-2% exit model over causal five-minute regime proposals.

This replaces the micro-target objective with literal market-entered moves.
Models fit on 2020-2021, thresholds and geometry are selected on 2022,
confirmed on 2023, and tested sequentially on 2024 then 2025-2026.
"""
from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import (HistGradientBoostingClassifier,
                              HistGradientBoostingRegressor)

import crypto_5m_regime_router_literal_oct6_audit as price
import crypto_raw_material_direction_model_oct21_audit as metrics
from src import research_crypto_replay as replay


OUT = price.ROOT / "crypto_5m_material_regime_model_oct22.json"
ROWS_OUT = price.ROOT / "crypto_5m_material_regime_model_oct22.pkl"
FIT = (2020, 2021)
SELECTION = (2022,)
CONFIRMATION = (2023,)
TEST = (2024,)
HIDDEN = (2025, 2026)
GEOMETRIES = ((1.0, .75, 144), (1.5, 1.0, 288), (2.0, 1.0, 288))
CONFIGS = ((15, 300, 5.0), (31, 600, 10.0))
TARGETS = ("literal_target", "signed_return")
QUANTILES = (.50, .70, .80, .90, .925, .95, .975)
TOP_EXACT = 20


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def add_outcomes(frame: pd.DataFrame, candles: dict,
                 geometry: tuple[float, float, int]) -> pd.DataFrame:
    target, stop, hold = geometry
    resolved = replay.add_outcomes(frame, candles, target, stop, hold)
    entry_price = np.empty(len(resolved), dtype=float)
    for symbol, part in resolved.groupby("symbol", sort=False):
        source = np.asarray(candles[symbol]["open"], dtype=float)
        entry_price[part.index.to_numpy(np.int64)] = source[
            part.entry_index.to_numpy(np.int64)]
    resolved["entry_price"] = entry_price
    resolved["target_hit"] = resolved.outcome_code.eq(2)
    return resolved


def fit_model(x: np.ndarray, frame: pd.DataFrame, mask: np.ndarray,
              target: str, config: tuple[int, int, float]):
    common = dict(
        learning_rate=.05, max_iter=180, max_leaf_nodes=config[0],
        min_samples_leaf=config[1], l2_regularization=config[2],
        early_stopping=False, random_state=20261022)
    if target == "literal_target":
        model = HistGradientBoostingClassifier(**common)
        y = frame.target_hit.astype(np.int8).to_numpy()
    else:
        model = HistGradientBoostingRegressor(**common)
        y = frame.signed_return.to_numpy(np.float32)
    model.fit(x[mask], y[mask])
    return model


def predict(model, x: np.ndarray, target: str) -> np.ndarray:
    return (model.predict_proba(x)[:, 1] if target == "literal_target"
            else model.predict(x))


def model_id(geometry: tuple[float, float, int], target: str,
             config: tuple[int, int, float]) -> str:
    return (f"tp{geometry[0]:g}_sl{geometry[1]:g}_h{geometry[2]}_"
            f"{target}_l{config[0]}_n{config[1]}")


def scored(frame: pd.DataFrame, scores: np.ndarray, name: str,
           threshold: float) -> pd.DataFrame:
    result = frame[ scores >= threshold ].copy()
    result["model_probability"] = scores[scores >= threshold]
    result["spec_name"] = name
    return result


def stage(frame: pd.DataFrame, years: tuple[int, ...], candles: dict,
          *, marked: bool) -> tuple[dict, list[dict]]:
    return metrics.exact_stage(frame, years, candles, marked=marked)


def rank(value: dict) -> tuple:
    result = value["margin_3pct"]
    floor = min(result.get("monthly_return_pct", {}).values(), default=-999.0)
    return (
        result["trades_per_calendar_day"] >= 2.5
        and result["return_pct"] > 0 and result["dd_pct"] <= 15,
        float(result["projected_30_day_pct"]), -result["negative_months"],
        floor, result["return_pct"], -result["dd_pct"],
    )


def gate(value: dict, minimum: int) -> bool:
    result = value["margin_3pct"]
    annual = list(value["annual_3pct"].values())
    return bool(
        result["n"] >= minimum and result["return_pct"] > 0
        and result["trades_per_calendar_day"] >= 2.5
        and max(result["dd_pct"], result["mtm_dd_pct"]) <= 12
        and all(item["return_pct"] > 0 for item in annual))


def public(item: dict) -> dict:
    return {key: value for key, value in item.items()
            if not key.startswith("_")}


def main() -> None:
    candles, sources = price.base.load()
    features = price.build_features(candles)
    proposals = price.proposals(features).reset_index(drop=True)
    candidates = []
    models = {}
    frames = {}
    for geometry in GEOMETRIES:
        frame = add_outcomes(proposals, candles, geometry)
        schema = price.columns(frame[frame.year.isin(FIT)])
        x = price.matrix(frame, schema)
        fit_mask = frame.year.isin(FIT).to_numpy()
        selection_mask = frame.year.isin(SELECTION).to_numpy()
        frames[geometry] = (frame, x, schema)
        for target, config in itertools.product(TARGETS, CONFIGS):
            name = model_id(geometry, target, config)
            model = fit_model(x, frame, fit_mask, target, config)
            scores = predict(model, x, target)
            models[name] = (model, scores)
            for quantile in QUANTILES:
                threshold = float(np.quantile(scores[selection_mask], quantile))
                selected = scored(frame, scores, name, threshold)
                value, _ = stage(selected, SELECTION, candles, marked=False)
                candidates.append({
                    "model_id": name,
                    "geometry": {"target_pct": geometry[0],
                                 "stop_pct": geometry[1],
                                 "hold_bars_5m": geometry[2]},
                    "model_target": target,
                    "config": {"max_leaf_nodes": config[0],
                               "min_samples_leaf": config[1],
                               "l2_regularization": config[2]},
                    "selection_quantile": quantile,
                    "threshold": threshold,
                    "selection_2022": value,
                    "_geometry": geometry,
                })
            print(json.dumps({"model": name, "candidates": len(candidates)}),
                  flush=True)

    candidates.sort(key=lambda item: rank(item["selection_2022"]), reverse=True)
    finalists = []
    for item in candidates[:TOP_EXACT]:
        frame, _, _ = frames[item["_geometry"]]
        scores = models[item["model_id"]][1]
        selected = scored(
            frame, scores, item["model_id"], float(item["threshold"]))
        selection, _ = stage(selected, SELECTION, candles, marked=True)
        confirmation, _ = stage(selected, CONFIRMATION, candles, marked=True)
        finalists.append({**item, "selection_2022": selection,
                           "confirmation_2023": confirmation,
                           "_frame": selected})
    finalists.sort(key=lambda item: (
        *rank(item["confirmation_2023"]), *rank(item["selection_2022"])),
        reverse=True)
    confirmed = [item for item in finalists
                 if gate(item["selection_2022"], 700)
                 and gate(item["confirmation_2023"], 700)]
    frozen = confirmed[0] if confirmed else None
    test = hidden = None
    accepted = []
    status = "REJECTED_CONFIRMATION"
    if frozen:
        test, accepted = stage(frozen["_frame"], TEST, candles, marked=True)
        status = "REJECTED_TEST"
        if gate(test, 700):
            hidden, accepted = stage(
                frozen["_frame"], HIDDEN, candles, marked=True)
            status = ("MATERIAL_REGIME_SHADOW_CANDIDATE"
                      if gate(hidden, 1000) else "REJECTED_HIDDEN")
    pd.DataFrame(accepted).to_pickle(ROWS_OUT)
    objective = bool(
        frozen and test and hidden
        and metrics.objective_ok(frozen["selection_2022"])
        and metrics.objective_ok(frozen["confirmation_2023"])
        and metrics.objective_ok(test) and metrics.objective_ok(hidden))
    report = {
        "status": status, "ready_for_live_orders": False,
        "orders_possible": False,
        "protocol": {
            "fit": list(FIT), "selection": list(SELECTION),
            "confirmation": list(CONFIRMATION),
            "untouched_test": list(TEST), "hidden_reveal": list(HIDDEN),
            "entry_universe": "five causal semantic regime families",
            "entry": "next five-minute market open",
            "exit": "literal 1-2% target; 0.75-1% stop; stop first",
            "orders": "market only",
            "money": "client-selected 3% current-equity margin; fixed 10x",
            "drawdown": "five-minute marked concurrent equity",
        },
        "grid": {"feature_rows": len(features),
                 "proposal_rows": len(proposals),
                 "models": len(models), "candidates": len(candidates),
                 "exact_finalists": len(finalists),
                 "confirmed": len(confirmed)},
        "selected_before_test": public(frozen) if frozen else None,
        "test_2024": test, "hidden_2025_2026": hidden,
        "all_stages_meet_monthly_objective": objective,
        "top_confirmation": [public(item) for item in finalists[:10]],
        "sources": {
            **sources,
            str(Path(price.__file__).resolve()):
                digest(Path(price.__file__).resolve()),
            str(Path(replay.__file__).resolve()):
                digest(Path(replay.__file__).resolve()),
            str(Path(__file__).resolve()): digest(Path(__file__).resolve()),
        },
        "limitations": [
            "All post-2024 history has been observed by earlier research, so "
            "a survivor still requires new shadow-forward evidence.",
            "Binance spot history remains a proxy for X-Perp execution.",
            "The 2026 archive ends in September.",
            "Gross results exclude fees at user request.",
        ],
    }
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({
        "status": status, "grid": report["grid"],
        "selected": report["selected_before_test"],
        "test": test, "hidden": hidden,
        "all_stages_meet_monthly_objective": objective,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
