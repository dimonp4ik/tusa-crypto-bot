"""Dynamic 1-2% geometry router over the material five-minute universe."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

import crypto_5m_material_regime_model_oct22_audit as base
import crypto_5m_regime_router_literal_oct6_audit as price
import crypto_raw_material_direction_model_oct21_audit as metrics


OUT = price.ROOT / "crypto_5m_dynamic_material_geometry_oct22.json"
ROWS_OUT = price.ROOT / "crypto_5m_dynamic_material_geometry_oct22.pkl"
FIT = (2020, 2021)
SELECTION = (2022,)
CONFIRMATION = (2023,)
TEST = (2024,)
HIDDEN = (2025, 2026)
GEOMETRIES = base.GEOMETRIES
CONFIG = (31, 600, 10.0)
QUANTILES = (.50, .70, .80, .90, .925, .95, .975)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def identity(frame: pd.DataFrame) -> pd.MultiIndex:
    return pd.MultiIndex.from_frame(
        frame[["symbol", "entry_time", "direction"]])


def build_router(candles: dict, proposals: pd.DataFrame,
                 kind: str) -> pd.DataFrame:
    scored = []
    reference = None
    reference_identity = None
    for geometry in GEOMETRIES:
        frame = base.add_outcomes(proposals, candles, geometry)
        current_identity = identity(frame)
        if reference is None:
            reference = frame.copy()
            reference_identity = current_identity
        elif not current_identity.equals(reference_identity):
            raise ValueError("geometry outcome frames lost identity alignment")
        schema = price.columns(frame[frame.year.isin(FIT)])
        x = price.matrix(frame, schema)
        target = "literal_target" if kind == "literal_ev" else "signed_return"
        model = base.fit_model(
            x, frame, frame.year.isin(FIT).to_numpy(), target, CONFIG)
        prediction = base.predict(model, x, target)
        if kind == "literal_ev":
            target_fraction = geometry[0] / 100
            stop_fraction = geometry[1] / 100
            prediction = (prediction * target_fraction
                          - (1 - prediction) * stop_fraction)
        scored.append((geometry, frame, prediction))
        print(json.dumps({"router": kind, "geometry": geometry}), flush=True)
    values = np.column_stack([item[2] for item in scored])
    choice = np.argmax(values, axis=1)
    output = reference.copy()
    output["model_probability"] = values[np.arange(len(values)), choice]
    output["chosen_geometry"] = [
        f"tp{GEOMETRIES[index][0]:g}_sl{GEOMETRIES[index][1]:g}_h{GEOMETRIES[index][2]}"
        for index in choice]
    outcome_columns = (
        "signed_return", "exit_time", "outcome_code", "target_pct",
        "stop_pct", "entry_price")
    for index, (_, frame, _) in enumerate(scored):
        mask = choice == index
        for column in outcome_columns:
            output.loc[mask, column] = frame.loc[mask, column].to_numpy()
    output["spec_name"] = f"dynamic_material_{kind}"
    return output


def selected(frame: pd.DataFrame, threshold: float) -> pd.DataFrame:
    return frame[frame.model_probability.ge(threshold)].copy()


def rank(stage: dict) -> tuple:
    value = stage["margin_3pct"]
    floor = min(value.get("monthly_return_pct", {}).values(), default=-999.)
    return (value["trades_per_calendar_day"] >= 2.5
            and value["return_pct"] > 0,
            value["projected_30_day_pct"], -value["negative_months"], floor,
            value["return_pct"], -value["dd_pct"])


def gate(stage: dict, minimum: int) -> bool:
    value = stage["margin_3pct"]
    return bool(value["n"] >= minimum and value["return_pct"] > 0
                and value["trades_per_calendar_day"] >= 2.5
                and max(value["dd_pct"], value["mtm_dd_pct"]) <= 12)


def public(item: dict) -> dict:
    return {key: value for key, value in item.items()
            if not key.startswith("_")}


def main() -> None:
    candles, sources = price.base.load()
    features = price.build_features(candles)
    proposals = price.proposals(features).reset_index(drop=True)
    routers = {kind: build_router(candles, proposals, kind)
               for kind in ("literal_ev", "signed_return")}
    candidates = []
    for kind, frame in routers.items():
        selection_scores = frame.loc[
            frame.year.isin(SELECTION), "model_probability"]
        for quantile in QUANTILES:
            threshold = float(selection_scores.quantile(quantile))
            routed = selected(frame, threshold)
            selection_stage, _ = base.stage(
                routed, SELECTION, candles, marked=True)
            confirmation_stage, _ = base.stage(
                routed, CONFIRMATION, candles, marked=True)
            candidates.append({
                "router_kind": kind, "selection_quantile": quantile,
                "threshold": threshold, "selection_2022": selection_stage,
                "confirmation_2023": confirmation_stage, "_frame": routed,
            })
    candidates.sort(key=lambda item: (
        *rank(item["confirmation_2023"]), *rank(item["selection_2022"])),
        reverse=True)
    confirmed = [item for item in candidates
                 if gate(item["selection_2022"], 700)
                 and gate(item["confirmation_2023"], 700)]
    frozen = confirmed[0] if confirmed else None
    test = hidden = None
    rows = []
    status = "REJECTED_CONFIRMATION"
    if frozen:
        test, rows = base.stage(frozen["_frame"], TEST, candles, marked=True)
        status = "REJECTED_TEST"
        if gate(test, 700) and test["margin_3pct"]["negative_months"] == 0:
            hidden, rows = base.stage(
                frozen["_frame"], HIDDEN, candles, marked=True)
            status = ("DYNAMIC_GEOMETRY_SHADOW_CANDIDATE"
                      if gate(hidden, 1000) else "REJECTED_HIDDEN")
    pd.DataFrame(rows).to_pickle(ROWS_OUT)
    objective = bool(frozen and test and hidden
                     and metrics.objective_ok(frozen["selection_2022"])
                     and metrics.objective_ok(frozen["confirmation_2023"])
                     and metrics.objective_ok(test)
                     and metrics.objective_ok(hidden))
    geometry_counts = {}
    if frozen:
        geometry_counts = {
            str(year): frozen["_frame"].loc[
                frozen["_frame"].year.eq(year), "chosen_geometry"
            ].value_counts().to_dict()
            for year in (*SELECTION, *CONFIRMATION, *TEST)
        }
    report = {
        "status": status, "ready_for_live_orders": False,
        "orders_possible": False,
        "protocol": {"fit": list(FIT), "selection": list(SELECTION),
                     "confirmation": list(CONFIRMATION),
                     "untouched_test": list(TEST),
                     "hidden_reveal": list(HIDDEN),
                     "geometry_choices": [list(x) for x in GEOMETRIES],
                     "entry": "next five-minute market open",
                     "orders": "market only",
                     "money": "client-selected 3% current-equity margin; fixed 10x"},
        "grid": {"feature_rows": len(features),
                 "proposal_rows": len(proposals),
                 "routers": len(routers), "candidates": len(candidates),
                 "confirmed": len(confirmed)},
        "selected_before_test": public(frozen) if frozen else None,
        "chosen_geometry_counts": geometry_counts,
        "test_2024": test, "hidden_2025_2026": hidden,
        "all_stages_meet_monthly_objective": objective,
        "top_confirmation": [public(item) for item in candidates[:10]],
        "sources": {**sources, str(base.OUT): digest(base.OUT),
                    str(Path(base.__file__).resolve()): digest(Path(base.__file__).resolve()),
                    str(Path(__file__).resolve()): digest(Path(__file__).resolve())},
        "limitations": ["The router chooses only from literal 1-2% market-entry geometries.",
                        "2025-2026 remain sealed unless 2024 passes.",
                        "Binance spot history is a proxy for X-Perp execution.",
                        "Gross results exclude fees at user request."],
    }
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"status": status, "grid": report["grid"],
                      "selected": report["selected_before_test"],
                      "geometry_counts": geometry_counts,
                      "test": test, "hidden": hidden,
                      "objective": objective}, indent=2), flush=True)


if __name__ == "__main__":
    main()
