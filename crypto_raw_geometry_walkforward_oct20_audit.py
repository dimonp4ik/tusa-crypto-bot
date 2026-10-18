"""Chronological geometry search for raw hourly crypto entries.

Entry features are inherited from the causal raw-hourly dataset.  Target,
stop, holding horizon, model target and score cutoff are selected using only
2020-2023.  The selected policy is then replayed on 2024+, followed by direct
X-Perp.  All exits use literal price moves and stop-first ambiguous bars.
"""
from __future__ import annotations

import collections
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import (HistGradientBoostingClassifier,
                              HistGradientBoostingRegressor)

import crypto_raw_hourly_direction_oct19_audit as parent


OUT = parent.ROOT / "crypto_raw_geometry_walkforward_oct20.json"
ROWS_OUT = parent.ROOT / "crypto_raw_geometry_walkforward_xperp_oct20.pkl"
GEOMETRIES = (
    (.005, .005, 16),
    (.005, .005, 32),
    (.0075, .005, 32),
    (.010, .005, 48),
    (.010, .0075, 48),
    (.015, .0075, 48),
)
MODEL_CONFIG = {
    "signed_return": (15, 300, 5.0),
    "profitable_exit": (7, 300, 5.0),
}
SOURCE_HASHES: dict[str, str] = {}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_values(path: Path) -> dict:
    raw = path.read_bytes()
    SOURCE_HASHES[str(path)] = hashlib.sha256(raw).hexdigest()
    return pickle.loads(raw)


def geometry_outcome(values: dict, index: int, direction_n: int,
                     target_return: float, stop_return: float,
                     hold_bars: int) -> tuple[float, str, float]:
    entry = float(values["open"][index])
    target = entry * (1 + direction_n * target_return)
    stop = entry * (1 - direction_n * stop_return)
    last = min(index + hold_bars - 1, len(values["time"]) - 1)
    for bar in range(index, last + 1):
        opened = float(values["open"][bar])
        high = float(values["high"][bar])
        low = float(values["low"][bar])
        stop_hit = low <= stop if direction_n > 0 else high >= stop
        if stop_hit:
            fill = min(opened, stop) if direction_n > 0 else max(opened, stop)
            return (direction_n * (fill / entry - 1), "SL",
                    float(values["time"][bar]) + 900)
        target_hit = high >= target if direction_n > 0 else low <= target
        if target_hit:
            return (target_return, "TP",
                    float(values["time"][bar]) + 900)
    close = float(values["close"][last])
    return (direction_n * (close / entry - 1), "TIME",
            float(values["time"][last]) + 900)


def relabel(frame: pd.DataFrame, geometry: tuple[float, float, int],
            *, direct: bool = False) -> pd.DataFrame:
    target_return, stop_return, hold_bars = geometry
    parts = []
    fields = ["symbol"] if direct else ["period", "symbol"]
    grouper = fields[0] if len(fields) == 1 else fields
    for identity, group in frame.groupby(grouper, sort=False):
        if direct:
            symbol = str(identity)
            path = parent.rails.DIRECT_CACHE / f"{symbol}_15min_current.pkl"
        else:
            period, symbol = identity
            path = parent.candle_path(str(period), str(symbol))
        values = load_values(path)
        lookup = {int(timestamp): index
                  for index, timestamp in enumerate(values["time"])}
        output = group.copy()
        signed, outcomes, exits = [], [], []
        for row in output.itertuples(index=False):
            index = lookup.get(int(row.entry_time))
            if index is None or index + hold_bars > len(values["time"]):
                signed.append(np.nan); outcomes.append("MISSING"); exits.append(np.nan)
                continue
            value, outcome, exit_time = geometry_outcome(
                values, index, int(row.direction_n), target_return,
                stop_return, hold_bars)
            signed.append(value); outcomes.append(outcome); exits.append(exit_time)
        output["signed_return"] = signed
        output["outcome"] = outcomes
        output["exit_time"] = exits
        parts.append(output[output.outcome.ne("MISSING")])
    return pd.concat(parts, ignore_index=True)


def fit_model(train: pd.DataFrame, schema: dict, target_name: str):
    config = MODEL_CONFIG[target_name]
    common = dict(
        learning_rate=.05, max_iter=150, max_leaf_nodes=config[0],
        min_samples_leaf=config[1], l2_regularization=config[2],
        early_stopping=False, random_state=20260908)
    if target_name == "signed_return":
        model = HistGradientBoostingRegressor(**common)
        target = train.signed_return.to_numpy(float)
    else:
        model = HistGradientBoostingClassifier(**common)
        target = (train.signed_return > 0).astype(int).to_numpy()
    model.fit(parent.matrix(train, schema), target)
    return model


def score(frame: pd.DataFrame, model, schema: dict,
          target_name: str) -> pd.DataFrame:
    result = frame.copy()
    values = parent.matrix(result, schema)
    result["model_quality"] = (
        model.predict(values) if target_name == "signed_return"
        else model.predict_proba(values)[:, 1])
    return result


def rank_key(item: dict) -> tuple:
    return (
        bool(item["admissible"]), float(item["development_floor_pct"]),
        -max(float(item["selection"]["dd_pct"]),
             float(item["confirmation"]["dd_pct"])),
        float(item["selection"]["return_pct"])
        + float(item["confirmation"]["return_pct"]),
    )


def evaluate_scored(scored: pd.DataFrame, target_name: str,
                    geometry: tuple[float, float, int]) -> list[dict]:
    selection = parent.stage(scored, parent.SELECT)
    candidates = []
    for quantile in parent.QUANTILES:
        threshold = float(selection.model_quality.quantile(quantile))
        for loss_pause in parent.LOSS_PAUSES:
            for hourly_cap in parent.HOURLY_CAPS:
                select_value, _ = parent.evaluate(
                    selection, threshold, loss_pause, hourly_cap, .03)
                confirm_value, _ = parent.evaluate(
                    parent.stage(scored, parent.CONFIRM), threshold,
                    loss_pause, hourly_cap, .03)
                floor = min(
                    parent.month_floor(
                        select_value, parent.rails.SELECTION_MONTHS),
                    parent.month_floor(
                        confirm_value, parent.rails.CONFIRMATION_MONTHS))
                candidates.append({
                    "target_return_pct": 100 * geometry[0],
                    "stop_return_pct": 100 * geometry[1],
                    "hold_bars": geometry[2],
                    "model_target": target_name,
                    "model_config": list(MODEL_CONFIG[target_name]),
                    "selection_quantile": quantile,
                    "threshold": threshold,
                    "loss_pause": loss_pause,
                    "hourly_cap": hourly_cap,
                    "selection": select_value,
                    "confirmation": confirm_value,
                    "development_floor_pct": round(floor, 4),
                    "admissible": (
                        parent.admissible(
                            select_value, parent.rails.SELECTION_MONTHS)
                        and parent.admissible(
                            confirm_value,
                            parent.rails.CONFIRMATION_MONTHS)),
                })
    return candidates


def geometry_tuple(item: dict) -> tuple[float, float, int]:
    return (float(item["target_return_pct"]) / 100,
            float(item["stop_return_pct"]) / 100,
            int(item["hold_bars"]))


def main() -> None:
    base_history = parent.build_history()
    schema = parent.design(parent.stage(base_history, parent.FIT))
    candidates = []
    for geometry in GEOMETRIES:
        labeled = relabel(base_history, geometry)
        fit = parent.stage(labeled, parent.FIT)
        for target_name in MODEL_CONFIG:
            model = fit_model(fit, schema, target_name)
            scored = score(labeled, model, schema, target_name)
            candidates.extend(evaluate_scored(scored, target_name, geometry))
    candidates.sort(key=rank_key, reverse=True)
    selected = candidates[0]
    geometry = geometry_tuple(selected)
    labeled = relabel(base_history, geometry)
    model = fit_model(parent.stage(labeled, parent.FIT), schema,
                      str(selected["model_target"]))
    scored = score(labeled, model, schema, str(selected["model_target"]))
    threshold = float(selected["threshold"])
    loss_pause = int(selected["loss_pause"])
    hourly_cap = int(selected["hourly_cap"])
    holdouts = {}
    for margin in (.03, .05):
        name = f"margin_{round(100 * margin)}pct"
        validation, _ = parent.evaluate(
            parent.stage(scored, parent.VALIDATE), threshold,
            loss_pause, hourly_cap, margin)
        later, _ = parent.evaluate(
            parent.stage(scored, parent.LATER), threshold,
            loss_pause, hourly_cap, margin)
        holdouts[name] = {"validation_2024": validation,
                          "later_2025_2026": later}

    direct_parts = []
    for symbol in parent.CORE_SYMBOLS:
        path = parent.rails.DIRECT_CACHE / f"{symbol}_15min_current.pkl"
        direct_parts.append(parent.build_symbol_period(
            "xperp", symbol, parent.load_candles(path)))
    direct = relabel(pd.concat(direct_parts, ignore_index=True),
                     geometry, direct=True)
    direct_scored = score(
        direct, model, schema, str(selected["model_target"]))
    direct_values = {}
    output_rows = []
    for margin in (.03, .05):
        value, rows = parent.evaluate(
            direct_scored, threshold, loss_pause, hourly_cap, margin)
        direct_values[f"margin_{round(100 * margin)}pct"] = value
        if margin == .03:
            output_rows = rows
    pd.DataFrame(output_rows).to_pickle(ROWS_OUT)

    status = "REJECTED_OBJECTIVE"
    if (selected["admissible"] and selected["development_floor_pct"] >= 30
            and holdouts["margin_5pct"]["validation_2024"]["negative_months"] == 0
            and holdouts["margin_5pct"]["later_2025_2026"]["negative_months"] == 0
            and direct_values["margin_5pct"]["negative_months"] == 0
            and min(direct_values["margin_5pct"]["monthly_return_pct"].values()) >= 30):
        status = "RAW_GEOMETRY_WALKFORWARD_SHADOW_CANDIDATE"
    report = {
        "status": status,
        "ready_for_live_orders": False,
        "protocol": {
            "geometries": [{"target_pct": 100 * item[0],
                             "stop_pct": 100 * item[1],
                             "hold_bars": item[2]} for item in GEOMETRIES],
            "fit": list(parent.FIT), "selection": list(parent.SELECT),
            "confirmation": list(parent.CONFIRM),
            "holdouts": [*parent.VALIDATE, *parent.LATER, "direct X-Perp"],
            "entry": "market at next 15-minute open; hourly decisions",
            "orders": "market only",
            "money": "client 3% or 5% current-equity margin; fixed 10x",
            "priority": "worst complete-month return; no win-rate gate",
        },
        "history_rows": len(base_history),
        "candidate_count": len(candidates),
        "admissible_count": sum(item["admissible"] for item in candidates),
        "selected_before_holdouts": selected,
        "top20_before_holdouts": candidates[:20],
        "holdouts": holdouts,
        "direct_xperp": direct_values,
        "sources": {**SOURCE_HASHES,
                    str(Path(__file__).resolve()): digest(Path(__file__).resolve())},
        "limitations": [
            "Only symbols with continuous 2020-2026 history are modeled.",
            "Historical named periods are available six-month windows.",
            "Direct X-Perp history is shorter than a full market cycle.",
            "Gross returns exclude fees at user request.",
            "No backtest guarantees a future monthly return.",
        ],
    }
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({
        "status": status, "candidate_count": len(candidates),
        "admissible_count": report["admissible_count"],
        "selected": selected, "holdouts": holdouts,
        "direct_xperp": direct_values,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
