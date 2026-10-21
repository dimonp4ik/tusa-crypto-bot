"""Chronological raw-state model for literal 2-3% crypto moves.

Unlike earlier classifiers over legacy setup proposals, this audit scores both
LONG and SHORT for every hourly closed-market state.  Models fit on 2020-2021,
thresholds are selected on 2022, policy choice uses 2023, and 2024 plus
2025-2026 are revealed sequentially only after earlier gates pass.
"""
from __future__ import annotations

import hashlib
import heapq
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import (HistGradientBoostingClassifier,
                              HistGradientBoostingRegressor)

import crypto_multiday_material_trend_oct21_audit as base
from src import research_crypto_replay as replay


OUT = base.ROOT / "crypto_raw_material_direction_model_oct21.json"
ROWS_OUT = base.ROOT / "crypto_raw_material_direction_model_oct21.pkl"
FIT = (2020, 2021)
SELECTION = (2022,)
CONFIRMATION = (2023,)
TEST = (2024,)
HIDDEN = (2025, 2026)
GEOMETRIES = (
    (2.0, .75, 288),
    (2.0, 1.0, 288),
    (3.0, 1.0, 576),
    (3.0, 1.5, 576),
)
CONFIGS = (
    (15, 400, 5.0),
    (31, 800, 10.0),
)
TARGETS = ("signed_return", "literal_target")
QUANTILES = (.50, .60, .70, .80, .85, .90, .925, .95)
TOP_SELECTION = 20


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def directional_frame(features: pd.DataFrame) -> pd.DataFrame:
    long = features.copy()
    long["direction"] = np.int8(1)
    short = features.copy()
    short["direction"] = np.int8(-1)
    frame = pd.concat((long, short), ignore_index=True)
    grouped = frame.groupby("entry_time", sort=False)
    # These same-bar market fields are fully known at the signal close.
    frame["market_ret24"] = grouped.ret24.transform("mean")
    frame["market_ret72"] = grouped.ret72.transform("mean")
    frame["dispersion24"] = grouped.ret24.transform("std")
    frame["dispersion72"] = grouped.ret72.transform("std")
    frame["breadth24"] = grouped.ret24.transform(
        lambda values: float((values > 0).mean()))
    frame["breadth72"] = grouped.ret72.transform(
        lambda values: float((values > 0).mean()))
    stamp = pd.to_datetime(frame.entry_time, unit="s", utc=True)
    hour = stamp.dt.hour.to_numpy(float)
    weekday = stamp.dt.weekday.to_numpy(float)
    frame["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    frame["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    frame["weekday_sin"] = np.sin(2 * np.pi * weekday / 7)
    frame["weekday_cos"] = np.cos(2 * np.pi * weekday / 7)
    return frame


def schema(frame: pd.DataFrame) -> dict:
    fit = frame[frame.year.isin(FIT)]
    raw = (
        "ret1", "ret4", "ret12", "ret24", "ret72", "ret168",
        "trend_z12", "trend_z24", "trend_z72", "trend_z168",
        "market_ret24", "market_ret72",
        "eff1", "eff4", "eff12", "eff24", "eff72", "eff168",
        "volume_ratio", "dispersion24", "dispersion72",
        "breadth24", "breadth72", "breakout_up24_atr",
        "breakout_down24_atr", "breakout_up72_atr",
        "breakout_down72_atr", "hour_sin", "hour_cos",
        "weekday_sin", "weekday_cos",
    )
    return {
        "raw_fields": list(raw),
        "medians": {name: float(fit[name].median()) for name in raw},
        "symbols": list(base.SYMBOLS),
    }


def matrix(frame: pd.DataFrame, design: dict) -> np.ndarray:
    direction = frame.direction.to_numpy(np.float32)
    columns = []
    directional = {
        "ret1", "ret4", "ret12", "ret24", "ret72", "ret168",
        "trend_z12", "trend_z24", "trend_z72", "trend_z168",
        "market_ret24", "market_ret72",
    }
    for name in design["raw_fields"]:
        values = pd.to_numeric(frame[name], errors="coerce").fillna(
            design["medians"][name]).to_numpy(np.float32)
        if name in directional:
            values = values * direction
        elif name.startswith("breadth"):
            values = np.where(direction > 0, values, 1 - values)
        elif name.startswith("breakout_up"):
            opposite = name.replace("breakout_up", "breakout_down")
            other = pd.to_numeric(frame[opposite], errors="coerce").fillna(
                design["medians"][opposite]).to_numpy(np.float32)
            values = np.where(direction > 0, values, other)
        elif name.startswith("breakout_down"):
            opposite = name.replace("breakout_down", "breakout_up")
            other = pd.to_numeric(frame[opposite], errors="coerce").fillna(
                design["medians"][opposite]).to_numpy(np.float32)
            values = np.where(direction > 0, values, other)
        columns.append(values)
    symbols = frame.symbol.astype(str).to_numpy()
    columns.extend((symbols == symbol).astype(np.float32)
                   for symbol in design["symbols"])
    columns.append(direction)
    return np.column_stack(columns).astype(np.float32, copy=False)


def label_identity(frame: pd.DataFrame, candles: dict,
                   geometry: tuple[float, float, int]) -> pd.DataFrame:
    columns = ["symbol", "entry_index", "entry_time", "year", "month",
               "direction"]
    target, stop, hold = geometry
    identity = frame[columns].copy()
    # add_outcomes evaluates one symbol at a time and concatenates those
    # groups.  Preserve the original row identity so labels stay aligned with
    # the feature matrix used by the model.
    identity["_row_id"] = np.arange(len(identity), dtype=np.int64)
    prices = np.empty(len(identity), dtype=float)
    for symbol, part in identity.groupby("symbol", sort=False):
        source = np.asarray(candles[symbol]["open"], dtype=float)
        prices[part.index.to_numpy(np.int64)] = source[
            part.entry_index.to_numpy(np.int64)]
    identity["entry_price"] = prices
    resolved = replay.add_outcomes(identity, candles, target, stop, hold)
    return resolved.sort_values("_row_id", kind="mergesort").drop(
        columns="_row_id").reset_index(drop=True)


def mtm_drawdown(rows: list[dict], candles: dict,
                 margin: float = base.MARGIN) -> dict:
    """Replay five-minute marked equity with entry-equity position sizing."""
    if not rows:
        return {"mtm_dd_pct": 0.0, "max_concurrent_positions": 0,
                "max_gross_notional_equity_pct": 0.0}
    priced = [dict(row) for row in rows]
    pending: list[tuple[float, int, float]] = []
    realized = 1.0
    sequence = 0
    for row in sorted(priced, key=lambda item: (
            float(item["entry_time"]), int(item["source_priority"]),
            str(item["symbol"]))):
        now = float(row["entry_time"])
        while pending and pending[0][0] < now:
            _, _, pnl = heapq.heappop(pending)
            realized += pnl
        stake = realized * margin * base.LEVERAGE
        row["_stake"] = stake
        sequence += 1
        heapq.heappush(pending, (
            float(row["exit_time"]), sequence,
            stake * float(row["signed_price_return"])))

    entries: dict[int, list[dict]] = {}
    exits: dict[int, list[dict]] = {}
    for row in priced:
        entries.setdefault(int(row["entry_time"]), []).append(row)
        exits.setdefault(int(row["exit_time"]), []).append(row)
    arrays = {
        symbol: (np.asarray(candle["time"], dtype=np.int64),
                 np.asarray(candle["close"], dtype=float))
        for symbol, candle in candles.items()
    }
    opened: dict[tuple[str, int], dict] = {}
    cash = peak = 1.0
    maximum_dd = maximum_open = maximum_gross = 0.0
    first = min(entries)
    last = max(exits)
    for close_time in range(first + 300, last + 1, 300):
        for row in entries.get(close_time - 300, ()):  # entered at bar open
            opened[(str(row["symbol"]), int(row["entry_time"]))] = row
        for row in exits.get(close_time, ()):  # exact stop/target/time fill
            key = (str(row["symbol"]), int(row["entry_time"]))
            opened.pop(key, None)
            cash += float(row["_stake"]) * float(row["signed_price_return"])
        unrealized = 0.0
        gross = 0.0
        bar_open = close_time - 300
        for row in opened.values():
            times, closes = arrays[str(row["symbol"])]
            index = int(np.searchsorted(times, bar_open))
            if index >= len(times) or int(times[index]) != bar_open:
                continue
            entry = float(row["entry_price"])
            direction = int(row["direction_n"])
            unrealized += float(row["_stake"]) * direction * (
                float(closes[index]) / entry - 1)
            gross += float(row["_stake"])
        equity = cash + unrealized
        peak = max(peak, equity)
        if peak > 0:
            maximum_dd = max(maximum_dd, (peak - equity) / peak)
            maximum_gross = max(maximum_gross, gross / peak)
        maximum_open = max(maximum_open, len(opened))
    return {
        "mtm_dd_pct": round(100 * maximum_dd, 4),
        "max_concurrent_positions": int(maximum_open),
        "max_gross_notional_equity_pct": round(100 * maximum_gross, 4),
    }


def exact_stage(frame: pd.DataFrame, years: tuple[int, ...],
                candles: dict, *, marked: bool = True) -> tuple[dict, list[dict]]:
    stage, rows = base.exact_stage(frame, years)
    if not marked:
        return stage, rows
    stage["margin_3pct"].update(mtm_drawdown(rows, candles))
    for year in years:
        annual_rows = [row for row in rows if pd.Timestamp(
            float(row["entry_time"]), unit="s", tz="UTC").year == year]
        stage["annual_3pct"][str(year)].update(
            mtm_drawdown(annual_rows, candles))
    return stage, rows


def fit_model(x: np.ndarray, labeled: pd.DataFrame, fit_mask: np.ndarray,
              model_target: str, config: tuple[int, int, float]):
    common = dict(
        learning_rate=.05, max_iter=160, max_leaf_nodes=config[0],
        min_samples_leaf=config[1], l2_regularization=config[2],
        early_stopping=False, random_state=20261021)
    if model_target == "signed_return":
        model = HistGradientBoostingRegressor(**common)
        target = labeled.signed_return.to_numpy(np.float32)[fit_mask]
    else:
        model = HistGradientBoostingClassifier(**common)
        target = labeled.outcome_code.eq(2).to_numpy(np.int8)[fit_mask]
    model.fit(x[fit_mask], target)
    return model


def predict(model, x: np.ndarray, model_target: str) -> np.ndarray:
    return (model.predict(x) if model_target == "signed_return"
            else model.predict_proba(x)[:, 1])


def scored_frame(labeled: pd.DataFrame, scores: np.ndarray,
                 model_id: str) -> pd.DataFrame:
    result = labeled.copy()
    result["model_probability"] = scores
    result["spec_name"] = model_id
    return result


def rank(stage: dict) -> tuple:
    value = stage["margin_3pct"]
    months = value.get("monthly_return_pct", {})
    floor = min(months.values(), default=-999.0)
    return (
        value["trades_per_calendar_day"] >= 2.5
        and value["return_pct"] > 0 and value["dd_pct"] <= 12,
        -int(value["negative_months"]), float(floor),
        float(value["projected_30_day_pct"]),
        float(value["return_pct"]), -float(value["dd_pct"]),
    )


def stage_ok(stage: dict, minimum_rows: int) -> bool:
    value = stage["margin_3pct"]
    annual = list(stage["annual_3pct"].values())
    return bool(
        value["n"] >= minimum_rows and value["return_pct"] > 0
        and max(value["dd_pct"], value["mtm_dd_pct"]) <= 12
        and value["trades_per_calendar_day"] >= 2.5
        and all(item["return_pct"] > 0 for item in annual)
    )


def objective_ok(stage: dict) -> bool:
    value = stage["margin_3pct"]
    months = value.get("monthly_return_pct", {})
    return bool(
        months and min(months.values()) >= 30
        and value["negative_months"] == 0
        and max(value["dd_pct"], value["mtm_dd_pct"]) <= 5
        and value["trades_per_calendar_day"] >= 3
        and value["projected_30_day_pct"] >= 30)


def public(item: dict) -> dict:
    return {key: value for key, value in item.items()
            if not key.startswith("_")}


def geometry_name(geometry: tuple[float, float, int]) -> str:
    return f"tp{geometry[0]:g}_sl{geometry[1]:g}_h{geometry[2]}"


def main() -> None:
    candles, sources = base.load()
    features = base.build_features(candles)
    directional = directional_frame(features)
    design = schema(directional)
    x = matrix(directional, design)
    fit_mask = directional.year.isin(FIT).to_numpy()
    selection_mask = directional.year.isin(SELECTION).to_numpy()

    candidates = []
    models = {}
    fit_diagnostics = {}
    for geometry in GEOMETRIES:
        labeled = label_identity(directional, candles, geometry)
        geometry_id = geometry_name(geometry)
        fit_part = labeled.loc[fit_mask]
        fit_diagnostics[geometry_id] = {
            "rows": len(fit_part),
            "literal_target_rate_pct": round(
                100 * float(fit_part.outcome_code.eq(2).mean()), 4),
            "profitable_exit_rate_pct": round(
                100 * float(fit_part.signed_return.gt(0).mean()), 4),
            "mean_signed_return_both_directions_pct": round(
                100 * float(fit_part.signed_return.mean()), 6),
        }
        for model_target, config in itertools.product(TARGETS, CONFIGS):
            model_id = (
                f"{geometry_id}_{model_target}_l{config[0]}_n{config[1]}")
            model = fit_model(x, labeled, fit_mask, model_target, config)
            models[model_id] = {
                "model": model, "geometry": geometry,
                "model_target": model_target, "config": config,
            }
            scores = predict(model, x, model_target)
            selection_scores = scores[selection_mask]
            selection_scored = scored_frame(
                labeled.loc[selection_mask].reset_index(drop=True),
                selection_scores, model_id)
            for quantile in QUANTILES:
                threshold = float(np.quantile(selection_scores, quantile))
                scored = selection_scored[
                    selection_scored.model_probability.ge(threshold)]
                stage, _ = exact_stage(
                    scored, SELECTION, candles, marked=False)
                candidates.append({
                    "model_id": model_id,
                    "geometry": {"target_pct": geometry[0],
                                 "stop_pct": geometry[1],
                                 "hold_bars_5m": geometry[2]},
                    "model_target": model_target,
                    "config": {"max_leaf_nodes": config[0],
                               "min_samples_leaf": config[1],
                               "l2_regularization": config[2]},
                    "selection_quantile": quantile,
                    "threshold": threshold,
                    "selection_2022": stage,
                })
            print(json.dumps({
                "geometry": geometry_id, "model": model_id,
                "candidates": len(candidates)}), flush=True)

    candidates.sort(key=lambda item: rank(item["selection_2022"]), reverse=True)
    finalists = []
    # Grouping avoids relabeling and rescoring the same model repeatedly.
    for model_id in sorted({item["model_id"] for item in candidates[:TOP_SELECTION]}):
        meta = models[model_id]
        labeled = label_identity(directional, candles, meta["geometry"])
        scores = predict(meta["model"], x, meta["model_target"])
        scored = scored_frame(labeled, scores, model_id)
        for item in [value for value in candidates[:TOP_SELECTION]
                     if value["model_id"] == model_id]:
            threshold = float(item["threshold"])
            selection_frame = scored[
                scored.year.isin(SELECTION)
                & scored.model_probability.ge(threshold)]
            selection, _ = exact_stage(
                selection_frame, SELECTION, candles)
            confirmation_frame = scored[
                scored.year.isin(CONFIRMATION)
                & scored.model_probability.ge(threshold)]
            confirmation, _ = exact_stage(
                confirmation_frame, CONFIRMATION, candles)
            finalists.append({**item, "selection_2022": selection,
                               "confirmation_2023": confirmation})
    finalists.sort(key=lambda item: (
        *rank(item["confirmation_2023"]), *rank(item["selection_2022"])),
        reverse=True)

    confirmed = [item for item in finalists
                 if stage_ok(item["selection_2022"], 700)
                 and stage_ok(item["confirmation_2023"], 700)]
    frozen = confirmed[0] if confirmed else None
    test = hidden = None
    accepted_rows = []
    status = "REJECTED_CONFIRMATION"
    if frozen:
        meta = models[frozen["model_id"]]
        labeled = label_identity(directional, candles, meta["geometry"])
        scores = predict(meta["model"], x, meta["model_target"])
        scored = scored_frame(labeled, scores, frozen["model_id"])
        threshold = float(frozen["threshold"])
        test_frame = scored[
            scored.year.isin(TEST) & scored.model_probability.ge(threshold)]
        test, test_rows = exact_stage(test_frame, TEST, candles)
        accepted_rows = test_rows
        status = "REJECTED_TEST"
        if stage_ok(test, 700):
            hidden_frame = scored[
                scored.year.isin(HIDDEN)
                & scored.model_probability.ge(threshold)]
            hidden, accepted_rows = exact_stage(hidden_frame, HIDDEN, candles)
            status = ("RAW_MATERIAL_MODEL_CANDIDATE"
                      if stage_ok(hidden, 1000) else "REJECTED_HIDDEN")

    diagnostic = finalists[0] if finalists else candidates[0]
    pd.DataFrame(accepted_rows).to_pickle(ROWS_OUT)
    objective = bool(
        frozen and test and hidden
        and objective_ok(frozen["selection_2022"])
        and objective_ok(frozen["confirmation_2023"])
        and objective_ok(test) and objective_ok(hidden))
    report = {
        "status": status, "ready_for_live_orders": False,
        "orders_possible": False,
        "protocol": {
            "model_fit": list(FIT), "threshold_selection": list(SELECTION),
            "policy_confirmation": list(CONFIRMATION),
            "untouched_test": list(TEST), "hidden_reveal": list(HIDDEN),
            "candidate_space": "both LONG and SHORT at every hourly state",
            "entry": "next five-minute market open after closed hourly state",
            "exit": "literal 2-3% target, 0.75-1.5% stop, stop first",
            "orders": "market only",
            "money": "client 3% or 5% current-equity margin; fixed 10x",
            "drawdown": (
                "five-minute marked equity across all concurrent positions; "
                "entry size follows realized equity used by account replay"),
        },
        "grid": {
            "feature_rows": len(features),
            "directional_rows": len(directional),
            "models": len(models), "threshold_candidates": len(candidates),
            "selection_finalists": len(finalists),
            "confirmed_candidates": len(confirmed),
        },
        "fit_diagnostics": fit_diagnostics,
        "selected_before_test": public(frozen) if frozen else None,
        "best_confirmation_diagnostic": public(diagnostic),
        "test_2024": test, "hidden_2025_2026": hidden,
        "all_stages_meet_monthly_objective": objective,
        "top_confirmation": [public(item) for item in finalists[:10]],
        "sources": {
            **sources,
            str(Path(base.__file__).resolve()):
                digest(Path(base.__file__).resolve()),
            str(Path(replay.__file__).resolve()):
                digest(Path(replay.__file__).resolve()),
            str(Path(__file__).resolve()): digest(Path(__file__).resolve()),
        },
        "limitations": [
            "Binance spot history remains a proxy for X-Perp execution.",
            "Tree probabilities are empirical scores, not calibrated guarantees.",
            "Five-minute MTM drawdown cannot observe sub-candle excursions, but "
            "literal stop-first exits bound them at the configured stop unless a gap occurs.",
            "The 2026 archive ends in September.",
            "Gross results exclude fees at user request.",
        ],
    }
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({
        "status": status, "grid": report["grid"],
        "selected": report["selected_before_test"],
        "best_confirmation_diagnostic":
            report["best_confirmation_diagnostic"],
        "test": test, "hidden": hidden,
        "all_stages_meet_monthly_objective": objective,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
