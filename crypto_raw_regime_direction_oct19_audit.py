"""Regime-aware extension of the raw hourly crypto direction model.

Adds only entry-time cross-sectional breadth, relative strength, and closed BTC
context to the frozen 1% target / 0.5% stop geometry.  Chronological fit,
selection and holdouts remain identical to the raw baseline audit.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import (HistGradientBoostingClassifier,
                              HistGradientBoostingRegressor)

import crypto_raw_hourly_direction_oct19_audit as parent


OUT = parent.ROOT / "crypto_raw_regime_direction_oct19.json"
ROWS_OUT = parent.ROOT / "crypto_raw_regime_direction_xperp_oct19.pkl"
CONTEXT = (
    "universe_n", "cross_mean_return_4", "cross_mean_return_16",
    "cross_std_return_4", "cross_abs_mean_return_4", "breadth_4",
    "breadth_16", "relative_return_4", "relative_return_16",
    "btc_return_4", "btc_return_16", "btc_return_96",
    "btc_atr14_fraction", "btc_range32_position",
    "direction_market_alignment_4", "direction_market_alignment_16",
    "direction_btc_alignment_4", "direction_btc_alignment_16",
)
FEATURES = (*parent.FEATURES, *CONTEXT)
CONFIGS = (
    (7, 100, 5.0), (7, 300, 5.0),
    (15, 100, 5.0), (15, 300, 5.0),
)
MODEL_TARGETS = ("signed_return", "profitable_exit")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def enrich(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    unique = result[result.direction_n.eq(1)].copy()
    grouped = unique.groupby("entry_time", sort=False)
    context = grouped.agg(
        universe_n=("symbol", "nunique"),
        cross_mean_return_4=("return_4", "mean"),
        cross_mean_return_16=("return_16", "mean"),
        cross_std_return_4=("return_4", "std"),
        cross_abs_mean_return_4=("return_4", lambda values: values.abs().mean()),
        breadth_4=("return_4", lambda values: (values > 0).mean()),
        breadth_16=("return_16", lambda values: (values > 0).mean()),
    ).reset_index()
    btc = unique[unique.symbol.eq("BTCUSDT")][[
        "entry_time", "return_4", "return_16", "return_96",
        "atr14_fraction", "range32_position",
    ]].rename(columns={
        "return_4": "btc_return_4",
        "return_16": "btc_return_16",
        "return_96": "btc_return_96",
        "atr14_fraction": "btc_atr14_fraction",
        "range32_position": "btc_range32_position",
    })
    result = result.merge(context, on="entry_time", how="left",
                          validate="many_to_one")
    result = result.merge(btc, on="entry_time", how="left",
                          validate="many_to_one")
    result["relative_return_4"] = (
        result.return_4 - result.cross_mean_return_4)
    result["relative_return_16"] = (
        result.return_16 - result.cross_mean_return_16)
    result["direction_market_alignment_4"] = (
        result.direction_n * result.cross_mean_return_4)
    result["direction_market_alignment_16"] = (
        result.direction_n * result.cross_mean_return_16)
    result["direction_btc_alignment_4"] = (
        result.direction_n * result.btc_return_4)
    result["direction_btc_alignment_16"] = (
        result.direction_n * result.btc_return_16)
    return result


def design(train: pd.DataFrame) -> dict:
    return {
        "symbols": list(parent.CORE_SYMBOLS),
        "medians": {name: float(pd.to_numeric(
            train[name], errors="coerce").median()) for name in FEATURES},
    }


def matrix(frame: pd.DataFrame, schema: dict) -> np.ndarray:
    columns = [pd.to_numeric(frame[name], errors="coerce").fillna(
        schema["medians"][name]).to_numpy(float) for name in FEATURES]
    symbols = frame.symbol.astype(str).to_numpy()
    columns.extend((symbols == symbol).astype(float)
                   for symbol in schema["symbols"])
    return np.column_stack(columns)


def fit_model(train: pd.DataFrame, schema: dict, target_name: str,
              config: tuple[int, int, float]):
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
    model.fit(matrix(train, schema), target)
    return model


def score(frame: pd.DataFrame, model, schema: dict,
          target_name: str) -> pd.DataFrame:
    result = frame.copy()
    values = matrix(result, schema)
    result["model_quality"] = (
        model.predict(values) if target_name == "signed_return"
        else model.predict_proba(values)[:, 1])
    return result


def main() -> None:
    if not parent.CACHE_OUT.exists():
        raise FileNotFoundError(
            "run crypto_raw_hourly_direction_oct19_audit.py first")
    historical = enrich(pd.read_pickle(parent.CACHE_OUT))
    fit = parent.stage(historical, parent.FIT)
    schema = design(fit)
    candidates = []
    for target_name in MODEL_TARGETS:
        for config in CONFIGS:
            model = fit_model(fit, schema, target_name, config)
            scored = score(historical, model, schema, target_name)
            selection = parent.stage(scored, parent.SELECT)
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
                            "model_target": target_name,
                            "max_leaf_nodes": config[0],
                            "min_samples_leaf": config[1],
                            "l2_regularization": config[2],
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
                            "_model": model,
                            "_scored": scored,
                        })
    candidates.sort(key=lambda item: (
        item["admissible"], item["development_floor_pct"],
        -max(item["selection"]["dd_pct"], item["confirmation"]["dd_pct"]),
        item["selection"]["return_pct"] + item["confirmation"]["return_pct"],
    ), reverse=True)
    selected = candidates[0]
    model = selected["_model"]
    scored = selected["_scored"]
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
    direct = enrich(pd.concat(direct_parts, ignore_index=True))
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

    public = {key: value for key, value in selected.items()
              if not key.startswith("_")}
    status = "REJECTED_OBJECTIVE"
    if (selected["admissible"] and selected["development_floor_pct"] >= 30
            and holdouts["margin_5pct"]["validation_2024"]["negative_months"] == 0
            and holdouts["margin_5pct"]["later_2025_2026"]["negative_months"] == 0
            and direct_values["margin_5pct"]["negative_months"] == 0
            and min(direct_values["margin_5pct"]["monthly_return_pct"].values()) >= 30):
        status = "RAW_REGIME_DIRECTION_SHADOW_CANDIDATE"
    report = {
        "status": status,
        "ready_for_live_orders": False,
        "protocol": {
            "base_geometry": "1% target / 0.5% stop / four-hour timeout",
            "fit": list(parent.FIT), "selection": list(parent.SELECT),
            "confirmation": list(parent.CONFIRM),
            "holdouts": [*parent.VALIDATE, *parent.LATER, "direct X-Perp"],
            "entry": "market at next 15-minute open; hourly decisions",
            "money": "client 3% or 5% current-equity margin; fixed 10x",
            "priority": "worst complete-month return; no win-rate gate",
            "features": list(FEATURES) + ["symbol one-hot"],
        },
        "history_rows": len(historical),
        "candidate_count": len(candidates),
        "admissible_count": sum(item["admissible"] for item in candidates),
        "selected_before_holdouts": public,
        "top20_before_holdouts": [
            {key: value for key, value in item.items() if not key.startswith("_")}
            for item in candidates[:20]],
        "holdouts": holdouts,
        "direct_xperp": direct_values,
        "sources": {
            str(parent.CACHE_OUT): digest(parent.CACHE_OUT),
            str(parent.OUT): digest(parent.OUT),
            **parent.SOURCE_HASHES,
            str(Path(__file__).resolve()): digest(Path(__file__).resolve()),
        },
        "limitations": [
            "Only symbols with continuous 2020-2026 history are modeled.",
            "Historical named periods are available six-month windows.",
            "Direct X-Perp history is shorter than a full market cycle.",
            "Gross results exclude fees at user request.",
            "No backtest guarantees a future monthly return.",
        ],
    }
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({
        "status": status, "candidate_count": len(candidates),
        "admissible_count": report["admissible_count"],
        "selected": public, "holdouts": holdouts,
        "direct_xperp": direct_values,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
