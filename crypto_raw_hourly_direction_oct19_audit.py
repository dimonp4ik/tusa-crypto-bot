"""Causal raw-candle crypto direction model with literal price targets.

Every hourly candidate is formed from a closed 15-minute candle, enters at the
next candle open, and exits by a 1% target, 0.5% stop, or four-hour timeout.
Models fit 2020-2021; policy selection uses 2022-2023; all later periods and
direct X-Perp data are holdouts.  Stop wins ambiguous OHLC bars.
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

import crypto_high_target_tree_oct5_audit as base
import crypto_precision_be_runner_oct18_audit as rails
from crypto_runtime_shadow_portfolio_oct3_audit import money


ROOT = Path("reports/audit_2026_09_08")
OUT = ROOT / "crypto_raw_hourly_direction_oct19.json"
ROWS_OUT = ROOT / "crypto_raw_hourly_direction_xperp_oct19.pkl"
CACHE_OUT = ROOT / "crypto_raw_hourly_direction_oct19.pkl"
CORE_SYMBOLS = ("BTCUSDT", "ETHUSDT", "LINKUSDT", "XLMUSDT", "XRPUSDT")
FIT = ("2020h1", "2020h2", "2021h1", "2021h2")
SELECT = ("2022h1",)
CONFIRM = ("2022h2", "2023")
VALIDATE = ("2024",)
LATER = ("2025", "2026")
TARGET_RETURN = .01
STOP_RETURN = .005
HOLD_BARS = 16
FEATURES = (
    "return_1", "return_2", "return_4", "return_8", "return_16",
    "return_32", "return_96", "ema8_deviation", "ema32_deviation",
    "atr14_fraction", "range32_position", "volume_z20", "body_fraction",
    "candle_range_fraction", "close_location", "efficiency16",
    "efficiency64", "hour_sin", "hour_cos", "weekday_sin", "weekday_cos",
    "direction_n",
)
CONFIGS = tuple(
    (leaves, leaf, l2)
    for leaves in (7, 15)
    for leaf in (100, 300)
    for l2 in (1.0, 5.0)
)
MODEL_TARGETS = ("signed_return", "profitable_exit")
QUANTILES = (.50, .60, .70, .75, .80, .85, .875, .90, .925, .95)
LOSS_PAUSES = (3, 5)
HOURLY_CAPS = (1, 2)
SOURCE_HASHES: dict[str, str] = {}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def candle_path(period: str, symbol: str) -> Path:
    return Path("backtest_cache") / (
        f"{symbol}_15min_{rails.PERIOD_CACHE[period]}.pkl")


def load_candles(path: Path) -> pd.DataFrame:
    raw = path.read_bytes()
    SOURCE_HASHES[str(path)] = hashlib.sha256(raw).hexdigest()
    values = pickle.loads(raw)
    return pd.DataFrame(values)


def _outcome(candles: pd.DataFrame, start: int, direction_n: int
             ) -> tuple[float, str, float]:
    entry = float(candles.open.iloc[start])
    target = entry * (1 + direction_n * TARGET_RETURN)
    stop = entry * (1 - direction_n * STOP_RETURN)
    last = min(start + HOLD_BARS - 1, len(candles) - 1)
    for index in range(start, last + 1):
        opened = float(candles.open.iloc[index])
        high = float(candles.high.iloc[index])
        low = float(candles.low.iloc[index])
        stop_hit = low <= stop if direction_n > 0 else high >= stop
        if stop_hit:
            fill = min(opened, stop) if direction_n > 0 else max(opened, stop)
            return (direction_n * (fill / entry - 1), "SL",
                    float(candles.time.iloc[index]) + 900)
        target_hit = high >= target if direction_n > 0 else low <= target
        if target_hit:
            return (TARGET_RETURN, "TP",
                    float(candles.time.iloc[index]) + 900)
    close = float(candles.close.iloc[last])
    return (direction_n * (close / entry - 1), "TIME",
            float(candles.time.iloc[last]) + 900)


def build_symbol_period(period: str, symbol: str, candles: pd.DataFrame
                        ) -> pd.DataFrame:
    frame = candles.copy()
    close = pd.to_numeric(frame.close, errors="coerce")
    opened = pd.to_numeric(frame.open, errors="coerce")
    high = pd.to_numeric(frame.high, errors="coerce")
    low = pd.to_numeric(frame.low, errors="coerce")
    volume = pd.to_numeric(frame.volume, errors="coerce")
    previous = close.shift(1)
    true_range = pd.concat([
        high - low, (high - previous).abs(), (low - previous).abs()], axis=1
    ).max(axis=1)
    features = pd.DataFrame(index=frame.index)
    for lag in (1, 2, 4, 8, 16, 32, 96):
        features[f"return_{lag}"] = close / close.shift(lag) - 1
    features["ema8_deviation"] = close / close.ewm(
        span=8, adjust=False).mean() - 1
    features["ema32_deviation"] = close / close.ewm(
        span=32, adjust=False).mean() - 1
    features["atr14_fraction"] = true_range.rolling(14).mean() / close
    low32 = low.rolling(32).min()
    high32 = high.rolling(32).max()
    features["range32_position"] = (close - low32) / (high32 - low32)
    features["volume_z20"] = ((volume - volume.rolling(20).mean())
                              / volume.rolling(20).std())
    features["body_fraction"] = (close - opened) / opened
    features["candle_range_fraction"] = (high - low) / opened
    features["close_location"] = (close - low) / (high - low)
    movement = close.diff().abs()
    for window in (16, 64):
        features[f"efficiency{window}"] = (
            (close - close.shift(window)).abs()
            / movement.rolling(window).sum())
    stamp = pd.to_datetime(frame.time, unit="s", utc=True)
    hour = stamp.dt.hour + stamp.dt.minute / 60
    features["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    features["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    features["weekday_sin"] = np.sin(2 * np.pi * stamp.dt.weekday / 7)
    features["weekday_cos"] = np.cos(2 * np.pi * stamp.dt.weekday / 7)

    rows = []
    for signal_index in range(96, len(frame) - HOLD_BARS - 1):
        entry_index = signal_index + 1
        entry_time = int(frame.time.iloc[entry_index])
        if entry_time % 3600 != 0:
            continue
        values = features.loc[signal_index, [name for name in FEATURES
                                             if name != "direction_n"]]
        if not np.isfinite(values.to_numpy(float)).all():
            continue
        common = {name: float(values[name]) for name in values.index}
        for direction_n, direction in ((1, "LONG"), (-1, "SHORT")):
            signed_return, outcome, exit_time = _outcome(
                frame, entry_index, direction_n)
            rows.append({
                **common,
                "period": period,
                "symbol": symbol,
                "entry_time": entry_time,
                "exit_time": exit_time,
                "entry": float(frame.open.iloc[entry_index]),
                "direction_n": direction_n,
                "direction": direction,
                "signed_return": signed_return,
                "outcome": outcome,
            })
    return pd.DataFrame(rows)


def build_history() -> pd.DataFrame:
    frames = []
    for period in base.PERIODS:
        for symbol in CORE_SYMBOLS:
            path = candle_path(period, symbol)
            if path.exists():
                frames.append(build_symbol_period(
                    period, symbol, load_candles(path)))
    result = pd.concat(frames, ignore_index=True)
    order = {period: index for index, period in enumerate(base.PERIODS)}
    result["_period_order"] = result.period.map(order)
    result = result.sort_values(["symbol", "entry_time", "direction_n",
                                 "_period_order"])
    result = result.drop_duplicates(
        ["symbol", "entry_time", "direction_n"], keep="last")
    return result.drop(columns="_period_order").reset_index(drop=True)


def design(train: pd.DataFrame) -> dict:
    return {
        "symbols": list(CORE_SYMBOLS),
        "medians": {name: float(train[name].median()) for name in FEATURES},
    }


def matrix(frame: pd.DataFrame, schema: dict) -> np.ndarray:
    columns = [pd.to_numeric(frame[name], errors="coerce").fillna(
        schema["medians"][name]).to_numpy(float) for name in FEATURES]
    symbols = frame.symbol.astype(str).to_numpy()
    columns.extend((symbols == symbol).astype(float)
                   for symbol in schema["symbols"])
    return np.column_stack(columns)


def stage(frame: pd.DataFrame, periods: tuple[str, ...]) -> pd.DataFrame:
    return frame[frame.period.isin(periods)].copy()


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


def candidate_rows(frame: pd.DataFrame, threshold: float) -> list[dict]:
    chosen = frame[frame.model_quality >= threshold].copy()
    chosen = chosen.sort_values(
        ["symbol", "entry_time", "model_quality", "direction_n"],
        ascending=[True, True, False, False], kind="mergesort")
    chosen = chosen.drop_duplicates(["symbol", "entry_time"], keep="first")
    output = []
    for row in chosen.to_dict("records"):
        signed = float(row["signed_return"])
        output.append({
            **row,
            "exit_reason": str(row["outcome"]),
            "outcome": "SL" if signed < 0 else "TP",
            "module": "raw_hourly_direction",
            "source": "raw_hourly_direction",
            "source_priority": 0,
            "module_priority": 0,
            "model_probability": float(row["model_quality"]),
            "signed_price_return": signed,
        })
    return output


def evaluate(frame: pd.DataFrame, threshold: float, loss_pause: int,
             hourly_cap: int, margin: float) -> tuple[dict, list[dict]]:
    rows = rails.ranked_runtime_gate(
        candidate_rows(frame, threshold), loss_pause=loss_pause,
        hourly_cap=hourly_cap)
    value = money(rows, margin)
    value.update({
        "literal_tp_rate_pct": round(100 * sum(
            row["exit_reason"] == "TP" for row in rows) / len(rows), 2)
        if rows else 0.0,
        "profitable_exit_rate_pct": round(100 * sum(
            float(row["signed_return"]) > 0 for row in rows) / len(rows), 2)
        if rows else 0.0,
        "max_entries_same_timestamp": max(collections.Counter(
            int(float(row["entry_time"])) for row in rows).values())
        if rows else 0,
    })
    return value, rows


def month_floor(value: dict, months: tuple[str, ...]) -> float:
    observed = value.get("monthly_return_pct", {})
    return (min(float(observed[month]) for month in months)
            if set(months).issubset(observed) else -999.0)


def admissible(value: dict, months: tuple[str, ...]) -> bool:
    return bool(value["n"] >= 100 and value["dd_pct"] <= 10
                and value["trades_per_calendar_day"] >= 3
                and month_floor(value, months) >= 0)


def main() -> None:
    historical = build_history()
    historical.to_pickle(CACHE_OUT)
    fit = stage(historical, FIT)
    schema = design(fit)
    candidates = []
    for target_name in MODEL_TARGETS:
        for config in CONFIGS:
            model = fit_model(fit, schema, target_name, config)
            scored = score(historical, model, schema, target_name)
            selection = stage(scored, SELECT)
            for quantile in QUANTILES:
                threshold = float(selection.model_quality.quantile(quantile))
                for loss_pause in LOSS_PAUSES:
                    for hourly_cap in HOURLY_CAPS:
                        select_value, _ = evaluate(
                            selection, threshold, loss_pause, hourly_cap, .03)
                        confirm_value, _ = evaluate(
                            stage(scored, CONFIRM), threshold,
                            loss_pause, hourly_cap, .03)
                        floor = min(
                            month_floor(select_value, rails.SELECTION_MONTHS),
                            month_floor(confirm_value, rails.CONFIRMATION_MONTHS))
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
                                admissible(select_value, rails.SELECTION_MONTHS)
                                and admissible(confirm_value,
                                               rails.CONFIRMATION_MONTHS)),
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
        validation, _ = evaluate(
            stage(scored, VALIDATE), threshold, loss_pause, hourly_cap, margin)
        later, _ = evaluate(
            stage(scored, LATER), threshold, loss_pause, hourly_cap, margin)
        holdouts[name] = {"validation_2024": validation,
                          "later_2025_2026": later}

    direct_frames = []
    for symbol in CORE_SYMBOLS:
        path = rails.DIRECT_CACHE / f"{symbol}_15min_current.pkl"
        direct_frames.append(build_symbol_period(
            "xperp", symbol, load_candles(path)))
    direct = pd.concat(direct_frames, ignore_index=True)
    direct_scored = score(
        direct, model, schema, str(selected["model_target"]))
    direct_values = {}
    output_rows = []
    for margin in (.03, .05):
        value, rows = evaluate(
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
        status = "RAW_HOURLY_DIRECTION_SHADOW_CANDIDATE"
    report = {
        "status": status,
        "ready_for_live_orders": False,
        "protocol": {
            "symbols": list(CORE_SYMBOLS),
            "fit": list(FIT), "selection": list(SELECT),
            "confirmation": list(CONFIRM),
            "holdouts": [*VALIDATE, *LATER, "direct X-Perp"],
            "entry": "market at next 15-minute open; hourly decisions",
            "exit": "market close at +1%, -0.5%, or four-hour timeout",
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
        "sources": {**SOURCE_HASHES,
                    str(Path(__file__).resolve()): digest(Path(__file__).resolve())},
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
        "status": status, "history_rows": len(historical),
        "candidate_count": len(candidates),
        "admissible_count": report["admissible_count"],
        "selected": public, "holdouts": holdouts,
        "direct_xperp": direct_values,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
