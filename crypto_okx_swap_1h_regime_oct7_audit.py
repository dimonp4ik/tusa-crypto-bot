"""Direct OKX global USDT-swap one-hour regime audit."""
from __future__ import annotations

import hashlib
import json
import math
import pickle
from pathlib import Path

import numba as nb
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

import crypto_5m_regime_router_literal_oct6_audit as portfolio


ROOT = Path("reports/audit_2026_09_08/crypto_okx_swap_1h_history")
MANIFEST = ROOT / "manifest.json"
OUT = Path("reports/audit_2026_09_08/crypto_okx_swap_1h_regime_oct7.json")
ROWS_OUT = Path("reports/audit_2026_09_08/crypto_okx_swap_1h_regime_oct7.pkl")
FAMILIES = ("trend", "pullback", "breakout", "exhaustion",
            "compression_release", "breadth", "relative_leader")
FEATURES = (
    "ret1", "ret3", "ret6", "ret12", "ret24", "ret72", "ret168",
    "body1", "range1", "atr14_pct", "volume_ratio", "range_ratio",
    "eff24", "eff72", "ema12_dist", "ema48_dist", "ema168_dist",
    "upper_wick", "lower_wick", "breakout_up", "breakout_down",
    "rv6", "rv24", "rv168", "fast_slow", "release3",
    "autocorr24", "autocorr168", "cross_z6", "cross_z24", "cross_z72",
    "breadth6", "breadth24", "breadth72", "btc_ret6", "btc_ret24",
    "btc_ret168",
)
TARGET_PCT = .5
STOP_PCT = 1.0
HOLD_BARS = 12
CONFIGS = (
    {"leaves": 15, "leaf": 160, "l2": 3.0},
    {"leaves": 31, "leaf": 280, "l2": 5.0},
)
QUANTILES = (.70, .75, .80, .85, .90, .925, .95, .96, .97, .98, .985, .99)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load() -> tuple[dict, dict]:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    candles, sources = {}, {str(MANIFEST): digest(MANIFEST)}
    for symbol, item in manifest["symbols"].items():
        path = Path(item["path"])
        raw = path.read_bytes()
        found = hashlib.sha256(raw).hexdigest()
        if found != item["sha256"]:
            raise ValueError(f"OKX 1h hash mismatch: {symbol}")
        candle = pickle.loads(raw)
        if any(right - left != 3600 for left, right in zip(
                candle["time"], candle["time"][1:])):
            raise ValueError(f"non-contiguous OKX 1h history: {symbol}")
        candles[symbol] = candle
        sources[str(path)] = found
    return candles, sources


def symbol_features(symbol: str, candle: dict) -> pd.DataFrame:
    frame = pd.DataFrame({key: np.asarray(candle[key], dtype=float)
                          for key in ("open", "high", "low", "close",
                                      "volume_quote")})
    times = np.asarray(candle["time"], dtype=np.int64)
    close = frame.close
    log_return = np.log(close).diff()
    previous = close.shift(1)
    true_range = pd.concat(((frame.high - frame.low),
                            (frame.high - previous).abs(),
                            (frame.low - previous).abs()), axis=1).max(axis=1)
    candle_range = (frame.high - frame.low).replace(0, np.nan)
    upper = frame.high - np.maximum(frame.open, frame.close)
    lower = np.minimum(frame.open, frame.close) - frame.low
    path24 = close.diff().abs().rolling(24, min_periods=18).sum()
    path72 = close.diff().abs().rolling(72, min_periods=54).sum()
    prior_high = frame.high.shift(1).rolling(24, min_periods=18).max()
    prior_low = frame.low.shift(1).rolling(24, min_periods=18).min()
    entry = np.arange(1, len(frame))
    signal = entry - 1
    valid = times[entry] - times[signal] == 3600
    entry, signal = entry[valid], signal[valid]
    stamp = pd.to_datetime(times[entry], unit="s", utc=True)
    values = {
        "symbol": symbol, "entry_index": entry, "entry_time": times[entry],
        "year": stamp.year.astype(int),
        "month": (stamp.year * 100 + stamp.month).astype(int),
        "body1": ((frame.close - frame.open) / frame.open).iloc[signal].to_numpy(float),
        "range1": (candle_range / frame.close).iloc[signal].to_numpy(float),
        "atr14_pct": (true_range.rolling(14).mean() / close).iloc[signal].to_numpy(float),
        "volume_ratio": (frame.volume_quote / frame.volume_quote.shift(1).rolling(
            168, min_periods=120).median()).iloc[signal].to_numpy(float),
        "range_ratio": (candle_range / candle_range.shift(1).rolling(
            168, min_periods=120).median()).iloc[signal].to_numpy(float),
        "eff24": ((close - close.shift(24)).abs() / path24).iloc[signal].to_numpy(float),
        "eff72": ((close - close.shift(72)).abs() / path72).iloc[signal].to_numpy(float),
        "ema12_dist": (close / close.ewm(span=12, adjust=False).mean() - 1).iloc[signal].to_numpy(float),
        "ema48_dist": (close / close.ewm(span=48, adjust=False).mean() - 1).iloc[signal].to_numpy(float),
        "ema168_dist": (close / close.ewm(span=168, adjust=False).mean() - 1).iloc[signal].to_numpy(float),
        "upper_wick": (upper / candle_range).iloc[signal].to_numpy(float),
        "lower_wick": (lower / candle_range).iloc[signal].to_numpy(float),
        "breakout_up": (close / prior_high - 1).iloc[signal].to_numpy(float),
        "breakout_down": (prior_low / close - 1).iloc[signal].to_numpy(float),
    }
    for window in (1, 3, 6, 12, 24, 72, 168):
        values[f"ret{window}"] = (close / close.shift(window) - 1).iloc[
            signal].to_numpy(float)
    for window in (6, 24, 168):
        values[f"rv{window}"] = log_return.rolling(
            window, min_periods=max(4, int(.75 * window))).std(ddof=0).iloc[
                signal].to_numpy(float)
    rv6 = log_return.rolling(6, min_periods=4).std(ddof=0)
    rv168 = log_return.rolling(168, min_periods=120).std(ddof=0)
    fast_slow = rv6 / rv168.replace(0, np.nan)
    values["fast_slow"] = fast_slow.iloc[signal].to_numpy(float)
    values["release3"] = (fast_slow / fast_slow.shift(3)).iloc[signal].to_numpy(float)
    values["autocorr24"] = log_return.rolling(24, min_periods=18).corr(
        log_return.shift(1)).iloc[signal].to_numpy(float)
    values["autocorr168"] = log_return.rolling(168, min_periods=120).corr(
        log_return.shift(1)).iloc[signal].to_numpy(float)
    return pd.DataFrame(values).replace([np.inf, -np.inf], np.nan).dropna()


def build_features(candles: dict) -> pd.DataFrame:
    frame = pd.concat([symbol_features(symbol, candle)
                       for symbol, candle in candles.items()], ignore_index=True)
    grouped = frame.groupby("entry_time", sort=False)
    for window in (6, 24, 72):
        mean = grouped[f"ret{window}"].transform("mean")
        std = grouped[f"ret{window}"].transform("std").replace(0, np.nan)
        frame[f"cross_z{window}"] = (frame[f"ret{window}"] - mean) / std
        frame[f"breadth{window}"] = grouped[f"ret{window}"].transform(
            lambda values: (values > 0).mean())
    btc = frame[frame.symbol.eq("BTCUSDT")][
        ["entry_time", "ret6", "ret24", "ret168"]].rename(columns={
            "ret6": "btc_ret6", "ret24": "btc_ret24",
            "ret168": "btc_ret168"})
    return frame.merge(btc, on="entry_time", how="left",
                       validate="many_to_one").dropna()


def proposals(frame: pd.DataFrame) -> pd.DataFrame:
    pieces = []

    def add(name: str, direction: pd.Series, mask: pd.Series) -> None:
        selected = mask & direction.ne(0)
        part = frame.loc[selected].copy()
        part["direction"] = direction.loc[selected].astype(np.int8)
        part[f"family_{name}"] = 1
        pieces.append(part)

    trend = np.sign(frame.ret24)
    add("trend", trend, frame.ret24.abs().ge(.01)
        & (frame.ret168 * trend).ge(.015) & frame.eff24.ge(.15))

    pullback = np.sign(frame.ret168)
    add("pullback", pullback, frame.ret168.abs().ge(.02)
        & (frame.ret6 * pullback).between(-.018, -.001)
        & (frame.body1 * pullback).gt(0))

    breakout = np.sign(frame.body1)
    distance = frame.breakout_up.where(breakout > 0, frame.breakout_down)
    add("breakout", breakout, frame.body1.abs().ge(.002)
        & frame.range_ratio.ge(1.2) & distance.ge(-.0015))

    exhaustion = -np.sign(frame.ret12)
    rejection = frame.lower_wick.where(exhaustion > 0, frame.upper_wick)
    add("exhaustion", exhaustion, frame.ret12.abs().ge(.015)
        & frame.cross_z6.abs().ge(.8) & rejection.ge(.25)
        & (frame.body1 * exhaustion).gt(0))

    release = np.sign(frame.ret3)
    add("compression_release", release, frame.fast_slow.between(.6, 1.5)
        & frame.release3.ge(1.5) & (frame.ret3 * release).ge(.002)
        & frame.range_ratio.ge(1.1))

    breadth = np.sign(frame.btc_ret24)
    directed_breadth = frame.breadth24.where(breadth > 0, 1 - frame.breadth24)
    add("breadth", breadth, frame.btc_ret24.abs().ge(.008)
        & directed_breadth.ge(.625) & (frame.ret24 * breadth).ge(.004))

    leader = np.sign(frame.cross_z24)
    add("relative_leader", leader, frame.cross_z24.abs().ge(1.0)
        & (frame.ret6 * leader).ge(.002) & frame.volume_ratio.ge(.8))

    combined = pd.concat(pieces, ignore_index=True)
    family_columns = [f"family_{name}" for name in FAMILIES]
    for column in family_columns:
        if column not in combined:
            combined[column] = 0
    keys = ["symbol", "entry_time", "direction"]
    flags = combined.groupby(keys, sort=False)[family_columns].max().reset_index()
    base = combined.sort_values(keys).drop_duplicates(keys).drop(columns=family_columns)
    return base.merge(flags, on=keys, how="left", validate="one_to_one")


@nb.njit(cache=True)
def resolve(open_, high, low, close, times, entries, directions):
    n = len(entries)
    signed = np.empty(n, dtype=np.float64)
    exits = np.empty(n, dtype=np.int64)
    hits = np.zeros(n, dtype=np.int8)
    reasons = np.empty(n, dtype=np.int8)
    target_move = TARGET_PCT / 100
    stop_move = STOP_PCT / 100
    for row in range(n):
        first = entries[row]
        direction = directions[row]
        entry = open_[first]
        target = entry * (1 + direction * target_move)
        stop = entry * (1 - direction * stop_move)
        last = first
        result = 0.0
        reason = 2
        end = min(len(times), first + HOLD_BARS)
        resolved = False
        for index in range(first, end):
            if index > first and times[index] - times[index - 1] != 3600:
                break
            last = index
            stop_hit = low[index] <= stop if direction > 0 else high[index] >= stop
            target_hit = high[index] >= target if direction > 0 else low[index] <= target
            if stop_hit:
                fill = min(open_[index], stop) if direction > 0 else max(open_[index], stop)
                result = direction * (fill / entry - 1)
                reason = 0
                resolved = True
                break
            if target_hit:
                result = target_move
                hits[row] = 1
                reason = 1
                resolved = True
                break
        if not resolved:
            result = direction * (close[last] / entry - 1)
            result = max(-stop_move, min(target_move, result))
        signed[row] = result
        exits[row] = times[last] + 3600
        reasons[row] = reason
    return signed, exits, hits, reasons


def add_outcomes(frame: pd.DataFrame, candles: dict) -> pd.DataFrame:
    parts = []
    for symbol, part in frame.groupby("symbol", sort=False):
        candle = candles[symbol]
        arrays = [np.asarray(candle[key], dtype=float)
                  for key in ("open", "high", "low", "close")]
        values, exits, hits, reasons = resolve(
            *arrays, np.asarray(candle["time"], dtype=np.int64),
            part.entry_index.to_numpy(np.int64),
            part.direction.to_numpy(np.int8))
        found = part.copy()
        found["signed_return"] = values
        found["exit_time"] = exits
        found["target_hit"] = hits.astype(bool)
        found["exit_reason"] = reasons
        parts.append(found)
    return pd.concat(parts, ignore_index=True)


def schema(frame: pd.DataFrame) -> list[str]:
    return [*FEATURES, *[f"family_{name}" for name in FAMILIES],
            *[f"symbol_{name}" for name in sorted(frame.symbol.unique())]]


def matrix(frame: pd.DataFrame, columns: list[str]) -> np.ndarray:
    output = pd.DataFrame(index=frame.index)
    direction = frame.direction.to_numpy(float)
    nondirectional = {"range1", "atr14_pct", "volume_ratio", "range_ratio",
                      "eff24", "eff72", "upper_wick", "lower_wick",
                      "rv6", "rv24", "rv168", "fast_slow", "release3",
                      "autocorr24", "autocorr168"}
    for name in FEATURES:
        if name.startswith("breadth"):
            output[name] = np.where(direction > 0, frame[name], 1 - frame[name])
        elif name in nondirectional:
            output[name] = frame[name]
        elif name == "breakout_up":
            output[name] = np.where(direction > 0, frame.breakout_up, frame.breakout_down)
        elif name == "breakout_down":
            output[name] = np.where(direction > 0, frame.breakout_down, frame.breakout_up)
        else:
            output[name] = frame[name] * direction
    for family in FAMILIES:
        output[f"family_{family}"] = frame[f"family_{family}"].astype(float)
    for name in columns:
        if name.startswith("symbol_"):
            output[name] = (frame.symbol == name[7:]).astype(float)
    return output.reindex(columns=columns).replace(
        [np.inf, -np.inf], np.nan).fillna(0).to_numpy(np.float32)


def selection_ok(value: dict) -> bool:
    return bool(value["n"] >= 700 and value["return_pct"] > 0
                and value["literal_target_hit_rate"] >= 68
                and value["dd_pct"] <= 7
                and value["trades_per_calendar_day"] >= 2.5)


def validation_ok(value: dict) -> bool:
    return bool(value["n"] >= 500 and value["return_pct"] > 0
                and value["literal_target_hit_rate"] >= 65
                and value["dd_pct"] <= 7
                and value["trades_per_calendar_day"] >= 2.0)


def public(item: dict | None) -> dict | None:
    if item is None:
        return None
    return {key: value for key, value in item.items()
            if key not in ("_frame", "_model")}


def main() -> None:
    candles, sources = load()
    raw = build_features(candles)
    frame = add_outcomes(proposals(raw), candles)
    fit = frame.year.isin((2020, 2021)).to_numpy()
    columns = schema(frame.loc[fit])
    x = matrix(frame, columns)
    labels = frame.target_hit.astype(int).to_numpy()
    selection = []
    for config in CONFIGS:
        model = HistGradientBoostingClassifier(
            max_iter=180, learning_rate=.06, max_leaf_nodes=config["leaves"],
            min_samples_leaf=config["leaf"], l2_regularization=config["l2"],
            random_state=20261007)
        model.fit(x[fit], labels[fit])
        scored = frame.copy()
        scored["quality"] = model.predict_proba(x)[:, 1]
        values = scored.loc[scored.year.eq(2022), "quality"]
        for quantile in QUANTILES:
            threshold = float(values.quantile(quantile))
            score = portfolio.evaluate(scored, (2022,), threshold)
            if selection_ok(score):
                selection.append({"model": config, "quantile": quantile,
                                  "threshold": threshold,
                                  "selection_2022": score,
                                  "_frame": scored, "_model": model})
    validated = []
    diagnostics = []
    for item in selection:
        value = portfolio.evaluate(item["_frame"], (2023,), item["threshold"])
        diagnostics.append(public({**item, "validation_2023": value}))
        if validation_ok(value):
            candidate = {**item, "validation_2023": value}
            candidate["score"] = round(
                min(item["selection_2022"]["return_pct"], value["return_pct"])
                - 2 * max(item["selection_2022"]["dd_pct"], value["dd_pct"])
                + min(item["selection_2022"]["trades_per_calendar_day"],
                      value["trades_per_calendar_day"]), 8)
            validated.append(candidate)
    validated.sort(key=lambda item: item["score"], reverse=True)
    frozen = validated[0] if validated else None
    test_2024 = later = checks = None
    if frozen:
        test_2024 = portfolio.evaluate(frozen["_frame"], (2024,), frozen["threshold"])
        if validation_ok(test_2024):
            later = {str(year): portfolio.evaluate(
                frozen["_frame"], (year,), frozen["threshold"])
                for year in (2025, 2026)}
            checks = {year: validation_ok(value) for year, value in later.items()}
        frozen["_frame"].loc[
            frozen["_frame"].quality.ge(frozen["threshold"])].to_pickle(ROWS_OUT)
    report = {
        "status": ("RESEARCH_CANDIDATE_AWAITING_DIRECT_FORWARD"
                   if checks and all(checks.values()) else "REJECTED"),
        "protocol": {
            "venue": "OKX global USDT swap", "bar": "1H",
            "fit": [2020, 2021], "selection": 2022,
            "validation_and_ranking": 2023, "untouched_test": 2024,
            "hidden_if_test_passes": [2025, 2026],
            "families": list(FAMILIES), "model_grid": list(CONFIGS),
            "entry": "next 1h market open", "target_pct": TARGET_PCT,
            "stop_pct": STOP_PCT, "hold_bars": HOLD_BARS,
            "money": "3% equity margin, fixed 10x gross",
        },
        "feature_rows": len(raw), "proposal_rows": len(frame),
        "fit_literal_target_prevalence_pct": round(100 * float(labels[fit].mean()), 4),
        "qualified_2022": len(selection), "qualified_2023": len(validated),
        "validation_diagnostics": diagnostics,
        "frozen_before_2024": public(frozen), "test_2024": test_2024,
        "later": later, "later_checks": checks,
        "sources": {**sources, str(Path(__file__)): digest(Path(__file__))},
        "limitations": [
            "The model uses hourly OHLCV aggregates and has no historical order-book depth.",
            "A historical pass still requires the separate EU X-Perp transfer and forward shadow observation.",
        ],
    }
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({key: report[key] for key in (
        "status", "feature_rows", "proposal_rows",
        "fit_literal_target_prevalence_pct", "qualified_2022",
        "qualified_2023", "frozen_before_2024", "test_2024", "later",
        "later_checks")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
