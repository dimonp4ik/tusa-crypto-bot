"""Frozen one-hour OKX regime model for forward shadow observation only.

The feature and proposal rules mirror crypto_okx_swap_1h_regime_oct7_audit.py.
The exported histogram model is evaluated by pure Python, so the production
image does not need numpy, pandas, or sklearn.  Incomplete cross-market input
fails closed because breadth and relative-strength values would otherwise no
longer mean what they meant during the historical audit.
"""
from __future__ import annotations

import hashlib
import json
import math
import statistics
import struct
from pathlib import Path

from src.portable_hgb import score_named


SUPPORTED_SYMBOLS = frozenset({
    "AAVEUSDT", "AVAXUSDT", "BTCUSDT", "ETHUSDT",
    "LINKUSDT", "SOLUSDT", "XLMUSDT", "XRPUSDT",
})
FAMILIES = (
    "trend", "pullback", "breakout", "exhaustion",
    "compression_release", "breadth", "relative_leader",
)
MODEL_PATH = Path(__file__).with_name("frozen_crypto_okx_1h_hgb.json")
_MODEL = json.loads(MODEL_PATH.read_text(encoding="utf-8"))
# Kept for reproducibility of earlier append audits.  The active Oct 22 route
# below uses one fixed material bracket and never evaluates this cascade.
CASCADE_PATH = Path(__file__).with_name("frozen_crypto_okx_1h_target_cascade.json")
_CASCADE = json.loads(CASCADE_PATH.read_text(encoding="utf-8"))
POLICY_PATH = Path(__file__).with_name(
    "frozen_crypto_okx_1h_stable_direction.json")
_POLICY = json.loads(POLICY_PATH.read_text(encoding="utf-8"))
if hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest() != _POLICY["base_model_sha256"]:
    raise RuntimeError("frozen OKX 1h model does not match stable route policy")
QUALITY_THRESHOLD = float(_POLICY["quality_threshold"])
TARGET_PCT = float(_POLICY["target_pct"])
STOP_PCT = float(_POLICY["stop_pct"])
HOLD_BARS = int(_POLICY["hold_bars"])


def _finite(value: float) -> bool:
    return math.isfinite(float(value))


def _float32(value: float) -> float:
    """Match the numpy.float32 design matrix used by the frozen audit."""
    return struct.unpack("!f", struct.pack("!f", float(value)))[0]


def _ret(close: list[float], index: int, window: int) -> float:
    return close[index] / close[index - window] - 1.0


def _pstdev(values: list[float]) -> float:
    return statistics.pstdev(values) if len(values) > 1 else 0.0


def _corr(left: list[float], right: list[float]) -> float:
    if len(left) != len(right) or len(left) < 2:
        return math.nan
    ml, mr = statistics.fmean(left), statistics.fmean(right)
    numerator = sum((x - ml) * (y - mr) for x, y in zip(left, right))
    dl = sum((x - ml) ** 2 for x in left)
    dr = sum((y - mr) ** 2 for y in right)
    return numerator / math.sqrt(dl * dr) if dl > 0 and dr > 0 else math.nan


def _ewm_last(values: list[float], span: int) -> float:
    alpha = 2.0 / (span + 1.0)
    mean = float(values[0])
    for value in values[1:]:
        mean = alpha * float(value) + (1.0 - alpha) * mean
    return mean


def _validate_candles(candle: dict) -> int | None:
    required = ("time", "open", "high", "low", "close", "volume_quote")
    if any(not candle.get(key) for key in required):
        return None
    lengths = {len(candle[key]) for key in required}
    if len(lengths) != 1 or next(iter(lengths)) < 220:
        return None
    times = [int(value) for value in candle["time"]]
    if any(right - left != 3600 for left, right in zip(times, times[1:])):
        return None
    return len(times) - 1


def _symbol_features(candle: dict) -> dict[str, float] | None:
    """Build the latest closed-bar feature row used by the frozen audit."""
    index = _validate_candles(candle)
    if index is None:
        return None
    open_ = [float(x) for x in candle["open"]]
    high = [float(x) for x in candle["high"]]
    low = [float(x) for x in candle["low"]]
    close = [float(x) for x in candle["close"]]
    volume = [float(x) for x in candle["volume_quote"]]
    if (any(value <= 0 for series in (open_, high, low, close)
            for value in series)
            or any(value < 0 for value in volume)):
        return None

    candle_range = [h - l for h, l in zip(high, low)]
    if candle_range[index] <= 0:
        return None
    log_returns = [math.nan] + [
        math.log(close[pos] / close[pos - 1]) for pos in range(1, len(close))
    ]
    true_ranges = [high[0] - low[0]]
    true_ranges.extend(max(high[pos] - low[pos],
                           abs(high[pos] - close[pos - 1]),
                           abs(low[pos] - close[pos - 1]))
                       for pos in range(1, len(close)))
    path24 = sum(abs(close[pos] - close[pos - 1])
                 for pos in range(index - 23, index + 1))
    path72 = sum(abs(close[pos] - close[pos - 1])
                 for pos in range(index - 71, index + 1))
    previous_volumes = volume[index - 168:index]
    previous_ranges = candle_range[index - 168:index]
    median_volume = statistics.median(previous_volumes)
    median_range = statistics.median(previous_ranges)
    if min(path24, path72, median_volume, median_range) <= 0:
        return None

    raw: dict[str, float] = {
        "body1": (close[index] - open_[index]) / open_[index],
        "range1": candle_range[index] / close[index],
        "atr14_pct": statistics.fmean(true_ranges[index - 13:index + 1]) / close[index],
        "volume_ratio": volume[index] / median_volume,
        "range_ratio": candle_range[index] / median_range,
        "eff24": abs(close[index] - close[index - 24]) / path24,
        "eff72": abs(close[index] - close[index - 72]) / path72,
        "ema12_dist": close[index] / _ewm_last(close, 12) - 1.0,
        "ema48_dist": close[index] / _ewm_last(close, 48) - 1.0,
        "ema168_dist": close[index] / _ewm_last(close, 168) - 1.0,
        "upper_wick": (high[index] - max(open_[index], close[index])) / candle_range[index],
        "lower_wick": (min(open_[index], close[index]) - low[index]) / candle_range[index],
        "breakout_up": close[index] / max(high[index - 24:index]) - 1.0,
        "breakout_down": min(low[index - 24:index]) / close[index] - 1.0,
    }
    for window in (1, 3, 6, 12, 24, 72, 168):
        raw[f"ret{window}"] = _ret(close, index, window)
    for window in (6, 24, 168):
        raw[f"rv{window}"] = _pstdev(log_returns[index - window + 1:index + 1])
    raw["fast_slow"] = raw["rv6"] / raw["rv168"] if raw["rv168"] else math.nan
    previous_fast = _pstdev(log_returns[index - 8:index - 2]) / _pstdev(
        log_returns[index - 170:index - 2])
    raw["release3"] = raw["fast_slow"] / previous_fast if previous_fast else math.nan
    for window in (24, 168):
        current = log_returns[index - window + 1:index + 1]
        lagged = log_returns[index - window:index]
        raw[f"autocorr{window}"] = _corr(current, lagged)
    if any(not _finite(value) for value in raw.values()):
        return None
    return raw


def _direction(value: float) -> int:
    return 1 if value > 0 else -1 if value < 0 else 0


def _family_directions(row: dict[str, float]) -> dict[int, set[str]]:
    found = {1: set(), -1: set()}
    direction = _direction(row["ret24"])
    if (direction and abs(row["ret24"]) >= .01
            and row["ret168"] * direction >= .015 and row["eff24"] >= .15):
        found[direction].add("trend")
    direction = _direction(row["ret168"])
    directed_ret6 = row["ret6"] * direction
    if (direction and abs(row["ret168"]) >= .02
            and -.018 <= directed_ret6 <= -.001
            and row["body1"] * direction > 0):
        found[direction].add("pullback")
    direction = _direction(row["body1"])
    distance = row["breakout_up"] if direction > 0 else row["breakout_down"]
    if (direction and abs(row["body1"]) >= .002
            and row["range_ratio"] >= 1.2 and distance >= -.0015):
        found[direction].add("breakout")
    direction = -_direction(row["ret12"])
    rejection = row["lower_wick"] if direction > 0 else row["upper_wick"]
    if (direction and abs(row["ret12"]) >= .015
            and abs(row["cross_z6"]) >= .8 and rejection >= .25
            and row["body1"] * direction > 0):
        found[direction].add("exhaustion")
    direction = _direction(row["ret3"])
    if (direction and .6 <= row["fast_slow"] <= 1.5
            and row["release3"] >= 1.5 and row["ret3"] * direction >= .002
            and row["range_ratio"] >= 1.1):
        found[direction].add("compression_release")
    direction = _direction(row["btc_ret24"])
    directed_breadth = row["breadth24"] if direction > 0 else 1-row["breadth24"]
    if (direction and abs(row["btc_ret24"]) >= .008
            and directed_breadth >= .625 and row["ret24"] * direction >= .004):
        found[direction].add("breadth")
    direction = _direction(row["cross_z24"])
    if (direction and abs(row["cross_z24"]) >= 1.0
            and row["ret6"] * direction >= .002 and row["volume_ratio"] >= .8):
        found[direction].add("relative_leader")
    return found


_NONDIRECTIONAL = {
    "range1", "atr14_pct", "volume_ratio", "range_ratio", "eff24", "eff72",
    "upper_wick", "lower_wick", "rv6", "rv24", "rv168", "fast_slow",
    "release3", "autocorr24", "autocorr168",
}


def _model_row(symbol: str, raw: dict[str, float], direction: int,
               families: set[str]) -> dict[str, float]:
    output: dict[str, float] = {}
    for name in _MODEL["features"]:
        if name.startswith("family_"):
            output[name] = float(name[7:] in families)
        elif name.startswith("symbol_"):
            output[name] = float(symbol == name[7:])
        elif name.startswith("breadth"):
            output[name] = raw[name] if direction > 0 else 1.0 - raw[name]
        elif name in _NONDIRECTIONAL:
            output[name] = raw[name]
        elif name == "breakout_up":
            output[name] = raw["breakout_up"] if direction > 0 else raw["breakout_down"]
        elif name == "breakout_down":
            output[name] = raw["breakout_down"] if direction > 0 else raw["breakout_up"]
        else:
            output[name] = raw[name] * direction
    return {name: _float32(value) for name, value in output.items()}


def _route_allowed(symbol: str, direction: int, families: set[str]) -> bool:
    """Frozen stable route: every short; pullback or AVAX longs only."""
    return bool(direction < 0 or "pullback" in families or symbol == "AVAXUSDT")


def shadow_setups(candles_by_symbol: dict[str, dict],
                  market_prices: dict[str, float]) -> dict[str, dict]:
    """Return at most one frozen hourly shadow setup per supported symbol."""
    if not SUPPORTED_SYMBOLS.issubset(candles_by_symbol):
        return {}
    latest = {int(candles_by_symbol[symbol]["time"][-1])
              for symbol in SUPPORTED_SYMBOLS}
    if len(latest) != 1:
        return {}
    signal_time = latest.pop()
    raw = {symbol: _symbol_features(candles_by_symbol[symbol])
           for symbol in SUPPORTED_SYMBOLS}
    if any(value is None for value in raw.values()):
        return {}
    rows: dict[str, dict[str, float]] = raw  # type: ignore[assignment]
    for window in (6, 24, 72):
        values = [rows[symbol][f"ret{window}"] for symbol in SUPPORTED_SYMBOLS]
        mean = statistics.fmean(values)
        std = statistics.stdev(values)
        if std <= 0:
            return {}
        breadth = sum(value > 0 for value in values) / len(values)
        for symbol in SUPPORTED_SYMBOLS:
            rows[symbol][f"cross_z{window}"] = (
                rows[symbol][f"ret{window}"] - mean) / std
            rows[symbol][f"breadth{window}"] = breadth
    btc = rows["BTCUSDT"]
    for row in rows.values():
        for window in (6, 24, 168):
            row[f"btc_ret{window}"] = btc[f"ret{window}"]

    output: dict[str, dict] = {}
    for symbol in sorted(SUPPORTED_SYMBOLS):
        market = float(market_prices.get(symbol) or 0)
        if market <= 0:
            continue
        options = []
        for direction, families in _family_directions(rows[symbol]).items():
            if not families or not _route_allowed(symbol, direction, families):
                continue
            feature_row = _model_row(symbol, rows[symbol], direction, families)
            quality = score_named(_MODEL, feature_row)
            if quality >= QUALITY_THRESHOLD:
                options.append((quality, direction, families, feature_row))
        if not options:
            continue
        quality, direction, families, feature_row = max(
            options, key=lambda item: item[0])
        # The Oct 22 material-exit transfer passed only for this literal bracket.
        # A target cascade would be a different, unvalidated exit contract.
        selected_target = TARGET_PCT
        hold_bars = HOLD_BARS
        label = "LONG" if direction > 0 else "SHORT"
        output[symbol] = {
            "direction": label, "module": "shadow_okx_1h_hgb",
            "btc_regime": "hourly_hgb", "entry": market,
            "atr": market * STOP_PCT / 200.0, "target_r": selected_target,
            "first_target_move_pct": selected_target,
            "sl": market * (1.0 - direction * STOP_PCT / 100.0),
            "tp": market * (1.0 + direction * selected_target / 100.0),
            "utc_session": f"hour_{(signal_time // 3600) % 24:02d}",
            "eff_ratio": rows[symbol]["eff72"],
            "signal_bar_ts": signal_time, "model_quality": quality,
            "target_model_scores": {},
            "selected_target_pct": selected_target, "hold_bars": hold_bars,
            "families": sorted(families), "entry_source": "MARKET",
            "route_policy": _POLICY["route_policy"],
            "_shadow_experiment": True, "_hourly_hgb_shadow": True,
        }
    return output
