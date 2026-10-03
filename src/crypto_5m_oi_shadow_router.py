"""Frozen shadow-only 5-minute X-Perp route confirmed by lagged OI.

This module proposes market brackets only.  It never places an order and it
fails closed when candles, alignment, or the one-hour-lagged OI series are
missing.  The route must remain shadow-only until its forward activation gate
is met.
"""
from __future__ import annotations

from bisect import bisect_right
from datetime import datetime, timezone
import math


MODULE_NAME = "shadow_5m_oi_trend_pullback"
OI_168H_MIN = -0.15010500977847596
BTC_RET12_DIR_ATR_MAX = -0.440283
BTC_MA20_96_DIR_ATR_MIN = 4.93787
STOP_ATR = 2.0
TARGET_R = 0.25


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def _atr(candles: dict, index: int, length: int = 14) -> float | None:
    if index < length or index >= len(candles.get("close", [])):
        return None
    close = candles["close"]
    values = []
    for current in range(index - length + 1, index + 1):
        high = float(candles["high"][current])
        low = float(candles["low"][current])
        previous = float(close[current - 1])
        values.append(max(high - low, abs(high - previous), abs(low - previous)))
    result = _mean(values)
    return result if math.isfinite(result) and result > 0 else None


def _trend_pullback_direction(candles: dict, index: int) -> int:
    if index < 95:
        return 0
    close = [float(value) for value in candles["close"]]
    atr = _atr(candles, index)
    if atr is None:
        return 0
    ma20 = _mean(close[index - 19:index + 1])
    ma96 = _mean(close[index - 95:index + 1])
    current, previous = close[index], close[index - 1]
    if ma20 > ma96 and current < ma20 - .5 * atr \
            and current > ma96 and current > previous:
        return 1
    if ma20 < ma96 and current > ma20 + .5 * atr \
            and current < ma96 and current < previous:
        return -1
    return 0


def _aligned_index(candles: dict, timestamp: int) -> int | None:
    times = [int(value) for value in candles.get("time", [])]
    index = bisect_right(times, int(timestamp)) - 1
    return index if index >= 0 and times[index] == int(timestamp) else None


def candidate_context(candles: dict, btc_candles: dict) -> dict | None:
    """Return a frozen price-context candidate before the costly OI request."""
    required = ("time", "open", "high", "low", "close", "volume")
    if any(len(candles.get(key, [])) < 97 for key in required):
        return None
    index = len(candles["close"]) - 1
    direction = _trend_pullback_direction(candles, index)
    if direction == 0 or direction == _trend_pullback_direction(candles, index - 1):
        return None
    signal_time = int(candles["time"][index])
    btc_index = _aligned_index(btc_candles, signal_time)
    if btc_index is None or btc_index < 95:
        return None
    btc_atr = _atr(btc_candles, btc_index)
    coin_atr = _atr(candles, index)
    if btc_atr is None or coin_atr is None:
        return None
    btc_close = [float(value) for value in btc_candles["close"]]
    btc_ma20 = _mean(btc_close[btc_index - 19:btc_index + 1])
    btc_ma96 = _mean(btc_close[btc_index - 95:btc_index + 1])
    btc_trend = direction * (btc_ma20 - btc_ma96) / btc_atr
    btc_return = direction * (btc_close[btc_index] - btc_close[btc_index - 12]) / btc_atr
    if (btc_return > BTC_RET12_DIR_ATR_MAX
            or btc_trend <= BTC_MA20_96_DIR_ATR_MIN):
        return None
    recent = [float(value) for value in candles["close"][index - 19:index + 1]]
    movement = sum(abs(recent[pos] - recent[pos - 1]) for pos in range(1, 20))
    efficiency = abs(recent[-1] - recent[0]) / movement if movement else 0.0
    return {
        "direction": "LONG" if direction > 0 else "SHORT",
        "direction_value": direction,
        "atr": coin_atr,
        "eff_ratio": efficiency,
        "signal_bar_ts": signal_time,
        "btc_ret12_dir_atr": btc_return,
        "btc_ma20_96_dir_atr": btc_trend,
    }


def _oi_change_168h(oi_history: dict, entry_time: float) -> float | None:
    times = [int(value) for value in oi_history.get("time", [])]
    values = [float(value) for value in oi_history.get("oi", [])]
    if len(times) != len(values) or len(times) < 169:
        return None
    # Match the audit: the newest usable snapshot is at least one hour old.
    index = bisect_right(times, int(entry_time) - 3600) - 1
    if index < 168 or values[index - 168] <= 0:
        return None
    change = values[index] / values[index - 168] - 1
    return change if math.isfinite(change) else None


def shadow_setup(candles: dict, btc_candles: dict, oi_history: dict,
                 *, entry_time: float, market_price: float,
                 symbol: str) -> dict | None:
    context = candidate_context(candles, btc_candles)
    if context is None or not symbol:
        return None
    oi_change = _oi_change_168h(oi_history, entry_time)
    if oi_change is None or oi_change < OI_168H_MIN:
        return None
    entry = float(market_price)
    if not math.isfinite(entry) or entry <= 0:
        return None
    return {
        "symbol": symbol,
        "module": MODULE_NAME,
        "family": "trend_pullback",
        "direction": context["direction"],
        "btc_regime": "oi_confirmed_trend",
        "utc_session": f"{datetime.fromtimestamp(entry_time, timezone.utc).hour:02d}",
        "entry": entry,
        "atr": context["atr"],
        "target_r": TARGET_R,
        "eff_ratio": context["eff_ratio"],
        "entry_time": float(entry_time),
        "signal_bar_ts": context["signal_bar_ts"],
        "oi_change_168h": oi_change,
        "btc_ret12_dir_atr": context["btc_ret12_dir_atr"],
        "btc_ma20_96_dir_atr": context["btc_ma20_96_dir_atr"],
        "_shadow_experiment": True,
        "_five_minute_oi_shadow": True,
    }
