"""Independent causal entry filters validated outside the legacy SMC stack."""
from __future__ import annotations

import math


def bearish_btc_regime(candles: dict) -> dict | None:
    """Frozen BTC trend regime, computed from closed 15-minute bars."""
    closes = list(map(float, candles.get("close", [])))
    if len(closes) < 96:
        return None
    recent = closes[-20:]
    fast = sum(recent) / 20
    slow = sum(closes[-96:]) / 96
    path = sum(abs(recent[j] - recent[j - 1]) for j in range(1, 20))
    efficiency = abs(recent[-1] - recent[0]) / path if path else 0.0
    if not all(math.isfinite(value) for value in (fast, slow, efficiency)):
        return None
    return {"fast_below_slow": fast < slow, "eff_ratio": efficiency,
            "accepted": fast < slow and efficiency >= .2}


def crypto_pullback_short(candles: dict, btc_candles: dict,
                          market_price: float | None = None,
                          target_r: float | None = None) -> dict | None:
    """Return the frozen bearish trend-pullback setup for the next market entry.

    Every feature uses closed candles only. The caller supplies the executable
    market price; historical research used the following bar's open.
    """
    btc_regime = bearish_btc_regime(btc_candles)
    if not btc_regime or not btc_regime["accepted"]:
        return None
    if target_r is None:
        target_r = .5 if btc_regime["eff_ratio"] >= .5 else .25
    closes = candles.get("close", [])
    highs = candles.get("high", [])
    lows = candles.get("low", [])
    n = len(closes)
    if n < 100 or len(highs) != n or len(lows) != n or target_r <= 0:
        return None
    last = float(closes[-1])
    mean = sum(map(float, closes[-20:])) / 20
    slow = sum(map(float, closes[-96:])) / 96
    variance = sum((float(value) - mean) ** 2 for value in closes[-20:]) / 20
    std = math.sqrt(variance)
    true_ranges = [max(
        float(highs[j]) - float(lows[j]),
        abs(float(highs[j]) - float(closes[j - 1])),
        abs(float(lows[j]) - float(closes[j - 1])),
    ) for j in range(n - 14, n)]
    atr = sum(true_ranges) / 14
    path = sum(abs(float(closes[j]) - float(closes[j - 1]))
               for j in range(n - 19, n))
    efficiency = abs(last - float(closes[-20])) / path if path else 0.0
    z = (last - mean) / std if std else float("inf")
    if not all(math.isfinite(value) for value in (last, mean, slow, atr, efficiency, z)):
        return None
    if not (atr > 0 and mean < slow and last > mean + 0.5 * atr
            and last < slow and last < float(closes[-2])
            and efficiency >= 0.25 and abs(z) <= 1.0):
        return None
    entry = float(market_price if market_price is not None else last)
    risk = 2 * atr
    if not math.isfinite(entry) or entry <= 0 or risk <= 0 or risk >= entry:
        return None
    return {
        "family": "trend_pullback_short",
        "direction": "SHORT",
        "entry": entry,
        "sl": entry + risk,
        "tp": entry - target_r * risk,
        "target_r": target_r,
        "atr": atr,
        "eff_ratio": efficiency,
        "z": z,
        "btc_eff_ratio": btc_regime["eff_ratio"],
    }
