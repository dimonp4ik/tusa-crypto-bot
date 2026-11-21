"""Observe one countertrend candle filter without altering signal admission."""

import math


def countertrend_wick_evidence(candles, btc, side):
    if side not in ("LONG", "SHORT"):
        return {}
    try:
        signal_bar = candles["time"][-1]
        if (btc["time"][-1] != signal_bar or len(btc["time"]) < 17
                or btc["time"][-17] != signal_bar - 16 * 900
                or not candles.get("confirmed", [True])[-1]
                or not btc.get("confirmed", [True])[-1]):
            return {}
        low, high, close = (float(candles[field][-1]) for field in ("low", "high", "close"))
        latest, previous = float(btc["close"][-1]), float(btc["close"][-17])
        if (not all(math.isfinite(price) and price > 0 for price in (low, high, close, latest, previous))
                or not low <= close <= high):
            return {}
    except (KeyError, IndexError, TypeError, ValueError):
        return {}

    sign = 1 if side == "LONG" else -1
    aligned = sign * (latest / previous - 1)
    if not math.isfinite(aligned):
        return {}
    fraction = (close - low) / (high - low) if high > low else .5
    directional = fraction if sign > 0 else 1 - fraction
    return dict(
        wick_shadow_profile="countertrend_wick_v1",
        wick_btc_aligned_4h=aligned,
        wick_directional_close=directional,
        wick_would_skip=int(aligned <= -.005 and directional < .15),
    )
