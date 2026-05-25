import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import RSI_OVERSOLD, RSI_OVERBOUGHT, VOLUME_SPIKE_MULTIPLIER, MIN_SIGNALS_TO_PASS
from src.indicators import get_indicators


def analyze_coin(df, symbol: str) -> dict | None:
    """
    Run technical analysis on OHLCV data.
    Returns a setup dict if a strong signal exists, otherwise None.

    Scoring system (4 possible signals):
      1. EMA alignment / fresh cross  (+1 or +2 for fresh cross)
      2. RSI extreme zone              (+1)
      3. Volume spike (confirms direction) (+1)
      4. Price breakout / breakdown    (+1)

    A coin passes only if bullish_score >= MIN_SIGNALS and bullish > bearish
    (or same logic for bearish).
    """
    if len(df) < 30:
        return None

    ind = get_indicators(df)
    bullish = 0
    bearish = 0
    details = []

    # --- 1. EMA trend / cross ---
    ema_bullish_cross = ind["ema9_prev"] <= ind["ema21_prev"] and ind["ema9"] > ind["ema21"]
    ema_bearish_cross = ind["ema9_prev"] >= ind["ema21_prev"] and ind["ema9"] < ind["ema21"]

    if ema_bullish_cross:
        bullish += 2
        details.append("EMA bullish cross (fresh)")
    elif ind["ema9"] > ind["ema21"]:
        bullish += 1
        details.append("EMA bullish trend")
    elif ema_bearish_cross:
        bearish += 2
        details.append("EMA bearish cross (fresh)")
    elif ind["ema9"] < ind["ema21"]:
        bearish += 1
        details.append("EMA bearish trend")

    # --- 2. RSI extreme ---
    rsi = ind["rsi"]
    if rsi < RSI_OVERSOLD:
        bullish += 1
        details.append(f"RSI oversold ({rsi:.1f})")
    elif rsi > RSI_OVERBOUGHT:
        bearish += 1
        details.append(f"RSI overbought ({rsi:.1f})")

    # --- 3. Volume spike (adds weight to dominant direction) ---
    vol_ratio = ind["volume_ratio"]
    if vol_ratio >= VOLUME_SPIKE_MULTIPLIER:
        details.append(f"Volume spike ({vol_ratio:.1f}x avg)")
        if bullish > bearish:
            bullish += 1
        elif bearish > bullish:
            bearish += 1
        # tied: no direction confirmed, skip

    # --- 4. Price breakout / breakdown ---
    price = ind["current_close"]
    if price > ind["recent_high"]:
        bullish += 1
        details.append("Breakout above 20-candle resistance")
    elif price < ind["recent_low"]:
        bearish += 1
        details.append("Breakdown below 20-candle support")

    # --- Decision ---
    direction = None
    if bullish >= MIN_SIGNALS_TO_PASS and bullish > bearish:
        direction = "LONG"
    elif bearish >= MIN_SIGNALS_TO_PASS and bearish > bullish:
        direction = "SHORT"

    if direction is None:
        return None

    return {
        "symbol":        symbol,
        "direction":     direction,
        "rsi":           round(rsi, 2),
        "ema9":          round(ind["ema9"], 6),
        "ema21":         round(ind["ema21"], 6),
        "volume_ratio":  round(vol_ratio, 2),
        "current_price": round(price, 6),
        "recent_high":   round(ind["recent_high"], 6),
        "recent_low":    round(ind["recent_low"], 6),
        "signals":       details,
        "bullish_score": bullish,
        "bearish_score": bearish,
    }
