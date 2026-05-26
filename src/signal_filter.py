import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    RSI_OVERSOLD, RSI_OVERBOUGHT, VOLUME_SPIKE_MULTIPLIER, MIN_SIGNALS_TO_PASS,
    SMC_MIN_CONFIRMATIONS, SMC_BOS_MIN_VOLUME, BTC_BLOCK_THRESHOLD_PCT,
)
from src.indicators import get_indicators, get_smc_indicators


# ── SMC filter ────────────────────────────────────────────────────────────────

def analyze_coin_smc(candles_15m: dict, candles_1h: dict, symbol: str,
                     candles_4h: dict = None, btc_change_pct: float = 0.0) -> dict | None:
    """
    SMC-based setup detector with multi-timeframe confirmation.

    Entry conditions (ALL must be true):
      1. BOS on 15m
      2. 1h trend matches BOS direction
      3. 4h trend matches BOS direction (or neutral)
      4. BOS candle had elevated volume (>= SMC_BOS_MIN_VOLUME)
      5. At least SMC_MIN_CONFIRMATIONS (default 2) from: FVG, OB, LiqSweep

    Returns setup dict or None.
    """
    if len(candles_15m.get("close", [])) < 30:
        return None

    ind = get_smc_indicators(candles_15m, candles_1h, candles_4h)

    bos      = ind["bos"]        # 'bullish' | 'bearish' | None
    trend_1h = ind["trend_1h"]
    trend_4h = ind["trend_4h"]

    # 1. Must have BOS
    if not bos:
        return None

    # 2. 1h trend must match (neutral is OK)
    if trend_1h != "neutral" and trend_1h != bos:
        return None

    # 3. 4h trend must match (neutral is OK)
    if trend_4h != "neutral" and trend_4h != bos:
        return None

    # 4. Volume on BOS candle
    if ind["volume_ratio"] < SMC_BOS_MIN_VOLUME:
        return None

    # 4b. BTC correlation filter — skip if BTC moving strongly against direction
    if bos == "bullish"  and btc_change_pct < -BTC_BLOCK_THRESHOLD_PCT:
        return None
    if bos == "bearish"  and btc_change_pct > +BTC_BLOCK_THRESHOLD_PCT:
        return None

    # 5. Build confirmation list — need >= SMC_MIN_CONFIRMATIONS
    if bos == "bullish":
        confirmations = []
        if ind["bullish_fvg"]:  confirmations.append("FVG")
        if ind["bull_ob"]:      confirmations.append("OB")
        if ind["bull_sweep"]:   confirmations.append("LiqSweep")
        direction = "LONG"
    elif bos == "bearish":
        confirmations = []
        if ind["bearish_fvg"]:  confirmations.append("FVG")
        if ind["bear_ob"]:      confirmations.append("OB")
        if ind["bear_sweep"]:   confirmations.append("LiqSweep")
        direction = "SHORT"
    else:
        return None

    if len(confirmations) < SMC_MIN_CONFIRMATIONS:
        return None

    signals = [f"BOS {bos}", f"Vol {ind['volume_ratio']:.1f}x"] + confirmations
    score   = len(signals)

    return {
        "symbol":        symbol,
        "direction":     direction,
        "trend_1h":      trend_1h,
        "trend_4h":      trend_4h,
        "bos":           bos,
        "fvg":           ind["bullish_fvg"] if direction == "LONG" else ind["bearish_fvg"],
        "order_block":   ind["bull_ob"]     if direction == "LONG" else ind["bear_ob"],
        "liq_sweep":     ind["bull_sweep"]  if direction == "LONG" else ind["bear_sweep"],
        "rsi":           ind["rsi"],
        "atr":           ind["atr"],
        "volume_ratio":  ind["volume_ratio"],
        "current_price": round(ind["current_close"], 8),
        "recent_high":   round(ind["recent_high"], 8),
        "recent_low":    round(ind["recent_low"], 8),
        "btc_change":    round(btc_change_pct, 2),
        "signals":       signals,
        "bullish_score": score if direction == "LONG"  else 0,
        "bearish_score": score if direction == "SHORT" else 0,
    }


# ── Legacy EMA/RSI filter (kept as fallback) ──────────────────────────────────

def analyze_coin(df, symbol: str) -> dict | None:
    """
    Original EMA+RSI filter. Not used in main scan anymore.
    Kept for reference / fallback testing.
    """
    if len(df) < 30:
        return None

    ind     = get_indicators(df)
    bullish = 0
    bearish = 0
    details = []

    ema_bullish_cross = ind["ema9_prev"] <= ind["ema21_prev"] and ind["ema9"] > ind["ema21"]
    ema_bearish_cross = ind["ema9_prev"] >= ind["ema21_prev"] and ind["ema9"] < ind["ema21"]

    if ema_bullish_cross:
        bullish += 2; details.append("EMA bullish cross (fresh)")
    elif ind["ema9"] > ind["ema21"]:
        bullish += 1; details.append("EMA bullish trend")
    elif ema_bearish_cross:
        bearish += 2; details.append("EMA bearish cross (fresh)")
    elif ind["ema9"] < ind["ema21"]:
        bearish += 1; details.append("EMA bearish trend")

    rsi = ind["rsi"]
    if rsi < RSI_OVERSOLD:
        bullish += 1; details.append(f"RSI oversold ({rsi:.1f})")
    elif rsi > RSI_OVERBOUGHT:
        bearish += 1; details.append(f"RSI overbought ({rsi:.1f})")

    vol_ratio = ind["volume_ratio"]
    if vol_ratio >= VOLUME_SPIKE_MULTIPLIER:
        details.append(f"Volume spike ({vol_ratio:.1f}x avg)")
        if bullish > bearish:   bullish += 1
        elif bearish > bullish: bearish += 1

    price = ind["current_close"]
    if price > ind["recent_high"]:
        bullish += 1; details.append("Breakout above 20-candle resistance")
    elif price < ind["recent_low"]:
        bearish += 1; details.append("Breakdown below 20-candle support")

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
