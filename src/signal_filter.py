import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    RSI_OVERSOLD, RSI_OVERBOUGHT, VOLUME_SPIKE_MULTIPLIER, MIN_SIGNALS_TO_PASS,
    SMC_MIN_CONFIRMATIONS, SMC_BOS_MIN_VOLUME, BTC_BLOCK_THRESHOLD_PCT,
    SMC_RSI_LONG_MAX, SMC_RSI_SHORT_MIN, MTF_MIN_SCORE,
    REQUIRE_ENTRY_ZONE, ENTRY_ZONE_SL_BUFFER_ATR,
)
from src.indicators import get_indicators, get_smc_indicators


# ── Entry zone helpers ────────────────────────────────────────────────────────

def _zone_payload(zone, source: str, current: float):
    """Normalize a (low, high) zone tuple into entry dict form."""
    if not zone:
        return None
    low, high = sorted([float(zone[0]), float(zone[1])])
    if low <= 0 or high <= 0 or high <= low:
        return None
    return {
        "entry_low":    round(low, 8),
        "entry_high":   round(high, 8),
        "entry_price":  round((low + high) / 2, 8),
        "entry_source": source,
        "market_price": round(current, 8),
    }


def _select_entry_zone(ind: dict, direction: str):
    """Prefer Order Block zone, then FVG zone as entry area."""
    current = ind["current_close"]
    if direction == "LONG":
        return (
            _zone_payload(ind.get("bull_ob_zone"), "OB", current)
            or _zone_payload(ind.get("bullish_fvg_zone"), "FVG", current)
        )
    return (
        _zone_payload(ind.get("bear_ob_zone"), "OB", current)
        or _zone_payload(ind.get("bearish_fvg_zone"), "FVG", current)
    )


# ── MTF Score ─────────────────────────────────────────────────────────────────

def _calc_mtf_score(ind: dict, bos: str, direction: str, confirmations: list,
                    btc_change_pct: float, entry_zone) -> tuple:
    """
    Deterministic quality score (max ~15) before Claude.
    Weak setups filtered here save Claude tokens.
    """
    score = 0
    tags = []

    score += 2; tags.append("BOS+2")

    if ind.get("trend_1h") == bos:
        score += 2; tags.append("1h+2")
    elif ind.get("trend_1h") == "neutral":
        score += 1; tags.append("1hN+1")

    if ind.get("trend_4h") == bos:
        score += 2; tags.append("4h+2")
    elif ind.get("trend_4h") == "neutral":
        score += 1; tags.append("4hN+1")

    vol = float(ind.get("volume_ratio", 0.0))
    if vol >= max(SMC_BOS_MIN_VOLUME * 1.35, 2.0):
        score += 2; tags.append("Vol+2")
    elif vol >= SMC_BOS_MIN_VOLUME:
        score += 1; tags.append("Vol+1")

    rsi = float(ind.get("rsi", 50.0))
    if direction == "LONG" and 38 <= rsi <= 68:
        score += 1; tags.append("RSI+1")
    elif direction == "SHORT" and 32 <= rsi <= 62:
        score += 1; tags.append("RSI+1")

    if direction == "LONG" and btc_change_pct >= 0:
        score += 2; tags.append("BTC+2")
    elif direction == "SHORT" and btc_change_pct <= 0:
        score += 2; tags.append("BTC+2")
    else:
        score += 1; tags.append("BTCok+1")

    for name in confirmations:
        if name in ("FVG", "OB", "LiqSweep"):
            score += 1; tags.append(f"{name}+1")

    if entry_zone:
        score += 1; tags.append(f"Zone:{entry_zone['entry_source']}+1")

    return score, tags


# ── SMC filter ────────────────────────────────────────────────────────────────

def analyze_coin_smc(candles_15m: dict, candles_1h: dict, symbol: str,
                     candles_4h: dict = None, btc_change_pct: float = 0.0) -> dict | None:
    """
    SMC-based setup detector with MTF score and zone entry.

    Filters (all must pass before Claude):
      1. BOS on closed candles
      2. 1h/4h trend not against setup
      3. Volume >= SMC_BOS_MIN_VOLUME on BOS context
      4. BTC not strongly against direction
      5. RSI not exhausted (SMC_RSI_LONG_MAX / SMC_RSI_SHORT_MIN)
      6. >= SMC_MIN_CONFIRMATIONS from FVG/OB/Sweep/Div/Wick/Stoch
      7. Active FVG/OB entry zone when REQUIRE_ENTRY_ZONE=True
      8. MTF score >= MTF_MIN_SCORE
    """
    if len(candles_15m.get("close", [])) < 30:
        return None

    ind = get_smc_indicators(candles_15m, candles_1h, candles_4h)

    bos      = ind["bos"]
    trend_1h = ind["trend_1h"]
    trend_4h = ind["trend_4h"]

    # 1. Must have BOS
    if not bos:
        return None

    # 2. Trend must match (neutral OK)
    if trend_1h != "neutral" and trend_1h != bos:
        return None
    if trend_4h != "neutral" and trend_4h != bos:
        return None

    # 3. Volume on BOS context
    if ind["volume_ratio"] < SMC_BOS_MIN_VOLUME:
        return None

    # 4. BTC correlation
    if bos == "bullish" and btc_change_pct < -BTC_BLOCK_THRESHOLD_PCT:
        return None
    if bos == "bearish" and btc_change_pct > +BTC_BLOCK_THRESHOLD_PCT:
        return None

    # 5. RSI not exhausted
    rsi = ind["rsi"]
    if bos == "bullish" and rsi > SMC_RSI_LONG_MAX:
        return None
    if bos == "bearish" and rsi < SMC_RSI_SHORT_MIN:
        return None

    # 6. Build confirmations
    wicks  = ind.get("wicks", {})
    div    = ind.get("divergence")
    sk, sd = ind.get("stoch_k", 50), ind.get("stoch_d", 50)

    if bos == "bullish":
        confirmations = []
        if ind["bullish_fvg"]:                              confirmations.append("FVG")
        if ind["bull_ob"]:                                  confirmations.append("OB")
        if ind["bull_sweep"]:                               confirmations.append("LiqSweep")
        if div == "bullish":                                confirmations.append("RSI_Div")
        if wicks.get("bull_pressure") or wicks.get("rejection") == "bullish":
                                                            confirmations.append("BullWick")
        if sk < 25 and sk > sd:                            confirmations.append("StochCross")
        direction = "LONG"
    elif bos == "bearish":
        confirmations = []
        if ind["bearish_fvg"]:                              confirmations.append("FVG")
        if ind["bear_ob"]:                                  confirmations.append("OB")
        if ind["bear_sweep"]:                               confirmations.append("LiqSweep")
        if div == "bearish":                                confirmations.append("RSI_Div")
        if wicks.get("bear_pressure") or wicks.get("rejection") == "bearish":
                                                            confirmations.append("BearWick")
        if sk > 75 and sk < sd:                            confirmations.append("StochCross")
        direction = "SHORT"
    else:
        return None

    if len(confirmations) < SMC_MIN_CONFIRMATIONS:
        return None

    # 7. Entry zone
    entry_zone = _select_entry_zone(ind, direction)
    if REQUIRE_ENTRY_ZONE and not entry_zone:
        return None

    # 8. MTF score
    mtf_score, score_tags = _calc_mtf_score(
        ind, bos, direction, confirmations, btc_change_pct, entry_zone
    )
    if mtf_score < MTF_MIN_SCORE:
        return None

    # Bonus signals for context
    session = ind.get("session", "OFF_HOURS")
    if session in ("LONDON", "NEW_YORK", "OVERLAP"):
        confirmations.append(f"Session:{session}")
    if ind.get("trend_1h_strong"):
        confirmations.append("StrongTrend1h")

    signals = [f"BOS {bos}", f"Vol {ind['volume_ratio']:.1f}x"] + confirmations
    if entry_zone:
        signals.append(f"Zone:{entry_zone['entry_source']}")
    signals.append(f"MTF {mtf_score}")

    # Use zone midpoint as entry price when available
    price_payload = entry_zone or {
        "entry_low":    round(ind["current_close"], 8),
        "entry_high":   round(ind["current_close"], 8),
        "entry_price":  round(ind["current_close"], 8),
        "entry_source": "MARKET",
        "market_price": round(ind["current_close"], 8),
    }

    return {
        "symbol":           symbol,
        "direction":        direction,
        "trend_1h":         trend_1h,
        "trend_4h":         ind["trend_4h"],
        "trend_1h_strong":  ind.get("trend_1h_strong", False),
        "session":          session,
        "bos":              bos,
        "bos_body_strong":  ind.get("bos_body_strong", False),
        "fvg":              ind["bullish_fvg"] if direction == "LONG" else ind["bearish_fvg"],
        "order_block":      ind["bull_ob"]     if direction == "LONG" else ind["bear_ob"],
        "liq_sweep":        ind["bull_sweep"]  if direction == "LONG" else ind["bear_sweep"],
        "rsi":              rsi,
        "stoch_k":          sk,
        "stoch_d":          sd,
        "divergence":       div,
        "wick_rejection":   wicks.get("rejection"),
        "atr":              ind["atr"],
        "volume_ratio":     ind["volume_ratio"],
        "current_price":    price_payload["entry_price"],
        "market_price":     price_payload["market_price"],
        "entry_low":        price_payload["entry_low"],
        "entry_high":       price_payload["entry_high"],
        "entry_source":     price_payload["entry_source"],
        "recent_high":      round(ind["recent_high"], 8),
        "recent_low":       round(ind["recent_low"], 8),
        "btc_change":       round(btc_change_pct, 2),
        "signals":          signals,
        "mtf_score":        mtf_score,
        "score_tags":       score_tags,
        "bullish_score":    mtf_score if direction == "LONG"  else 0,
        "bearish_score":    mtf_score if direction == "SHORT" else 0,
    }


# ── Legacy EMA/RSI filter (kept as fallback) ──────────────────────────────────

def analyze_coin(df, symbol: str) -> dict | None:
    """Original EMA+RSI filter. Not used in main scan. Kept for reference."""
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
