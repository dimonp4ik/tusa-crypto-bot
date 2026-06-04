import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    RSI_OVERSOLD, RSI_OVERBOUGHT, VOLUME_SPIKE_MULTIPLIER, MIN_SIGNALS_TO_PASS,
    SMC_MIN_CONFIRMATIONS, SMC_BOS_MIN_VOLUME, BTC_BLOCK_THRESHOLD_PCT,
    SMC_RSI_LONG_MAX, SMC_RSI_SHORT_MIN, MTF_MIN_SCORE,
    REQUIRE_ENTRY_ZONE, ENTRY_ZONE_SL_BUFFER_ATR,
    REQUIRE_HTF_TREND, REQUIRE_RETEST, RETEST_MAX_DIST_PCT,
    VOL_REGIME_FILTER, VOL_MIN_ATR_PCT, VOL_MIN_RATIO, VOL_MAX_RATIO,
    REQUIRE_STRONG_BOS, STRONG_BOS_VOL_MULT,
    REQUIRE_STRONG_CONFIRM,
    EFF_RATIO_FILTER, EFF_RATIO_MIN,
    REQUIRE_STRICT_HTF,
)
from src.indicators import get_indicators, get_smc_indicators


# ── Entry zone helpers ────────────────────────────────────────────────────────

def _zones_overlap(z1, z2, buffer_pct: float = 0.005) -> bool:
    """True when two (low, high) price zones overlap or are within buffer_pct of each other."""
    if not z1 or not z2:
        return False
    l1, h1 = float(z1[0]), float(z1[1])
    l2, h2 = float(z2[0]), float(z2[1])
    h1_b = h1 * (1 + buffer_pct)
    l1_b = l1 * (1 - buffer_pct)
    return l2 <= h1_b and h2 >= l1_b


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


_FVG_MAX_FILL = 0.80   # skip FVG if price already through > 80% of the zone


def _fvg_fresh(zone, current: float, direction: str) -> bool:
    """Return True when price has not yet gone through > 80% of the FVG zone.

    LONG bullish FVG (support below): price enters from the TOP (high) moving down.
        fill=0 → price just touched the top (fresh ideal entry)
        fill=1 → price reached the bottom (zone exhausted, likely breaking)

    SHORT bearish FVG (resistance above): price enters from the BOTTOM (low) moving up.
        fill=0 → price just touched the bottom (fresh ideal entry)
        fill=1 → price reached the top (zone exhausted, likely breaking through)
    """
    if not zone:
        return False
    low, high = float(zone[0]), float(zone[1])
    rng = high - low
    if rng <= 0:
        return False
    if direction == "LONG":
        fill = (high - current) / rng   # 0 = just entered from top (fresh), 1 = at bottom
    else:
        fill = (current - low) / rng    # 0 = just entered from bottom (fresh), 1 = at top
    return fill <= _FVG_MAX_FILL


def _select_entry_zone(ind: dict, direction: str):
    """Prefer OB zone, then FVG zone. Skip FVG if > 60% already filled."""
    current = ind["current_close"]
    if direction == "LONG":
        ob_z  = _zone_payload(ind.get("bull_ob_zone"), "OB", current)
        fvg_z = ind.get("bullish_fvg_zone")
        fvg_p = _zone_payload(fvg_z, "FVG", current) if _fvg_fresh(fvg_z, current, "LONG") else None
        return ob_z or fvg_p
    ob_z  = _zone_payload(ind.get("bear_ob_zone"), "OB", current)
    fvg_z = ind.get("bearish_fvg_zone")
    fvg_p = _zone_payload(fvg_z, "FVG", current) if _fvg_fresh(fvg_z, current, "SHORT") else None
    return ob_z or fvg_p


def _ob_fvg_overlap(ind: dict, direction: str) -> bool:
    """True when an Order Block and FVG zone overlap (double confluence, no sweep req)."""
    if direction == "LONG":
        ob_z, fvg_z = ind.get("bull_ob_zone"), ind.get("bullish_fvg_zone")
    else:
        ob_z, fvg_z = ind.get("bear_ob_zone"), ind.get("bearish_fvg_zone")
    if not ob_z or not fvg_z:
        return False
    return _zones_overlap(ob_z, fvg_z)


def _premium_setup(ind: dict, direction: str) -> bool:
    """Institutional TRIPLE confluence: OB + FVG zones overlap AND liquidity sweep.

    Research consensus: an OB+FVG overlap zone is the single highest-probability
    ICT setup (~65% WR vs ~52% for a lone OB). Adding a liquidity sweep (stop-hunt
    before the move) confirms smart-money intent. These are rare but premium.
    """
    if not _ob_fvg_overlap(ind, direction):
        return False
    sweep = ind.get("bull_sweep") if direction == "LONG" else ind.get("bear_sweep")
    return bool(sweep)


# ── MTF Score ─────────────────────────────────────────────────────────────────

def _calc_mtf_score(ind: dict, bos: str, direction: str, confirmations: list,
                    btc_change_pct: float, entry_zone, premium: bool = False) -> tuple:
    """
    Deterministic quality score (max ~20) before Claude.
    Weak setups filtered here save Claude tokens.
    """
    score = 0
    tags = []

    score += 2; tags.append("BOS+2")

    # Clean break body (not a thin-wick poke) — research: false-break wicks → SL.
    if ind.get("bos_body_strong"):
        score += 1; tags.append("BodyStrong+1")

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

    # Confirmations — RSI_Div, Wicks, StochCross now score too (previously missed)
    _SCORED = ("FVG", "OB", "LiqSweep", "ChoCH", "MACD_Div", "Engulfing",
               "Discount", "Premium", "RSI_Div", "BullWick", "BearWick", "StochCross")
    for name in confirmations:
        if name in _SCORED:
            score += 1; tags.append(f"{name}+1")

    if entry_zone:
        score += 1; tags.append(f"Zone:{entry_zone['entry_source']}+1")

    # Session: informational only — backtest showed +2/-1 gating cuts 80% of
    # signals without quality improvement (WR 23% → 13%, -38R vs +13R).
    # Session label still passed in tags for the signal text display.
    session = ind.get("session", "OFF_HOURS")
    tags.append(f"Sess:{session}")

    # Strong HTF trend alignment (EMA stack confirmed)
    if ind.get("trend_1h_strong") and ind.get("trend_1h") == bos:
        score += 1; tags.append("Strong1h+1")
    if ind.get("trend_4h_strong") and ind.get("trend_4h") == bos:
        score += 1; tags.append("Strong4h+1")

    # Nested OB: 1h OB overlaps 15m entry zone → double confluence
    if entry_zone:
        ob_1h_z = ind.get("bull_ob_1h_zone") if direction == "LONG" else ind.get("bear_ob_1h_zone")
        if ob_1h_z and _zones_overlap(ob_1h_z, (entry_zone["entry_low"], entry_zone["entry_high"])):
            score += 2; tags.append("NestedOB_1h+2")

    # Premium triple confluence (OB+FVG overlap + sweep) — highest-WR ICT setup.
    if premium:
        score += 3; tags.append("💎Premium+3")

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

    # 2b. Regime filter — reject chop: no established HTF trend (both neutral)
    if REQUIRE_HTF_TREND and trend_1h == "neutral" and trend_4h == "neutral":
        return None

    # 2b-A. Efficiency-Ratio chop gate — false BOS in ranges → SL clusters
    if EFF_RATIO_FILTER and ind.get("eff_ratio", 1.0) < EFF_RATIO_MIN:
        return None

    # 2b-B. Strict HTF alignment — both 1h AND 4h must back the signal
    if REQUIRE_STRICT_HTF and (trend_1h != bos or trend_4h != bos):
        return None

    # 2c. Volatility regime — skip dead markets (→ EXPIRED) and spikes (→ SL)
    if VOL_REGIME_FILTER:
        atr_pct = ind.get("vol_atr_pct", 0.0)
        v_ratio = ind.get("vol_ratio_regime", 1.0)
        if atr_pct < VOL_MIN_ATR_PCT:
            return None
        if v_ratio < VOL_MIN_RATIO or v_ratio > VOL_MAX_RATIO:
            return None

    # 3. Volume on BOS context
    if ind["volume_ratio"] < SMC_BOS_MIN_VOLUME:
        return None

    # 3b. Strong BOS — real break needs decisive body OR volume surge, not a
    #     thin-wick poke (classic false breakout → SL).
    if REQUIRE_STRONG_BOS:
        strong_body = ind.get("bos_body_strong", False)
        vol_surge   = ind["volume_ratio"] >= SMC_BOS_MIN_VOLUME * STRONG_BOS_VOL_MULT
        if not (strong_body or vol_surge):
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
        if ind["bullish_fvg"]:                               confirmations.append("FVG")
        if ind["bull_ob"]:                                   confirmations.append("OB")
        if ind["bull_sweep"]:                                confirmations.append("LiqSweep")
        if div == "bullish":                                 confirmations.append("RSI_Div")
        if ind.get("macd_divergence") == "bullish":          confirmations.append("MACD_Div")
        if ind.get("choch") == "bullish":                    confirmations.append("ChoCH")
        if ind.get("engulfing") == "bullish":                confirmations.append("Engulfing")
        if ind.get("in_discount"):                           confirmations.append("Discount")
        if wicks.get("bull_pressure") or wicks.get("rejection") == "bullish":
                                                             confirmations.append("BullWick")
        if sk < 25 and sk > sd:                             confirmations.append("StochCross")
        direction = "LONG"
    elif bos == "bearish":
        confirmations = []
        if ind["bearish_fvg"]:                               confirmations.append("FVG")
        if ind["bear_ob"]:                                   confirmations.append("OB")
        if ind["bear_sweep"]:                                confirmations.append("LiqSweep")
        if div == "bearish":                                 confirmations.append("RSI_Div")
        if ind.get("macd_divergence") == "bearish":          confirmations.append("MACD_Div")
        if ind.get("choch") == "bearish":                    confirmations.append("ChoCH")
        if ind.get("engulfing") == "bearish":                confirmations.append("Engulfing")
        if ind.get("in_premium"):                            confirmations.append("Premium")
        if wicks.get("bear_pressure") or wicks.get("rejection") == "bearish":
                                                             confirmations.append("BearWick")
        if sk > 75 and sk < sd:                             confirmations.append("StochCross")
        direction = "SHORT"
    else:
        return None

    if len(confirmations) < SMC_MIN_CONFIRMATIONS:
        return None

    # 6b. Require >=1 STRUCTURAL confirmation — two weak candle signals
    #     (Engulfing + Wick) alone are noise, not smart-money structure.
    if REQUIRE_STRONG_CONFIRM:
        _STRUCTURAL = {"FVG", "OB", "LiqSweep", "ChoCH"}
        if not any(c in _STRUCTURAL for c in confirmations):
            return None

    # 7. Entry zone
    entry_zone = _select_entry_zone(ind, direction)
    if REQUIRE_ENTRY_ZONE and not entry_zone:
        return None

    # 7b. Retest — price must currently be at/near the zone (true retest, not chase)
    if REQUIRE_RETEST and entry_zone:
        cur    = ind["current_close"]
        z_low  = entry_zone["entry_low"]
        z_high = entry_zone["entry_high"]
        if cur < z_low:
            dist = (z_low - cur) / cur
        elif cur > z_high:
            dist = (cur - z_high) / cur
        else:
            dist = 0.0  # price inside the zone
        if dist > RETEST_MAX_DIST_PCT:
            return None

    # 8. MTF score (premium triple-confluence boosts score)
    premium = _premium_setup(ind, direction)
    ob_fvg_overlap = _ob_fvg_overlap(ind, direction)
    mtf_score, score_tags = _calc_mtf_score(
        ind, bos, direction, confirmations, btc_change_pct, entry_zone, premium
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
    if premium:
        signals.append("💎PREMIUM")
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
        "tp1_level":        ind.get("bull_tp1") if direction == "LONG" else ind.get("bear_tp1"),
        "tp2_level":        ind.get("bull_tp2") if direction == "LONG" else ind.get("bear_tp2"),
        "btc_change":       round(btc_change_pct, 2),
        "signals":          signals,
        "mtf_score":        mtf_score,
        "premium":          premium,
        "ob_fvg_overlap":   ob_fvg_overlap,
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
