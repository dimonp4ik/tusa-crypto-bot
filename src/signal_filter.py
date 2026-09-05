import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (EFF_RATIO_MAX, COUNTER_STRUCTURE_SCORE_BONUS, HTF_ALIGNED_SCORE, HTF_NEUTRAL_SCORE, HTF_STRONG_SCORE)
from config import (
    RSI_OVERSOLD, RSI_OVERBOUGHT, VOLUME_SPIKE_MULTIPLIER, MIN_SIGNALS_TO_PASS,
    SMC_MIN_CONFIRMATIONS, SMC_BOS_MIN_VOLUME, BTC_BLOCK_THRESHOLD_PCT,
    SMC_RSI_LONG_MAX, SMC_RSI_SHORT_MIN, MTF_MIN_SCORE, SHADOW_MIN_SCORE,
    REQUIRE_ENTRY_ZONE, ENTRY_ZONE_SL_BUFFER_ATR,
    REQUIRE_HTF_TREND, REQUIRE_RETEST, RETEST_MAX_DIST_PCT,
    HTF_FULL_ALIGN_SKIP,
    VOL_REGIME_FILTER, VOL_MIN_ATR_PCT, VOL_MIN_RATIO, VOL_MAX_RATIO,
    REQUIRE_STRONG_BOS, STRONG_BOS_VOL_MULT,
    REQUIRE_STRONG_CONFIRM,
    MACD_CHOCH_NOISE_FILTER, OVERLAP_BEARISH_1H_GUARD,
    DAILY_TREND_FILTER, DOUBLE_NEUTRAL_LONG_FILTER, DAILY_TREND_SHORT_FILTER,
    EFF_RATIO_FILTER, EFF_RATIO_MIN,
    REQUIRE_STRICT_HTF,
    ADAPTIVE_FILTER_PACKS, ADAPTIVE_MIXED_SCORE_BUMP, ADAPTIVE_CHOP_SCORE_BUMP,
    ADAPTIVE_HOT_SCORE_BUMP, ADAPTIVE_MIXED_EFF_MIN, ADAPTIVE_CHOP_EFF_MIN,
    ADAPTIVE_HOT_EFF_MIN, ADAPTIVE_CHOP_MIN_VOLUME, ADAPTIVE_HOT_MIN_VOLUME,
    ADAPTIVE_HOT_VOL_RATIO, ADAPTIVE_EXTREME_VOL_RATIO, ADAPTIVE_EXTREME_ATR_PCT,
    ADAPTIVE_MIXED_RISK_MULT, ADAPTIVE_CHOP_RISK_MULT, ADAPTIVE_HOT_RISK_MULT,
    ADAPTIVE_BEAR_SQUEEZE_GUARD, ADAPTIVE_BEAR_SKIP_NEW_YORK,
    ADAPTIVE_BEAR_VOL_MIN_RATIO, ADAPTIVE_BEAR_VOL_MAX_RATIO,
    BEAR_TREND_HOT_VOL_GUARD, BEAR_TREND_HOT_VOL_MIN_RATIO, BEAR_TREND_SKIP_SESSIONS,
    DIRECTIONAL_RSI_MIDLINE_FILTER, RSI_LONG_MIN_MIDLINE, RSI_SHORT_MAX_MIDLINE,
    SYMBOL_EDGE_FILTER, LOW_EDGE_SYMBOLS,
    SOURCE_EDGE_FILTER, LOW_EDGE_FVG_SYMBOLS, LOW_EDGE_OB_SYMBOLS,
    DIRECTION_EDGE_FILTER, LOW_EDGE_LONG_SYMBOLS, LOW_EDGE_SHORT_SYMBOLS,
    RELATIVE_STRENGTH_LOOKBACK_HOURS,
    LONG_RELATIVE_WEAKNESS_FILTER, LONG_RELATIVE_WEAKNESS_MAX_PCT,
    BULL_NEUTRAL_LONG_NARROW_ZONE_FILTER, BULL_NEUTRAL_LONG_MAX_ZONE_WIDTH_PCT,
    LONG_NY_COIN_MOMENTUM_FILTER, LONG_NY_MIN_COIN_CHANGE_1H,
    SHORT_FVG_COIN_MOMENTUM_FILTER, SHORT_FVG_MAX_COIN_CHANGE_1H,
    FVG_LONDON_BTC_UP_FILTER, FVG_LONDON_BTC_UP_MIN_PCT,
    QUALITY_RISK_OVERLAY, QUALITY_RISK_MULT, QUALITY_RISK_MAX_MULT,
    QUALITY_RISK_VOL_MIN, QUALITY_RISK_VOL_MAX,
    QUALITY_RISK_RSI_MIN, QUALITY_RISK_RSI_MAX, HIGH_EDGE_RISK_SYMBOLS,
    REL_STRENGTH_RISK_UP, REL_STRENGTH_RISK_UP_MIN_PCT, REL_STRENGTH_RISK_UP_MAX_PCT,
    REL_STRENGTH_RISK_UP_MULT, REL_STRENGTH_RISK_UP_MAX_MULT,
    TREND_PAIR_RISK_UP, TREND_PAIR_RISK_UP_1H, TREND_PAIR_RISK_UP_4H,
    TREND_PAIR_RISK_UP_MULT, TREND_PAIR_RISK_UP_MAX_MULT,
    SNIPER_TAG_ENABLED,
    RISK_MIN_PCT, RISK_MAX_PCT, SL_ATR_BUFFER,
    STABILITY_FILTERS_ENABLED, STABILITY_SKIP_PACKS, STABILITY_SKIP_SESSIONS,
    STABILITY_MIN_EFF_RATIO, STABILITY_MIN_VOLUME_RATIO, STABILITY_MIN_QUALITY_SCORE,
    SKIP_RSI_DIV_SETUPS, SKIP_UTC_HOURS, SKIP_WEEKDAYS,
)
from src.indicators import get_indicators, get_smc_indicators


# ── Rejection funnel (diagnostic) ─────────────────────────────────────────────
# The filter has 20+ places where a setup dies, and until 2026-08-24 only two of
# them were counted. That made "which gate is costing us volume" unanswerable —
# every tuning discussion had to guess. _rej() names each exit so the funnel can
# be printed and each gate judged on what it actually removes.
#
# Costs one dict increment on a path that already returns; production reads
# nothing unless asked.
REJECT_COUNTS: dict = {}


def _regime(ind: dict) -> float:
    """Volatility-regime ratio, defaulting only when it is genuinely absent.

    `ind.get("vol_ratio_regime", 1.0) or 1.0` read a ratio of exactly 0.0 as
    1.0 — "normal volatility". Zero is reachable and it is not normal: it means
    the last three bars were perfectly flat against a median that moved, i.e.
    the market has stopped. That is precisely the reading the dead-market rules
    exist to catch, and `or` handed back its opposite. Written out in four
    places, so it is a function now rather than a fifth copy.
    """
    v = ind.get("vol_ratio_regime")
    try:
        return float(v) if v is not None else 1.0
    except (TypeError, ValueError):
        return 1.0


def _rej(reason: str):
    REJECT_COUNTS[reason] = REJECT_COUNTS.get(reason, 0) + 1
    return None


def reset_reject_counts() -> None:
    REJECT_COUNTS.clear()


def _norm_symbol(symbol: str) -> str:
    return str(symbol or "").upper().replace("-", "").replace("/", "").replace("_", "")


def _daily_trend(candles_1d: dict) -> str:
    """
    Macro trend from daily candles (3-day momentum).
    Returns 'bullish' / 'bearish' / 'neutral'.
    Uses 3-day close change to avoid single-candle noise.
    Threshold ±1% — neutral band absorbs normal daily noise.
    """
    if not candles_1d:
        return "neutral"
    closes = candles_1d.get("close", [])
    if len(closes) < 3:
        return "neutral"
    c_old = closes[-3]
    c_new = closes[-1]
    if not c_old or c_old == 0:
        return "neutral"
    change_pct = (c_new - c_old) / c_old * 100.0
    if change_pct > 1.0:
        return "bullish"
    if change_pct < -1.0:
        return "bearish"
    return "neutral"


_LOW_EDGE_SYMBOLS_NORM       = {_norm_symbol(s) for s in LOW_EDGE_SYMBOLS}
_LOW_EDGE_FVG_SYMBOLS_NORM   = {_norm_symbol(s) for s in LOW_EDGE_FVG_SYMBOLS}
_LOW_EDGE_OB_SYMBOLS_NORM    = {_norm_symbol(s) for s in LOW_EDGE_OB_SYMBOLS}
_LOW_EDGE_LONG_SYMBOLS_NORM  = {_norm_symbol(s) for s in LOW_EDGE_LONG_SYMBOLS}
_LOW_EDGE_SHORT_SYMBOLS_NORM = {_norm_symbol(s) for s in LOW_EDGE_SHORT_SYMBOLS}
_HIGH_EDGE_RISK_SYMBOLS_NORM = {_norm_symbol(s) for s in HIGH_EDGE_RISK_SYMBOLS}


def _change_pct_from_1h(candles_1h: dict, lookback_hours: int = 1) -> float:
    """Coin % change over the last lookback_hours 1h candles."""
    closes  = candles_1h.get("close", []) if candles_1h else []
    lookback = max(1, int(lookback_hours or 1))
    if len(closes) <= lookback:
        return 0.0
    prev = float(closes[-1 - lookback])
    cur  = float(closes[-1])
    if prev <= 0:
        return 0.0
    return (cur - prev) / prev * 100.0


# ⚠ 2026-09-05: the three overlays below, and the kNN fold-in in main.py, all
# write to `risk_mult` — and NOTHING SIZES FROM IT. src/autotrader.py never
# reads the field, and backtest.py records it as a column while sizing from
# _size_mult_for(), which mirrors the autotrader instead. Proof: running the
# book with TREND_PAIR_RISK_UP=0 reproduces the baseline byte for byte in all
# three windows — 1121сд/+435.84R, 869сд/+226.17R, 695сд/+189.70R.
#
# So these knobs describe an intention, not a behaviour. Two consequences:
# a config change here moves nothing, and make_dashboard/monte_carlo/
# significance_check run with --use-risk-mult weight past results by a
# multiplier no position ever carried. Wire them into _size_mult_for on both
# sides before trusting any of them, or delete them.
def _apply_quality_risk_overlay(
    risk_mult: float,
    *,
    symbol: str,
    entry_source: str,
    vol_ratio_regime: float,
    rsi: float,
) -> tuple[float, str]:
    """Boost risk_mult by x1.15 for OB entries / optimal vol-RSI / high-edge symbols."""
    if not QUALITY_RISK_OVERLAY:
        return risk_mult, ""
    quality_context = (
        entry_source == "OB"
        or QUALITY_RISK_VOL_MIN <= vol_ratio_regime <= QUALITY_RISK_VOL_MAX
        or QUALITY_RISK_RSI_MIN <= rsi < QUALITY_RISK_RSI_MAX
        or _norm_symbol(symbol) in _HIGH_EDGE_RISK_SYMBOLS_NORM
    )
    if not quality_context:
        return risk_mult, ""
    boosted = min(float(QUALITY_RISK_MAX_MULT), float(risk_mult) * float(QUALITY_RISK_MULT))
    if boosted <= risk_mult:
        return risk_mult, ""
    return boosted, f"QualityRisk:{boosted:.2f}"


def _apply_relative_strength_risk_overlay(
    risk_mult: float,
    *,
    rel_strength: float,
) -> tuple[float, str]:
    """Boost risk_mult when coin outperforms BTC by 0.5–2% (strong relative momentum)."""
    if not REL_STRENGTH_RISK_UP:
        return risk_mult, ""
    if not (REL_STRENGTH_RISK_UP_MIN_PCT < rel_strength <= REL_STRENGTH_RISK_UP_MAX_PCT):
        return risk_mult, ""
    boosted = min(float(REL_STRENGTH_RISK_UP_MAX_MULT), float(risk_mult) * float(REL_STRENGTH_RISK_UP_MULT))
    if boosted <= risk_mult:
        return risk_mult, ""
    return boosted, f"RelStrengthRisk:{boosted:.2f}"


def _apply_trend_pair_risk_overlay(
    risk_mult: float,
    *,
    trend_1h: str,
    trend_4h: str,
) -> tuple[float, str]:
    """Boost risk_mult when both 1h and 4h trends are fully bullish."""
    if not TREND_PAIR_RISK_UP:
        return risk_mult, ""
    if str(trend_1h or "").lower() != TREND_PAIR_RISK_UP_1H:
        return risk_mult, ""
    if str(trend_4h or "").lower() != TREND_PAIR_RISK_UP_4H:
        return risk_mult, ""
    boosted = min(float(TREND_PAIR_RISK_UP_MAX_MULT), float(risk_mult) * float(TREND_PAIR_RISK_UP_MULT))
    if boosted <= risk_mult:
        return risk_mult, ""
    return boosted, f"TrendPairRisk:{boosted:.2f}"


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


def _zone_payload(zone, source: str, current: float, age=None):
    """Normalize a (low, high) zone tuple into entry dict form."""
    if not zone:
        return None
    low, high = sorted([float(zone[0]), float(zone[1])])
    if low <= 0 or high <= 0 or high <= low:
        return None
    mid = (low + high) / 2
    return {
        "entry_low":     round(low, 8),
        "entry_high":    round(high, 8),
        "entry_price":   round(mid, 8),
        "entry_source":  source,
        "market_price":  round(current, 8),
        "zone_age_bars": int(age) if age is not None else -1,
        "zone_width_pct": round((high - low) / mid, 8) if mid > 0 else 0.0,
    }


from config import SMC_FVG_MAX_FILL as _FVG_MAX_FILL   # see config.py
from config import (SMC_VOL_STRONG_TIER,
                    RSI_SCORE_LONG_MIN, RSI_SCORE_LONG_MAX,
                    RSI_SCORE_SHORT_MIN, RSI_SCORE_SHORT_MAX)


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
    """Prefer OB zone, then FVG zone. Skip FVG if > 80% already filled."""
    current = ind["current_close"]
    if direction == "LONG":
        ob_z  = _zone_payload(ind.get("bull_ob_zone"), "OB", current, ind.get("bull_ob_age"))
        fvg_z = ind.get("bullish_fvg_zone")
        fvg_p = _zone_payload(fvg_z, "FVG", current, ind.get("bullish_fvg_age")) if _fvg_fresh(fvg_z, current, "LONG") else None
        return ob_z or fvg_p
    ob_z  = _zone_payload(ind.get("bear_ob_zone"), "OB", current, ind.get("bear_ob_age"))
    fvg_z = ind.get("bearish_fvg_zone")
    fvg_p = _zone_payload(fvg_z, "FVG", current, ind.get("bearish_fvg_age")) if _fvg_fresh(fvg_z, current, "SHORT") else None
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
        score += HTF_ALIGNED_SCORE; tags.append(f"1h+{HTF_ALIGNED_SCORE}")
    elif ind.get("trend_1h") == "neutral":
        score += HTF_NEUTRAL_SCORE; tags.append(f"1hN+{HTF_NEUTRAL_SCORE}")

    if ind.get("trend_4h") == bos:
        score += HTF_ALIGNED_SCORE; tags.append(f"4h+{HTF_ALIGNED_SCORE}")
    elif ind.get("trend_4h") == "neutral":
        score += HTF_NEUTRAL_SCORE; tags.append(f"4hN+{HTF_NEUTRAL_SCORE}")

    vol = float(ind.get("volume_ratio", 0.0))
    _vol_strong = (SMC_VOL_STRONG_TIER if SMC_VOL_STRONG_TIER > 0
                   else max(SMC_BOS_MIN_VOLUME * 1.35, 2.0))
    if vol >= _vol_strong:
        score += 2; tags.append("Vol+2")
    elif vol >= SMC_BOS_MIN_VOLUME:
        score += 1; tags.append("Vol+1")

    rsi = float(ind.get("rsi", 50.0))
    if direction == "LONG" and RSI_SCORE_LONG_MIN <= rsi <= RSI_SCORE_LONG_MAX:
        score += 1; tags.append("RSI+1")
    elif direction == "SHORT" and RSI_SCORE_SHORT_MIN <= rsi <= RSI_SCORE_SHORT_MAX:
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
        score += HTF_STRONG_SCORE; tags.append(f"Strong1h+{HTF_STRONG_SCORE}")
    if ind.get("trend_4h_strong") and ind.get("trend_4h") == bos:
        score += HTF_STRONG_SCORE; tags.append(f"Strong4h+{HTF_STRONG_SCORE}")

    # Nested OB: 1h OB overlaps 15m entry zone → double confluence
    if entry_zone:
        ob_1h_z = ind.get("bull_ob_1h_zone") if direction == "LONG" else ind.get("bear_ob_1h_zone")
        if ob_1h_z and _zones_overlap(ob_1h_z, (entry_zone["entry_low"], entry_zone["entry_high"])):
            score += 2; tags.append("NestedOB_1h+2")

    # Premium triple confluence (OB+FVG overlap + sweep) — highest-WR ICT setup.
    if premium:
        score += 3; tags.append("💎Premium+3")

    return score, tags


# ── Adaptive market-regime packs (ported from friend's v2, DEFAULT OFF) ────────

def _has_structural_confirmation(confirmations: list) -> bool:
    structural = {"FVG", "OB", "LiqSweep", "ChoCH"}
    return any(c in structural for c in confirmations)


def _adaptive_filter_pack(ind: dict, bos: str, direction: str,
                          confirmations: list, mtf_score: int) -> tuple:
    """
    Regime-aware final gate. Requires progressively higher quality as the market
    regime worsens (clean trend → mixed → choppy) and returns a per-regime
    risk_mult for position sizing.

    Returns: (allowed, pack_name, reason, risk_mult).
    """
    trend_1h = ind.get("trend_1h", "neutral")
    trend_4h = ind.get("trend_4h", "neutral")
    eff       = float(ind.get("eff_ratio", 1.0) or 0.0)
    vol_ratio = float(ind.get("volume_ratio", 0.0) or 0.0)
    vol_regime = _regime(ind)
    atr_pct   = float(ind.get("vol_atr_pct", 0.0) or 0.0)
    structural = _has_structural_confirmation(confirmations)
    strong_bos = bool(ind.get("bos_body_strong", False))
    strong_htf = bool(ind.get("trend_1h_strong")) or bool(ind.get("trend_4h_strong"))

    aligned = int(trend_1h == bos) + int(trend_4h == bos)
    neutral = int(trend_1h == "neutral") + int(trend_4h == "neutral")
    hot = vol_regime >= ADAPTIVE_HOT_VOL_RATIO

    if ADAPTIVE_BEAR_SQUEEZE_GUARD and direction == "SHORT" and aligned == 2:
        session = ind.get("session", "OFF_HOURS")
        if ADAPTIVE_BEAR_SKIP_NEW_YORK and session == "NEW_YORK":
            return False, "bear_squeeze", "skip full-trend shorts during New York", 0.0
        if vol_regime < ADAPTIVE_BEAR_VOL_MIN_RATIO or vol_regime >= ADAPTIVE_BEAR_VOL_MAX_RATIO:
            return False, "bear_squeeze", "skip full-trend shorts outside bear vol corridor", 0.0

    if vol_regime >= ADAPTIVE_EXTREME_VOL_RATIO or atr_pct >= ADAPTIVE_EXTREME_ATR_PCT:
        need_score = MTF_MIN_SCORE + ADAPTIVE_HOT_SCORE_BUMP + 1
        if not (aligned == 2 and structural and strong_bos and mtf_score >= need_score):
            return False, "extreme_vol", "skip extreme volatility", 0.0
        return True, "extreme_trend", "extreme vol with full trend+structure", ADAPTIVE_HOT_RISK_MULT

    if aligned == 2:
        pack = "trend_up" if direction == "LONG" else "trend_down"
        if hot:
            need_score = MTF_MIN_SCORE + ADAPTIVE_HOT_SCORE_BUMP
            if mtf_score < need_score:
                return False, "hot_vol", f"score {mtf_score} < {need_score}", ADAPTIVE_HOT_RISK_MULT
            if eff < ADAPTIVE_HOT_EFF_MIN:
                return False, "hot_vol", f"eff {eff:.2f} < {ADAPTIVE_HOT_EFF_MIN:.2f}", ADAPTIVE_HOT_RISK_MULT
            if vol_ratio < ADAPTIVE_HOT_MIN_VOLUME:
                return False, "hot_vol", f"volume {vol_ratio:.2f} < {ADAPTIVE_HOT_MIN_VOLUME:.2f}", ADAPTIVE_HOT_RISK_MULT
            if not (structural and (strong_bos or strong_htf)):
                return False, "hot_vol", "needs structure and strong BOS/HTF", ADAPTIVE_HOT_RISK_MULT
            return True, f"{pack}_hot", "aligned trend with hot-vol guard", ADAPTIVE_HOT_RISK_MULT
        return True, pack, "full HTF alignment", 1.0

    if aligned == 1 and neutral == 1:
        need_score = MTF_MIN_SCORE + ADAPTIVE_MIXED_SCORE_BUMP
        if mtf_score < need_score:
            return False, "mixed", f"score {mtf_score} < {need_score}", ADAPTIVE_MIXED_RISK_MULT
        if eff < ADAPTIVE_MIXED_EFF_MIN:
            return False, "mixed", f"eff {eff:.2f} < {ADAPTIVE_MIXED_EFF_MIN:.2f}", ADAPTIVE_MIXED_RISK_MULT
        if not structural:
            return False, "mixed", "needs structural confirmation", ADAPTIVE_MIXED_RISK_MULT
        if hot and (vol_ratio < ADAPTIVE_HOT_MIN_VOLUME or not strong_bos):
            return False, "mixed_hot", "hot mixed needs volume and strong BOS", ADAPTIVE_HOT_RISK_MULT
        return True, "mixed", "one HTF aligned, one neutral", ADAPTIVE_MIXED_RISK_MULT

    if neutral == 2:
        need_score = MTF_MIN_SCORE + ADAPTIVE_CHOP_SCORE_BUMP
        if mtf_score < need_score:
            return False, "choppy", f"score {mtf_score} < {need_score}", ADAPTIVE_CHOP_RISK_MULT
        if eff < ADAPTIVE_CHOP_EFF_MIN:
            return False, "choppy", f"eff {eff:.2f} < {ADAPTIVE_CHOP_EFF_MIN:.2f}", ADAPTIVE_CHOP_RISK_MULT
        if vol_ratio < ADAPTIVE_CHOP_MIN_VOLUME:
            return False, "choppy", f"volume {vol_ratio:.2f} < {ADAPTIVE_CHOP_MIN_VOLUME:.2f}", ADAPTIVE_CHOP_RISK_MULT
        if not (structural and strong_bos):
            return False, "choppy", "needs structure and strong BOS", ADAPTIVE_CHOP_RISK_MULT
        return True, "choppy", "range market top-quality retest", ADAPTIVE_CHOP_RISK_MULT

    return False, "conflict", "HTF conflict", 0.0


def _quality_breakdown(ind: dict, bos: str, entry_zone, adaptive_pack: str) -> dict:
    trend_score = 0
    if ind.get("trend_1h") == bos:
        trend_score += 35
    elif ind.get("trend_1h") == "neutral":
        trend_score += 15
    if ind.get("trend_4h") == bos:
        trend_score += 45
    elif ind.get("trend_4h") == "neutral":
        trend_score += 20
    if ind.get("trend_1h_strong"):
        trend_score += 10
    if ind.get("trend_4h_strong"):
        trend_score += 10
    trend_score = min(100, trend_score)

    eff = float(ind.get("eff_ratio", 0.0) or 0.0)
    vol_ratio = _regime(ind)
    volatility_score = 40 + min(40, eff * 120)
    if 0.8 <= vol_ratio <= 1.8:
        volatility_score += 20
    elif 0.55 <= vol_ratio <= 3.0:
        volatility_score += 10
    volatility_score = int(max(0, min(100, volatility_score)))

    entry_score = 35 if entry_zone else 10
    if entry_zone and entry_zone.get("entry_source") == "OB":
        entry_score += 25
    elif entry_zone and entry_zone.get("entry_source") == "FVG":
        entry_score += 15
    if ind.get("bos_body_strong"):
        entry_score += 20
    if float(ind.get("volume_ratio", 0.0) or 0.0) >= 2.0:
        entry_score += 20
    entry_score = min(100, entry_score)

    portfolio_score = 80
    if adaptive_pack in ("mixed",):
        portfolio_score -= 10
    if adaptive_pack in ("choppy", "trend_up_hot", "trend_down_hot", "extreme_trend"):
        portfolio_score -= 25
    if adaptive_pack == "bear_squeeze":
        portfolio_score -= 50
    portfolio_score = max(0, min(100, portfolio_score))

    total = round(
        trend_score * 0.35
        + volatility_score * 0.20
        + entry_score * 0.30
        + portfolio_score * 0.15,
        1,
    )
    return {
        "trend_score": int(trend_score),
        "volatility_score": int(volatility_score),
        "entry_quality_score": int(entry_score),
        "portfolio_risk_score": int(portfolio_score),
        "quality_score": total,
    }


def _is_sniper(ind: dict, price: float, direction: str,
               trend_1h: str, trend_4h: str) -> bool:
    """Counter-structure entry: the one marker that has survived every change.

    Telemetry only — no gate reads it, nothing is shown to the user. It exists
    so the split stays measurable in setup_log (column kept as `sniper` to
    avoid a pointless migration; the name is historical).

    True when the entry cuts AGAINST the 15m swing structure — a LONG while the
    structure is bearish, a SHORT while it is bullish. Counter-intuitive, and
    consistent across two independent validations on different populations:
      seed, 10,300 trades, fill-at-zone:  83.1% / +0.552R vs 80.7% / +0.442R,
        same sign in all five years 2022-2026
      zone-watch population, 1,353 trades: 77.3% / +0.286R vs 73.8% / +0.168R,
        same sign in both windows (75%/+0.232 and 79%/+0.339)
    Mechanically it fits what this strategy is: entry at an FVG/OB retest, and a
    retest that cuts against the swing IS the deep pullback the zone exists to
    catch.

    ⚠️ The previous definition (stop < 2.09 ATR AND no full 1h/4h agreement)
    was retired 2026-08-13. It measured 90% win rate under the old
    chase-the-price entry and collapsed to 71.1% / +0.162R — i.e. below the
    73.8% baseline — once ZONE_WATCH started filling at the zone instead. Its
    stop-distance condition was selecting for roughly what waiting for the zone
    now selects for, so the two overlapped and the edge vanished. The lesson is
    general: a marker validated on one population is not validated on another.
    """
    if not SNIPER_TAG_ENABLED:
        return False
    try:
        want = "bullish" if direction == "LONG" else "bearish"
        swing = str(ind.get("swing_trend") or "")
        opposite = "bear" if direction == "LONG" else "bull"
        return swing == opposite
    except (TypeError, ValueError):
        return False


def _stability_overlay_pass(ind: dict, adaptive_pack: str, quality_score: float = 0.0) -> bool:
    """Final deterministic cut for regimes/sessions that validated poorly."""
    if not STABILITY_FILTERS_ENABLED:
        return True
    pack = (adaptive_pack or "").lower()
    session = str(ind.get("session", "") or "").upper()
    if pack in STABILITY_SKIP_PACKS:
        return False
    if session in STABILITY_SKIP_SESSIONS:
        return False
    if float(ind.get("eff_ratio", 0.0) or 0.0) < STABILITY_MIN_EFF_RATIO:
        return False
    if float(ind.get("volume_ratio", 0.0) or 0.0) < STABILITY_MIN_VOLUME_RATIO:
        return False
    if quality_score < STABILITY_MIN_QUALITY_SCORE:
        return False
    return True


# ── SMC filter ────────────────────────────────────────────────────────────────

def analyze_coin_smc(candles_15m: dict, candles_1h: dict, symbol: str,
                     candles_4h: dict = None, btc_change_pct: float = 0.0,
                     candles_1d: dict = None, diag: dict = None,
                     include_shadow: bool = False) -> dict | None:
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
    if SYMBOL_EDGE_FILTER and _norm_symbol(symbol) in _LOW_EDGE_SYMBOLS_NORM:
        return None
    symbol_norm = _norm_symbol(symbol)

    # Filter-variant experiment plumbing (see src/filter_variants.py).
    # A gate wired through _soft_fail() does not drop the setup outright when
    # include_shadow is on — it marks it shadow-only, so the variant that
    # relaxes exactly THAT gate can be measured live. Shadow setups are never
    # real signals (main.py routes them to Claude + setup_log only).
    # Soft-failing a SECOND, different gate drops the setup: no single variant
    # would have admitted it, so it belongs to no arm.
    shadow_only = False
    shadow_reason = ""

    def _soft_fail(reason: str) -> bool:
        """True = caller must return None. False = continue as shadow-only."""
        nonlocal shadow_only, shadow_reason
        if not include_shadow:
            return True
        if shadow_only and shadow_reason != reason:
            return True
        shadow_only = True
        shadow_reason = reason
        return False

    ind = get_smc_indicators(candles_15m, candles_1h, candles_4h)

    bos      = ind["bos"]
    trend_1h = ind["trend_1h"]
    trend_4h = ind["trend_4h"]
    trend_1d = _daily_trend(candles_1d)

    # 1. Must have BOS
    if not bos:
        return None

    # 1b. Macro daily trend filter (LONG only).
    #     Skip LONG when daily trend is bearish — price is in a day-scale downtrend.
    if DAILY_TREND_FILTER and bos == "bullish" and trend_1d == "bearish":
        return None

    # 1c. Double-neutral LONG block. 4h neutral + 1D neutral = full macro chop;
    #     longs get range-swept. Was soft-failable for variant C until
    #     2026-08-03: over a week it bound on exactly ONE setup, so the arm was
    #     measuring nothing and the slot was re-pointed (see filter_variants.py).
    if DOUBLE_NEUTRAL_LONG_FILTER and bos == "bullish" and trend_4h == "neutral" and trend_1d == "neutral":
        return None

    # 1d. Daily SHORT guard — don't short into a bullish daily trend.
    if DAILY_TREND_SHORT_FILTER and bos == "bearish" and trend_1d == "bullish":
        return None

    # 1e. Premium/discount dealing-range filter — TESTED AND DROPPED (default off).
    #     2026-06-11 A/B: strategy enters on structure breaks (price at range edge
    #     by design), so PD cut kills working entries: 0.5→3tr, 0.8→+25R, 0.9→+38R
    #     vs +71R baseline. Kept env-gated for re-testing on other entry models.
    if os.getenv("PD_RANGE_FILTER", "0") != "0":
        _pd_look = int(os.getenv("PD_RANGE_LOOKBACK", "96"))  # 96×15m = 24h
        _highs = candles_15m.get("high", [])[-_pd_look:]
        _lows  = candles_15m.get("low",  [])[-_pd_look:]
        if _highs and _lows:
            _rng_hi, _rng_lo = max(_highs), min(_lows)
            if _rng_hi > _rng_lo:
                _pos = (ind["current_close"] - _rng_lo) / (_rng_hi - _rng_lo)
                _pd_max = float(os.getenv("PD_RANGE_MAX", "0.5"))
                if bos == "bullish" and _pos > _pd_max:
                    return None
                if bos == "bearish" and _pos < (1.0 - _pd_max):
                    return None

    # 1f. BOS staleness/extension filter — VALIDATED, default ON (2026-07-16,
    #     in response to a chop cluster of stops chasing already-extended
    #     BOS). detect_bos scans a 10-candle window and returns a break
    #     anywhere in it — the signal can fire several candles after the
    #     actual break, after price has already run the move and is closer
    #     to exhaustion than continuation. Cuts entries that are either too
    #     old (candles_ago > max) or already extended too far past the break
    #     level (extension_atr > max).
    #     Thresholds WIDENED 2026-07-18: full-year backtest (365d/20sym) showed
    #     the extension_atr metric is NOT monotonic — the 2-3 ATR bucket
    #     (previously cut by the old 2.0 threshold) actually outperformed the
    #     passing 1-2 ATR bucket (netR/tr +0.473 vs +0.443); the real cliff is
    #     at 3+ ATR (WR craters to 50.3%). candles_ago degrades smoothly, no
    #     cliff before 8-9. Widening age 3→6 and extension 2.0→3.0 recovers
    #     ~1600 trades/yr at unchanged quality (WR 63.1% vs 63.4%, netR/tr
    #     +0.441 vs +0.440) — total net R +1932 vs +1229 (+57%).
    if os.getenv("BOS_STALENESS_FILTER", "1") != "0":
        _candles_ago = ind.get("bos_candles_ago")
        _ext_atr = ind.get("bos_extension_atr")
        _max_age = int(os.getenv("BOS_MAX_CANDLES_AGO", "6"))
        _max_ext = float(os.getenv("BOS_MAX_EXTENSION_ATR", "3.0"))
        if _candles_ago is not None and _candles_ago > _max_age:
            return None
        if _ext_atr is not None and _ext_atr > _max_ext:
            return None

    # 1g. Fully-aligned bullish HTF guard — VALIDATED, default OFF (2026-07-18).
    #     When BOTH 1h and 4h have already flipped bullish, the move has
    #     typically already run on every timeframe — the same "chasing an
    #     exhausted move" pattern as BOS staleness (1f), but at the HTF-
    #     structure level instead of the single-candle level. A LONG where 1h
    #     is still neutral (4h leading, 1h hasn't caught up) is meaningfully
    #     fresher. Backtest (365d/20sym, combined with the widened BOS
    #     threshold above):
    #       1h=bull & 4h=bull LONG:    WR 61.4%, SL% 20.4%, netR/tr +0.376
    #       1h=neutral & 4h=bull LONG: WR 70.2%, SL% 10.5%, netR/tr +0.660
    #     Cutting the fully-aligned bucket: WR 63.4%→66.0%, netR/tr
    #     +0.440→+0.523, maxDD -22.28R→-17.79R (fewer but cleaner LONGs) —
    #     but total net R drops ~43% (+1929.7R→+1102.6R, trades 4383→2123)
    #     since this bucket is most of all LONG volume. Verified live-code
    #     backtest with guard OFF: 4383 trades, WR 63.0%, netR/tr +0.440,
    #     netR +1929.69, maxDD -22.24R — essentially the pre-guard baseline
    #     drawdown, but ~75% more total profit. User chose profit over the
    #     smaller drawdown (2026-07-18) — default flipped OFF. Set
    #     HTF_ALIGNED_LONG_GUARD=1 to re-enable the quality/lower-DD variant.
    #     Bearish side is NOT symmetric — 1h=bear&4h=bear is a strong bucket
    #     (see TREND_PAIR_RISK_UP), so this guard is LONG-only.
    if os.getenv("HTF_ALIGNED_LONG_GUARD", "0") != "0":
        if bos == "bullish" and trend_1h == "bullish" and trend_4h == "bullish":
            return None

    # 2. Trend must match (neutral OK)
    if trend_1h != "neutral" and trend_1h != bos:
        return None
    if trend_4h != "neutral" and trend_4h != bos:
        return None

    # 2b. Regime filter — reject chop: no established HTF trend (both neutral)
    if REQUIRE_HTF_TREND and trend_1h == "neutral" and trend_4h == "neutral":
        return None

    # 2b-B. Fully-aligned strong HTF trend — see HTF_FULL_ALIGN_SKIP in config.
    # Buying a retest while both higher timeframes already run hard is a chase.
    if (HTF_FULL_ALIGN_SKIP and trend_1h == bos and trend_4h == bos
            and ind.get("trend_1h_strong")):
        return None

    # 2b-A. Efficiency-Ratio chop gate — false BOS in ranges → SL clusters
    if EFF_RATIO_MAX > 0 and float(ind.get("eff_ratio", 0.0) or 0.0) >= EFF_RATIO_MAX:
        return _rej("eff_ratio_too_clean")
    if EFF_RATIO_FILTER and ind.get("eff_ratio", 1.0) < EFF_RATIO_MIN:
        return None

    # 2b-B. Strict HTF alignment — both 1h AND 4h must back the signal
    if REQUIRE_STRICT_HTF and (trend_1h != bos or trend_4h != bos):
        return None

    # 2c. Volatility regime — skip dead markets (→ EXPIRED) and spikes (→ SL)
    if VOL_REGIME_FILTER:
        atr_pct = ind.get("vol_atr_pct", 0.0)
        v_ratio = _regime(ind)
        if atr_pct < VOL_MIN_ATR_PCT:
            return None
        if v_ratio < VOL_MIN_RATIO or v_ratio > VOL_MAX_RATIO:
            return None

    # 2d. Asymmetric bear-squeeze guard.
    #     Full bearish HTF shorts with hot volume = crowded late entries → squeeze.
    if (
        BEAR_TREND_HOT_VOL_GUARD
        and bos == "bearish"
        and trend_1h == "bearish"
        and trend_4h == "bearish"
        and _regime(ind) >= BEAR_TREND_HOT_VOL_MIN_RATIO
    ):
        return None
    if (
        BEAR_TREND_SKIP_SESSIONS
        and bos == "bearish"
        and trend_1h == "bearish"
        and trend_4h == "bearish"
        and str(ind.get("session", "") or "").upper() in BEAR_TREND_SKIP_SESSIONS
    ):
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

    # 5b. Directional RSI midline — BOS without momentum = higher false-break rate.
    #     LONG needs RSI ≥ 50 (midline reclaimed), SHORT needs RSI < 40.
    #     Was soft-failable for variant I until 2026-08-03: it bound on only 4
    #     setups in a week (net -0.6R), too little to ever conclude, so the slot
    #     was re-pointed (see filter_variants.py).
    if DIRECTIONAL_RSI_MIDLINE_FILTER:
        if bos == "bullish" and rsi < RSI_LONG_MIN_MIDLINE:
            return None
        if bos == "bearish" and rsi >= RSI_SHORT_MAX_MIDLINE:
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
        return _rej("no_direction")

    # 6a. Entry-candle wick-exhaustion gate — TESTED AND DROPPED (default off,
    #     2026-07-16). Hypothesis: an entry candle with an OPPOSING pin-bar
    #     wick (sellers hitting highs on a LONG / buyers defending lows on a
    #     SHORT) signals exhaustion, should be blocked. Backtest disagreed —
    #     hurt on every metric, 12d/20sym: WR 55.4%→50.0%, R/tr 0.488→0.303,
    #     maxDD unchanged. Combined with the (good) staleness filter above it
    #     also dragged that one down (0.638→0.381 R/tr). Kept env-gated for
    #     re-testing against a different entry model, same as PD_RANGE_FILTER.
    if os.getenv("WICK_EXHAUSTION_FILTER", "0") != "0":
        if bos == "bullish" and wicks.get("rejection") == "bearish":
            return _rej("wick_reject")
        if bos == "bearish" and wicks.get("rejection") == "bullish":
            return _rej("wick_reject")

    # 6-exp. Research-validated cuts (2026-06-11 A/B, 30/60/90d windows).
    #   RSI_Div setups: WR 23%, -0.21R/tr — 15m divergence in chop = noise.
    #   Monday + 18-20 UTC: near-zero R segments, cutting lifts WR ~2pp.
    if SKIP_RSI_DIV_SETUPS and "RSI_Div" in confirmations:
        return _rej("rsi_div")
    if SKIP_UTC_HOURS or SKIP_WEEKDAYS:
        _ts = (candles_15m.get("time") or [None])[-1]
        if _ts:
            from datetime import datetime as _dt, timezone as _tzz
            _d = _dt.fromtimestamp(int(_ts), tz=_tzz.utc)
            if str(_d.hour) in SKIP_UTC_HOURS:
                return _rej("skip_hour")
            if str(_d.weekday()) in SKIP_WEEKDAYS:
                return _rej("skip_weekday")

    # 6a. Direction edge filter — skip symbol/direction combos with proven poor edge.
    if DIRECTION_EDGE_FILTER:
        if direction == "LONG" and symbol_norm in _LOW_EDGE_LONG_SYMBOLS_NORM:
            return _rej("low_edge_long")
        if direction == "SHORT" and symbol_norm in _LOW_EDGE_SHORT_SYMBOLS_NORM:
            return _rej("low_edge_short")

    # 6a-1. Context momentum filters — coin relative to BTC + session momentum.
    coin_change_1h = _change_pct_from_1h(candles_1h or {}, RELATIVE_STRENGTH_LOOKBACK_HOURS)
    rel_strength   = coin_change_1h - float(btc_change_pct or 0.0)

    #     All five "context momentum pack" gates (here + 7b-1/7b-2/7b-3 below) are
    #     soft-failable under the shared "ctxmom" reason — variant F measures the
    #     whole pack switched OFF, since they were validated together on a single
    #     window and never walk-forward tested.
    if (
        LONG_RELATIVE_WEAKNESS_FILTER
        and direction == "LONG"
        and rel_strength <= LONG_RELATIVE_WEAKNESS_MAX_PCT
    ):
        if _soft_fail("ctxmom"):
            return _rej("ctx_momentum")
    if (
        LONG_NY_COIN_MOMENTUM_FILTER
        and direction == "LONG"
        and ind.get("session") == "NEW_YORK"
        and coin_change_1h <= LONG_NY_MIN_COIN_CHANGE_1H
    ):
        if _soft_fail("ctxmom"):
            return _rej("ctx_momentum")

    if len(confirmations) < SMC_MIN_CONFIRMATIONS:
        return _rej("few_confirmations")

    # 6b. Require >=1 STRUCTURAL confirmation — two weak candle signals
    #     (Engulfing + Wick) alone are noise, not smart-money structure.
    if REQUIRE_STRONG_CONFIRM:
        _STRUCTURAL = {"FVG", "OB", "LiqSweep", "ChoCH"}
        if not any(c in _STRUCTURAL for c in confirmations):
            return _rej("no_structural")

    # 6c. MACD+ChoCH noise — both on same bar = double-counted signal, not added confluence.
    if MACD_CHOCH_NOISE_FILTER and "MACD_Div" in confirmations and "ChoCH" in confirmations:
        return _rej("macd_choch_noise")

    # 6d. Overlap-session bearish 1h guard — A/B 8640×15m: +9.39R net, +0.5pp WR.
    #     Expansion session + bearish 1h = latecomers get squeezed at NYSE open.
    if OVERLAP_BEARISH_1H_GUARD and ind.get("session") == "OVERLAP" and trend_1h == "bearish":
        return _rej("overlap_bear_guard")

    # 7. Entry zone
    entry_zone = _select_entry_zone(ind, direction)
    if REQUIRE_ENTRY_ZONE and not entry_zone:
        return _rej("no_entry_zone")

    # 7a. Source edge filter — skip entry sources with proven poor edge per symbol.
    if SOURCE_EDGE_FILTER and entry_zone:
        _src = str(entry_zone.get("entry_source") or "").upper()
        if _src == "FVG" and symbol_norm in _LOW_EDGE_FVG_SYMBOLS_NORM:
            return _rej("low_edge_fvg")
        if _src == "OB" and symbol_norm in _LOW_EDGE_OB_SYMBOLS_NORM:
            return _rej("low_edge_ob")

    # 7b-1. Bull/neutral LONG narrow-zone filter — mixed-trend LONGs into tight zones
    #       wicked through and reversed to SL in backtest.
    if (
        BULL_NEUTRAL_LONG_NARROW_ZONE_FILTER
        and direction == "LONG"
        and trend_1h == "bullish"
        and trend_4h == "neutral"
        and entry_zone
        and float(entry_zone.get("zone_width_pct", 0.0) or 0.0) <= BULL_NEUTRAL_LONG_MAX_ZONE_WIDTH_PCT
    ):
        if _soft_fail("ctxmom"):
            return _rej("ctx_momentum")

    # 7b-2. Short FVG coin-momentum filter — coin still trending up fills FVG as support
    #       before the SHORT move materialises.
    if (
        SHORT_FVG_COIN_MOMENTUM_FILTER
        and direction == "SHORT"
        and entry_zone
        and str(entry_zone.get("entry_source") or "").upper() == "FVG"
        and coin_change_1h >= SHORT_FVG_MAX_COIN_CHANGE_1H
    ):
        if _soft_fail("ctxmom"):
            return _rej("ctx_momentum")

    # 7b-3. FVG London BTC-up filter — FVG LONGs in London when BTC already up >0.29%
    #       are late entries; expansion stalls then reverses at NYC open.
    if (
        FVG_LONDON_BTC_UP_FILTER
        and entry_zone
        and str(entry_zone.get("entry_source") or "").upper() == "FVG"
        and ind.get("session") == "LONDON"
        and btc_change_pct >= FVG_LONDON_BTC_UP_MIN_PCT
    ):
        if _soft_fail("ctxmom"):
            return _rej("ctx_momentum")

    # 7c. Retest — price must currently be at/near the zone (true retest, not chase)
    if REQUIRE_RETEST and entry_zone:
        cur    = ind["current_close"]
        z_low  = entry_zone["entry_low"]
        z_high = entry_zone["entry_high"]
        if cur < z_low:
            dist = (z_low - cur) / cur
        elif cur > z_high:
            dist = (cur - z_high) / cur
        else:
            dist = 0.0
        if dist > RETEST_MAX_DIST_PCT:
            return _rej("retest_too_far")

    # 8. MTF score (premium triple-confluence boosts score)
    premium = _premium_setup(ind, direction)
    ob_fvg_overlap = _ob_fvg_overlap(ind, direction)
    mtf_score, score_tags = _calc_mtf_score(
        ind, bos, direction, confirmations, btc_change_pct, entry_zone, premium
    )
    # Diagnostics: this coin survived ALL structural gates and got scored.
    # Lets run_scan log how many of N coins reach scoring + the best score,
    # so we can tell "strict gate" (close misses) from "no structure" (0 reach).
    if diag is not None:
        diag["reached_score"] = diag.get("reached_score", 0) + 1
        if mtf_score > diag.get("best_score", -1):
            diag["best_score"]  = mtf_score
            diag["best_symbol"] = symbol
    # Counter-structure setups clear a lower bar — see the bonus in config.py.
    # Computed here rather than reusing the telemetry call further down, because
    # that one runs after this gate has already thrown the setup away.
    _cs_bonus = 0
    if COUNTER_STRUCTURE_SCORE_BONUS:
        try:
            if _is_sniper(ind, price_payload["entry_price"], direction, trend_1h, trend_4h):
                _cs_bonus = int(COUNTER_STRUCTURE_SCORE_BONUS)
        except Exception:
            _cs_bonus = 0
    if mtf_score + _cs_bonus < MTF_MIN_SCORE:
        if diag is not None:
            diag["score_fail"] = diag.get("score_fail", 0) + 1
        # Variant D removed 2026-08-07 at the user's request: score near-misses
        # are no longer soft-failed, so nothing below MTF_MIN_SCORE survives and
        # the score-shadow batch stops costing Claude budget. To revive, restore
        #   if mtf_score < SHADOW_MIN_SCORE or _soft_fail("score"):
        # here and re-add the D arm in src/filter_variants.py.
        return _rej("low_mtf_score")

    # 8b. Adaptive regime pack gate (DEFAULT OFF — under backtest evaluation).
    #     Requires higher quality as the regime worsens + sets a per-regime risk_mult.
    adaptive_pack   = "base"
    adaptive_reason = "adaptive disabled"
    risk_mult       = 1.0
    if ADAPTIVE_FILTER_PACKS:
        allowed, adaptive_pack, adaptive_reason, risk_mult = _adaptive_filter_pack(
            ind, bos, direction, confirmations, mtf_score
        )
        if not allowed:
            return _rej("not_allowed")
    quality = _quality_breakdown(ind, bos, entry_zone, adaptive_pack)
    # NOTE: this also blocks 11:00-12:59 UTC outright (STABILITY_SKIP_SESSIONS
    # = OVERLAP), on the strength of 19 trades measured 2026-06-11 — before the
    # TP_WINDOW fix, the OKX migration, the Strong1h window fix and the real-BTC
    # fix, i.e. on a base now known to be wrong. Not one of the 10,300 seed rows
    # carries session=OVERLAP, so those two hours are a complete blind spot.
    # Variant C briefly measured them (2026-08-03) and was removed the same day
    # at the user's request; re-add a soft_fail here if that changes.
    if not _stability_overlay_pass(ind, adaptive_pack, quality["quality_score"]):
        return _rej("stability_overlay")

    # Risk multiplier overlays — boost size on statistically stronger setups (no filtering).
    risk_mult, quality_risk_tag = _apply_quality_risk_overlay(
        risk_mult,
        symbol=symbol,
        entry_source=entry_zone["entry_source"] if entry_zone else "MARKET",
        vol_ratio_regime=_regime(ind),
        rsi=float(rsi),
    )
    risk_mult, trend_pair_risk_tag = _apply_trend_pair_risk_overlay(
        risk_mult,
        trend_1h=trend_1h,
        trend_4h=trend_4h,
    )
    risk_mult, rel_strength_risk_tag = _apply_relative_strength_risk_overlay(
        risk_mult,
        rel_strength=round(float(rel_strength), 2),
    )

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
    if ADAPTIVE_FILTER_PACKS:
        signals.append(f"Q {quality['quality_score']:.1f}")
        signals.append(f"Pack:{adaptive_pack}")
        if abs(risk_mult - 1.0) > 1e-9:
            signals.append(f"Risk x{risk_mult:.2f}")
        score_tags.append(f"Pack:{adaptive_pack}")
        score_tags.append(f"RiskMult:{risk_mult:.2f}")
    elif quality_risk_tag or trend_pair_risk_tag or rel_strength_risk_tag:
        signals.append(f"Risk x{risk_mult:.2f}")
    if quality_risk_tag:
        score_tags.append(quality_risk_tag)
    if trend_pair_risk_tag:
        score_tags.append(trend_pair_risk_tag)
    if rel_strength_risk_tag:
        score_tags.append(rel_strength_risk_tag)

    # Use zone midpoint as entry price when available
    price_payload = entry_zone or {
        "entry_low":     round(ind["current_close"], 8),
        "entry_high":    round(ind["current_close"], 8),
        "entry_price":   round(ind["current_close"], 8),
        "entry_source":  "MARKET",
        "market_price":  round(ind["current_close"], 8),
        "zone_age_bars": -1,
        "zone_width_pct": 0.0,
    }

    return {
        "symbol":           symbol,
        "direction":        direction,
        "trend_1h":         trend_1h,
        "trend_4h":         ind["trend_4h"],
        "trend_1d":         trend_1d,
        "trend_1h_strong":  ind.get("trend_1h_strong", False),
        "swing_trend":      ind.get("swing_trend", ""),
        # Exposed for the filter-variant A/B experiment (variant I gate).
        "bos_candles_ago":  ind.get("bos_candles_ago"),
        "bos_extension_atr": ind.get("bos_extension_atr"),
        "bos_break_level":  ind.get("bos_break_level"),
        "overhead_atr":     ind.get("overhead_atr"),
        "underfoot_atr":    ind.get("underfoot_atr"),
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
        # Label only — no gate reads this. See config.py SNIPER_TAG_ENABLED.
        "sniper":           _is_sniper(ind, price_payload["entry_price"], direction,
                                       trend_1h, trend_4h),
        "eff_ratio":        ind.get("eff_ratio"),
        "vol_atr_pct":      ind.get("vol_atr_pct"),
        "vol_ratio_regime": ind.get("vol_ratio_regime"),
        "adaptive_pack":    adaptive_pack,
        "adaptive_reason":  adaptive_reason,
        "risk_mult":        round(float(risk_mult), 4),
        "quality_score":    quality["quality_score"],
        "trend_score":      quality["trend_score"],
        "volatility_score": quality["volatility_score"],
        "entry_quality_score":  quality["entry_quality_score"],
        "portfolio_risk_score": quality["portfolio_risk_score"],
        "volume_ratio":     ind["volume_ratio"],
        "absorption":       ind.get("absorption", 0.0),
        "buy_pressure":     ind.get("buy_pressure", 0.0),
        "accel_ratio":      ind.get("accel_ratio", 1.0),
        "obv_agree":        ind.get("obv_agree", 0.0),
        "obv_strength":     ind.get("obv_strength", 0.0),
        "current_price":    price_payload["entry_price"],
        "market_price":     price_payload["market_price"],
        "entry_low":        price_payload["entry_low"],
        "entry_high":       price_payload["entry_high"],
        "entry_source":     price_payload["entry_source"],
        "zone_age_bars":    price_payload.get("zone_age_bars", -1),
        "zone_width_pct":   price_payload.get("zone_width_pct", 0.0),
        "recent_high":      round(ind["recent_high"], 8),
        "recent_low":       round(ind["recent_low"], 8),
        "tp1_level":        ind.get("bull_tp1") if direction == "LONG" else ind.get("bear_tp1"),
        "tp2_level":        ind.get("bull_tp2") if direction == "LONG" else ind.get("bear_tp2"),
        "btc_change":       round(btc_change_pct, 2),
        "coin_change_1h":   round(coin_change_1h, 2),
        "rel_strength":     round(rel_strength, 2),
        "signals":          signals,
        "mtf_score":        mtf_score,
        "premium":          premium,
        "ob_fvg_overlap":   ob_fvg_overlap,
        "score_tags":       score_tags,
        "bullish_score":    mtf_score if direction == "LONG"  else 0,
        "bearish_score":    mtf_score if direction == "SHORT" else 0,
        "confirmations":    confirmations,
        "_shadow_only":     shadow_only,
        "_shadow_reason":   shadow_reason,
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
