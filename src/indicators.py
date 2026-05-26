"""
Pure Python technical indicators + Smart Money Concepts (SMC).
No pandas, no numpy — works on any Python version.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import SMC_SWING_LOOKBACK, SMC_FVG_MIN_PCT, SMC_OB_LOOKBACK, ATR_PERIOD


# ── Basic indicators ──────────────────────────────────────────────────────────

def calculate_ema(values: list, period: int) -> list:
    """Exponential Moving Average."""
    k = 2 / (period + 1)
    ema = [values[0]]
    for v in values[1:]:
        ema.append(v * k + ema[-1] * (1 - k))
    return ema


def calculate_atr(highs: list, lows: list, closes: list, period: int = ATR_PERIOD) -> float:
    """
    Average True Range — measures volatility.
    Returns latest ATR value (in price units).
    """
    if len(closes) < period + 1:
        return 0.0

    trs = []
    for i in range(1, len(closes)):
        h_l    = highs[i] - lows[i]
        h_pc   = abs(highs[i] - closes[i - 1])
        l_pc   = abs(lows[i]  - closes[i - 1])
        trs.append(max(h_l, h_pc, l_pc))

    # Wilder smoothing
    atr = sum(trs[:period]) / period
    for i in range(period, len(trs)):
        atr = (atr * (period - 1) + trs[i]) / period

    return atr


def calculate_rsi(closes: list, period: int = 14) -> float:
    """RSI — returns only the latest value."""
    if len(closes) < period + 2:
        return 50.0

    deltas = [closes[i] - closes[i - 1] for i in range(1, len(closes))]
    gains  = [max(d, 0.0) for d in deltas]
    losses = [abs(min(d, 0.0)) for d in deltas]

    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period

    for i in range(period, len(deltas)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period

    rs = avg_gain / (avg_loss + 1e-10)
    return 100 - 100 / (1 + rs)


# ── SMC: Swing Points ─────────────────────────────────────────────────────────

def find_swing_points(highs: list, lows: list, lookback: int = SMC_SWING_LOOKBACK):
    """
    Find confirmed swing highs and lows.
    A swing high/low needs `lookback` candles on each side to confirm.
    Returns: (swing_highs, swing_lows) each as list of (index, price).
    """
    swing_highs = []
    swing_lows  = []
    n = len(highs)

    for i in range(lookback, n - lookback):
        window_h = highs[i - lookback: i + lookback + 1]
        window_l = lows[i - lookback: i + lookback + 1]

        if highs[i] == max(window_h):
            swing_highs.append((i, highs[i]))
        if lows[i] == min(window_l):
            swing_lows.append((i, lows[i]))

    return swing_highs, swing_lows


# ── SMC: Break of Structure ───────────────────────────────────────────────────

def detect_bos(closes: list, swing_highs: list, swing_lows: list,
               recent_candles: int = 5) -> str | None:
    """
    Detect Break of Structure in the last `recent_candles` candles.
    Returns 'bullish', 'bearish', or None.

    Bullish BOS: recent close breaks above a confirmed swing high.
    Bearish BOS: recent close breaks below a confirmed swing low.
    """
    if not swing_highs or not swing_lows or len(closes) < recent_candles:
        return None

    # Use last confirmed swing high/low (exclude very recent — not yet confirmed)
    last_sh = swing_highs[-1][1] if swing_highs else None
    last_sl = swing_lows[-1][1]  if swing_lows  else None

    # Check if any of the last N closes broke structure
    recent_closes = closes[-recent_candles:]

    if last_sh and any(c > last_sh for c in recent_closes):
        return "bullish"
    if last_sl and any(c < last_sl for c in recent_closes):
        return "bearish"

    return None


# ── SMC: Fair Value Gap ───────────────────────────────────────────────────────

def detect_fvg(opens: list, highs: list, lows: list, closes: list,
               min_pct: float = SMC_FVG_MIN_PCT) -> dict:
    """
    Detect Fair Value Gaps in last 40 candles, active near current price.

    Bullish FVG: highs[i] < lows[i+2]  (gap above candle i, below candle i+2)
    Bearish FVG: lows[i]  > highs[i+2] (gap below candle i, above candle i+2)

    Returns {'bullish': bool, 'bearish': bool}.
    """
    current = closes[-1]
    n = len(closes)
    search_from = max(0, n - 40)

    bullish_fvg = False
    bearish_fvg = False

    for i in range(search_from, n - 2):
        # Bullish FVG
        if highs[i] < lows[i + 2]:
            gap_bot = highs[i]
            gap_top = lows[i + 2]
            size = (gap_top - gap_bot) / gap_bot
            if size >= min_pct:
                # Active if price is testing or inside the gap
                if gap_bot * 0.999 <= current <= gap_top * 1.01:
                    bullish_fvg = True

        # Bearish FVG
        elif lows[i] > highs[i + 2]:
            gap_bot = highs[i + 2]
            gap_top = lows[i]
            size = (gap_top - gap_bot) / gap_bot
            if size >= min_pct:
                if gap_bot * 0.99 <= current <= gap_top * 1.001:
                    bearish_fvg = True

    return {"bullish": bullish_fvg, "bearish": bearish_fvg}


# ── SMC: Order Block ──────────────────────────────────────────────────────────

def detect_order_block(opens: list, highs: list, lows: list, closes: list,
                       lookback: int = SMC_OB_LOOKBACK) -> dict:
    """
    Detect Order Blocks near current price.

    Bullish OB: last bearish candle before a strong bullish impulse (3+ bull candles).
    Bearish OB: last bullish candle before a strong bearish impulse (3+ bear candles).

    Active if current price is retesting the OB zone.
    Returns {'bullish': bool, 'bearish': bool}.
    """
    current = closes[-1]
    n = len(closes)
    start = max(0, n - lookback)

    bull_ob = False
    bear_ob = False

    for i in range(start, n - 4):
        # Bullish OB: bearish candle → strong bullish impulse
        if closes[i] < opens[i]:
            next3_bull = all(closes[j] > opens[j] for j in range(i + 1, min(i + 4, n)))
            if next3_bull:
                move = (closes[min(i + 3, n - 1)] - closes[i]) / (closes[i] + 1e-10)
                if move > 0.005:  # at least 0.5% impulse
                    ob_top = max(opens[i], closes[i])
                    ob_bot = min(opens[i], closes[i])
                    # Active if price returned to OB zone
                    if ob_bot * 0.998 <= current <= ob_top * 1.005:
                        bull_ob = True

        # Bearish OB: bullish candle → strong bearish impulse
        elif closes[i] > opens[i]:
            next3_bear = all(closes[j] < opens[j] for j in range(i + 1, min(i + 4, n)))
            if next3_bear:
                move = (closes[i] - closes[min(i + 3, n - 1)]) / (closes[i] + 1e-10)
                if move > 0.005:
                    ob_top = max(opens[i], closes[i])
                    ob_bot = min(opens[i], closes[i])
                    if ob_bot * 0.995 <= current <= ob_top * 1.002:
                        bear_ob = True

    return {"bullish": bull_ob, "bearish": bear_ob}


# ── SMC: Liquidity Sweep ──────────────────────────────────────────────────────

def detect_liquidity_sweep(highs: list, lows: list, closes: list,
                            swing_highs: list, swing_lows: list,
                            check_last: int = 4) -> dict:
    """
    Detect liquidity sweeps (stop hunts).

    Bullish sweep: candle wicked below swing low then closed above it → reversal up.
    Bearish sweep: candle wicked above swing high then closed below it → reversal down.

    Returns {'bullish': bool, 'bearish': bool}.
    """
    if not swing_highs or not swing_lows:
        return {"bullish": False, "bearish": False}

    recent_sh = [p for _, p in swing_highs[-4:]]
    recent_sl = [p for _, p in swing_lows[-4:]]

    n = len(closes)
    bull_sweep = False
    bear_sweep = False

    for i in range(max(0, n - check_last), n):
        # Bullish sweep: wick below swing low, close above it
        for level in recent_sl:
            if lows[i] < level * 0.999 and closes[i] > level:
                bull_sweep = True

        # Bearish sweep: wick above swing high, close below it
        for level in recent_sh:
            if highs[i] > level * 1.001 and closes[i] < level:
                bear_sweep = True

    return {"bullish": bull_sweep, "bearish": bear_sweep}


# ── 1h Trend ──────────────────────────────────────────────────────────────────

def get_1h_trend(candles_1h: dict) -> str:
    """
    Determine trend from 1h candles using EMA9 vs EMA21.
    Returns 'bullish', 'bearish', or 'neutral'.
    """
    closes = candles_1h.get("close", [])
    if len(closes) < 22:
        return "neutral"

    ema9  = calculate_ema(closes, 9)
    ema21 = calculate_ema(closes, 21)

    if ema9[-1] > ema21[-1] * 1.001:
        return "bullish"
    if ema9[-1] < ema21[-1] * 0.999:
        return "bearish"
    return "neutral"


# ── Combined SMC indicator dict ───────────────────────────────────────────────

def get_smc_indicators(candles_15m: dict, candles_1h: dict = None,
                        candles_4h: dict = None) -> dict:
    """
    Run all SMC indicators on 15m candles + optional 1h for trend.
    Returns a flat dict of all signals.
    """
    closes  = candles_15m["close"]
    opens   = candles_15m["open"]
    highs   = candles_15m["high"]
    lows    = candles_15m["low"]
    volumes = candles_15m["volume"]

    # Swing points
    swing_highs, swing_lows = find_swing_points(highs, lows)

    # BOS
    bos = detect_bos(closes, swing_highs, swing_lows)

    # FVG
    fvg = detect_fvg(opens, highs, lows, closes)

    # Order Block
    ob = detect_order_block(opens, highs, lows, closes)

    # Liquidity sweep
    sweep = detect_liquidity_sweep(highs, lows, closes, swing_highs, swing_lows)

    # Volume
    avg_vol    = sum(volumes[-21:-1]) / 20 if len(volumes) >= 21 else sum(volumes) / len(volumes)
    vol_ratio  = volumes[-1] / (avg_vol + 1e-10)

    # RSI
    rsi = calculate_rsi(closes, 14)

    # ATR for stops/takes
    atr = calculate_atr(highs, lows, closes)

    # TP/SL reference levels
    recent_high = max(highs[-21:-1]) if len(highs) >= 22 else max(highs)
    recent_low  = min(lows[-21:-1])  if len(lows)  >= 22 else min(lows)

    # 1h + 4h trend
    trend_1h = get_1h_trend(candles_1h) if candles_1h else "neutral"
    trend_4h = get_1h_trend(candles_4h) if candles_4h else "neutral"  # same EMA logic

    return {
        "bos":           bos,              # 'bullish' | 'bearish' | None
        "bullish_fvg":   fvg["bullish"],
        "bearish_fvg":   fvg["bearish"],
        "bull_ob":       ob["bullish"],
        "bear_ob":       ob["bearish"],
        "bull_sweep":    sweep["bullish"],
        "bear_sweep":    sweep["bearish"],
        "trend_1h":      trend_1h,
        "trend_4h":      trend_4h,
        "rsi":           round(rsi, 2),
        "atr":           atr,
        "volume_ratio":  round(vol_ratio, 2),
        "current_close": closes[-1],
        "current_open":  opens[-1],
        "recent_high":   recent_high,
        "recent_low":    recent_low,
    }


# ── Legacy helper (kept for compatibility) ────────────────────────────────────

def get_indicators(candles: dict) -> dict:
    """Legacy indicator dict — used by old signal_filter."""
    closes  = candles["close"]
    highs   = candles["high"]
    lows    = candles["low"]
    volumes = candles["volume"]

    ema9  = calculate_ema(closes, 9)
    ema21 = calculate_ema(closes, 21)
    rsi   = calculate_rsi(closes, 14)

    avg_volume     = sum(volumes[-21:-1]) / 20 if len(volumes) >= 21 else sum(volumes) / len(volumes)
    current_volume = volumes[-1]
    volume_ratio   = current_volume / avg_volume if avg_volume > 0 else 1.0

    recent_high = max(highs[-21:-1])
    recent_low  = min(lows[-21:-1])

    return {
        "ema9":          ema9[-1],
        "ema21":         ema21[-1],
        "ema9_prev":     ema9[-2],
        "ema21_prev":    ema21[-2],
        "rsi":           rsi,
        "volume_ratio":  volume_ratio,
        "recent_high":   recent_high,
        "recent_low":    recent_low,
        "current_close": closes[-1],
        "current_open":  candles["open"][-1],
    }
