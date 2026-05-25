"""
Pure Python technical indicators — no pandas, no numpy.
Works on any Python version.
"""


def calculate_ema(values: list, period: int) -> list:
    """Exponential Moving Average."""
    k = 2 / (period + 1)
    ema = [values[0]]
    for v in values[1:]:
        ema.append(v * k + ema[-1] * (1 - k))
    return ema


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


def get_indicators(candles: dict) -> dict:
    """Calculate all indicators from candles dict. Returns latest values."""
    closes  = candles["close"]
    highs   = candles["high"]
    lows    = candles["low"]
    volumes = candles["volume"]

    ema9  = calculate_ema(closes, 9)
    ema21 = calculate_ema(closes, 21)
    rsi   = calculate_rsi(closes, 14)

    # Volume: last candle vs average of previous 20
    avg_volume     = sum(volumes[-21:-1]) / 20 if len(volumes) >= 21 else sum(volumes) / len(volumes)
    current_volume = volumes[-1]
    volume_ratio   = current_volume / avg_volume if avg_volume > 0 else 1.0

    # Breakout levels: high/low of previous 20 candles
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
