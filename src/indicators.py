import pandas as pd


def calculate_ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(span=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-10)  # avoid division by zero
    return 100 - (100 / (1 + rs))


def get_indicators(df: pd.DataFrame) -> dict:
    """Calculate all indicators and return latest values as a dict."""
    close = df["close"]
    volume = df["volume"]

    ema9 = calculate_ema(close, 9)
    ema21 = calculate_ema(close, 21)
    rsi = calculate_rsi(close, 14)

    # Volume spike: last candle vs average of previous 20 candles
    avg_volume = volume.iloc[-21:-1].mean()
    current_volume = volume.iloc[-1]
    volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0

    # Breakout: current close vs highest/lowest of previous 20 candles
    recent_high = df["high"].iloc[-21:-1].max()
    recent_low = df["low"].iloc[-21:-1].min()

    return {
        "ema9":         ema9.iloc[-1],
        "ema21":        ema21.iloc[-1],
        "ema9_prev":    ema9.iloc[-2],
        "ema21_prev":   ema21.iloc[-2],
        "rsi":          rsi.iloc[-1],
        "volume_ratio": volume_ratio,
        "recent_high":  recent_high,
        "recent_low":   recent_low,
        "current_close": close.iloc[-1],
        "current_open":  df["open"].iloc[-1],
    }
