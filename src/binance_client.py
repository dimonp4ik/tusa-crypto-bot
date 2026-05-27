"""
Market data via KuCoin API — accessible from US cloud servers (Render).
KuCoin symbol format: BTC-USDT (with dash).
KuCoin candle columns: [timestamp, open, close, high, low, volume, turnover]
"""
import time
import requests
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    KUCOIN_BASE_URL, QUOTE_ASSET, TOP_COINS_COUNT,
    MIN_24H_QUOTE_VOLUME_USDT, MAX_SPREAD_PCT, ALLOWED_SYMBOLS, BLOCKED_SYMBOLS,
    BLOCK_STABLE_BASES, LEVERAGED_TOKEN_SUFFIXES,
    KLINES_LIMIT, TIMEFRAME_KUCOIN, KLINES_INTERVAL_SEC,
    TIMEFRAME_1H_KUCOIN, KLINES_1H_LIMIT, KLINES_1H_INTERVAL_SEC,
    TIMEFRAME_4H_KUCOIN, KLINES_4H_LIMIT, KLINES_4H_INTERVAL_SEC,
)


def _drop_unclosed_candle(candles: list, interval_sec: int, now_ts: int) -> list:
    """
    KuCoin can return the currently forming candle. For signal logic we only want
    fully closed candles to avoid repaint / mid-candle fake BOS or volume spikes.
    """
    if not candles:
        return candles
    try:
        last_open_ts = int(float(candles[-1][0]))
    except (TypeError, ValueError, IndexError):
        return candles[:-1] if len(candles) > 1 else candles
    if now_ts < last_open_ts + interval_sec:
        return candles[:-1]
    return candles


def _safe_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _is_bad_symbol(symbol: str) -> bool:
    """Return True for synthetic/stable/blocked pairs that create noisy signals."""
    symbol = symbol.upper()
    base = symbol.split("-")[0]
    if ALLOWED_SYMBOLS and symbol not in ALLOWED_SYMBOLS:
        return True
    if symbol in BLOCKED_SYMBOLS:
        return True
    if base in BLOCK_STABLE_BASES:
        return True
    if base.endswith(LEVERAGED_TOKEN_SUFFIXES):
        return True
    return False


def _ticker_spread_pct(ticker: dict):
    """Return bid/ask spread in percent when KuCoin provides buy/sell quotes."""
    bid = _safe_float(ticker.get("buy"), 0.0)
    ask = _safe_float(ticker.get("sell"), 0.0)
    if bid <= 0 or ask <= 0 or ask < bid:
        return None
    mid = (ask + bid) / 2
    return ((ask - bid) / mid) * 100


def get_top_coins():
    """Fetch top liquid USDT pairs ranked by 24h USDT volume.
    Filters out no-name, leveraged, stablecoin, and low-volume pairs before
    any candle downloads or Claude calls are made.
    """
    url = f"{KUCOIN_BASE_URL}/api/v1/market/allTickers"
    response = requests.get(url, timeout=15)
    response.raise_for_status()
    data = response.json()

    tickers = data["data"]["ticker"]
    filtered = []

    for t in tickers:
        symbol = str(t.get("symbol", "")).upper()
        if not symbol.endswith(f"-{QUOTE_ASSET}"):
            continue
        if _is_bad_symbol(symbol):
            continue
        quote_volume = _safe_float(t.get("volValue"), 0.0)
        if quote_volume < MIN_24H_QUOTE_VOLUME_USDT:
            continue
        spread_pct = _ticker_spread_pct(t)
        if spread_pct is not None and spread_pct > MAX_SPREAD_PCT:
            continue
        filtered.append(t)

    filtered.sort(key=lambda x: _safe_float(x.get("volValue"), 0.0), reverse=True)
    return [t["symbol"] for t in filtered[:TOP_COINS_COUNT]]


def get_klines(symbol, interval=TIMEFRAME_KUCOIN, limit=KLINES_LIMIT,
               interval_sec=KLINES_INTERVAL_SEC, closed_only: bool = True):
    """
    Fetch OHLCV data from KuCoin.
    Returns plain dict of lists (oldest → newest):
    {"time": [...], "open": [...], "high": [...], "low": [...], "close": [...], "volume": [...]}

    closed_only=True removes the currently forming candle to avoid mid-candle
    fake BOS, volume spikes, and indicator repainting.
    """
    url = f"{KUCOIN_BASE_URL}/api/v1/market/candles"
    now = int(time.time())
    # Fetch one extra candle because closed_only can drop the latest one.
    start_at = now - ((limit + 1) * interval_sec)

    params = {
        "symbol":  symbol,
        "type":    interval,
        "startAt": start_at,
        "endAt":   now,
    }

    response = requests.get(url, params=params, timeout=15)
    response.raise_for_status()
    data = response.json()

    # KuCoin returns newest-first — reverse to oldest-first
    candles = list(reversed(data["data"]))
    if closed_only:
        candles = _drop_unclosed_candle(candles, interval_sec, now)
    candles = candles[-limit:]

    if not candles:
        raise ValueError(f"No candle data for {symbol}")

    return {
        "time":   [int(float(c[0])) for c in candles],
        "open":   [float(c[1]) for c in candles],
        "high":   [float(c[3]) for c in candles],   # index 3 = high
        "low":    [float(c[4]) for c in candles],   # index 4 = low
        "close":  [float(c[2]) for c in candles],   # index 2 = close
        "volume": [float(c[5]) for c in candles],
    }


def get_klines_1h(symbol):
    """Fetch closed 1h candles for trend direction."""
    return get_klines(
        symbol,
        interval=TIMEFRAME_1H_KUCOIN,
        limit=KLINES_1H_LIMIT,
        interval_sec=KLINES_1H_INTERVAL_SEC,
        closed_only=True,
    )


def get_klines_4h(symbol):
    """Fetch closed 4h candles for higher timeframe bias."""
    return get_klines(
        symbol,
        interval=TIMEFRAME_4H_KUCOIN,
        limit=KLINES_4H_LIMIT,
        interval_sec=KLINES_4H_INTERVAL_SEC,
        closed_only=True,
    )


def get_btc_change_1h() -> float:
    """Return BTC price change over the last closed hour (%)."""
    try:
        candles = get_klines_1h("BTC-USDT")
        closes = candles["close"]
        if len(closes) < 2:
            return 0.0
        return (closes[-1] - closes[-2]) / closes[-2] * 100.0
    except Exception:
        return 0.0


def get_funding_rate(symbol: str):
    """Get current funding rate from KuCoin futures. Returns None if unavailable."""
    base = symbol.replace("-USDT", "")
    futures_symbol = "XBTUSDTM" if base == "BTC" else f"{base}USDTM"
    try:
        url = f"https://api-futures.kucoin.com/api/v1/funding-rate/{futures_symbol}/current"
        resp = requests.get(url, timeout=10)
        if resp.status_code != 200:
            return None
        rate = resp.json().get("data", {}).get("value")
        return float(rate) if rate is not None else None
    except Exception:
        return None
