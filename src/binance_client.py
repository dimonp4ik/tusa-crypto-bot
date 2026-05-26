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
    KLINES_LIMIT, TIMEFRAME_KUCOIN, KLINES_INTERVAL_SEC,
    TIMEFRAME_1H_KUCOIN, KLINES_1H_LIMIT, KLINES_1H_INTERVAL_SEC,
    TIMEFRAME_4H_KUCOIN, KLINES_4H_LIMIT, KLINES_4H_INTERVAL_SEC,
)


def get_top_coins():
    """Fetch top N USDT pairs ranked by 24h USDT volume."""
    url = f"{KUCOIN_BASE_URL}/api/v1/market/allTickers"
    response = requests.get(url, timeout=15)
    response.raise_for_status()
    data = response.json()

    tickers = data["data"]["ticker"]

    # Keep only -USDT pairs with real volume
    usdt = [
        t for t in tickers
        if t["symbol"].endswith(f"-{QUOTE_ASSET}")
        and float(t.get("volValue", 0)) > 0
    ]

    # Sort by USDT volume (highest first)
    usdt.sort(key=lambda x: float(x["volValue"]), reverse=True)

    return [t["symbol"] for t in usdt[:TOP_COINS_COUNT]]


def get_klines(symbol, interval=TIMEFRAME_KUCOIN, limit=KLINES_LIMIT,
               interval_sec=KLINES_INTERVAL_SEC):
    """
    Fetch OHLCV data from KuCoin.
    Returns plain dict of lists (oldest → newest):
    { "open": [...], "high": [...], "low": [...], "close": [...], "volume": [...] }

    KuCoin candle column order: [time, open, close, high, low, volume, turnover]
    Note: close is index 2, high is index 3, low is index 4
    """
    url = f"{KUCOIN_BASE_URL}/api/v1/market/candles"
    now = int(time.time())
    start_at = now - (limit * interval_sec)

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

    if not candles:
        raise ValueError(f"No candle data for {symbol}")

    return {
        "open":   [float(c[1]) for c in candles],
        "high":   [float(c[3]) for c in candles],   # index 3 = high
        "low":    [float(c[4]) for c in candles],   # index 4 = low
        "close":  [float(c[2]) for c in candles],   # index 2 = close
        "volume": [float(c[5]) for c in candles],
    }


def get_klines_1h(symbol):
    """Fetch 1h candles for trend direction."""
    return get_klines(
        symbol,
        interval=TIMEFRAME_1H_KUCOIN,
        limit=KLINES_1H_LIMIT,
        interval_sec=KLINES_1H_INTERVAL_SEC,
    )


def get_klines_4h(symbol):
    """Fetch 4h candles for higher timeframe bias."""
    return get_klines(
        symbol,
        interval=TIMEFRAME_4H_KUCOIN,
        limit=KLINES_4H_LIMIT,
        interval_sec=KLINES_4H_INTERVAL_SEC,
    )
