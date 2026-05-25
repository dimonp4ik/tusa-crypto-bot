"""
Market data via Bybit API — no geo-restrictions, works from any server.
Drop-in replacement for the previous Binance client.
"""
import requests
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import BYBIT_BASE_URL, QUOTE_ASSET, TOP_COINS_COUNT, KLINES_LIMIT, TIMEFRAME_BYBIT


def get_top_coins():
    """Fetch top N USDT spot pairs ranked by 24h turnover (USDT volume)."""
    url = f"{BYBIT_BASE_URL}/v5/market/tickers"
    response = requests.get(url, params={"category": "spot"}, timeout=15)
    response.raise_for_status()
    data = response.json()

    tickers = data["result"]["list"]

    # Keep only USDT pairs with real volume
    usdt = [
        t for t in tickers
        if t["symbol"].endswith(QUOTE_ASSET) and float(t.get("turnover24h", 0)) > 0
    ]

    # Sort by USDT turnover (highest volume first)
    usdt.sort(key=lambda x: float(x["turnover24h"]), reverse=True)

    return [t["symbol"] for t in usdt[:TOP_COINS_COUNT]]


def get_klines(symbol, interval=TIMEFRAME_BYBIT, limit=KLINES_LIMIT):
    """
    Fetch OHLCV candlestick data from Bybit.
    Returns plain dict of lists (oldest → newest):
    { "open": [...], "high": [...], "low": [...], "close": [...], "volume": [...] }
    """
    url = f"{BYBIT_BASE_URL}/v5/market/kline"
    params = {
        "category": "spot",
        "symbol":   symbol,
        "interval": interval,
        "limit":    limit,
    }

    response = requests.get(url, params=params, timeout=15)
    response.raise_for_status()
    data = response.json()

    # Bybit returns newest-first — reverse to oldest-first
    candles = list(reversed(data["result"]["list"]))

    # Format: [startTime, open, high, low, close, volume, turnover]
    return {
        "open":   [float(c[1]) for c in candles],
        "high":   [float(c[2]) for c in candles],
        "low":    [float(c[3]) for c in candles],
        "close":  [float(c[4]) for c in candles],
        "volume": [float(c[5]) for c in candles],
    }
