import requests
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import BINANCE_BASE_URL, QUOTE_ASSET, TOP_COINS_COUNT, TIMEFRAME, KLINES_LIMIT


def get_top_coins():
    """Fetch top N USDT pairs ranked by 24h trading volume."""
    url = f"{BINANCE_BASE_URL}/api/v3/ticker/24hr"
    response = requests.get(url, timeout=15)
    response.raise_for_status()
    data = response.json()

    usdt_pairs = [
        d for d in data
        if d["symbol"].endswith(QUOTE_ASSET) and float(d["quoteVolume"]) > 0
    ]
    usdt_pairs.sort(key=lambda x: float(x["quoteVolume"]), reverse=True)

    return [p["symbol"] for p in usdt_pairs[:TOP_COINS_COUNT]]


def get_klines(symbol, interval=TIMEFRAME, limit=KLINES_LIMIT):
    """
    Fetch OHLCV data. Returns a plain dict of lists:
    { "open": [...], "high": [...], "low": [...], "close": [...], "volume": [...] }
    """
    url = f"{BINANCE_BASE_URL}/api/v3/klines"
    params = {"symbol": symbol, "interval": interval, "limit": limit}

    response = requests.get(url, params=params, timeout=15)
    response.raise_for_status()
    data = response.json()

    return {
        "open":   [float(c[1]) for c in data],
        "high":   [float(c[2]) for c in data],
        "low":    [float(c[3]) for c in data],
        "close":  [float(c[4]) for c in data],
        "volume": [float(c[5]) for c in data],
    }
