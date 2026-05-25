import requests
import pandas as pd
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

    # Keep only USDT pairs with real volume
    usdt_pairs = [
        d for d in data
        if d["symbol"].endswith(QUOTE_ASSET)
        and float(d["quoteVolume"]) > 0
    ]

    # Sort by USDT volume (highest first)
    usdt_pairs.sort(key=lambda x: float(x["quoteVolume"]), reverse=True)

    return [p["symbol"] for p in usdt_pairs[:TOP_COINS_COUNT]]


def get_klines(symbol, interval=TIMEFRAME, limit=KLINES_LIMIT):
    """Fetch OHLCV candlestick data for a symbol. Returns a pandas DataFrame."""
    url = f"{BINANCE_BASE_URL}/api/v3/klines"
    params = {"symbol": symbol, "interval": interval, "limit": limit}

    response = requests.get(url, params=params, timeout=15)
    response.raise_for_status()
    data = response.json()

    df = pd.DataFrame(data, columns=[
        "timestamp", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "trades",
        "taker_buy_base", "taker_buy_quote", "ignore"
    ])

    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)

    return df[["timestamp", "open", "high", "low", "close", "volume"]].copy()
