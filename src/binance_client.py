"""
Market data via Bybit API.
Bybit symbol format: BTCUSDT (no dash).
Bybit candle columns: [timestamp, open, high, low, close, volume, ...]
"""
import time
import requests
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    QUOTE_ASSET, TOP_COINS_COUNT,
    MIN_24H_QUOTE_VOLUME_USDT, MAX_SPREAD_PCT, ALLOWED_SYMBOLS, BLOCKED_SYMBOLS,
    BLOCK_STABLE_BASES, LEVERAGED_TOKEN_SUFFIXES,
    KLINES_LIMIT, TIMEFRAME_KUCOIN, KLINES_INTERVAL_SEC,
    TIMEFRAME_1H_KUCOIN, KLINES_1H_LIMIT, KLINES_1H_INTERVAL_SEC,
    TIMEFRAME_4H_KUCOIN, KLINES_4H_LIMIT, KLINES_4H_INTERVAL_SEC,
)

# Bybit geoblocks US cloud IPs (e.g. Render) with HTTP 403 on every domain.
# Workaround: route requests through a non-US proxy.
#
# BYBIT_PROXY_BASE — full base URL of a proxy that forwards to Bybit, e.g. a
#   Cloudflare Worker "https://bybit-proxy.xxx.workers.dev". When set, it
#   REPLACES the Bybit host (path + params are appended unchanged).
# BYBIT_HTTPS_PROXY — standard HTTP(S) proxy URL, e.g. "http://user:pass@ip:port".
#   When set, requests are tunnelled through it to the real Bybit host.
_PROXY_BASE  = os.getenv("BYBIT_PROXY_BASE", "").strip().rstrip("/")
_HTTPS_PROXY = os.getenv("BYBIT_HTTPS_PROXY", "").strip()

# Host candidates (used when no BYBIT_PROXY_BASE override). api.bytick.com is
# Bybit's mirror; both share the same geoblock but kept as a cheap fallback.
BYBIT_HOSTS = [
    "https://api.bybit.com",
    "https://api.bytick.com",
]
_working_host = {"url": None}

# Map KuCoin timeframe strings to Bybit interval param
TIMEFRAME_MAP = {
    "15min": "15",
    "1h": "60",
    "4h": "240",
}
BYBIT_INTERVAL_15M = TIMEFRAME_MAP.get(TIMEFRAME_KUCOIN, "15")
BYBIT_INTERVAL_1H = TIMEFRAME_MAP.get(TIMEFRAME_1H_KUCOIN, "60")
BYBIT_INTERVAL_4H = TIMEFRAME_MAP.get(TIMEFRAME_4H_KUCOIN, "240")


def _bybit_get(path: str, params: dict, timeout: int = 15):
    """
    GET a Bybit endpoint with geoblock workarounds:
      1. If BYBIT_PROXY_BASE set → hit that URL directly (proxy forwards to Bybit).
      2. Else try each Bybit host, optionally tunnelled via BYBIT_HTTPS_PROXY.
    Caches the first working host so later calls hit it directly.
    Raises the last error if every attempt fails.
    """
    # Re-read proxy from env each call — handles cases where env is set after module import
    https_proxy = os.getenv("BYBIT_HTTPS_PROXY", "").strip() or _HTTPS_PROXY
    proxies = {"http": https_proxy, "https": https_proxy} if https_proxy else None

    # Proxy-base override: single endpoint, no host rotation needed.
    if _PROXY_BASE:
        resp = requests.get(f"{_PROXY_BASE}{path}", params=params,
                            timeout=timeout, proxies=proxies)
        resp.raise_for_status()
        return resp

    # Order hosts so the last known-good one is tried first.
    hosts = list(BYBIT_HOSTS)
    if _working_host["url"] in hosts:
        hosts.remove(_working_host["url"])
        hosts.insert(0, _working_host["url"])

    import logging as _log
    _logger = _log.getLogger(__name__)

    last_err = None
    for base in hosts:
        try:
            resp = requests.get(f"{base}{path}", params=params,
                                timeout=timeout, proxies=proxies)
            resp.raise_for_status()
            _working_host["url"] = base
            return resp
        except Exception as e:
            _logger.warning(f"Bybit FAIL {base}: {e}")
            last_err = e
            continue
    raise last_err


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
    # Bybit format: BTCUSDT → base = BTC
    base = symbol.replace(QUOTE_ASSET, "")
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
    response = _bybit_get("/v5/market/tickers", {"category": "linear"})
    data = response.json()

    tickers = data.get("result", {}).get("list", [])
    filtered = []

    for t in tickers:
        symbol = str(t.get("symbol", "")).upper()
        if not symbol.endswith(QUOTE_ASSET):
            continue
        if _is_bad_symbol(symbol):
            continue
        # Bybit uses turnover24h (quote volume)
        quote_volume = _safe_float(t.get("turnover24h", "0"))
        if quote_volume < MIN_24H_QUOTE_VOLUME_USDT:
            continue
        bid = _safe_float(t.get("bid1Price", "0"))
        ask = _safe_float(t.get("ask1Price", "0"))
        if bid > 0 and ask > 0 and ask > bid:
            mid = (ask + bid) / 2
            spread = ((ask - bid) / mid) * 100
            if spread > MAX_SPREAD_PCT:
                continue
        filtered.append(t)

    filtered.sort(key=lambda x: _safe_float(x.get("turnover24h", "0"), 0.0), reverse=True)
    return [t["symbol"] for t in filtered[:TOP_COINS_COUNT]]


def get_klines(symbol, interval=TIMEFRAME_KUCOIN, limit=KLINES_LIMIT,
               interval_sec=KLINES_INTERVAL_SEC, closed_only: bool = True):
    """
    Fetch OHLCV data from Bybit.
    Returns plain dict of lists (oldest → newest):
    {"time": [...], "open": [...], "high": [...], "low": [...], "close": [...], "volume": [...]}

    closed_only=True removes the currently forming candle to avoid mid-candle
    fake BOS, volume spikes, and indicator repainting.

    Bybit candle format: [timestamp (ms), open, high, low, close, volume, turnover]
    """
    # Map interval (e.g. "15min") to Bybit format ("15")
    bybit_interval = TIMEFRAME_MAP.get(interval, "15")

    params = {
        "category": "linear",
        "symbol":   symbol,
        "interval": bybit_interval,
        "limit":    limit + 1,  # Fetch one extra; closed_only may drop it
    }

    response = _bybit_get("/v5/market/kline", params)
    data = response.json()

    # Bybit returns newest-first — reverse to oldest-first so index[-1] = latest
    candles = list(reversed(data.get("result", {}).get("list", [])))
    if not candles:
        raise ValueError(f"No candle data for {symbol}")

    now = int(time.time())
    if closed_only:
        candles = _drop_unclosed_candle(candles, interval_sec, now)
    candles = candles[-limit:]

    if not candles:
        raise ValueError(f"No closed candle data for {symbol}")

    return {
        "time":   [int(float(c[0])) // 1000 for c in candles],  # Bybit in ms, convert to seconds
        "open":   [float(c[1]) for c in candles],
        "high":   [float(c[2]) for c in candles],
        "low":    [float(c[3]) for c in candles],
        "close":  [float(c[4]) for c in candles],
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
        candles = get_klines_1h("BTCUSDT")
        closes = candles["close"]
        if len(closes) < 2:
            return 0.0
        return (closes[-1] - closes[-2]) / closes[-2] * 100.0
    except Exception:
        return 0.0


def get_current_price(symbol: str):
    """Fetch last traded price from Bybit linear ticker. Returns None on error."""
    try:
        resp = _bybit_get("/v5/market/tickers", {"category": "linear", "symbol": symbol}, timeout=8)
        lst  = resp.json().get("result", {}).get("list", [])
        if lst:
            return float(lst[0].get("lastPrice", 0)) or None
    except Exception:
        pass
    return None


def get_funding_rate(symbol: str):
    """Get current funding rate from Bybit futures. Returns None if unavailable."""
    # Convert BTCUSDT to BTCUSDT (already correct format for Bybit)
    try:
        params = {
            "category": "linear",
            "symbol":   symbol,
            "limit":    1,
        }
        resp = _bybit_get("/v5/market/funding/history", params, timeout=10)
        data = resp.json().get("result", {}).get("list", [])
        if not data:
            return None
        rate = data[0].get("fundingRate")
        return float(rate) if rate is not None else None
    except Exception:
        return None
