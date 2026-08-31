"""
Market data via OKX API (v5, public endpoints — no key needed).

Analysis feed:  OKX global USDT perpetuals (deep liquidity, long history),
                instId format "BTC-USDT-SWAP".
Execution ref:  OKX EU X-Perps ("BTC-USD_UM_XPERP-<expiry>") — signal levels are
                re-anchored to the X-Perp price at publish time in main.py.

Internal symbol format stays Bybit-style "BTCUSDT" (no dash) everywhere — DB,
Claude memory and setup history keep full continuity across the migration.

OKX candle columns: [ts_ms, open, high, low, close, vol(contracts),
                     volCcy(base), volCcyQuote(quote), confirm]
OKX returns candles newest-first; we reverse to oldest-first.
"""
import time
import threading
import requests
import sys
import os
import logging as _log

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    QUOTE_ASSET, TOP_COINS_COUNT,
    MIN_24H_QUOTE_VOLUME_USDT, MAX_SPREAD_PCT, ALLOWED_SYMBOLS, BLOCKED_SYMBOLS,
    BLOCK_STABLE_BASES, LEVERAGED_TOKEN_SUFFIXES,
    KLINES_LIMIT, TIMEFRAME_KUCOIN, KLINES_INTERVAL_SEC,
    TIMEFRAME_1H_KUCOIN, KLINES_1H_LIMIT, KLINES_1H_INTERVAL_SEC,
    TIMEFRAME_4H_KUCOIN, KLINES_4H_LIMIT, KLINES_4H_INTERVAL_SEC,
    TIMEFRAME_1D_KUCOIN, KLINES_1D_LIMIT, KLINES_1D_INTERVAL_SEC,
)

_logger = _log.getLogger(__name__)

# OKX host candidates. OKX_BASE_URL env overrides (e.g. a proxy), otherwise
# main host with the AWS mirror as fallback. Railway EU region (Amsterdam)
# reaches both directly — no geoblock workaround needed.
OKX_HOSTS = [
    "https://www.okx.com",
    "https://aws.okx.com",
]
_working_host = {"url": None}

# Map legacy timeframe strings (KuCoin-era, used across config) to OKX bars.
# NOTE: OKX 1D default is UTC+8 aligned — "1Dutc" keeps daily candles on UTC
# boundaries (same as the old Bybit feed, so the daily trend filter is unmoved).
TIMEFRAME_MAP = {
    "15min": "15m",
    "1h":    "1H",
    "1hour": "1H",
    "4h":    "4H",
    "4hour": "4H",
    "1d":    "1Dutc",
    "1day":  "1Dutc",
}


def _okx_get(path: str, params: dict, timeout: int = 15):
    """GET an OKX public endpoint with host fallback.

    OKX_BASE_URL env (re-read each call) overrides the host list entirely.
    Caches the first working host so later calls hit it directly.
    OKX wraps errors in HTTP 200 + {"code": "..."} — raise on non-zero code.
    """
    base_override = os.getenv("OKX_BASE_URL", "").strip().rstrip("/")
    hosts = [base_override] if base_override else list(OKX_HOSTS)
    if not base_override and _working_host["url"] in hosts:
        hosts.remove(_working_host["url"])
        hosts.insert(0, _working_host["url"])

    last_err = None
    for base in hosts:
        try:
            resp = requests.get(f"{base}{path}", params=params, timeout=timeout)
            resp.raise_for_status()
            body = resp.json()
            if str(body.get("code", "0")) != "0":
                raise ValueError(f"OKX error {body.get('code')}: {body.get('msg', '')}")
            _working_host["url"] = base
            return body
        except Exception as e:
            _logger.warning(f"OKX FAIL {base}{path}: {e}")
            last_err = e
            continue
    raise last_err


# ── Symbol conversion (internal "BTCUSDT" ↔ OKX instIds) ─────────────────────

def _base_of(symbol: str) -> str:
    """Internal 'BTCUSDT' → base asset 'BTC'."""
    s = symbol.upper()
    return s[:-len(QUOTE_ASSET)] if s.endswith(QUOTE_ASSET) else s


def _swap_inst_id(symbol: str) -> str:
    """Internal 'BTCUSDT' → OKX analysis feed 'BTC-USDT-SWAP'."""
    return f"{_base_of(symbol)}-{QUOTE_ASSET}-SWAP"


def _safe_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _is_bad_symbol(symbol: str) -> bool:
    """Return True for synthetic/stable/blocked pairs that create noisy signals."""
    symbol = symbol.upper()
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


# ── X-Perps universe (what the user can actually trade on OKX EU) ────────────

_xperp_cache = {"at": 0.0, "by_base": {}, "ctval": {}}
_XPERP_TTL = 24 * 3600  # instrument list changes only on listings/rollover


def get_xperp_instruments() -> dict:
    """Map base asset → current X-Perp instId, e.g. {"BTC": "BTC-USD_UM_XPERP-310404"}.

    Cached 24h. The expiry date embedded in the instId changes on rollover
    (a new far-dated contract is auto-generated) — dynamic resolution here
    means rollover never breaks the bot. Returns last good cache on API failure.
    """
    now = time.time()
    if now - _xperp_cache["at"] < _XPERP_TTL and _xperp_cache["by_base"]:
        return _xperp_cache["by_base"]
    try:
        body = _okx_get("/api/v5/public/instruments",
                        {"instType": "FUTURES", "ruleType": "xperp"})
        by_base = {}
        _ctval = {}
        for inst in body.get("data", []):
            inst_id = str(inst.get("instId", ""))
            if "_UM_XPERP-" not in inst_id or inst.get("state") != "live":
                continue
            base = inst_id.split("-")[0]
            # Keep the farthest-dated contract per base (rollover overlap window)
            if base not in by_base or inst_id > by_base[base]:
                by_base[base] = inst_id
                # ctVal = units of the base asset per contract. Needed to turn
                # book sizes (which OKX quotes in CONTRACTS) into notional —
                # without it PEPE's book reads as $0.16 and BTC's as $1bn.
                try:
                    _ctval[inst_id] = float(inst.get("ctVal") or 1.0)
                except (TypeError, ValueError):
                    _ctval[inst_id] = 1.0
        if by_base:
            _xperp_cache["by_base"] = by_base
            _xperp_cache["ctval"] = _ctval
            _xperp_cache["at"] = now
    except Exception as e:
        _logger.warning(f"get_xperp_instruments failed (using stale cache): {e}")
    return _xperp_cache["by_base"]


def get_xperp_price(symbol: str):
    """Live X-Perp price for an internal symbol ('BTCUSDT' → BTC X-Perp last).
    This is the price the user actually trades on OKX EU. None if unavailable.
    """
    try:
        inst_id = get_xperp_instruments().get(_base_of(symbol))
        if not inst_id:
            return None
        body = _okx_get("/api/v5/market/ticker", {"instId": inst_id}, timeout=8)
        lst = body.get("data", [])
        if lst:
            px = _safe_float(lst[0].get("last"))
            return px if px > 0 else None
    except Exception as e:
        _logger.warning(f"get_xperp_price failed for {symbol}: {e}")
    return None


# ── Market data (analysis feed: OKX global USDT swaps) ───────────────────────

_crypto_swaps_cache = {"at": 0.0, "ids": set()}


def _crypto_swap_ids() -> set:
    """instIds of CRYPTO swaps only. OKX also lists stock/commodity-tracking
    swaps (MU, SNDK, SPCX… instCategory="3") as plain USDT swaps — they follow
    stock-market hours (dead candles overnight) and would poison SMC signals.
    Crypto = instCategory "1". Cached 24h; stale cache returned on API failure.
    """
    now = time.time()
    if now - _crypto_swaps_cache["at"] < _XPERP_TTL and _crypto_swaps_cache["ids"]:
        return _crypto_swaps_cache["ids"]
    try:
        body = _okx_get("/api/v5/public/instruments", {"instType": "SWAP"})
        ids = {
            str(i.get("instId", "")) for i in body.get("data", [])
            if str(i.get("instCategory", "1")) == "1" and i.get("state") == "live"
        }
        if ids:
            _crypto_swaps_cache["ids"] = ids
            _crypto_swaps_cache["at"] = now
    except Exception as e:
        _logger.warning(f"_crypto_swap_ids failed (using stale cache): {e}")
    return _crypto_swaps_cache["ids"]


def get_top_coins():
    """Top liquid CRYPTO pairs by 24h USD turnover, as internal 'BTCUSDT' names.
    Filters stock-tracking swaps, no-name/leveraged/stablecoin/low-volume/
    wide-spread pairs first.
    """
    body = _okx_get("/api/v5/market/tickers", {"instType": "SWAP"})
    rows = []
    suffix = f"-{QUOTE_ASSET}-SWAP"
    crypto_ids = _crypto_swap_ids()

    for t in body.get("data", []):
        inst_id = str(t.get("instId", ""))
        if not inst_id.endswith(suffix):
            continue
        if crypto_ids and inst_id not in crypto_ids:
            continue  # stock/commodity-tracking swap — not a crypto market
        symbol = inst_id.replace(suffix, "") + QUOTE_ASSET  # BTC-USDT-SWAP → BTCUSDT
        if _is_bad_symbol(symbol):
            continue
        last = _safe_float(t.get("last"))
        # volCcy24h is base-currency volume → × last = quote (USD) turnover
        turnover = _safe_float(t.get("volCcy24h")) * last
        if turnover < MIN_24H_QUOTE_VOLUME_USDT:
            continue
        bid = _safe_float(t.get("bidPx"))
        ask = _safe_float(t.get("askPx"))
        if bid > 0 and ask > 0 and ask > bid:
            mid = (ask + bid) / 2
            if ((ask - bid) / mid) * 100 > MAX_SPREAD_PCT:
                continue
        rows.append((symbol, turnover))

    rows.sort(key=lambda r: r[1], reverse=True)
    return [sym for sym, _ in rows[:TOP_COINS_COUNT]]


# Bar-aware candle cache (2026-07-31). The scanner runs every SCAN_INTERVAL_
# MINUTES (5) but closed candles only change when a new bar closes, so a 4h
# series was being re-downloaded ~48 times per bar and a 1d series ~288 times,
# for byte-identical data. Measured cost per symbol before this: 15m 869ms +
# 1h 410ms + 4h 391ms + 1d 369ms ~= 2.0s, times 25 symbols ~= 56s of pure
# waiting per scan — the bulk of the publish latency that lets price drift out
# of the setup zone (see the stale-entry guard in main.py).
#
# An entry stays valid until the CURRENTLY-FORMING bar closes, i.e. until
# newest_closed_open_time + 2*interval. A few seconds of grace past that avoids
# hammering the API at the exact boundary before the exchange has published the
# new candle; _MIN_REFETCH_SEC stops a retry storm if it is late.
_kl_cache: dict = {}
_KL_CACHE_MAX = 400          # 25 symbols x 4 timeframes leaves plenty of room
_KL_BOUNDARY_GRACE = 3       # seconds past the expected close before refetching
_MIN_REFETCH_SEC = 10        # never re-hit the API faster than this per key


# Candles are fetched 6-wide by the scan's ThreadPoolExecutor, so the eviction
# below (which iterates the whole dict) can race an insert from another worker
# and raise "dictionary changed size during iteration" — a hard failure, not
# just a lost cache entry. One lock over get/put; both are pure dict work,
# never holding it across a network call.
_kl_lock = threading.Lock()


def _kl_cache_get(key, interval_sec: int):
    with _kl_lock:
        e = _kl_cache.get(key)
        if not e:
            return None
        now = time.time()
        if now - e["fetched_at"] < _MIN_REFETCH_SEC:
            return e["data"]
        if now < e["t_newest"] + 2 * interval_sec + _KL_BOUNDARY_GRACE:
            return e["data"]
        return None


def _kl_cache_put(key, data: dict):
    times = data.get("time") or [0]
    entry = {"data": data, "t_newest": int(times[-1]), "fetched_at": time.time()}
    with _kl_lock:
        if len(_kl_cache) >= _KL_CACHE_MAX:
            oldest = min(_kl_cache, key=lambda k: _kl_cache[k]["fetched_at"])
            _kl_cache.pop(oldest, None)
        _kl_cache[key] = entry


def get_klines(symbol, interval=TIMEFRAME_KUCOIN, limit=KLINES_LIMIT,
               interval_sec=KLINES_INTERVAL_SEC, closed_only: bool = True):
    """
    Fetch OHLCV from the OKX analysis feed (global USDT swap).
    Returns plain dict of lists (oldest → newest):
    {"time": [...], "open": [...], "high": [...], "low": [...], "close": [...], "volume": [...]}

    closed_only=True drops the forming candle using OKX's own confirm flag
    (index 8: "1" = closed) — no repaint / mid-candle fake BOS.

    Closed-candle results are cached until the next bar actually closes (see
    _kl_cache above). closed_only=False is never cached — that caller wants the
    live forming bar.
    """
    _ck = (symbol, interval, limit) if closed_only else None
    if _ck is not None:
        _hit = _kl_cache_get(_ck, interval_sec)
        if _hit is not None:
            return _hit

    bar = TIMEFRAME_MAP.get(interval, "15m")
    inst_id = _swap_inst_id(symbol)
    want = limit + 2  # extra covers the forming candle drop

    # OKX caps /market/candles at 300 per request — paginate backwards with
    # "after" (= return records older than ts) for deep fetches (e.g. kNN's 1000).
    raw: list = []
    after = None
    while len(raw) < want:
        params = {"instId": inst_id, "bar": bar, "limit": min(want - len(raw), 300)}
        if after is not None:
            params["after"] = after
        page = _okx_get("/api/v5/market/candles", params).get("data", [])
        if not page:
            break
        raw.extend(page)  # newest-first within and across pages
        if len(page) < 300 and len(raw) < want:
            break  # feed exhausted
        after = page[-1][0]  # oldest ts of this page → next page is older

    # Collected newest-first — reverse to oldest-first so index[-1] = latest
    candles = list(reversed(raw))
    if not candles:
        raise ValueError(f"No candle data for {symbol}")

    if closed_only:
        candles = [c for c in candles if len(c) > 8 and c[8] == "1"]
    candles = candles[-limit:]

    if not candles:
        raise ValueError(f"No closed candle data for {symbol}")

    out = {
        "time":   [int(float(c[0])) // 1000 for c in candles],  # ms → seconds
        "open":   [float(c[1]) for c in candles],
        "high":   [float(c[2]) for c in candles],
        "low":    [float(c[3]) for c in candles],
        "close":  [float(c[4]) for c in candles],
        "volume": [float(c[6]) for c in candles],  # volCcy = base-currency volume
    }
    if _ck is not None:
        _kl_cache_put(_ck, out)
    return out


def get_klines_xperp(symbol, limit=60, include_forming=False):
    """Closed 15m candles of the X-Perp contract (the user's actual market).

    Used by the open-position monitor so TP/SL hits are judged on the prices
    the user's position actually experiences (X-Perp wicks can differ slightly
    from the global feed). Returns None on any failure — caller falls back to
    the global analysis feed.

    include_forming=True appends the currently-forming (unclosed) candle as
    the last row, with its live high/low-so-far and last-traded close. Without
    this, a SL/TP touch is only detectable once the 15m candle CLOSES — since
    the 1-min monitor polls every minute but candles close every 15, that adds
    up to ~15min (avg ~7.5min) of pure notification lag versus the real fill
    (autotrade's exchange-side stop order fires the instant price touches it,
    independent of candle closes). Safe for breach detection: a real price
    touch already happened and can't un-happen even if the candle's close
    later differs. Only enable where immediate SL/TP detection matters, not
    for logic that specifically wants a settled/closed candle's close price.
    """
    try:
        inst_id = get_xperp_instruments().get(_base_of(symbol))
        if not inst_id:
            return None
        raw = _okx_get("/api/v5/market/candles",
                       {"instId": inst_id, "bar": "15m",
                        "limit": min(limit + 2, 300)}).get("data", [])
        raw_newest_first = raw  # OKX order: newest-first
        closed = [c for c in reversed(raw_newest_first) if len(c) > 8 and c[8] == "1"]
        closed = closed[-limit:]
        candles = list(closed)
        if include_forming and raw_newest_first:
            forming = raw_newest_first[0]
            if len(forming) > 8 and forming[8] == "0":
                candles = candles + [forming]
        if not candles:
            return None
        return {
            "time":   [int(float(c[0])) // 1000 for c in candles],
            "open":   [float(c[1]) for c in candles],
            "high":   [float(c[2]) for c in candles],
            "low":    [float(c[3]) for c in candles],
            "close":  [float(c[4]) for c in candles],
            "volume": [float(c[6]) for c in candles],
            # 1 = candle is closed, 0 = still forming. Callers that judge on a
            # CLOSE price (close-confirmed stop) must ignore unconfirmed rows:
            # a forming candle's "close" is just the current price, so treating
            # it as a settled close would fire on the very intrabar touch the
            # close-confirmation exists to ignore.
            "confirmed": [1 if (len(c) > 8 and c[8] == "1") else 0 for c in candles],
        }
    except Exception as e:
        _logger.debug(f"get_klines_xperp failed for {symbol}: {e}")
        return None


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


def get_klines_1d(symbol):
    """Fetch closed daily candles (UTC-aligned) for macro trend direction."""
    return get_klines(
        symbol,
        interval=TIMEFRAME_1D_KUCOIN,
        limit=KLINES_1D_LIMIT,
        interval_sec=KLINES_1D_INTERVAL_SEC,
        closed_only=True,
    )


def get_btc_change_1d() -> float:
    """Return BTC price change over the last closed day (%)."""
    try:
        candles = get_klines_1d("BTCUSDT")
        closes = candles["close"]
        if len(closes) < 2:
            return 0.0
        return (closes[-1] - closes[-2]) / closes[-2] * 100.0
    except Exception as e:
        # 0.0 reads as "BTC flat", which is not neutral: the correlation
        # and trend guards that consume it go inert, so a feed outage
        # quietly lets through trades those guards exist to block.
        # The value stays 0.0 (every caller assumes a float), but the
        # failure is no longer invisible.
        _logger.warning(f"BTC change feed failed, filters see flat: {e}")
        return 0.0


def get_btc_change_1h() -> float:
    """Return BTC price change over the last closed hour (%)."""
    try:
        candles = get_klines_1h("BTCUSDT")
        closes = candles["close"]
        if len(closes) < 2:
            return 0.0
        return (closes[-1] - closes[-2]) / closes[-2] * 100.0
    except Exception as e:
        # 0.0 reads as "BTC flat", which is not neutral: the correlation
        # and trend guards that consume it go inert, so a feed outage
        # quietly lets through trades those guards exist to block.
        # The value stays 0.0 (every caller assumes a float), but the
        # failure is no longer invisible.
        _logger.warning(f"BTC change feed failed, filters see flat: {e}")
        return 0.0


def get_current_price(symbol: str):
    """Last traded price from the OKX analysis feed (global swap).
    Falls back to last kline close if the ticker endpoint fails.
    Returns None only if both attempts fail.
    """
    try:
        body = _okx_get("/api/v5/market/ticker",
                        {"instId": _swap_inst_id(symbol)}, timeout=8)
        lst = body.get("data", [])
        if lst:
            price = _safe_float(lst[0].get("last"))
            if price > 0:
                return price
        _logger.warning(f"get_current_price: empty/zero result for {symbol}")
    except Exception as e:
        _logger.warning(f"get_current_price ticker failed for {symbol}: {e}")
    try:
        klines = get_klines(symbol, limit=2)
        close = klines["close"][-1]
        return close if close > 0 else None
    except Exception as e2:
        _logger.warning(f"get_current_price kline fallback failed for {symbol}: {e2}")
    return None


def get_open_interest(symbol: str, interval: str = "15min", limit: int = 5):
    """Open Interest series from OKX (oldest→newest list of floats, USD terms).

    SHADOW feature: rising OI with a same-direction price move = real money
    behind the break; falling OI = short-cover / long-liquidation (weak move).
    Uses the global swap (same instrument as the analysis feed).
    Returns [] on any failure.
    """
    try:
        body = _okx_get(
            "/api/v5/rubik/stat/contracts/open-interest-history",
            {
                "instId": _swap_inst_id(symbol),
                "period": "15m" if interval == "15min" else interval,
                "limit":  limit,
            },
            timeout=10,
        )
        # Rows: [ts_ms, oi_contracts, oi_base, oi_usd] — newest-first, reverse.
        vals = [float(r[3]) for r in reversed(body.get("data", []))
                if len(r) > 3 and r[3] not in (None, "")]
        return vals
    except Exception as e:
        _logger.debug(f"get_open_interest failed for {symbol}: {e}")
        return []


def get_funding_rate(symbol: str):
    """Current funding rate from OKX swap. Returns None if unavailable."""
    try:
        body = _okx_get("/api/v5/public/funding-rate",
                        {"instId": _swap_inst_id(symbol)}, timeout=10)
        data = body.get("data", [])
        if not data:
            return None
        rate = data[0].get("fundingRate")
        return float(rate) if rate not in (None, "") else None
    except Exception:
        return None


def get_xperp_book(symbol: str, levels: int = 20) -> dict | None:
    """Order-book snapshot of the X-Perp this symbol actually trades on.

    SHADOW feature (2026-08-07): recorded to setup_log, never read by the
    filter or shown to Claude. It exists because a third of the stops in the
    2026-08-07 report were "X-Perp only" — the deep global feed never confirmed
    the break — i.e. the stop came from a thin local book rather than a real
    move. Depth and spread measure that thinness directly, so this is aimed at
    a failure mode already observed rather than at a guess.

    Returns {spread_bps, depth_usd, imbalance} or None. Never raises.
      spread_bps — (ask-bid)/mid in basis points; wide = easy to wick through
      depth_usd  — notional resting across `levels` on BOTH sides
      imbalance  — (bid_depth - ask_depth) / total, -1..+1
    """
    try:
        inst_id = get_xperp_instruments().get(_base_of(symbol))
        if not inst_id:
            return None
        data = _okx_get("/api/v5/market/books",
                        {"instId": inst_id, "sz": levels}, timeout=8).get("data", [])
        if not data:
            return None
        bids = [(float(p), float(s)) for p, s, *_ in data[0].get("bids", [])]
        asks = [(float(p), float(s)) for p, s, *_ in data[0].get("asks", [])]
        if not bids or not asks:
            return None
        mid = (bids[0][0] + asks[0][0]) / 2.0
        if mid <= 0:
            return None
        # Book sizes are in CONTRACTS; ctVal converts to base units.
        ctv = _xperp_cache.get("ctval", {}).get(inst_id, 1.0) or 1.0
        bid_d = sum(p * s * ctv for p, s in bids)
        ask_d = sum(p * s * ctv for p, s in asks)
        total = bid_d + ask_d
        return {
            "spread_bps": round((asks[0][0] - bids[0][0]) / mid * 1e4, 3),
            "depth_usd":  round(total, 2),
            "imbalance":  round((bid_d - ask_d) / total, 4) if total else 0.0,
        }
    except Exception as e:
        _logger.debug(f"get_xperp_book failed for {symbol}: {e}")
        return None


def get_xperp_prices_bulk() -> dict:
    """{base_asset: last_price} for every live X-Perp, in ONE request.

    The zone-watch job polls every few seconds; asking per symbol would be one
    HTTP call per watched setup per tick. This is a single call regardless of
    how many are being watched.
    """
    try:
        body = _okx_get("/api/v5/market/tickers", {"instType": "FUTURES"}, timeout=10)
        out = {}
        for t in body.get("data", []):
            iid = str(t.get("instId", ""))
            if "_UM_XPERP-" not in iid:
                continue
            px = _safe_float(t.get("last"))
            if px > 0:
                out[iid.split("-")[0]] = px
        return out
    except Exception as e:
        _logger.debug(f"get_xperp_prices_bulk failed: {e}")
        return {}
