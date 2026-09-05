"""
Fast SMC backtest runner.

Usage:
    python backtest.py
    python backtest.py --symbols BTC-USDT,ETH-USDT --workers 4
    python backtest.py --stride 4 --export-trades backtest_trades.csv

The strategy logic still comes from src.signal_filter.analyze_coin_smc.
This file speeds up the runner around it:
  - process-level parallelism by symbol
  - zero-copy candle windows
  - exact cheap prefilter for BOS + volume before the expensive SMC stack
  - time-aligned 1h/4h snapshots
  - direct bracket simulation without per-bar future dict copies
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import os
import pickle
import sys
import time
import types
from bisect import bisect_right
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Iterable
from urllib.parse import urlencode  # noqa: F401 (kept for potential future use)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Backtests should run in a clean research environment even when optional app
# dependencies are not installed. The real bot still uses python-dotenv when
# present; this only lets config.py import with a no-op load_dotenv fallback.
if importlib.util.find_spec("dotenv") is None:
    dotenv_stub = types.ModuleType("dotenv")
    dotenv_stub.load_dotenv = lambda *args, **kwargs: None
    sys.modules["dotenv"] = dotenv_stub

from config import (  # noqa: E402
    BACKTEST_CANDLES,
    BACKTEST_FEE_RATE,
    BACKTEST_SLIPPAGE_RATE,
    BACKTEST_TP_WINDOW,
    BLOCKED_SYMBOLS,
    BLOCK_STABLE_BASES,
    KLINES_1H_INTERVAL_SEC,
    KLINES_4H_INTERVAL_SEC,
    KLINES_INTERVAL_SEC,
    LEVERAGED_TOKEN_SUFFIXES,
    QUOTE_ASSET,
    RISK_MAX_PCT,
    RISK_MIN_PCT,
    SL_ATR_BUFFER,
    SMC_BOS_MIN_VOLUME,
    SMC_SWING_LOOKBACK,
    TIMEFRAME_1H_KUCOIN,
    TIMEFRAME_4H_KUCOIN,
    TIMEFRAME_KUCOIN,
    TP1_R_MULT,
    TP2_R_MULT,
    SYMBOL_SIZE_MULT,
    COUNTER_STRUCTURE_SIZE_MULT,
    SESSION_SIZE_MULT,
    SYMBOL_TIER_MULT,
    SIZE_MULT_MAX,
    EXTENSION_ATR_THRESHOLD,
    EXTENSION_SIZE_MULT,
    VOL_ATR_BOOST_THRESHOLD,
    VOL_ATR_BOOST_MULT,
    HTF_NEUTRAL_4H_SIZE_MULT,
    BE_ARM_PROGRESS,
    MAX_SAME_DIRECTION_POSITIONS,
    ZONE_WATCH_ENABLED,
    ZONE_WATCH_MINUTES,
    SIGNAL_COOLDOWN_HOURS,
    KILL_SWITCH_SL_STREAK,
    LONDON_VOL_MIN, LONDON_THIN_SIZE_MULT,
    OVERLAP_VOL_MAX, OVERLAP_CALM_SIZE_MULT,
    CHOP_VOL_MIN, CHOP_EFF_MAX, CHOP_ATR_MIN, CHOP_SIZE_MULT,
    OPEN_SPACE_ROOM_MIN, OPEN_SPACE_SIZE_MULT,
    PARABOLIC_ACCEL_MIN, PARABOLIC_SIZE_MULT,
    RSI_STRETCH_LONG_MIN, RSI_STRETCH_SIZE_MULT,
    DEAD_THIN_VOL_MAX, DEAD_THIN_SIZE_MULT,
    HTF_NEUTRAL_1H_SIZE_MULT,
    TRAIL_ATR_MULT,
    TP1_CLOSE_FRAC,
    EXIT_PROFILE,
    POST_TP1_STRONG_TRAIL_ATR_MULT,
    POST_TP1_WEAK_TRAIL_ATR_MULT,
    POST_TP1_STRONG_CLOSE_PROGRESS,
    POST_TP1_STRONG_WICK_PROGRESS,
    POST_TP1_WEAK_CLOSE_PROGRESS,
    MIN_24H_QUOTE_VOLUME_USDT,
)
from src.signal_filter import analyze_coin_smc  # noqa: E402

# run_scan publishes at most this many signals per scan. It lives as a literal
# there rather than in config, so it is mirrored (not imported) — keep in sync.
# Swept 2026-08-28 against the hope that this was the throughput bottleneck.
# It is not — it barely binds:
#   cap   2026 trades  profit     ratios
#    3      1192      +396.98R   80.3/216.4   <- live value
#    4      1203      +401.73R   81.2/220.7
#    5      1205      +402.93R   81.5/221.6
# Going 3 -> 5 buys THIRTEEN trades (+1.1%) and +1.5% profit. Not worth a live
# change that would also raise concurrent correlated exposure and the per-scan
# Claude spend. NOTE the live side is not this variable: main.py hardcodes
# MAX_SIGNALS_PER_SCAN = 3 as a LOCAL inside the scan function, so raising it
# there is a code edit, not a setting.
_LIVE_MAX_PER_SCAN = int(os.getenv("BT_LIVE_MAX_PER_SCAN", "3"))
# Research handle, default 0 = honest. 1 restores the old kill-switch replay
# that read a trade's eventual outcome while walking entries, for A/B only.
_BT_KILL_LOOKAHEAD = os.getenv("BT_KILL_LOOKAHEAD", "0") == "1"

# Which setup wins a contested slot. The caps below (3/scan, 8/dir) bind
# constantly — the current window keeps 1172 trades and refuses 800 — so the
# tie-break inside one bar decides a large share of the book, not a detail.
#
#   alpha  : by symbol name. Arbitrary, and what this file did historically:
#            AAVE always beats ZEC, which models nothing real.
#   volume : by volume_ratio, then score. Models main.py:_setup_rank as it
#            stands today.
#   score  : by mtf_score, then volume. The reverse of that.
#
# Measured on the current window, and the answer is that it does not matter:
#   alpha   1172 trades  75.9%  +369.51R   ww 32.4  ulcer 127.7
#   volume  1174 trades  75.5%  +362.64R   ww 31.5  ulcer 116.4
#   score   1164 trades  75.8%  +365.14R   ww 31.9  ulcer 114.2
# Both orderings that model something real come out slightly BEHIND the
# arbitrary one, which is how you know the spread is noise rather than signal.
#
# Two reasons it cannot matter much. Contests are rare: 70.8% of trades are
# alone at their timestamp and only 10.5% come from a bar with three at once,
# so most of the 800 refusals are the cooldown and direction cap, not the scan
# cap. And neither ranking key predicts the outcome — logistic slope on the
# honest book gives mtf_score t=+0.72 and volume_ratio t=-0.78 in the current
# window, t=-0.01 and t=+1.29 in 2023, i.e. insignificant with signs that flip
# between windows.
#
# So the default stays "alpha" despite modelling nothing: the mismatch with
# live is real but measured immaterial, and changing it would move every
# baseline in this file by ~1% to buy nothing. Kept as an instrument.
#
# NOTE for anyone re-opening main.py:_setup_rank — its docstring justifies
# ranking on volume with win rates in the 80s, which are pre-2026-08-23
# fantasy-fill numbers. That justification is void. The replacement is not
# "rank by score" though: on honest data neither key clears significance.
_BT_SCAN_ORDER = os.getenv("BT_SCAN_ORDER", "alpha").lower()


def _scan_order_key(t):
    if _BT_SCAN_ORDER == "volume":
        return (t.entry_time or 0, -float(t.volume_ratio or 0),
                -int(t.mtf_score or 0), t.symbol, t.entry_bar)
    if _BT_SCAN_ORDER == "score":
        return (t.entry_time or 0, -int(t.mtf_score or 0),
                -float(t.volume_ratio or 0), t.symbol, t.entry_bar)
    return (t.entry_time or 0, t.symbol, t.entry_bar)
from src.knn_analog import knn_direction_score  # noqa: E402


PROJECT_DIR = Path(__file__).resolve().parent
CACHE_DIR = PROJECT_DIR / "backtest_cache"
# NOTE for anyone comparing runs: a backtest with no --end-date is anchored to
# NOW, and this TTL means it refetches every two hours and slides the window
# forward — old candles fall off the back, new ones arrive at the front. Two
# runs of the SAME config straddling a refresh will not agree: that is how a
# 1172-trade baseline came back as 1173 and briefly looked like config drift.
# The dated windows are immune, because their range is pinned. For any
# comparison that has to be exact, pass --end-date on the current window too.
CACHE_TTL_SEC = 2 * 3600

# OKX history — the SAME venue the live bot reads (2026-07-31). This used to
# pull Bybit candles while production ran on OKX: same asset, different order
# book, so the filter was being validated on data it never actually sees. The
# last structural live-vs-backtest divergence left after that day's parity work.
# Depth is not a constraint: probed back to 2022-07 on BTC-USDT-SWAP.
OKX_HOSTS = ["https://www.okx.com", "https://aws.okx.com"]
OKX_PAGE_LIMIT = 300   # OKX max candles per history-candles request

# internal interval → OKX bar string (mirrors src.binance_client.TIMEFRAME_MAP)
OKX_INTERVAL_MAP = {
    "15min": "15m", "1hour": "1H", "4hour": "4H",
    "15m": "15m", "1H": "1H", "4H": "4H",
    "1d": "1Dutc", "1Dutc": "1Dutc",
}


def _inst_id(symbol: str) -> str:
    """Internal 'BTCUSDT' → OKX analysis-feed instId 'BTC-USDT-SWAP'."""
    s = symbol.upper()
    base = s[:-len("USDT")] if s.endswith("USDT") else s
    return f"{base}-USDT-SWAP"

WINDOW_15M = 300
WINDOW_1H = 90
WINDOW_4H = 50
DEFAULT_WARMUP = 50

# Fixed symbol set: reproducible A/B runs. Internal format (no dashes) —
# resolved to '<BASE>-USDT-SWAP' instIds for the OKX feed by _inst_id().
# XMRUSDT and TONUSDT dropped 2026-07-31: neither exists as an OKX USDT swap,
# so the live bot can never trade them either. They were only ever testable
# because the backtest used to read Bybit. Removing them is a parity fix, not a
# loss — the pinned set now equals the universe production can actually reach.
BACKTEST_SYMBOLS = [
    "BTCUSDT", "ETHUSDT", "XRPUSDT", "SOLUSDT",
    "DOTUSDT", "XLMUSDT", "LINKUSDT", "SUIUSDT", "HYPEUSDT",
    "ZECUSDT", "SEIUSDT", "AAVEUSDT", "TAOUSDT", "NEARUSDT",
    "BILLUSDT", "LABUSDT", "ADAUSDT", "AVAXUSDT",
]


class Window:
    """Read-only list-like view over base[start:stop] without copying."""

    __slots__ = ("_base", "_start", "_stop")

    def __init__(self, base: list, start: int = 0, stop: int | None = None):
        self._base = base
        self._start = max(0, start)
        self._stop = len(base) if stop is None else max(self._start, min(stop, len(base)))

    def __len__(self) -> int:
        return self._stop - self._start

    def __iter__(self):
        base = self._base
        for i in range(self._start, self._stop):
            yield base[i]

    def __getitem__(self, idx):
        n = len(self)
        if isinstance(idx, slice):
            start, stop, step = idx.indices(n)
            base = self._base
            offset = self._start
            return [base[offset + i] for i in range(start, stop, step)]
        if idx < 0:
            idx += n
        if idx < 0 or idx >= n:
            raise IndexError(idx)
        return self._base[self._start + idx]

    def materialize(self) -> list:
        return self._base[self._start:self._stop]


def candle_window(candles: dict[str, list], start: int, stop: int) -> dict[str, Window]:
    return {k: Window(v, start, stop) for k, v in candles.items()}


def candle_slice(candles: dict[str, list], start: int, stop: int) -> dict[str, list]:
    return {k: v[start:stop] for k, v in candles.items()}


def parse_symbols(value: str | None) -> list[str]:
    if value:
        return [s.strip().upper() for s in value.split(",") if s.strip()]
    env_symbols = os.getenv("BACKTEST_SYMBOLS", "").strip()
    if env_symbols:
        return [s.strip().upper() for s in env_symbols.split(",") if s.strip()]
    return list(BACKTEST_SYMBOLS)


def _okx_get_bt(path: str, params: dict, timeout: int = 20, retries: int = 4):
    """OKX GET for backtest — host fallback + exponential backoff.

    Deep pagination across many symbols trips OKX rate limits and transient DNS
    failures; retrying with backoff makes a cold-cache prefetch reliable.
    """
    import requests as _req
    base = os.getenv("OKX_BASE_URL", "").strip().rstrip("/")
    hosts = [base] if base else OKX_HOSTS
    last_exc = None
    for attempt in range(retries):
        for host in hosts:
            try:
                r = _req.get(f"{host}{path}", params=params, timeout=timeout)
                r.raise_for_status()
                return r
            except Exception as e:
                last_exc = e
                continue
        time.sleep(1.5 * (attempt + 1))  # 1.5s, 3s, 4.5s backoff
    raise RuntimeError(f"All OKX hosts failed for {path}: {last_exc}")


def fetch_top_symbols(limit: int) -> list[str]:
    """Top OKX USDT swaps by 24h quote volume — same venue as the live bot.

    instId 'BTC-USDT-SWAP' is converted back to the internal 'BTCUSDT' format
    the rest of the pipeline (and the DB) uses.
    """
    resp = _okx_get_bt("/api/v5/market/tickers", {"instType": "SWAP"})
    tickers = resp.json().get("data", [])
    blocked = set(BLOCKED_SYMBOLS or [])
    rows = []
    for t in tickers:
        inst = str(t.get("instId", "")).upper()
        if not inst.endswith(f"-{QUOTE_ASSET}-SWAP"):
            continue
        base = inst.split("-")[0]
        symbol = f"{base}{QUOTE_ASSET}"
        if symbol in blocked or base in BLOCK_STABLE_BASES:
            continue
        if any(base.endswith(s) for s in LEVERAGED_TOKEN_SUFFIXES):
            continue
        try:
            vol = float(t.get("volCcyQuote") or 0.0)  # 24h turnover in quote ccy
        except (TypeError, ValueError):
            vol = 0.0
        if vol < MIN_24H_QUOTE_VOLUME_USDT:
            continue
        rows.append((vol, symbol))
    rows.sort(reverse=True)
    return [s for _, s in rows[:limit]]


def choose_workers(symbol_count: int, candles: int, stride: int) -> int:
    """Pick a low-overhead default for the common pinned-symbol backtest."""
    if symbol_count <= 1:
        return 1

    cpu = os.cpu_count() or 2
    effective_bars = max(1, candles // max(1, stride))

    if symbol_count <= 24 and effective_bars <= 2_000:
        return max(1, min(4, cpu, symbol_count))
    return max(1, min(8, cpu, symbol_count))


def cache_path(symbol: str, interval: str, count: int, end_date_ms: int | None = None) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    safe = symbol.replace("/", "_").replace("-", "_")
    suffix = f"_end{end_date_ms}" if end_date_ms else ""
    return CACHE_DIR / f"{safe}_{interval}_{count}{suffix}.pkl"


def _normalize_cached_candles(obj) -> dict[str, list] | None:
    if not isinstance(obj, dict):
        return None
    required = ("time", "open", "high", "low", "close", "volume")
    if any(k not in obj for k in required):
        return None
    lengths = {len(obj[k]) for k in required}
    if len(lengths) != 1 or not next(iter(lengths), 0):
        return None
    return {k: list(obj[k]) for k in required}


def fetch_history(
    symbol: str,
    interval: str,
    interval_sec: int,
    count: int,
    *,
    refresh_cache: bool = False,
    end_date_ms: int | None = None,
) -> dict[str, list]:
    """Fetch historical OKX candles with a local pickle cache.

    Same venue the live bot reads. Only closed candles (confirm == "1") are
    kept. OKX returns newest-first; we sort to oldest-first and paginate
    backwards via `after` (records strictly older than that ts).

    end_date_ms anchors the window's newest candle to a specific past moment
    instead of "now" — lets a seed batch target an exact historical range
    (e.g. 2022-2024) without re-downloading the overlap already covered by an
    earlier "last N candles from now" batch.
    """
    path = cache_path(symbol, interval, count, end_date_ms)
    if not refresh_cache and path.exists():
        age = time.time() - path.stat().st_mtime
        if age < CACHE_TTL_SEC:
            try:
                with path.open("rb") as f:
                    cached = _normalize_cached_candles(pickle.load(f))
                if cached:
                    return cached
            except Exception:
                pass

    okx_bar = OKX_INTERVAL_MAP.get(str(interval), "15m")
    inst_id = _inst_id(symbol)
    anchor_ms = int(end_date_ms) if end_date_ms else None
    after = anchor_ms  # OKX 'after' = records strictly older than this ts
    by_time: dict[int, list] = {}
    cutoff_ms = (anchor_ms or int(time.time() * 1000)) - count * interval_sec * 1000

    while len(by_time) < count:
        params = {"instId": inst_id, "bar": okx_bar, "limit": OKX_PAGE_LIMIT}
        if after is not None:
            params["after"] = str(after)
        resp = _okx_get_bt("/api/v5/market/history-candles", params)
        raw = resp.json().get("data", [])
        if not raw:
            break

        for c in raw:
            if len(c) > 8 and c[8] != "1":
                continue  # unclosed candle — skip (no repaint)
            ts_s = int(float(c[0])) // 1000
            if ts_s not in by_time:
                by_time[ts_s] = c

        oldest_ts_ms = int(float(raw[-1][0]))
        if len(raw) < OKX_PAGE_LIMIT or oldest_ts_ms <= cutoff_ms:
            break
        after = oldest_ts_ms  # next page = strictly older

    candles = [by_time[ts] for ts in sorted(by_time)][-count:]
    if not candles:
        raise ValueError(f"No OKX data for {inst_id} {interval}")

    # OKX columns: [ts_ms, o, h, l, c, vol(contracts), volCcy(base),
    # volCcyQuote, confirm]. volCcy (index 6) matches the live client exactly.
    data = {
        "time":   [int(float(c[0])) // 1000 for c in candles],
        "open":   [float(c[1]) for c in candles],
        "high":   [float(c[2]) for c in candles],
        "low":    [float(c[3]) for c in candles],
        "close":  [float(c[4]) for c in candles],
        "volume": [float(c[6]) for c in candles],
    }

    with path.open("wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

    return data


# RESEARCH ONLY: how far a structural level must be before TP1 snaps to it.
# Live hardcodes 1.0R in telegram_notifier.calculate_tp_sl; this makes the
# threshold sweepable so the 21%-of-trades structural bucket can be widened.
_STRUCT_TP1_MIN_R = float(os.getenv("BT_STRUCT_TP1_MIN_R", "1.0") or 1.0)
# RESEARCH ONLY: TP1 as a FIXED % of price instead of a multiple of risk.
# The user's idea — "take 0.5-1% of price movement". Worth testing because it
# behaves very differently per coin: on a 1.2% stop, 0.5% is 0.42R (break-even
# ~70%); on a 3% stop the same 0.5% is only 0.17R, which needs ~85% just to
# break even. 0 disables.
_TP1_FIXED_PCT = float(os.getenv("BT_TP1_FIXED_PCT", "0") or 0)


def calculate_tp_sl_local(
    price: float,
    direction: str,
    atr: float = 0.0,
    recent_high: float = 0.0,
    recent_low: float = 0.0,
    tp1_level: float | None = None,
    tp2_level: float | None = None,
) -> tuple[float, float, float]:
    """Local copy of telegram_notifier.calculate_tp_sl without requests import."""

    min_risk = price * RISK_MIN_PCT
    max_risk = price * RISK_MAX_PCT
    buf = atr * SL_ATR_BUFFER if atr and atr > 0 else 0.0

    if direction == "LONG":
        struct_sl = (recent_low - buf) if recent_low and recent_low > 0 else price - max_risk
        risk = min(max(price - struct_sl, min_risk), max_risk)
        sl = price - risk

        if _TP1_FIXED_PCT > 0:
            tp1 = price * (1 + _TP1_FIXED_PCT / 100.0)
        elif tp1_level and tp1_level > price * 1.001 and (tp1_level - price) >= risk * _STRUCT_TP1_MIN_R:
            tp1 = tp1_level
        else:
            tp1 = price + risk * TP1_R_MULT

        if tp2_level and tp2_level > tp1 * 1.001 and (tp2_level - price) >= risk * 1.5:
            tp2 = tp2_level
        else:
            tp2 = price + risk * TP2_R_MULT
            if tp2 <= tp1:
                tp2 = tp1 * 1.02
    else:
        struct_sl = (recent_high + buf) if recent_high and recent_high > 0 else price + max_risk
        risk = min(max(struct_sl - price, min_risk), max_risk)
        sl = price + risk

        if _TP1_FIXED_PCT > 0:
            tp1 = price * (1 - _TP1_FIXED_PCT / 100.0)
        elif tp1_level and tp1_level < price * 0.999 and (price - tp1_level) >= risk * _STRUCT_TP1_MIN_R:
            tp1 = tp1_level
        else:
            tp1 = price - risk * TP1_R_MULT

        if tp2_level and tp2_level < tp1 * 0.999 and (price - tp2_level) >= risk * 1.5:
            tp2 = tp2_level
        else:
            tp2 = price - risk * TP2_R_MULT
            if tp2 >= tp1:
                tp2 = tp1 * 0.98

    return round(tp1, 8), round(tp2, 8), round(sl, 8)


def _last_swing_high(highs: list[float], start: int, stop: int, lookback: int) -> float | None:
    for i in range(stop - lookback - 1, start + lookback - 1, -1):
        h = highs[i]
        if h == max(highs[i - lookback:i + lookback + 1]):
            return h
    return None


def _last_swing_low(lows: list[float], start: int, stop: int, lookback: int) -> float | None:
    for i in range(stop - lookback - 1, start + lookback - 1, -1):
        l = lows[i]
        if l == min(lows[i - lookback:i + lookback + 1]):
            return l
    return None


def cheap_prefilter_at(candles_15m: dict[str, list], end: int, window: int) -> bool:
    """
    Exact early reject for gates analyze_coin_smc also requires:
    enough candles, BOS present, and BOS-context volume threshold.
    """

    start = max(0, end - window)
    n = end - start
    if n < 30:
        return False

    volumes = candles_15m["volume"]
    if n >= 21:
        avg_vol = sum(volumes[end - 21:end - 1]) / 20
    else:
        avg_vol = sum(volumes[start:end]) / n
    volume_ratio = round(volumes[end - 1] / (avg_vol + 1e-10), 2)
    if volume_ratio < SMC_BOS_MIN_VOLUME:
        return False

    highs = candles_15m["high"]
    lows = candles_15m["low"]
    closes = candles_15m["close"]
    swing_lookback = SMC_SWING_LOOKBACK

    last_sh = _last_swing_high(highs, start, end, swing_lookback)
    if last_sh is None:
        return False
    last_sl = _last_swing_low(lows, start, end, swing_lookback)
    if last_sl is None:
        return False

    for i in range(max(start, end - 10), end - 1):
        c = closes[i]
        if c > last_sh or c < last_sl:
            return True
    return False


def aligned_slice_by_time(
    candles: dict[str, list],
    t_cur: int | None,
    lookback: int,
    fallback_end: int,
) -> dict[str, list]:
    if not candles or not candles.get("close"):
        return {}

    if t_cur is not None and candles.get("time"):
        end = bisect_right(candles["time"], t_cur)
    else:
        end = fallback_end

    end = max(1, min(end, len(candles["close"])))
    start = max(0, end - lookback)
    return candle_slice(candles, start, end)


_TP1_CLOSE_FRAC = max(0.0, min(1.0, float(TP1_CLOSE_FRAC)))
_RUNNER_FRAC = 1.0 - _TP1_CLOSE_FRAC


def _fld(setup: dict, key: str, missing: float) -> float:
    """Read a numeric setup field, substituting ONLY when it is absent.

    `float(setup.get(k) or default)` is the obvious spelling and it is wrong
    wherever zero is a legitimate value, because 0 is falsy: a zone that formed
    on the current bar has zone_age_bars == 0, and `0 or -1` is -1, i.e. "no
    zone". The fresh-zone trim silently skipped every age-0 trade — half its
    intended subset — until a direct call was checked against expected output.
    Same latent trap for eff_ratio (0 = perfectly non-directional) and for the
    volume fields that default to 99.
    """
    v = setup.get(key)
    if v is None or v == "":
        return missing
    try:
        return float(v)
    except (TypeError, ValueError):
        return missing


def _size_mult_for(symbol: str, setup: dict) -> float:
    """Mirror of the live sizing rules in src/autotrader.py."""
    m = float(SYMBOL_SIZE_MULT.get(str(symbol).upper(), 1.0))
    m *= float(SYMBOL_TIER_MULT.get(str(symbol).upper(), 1.0))
    if setup.get("sniper"):
        m *= float(COUNTER_STRUCTURE_SIZE_MULT)
    _sess = str(setup.get("session") or "").upper()
    m *= float(SESSION_SIZE_MULT.get(_sess, 1.0))
    # London boost only where volume confirms — see LONDON_VOL_MIN in config.
    # A calm OVERLAP rides bigger — see OVERLAP_VOL_MAX in config.py.
    if OVERLAP_CALM_SIZE_MULT != 1.0 and _sess == "OVERLAP":
        try:
            if _fld(setup, "volume_ratio", 99.0) < OVERLAP_VOL_MAX:
                m *= float(OVERLAP_CALM_SIZE_MULT)
        except (TypeError, ValueError):
            pass
    if LONDON_VOL_MIN > 0 and _sess == "LONDON":
        try:
            if _fld(setup, "volume_ratio", 0.0) < LONDON_VOL_MIN:
                m *= float(LONDON_THIN_SIZE_MULT) / float(
                    SESSION_SIZE_MULT.get("LONDON", 1.0) or 1.0)
        except (TypeError, ValueError, ZeroDivisionError):
            pass
    if str(setup.get("trend_4h") or "").lower() == "neutral":
        m *= float(HTF_NEUTRAL_4H_SIZE_MULT)
    if str(setup.get("trend_1h") or "").lower() == "neutral":
        m *= float(HTF_NEUTRAL_1H_SIZE_MULT)
    try:
        if _fld(setup, "bos_extension_atr", 0.0) > EXTENSION_ATR_THRESHOLD:
            m *= float(EXTENSION_SIZE_MULT)
    except (TypeError, ValueError):
        pass
    try:
        if _fld(setup, "vol_atr_pct", 0.0) >= VOL_ATR_BOOST_THRESHOLD:
            m *= float(VOL_ATR_BOOST_MULT)
    except (TypeError, ValueError):
        pass
    # Thin dead zone rides smaller — see DEAD_THIN_VOL_MAX in config.py.
    if DEAD_THIN_SIZE_MULT != 1.0 and _sess == "DEAD_ZONE":
        try:
            if _fld(setup, "volume_ratio", 99.0) < DEAD_THIN_VOL_MAX:
                m *= float(DEAD_THIN_SIZE_MULT)
        except (TypeError, ValueError):
            pass
    # Parabolic arc rides smaller — see PARABOLIC_ACCEL_MIN in config.py.
    if PARABOLIC_SIZE_MULT != 1.0:
        try:
            if _fld(setup, "accel_ratio", 1.0) >= PARABOLIC_ACCEL_MIN:
                m *= float(PARABOLIC_SIZE_MULT)
        except (TypeError, ValueError):
            pass
    # A long bought while RSI is already at the top of its corridor rides
    # smaller — see RSI_STRETCH_LONG_MIN in config.py. A missing rsi defaults
    # BELOW the threshold, so an absent field means no trim rather than one
    # applied on no evidence.
    if (RSI_STRETCH_SIZE_MULT != 1.0
            and str(setup.get("direction") or "").upper() == "LONG"):
        try:
            if _fld(setup, "rsi", 0.0) >= RSI_STRETCH_LONG_MIN:
                m *= float(RSI_STRETCH_SIZE_MULT)
        except (TypeError, ValueError):
            pass
    # Open space rides smaller — see OPEN_SPACE_ROOM_MIN in config.py.
    # NOTE: there is no "room_atr" key on the setup. The exported column of
    # that name is DERIVED at trade-record time by picking overhead_atr for a
    # long and underfoot_atr for a short. Reading setup["room_atr"] returns
    # None, and the first version of this rule did exactly that: six runs
    # across two multipliers came back byte-identical to the base, which is
    # the signature of a knob that never reached the code.
    if OPEN_SPACE_SIZE_MULT != 1.0:
        try:
            _dir = str(setup.get("direction") or "").upper()
            _rm = setup.get("overhead_atr") if _dir == "LONG" else setup.get("underfoot_atr")
            if _rm is not None and float(_rm) >= OPEN_SPACE_ROOM_MIN:
                m *= float(OPEN_SPACE_SIZE_MULT)
        except (TypeError, ValueError):
            pass
    # Active chop rides bigger — see CHOP_SIZE_MULT in config.py.
    if CHOP_SIZE_MULT != 1.0:
        try:
            if (_fld(setup, "volume_ratio", 0.0) >= CHOP_VOL_MIN
                    and _fld(setup, "eff_ratio", 99.0) < CHOP_EFF_MAX
                    and _fld(setup, "vol_atr_pct", 0.0) >= CHOP_ATR_MIN):
                m *= float(CHOP_SIZE_MULT)
        except (TypeError, ValueError):
            pass
    return min(m, float(SIZE_MULT_MAX))


def _post_tp1_trail_mult_bt(direction: str, entry: float, tp1: float, tp2: float,
                            high: float, low: float, close: float) -> float:
    """Context-aware runner trail from the TP1 candle (mirrors live _post_tp1_trail_mult)."""
    base = max(0.0, float(TRAIL_ATR_MULT))
    if str(EXIT_PROFILE).lower() != "post_tp1_v2":
        return base
    leg = abs(float(tp2) - float(tp1))
    if leg <= 0:
        return base
    if str(direction).upper() == "LONG":
        close_progress = (float(close) - float(tp1)) / leg
        wick_progress = (float(high) - float(tp1)) / leg
        failed_close = float(close) < float(tp1)
    else:
        close_progress = (float(tp1) - float(close)) / leg
        wick_progress = (float(tp1) - float(low)) / leg
        failed_close = float(close) > float(tp1)
    if close_progress >= POST_TP1_STRONG_CLOSE_PROGRESS or wick_progress >= POST_TP1_STRONG_WICK_PROGRESS:
        return max(base, float(POST_TP1_STRONG_TRAIL_ATR_MULT))
    if failed_close or close_progress <= POST_TP1_WEAK_CLOSE_PROGRESS:
        return min(base, float(POST_TP1_WEAK_TRAIL_ATR_MULT))
    return base


# Close-confirmed stop — mirrors the live setting so backtest and live stay in
# parity (config already applies the STOP_CLOSE_CONFIRM env override). Set
# STOP_CLOSE_CONFIRM=0 to measure the old wick-touch stop as a baseline.
# A time-stop ("scratch the trade if it hasn't moved after N bars") was tested
# here on 2026-07-26 and REJECTED: 32 bars gave WR 80.8%→71.1% and netR
# +904→+850 on the 6-month window — it scratches trades that would have won.
from config import STOP_CLOSE_CONFIRM as _STOP_CLOSE_CONFIRM

# 🔴 CORRECTED 2026-08-29, hours after being added with the wrong default.
#
# The claim was: an autotraded position is closed by the exchange OCO on a TOUCH
# of the stop, so modelling close-confirmation describes a bot the account does
# not run. That is WRONG. src/autotrader.py deliberately moves the exchange stop
# OUT to STOP_EXCHANGE_BACKSTOP_R (1.5R crypto / 2.5R stocks) measured from the
# real fill, precisely so it cannot fire on the wicks the close-confirmed rule
# exists to ignore. Its own comment says so: the exchange leg is a disaster
# backstop for when the bot is down, not the working stop. The working stop is
# the monitor's, and it IS close-confirmed.
#
# So STOP_CLOSE_CONFIRM=1 is right for an autotrading account and the default
# here goes back to OFF. Kept as a research handle: BT_EXCHANGE_STOP=1 models
# touch-triggered stops, which is what the book would look like if the backstop
# were ever moved onto the signal level.
_BT_EXCHANGE_STOP = os.getenv("BT_EXCHANGE_STOP", "0") != "0"
_STOP_ON_CLOSE = _STOP_CLOSE_CONFIRM and not _BT_EXCHANGE_STOP

# Research only, default OFF. When a single bar satisfies BOTH the stop and a
# target, the loop below resolves it as a stop because that check comes first.
# 15m OHLC does not record which level was touched first, so that ordering is a
# convention, not data — a pessimistic one. Set BT_TP_FIRST=1 to flip it to the
# optimistic convention; the gap between the two runs is the size of the
# uncertainty this convention hides. Measured 2026-08-13 after a spike-strategy
# test where 65 of 87 trades were decided by tie-break alone.
_BT_TP_FIRST = os.getenv("BT_TP_FIRST", "0") == "1"

# DEFAULT ON (the "research only, default OFF" this comment used to carry was
# copied from the flag above and was wrong — the code has always defaulted to
# "1"). Anchors the runner trail to prior bars only, so a trail exit can never
# be filled off the same bar that printed the peak. Load-bearing: without it a
# trail narrower than the average bar range pays out on essentially every bar,
# which is exactly the regime the trail now runs in (0.02 ATR).
_BT_TRAIL_LAG = os.getenv("BT_TRAIL_LAG", "1") == "1"

# Research flag, default 0 = OFF. Hours after which an open position is closed
# at market. Added 2026-08-29 to answer "is holding past 12h worth it". Answer:
# yes, holding is worth it — every horizon tested is worse.
#   stop   2023 profit  ratios      2024 profit  ratios      2026 profit  ratios
#    off   +148.46R  23.2/53.7      +206.87R  74.5/102.3     +416.45R  87.2/230.9
#    24h   +151.36R  21.2/59.3      +187.34R  28.7/75.3      +416.85R  88.8/238.6
#    12h                                                     +352.74R  32.2/129.7
#     6h                                                     +295.55R  27.3/109.0
# 12h costs 15% of profit and halves both risk ratios; 6h is worse still. 24h is
# flat-to-positive on the current window and MIXED on 2023, but 2024 rejects it
# outright: -9.4% profit and the worst-windows ratio falls 74.5 -> 28.7.
#
# Duration decays monotonically in the book (1-3h +0.582 R/trade, 6-12h +0.137,
# 24-48h -0.103), which is what made a time stop look obvious. It is not: a
# trade still open at 24h is one that has not resolved, so closing it banks
# whatever it is worth then — usually near breakeven — while forfeiting the tail
# of winners that resolve late. Note also that the trade count RISES with a time
# stop (+31/+33/+64) purely because closing frees slots under
# MAX_SAME_DIRECTION_POSITIONS; that is not new signal, it is capital turnover.
#
# ⚠️ Do not re-derive this from a "keep only trades under N hours" filter on an
# export. That deletes long trades rather than closing them at N hours, and so
# assumes their eventual outcome was knowable at the cut. It reads +9R where the
# real simulation reads -19R on the same window.
# 🔴 SWEPT AND REJECTED 2026-09-05, with the hours recorded so nobody repeats it.
# The motivation is real and worth restating, because it will look compelling
# again: expected unit R decays monotonically with time in trade, conditional on
# the trade STILL BEING OPEN — which is observable live, not hindsight.
#
#   всё ещё открыта   2026     2024     2023
#   на входе         +0.344   +0.265   +0.217
#   через 4 часа     +0.150   +0.075   -0.009
#   через 8 часов    +0.003   +0.037   -0.106
#   через 12 часов   -0.032   +0.013   -0.176
#
# Winners resolve in a median 15-16 bars, stops in 30-36; two hours in, a third
# of the winners have closed against 8-12% of the stops.
#
# And yet closing on it does not pay:
#
#   вариант   2026            2024            2023
#   база     +435.84 pd 56.2  +226.17 pd 35.8  +189.70 pd 26.0
#   6 часов  +390.21 pd 49.2  +185.46 pd 25.8  +164.93 pd 24.0
#   8 часов  +426.35 pd 52.5  +201.39 pd 26.0  +195.38 pd 25.7
#   12 часов +445.10 pd 63.1  +197.07 pd 28.4  +182.73 pd 28.0
#
# 6h and 8h lose on every measure; 12h wins one window and costs profit in the
# other two. The reason the curve misleads: the trades still open at hour N are
# not all dying — many are IN PROFIT and trailing, and a time exit cuts those
# together with the hopeless ones. The decay is real; it just cannot be
# harvested by closing on the clock.
_BT_TIME_STOP_H = float(os.getenv("BT_TIME_STOP_H", "0"))

# Average-down research flag, default OFF. When set, a SECOND unit of the same
# size is added by resting limit order once price trades BT_AVG_DOWN_R risk
# units against the entry, before TP1. The stop stays where structure put it,
# so total risk is 1.0 + (1.0 - BT_AVG_DOWN_R) units and every R below is
# reported per unit of that combined risk — otherwise the variant would just be
# betting more money and would not be comparable.
# The fill level sits BETWEEN entry and stop, so on any bar that reaches the
# stop, price has already passed the add level: unlike a TP-vs-SL tie this
# ordering is forced by geometry, not assumed. Fills are modelled AT the level
# (backtest.py carries no opens, so a gap through it is filled optimistically).
_BT_AVG_DOWN_R = float(os.getenv("BT_AVG_DOWN_R", "0") or 0)

# Structure-invalidation exit, research flag, default OFF. Exits at the close
# when price closes back BEYOND the level whose break justified the entry —
# i.e. the reason for being in the trade has gone, which is what a person means
# by "it obviously broke, why are we still in it".
# This is a different test from the MAE one (2026-08-24), which asked how DEEP
# the trade had gone and found no information below the stop. Depth is not
# structure.
_BT_STRUCT_EXIT = os.getenv("BT_STRUCT_EXIT", "0") == "1"

# Mirror mode, research flag, default OFF. Trades every setup in the OPPOSITE
# direction with the bracket reflected through the entry, to answer "would this
# have reached the target faster the other way". The mirror experiment in db.py
# only ever ran on REJECTED setups; this runs it on the whole book.
_BT_MIRROR = os.getenv("BT_MIRROR", "0") == "1"

# Two-stage trail, research flag. The trail is deliberately tight (0.05 ATR)
# because that measured best — but the cost is visible in the exit distribution:
# TP1 arms the trail at +0.6R and the median trail exit is +0.65R, so the runner
# contributes five hundredths of an R. Wins average +0.65R against -1.0R losses,
# and NO trail exit in the book reaches +2.0R.
# The question this tests is whether a trade that has already proved itself
# deserves more room: keep the tight trail until BT_TRAIL_STAGE_R, then widen to
# BT_TRAIL_STAGE_MULT. 0 = off.
_BT_TRAIL_STAGE_R    = float(os.getenv("BT_TRAIL_STAGE_R", "0") or 0)
_BT_TRAIL_STAGE_MULT = float(os.getenv("BT_TRAIL_STAGE_MULT", "0.35") or 0.35)

# Research only, both default OFF. Conditional stale exit: once a trade is at
# least BT_STALE_EXIT_BARS old and STILL has not reached TP1, close it if it is
# currently underwater by more than BT_STALE_EXIT_MAX_R.
#
# This is NOT the flat time-stop rejected on 2026-07-26 (scratch everything at 32
# bars: WR 80.8->71.1%, netR +904->+850). That one killed trades that were merely
# slow, most of which still won. This one only scratches trades that are slow AND
# losing, which is a different population: measured 2026-08-16, trades lasting
# 48+ bars run 42.9% stops and -0.090R/trade while trades under 4 bars run 0.1%
# stops and +0.722R. Duration is unknowable at entry, so the only place to act on
# it is mid-trade.
# ── Execution model: match what the live bot actually does ────────────────────
# Until 2026-08-23 the backtest filled every setup at the ZONE MIDPOINT on the
# signal bar, without requiring that price ever traded there. Live does not do
# that: zone-watch parks an approved setup and fires only when price genuinely
# re-enters [entry_low, entry_high] within ZONE_WATCH_MINUTES, at whatever the
# market shows — which on approach is the near EDGE of the zone.
#
# Isolated on 18k candles, 2x2:
#     A середина, без ожидания   1819 сд  84.8%  +955.07R  DD -10.22R
#     B край,     без ожидания   1819 сд  81.7%  +772.62R  DD -18.24R
#     C середина, с ожиданием     738 сд  73.3%  +164.77R  DD -14.74R
#     D край,     с ожиданием     998 сд  73.2%  +203.31R  DD -24.22R
#
# The fill PRICE costs 3.1pp of win rate and 19% of profit (A->B). The WAIT
# costs 11.5pp and 83%, because 59% of setups never come back (A->C). The wait
# is the whole story; the price is a footnote.
#
# D is the live model and is now the default, so the headline stops describing
# an execution nobody has. Note D beats C: given that we wait, filling at the
# edge yields MORE trades and MORE profit than holding out for the midpoint —
# so the live behaviour is already the best of the realistic options, and the
# "enter deeper into the zone" idea is dead.
#
# ⚠️ Every figure recorded before this date used model A. Relative comparisons
# between variants survive (they all ran at A), but absolute levels do not.
#
# BT_ZONE_DEPTH=-1 and BT_LIMIT_WAIT_BARS=0 restore the old behaviour.
_DEFAULT_ZONE_DEPTH = 0.0 if ZONE_WATCH_ENABLED else -1.0
_DEFAULT_WAIT_BARS = (
    max(1, int(ZONE_WATCH_MINUTES * 60 // (KLINES_INTERVAL_SEC or 900)))
    if ZONE_WATCH_ENABLED else 0
)

# 🔴 SWEPT AND REJECTED 2026-09-05. This is the refined version of the time
# stop above: close a trade only if it has hung for N bars AND is STILL
# underwater by more than X R, so the winners that are trailing are left alone.
# That reasoning is sound and the screening window agreed with it — there is a
# clean ladder where later and deeper is better (2026-08-26, base 1121 trades,
# +435.84R, DD -7.76, pd 56.2):
#
#   16 баров / -0.3R  1356сд  +388.44R  DD-7.96  pd 48.8
#   24 / -0.5         1256сд  +416.14R  DD-8.04  pd 51.8
#   32 / -0.5         1239сд  +449.76R  DD-8.12  pd 55.4
#   32 / -0.7         1189сд  +453.28R  DD-8.30  pd 54.6
#   48 / -0.7         1170сд  +442.67R  DD-7.67  pd 57.7   <- beats base on both
#
# And it does not survive the other windows:
#
#   48 / -0.7   2024  903сд +218.10R DD-6.80 pd 32.1  (база +226.17 -6.32 35.8)
#   48 / -0.7   2023  724сд +191.85R DD-7.54 pd 25.5  (база +189.70 -7.29 26.0)
#
# Better on 2026, worse on 2024, worse on 2023. Pushing further out (64 bars)
# improves 2026 again to +447.94R / DD-7.38 / pd 60.7 — which is the shape of
# fitting one window, not of an edge, so the sweep stopped there.
#
# Note -1.0R reproduces the baseline exactly: the stop already sits at -1R, so
# a trade cannot be that far underwater and still be open. The usable range is
# -0.5 to -0.9 and it was covered.
_BT_STALE_BARS  = int(os.getenv("BT_STALE_EXIT_BARS", "0") or 0)
_BT_STALE_MAX_R = float(os.getenv("BT_STALE_EXIT_MAX_R", "0") or 0.0)


def _r_from_price(entry: float, exit_px: float, sl: float, direction: str) -> float:
    """Actual R of an exit at an arbitrary price (pre-TP1, full position open).

    Needed because gross_r_for_outcome() hardcodes SL = -1.0R, which is only
    true when the exit really happens AT the stop level. A close-confirmed stop
    exits at the candle CLOSE, which can be well beyond the level — counting
    that as -1.0R would invent an edge that does not exist.
    """
    risk = abs(entry - sl)
    if risk <= 0:
        return 0.0
    return ((exit_px - entry) if direction == "LONG" else (entry - exit_px)) / risk


def gross_r_for_outcome(outcome: str, entry: float, tp1: float, tp2: float, sl: float) -> float:
    risk = abs(entry - sl)
    if risk <= 0:
        return 0.0

    tp1_r = abs(tp1 - entry) / risk
    tp2_r = abs(tp2 - entry) / risk

    if outcome == "TP2":
        return _TP1_CLOSE_FRAC * tp1_r + _RUNNER_FRAC * tp2_r
    if outcome == "TP1":
        return _TP1_CLOSE_FRAC * tp1_r
    if outcome == "SL":
        return -1.0
    if outcome == "BE":
        return 0.0
    return 0.0


def gross_r_for_trailing_exit(entry: float, tp1: float, trail_exit: float, sl: float, direction: str) -> float:
    risk = abs(entry - sl)
    if risk <= 0:
        return 0.0
    tp1_r = abs(tp1 - entry) / risk
    if direction == "LONG":
        trail_r = (trail_exit - entry) / risk
    else:
        trail_r = (entry - trail_exit) / risk
    return _TP1_CLOSE_FRAC * tp1_r + _RUNNER_FRAC * max(0.0, trail_r)


def execution_fill_price(
    direction: str,
    planned_entry: float,
    candles_15m: dict[str, list],
    entry_bar: int,
    delay_bars: int,
    adverse_bps: float,
) -> tuple[float, int]:
    fill_bar = min(max(entry_bar, entry_bar + max(0, delay_bars)), len(candles_15m["close"]) - 1)
    price = planned_entry if delay_bars <= 0 else float(candles_15m["close"][fill_bar])
    adverse = adverse_bps / 10_000.0
    if direction == "LONG":
        price *= 1.0 + adverse
    else:
        price *= 1.0 - adverse
    return price, fill_bar


def estimate_cost_r(entry: float, sl: float, fee_rate: float, slippage_rate: float) -> float:
    risk = abs(entry - sl)
    if entry <= 0 or risk <= 0:
        return 0.0
    round_trip_cost_pct = 2.0 * (fee_rate + slippage_rate)
    return round_trip_cost_pct * entry / risk


@dataclass
class TradeRecord:
    symbol: str
    entry_bar: int
    exit_bar: int
    entry_time: int | None
    exit_time: int | None
    direction: str
    outcome: str
    entry: float
    tp1: float
    tp2: float
    sl: float
    gross_r: float
    net_r: float
    cost_r: float
    mtf_score: int = 0
    volume_ratio: float = 0.0
    rsi: float = 0.0
    eff_ratio: float = 0.0
    vol_atr_pct: float = 0.0
    vol_ratio_regime: float = 0.0
    adaptive_pack: str = ""
    adaptive_reason: str = ""
    risk_mult: float = 1.0
    quality_score: float = 0.0
    trend_score: int = 0
    volatility_score: int = 0
    entry_quality_score: int = 0
    portfolio_risk_score: int = 0
    session: str = ""
    trend_1h: str = ""
    trend_4h: str = ""
    entry_source: str = ""
    signals: str = ""
    score_tags: str = ""
    premium: int = 0
    sniper: int = 0   # label only, never a filter (config.py SNIPER_TAG_ENABLED)
    knn_score: float = -1.0
    swing_trend: str = ""  # 15m structure (bull/bear/range) — feeds Claude memory seeding
    # Worst adverse excursion while the trade was open, in R, measured on WICKS.
    # The engine stops on a candle CLOSE beyond 1R, but the exchange backstop is
    # a plain trigger at STOP_EXCHANGE_BACKSTOP_R — a wick that deep fires it for
    # real while the bot still believes the trade is alive. This column is how
    # that divergence gets counted instead of assumed away.
    # What price did AFTER the stop, recorded during the run so the candles and
    # the trade are the same object — matching an export back to cached candles
    # afterwards was tried and failed (3.5% of bar indices and 23.7% of
    # timestamps landed on the right bar), which is why this lives here.
    #   post_sl_tp1  1 if TP1 was reached within the normal window after the stop
    #   post_sl_max_r how far past the stop price ran, in units of the trade's risk
    # Together they separate "the stop stood in noise" from "the entry was wrong".
    # Shape of the price action AT the entry, measured from the bars themselves.
    # The existing columns are all derived INDICATORS; none of them describes
    # what the chart actually looked like, which is why a feature sweep kept
    # coming up empty on "is this a stupid entry".
    #   pullback_frac  how far price retraced the recent swing before we bought.
    #                  Near 0 = buying a shallow dip after a big run (a chase),
    #                  near 1 = buying near the base of the move.
    #   run_len_before consecutive bars closing OUR way immediately before entry
    #   impulse6_atr   size of the last six bars' range, in ATR
    #
    # 🔴 ALL FIVE MEASURED AND NULL, 2026-09-05. Kept because their absence is
    # the finding: the shape of the chart at entry does not separate winners
    # from losers on this book, which is why every indicator sweep came up
    # empty too. Unit R by window (2023 / 2024 / 2026):
    #
    #   pullback_frac <0.20 (buying right under the swing high, the "chase")
    #        +0.128 / -0.036 / +0.017 against the book — windows disagree
    #   pullback_frac >0.60 (buying near the base)
    #        -0.005 / -0.003 / -0.147 — if anything worse
    #   run_len_before 0 bars   +0.18 / +0.28 / +0.31
    #                  4+ bars  +0.24 / +0.11 / +0.33 — no order
    #   impulse6_atr — windows disagree on which end is better
    #   retrace_bars_ratio — degenerate, 97% of trades fall in one bucket
    #   pullback_vol_ratio — the textbook "quiet pullback is healthy" reverses:
    #        quiet <0.7 runs -0.199 in 2026, loud 1.0-1.5 runs +0.050 in 2024
    #
    # Reading: local 15m "chasing" costs nothing. What does predict, in every
    # window, is how extended the HIGHER timeframes are — see
    # HTF_FULL_ALIGN_SKIP in config.py. Same conclusion reached from the
    # opposite direction, which is why it is worth trusting.
    #   retrace_bars_ratio  bars spent retracing / bars spent on the impulse.
    #                  Above 1 = the pullback is taking LONGER than the move it
    #                  is correcting, which is opposition, not a pause.
    #   pullback_vol_ratio  average volume while retracing / average volume of
    #                  the impulse. The textbook healthy pullback is quiet; a
    #                  loud one is distribution.
    retrace_bars_ratio: float = -1.0
    pullback_vol_ratio: float = -1.0
    pullback_frac: float = -1.0
    run_len_before: int = 0
    impulse6_atr: float = 0.0
    post_sl_tp1: int = -1
    post_sl_max_r: float = 0.0
    mae_r: float = 0.0
    mfe_tp1: float = 0.0   # доля пути до TP1, пройденная в лучшей точке
    accel_ratio: float = 1.0   # ускорение движения: >1 = параболическая дуга
    buy_pressure: float = 0.0  # давление покупателей внутри баров, [-1,+1]
    absorption: float = 0.0    # усилие/результат по Вайкоффу: объём на единицу хода
    obv_agree: float = 0.0
    obv_strength: float = 0.0
    # Live per-symbol position-size trim (config.SYMBOL_SIZE_MULT). Deliberately
    # not merged into risk_mult — see the construction site for why.
    size_mult: float = 1.0
    # Microstructure at the moment of entry, added 2026-08-25 to answer "in
    # which scenario are stops more likely". The filter already computes the
    # first three and threw them away at the export boundary.
    zone_age_bars: int = -1      # how stale the zone was when price returned
    bos_candles_ago: int = -1    # how long ago structure actually broke
    room_atr: float = -1.0       # distance to the nearest HTF level IN OUR WAY
    tp1_beyond_level: int = 0    # 1 = TP1 sits past that level, i.e. through a wall
    extension_atr: float = 0.0   # distance from the BOS level to entry, in ATR
    entry_range_atr: float = 0.0 # range of the entry bar itself, in ATR
    run_bars: int = 0            # consecutive same-direction closes before entry
    # Scan bar that produced the setup. entry_bar is the FILL bar, which under
    # the zone-wait model can be several bars later — so entry_bar/entry_time
    # cannot identify a setup across execution models, and joining on them
    # silently mismatches. This is the stable key.
    signal_bar: int = -1


@dataclass
class SymbolResult:
    symbol: str
    bars: int = 0
    scanned: int = 0
    prefiltered: int = 0
    analyzed: int = 0
    trades: int = 0
    tp1: int = 0
    tp2: int = 0
    sl: int = 0
    expired: int = 0
    gross_r: float = 0.0
    net_r: float = 0.0
    elapsed_sec: float = 0.0
    error: str | None = None
    trade_records: list[TradeRecord] = field(default_factory=list)


def simulate_trade_direct(
    symbol: str,
    setup: dict,
    candles_15m: dict[str, list],
    entry_bar: int,
    window: int,
    fee_rate: float,
    slippage_rate: float,
    execution_delay_bars: int = 0,
    adverse_entry_bps: float = 0.0,
    exit_policy: str = "classic",
    trail_atr_mult: float = 0.75,
) -> TradeRecord:
    direction = setup["direction"]
    planned_entry = float(setup["current_price"])
    entry, fill_bar = execution_fill_price(
        direction,
        planned_entry,
        candles_15m,
        entry_bar,
        execution_delay_bars,
        adverse_entry_bps,
    )
    # 🔴 The measurement that motivated this switch was WRONG, and the switch is
    # kept only as a research handle. The claim was that live fills land ~0.37%
    # (stocks) / ~0.22% (crypto) worse than the model's, costing 80% / 42% of
    # profit. What was actually measured is the fill against the ZONE MIDPOINT —
    # and the model does not fill at the zone either. planned_entry is
    # setup["current_price"], the price at signal time, which is what the live
    # bot records as its fill too. The offset from the zone is a property of the
    # strategy that BOTH sides already carry, so feeding it back through
    # --adverse-entry-bps counted it twice.
    #
    # What IS true and was verified: execution slip is 0.000% on both desks —
    # the recorded fill equals the recorded market price to the digit. There is
    # nothing for a limit order to recover.
    #
    # The residual gap that cannot be measured from the exports is the drift
    # between the bar CLOSE the model fills at and the market price ~2 minutes
    # later when the order lands. Bounding that needs the bar close stored on
    # the signal, which is not logged today.
    #
    # BT_ADVERSE_KEEP_LEVELS remains correct in itself: if an adverse fill is
    # ever modelled, the levels must NOT move with it, because live computes
    # them at scan time and stores them.
    _lvl_src = (planned_entry
                if os.getenv("BT_ADVERSE_KEEP_LEVELS", "0") == "1" else entry)
    tp1, tp2, sl = calculate_tp_sl_local(
        _lvl_src,
        direction,
        atr=setup.get("atr", 0.0),
        recent_high=setup.get("recent_high", 0.0),
        recent_low=setup.get("recent_low", 0.0),
        tp1_level=setup.get("tp1_level"),
        tp2_level=setup.get("tp2_level"),
    )

    highs = candles_15m["high"]
    lows = candles_15m["low"]
    closes = candles_15m["close"]
    times = candles_15m.get("time") or []
    end = min(fill_bar + window, len(highs))
    outcome = "EXPIRED"
    tp1_reached = False
    closed = False
    exit_bar = max(fill_bar, end - 1)
    trailing_stop = entry
    trail_exit_price = entry
    best_price = entry
    if _BT_MIRROR:
        # Reflect the bracket through the entry and flip the side.
        direction = "SHORT" if direction == "LONG" else "LONG"
        tp1 = entry - (tp1 - entry)
        tp2 = entry - (tp2 - entry)
        sl  = entry - (sl - entry)

    trail_mult_eff = max(0.0, float(trail_atr_mult))  # context-frozen at TP1 candle

    stop_exit_price = None  # set when we exit at a price other than the SL level
    add_price = None        # fill price of the averaging-down unit, when armed
    try:
        _bos_lvl_num = float(setup.get("bos_break_level"))
    except (TypeError, ValueError):
        _bos_lvl_num = None
    _risk_abs = abs(entry - sl)
    # --- entry-context candle shape (recorded only, changes no outcome) ---
    _o = candles_15m.get("open") or []
    _h_all, _l_all, _c_all = candles_15m["high"], candles_15m["low"], candles_15m["close"]
    _atr_e = float(setup.get("atr", 0.0) or 0.0)
    _lo_i = max(0, fill_bar - 20)
    _swing_hi = max(_h_all[_lo_i:fill_bar]) if fill_bar > _lo_i else entry
    _swing_lo = min(_l_all[_lo_i:fill_bar]) if fill_bar > _lo_i else entry
    _span = _swing_hi - _swing_lo
    if _span > 0:
        _pullback_frac = ((_swing_hi - entry) / _span if direction == "LONG"
                          else (entry - _swing_lo) / _span)
        _pullback_frac = max(0.0, min(1.0, _pullback_frac))
    else:
        _pullback_frac = -1.0
    _run_len = 0
    for _k in range(fill_bar - 1, max(-1, fill_bar - 12), -1):
        if _k <= 0 or _k >= len(_c_all): break
        _up = _c_all[_k] > _c_all[_k - 1]
        if _up == (direction == "LONG"): _run_len += 1
        else: break
    _i6 = max(0, fill_bar - 6)
    _impulse6 = ((max(_h_all[_i6:fill_bar]) - min(_l_all[_i6:fill_bar])) / _atr_e
                 if fill_bar > _i6 and _atr_e > 0 else 0.0)
    # Where the swing extreme sits tells us how long the retrace has run
    # against how long the impulse took, and on what volume.
    _retrace_ratio = -1.0
    _pb_vol_ratio = -1.0
    if fill_bar > _lo_i:
        _seg_h = _h_all[_lo_i:fill_bar]
        _seg_l = _l_all[_lo_i:fill_bar]
        _ext_i = (_lo_i + _seg_h.index(max(_seg_h))) if direction == "LONG"             else (_lo_i + _seg_l.index(min(_seg_l)))
        _retr_bars = fill_bar - _ext_i
        _imp_bars = _ext_i - _lo_i
        if _imp_bars > 0 and _retr_bars > 0:
            _retrace_ratio = _retr_bars / _imp_bars
            _vol = candles_15m.get("volume") or []
            if len(_vol) > fill_bar:
                _iv = _vol[_lo_i:_ext_i]; _rv = _vol[_ext_i:fill_bar]
                _im = sum(_iv) / len(_iv) if _iv else 0.0
                _rm = sum(_rv) / len(_rv) if _rv else 0.0
                if _im > 0: _pb_vol_ratio = _rm / _im
    _mae_r = 0.0
    # Best move TOWARD the target before the trade resolved, as a fraction of
    # the distance to TP1. 1.0 means it touched TP1; 0.98 means it stopped two
    # percent short. Recorded to answer a specific question: how often does a
    # trade die after coming within a whisker of its target?
    _mfe_tp1 = 0.0
    _tp1_dist = abs(tp1 - entry)
    for j in range(fill_bar, end):
        h = highs[j]
        l = lows[j]
        # Time stop (research flag, default 0 = off). Closes at the bar CLOSE
        # once the position has been open N hours. Note this is a real exit at
        # the price then trading — NOT the same as deleting long trades from the
        # book, which is what a naive "keep only trades under N hours" filter
        # does and which silently assumes the eventual outcome was knowable at
        # the cut.
        if _BT_TIME_STOP_H > 0 and (j - fill_bar) * (KLINES_INTERVAL_SEC or 900) >= _BT_TIME_STOP_H * 3600:
            outcome = "TIME"
            stop_exit_price = closes[j]
            exit_bar = j
            closed = True
            break
        if not tp1_reached:
            # Conditional stale exit (research flag, default off) — see above.
            if _BT_STALE_BARS > 0 and (j - fill_bar) >= _BT_STALE_BARS and _risk_abs > 0:
                _unreal = ((closes[j] - entry) if direction == "LONG"
                           else (entry - closes[j])) / _risk_abs
                if _unreal < -abs(_BT_STALE_MAX_R):
                    outcome = "STALE"
                    stop_exit_price = closes[j]
                    exit_bar = j
                    closed = True
                    break
            # Only the pre-TP1 phase counts: once TP1 prints, the autotrader
            # amends the exchange stop to breakeven, so the 2R backstop is no
            # longer the thing that can fire behind the engine's back.
            if _risk_abs > 0:
                _adv = (entry - l) if direction == "LONG" else (h - entry)
                if _adv / _risk_abs > _mae_r:
                    _mae_r = _adv / _risk_abs
            if _tp1_dist > 0:
                _fav = (h - entry) if direction == "LONG" else (entry - l)
                if _fav / _tp1_dist > _mfe_tp1:
                    _mfe_tp1 = _fav / _tp1_dist
            # Early breakeven — see BE_ARM_PROGRESS in config.py. Armed once the
            # trade has travelled far enough toward TP1; from then on the stop
            # sits at entry, which is TIGHTER than the original, so it is
            # checked BEFORE the normal stop.
            if BE_ARM_PROGRESS > 0 and _mfe_tp1 >= BE_ARM_PROGRESS:
                _be_hit = ((closes[j] <= entry) if _STOP_ON_CLOSE else (l <= entry))                     if direction == "LONG" else                     ((closes[j] >= entry) if _STOP_ON_CLOSE else (h >= entry))
                if _be_hit:
                    outcome = "BE"
                    stop_exit_price = entry
                    exit_bar = j
                    closed = True
                    break
            if _BT_STRUCT_EXIT and _bos_lvl_num is not None:
                _broke_back = (closes[j] < _bos_lvl_num) if direction == "LONG"                     else (closes[j] > _bos_lvl_num)
                if _broke_back:
                    outcome = "STRUCT"
                    stop_exit_price = closes[j]
                    exit_bar = j
                    closed = True
                    break
            if _BT_AVG_DOWN_R > 0 and add_price is None and _risk_abs > 0:
                _lvl = (entry - _risk_abs * _BT_AVG_DOWN_R) if direction == "LONG"                     else (entry + _risk_abs * _BT_AVG_DOWN_R)
                if (l <= _lvl) if direction == "LONG" else (h >= _lvl):
                    add_price = _lvl
            _stop_hit = ((closes[j] <= sl) if _STOP_ON_CLOSE else (l <= sl)) if direction == "LONG" \
                else ((closes[j] >= sl) if _STOP_ON_CLOSE else (h >= sl))
            _tgt_hit = (h >= tp1) if direction == "LONG" else (l <= tp1)
            if _stop_hit and _tgt_hit:
                globals()["_BT_AMBIGUOUS"] = globals().get("_BT_AMBIGUOUS", 0) + 1
            if _BT_TP_FIRST:
                # optimistic tie-break: a bar that reaches a target counts as the
                # target even if it also satisfies the stop
                if direction == "LONG" and h >= tp1:
                    if h >= tp2:
                        outcome = "TP2"
                        exit_bar = j
                        closed = True
                        break
                    outcome = "TP1"
                    tp1_reached = True
                    exit_bar = j
                    trail_mult_eff = _post_tp1_trail_mult_bt(direction, entry, tp1, tp2, h, l, closes[j])
                    continue
                if direction == "SHORT" and l <= tp1:
                    if l <= tp2:
                        outcome = "TP2"
                        exit_bar = j
                        closed = True
                        break
                    outcome = "TP1"
                    tp1_reached = True
                    exit_bar = j
                    trail_mult_eff = _post_tp1_trail_mult_bt(direction, entry, tp1, tp2, h, l, closes[j])
                    continue
            if direction == "LONG":
                if (closes[j] <= sl) if _STOP_ON_CLOSE else (l <= sl):
                    outcome = "SL"
                    if _STOP_ON_CLOSE:
                        stop_exit_price = closes[j]
                    exit_bar = j
                    closed = True
                    break
                if h >= tp2:
                    outcome = "TP2"
                    exit_bar = j
                    closed = True
                    break
                if h >= tp1:
                    outcome = "TP1"
                    tp1_reached = True
                    exit_bar = j
                    trail_mult_eff = _post_tp1_trail_mult_bt(direction, entry, tp1, tp2, h, l, closes[j])
                    continue
            else:
                if (closes[j] >= sl) if _STOP_ON_CLOSE else (h >= sl):
                    outcome = "SL"
                    if _STOP_ON_CLOSE:
                        stop_exit_price = closes[j]
                    exit_bar = j
                    closed = True
                    break
                if l <= tp2:
                    outcome = "TP2"
                    exit_bar = j
                    closed = True
                    break
                if l <= tp1:
                    outcome = "TP1"
                    tp1_reached = True
                    exit_bar = j
                    trail_mult_eff = _post_tp1_trail_mult_bt(direction, entry, tp1, tp2, h, l, closes[j])
                    continue
        else:
            if direction == "LONG":
                if exit_policy == "trail":
                    if _BT_TRAIL_STAGE_R > 0 and _risk_abs > 0:
                        _gain = (best_price - entry) / _risk_abs
                        if _gain >= _BT_TRAIL_STAGE_R:
                            trail_mult_eff = max(trail_mult_eff, _BT_TRAIL_STAGE_MULT)
                    # Under BT_TRAIL_LAG the trail is anchored to the peak of
                    # PRIOR bars only. Anchoring to this bar's own high and then
                    # testing this bar's low assumes the high printed before the
                    # low, which 15m OHLC does not record — an optimistic
                    # convention that pays out on every bar once the trail is
                    # narrower than the average bar range.
                    if not _BT_TRAIL_LAG:
                        best_price = max(best_price, h)
                    trailing_stop = max(entry, best_price - max(0.0, float(setup.get("atr", 0.0) or 0.0)) * trail_mult_eff)
                    if _BT_TRAIL_LAG:
                        best_price = max(best_price, h)
                    if l <= trailing_stop:
                        outcome = "TRAIL"
                        trail_exit_price = trailing_stop
                        exit_bar = j
                        closed = True
                        break
                if l <= entry:
                    outcome = "TP1"
                    exit_bar = j
                    closed = True
                    break
                if h >= tp2:
                    outcome = "TP2"
                    exit_bar = j
                    closed = True
                    break
            else:
                if exit_policy == "trail":
                    if _BT_TRAIL_STAGE_R > 0 and _risk_abs > 0:
                        _gain = (entry - best_price) / _risk_abs
                        if _gain >= _BT_TRAIL_STAGE_R:
                            trail_mult_eff = max(trail_mult_eff, _BT_TRAIL_STAGE_MULT)
                    if not _BT_TRAIL_LAG:
                        best_price = min(best_price, l)
                    trailing_stop = min(entry, best_price + max(0.0, float(setup.get("atr", 0.0) or 0.0)) * trail_mult_eff)
                    if _BT_TRAIL_LAG:
                        best_price = min(best_price, l)
                    if h >= trailing_stop:
                        outcome = "TRAIL"
                        trail_exit_price = trailing_stop
                        exit_bar = j
                        closed = True
                        break
                if h >= entry:
                    outcome = "TP1"
                    exit_bar = j
                    closed = True
                    break
                if l <= tp2:
                    outcome = "TP2"
                    exit_bar = j
                    closed = True
                    break

    if tp1_reached and outcome == "TP1" and not closed:
        exit_bar = max(fill_bar, end - 1)

    if outcome == "TRAIL":
        gross_r = gross_r_for_trailing_exit(entry, tp1, trail_exit_price, sl, direction)
    elif stop_exit_price is not None:
        # Close-confirmed stop or time-stop: full position still open, so R is
        # the real move to the exit price — NOT the -1.0R that a level-touch
        # stop would have booked. Can be worse than -1R (gap through the level).
        gross_r = _r_from_price(entry, stop_exit_price, sl, direction)
    else:
        gross_r = gross_r_for_outcome(outcome, entry, tp1, tp2, sl)
    cost_r = estimate_cost_r(entry, sl, fee_rate, slippage_rate)
    if add_price is not None and _risk_abs > 0:
        # Both units leave at the same price; the second simply entered closer
        # to it. Reported per unit of COMBINED risk so the comparison against a
        # single unit is like-for-like rather than "we bet more".
        _edge = ((entry - add_price) if direction == "LONG" else (add_price - entry)) / _risk_abs
        _total_risk = 1.0 + max(0.0, 1.0 - _edge)
        gross_r = (gross_r + (gross_r + _edge)) / _total_risk
        cost_r  = (cost_r * 2.0) / _total_risk
    # Post-stop walk-forward (research only; changes no outcome). Scans the same
    # `window` the trade itself was allowed, starting at the stop bar.
    _post_sl_tp1 = -1
    _post_sl_max_r = 0.0
    if outcome == "SL" and _risk_abs > 0:
        _post_sl_tp1 = 0
        for _j in range(exit_bar, min(exit_bar + 1 + window, len(highs))):
            _adv = (sl - lows[_j]) if direction == "LONG" else (highs[_j] - sl)
            if _adv > _post_sl_max_r * _risk_abs:
                _post_sl_max_r = _adv / _risk_abs
            if _j > exit_bar and ((highs[_j] >= tp1) if direction == "LONG"
                                  else (lows[_j] <= tp1)):
                _post_sl_tp1 = 1
                break

    net_r = gross_r - cost_r
    # Live position-size rules scale BOTH the win and the loss, so they belong
    # in the R the summary reports. Until 2026-08-24 size_mult was recorded on
    # the record but never applied, which meant no backtest figure in this
    # project reflected the BTC half-size trim or the counter-structure boost.
    # --- microstructure ---
    _atr = max(1e-12, float(setup.get("atr", 0.0) or 0.0))
    try:
        _ext = float(setup.get("bos_extension_atr") or 0.0)
    except (TypeError, ValueError):
        _ext = 0.0
    _b = max(0, min(fill_bar, len(highs) - 1))
    _rng = (highs[_b] - lows[_b]) / _atr if _b < len(highs) else 0.0
    _run = 0
    for _k in range(_b - 1, max(-1, _b - 13), -1):
        if _k <= 0 or _k >= len(closes):
            break
        _up = closes[_k] > closes[_k - 1]
        if (_up and direction == "LONG") or ((not _up) and direction == "SHORT"):
            _run += 1
        else:
            break

    # How much room to the nearest higher-timeframe level standing in our way,
    # and whether TP1 is on the far side of it. A target that requires breaking
    # a 4h swing is a different proposition from one in open air — the thing a
    # human reads off the chart instantly and the bot never measured.
    _room = setup.get("overhead_atr") if direction == "LONG" else setup.get("underfoot_atr")
    try:
        _room = float(_room) if _room is not None else -1.0
    except (TypeError, ValueError):
        _room = -1.0
    _tp1_atr = abs(float(tp1) - float(entry)) / _atr if _atr > 0 else 0.0
    _beyond = 1 if (0 <= _room < _tp1_atr) else 0

    _sz = _size_mult_for(symbol, setup)
    gross_r *= _sz
    net_r   *= _sz
    cost_r  *= _sz

    return TradeRecord(
        symbol=symbol,
        entry_bar=fill_bar,
        exit_bar=exit_bar,
        entry_time=times[fill_bar - 1] if 0 <= fill_bar - 1 < len(times) else None,
        exit_time=times[exit_bar] if 0 <= exit_bar < len(times) else None,
        direction=direction,
        outcome=outcome,
        entry=entry,
        tp1=tp1,
        tp2=tp2,
        sl=sl,
        gross_r=gross_r,
        net_r=net_r,
        cost_r=cost_r,
        mtf_score=int(setup.get("mtf_score", 0) or 0),
        volume_ratio=float(setup.get("volume_ratio", 0.0) or 0.0),
        rsi=float(setup.get("rsi", 0.0) or 0.0),
        eff_ratio=float(setup.get("eff_ratio", 0.0) or 0.0),
        vol_atr_pct=float(setup.get("vol_atr_pct", 0.0) or 0.0),
        vol_ratio_regime=float(setup.get("vol_ratio_regime", 0.0) or 0.0),
        adaptive_pack=str(setup.get("adaptive_pack", "") or ""),
        adaptive_reason=str(setup.get("adaptive_reason", "") or ""),
        risk_mult=float(setup.get("risk_mult", 1.0) or 1.0),
        # Kept SEPARATE from risk_mult on purpose. risk_mult is the kNN adaptive
        # multiplier, which the live autotrader does NOT apply (_margin_for
        # ignores it) — folding the two together would make every --use-risk-mult
        # analysis model a book the bot never trades. This column is the trim the
        # autotrader really applies, and nothing else.
        size_mult=_sz,
        quality_score=float(setup.get("quality_score", 0.0) or 0.0),
        trend_score=int(setup.get("trend_score", 0) or 0),
        volatility_score=int(setup.get("volatility_score", 0) or 0),
        entry_quality_score=int(setup.get("entry_quality_score", 0) or 0),
        portfolio_risk_score=int(setup.get("portfolio_risk_score", 0) or 0),
        session=str(setup.get("session", "") or ""),
        trend_1h=str(setup.get("trend_1h", "") or ""),
        trend_4h=str(setup.get("trend_4h", "") or ""),
        entry_source=str(setup.get("entry_source", "") or ""),
        signals=" | ".join(setup.get("signals", [])),
        score_tags=" | ".join(setup.get("score_tags", [])),
        premium=int(bool(setup.get("premium"))),
        sniper=int(bool(setup.get("sniper"))),
        knn_score=float(setup.get("_knn_score", -1.0)),
        swing_trend=str(setup.get("swing_trend", "") or ""),
        retrace_bars_ratio=round(_retrace_ratio, 4),
        pullback_vol_ratio=round(_pb_vol_ratio, 4),
        pullback_frac=round(_pullback_frac, 4),
        run_len_before=_run_len,
        impulse6_atr=round(_impulse6, 4),
        post_sl_tp1=_post_sl_tp1,
        post_sl_max_r=round(_post_sl_max_r, 4),
        mae_r=round(_mae_r, 4),
        mfe_tp1=round(_mfe_tp1, 4),
        accel_ratio=_fld(setup, "accel_ratio", 1.0),
        buy_pressure=_fld(setup, "buy_pressure", 0.0),
        absorption=_fld(setup, "absorption", 0.0),
        obv_agree=float(setup.get("obv_agree") or 0.0),
        obv_strength=float(setup.get("obv_strength") or 0.0),
        # NOT `or -1`: zone_age_bars == 0 is a real value (the zone formed on
        # the entry bar) and 0 is falsy, so the obvious spelling wrote -1 —
        # "no zone" — into the export for every age-0 trade. That corrupted the
        # feature analysis before it corrupted anything else: age 0 appeared to
        # occur zero times in 2712 trades, which is what made two otherwise
        # identical sizing runs disagree by 10%.
        zone_age_bars=int(_fld(setup, "zone_age_bars", -1.0)),
        bos_candles_ago=int(setup.get("bos_candles_ago") or -1),
        room_atr=round(_room, 3),
        tp1_beyond_level=_beyond,
        extension_atr=round(_ext, 3),
        entry_range_atr=round(_rng, 3),
        run_bars=_run,
        signal_bar=int(setup.get("_signal_bar", -1)),
    )


def backtest_symbol(
    symbol: str,
    *,
    candles: int,
    tp_window: int,
    warmup: int,
    stride: int,
    window_15m: int,
    window_1h: int,
    window_4h: int,
    use_prefilter: bool,
    refresh_cache: bool,
    fee_rate: float,
    slippage_rate: float,
    execution_delay_bars: int,
    adverse_entry_bps: float,
    exit_policy: str,
    trail_atr_mult: float,
    end_date_ms: int | None = None,
) -> SymbolResult:
    started = time.perf_counter()
    result = SymbolResult(symbol=symbol)

    try:
        c15 = fetch_history(symbol, TIMEFRAME_KUCOIN, KLINES_INTERVAL_SEC, candles,
                            refresh_cache=refresh_cache, end_date_ms=end_date_ms)
        c1h = fetch_history(
            symbol,
            TIMEFRAME_1H_KUCOIN,
            KLINES_1H_INTERVAL_SEC,
            max(10, math.ceil(candles / 4) + 4),
            refresh_cache=refresh_cache,
            end_date_ms=end_date_ms,
        )
        c4h = fetch_history(
            symbol,
            TIMEFRAME_4H_KUCOIN,
            KLINES_4H_INTERVAL_SEC,
            max(10, math.ceil(candles / 16) + 4),
            refresh_cache=refresh_cache,
            end_date_ms=end_date_ms,
        )
        try:
            c1d = fetch_history(
                symbol, "1d", 86400,
                max(8, math.ceil(candles / 96) + 4),
                refresh_cache=refresh_cache,
                end_date_ms=end_date_ms,
            )
        except Exception:
            c1d = {}
        # BTC 1h series, so btc_change_pct can be REAL instead of a constant 0.0.
        # Passing 0.0 (as this did until 2026-07-31) silently handed every single
        # backtest trade the maximum BTC score bonus — `0.0 >= 0` and `0.0 <= 0`
        # are both true, so BTC+2 fired on 100% of trades and the BTCok+1 branch
        # never ran — while also disabling two live filters outright
        # (BTC_BLOCK_THRESHOLD_PCT and FVG_LONDON_BTC_UP_FILTER can never trigger
        # at 0.0) and feeding a wrong rel_strength to the momentum pack.
        try:
            btc_1h = fetch_history(
                "BTCUSDT", TIMEFRAME_1H_KUCOIN, KLINES_1H_INTERVAL_SEC,
                max(10, math.ceil(candles / 4) + 4),
                refresh_cache=refresh_cache,
                end_date_ms=end_date_ms,
            )
        except Exception:
            btc_1h = {}
    except Exception as exc:
        result.error = str(exc)
        result.elapsed_sec = time.perf_counter() - started
        return result

    n = len(c15["close"])
    result.bars = n
    if n < warmup + tp_window + 2:
        result.elapsed_sec = time.perf_counter() - started
        return result

    for i in range(warmup, n - tp_window, max(1, stride)):
        result.scanned += 1

        if use_prefilter and not cheap_prefilter_at(c15, i, window_15m):
            result.prefiltered += 1
            continue

        snap_15 = candle_slice(c15, max(0, i - window_15m), i)
        t_cur = c15["time"][i - 1] if c15.get("time") and i > 0 else None
        snap_1h = aligned_slice_by_time(c1h, t_cur, window_1h, max(1, i // 4))
        snap_4h = aligned_slice_by_time(c4h, t_cur, window_4h, max(1, i // 16))
        snap_1d = aligned_slice_by_time(c1d, t_cur, 8, max(1, i // 96)) if c1d else None

        # Same definition the live bot uses (get_btc_change_1h): pct move of the
        # last CLOSED 1h BTC candle vs the one before it, as of this scan bar.
        _btc_chg = 0.0
        if btc_1h:
            _bsnap = aligned_slice_by_time(btc_1h, t_cur, 3, max(1, i // 4))
            _bc = (_bsnap or {}).get("close") or []
            if len(_bc) >= 2 and _bc[-2]:
                _btc_chg = (_bc[-1] - _bc[-2]) / _bc[-2] * 100.0

        result.analyzed += 1
        setup = analyze_coin_smc(snap_15, snap_1h, symbol, snap_4h,
                                 btc_change_pct=_btc_chg,
                                 candles_1d=snap_1d)
        if not setup:
            continue

        # k-NN price-shape analog score (research column, no look-ahead).
        # KNN_MAXHIST env caps the analog pool to test required live candle depth.
        _mh = os.getenv("KNN_MAXHIST", "").strip()
        knn = knn_direction_score(
            c15, i, setup["direction"],
            max_history=int(_mh) if _mh else None,
        )
        setup["_knn_score"] = -1.0 if knn is None else knn

        # RESEARCH ONLY (2026-08-10): model the live stale-entry guard, which
        # refuses to publish when price has already drifted off the zone. The
        # backtest otherwise fills at the zone midpoint unconditionally — a
        # price the market offered on only ~20% of setups — so without this the
        # entry-quality/volume trade-off cannot be measured at all.
        _mx = float(os.getenv("BT_MAX_ENTRY_DRIFT_PCT", "0") or 0)
        if _mx > 0:
            _pe = float(setup["current_price"])
            _cl = float(c15["close"][i - 1]) if i >= 1 else _pe
            _dr = ((_cl - _pe) if setup["direction"] == "LONG" else (_pe - _cl)) / _pe
            if _dr > _mx / 100.0:
                continue
            _mn = float(os.getenv("BT_MIN_ENTRY_DRIFT_PCT", "0") or 0)
            if _mn > 0 and _dr <= _mn / 100.0:
                continue
            # ...and fill where the market actually was when the signal fired,
            # not at the zone midpoint. Gating without this measures a fantasy
            # fill on a filtered subset, which is the wrong question.
            setup = dict(setup, current_price=_cl)

        # RESEARCH ONLY (2026-08-10): LIMIT-ORDER mode. The default backtest
        # fills at the zone midpoint unconditionally — a price the market
        # actually offered on only ~20% of setups within the signal bar — which
        # is why its +0.400R/trade is unreachable live, where we market-buy at
        # whatever price is showing. This models the honest alternative: rest a
        # limit AT the zone, fill only if price trades back into it within
        # BT_LIMIT_WAIT_BARS, and start the trade from THAT bar. Unfilled
        # setups are simply not traded.
        # RESEARCH ONLY (2026-08-23): ZONE DEPTH. The live bot parks an approved
        # setup and fires the moment price touches ANYWHERE in [entry_low,
        # entry_high]. Approaching a LONG zone from above, that first touch is
        # the TOP edge — the worst price in the zone. The backtest meanwhile
        # fills at the zone MIDPOINT (signal_filter sets entry_price = mid), so
        # the two disagree by half a zone width on every trade.
        #
        # Seen live on XPL 2026-08-23: zone 0.108048-0.110200, midpoint
        # 0.108995, actual fill 0.110000 — 0.92% worse, which on that trade's
        # 3.0% stop is 0.31R given away before the trade even started.
        #
        # depth 0.0 = zone edge (what live does today), 0.5 = midpoint (what the
        # backtest assumes), 1.0 = far edge. Requires price to actually reach
        # that level within BT_LIMIT_WAIT_BARS; unfilled setups are not traded.
        _depth = float(os.getenv("BT_ZONE_DEPTH", str(_DEFAULT_ZONE_DEPTH)) or -1)
        if _depth >= 0.0:
            _lo_z = float(setup.get("entry_low") or 0)
            _hi_z = float(setup.get("entry_high") or 0)
            if _hi_z > _lo_z > 0:
                _target = (_hi_z - (_hi_z - _lo_z) * _depth
                           if setup["direction"] == "LONG"
                           else _lo_z + (_hi_z - _lo_z) * _depth)
                setup = dict(setup, current_price=_target)

        setup = dict(setup, _signal_bar=i)
        _entry_bar = i
        _wait = int(os.getenv("BT_LIMIT_WAIT_BARS", str(_DEFAULT_WAIT_BARS)) or 0)
        if _wait > 0:
            _px = float(setup["current_price"])
            _lo, _hi = c15["low"], c15["high"]
            _hit = None
            for _j in range(i, min(i + _wait, len(_lo))):
                if _lo[_j] <= _px <= _hi[_j]:
                    _hit = _j
                    break
            if _hit is None:
                continue
            _entry_bar = _hit

        trade = simulate_trade_direct(
            symbol,
            setup,
            c15,
            _entry_bar,
            tp_window,
            fee_rate,
            slippage_rate,
            execution_delay_bars=execution_delay_bars,
            adverse_entry_bps=adverse_entry_bps,
            exit_policy=exit_policy,
            trail_atr_mult=trail_atr_mult,
        )
        result.trade_records.append(trade)
        result.trades += 1
        result.gross_r += trade.gross_r
        result.net_r += trade.net_r

        if trade.outcome in ("TP1", "TRAIL"):
            result.tp1 += 1
        elif trade.outcome == "TP2":
            result.tp2 += 1
        elif trade.outcome == "SL":
            result.sl += 1
        else:
            result.expired += 1

    result.elapsed_sec = time.perf_counter() - started
    return result


def merge_results(results: Iterable[SymbolResult]) -> SymbolResult:
    total = SymbolResult(symbol="TOTAL")
    for r in results:
        total.bars += r.bars
        total.scanned += r.scanned
        total.prefiltered += r.prefiltered
        total.analyzed += r.analyzed
        total.trades += r.trades
        total.tp1 += r.tp1
        total.tp2 += r.tp2
        total.sl += r.sl
        total.expired += r.expired
        total.gross_r += r.gross_r
        total.net_r += r.net_r
        total.elapsed_sec += r.elapsed_sec
        total.trade_records.extend(r.trade_records)
    return total


def risk_profile(rs: list[float]) -> dict:
    """Downside measures that do NOT hang on a single week.

    Max drawdown is one number produced by one stretch: on the 922-trade book
    of 2026-08-25 the whole -6.87R came from FIFTEEN trades across five days in
    April. Every equal-risk ranking divided by that, which is why thresholds
    jumped and halves disagreed on changes that were really noise.

    worst_windows averages the k deepest rolling N-trade stretches, so it takes
    several bad patches to move. ulcer is RMS of the underwater curve — it
    counts how long we spend below water, not just how far. Both are downside
    only: plain volatility penalises big wins too, which is not the risk here.
    """
    import statistics as _st
    if not rs:
        return {"max_dd": 0.0, "worst_windows": 0.0, "ulcer": 0.0}
    cum = peak = 0.0
    worst = 0.0
    sq = []
    for x in rs:
        cum += x
        peak = max(peak, cum)
        worst = min(worst, cum - peak)
        sq.append((cum - peak) ** 2)
    win, k = 25, 5
    ww = 0.0
    if len(rs) >= win:
        sums = [sum(rs[i:i + win]) for i in range(len(rs) - win + 1)]
        ww = -_st.mean(sorted(sums)[:k])
    return {"max_dd": -worst,
            "worst_windows": ww,
            "ulcer": (sum(sq) / len(sq)) ** 0.5}


def apply_live_gates(trades: list[TradeRecord]) -> list[TradeRecord]:
    """Trades that survive the live bot's throughput gates.

    An audit on 2026-08-16 listed every gate run_scan applies and grepped for it
    in this file. NINE gates, ZERO of them modelled here: funding, news, spread,
    stale-entry, kill-switch, auto-blocked symbols, reject cooldown, the
    per-scan signal cap, and the per-coin signal cooldown. The backtest has
    always reported a strategy taking far more trades than production can.

    The four that are reproducible from a trade list alone are replayed here.
    News, spread, funding and the two Claude-dependent gates need live state and
    stay unmodelled — so even this figure is an upper bound.

    Measured effect on the 18k-candle window: 1758 -> 1248 trades (-29%) and
    +819.9R -> +589.3R (-28%), while max drawdown halves (-11.83R -> -6.46R).
    Win rate is unchanged (85.7% -> 85.6%), so the gates drop average trades
    rather than bad ones — the live book is smaller and steadier, not better
    selected. Risk-adjusted it is the stronger number: 91.2 against 69.3.
    """
    # Bound the cost of a no-skill filter that rejects a share of the book.
    # Claude is LIVE-ONLY: he rejects roughly half the crypto setups and so
    # decides which trades take the per-scan slots, and none of that is here.
    # Dropping the same share at random says what that costs with zero skill:
    # live doing worse than this is Claude choosing badly, better is real
    # selection. Deterministic per trade so runs reproduce.
    _rej = float(os.getenv("BT_RANDOM_REJECT", "0") or 0)
    if _rej > 0:
        import hashlib as _hl
        _seed = os.getenv("BT_RANDOM_REJECT_SEED", "1")
        def _drop(t) -> bool:
            key = f"{_seed}|{t.symbol}|{t.entry_time}|{t.direction}"
            h = int(_hl.md5(key.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
            return h < _rej
        trades = [t for t in trades if not _drop(t)]
    ordered = sorted(trades, key=_scan_order_key)
    last_sig: dict = {}
    per_bar: dict = {}
    open_by_dir: dict = {}
    kept: list[TradeRecord] = []
    closed: list[tuple[float, str]] = []   # (exit_ts, outcome) of kept trades
    streak = 0
    cur_day = None
    blocked_day = None

    def _sl_streak_at(now: float, dy: int) -> int:
        """Consecutive SLs among trades that had already CLOSED by `now`, today.

        This is what the live kill-switch can actually see. The original replay
        walked trades in ENTRY order and read t.outcome, which is only known at
        exit — so it paused the day at the entry of the third trade that would
        LATER stop out. Losses cluster on this book, so peeking at that timing
        removed the rest of a bad patch and flattered every figure downstream of
        the gate, drawdown worst of all.
        """
        done = sorted((e, o) for e, o in closed if e <= now and int(e // 86400) == dy)
        n = 0
        for _, outcome in reversed(done):
            if outcome != "SL":
                break
            n += 1
        return n

    for t in ordered:
        raw = t.entry_time or 0
        ts = raw / 1000 if raw > 1e11 else raw
        day = int(ts // 86400)
        if day != cur_day:
            cur_day, streak, blocked_day, closed = day, 0, None, []
        if KILL_SWITCH_SL_STREAK > 0:
            if blocked_day == day:
                continue
            if not _BT_KILL_LOOKAHEAD and _sl_streak_at(ts, day) >= KILL_SWITCH_SL_STREAK:
                blocked_day = day
                continue
        key = (t.symbol, t.direction)
        if SIGNAL_COOLDOWN_HOURS > 0 and key in last_sig                 and (ts - last_sig[key]) / 3600 < SIGNAL_COOLDOWN_HOURS:
            continue
        bar = int(ts // (KLINES_INTERVAL_SEC or 900))
        if _LIVE_MAX_PER_SCAN > 0 and per_bar.get(bar, 0) >= _LIVE_MAX_PER_SCAN:
            continue
        if MAX_SAME_DIRECTION_POSITIONS > 0:
            live = [o for o in open_by_dir.get(t.direction, [])
                    if (o.exit_time or 0) > raw]
            if len(live) >= MAX_SAME_DIRECTION_POSITIONS:
                continue
            live.append(t)
            open_by_dir[t.direction] = live
        last_sig[key] = ts
        per_bar[bar] = per_bar.get(bar, 0) + 1
        kept.append(t)
        if KILL_SWITCH_SL_STREAK > 0:
            ex = t.exit_time or 0
            ex = ex / 1000 if ex > 1e11 else ex
            if ex > 0:
                closed.append((ex, t.outcome))
            if _BT_KILL_LOOKAHEAD:
                streak = streak + 1 if t.outcome == "SL" else 0
                if streak >= KILL_SWITCH_SL_STREAK:
                    blocked_day = day
    return kept


def apply_direction_cap(trades: list[TradeRecord], cap: int) -> list[TradeRecord]:
    """Trades that survive the live bot's MAX_SAME_DIRECTION_POSITIONS cap.

    The engine cannot enforce this while scanning: symbols run in separate
    processes and none of them can see the shared book. So the cap has simply
    never been modelled, and every headline figure this file has ever printed
    counted setups the live bot would have refused — 1758 against 1644 on the
    18k-candle window, i.e. profit overstated by about 6%.

    Replayed here in entry order over the merged trade list, which is how the
    live book actually fills. Reported alongside the uncapped total rather than
    replacing it: the uncapped number measures the strategy, this one measures
    the book the bot is allowed to carry, and past results stay comparable.
    """
    if cap <= 0:
        return list(trades)
    ordered = sorted(trades, key=_scan_order_key)
    open_by_dir: dict[str, list[TradeRecord]] = {}
    kept: list[TradeRecord] = []
    for t in ordered:
        now = t.entry_time or 0
        live = [o for o in open_by_dir.get(t.direction, []) if (o.exit_time or 0) > now]
        if len(live) >= cap:
            continue
        live.append(t)
        open_by_dir[t.direction] = live
        kept.append(t)
    return kept


def max_drawdown_r(trades: list[TradeRecord], *, net: bool = True) -> float:
    equity = peak = 0.0
    max_dd = 0.0
    ordered = sorted(trades, key=lambda t: (t.entry_time or 0, t.symbol, t.entry_bar))
    for trade in ordered:
        equity += trade.net_r if net else trade.gross_r
        peak = max(peak, equity)
        max_dd = min(max_dd, equity - peak)
    return max_dd


def print_symbol_result(r: SymbolResult) -> None:
    if r.error:
        print(f"  {r.symbol:<13} ERROR {r.error}")
        return
    rate = r.scanned / r.elapsed_sec if r.elapsed_sec > 0 else 0.0
    print(
        f"  {r.symbol:<13} tr={r.trades:<4} "
        f"TP1={r.tp1:<3} TP2={r.tp2:<3} SL={r.sl:<3} EXP={r.expired:<3} "
        f"netR={r.net_r:+7.2f} "
        f"bars={r.scanned:<5} heavy={r.analyzed:<5} "
        f"{rate:7.0f} bars/s"
    )


def write_trades_csv(path: str, trades: list[TradeRecord]) -> None:
    fields = [
        "symbol", "entry_bar", "exit_bar", "entry_time", "exit_time",
        "direction", "outcome", "entry", "tp1", "tp2", "sl",
        "gross_r", "net_r", "cost_r", "mtf_score", "volume_ratio",
        "rsi", "eff_ratio", "vol_atr_pct", "vol_ratio_regime",
        "adaptive_pack", "adaptive_reason", "risk_mult",
        "quality_score", "trend_score", "volatility_score",
        "entry_quality_score", "portfolio_risk_score",
        "session", "trend_1h", "trend_4h", "entry_source",
        "signals", "score_tags", "premium", "sniper", "knn_score", "swing_trend",
        "pullback_frac", "run_len_before", "impulse6_atr", "retrace_bars_ratio", "pullback_vol_ratio", "post_sl_tp1", "post_sl_max_r", "mae_r", "mfe_tp1", "accel_ratio", "buy_pressure", "absorption", "obv_agree", "obv_strength", "size_mult", "signal_bar",
        "zone_age_bars", "bos_candles_ago", "extension_atr",
        "room_atr", "tp1_beyond_level",
        "entry_range_atr", "run_bars",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for trade in sorted(trades, key=lambda t: (t.entry_time or 0, t.symbol, t.entry_bar)):
            writer.writerow({name: getattr(trade, name) for name in fields})


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Fast SMC backtest")
    p.add_argument("--symbols", default=None, help="Comma-separated KuCoin symbols. Default: pinned set/env BACKTEST_SYMBOLS.")
    p.add_argument("--top", type=int, default=0, help="Use current top N KuCoin USDT pairs by 24h volume.")
    p.add_argument("--candles", type=int, default=BACKTEST_CANDLES, help="15m candles per symbol.")
    p.add_argument(
        "--tp-window",
        type=int,
        default=BACKTEST_TP_WINDOW,
        help="Forward 15m candles for TP/SL simulation. Default mirrors SIGNAL_EXPIRY_HOURS.",
    )
    p.add_argument("--workers", type=int, default=0, help="Parallel worker processes. 0 = auto.")
    p.add_argument("--serial", action="store_true", help="Run without multiprocessing.")
    p.add_argument("--quiet", action="store_true", help="Print only the final summary.")
    p.add_argument("--stride", type=int, default=1, help="Scan every Nth candle. Use 4/8 for very fast rough sweeps.")
    p.add_argument("--warmup", type=int, default=DEFAULT_WARMUP, help="First scan bar.")
    p.add_argument("--window-15m", type=int, default=WINDOW_15M, help="15m lookback window passed to strategy.")
    p.add_argument("--window-1h", type=int, default=WINDOW_1H, help="1h lookback window passed to strategy.")
    p.add_argument("--window-4h", type=int, default=WINDOW_4H, help="4h lookback window passed to strategy.")
    p.add_argument("--no-prefilter", action="store_true", help="Disable exact BOS/volume early reject.")
    p.add_argument("--refresh-cache", action="store_true", help="Ignore cached candle files.")
    p.add_argument("--end-date", default=None,
                   help="ISO date (YYYY-MM-DD, UTC) to anchor the candle window's newest "
                        "bar to, instead of now. Lets --candles target an exact past range "
                        "(e.g. --end-date 2024-01-01 --candles 70080 = 2022-01-01..2024-01-01) "
                        "without re-downloading a range already covered by another batch.")
    p.add_argument("--fee-rate", type=float, default=BACKTEST_FEE_RATE, help="Per-side fee rate for net R estimate.")
    p.add_argument("--slippage-rate", type=float, default=BACKTEST_SLIPPAGE_RATE, help="Per-side slippage rate for net R estimate.")
    p.add_argument("--execution-delay-bars", type=int, default=0, help="Delay entry by N 15m bars for execution realism.")
    # 2026-09-02: the live-vs-model fill gap was measured for this bot at a
    # median +0.081% (8 bps) -- zone_entry_price is the analyzer price this model
    # fills at, market_price is the live price at order time, and both sit on the
    # signal row. Adverse in BOTH directions (LONG +0.104%, SHORT +0.051%, 100%
    # each), so it is neither an exchange basis nor the zone-midpoint double
    # count. At 8 bps: 2026 +476.48 -> +390.70R (WR 73.9 -> 71.4), 2024 +207.17 ->
    # +154.17R. That closes only part of the gap to the live 61.5% WR, unlike the
    # stocks bot where the same correction lands on the live numbers -- consistent
    # with this bot waiting for the zone instead of entering at market.
    # Default stays 0 so the recorded tables remain comparable.
    p.add_argument("--adverse-entry-bps", type=float, default=0.0, help="Extra adverse fill in basis points.")
    p.add_argument("--exit-policy", choices=["classic", "trail"], default="trail", help="Exit model after TP1 (default mirrors live TRAIL_RUNNER_ENABLED).")
    p.add_argument("--trail-atr-mult", type=float, default=TRAIL_ATR_MULT, help="ATR multiple for --exit-policy trail (default mirrors live config).")
    p.add_argument("--export-trades", default=None, help="Write trade list CSV.")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    end_date_ms = None
    if args.end_date:
        from datetime import datetime as _dt, timezone as _tz
        end_date_ms = int(_dt.strptime(args.end_date, "%Y-%m-%d")
                          .replace(tzinfo=_tz.utc).timestamp() * 1000)
    # ⚠ THE MODEL AND THE LIVE BOT DO NOT TRADE THE SAME COINS. Measured
    # 2026-09-05. --top defaults to 0, so the default universe is the FIXED
    # 18-symbol list in parse_symbols(). The live bot re-ranks its top-25 by
    # turnover on every scan, so over three months it traded 47 distinct
    # symbols. The 29 it never tests are not a rounding error, they are half
    # the live book:
    #
    #   живые сделки крипты   сделок  винрейт   итого
    #   монета есть в модели     114    68.4%  +22.05R
    #   монеты в модели НЕТ      118    55.1%   -8.71R
    #
    # The tested half lands near the model (68.4% against 70.6% over the same
    # days); the untested half loses. So the model's optimism is structural,
    # not a mis-tuned knob, and any win rate from this file describes those 18
    # coins — not the universe the bot actually trades.
    #
    # Do NOT reach for --top as the fix: fetch_top_symbols() ranks on
    # volCcyQuote, and that field comes back 0.0 from this endpoint for the
    # majors (BTC and ETH included). Its top-25 and the live bot's top-25 agree
    # on 2 of 25 — it selects obscure and stock-tracking swaps. Ranking has to
    # match get_top_coins() in src/binance_client.py (volCcy24h * last, with the
    # crypto-only and spread filters) before --top means anything.
    if args.symbols:
        symbols = parse_symbols(args.symbols)
    elif args.top > 0:
        symbols = fetch_top_symbols(args.top)
    else:
        symbols = parse_symbols(None)
    worker_count = 1 if args.serial else (choose_workers(len(symbols), args.candles, args.stride) if args.workers <= 0 else args.workers)

    print(f"Fast backtest: {len(symbols)} symbols, {args.candles} candles, TP window {args.tp_window}")
    print(
        f"workers={worker_count}, stride={args.stride}, "
        f"prefilter={'off' if args.no_prefilter else 'on'}, cache={'refresh' if args.refresh_cache else 'ttl'}"
    )
    print()

    started = time.perf_counter()
    kwargs = dict(
        candles=args.candles,
        tp_window=args.tp_window,
        warmup=args.warmup,
        stride=max(1, args.stride),
        window_15m=args.window_15m,
        window_1h=args.window_1h,
        window_4h=args.window_4h,
        use_prefilter=not args.no_prefilter,
        refresh_cache=args.refresh_cache,
        fee_rate=args.fee_rate,
        slippage_rate=args.slippage_rate,
        execution_delay_bars=max(0, args.execution_delay_bars),
        adverse_entry_bps=max(0.0, args.adverse_entry_bps),
        exit_policy=args.exit_policy,
        trail_atr_mult=max(0.0, args.trail_atr_mult),
        end_date_ms=end_date_ms,
    )

    results: list[SymbolResult] = []
    if worker_count == 1 or len(symbols) == 1:
        for symbol in symbols:
            r = backtest_symbol(symbol, **kwargs)
            results.append(r)
            if not args.quiet:
                print_symbol_result(r)
    else:
        workers = max(1, min(worker_count, len(symbols)))
        with ProcessPoolExecutor(max_workers=workers) as pool:
            future_map = {pool.submit(backtest_symbol, symbol, **kwargs): symbol for symbol in symbols}
            for fut in as_completed(future_map):
                r = fut.result()
                results.append(r)
                if not args.quiet:
                    print_symbol_result(r)

    wall_sec = time.perf_counter() - started
    total = merge_results(results)
    errors = [r for r in results if r.error]
    wins = total.tp1 + total.tp2
    win_rate = wins / total.trades * 100 if total.trades else 0.0
    gross_rpt = total.gross_r / total.trades if total.trades else 0.0
    net_rpt = total.net_r / total.trades if total.trades else 0.0
    total_rate = total.scanned / wall_sec if wall_sec > 0 else 0.0

    print("\n" + "=" * 72)
    print("BACKTEST RESULTS")
    print("=" * 72)
    print(f"Symbols:       {len(symbols)} ({len(errors)} errors)")
    print(f"Bars scanned:  {total.scanned} ({total_rate:,.0f} bars/s wall-clock)")
    print(f"Heavy scans:   {total.analyzed}  skipped by prefilter: {total.prefiltered}")
    print(f"Trades:        {total.trades}")
    print(f"  TP1 hit:     {total.tp1}")
    print(f"  TP2 hit:     {total.tp2}")
    print(f"  SL hit:      {total.sl}")
    print(f"  Expired:     {total.expired}")
    print(f"Win rate:      {win_rate:.1f}%")
    print(f"Gross R:       {total.gross_r:+.2f}R total ({gross_rpt:+.3f}R/trade)")
    print(f"Net R est.:    {total.net_r:+.2f}R total ({net_rpt:+.3f}R/trade)")
    print(f"Max DD gross:  {max_drawdown_r(total.trade_records, net=False):+.2f}R")
    print(f"Max DD net:    {max_drawdown_r(total.trade_records, net=True):+.2f}R")
    print(f"Elapsed:       {wall_sec:.2f}s wall-clock")

    # What the live bot could actually have carried. Everything above counts
    # setups production would have refused: the per-coin cooldown, the per-scan
    # signal cap, the direction cap and the kill-switch. Still an upper bound —
    # news, spread, funding and the Claude gates need live state to model.
    if total.trade_records:
        gated = apply_live_gates(total.trade_records)
        if len(gated) != len(total.trade_records):
            g_net = sum(t.net_r for t in gated)
            # A win is an outcome that reached TP1: TP1, TP2, or TRAIL (TP1 hit,
            # then the runner trailed out). NOT merely "outcome != SL" — that
            # counted EXPIRED and the research STALE exit as wins and reported
            # 96% on a run where the stale-exit flag was converting stops into
            # scratches.
            g_wins = sum(1 for t in gated if t.outcome in ("TP1", "TP2", "TRAIL"))
            g_dd = max_drawdown_r(gated, net=True)

            print()
            # The candle count belongs on the comparison line. The live default
            # is 1152 (12 days), while window comparisons need 18000, and a run
            # that silently used the small default still prints a plausible
            # "18 symbols (0 errors)" header. One such run produced a variant
            # and a control that matched to the last decimal because both had
            # measured twelve days instead of the window.
            print(
                f"[{args.candles} candles"
                f"{' to ' + args.end_date if args.end_date else ''}] "
                f"With live gates (cooldown {SIGNAL_COOLDOWN_HOURS}h, "
                f"{_LIVE_MAX_PER_SCAN}/scan, {MAX_SAME_DIRECTION_POSITIONS}/dir, "
                f"kill {KILL_SWITCH_SL_STREAK}): "
                f"{len(gated)} trades "
                f"({len(total.trade_records) - len(gated)} refused), "
                f"WR {g_wins / len(gated) * 100:.1f}%, "
                f"net {g_net:+.2f}R, "
                f"Max DD {g_dd:+.2f}R, "
                f"profit/DD {g_net / abs(g_dd):.1f}"
            )
            # Max DD is one number from one stretch — see risk_profile. Print
            # the measures that need several bad patches to move, so variants
            # stop being ranked by a single week.
            _rp = risk_profile([t.net_r for t in gated])
            # The old guard dropped the WHOLE line when worst_windows was not
            # positive, taking ulcer with it — but ulcer stays meaningful there.
            # worst_windows is the mean of the 5 most negative 25-trade sums,
            # negated, so on a book where even the worst 25-trade stretch made
            # money it goes negative and its ratio prints as a large negative
            # number that reads like a catastrophe while meaning the opposite.
            _ww = _rp["worst_windows"]
            _ulc = (f"ulcer {_rp['ulcer']:.2f} (прибыль/ulcer "
                    f"{g_net / _rp['ulcer']:.1f})" if _rp['ulcer'] > 0
                    else "ulcer 0 (не применим)")
            if _ww > 0:
                print(f"  устойчивый риск: худшие окна {_ww:.2f}R "
                      f"(прибыль/риск {g_net / _ww:.1f})   {_ulc}")
            else:
                print(f"  устойчивый риск: худшие окна НЕ ПРИМЕНИМЫ "
                      f"(ни один отрезок из 25 сделок не убыточен; сырое {_ww:.2f})"
                      f"   {_ulc}")

    if args.export_trades:
        # Export the gated book by default — an ungated dump cannot be compared
        # against the headline numbers, which are all post-gate. BT_EXPORT_RAW=1
        # restores the full pre-gate list.
        _export = total.trade_records if os.getenv("BT_EXPORT_RAW") == "1"             else apply_live_gates(total.trade_records)
        write_trades_csv(args.export_trades, _export)
        print(f"Trades CSV:    {args.export_trades}")

    return 1 if errors and len(errors) == len(symbols) else 0


if __name__ == "__main__":
    raise SystemExit(main())
