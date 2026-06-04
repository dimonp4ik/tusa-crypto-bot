"""
Historical backtest: compare RETEST_MAX_DIST_PCT=0.015 vs 0.004
across 5 weeks spread over the last 6 months.

Both configs run on the SAME cached data (no double API calls).
Each week = 7 days of 15m signal data + 12h forward TP/SL window.

Usage:
    python backtest_historical.py
"""

import os
import sys
import time
import pickle
import requests
import concurrent.futures
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# -- Symbols (top liquid, available on KuCoin) ---------------------------------
SYMBOLS = [
    "BTC-USDT", "ETH-USDT", "SOL-USDT", "BNB-USDT", "XRP-USDT",
    "LINK-USDT", "DOT-USDT", "ADA-USDT", "SUI-USDT", "XLM-USDT",
]

# -- 5 random weeks from last 6 months (UTC end-of-week timestamps) ------------
WEEK_ENDS = [
    (datetime(2025, 12, 15, tzinfo=timezone.utc), "2025-12-08 -> 2025-12-15"),
    (datetime(2026,  1, 19, tzinfo=timezone.utc), "2026-01-12 -> 2026-01-19"),
    (datetime(2026,  2, 23, tzinfo=timezone.utc), "2026-02-16 -> 2026-02-23"),
    (datetime(2026,  3, 30, tzinfo=timezone.utc), "2026-03-23 -> 2026-03-30"),
    (datetime(2026,  5,  5, tzinfo=timezone.utc), "2026-04-28 -> 2026-05-05"),
]

KUCOIN_BASE  = "https://api.kucoin.com"
CACHE_DIR    = "backtest_cache_hist"
CACHE_TTL    = 72 * 3600      # 72h -- historical data doesn't change

# Signal window: 7 days + 2-day buffer so forward window doesn't overflow
CANDLES_15M  = 672 + 192      # 864 x 15min ~= 9 days
CANDLES_1H   = CANDLES_15M // 4
CANDLES_4H   = CANDLES_15M // 16
TP_WINDOW    = 192            # 48h forward simulation (matches live SIGNAL_EXPIRY_HOURS)

# Parallel workers for symbol scanning (set to 1 to disable)
MAX_WORKERS  = min(4, (os.cpu_count() or 2))

RETEST_CONFIGS = [
    ("CURRENT (adaptive off)", "current"),
    ("ADAPTIVE packs",         "adaptive"),
]


# -- Data fetching with cache --------------------------------------------------

def _cache_path(symbol: str, interval: str, end_ts: int) -> str:
    os.makedirs(CACHE_DIR, exist_ok=True)
    safe = symbol.replace("/", "_")
    return os.path.join(CACHE_DIR, f"{safe}_{interval}_{end_ts}.pkl")


def fetch_hist(symbol: str, interval: str, interval_sec: int,
               count: int, end_ts: int) -> dict:
    """Fetch historical candles ending at `end_ts`, with local cache."""
    path = _cache_path(symbol, interval, end_ts)
    if os.path.exists(path) and (time.time() - os.path.getmtime(path) < CACHE_TTL):
        with open(path, "rb") as f:
            return pickle.load(f)

    start_at = end_ts - count * interval_sec
    for attempt in range(4):
        try:
            resp = requests.get(
                f"{KUCOIN_BASE}/api/v1/market/candles",
                params={"symbol": symbol, "type": interval,
                        "startAt": start_at, "endAt": end_ts},
                timeout=20,
            )
            if resp.status_code == 429:
                wait = 4 * (attempt + 1)
                print(f"    429 rate-limit -- wait {wait}s...")
                time.sleep(wait)
                continue
            resp.raise_for_status()
            raw = list(reversed(resp.json().get("data", [])))
            if not raw:
                raise ValueError("empty response")
            data = {
                "time":   [int(c[0])   for c in raw],
                "open":   [float(c[1]) for c in raw],
                "high":   [float(c[3]) for c in raw],
                "low":    [float(c[4]) for c in raw],
                "close":  [float(c[2]) for c in raw],
                "volume": [float(c[5]) for c in raw],
            }
            with open(path, "wb") as f:
                pickle.dump(data, f)
            # Only sleep after a real API call, not on cache hits
            time.sleep(0.3)
            return data
        except Exception as e:
            if attempt == 3:
                raise
            time.sleep(2 * (attempt + 1))

    raise RuntimeError(f"Failed to fetch {symbol} {interval}")


# -- Pre-build time-alignment index (O(n) instead of O(n^2) per candle) -------

def _build_align_index(ref_times: list, htf_times: list) -> list:
    """
    For each index i in ref_times, find the largest j such that
    htf_times[j] <= ref_times[i].  Returns list of length len(ref_times).
    One linear pass: O(n + m) instead of O(n * m).
    """
    n, m = len(ref_times), len(htf_times)
    align = [0] * n
    j = 0
    for i in range(n):
        while j < m - 1 and htf_times[j + 1] <= ref_times[i]:
            j += 1
        align[i] = j
    return align


# -- Trade simulation ----------------------------------------------------------

def simulate_trade(setup: dict, future: dict) -> str:
    from src.telegram_notifier import calculate_tp_sl
    direction = setup["direction"]
    entry     = setup["current_price"]
    tp1, tp2, sl = calculate_tp_sl(
        entry, direction,
        atr=setup.get("atr", 0.0),
        recent_high=setup.get("recent_high", 0.0),
        recent_low=setup.get("recent_low", 0.0),
        tp1_level=setup.get("tp1_level"),
        tp2_level=setup.get("tp2_level"),
    )
    for h, l in zip(future["high"][:TP_WINDOW], future["low"][:TP_WINDOW]):
        if direction == "LONG":
            if l <= sl:  return "SL"
            if h >= tp2: return "TP2"
            if h >= tp1: return "TP1"
        else:
            if h >= sl:  return "SL"
            if l <= tp2: return "TP2"
            if l <= tp1: return "TP1"
    return "EXPIRED"


# -- Core: backtest one symbol/week/retest combo -------------------------------

def backtest_symbol_week(symbol: str, end_ts: int, mode: str) -> dict:
    """Run signal scan + trade sim on one symbol, one historical week.

    mode='current'  → our current fixed MTF gate (adaptive packs OFF).
    mode='adaptive' → friend's regime-aware adaptive pack gate ON.
    """
    import src.signal_filter as _sf
    from src.signal_filter import analyze_coin_smc

    # Toggle the adaptive regime-pack gate at module level for this config.
    _sf.ADAPTIVE_FILTER_PACKS = (mode == "adaptive")

    try:
        c15 = fetch_hist(symbol, "15min", 15*60,  CANDLES_15M, end_ts)
        c1h = fetch_hist(symbol, "1hour", 3600,   CANDLES_1H,  end_ts)
        c4h = fetch_hist(symbol, "4hour", 4*3600, CANDLES_4H,  end_ts)
    except Exception as e:
        return {"trades": 0, "tp1": 0, "tp2": 0, "sl": 0, "expired": 0, "error": str(e)}

    n = len(c15["close"])
    if n < 100:
        return {"trades": 0, "tp1": 0, "tp2": 0, "sl": 0, "expired": 0}

    W15, W1H, W4H = 300, 90, 50

    # Pre-build alignment index once: O(n) instead of O(n^2) in the inner loop
    align_1h = _build_align_index(c15["time"], c1h["time"])
    align_4h = _build_align_index(c15["time"], c4h["time"])

    trades = tp1 = tp2 = sl_hit = expired = 0

    for i in range(50, n - TP_WINDOW):
        snap_15 = {k: v[max(0, i - W15):i] for k, v in c15.items()}

        i1h = align_1h[i - 1]
        i4h = align_4h[i - 1]
        snap_1h = {k: v[max(0, i1h - W1H):i1h] for k, v in c1h.items()}
        snap_4h = {k: v[max(0, i4h - W4H):i4h] for k, v in c4h.items()}

        setup = analyze_coin_smc(snap_15, snap_1h, symbol, snap_4h, btc_change_pct=0.0)
        if not setup:
            continue

        future  = {k: v[i:i + TP_WINDOW] for k, v in c15.items()}
        outcome = simulate_trade(setup, future)
        trades += 1
        if   outcome == "TP1": tp1     += 1
        elif outcome == "TP2": tp2     += 1
        elif outcome == "SL":  sl_hit  += 1
        else:                  expired += 1

    return {"trades": trades, "tp1": tp1, "tp2": tp2, "sl": sl_hit, "expired": expired}


# -- Parallel worker wrapper ---------------------------------------------------

def _worker(args):
    """Top-level function (picklable) for ProcessPoolExecutor."""
    symbol, end_ts, label, mode = args
    try:
        result = backtest_symbol_week(symbol, end_ts, mode)
    except Exception as e:
        result = {"trades": 0, "tp1": 0, "tp2": 0, "sl": 0, "expired": 0, "error": str(e)}
    return symbol, label, result


# -- Main ----------------------------------------------------------------------

def _totals():
    return {"trades": 0, "tp1": 0, "tp2": 0, "sl": 0, "expired": 0}


def _add(acc, d):
    for k in acc:
        acc[k] += d.get(k, 0)


def _pnl(t):
    # R model (50/50 split, SL→BE after TP1, TP1=1.5R TP2=2.0R):
    #   TP2_HIT = 0.75 + 1.0 = +1.75R | TP1 only = +0.75R | SL = -1.0R
    return t["tp2"] * 1.75 + t["tp1"] * 0.75 - t["sl"] * 1.0


def main():
    print("=" * 65)
    print("HISTORICAL BACKTEST  --  5 weeks x 10 symbols")
    print(f"Comparing SWEEP-only vs PREMIUM(+sweep) (48h window, workers={MAX_WORKERS})")
    print("=" * 65)

    all_results = {label: {"weeks": [], "total": _totals()}
                   for label, _ in RETEST_CONFIGS}

    for week_dt, week_label in WEEK_ENDS:
        end_ts = int(week_dt.timestamp())
        print(f"\n-- Week: {week_label} --")

        # Collect results per (label, symbol) — parallel execution
        week_data = {label: {} for label, _ in RETEST_CONFIGS}

        tasks = [
            (sym, end_ts, label, rp)
            for sym in SYMBOLS
            for label, rp in RETEST_CONFIGS
        ]

        if MAX_WORKERS > 1:
            with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as exe:
                futures = {exe.submit(_worker, t): t for t in tasks}
                for fut in concurrent.futures.as_completed(futures):
                    sym, label, r = fut.result()
                    week_data[label][sym] = r
        else:
            for t in tasks:
                sym, label, r = _worker(t)
                week_data[label][sym] = r

        # Print + accumulate in symbol order
        for label, _ in RETEST_CONFIGS:
            week_total = _totals()
            for sym in SYMBOLS:
                r = week_data[label].get(sym, _totals())
                _add(week_total, r)
                err = r.get("error", "")
                if err:
                    print(f"  [{label}] {sym}... skip ({err[:40]})")
                else:
                    print(f"  [{label}] {sym}... "
                          f"tr={r['trades']} TP1={r['tp1']} TP2={r['tp2']} SL={r['sl']}")

            _add(all_results[label]["total"], week_total)
            all_results[label]["weeks"].append(week_total)
            t = week_total
            wins = t["tp1"] + t["tp2"]
            wr   = wins / t["trades"] * 100 if t["trades"] else 0
            pnl  = _pnl(t)
            print(f"  -> [{label}] week total: {t['trades']} trades  "
                  f"WR={wr:.1f}%  R={pnl:+.2f}R")

    # -- Summary ---------------------------------------------------------------
    print("\n" + "=" * 65)
    print("SUMMARY  (5 weeks x 10 symbols combined)")
    print("=" * 65)
    print(f"{'Metric':<22} {'SWEEP':>14} {'PREMIUM':>14}  {'diff':>8}")
    print("-" * 65)

    totals = [all_results[label]["total"] for label, _ in RETEST_CONFIGS]
    labels = [label for label, _ in RETEST_CONFIGS]

    rows = {}
    for i, (label, t) in enumerate(zip(labels, totals)):
        wins = t["tp1"] + t["tp2"]
        wr   = wins / t["trades"] * 100 if t["trades"] else 0.0
        pnl  = _pnl(t)
        rpt  = pnl / t["trades"] if t["trades"] else 0.0
        rows[i] = {"trades": t["trades"], "wr": wr, "pnl": pnl, "rpt": rpt,
                   "tp1": t["tp1"], "tp2": t["tp2"], "sl": t["sl"]}

    def _row(name, fmt, k):
        v0, v1 = rows[0][k], rows[1][k]
        delta  = v1 - v0
        print(f"{name:<22} {fmt.format(v0):>14} {fmt.format(v1):>14}  {delta:>+8.2f}")

    _row("Trades",      "{:.0f}",  "trades")
    _row("Win rate %",  "{:.1f}%", "wr")
    _row("TP1 hits",    "{:.0f}",  "tp1")
    _row("TP2 hits",    "{:.0f}",  "tp2")
    _row("SL hits",     "{:.0f}",  "sl")
    _row("Total R",     "{:+.2f}R","pnl")
    _row("R/trade",     "{:+.3f}R","rpt")

    print("=" * 65)
    best = "PREMIUM (+sweep)" if rows[1]["rpt"] > rows[0]["rpt"] else "SWEEP only"
    print(f"Winner by R/trade: {best}")


if __name__ == "__main__":
    main()
