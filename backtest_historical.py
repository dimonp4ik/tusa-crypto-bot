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
TP_WINDOW    = 48             # 12h forward simulation

RETEST_CONFIGS = [
    ("0.015 (current)", 0.015),
    ("0.004 (v3 tight)", 0.004),
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
            return data
        except Exception as e:
            if attempt == 3:
                raise
            time.sleep(2 * (attempt + 1))

    raise RuntimeError(f"Failed to fetch {symbol} {interval}")


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

def backtest_symbol_week(symbol: str, end_ts: int, retest_pct: float) -> dict:
    """Run signal scan + trade sim on one symbol, one historical week."""
    import src.signal_filter as _sf
    from src.signal_filter import analyze_coin_smc

    # Patch retest value at module level (imported variable, not re-read from config)
    _sf.RETEST_MAX_DIST_PCT = retest_pct

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
    trades = tp1 = tp2 = sl_hit = expired = 0

    for i in range(50, n - TP_WINDOW):
        snap_15 = {k: v[max(0, i - W15):i] for k, v in c15.items()}

        # Time-align 1h/4h by timestamp (proper alignment, not i//4)
        t_cur = c15["time"][i - 1]
        i1h = max(1, next(
            (j for j in range(len(c1h["time"]) - 1, -1, -1) if c1h["time"][j] <= t_cur),
            1,
        ))
        i4h = max(1, next(
            (j for j in range(len(c4h["time"]) - 1, -1, -1) if c4h["time"][j] <= t_cur),
            1,
        ))
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


# -- Main ----------------------------------------------------------------------

def _totals():
    return {"trades": 0, "tp1": 0, "tp2": 0, "sl": 0, "expired": 0}


def _add(acc, d):
    for k in acc:
        acc[k] += d.get(k, 0)


def _pnl(t):
    return t["tp2"] * 3.75 + t["tp1"] * 1.25 - t["sl"] * 1.0


def main():
    print("=" * 65)
    print("HISTORICAL BACKTEST  --  5 weeks x 10 symbols")
    print("Comparing RETEST_MAX_DIST_PCT: 0.015 vs 0.004")
    print("=" * 65)

    all_results = {label: {"weeks": [], "total": _totals()}
                   for label, _ in RETEST_CONFIGS}

    for week_dt, week_label in WEEK_ENDS:
        end_ts = int(week_dt.timestamp())
        print(f"\n-- Week: {week_label} --")

        for label, retest_pct in RETEST_CONFIGS:
            week_total = _totals()
            for sym in SYMBOLS:
                print(f"  [{label}] {sym}...", end=" ", flush=True)
                try:
                    r = backtest_symbol_week(sym, end_ts, retest_pct)
                    _add(week_total, r)
                    err = r.get("error", "")
                    if err:
                        print(f"skip ({err[:40]})")
                    else:
                        print(f"tr={r['trades']} TP1={r['tp1']} TP2={r['tp2']} SL={r['sl']}")
                except Exception as e:
                    print(f"ERROR: {e}")
                time.sleep(0.3)  # gentle rate limiting

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
    print(f"{'Metric':<22} {'RETEST=0.015':>14} {'RETEST=0.004':>14}  {'diff':>8}")
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
    best = "0.004" if rows[1]["rpt"] > rows[0]["rpt"] else "0.015"
    print(f"Winner by R/trade: RETEST={best}")


if __name__ == "__main__":
    main()
