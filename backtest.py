"""
Backtest the SMC strategy on historical KuCoin data.

Usage:  python backtest.py
Outputs: total trades, win rate, avg R:R, max drawdown.
"""

import time
import sys
import os
import pickle
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import requests
from config import (
    KUCOIN_BASE_URL, TIMEFRAME_KUCOIN, KLINES_INTERVAL_SEC,
    TIMEFRAME_1H_KUCOIN, KLINES_1H_INTERVAL_SEC,
    TIMEFRAME_4H_KUCOIN, KLINES_4H_INTERVAL_SEC,
    BACKTEST_CANDLES, BACKTEST_TP_WINDOW,
)
from src.signal_filter import analyze_coin_smc
from src.binance_client import get_top_coins
from src.telegram_notifier import calculate_tp_sl


_CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "backtest_cache")
_CACHE_TTL  = 2 * 3600  # 2 hours — refresh if older


def _cache_path(symbol: str, interval: str, count: int) -> str:
    os.makedirs(_CACHE_DIR, exist_ok=True)
    safe = symbol.replace("/", "_")
    return os.path.join(_CACHE_DIR, f"{safe}_{interval}_{count}.pkl")


def fetch_history(symbol: str, interval: str, interval_sec: int, count: int) -> dict:
    """Fetch `count` historical candles. Uses local cache (TTL 2h) to avoid re-fetching."""
    path = _cache_path(symbol, interval, count)

    # Return cached data if fresh
    if os.path.exists(path):
        age = time.time() - os.path.getmtime(path)
        if age < _CACHE_TTL:
            with open(path, "rb") as f:
                return pickle.load(f)

    url = f"{KUCOIN_BASE_URL}/api/v1/market/candles"
    now = int(time.time())
    start_at = now - count * interval_sec

    params = {"symbol": symbol, "type": interval, "startAt": start_at, "endAt": now}
    resp = requests.get(url, params=params, timeout=15)
    resp.raise_for_status()
    candles = list(reversed(resp.json().get("data", [])))

    if not candles:
        raise ValueError(f"No data for {symbol}")

    data = {
        "open":   [float(c[1]) for c in candles],
        "high":   [float(c[3]) for c in candles],
        "low":    [float(c[4]) for c in candles],
        "close":  [float(c[2]) for c in candles],
        "volume": [float(c[5]) for c in candles],
    }

    with open(path, "wb") as f:
        pickle.dump(data, f)

    return data


def simulate_trade(setup: dict, future_candles: dict, window: int) -> str:
    """
    Walk forward `window` candles after entry, return outcome:
    'TP2', 'TP1', 'SL', or 'EXPIRED'.

    Uses the same structure-based calculate_tp_sl as the live bot so the
    backtest reflects real TP/SL placement.
    """
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

    highs = future_candles["high"][:window]
    lows  = future_candles["low"][:window]

    for h, l in zip(highs, lows):
        if direction == "LONG":
            if l <= sl:  return "SL"
            if h >= tp2: return "TP2"
            if h >= tp1: return "TP1"
        else:
            if h >= sl:  return "SL"
            if l <= tp2: return "TP2"
            if l <= tp1: return "TP1"

    return "EXPIRED"


def backtest_symbol(symbol: str) -> dict:
    """Run backtest on a single symbol. Returns stats dict."""
    print(f"  {symbol}...", end=" ", flush=True)

    try:
        c15 = fetch_history(symbol, TIMEFRAME_KUCOIN,    KLINES_INTERVAL_SEC,    BACKTEST_CANDLES)
        c1h = fetch_history(symbol, TIMEFRAME_1H_KUCOIN, KLINES_1H_INTERVAL_SEC, BACKTEST_CANDLES // 4)
        c4h = fetch_history(symbol, TIMEFRAME_4H_KUCOIN, KLINES_4H_INTERVAL_SEC, BACKTEST_CANDLES // 16)
    except Exception as e:
        print(f"skip ({e})")
        return {"trades": 0, "tp1": 0, "tp2": 0, "sl": 0, "expired": 0}

    n = len(c15["close"])
    trades = tp1 = tp2 = sl_hit = expired = 0

    # Analysis only looks back ~50-70 candles; cap snapshot length so each
    # call is near-constant time instead of growing to the full history.
    # (Quadratic Stoch-RSI/divergence cost dominated runtime otherwise.)
    W15, W1H, W4H = 300, 90, 50

    # Slide a window forward through the data
    for i in range(50, n - BACKTEST_TP_WINDOW):
        # Build a "candles up to i" snapshot, capped to the lookback window
        snap_15 = {k: v[max(0, i - W15):i] for k, v in c15.items()}

        # Approximate 1h/4h snapshots (rough — assume time aligned)
        i_1h = max(1, i // 4)
        i_4h = max(1, i // 16)
        snap_1h = {k: v[max(0, i_1h - W1H):i_1h] for k, v in c1h.items()} if i_1h <= len(c1h["close"]) else c1h
        snap_4h = {k: v[max(0, i_4h - W4H):i_4h] for k, v in c4h.items()} if i_4h <= len(c4h["close"]) else c4h

        setup = analyze_coin_smc(snap_15, snap_1h, symbol, snap_4h, btc_change_pct=0.0)
        if not setup:
            continue

        # Simulate from entry
        future = {k: v[i:i + BACKTEST_TP_WINDOW] for k, v in c15.items()}
        outcome = simulate_trade(setup, future, BACKTEST_TP_WINDOW)

        trades += 1
        if   outcome == "TP1":     tp1 += 1
        elif outcome == "TP2":     tp2 += 1
        elif outcome == "SL":      sl_hit += 1
        else:                      expired += 1

    print(f"trades={trades}, TP1={tp1}, TP2={tp2}, SL={sl_hit}, EXP={expired}")
    return {"trades": trades, "tp1": tp1, "tp2": tp2, "sl": sl_hit, "expired": expired}


# Fixed symbol set — pinned so every A/B run uses the SAME coins.
# (Live get_top_coins() reshuffles by 24h volume each run, which contaminated
#  filter comparisons: a big winner like SUI drifting in/out swung the totals.)
BACKTEST_SYMBOLS = [
    "BTC-USDT", "ETH-USDT", "XRP-USDT", "SOL-USDT", "XMR-USDT",
    "DOT-USDT", "XLM-USDT", "LINK-USDT", "SUI-USDT", "HYPE-USDT",
    "ZEC-USDT", "SEI-USDT", "AAVE-USDT", "TAO-USDT", "NEAR-USDT",
    "TON-USDT", "BILL-USDT", "LAB-USDT", "PIEVERSE-USDT", "NEX-USDT",
]


def main():
    env_syms = os.getenv("BACKTEST_SYMBOLS", "").strip()
    if env_syms:
        coins = [s.strip().upper() for s in env_syms.split(",") if s.strip()]
        print(f"Using {len(coins)} symbols from BACKTEST_SYMBOLS env...")
    else:
        coins = BACKTEST_SYMBOLS
        print(f"Using {len(coins)} pinned symbols (reproducible A/B)...")
    print(f"Backtesting {len(coins)} coins on last ~21 days of 15m data...\n")

    total = {"trades": 0, "tp1": 0, "tp2": 0, "sl": 0, "expired": 0}
    for sym in coins:
        s = backtest_symbol(sym)
        for k in total:
            total[k] += s[k]
        time.sleep(0.5)

    print("\n" + "=" * 50)
    print("BACKTEST RESULTS")
    print("=" * 50)
    print(f"Total trades: {total['trades']}")
    print(f"  TP1 hit:    {total['tp1']}")
    print(f"  TP2 hit:    {total['tp2']}")
    print(f"  SL hit:     {total['sl']}")
    print(f"  Expired:    {total['expired']}")

    if total["trades"] > 0:
        wins = total["tp1"] + total["tp2"]
        win_rate = wins / total["trades"] * 100
        # Structure-based R-multiples (TP1=2.5R, TP2=5R), 50% close at TP1 then SL→BE:
        #   TP2  → 0.5*2.5R + 0.5*5R = +3.75R
        #   TP1  → 0.5*2.5R + 0.5*0  = +1.25R (rest stopped at breakeven)
        #   SL   → -1.0R
        #   EXP  →  0
        pnl = total["tp2"] * 3.75 + total["tp1"] * 1.25 - total["sl"] * 1.0
        print(f"\nWin rate:   {win_rate:.1f}%  (lower than scalping — high R:R is normal)")
        print(f"Expected R: {pnl:+.1f}R total ({pnl / total['trades']:+.2f}R per trade)")


if __name__ == "__main__":
    main()
