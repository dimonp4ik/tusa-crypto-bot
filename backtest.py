"""
Backtest the SMC strategy on historical KuCoin data.

Usage:  python backtest.py
Outputs: total trades, win rate, avg R:R, max drawdown.
"""

import time
import sys
import os
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import requests
from config import (
    KUCOIN_BASE_URL, TIMEFRAME_KUCOIN, KLINES_INTERVAL_SEC,
    TIMEFRAME_1H_KUCOIN, KLINES_1H_INTERVAL_SEC,
    TIMEFRAME_4H_KUCOIN, KLINES_4H_INTERVAL_SEC,
    BACKTEST_CANDLES, BACKTEST_TP_WINDOW,
    ATR_SL_MULT, ATR_TP1_MULT, ATR_TP2_MULT,
)
from src.signal_filter import analyze_coin_smc
from src.binance_client import get_top_coins


def fetch_history(symbol: str, interval: str, interval_sec: int, count: int) -> dict:
    """Fetch `count` historical candles."""
    url = f"{KUCOIN_BASE_URL}/api/v1/market/candles"
    now = int(time.time())
    start_at = now - count * interval_sec

    params = {"symbol": symbol, "type": interval, "startAt": start_at, "endAt": now}
    resp = requests.get(url, params=params, timeout=15)
    resp.raise_for_status()
    candles = list(reversed(resp.json().get("data", [])))

    if not candles:
        raise ValueError(f"No data for {symbol}")

    return {
        "open":   [float(c[1]) for c in candles],
        "high":   [float(c[3]) for c in candles],
        "low":    [float(c[4]) for c in candles],
        "close":  [float(c[2]) for c in candles],
        "volume": [float(c[5]) for c in candles],
    }


def simulate_trade(direction: str, entry: float, atr: float, future_candles: dict,
                   window: int) -> str:
    """
    Walk forward `window` candles after entry, return outcome:
    'TP2', 'TP1', 'SL', or 'EXPIRED'.
    """
    if atr <= 0:
        return "EXPIRED"

    if direction == "LONG":
        sl  = entry - atr * ATR_SL_MULT
        tp1 = entry + atr * ATR_TP1_MULT
        tp2 = entry + atr * ATR_TP2_MULT
    else:
        sl  = entry + atr * ATR_SL_MULT
        tp1 = entry - atr * ATR_TP1_MULT
        tp2 = entry - atr * ATR_TP2_MULT

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

    # Slide a window forward through the data
    for i in range(50, n - BACKTEST_TP_WINDOW):
        # Build a "candles up to i" snapshot
        snap_15 = {k: v[:i] for k, v in c15.items()}

        # Approximate 1h/4h snapshots (rough — assume time aligned)
        i_1h = max(1, i // 4)
        i_4h = max(1, i // 16)
        snap_1h = {k: v[:i_1h] for k, v in c1h.items()} if i_1h <= len(c1h["close"]) else c1h
        snap_4h = {k: v[:i_4h] for k, v in c4h.items()} if i_4h <= len(c4h["close"]) else c4h

        setup = analyze_coin_smc(snap_15, snap_1h, symbol, snap_4h, btc_change_pct=0.0)
        if not setup:
            continue

        # Simulate from entry
        entry  = snap_15["close"][-1]
        atr    = setup.get("atr", 0)
        future = {k: v[i:i + BACKTEST_TP_WINDOW] for k, v in c15.items()}
        outcome = simulate_trade(setup["direction"], entry, atr, future, BACKTEST_TP_WINDOW)

        trades += 1
        if   outcome == "TP1":     tp1 += 1
        elif outcome == "TP2":     tp2 += 1
        elif outcome == "SL":      sl_hit += 1
        else:                      expired += 1

    print(f"trades={trades}, TP1={tp1}, TP2={tp2}, SL={sl_hit}, EXP={expired}")
    return {"trades": trades, "tp1": tp1, "tp2": tp2, "sl": sl_hit, "expired": expired}


def main():
    print("Fetching top coins...")
    coins = get_top_coins()[:20]  # backtest top 20 to keep it fast
    print(f"Backtesting {len(coins)} coins on last ~10 days of 15m data...\n")

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
        # Approx P&L assuming 1R risk per trade, partial at TP1 (1R), rest at TP2 (2R)
        # TP1-only ≈ +1R*0.5 + (-1R)*0.5 = 0 (breakeven on the rest)
        # Actually if we move SL to BE after TP1, TP1-only ≈ +0.5R, TP2 ≈ +1.5R, SL ≈ -1R
        pnl = total["tp2"] * 1.5 + total["tp1"] * 0.5 - total["sl"] * 1.0
        print(f"\nWin rate:   {win_rate:.1f}%")
        print(f"Expected R: {pnl:+.1f}R total ({pnl / total['trades']:+.2f}R per trade)")


if __name__ == "__main__":
    main()
