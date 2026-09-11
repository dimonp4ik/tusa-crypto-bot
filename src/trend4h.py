"""4h breakout trend strategy with a BTC calm-market gate.

Research: reports/NIGHT_2026_09_11.md. Every rule here reads CLOSED bars only:

  * 4h bars are built from 15m candles and used only when all 16 are present.
  * Entry signal on closed 4h bar i: close breaks the high (long) / low (short)
    of the previous N bars, and EMA50 of 4h closes agrees over the last 6 bars.
  * Gate: |BTC 20-day return| < BTC_MOM_MAX, measured on the last CLOSED daily
    bar before the entry bar opens. Breakouts out of a calm market start new
    trends; breakouts after BTC has already run are late.
  * Entry at the next 4h open. Initial stop STOP_ATR x ATR14(4h).
  * Trailing stop TRAIL_ATR x ATR from the best 4h close, never loosened. The
    stop is checked against a bar BEFORE that bar's close moves it.

The module is pure: no exchange calls. The live wiring and the research
backtest both call it, so they cannot drift apart.
"""
from __future__ import annotations

import bisect
from dataclasses import dataclass

import numpy as np

BAR_SEC = 14400
N_BREAKOUT = 20
EMA_LEN = 50
EMA_SLOPE_BARS = 6
ATR_LEN = 14
STOP_ATR = 2.0
TRAIL_ATR = 3.0
BTC_MOM_DAYS = 20
BTC_MOM_MAX = 0.075
ENTRY_ADVERSE = 0.0008
ROUND_TRIP_COST = 0.0004


def build_4h(times, opens, highs, lows, closes):
    """Aggregate 15m candles (seconds) into complete 4h bars."""
    bars = {}
    for t, o, h, l, c in zip(times, opens, highs, lows, closes):
        k = int(t) - int(t) % BAR_SEC
        b = bars.get(k)
        if b is None:
            bars[k] = [o, h, l, c, 1]
        else:
            b[1] = max(b[1], h); b[2] = min(b[2], l); b[3] = c; b[4] += 1
    keys = [k for k in sorted(bars) if bars[k][4] == BAR_SEC // 900]
    arr = np.array([bars[k][:4] for k in keys], dtype=float).reshape(-1, 4)
    return np.array(keys, dtype=np.int64), arr


def indicators(bars):
    o, h, l, c = bars.T
    n = len(c)
    pc = np.r_[c[0], c[:-1]] if n else c
    tr = np.maximum(h - l, np.maximum(np.abs(h - pc), np.abs(l - pc)))
    atr = np.convolve(tr, np.ones(ATR_LEN) / ATR_LEN, "full")[:n]
    atr[:ATR_LEN] = np.nan
    ema = np.zeros(n)
    if n:
        a = 2 / (EMA_LEN + 1); ema[0] = c[0]
        for i in range(1, n):
            ema[i] = a * c[i] + (1 - a) * ema[i - 1]
    return atr, ema


def btc_momentum(btc_times, btc_closes, ts):
    """20-day BTC return on the last daily bar that had CLOSED by ts."""
    i = bisect.bisect_right(btc_times, ts - 86400) - 1
    if i < BTC_MOM_DAYS + 1:
        return None
    return btc_closes[i] / btc_closes[i - BTC_MOM_DAYS] - 1


def entry_signal(times, bars, atr, ema, i):
    """'LONG' / 'SHORT' / None for CLOSED bar i (entry would be at bar i+1)."""
    if i < max(N_BREAKOUT, 60) or np.isnan(atr[i]):
        return None
    if times[i] - times[i - N_BREAKOUT] != N_BREAKOUT * BAR_SEC:
        return None
    h, l, c = bars[:, 1], bars[:, 2], bars[:, 3]
    if c[i] > h[i - N_BREAKOUT:i].max() and ema[i] > ema[i - EMA_SLOPE_BARS]:
        return "LONG"
    if c[i] < l[i - N_BREAKOUT:i].min() and ema[i] < ema[i - EMA_SLOPE_BARS]:
        return "SHORT"
    return None


@dataclass
class Position:
    side: str
    entry: float
    risk: float
    stop: float
    best: float


def open_position(side, open_px, atr_at_signal):
    e = open_px * (1 + ENTRY_ADVERSE if side == "LONG" else 1 - ENTRY_ADVERSE)
    risk = STOP_ATR * atr_at_signal
    stop = e - risk if side == "LONG" else e + risk
    return Position(side, e, risk, stop, e)


def step(pos, bar, atr_now):
    """Advance one CLOSED 4h bar. Returns exit R (costs included) or None."""
    _o, h, l, c = bar
    if pos.side == "LONG":
        if l <= pos.stop:
            return (pos.stop - pos.entry) / pos.risk - ROUND_TRIP_COST * pos.entry / pos.risk
        pos.best = max(pos.best, c)
        pos.stop = max(pos.stop, pos.best - TRAIL_ATR * atr_now)
    else:
        if h >= pos.stop:
            return (pos.entry - pos.stop) / pos.risk - ROUND_TRIP_COST * pos.entry / pos.risk
        pos.best = min(pos.best, c)
        pos.stop = min(pos.stop, pos.best + TRAIL_ATR * atr_now)
    return None


def advance(state, times, bars, btc_times, btc_closes, *, gate=True):
    """Process every CLOSED 4h bar not yet seen; return events.

    This is the LIVE code path and the backtest calls it too (see simulate), so
    the two cannot disagree. One position per symbol: an exchange in one-way
    mode cannot hold a long and a short on the same contract at once.

    state (mutated): {"last": ts of last processed bar, "pos": dict|None,
                      "pending": [side, atr_at_signal]|None}
    events: ("OPEN", ts, side, entry, stop), ("STOP", ts, new_stop),
            ("EXIT", ts, side, r)
    """
    atr, ema = indicators(bars)
    ev = []
    last = state.get("last")
    start = 0 if last is None else int(np.searchsorted(times, last, "right"))
    for j in range(start, len(bars)):
        state["last"] = int(times[j])
        exited = False
        if state.get("pending") and state.get("pos") is None:
            side, a = state["pending"]
            state["pending"] = None
            p = open_position(side, bars[j][0], a)
            state["pos"] = {"side": p.side, "entry": p.entry, "risk": p.risk,
                            "stop": p.stop, "best": p.best, "open_ts": int(times[j])}
            ev.append(("OPEN", int(times[j]), p.side, p.entry, p.stop))
        pd = state.get("pos")
        if pd is not None:
            p = Position(pd["side"], pd["entry"], pd["risk"], pd["stop"], pd["best"])
            old = p.stop
            res = step(p, bars[j], atr[j])
            if res is not None:
                ev.append(("EXIT", int(times[j]) + BAR_SEC, p.side, float(res), pd["open_ts"]))
                state["pos"] = None
                exited = True
            else:
                pd.update(stop=p.stop, best=p.best)
                if p.stop != old:
                    ev.append(("STOP", int(times[j]), p.stop))
        if state.get("pos") is None and not exited and not state.get("pending"):
            sig = entry_signal(times, bars, atr, ema, j)
            if sig is not None and gate:
                m = btc_momentum(btc_times, btc_closes, int(times[j]) + BAR_SEC)
                if m is None or abs(m) >= BTC_MOM_MAX:
                    sig = None
            if sig is not None:
                state["pending"] = [sig, float(atr[j])]
    return ev


def simulate(times, bars, btc_times, btc_closes, *, gate=True):
    """Backtest = the live path fed one closed bar at a time (batch here, which
    is equivalent because indicators only read the past). Returns closed trades
    as (entry_ts, exit_ts, r, side)."""
    state = {}
    out = []
    for e in advance(state, times, bars, btc_times, btc_closes, gate=gate):
        if e[0] == "EXIT":
            out.append((e[4], e[1], e[3], e[2]))
    return out
