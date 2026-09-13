"""A wider feature set than the bank's nine, built causally on closed hourly bars.

The bank looks at nine things. Seven mining attempts on those nine have failed honest validation, so
the next attempt should not be the eighth pass over the same ground with the same method. Two things
change here.

First the inputs: about forty features instead of nine, including three kinds the bank has never
seen - market breadth (how many of the sixteen coins are rising at this hour, which needs all coins
aligned and is the only genuinely cross-sectional signal available), volume structure, and BTC
context at several horizons rather than one.

Second, and more important, the evaluation, which lives in the next file: edge over a side-matched
random entry rather than raw return, and the walk-forward run FIRST rather than last. Every previous
candidate died on the walk-forward after passing everything else; running it first turns a month of
wasted validation into a fast rejection.

Everything here reads only bars at or before i. The breadth features use each coin's own closed bar
at the same timestamp, which is available live because every coin's hour closes together.
"""
import sys

import numpy as np

sys.path.insert(0, 'C:/Users/Lenovo/Desktop/Торговля/crypto-bot')
import struct_params as SP
from src import pullback_bank as PB


def _lag(x, n):
    o = np.full(len(x), np.nan)
    if n < len(x):
        o[n:] = x[:-n]
    return o


def _roll(x, n, fn):
    o = np.full(len(x), np.nan)
    for i in range(n - 1, len(x)):
        o[i] = fn(x[i - n + 1:i + 1])
    return o


def _rsi(c, n):
    return PB._rsi(c, n)


def build(coin):
    """~40 causal features on this coin's closed hourly bars."""
    c = SP.CTX[coin]
    T1, B1, F = c['T1'], c['B1'], c['F']
    o, h, l, cl, v = B1[:, 0], B1[:, 1], B1[:, 2], B1[:, 3], B1[:, 4]
    at = F['atr']
    out = {}

    # --- momentum at several horizons, normalised by ATR so coins are comparable
    for n in (1, 3, 6, 12, 24, 48, 96, 168):
        out['ret%d' % n] = (cl - _lag(cl, n)) / at

    # --- where price sits in its own recent range
    for n in (12, 24, 48, 96):
        hi = _lag(_roll(h, n, np.max), 1)
        lo = _lag(_roll(l, n, np.min), 1)
        out['pos%d' % n] = (cl - lo) / (hi - lo + 1e-12)

    # --- volatility regime: current ATR against its own history
    atrp = at / cl
    for n in (24, 168, 720):
        out['vol%d' % n] = atrp / _roll(atrp, n, np.median)

    # --- RSI family
    for n in (2, 6, 14, 48):
        out['rsi%d' % n] = _rsi(cl, n)

    # --- candle structure of the closed bar
    rng = (h - l) + 1e-12
    out['body'] = (cl - o) / rng
    out['upwick'] = (h - np.maximum(o, cl)) / rng
    out['dnwick'] = (np.minimum(o, cl) - l) / rng
    out['range_atr'] = rng / at

    # --- volume structure
    vmed = _roll(v, 168, np.median)
    out['vol_rel'] = v / (vmed + 1e-12)
    out['vol_trend'] = _roll(v, 6, np.mean) / (_roll(v, 48, np.mean) + 1e-12)

    # --- trend shape
    e50 = PB._ema(cl, 50)
    e200 = PB._ema(cl, 200)
    out['ema_gap'] = (e50 - e200) / at
    out['ema_slope'] = (e200 - _lag(e200, 24)) / at
    out['dist_e50'] = (cl - e50) / at

    # --- how far since the last extreme
    out['bars_since_hi'] = _roll(h, 48, lambda w: len(w) - 1 - int(np.argmax(w)))
    out['bars_since_lo'] = _roll(l, 48, lambda w: len(w) - 1 - int(np.argmin(w)))

    # --- BTC context at several horizons (the bank has one)
    bT1, bB1 = PB.build_bars(SP.CTX['BTCUSDT']['t15'], SP.CTX['BTCUSDT']['a15'], PB.HOUR)
    bmap = dict(zip(bT1.tolist(), bB1[:, 3]))
    bc = np.array([bmap.get(int(t), np.nan) for t in T1])
    for n in (6, 24, 72, 168):
        prev = np.array([bmap.get(int(t) - n * 3600, np.nan) for t in T1])
        out['btc%d' % n] = bc / prev - 1
    bat = PB._atr(bB1)
    batmap = dict(zip(bT1.tolist(), bat))
    out['btc_vol'] = np.array([batmap.get(int(t), np.nan) for t in T1]) / bc

    # --- relative strength against BTC
    out['rs24'] = out['ret24'] - (bc / np.array([bmap.get(int(t) - 24 * 3600, np.nan)
                                                 for t in T1]) - 1) * cl / at
    out['hour'] = (T1 // PB.HOUR) % 24
    return T1, out


def breadth(coins):
    """How many coins are up over 24h at each hour - the only cross-sectional signal available."""
    per = {}
    for s in coins:
        T1, f = build(s)
        per[s] = dict(zip(T1.tolist(), f['ret24'].tolist()))
    stamps = sorted(set().union(*[set(d) for d in per.values()]))
    out = {}
    for t in stamps:
        vals = [per[s][t] for s in coins if t in per[s] and np.isfinite(per[s][t])]
        if len(vals) >= 8:
            out[t] = (float(np.mean([x > 0 for x in vals])), float(np.median(vals)))
    return out


if __name__ == '__main__':
    T1, f = build('SOLUSDT')
    ok = [k for k, v in f.items() if np.isfinite(v[PB.MIN_HOURS:]).mean() > 0.9]
    print('  признаков: %d, из них годных: %d' % (len(f), len(ok)), flush=True)
    print('  ' + ', '.join(sorted(f)), flush=True)
