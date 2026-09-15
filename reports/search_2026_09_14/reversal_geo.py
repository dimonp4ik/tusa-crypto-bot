"""Reversals from weakness, on a geometry built for reversals.

The reversal search judged every condition with the pullback geometry: stop 3 ATR, take 1 ATR, 48h.
That shape suits a dip inside strength, where the price only has to recover a little. The moves the
system misses average +12% in a day - a 1 ATR take captures a sliver of them while carrying a wide
stop. A reversal wants the opposite shape: a tighter stop, a longer target.

The best LONG pairs from reversal_search are re-simulated on stop 1/1.5/2/3 ATR, take 1/2/3/4 ATR and
hold 24/48/72h, market entry at the next 15m open, measured costs both legs, stops beyond 10% dropped,
one position per coin, no regime requirement and no BTC filter. Each pair's book is judged alone as
money under the live sizing at a fixed 1.4% risk, and the (pair, geometry) choice is made blind on four
years and read on the fifth.
"""
import collections
import csv
import datetime
import itertools
import pickle
import sys

import numpy as np

sys.path.insert(0, 'C:/Users/Lenovo/Desktop/Торговля/crypto-bot')
import struct_params as SP
import sim_w
import live_rules_sim as L
from src import pullback_bank as PB

cost = collections.defaultdict(list)
for r in csv.DictReader(open('book_frozen.csv')):
    cost[r['coin'] + 'USDT'].append(float(r['cost109']))
SLIP = {k: float(np.median(v)) for k, v in cost.items()}
FEE = 0.0004
COINS = [c for c in SP.COINS if c not in ('BILLUSDT', 'XLMUSDT', 'AAVEUSDT')]
YEARS = [2022, 2023, 2024, 2025, 2026]
YT = {y: int(datetime.datetime(y, 1, 1, tzinfo=datetime.UTC).timestamp()) for y in YEARS + [2027]}
WARM, MAXH = 800, 288

CONDS = pickle.load(open('reversal_conds.pkl', 'rb'))
PAIRS = pickle.load(open('reversal_pairs.pkl', 'rb'))
longs = [(k, v[0]) for k, v in PAIRS.items() if k[0] is True and v[0]['tpy'] >= 60]
longs.sort(key=lambda kv: -kv[1]['avg'])
TOP = [k for k, _ in longs[:12]]
print('  разворотных LONG пар на проверку: %d' % len(TOP), flush=True)

FE = {}
for s in COINS:
    c = SP.CTX[s]
    cl = c['B1'][:, 3]
    F = c['F']
    FE[s] = dict(T1=c['T1'], atr=F['atr'], rsi6=PB._rsi(cl, 6), rsi48=PB._rsi(cl, 48), rsi14=F['rsi14'],
                 rng=F['rng'], ret24=F['ret24'], volreg=F['volreg'], btc24=F['btc24'], slope=F['slope'])


def cond_mask(s, cond):
    n, op, thr, q = cond
    x = FE[s][n]
    with np.errstate(invalid='ignore'):
        return (x <= thr) if op == '<=' else (x >= thr)


def signals(key):
    lg, a, b = key
    out = []
    for s in COINS:
        c = SP.CTX[s]
        t15, a15, pos = c['t15'], c['a15'], c['pos']
        slip = SLIP.get(s, 0.0003)
        m = cond_mask(s, CONDS[lg][a]) & cond_mask(s, CONDS[lg][b])
        for i in np.flatnonzero(m):
            atr = FE[s]['atr'][i]
            if i < WARM or not np.isfinite(atr) or atr <= 0:
                continue
            close = int(FE[s]['T1'][i]) + 3600
            j = pos.get(close)
            if j is None or j + MAXH > len(t15):
                continue
            o, h, l, cc = (a15[j:j + MAXH, x] for x in range(4))
            e = o[0] * (1 + slip)
            out.append((close, s, e, atr, np.maximum.accumulate((h - e) / atr), np.maximum.accumulate((e - l) / atr),
                        (o - e) / atr, (cc - e) / atr, t15[j:j + MAXH], slip))
    out.sort(key=lambda x: x[0])
    return out


def book(sig, sl, tp, hh):
    H = hh * 4
    busy, rows = {}, []
    for close, s, e, atr, Mu, Md, on, cn, tt, slip in sig:
        if busy.get(s, 0) > close:
            continue
        sf = sl * atr / e
        if sf > 0.10:
            continue
        js = int(np.searchsorted(Md[:H], sl, side='left'))
        jt = int(np.searchsorted(Mu[:H], tp, side='left'))
        if js < H and js <= jt:
            jj, x = js, min(-sl, on[js])
        elif jt < H:
            jj, x = jt, tp
        else:
            jj, x = H - 1, cn[H - 1]
        end = int(tt[jj]) + 900
        rows.append((close, end, (x * atr / e - slip - FEE) / sf, sf, s, 1.0))
        busy[s] = end
    rows.sort()
    return rows


def money(rows):
    if len(rows) < 30:
        return float('nan'), float('nan')
    r = sim_w.simulate(rows, 0.014)
    return r['eq'], abs(L.dd_of(r['curve']))


GRID = list(itertools.product((1.0, 1.5, 2.0, 3.0), (1.0, 2.0, 3.0, 4.0), (24, 48, 72)))
RES = {}
for key in TOP:
    sig = signals(key)
    ca, cb = CONDS[True][key[1]], CONDS[True][key[2]]
    lbl = '%s<=%.3g + %s%s%.3g' % (ca[0], ca[2], cb[0], cb[1], cb[2])
    best = None
    for g in GRID:
        rows = book(sig, *g)
        eq, dd = money(rows)
        RES[(key, g)] = rows
        if np.isfinite(eq) and (best is None or eq > best[1]):
            best = (g, eq, dd, rows)
    cur = RES[(key, (3.0, 1.0, 48))]
    ceq, cdd = money(cur)
    g, eq, dd, rows = best
    v = np.array([x[2] for x in rows])
    yrs = [money([x for x in rows if YT[y] <= x[0] < YT[y + 1]])[0] for y in YEARS]
    years = (rows[-1][0] - rows[0][0]) / (365.25 * 86400) if rows else 1
    print('  %-34s 3/1/48: $%6.0f %4.1f%% | лучшая %s: $%6.0f %4.1f%% %4.0f сд/год ВР %.1f%% ср %+.3f | годы %s'
          % (lbl, ceq, 100 * cdd, g, eq, 100 * dd, len(v) / years, 100 * np.mean(v > 0), v.mean(),
             ' '.join('%4.0f' % y if np.isfinite(y) else '   -' for y in yrs)), flush=True)

print('', flush=True)
print('  -- слепой выбор (пара, геометрия) по деньгам при 1.4%: 4 года выбор, 5-й замер (старт $120) --', flush=True)
prof = 0
for hy in YEARS:
    lo, hi = YT[hy], YT[hy + 1]
    best, bk = -1, None
    for k, rows in RES.items():
        tr = [x for x in rows if not (lo <= x[0] < hi)]
        if len(tr) < 150:
            continue
        m = sim_w.simulate(tr, 0.014)['eq']
        if m > best:
            best, bk = m, k
    held = [x for x in RES[bk] if lo <= x[0] < hi]
    eq, dd = money(held)
    v = np.array([x[2] for x in held]) if held else np.zeros(0)
    prof += int(np.isfinite(eq) and eq > 120)
    ca, cb = CONDS[True][bk[0][1]], CONDS[True][bk[0][2]]
    print('    %d выбрано %s<=%.3g + %s%s%.3g %s | слепой год: $%s, сделок %d, ср R %s'
          % (hy, ca[0], ca[2], cb[0], cb[1], cb[2], bk[1], ('%.0f' % eq) if np.isfinite(eq) else '-', len(held),
             ('%+.3f' % v.mean()) if len(v) else '-'), flush=True)
print('    слепой выбор в плюсе в %d из 5 лет' % prof, flush=True)
