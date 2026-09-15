"""Reversals from weakness: the class of move the system misses.

The missed-move analysis showed the system catches big up-moves only when they start inside
strength (rsi48 ~57, LONG regime). The big up-moves it misses start from weakness: the coin fell over
24h, rsi6 is oversold, price sits near the bottom of its range, BTC is falling, and the regime is
absent or SHORT. Every reversal rule searched before was locked inside the LONG regime; this one is
not - no regime requirement and no BTC filter, since the missed moves start while BTC is down.

Pairs of conditions drawn from that profile are tested for LONG, and the mirror profile (overbought,
near the top of range, coin up) for SHORT. Outcomes for every hour are simulated once - market entry
at the next 15m open, stop 3 / take 1 ATR, 48h, measured costs both legs, stops beyond 10% dropped -
and each pair's book applies one position per coin. Null: every valid hour. Mirror: the pair traded
the other way. Selection is blind: pairs are chosen on four years and read on the fifth.

Thresholds are pooled quantiles over all years - they describe the shape of a condition, and the
blind protocol decides which pairs are kept.
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
from src import pullback_bank as PB

cost = collections.defaultdict(list)
for r in csv.DictReader(open('book_frozen.csv')):
    cost[r['coin'] + 'USDT'].append(float(r['cost109']))
SLIP = {k: float(np.median(v)) for k, v in cost.items()}
FEE = 0.0004
COINS = [c for c in SP.COINS if c not in ('BILLUSDT', 'XLMUSDT', 'AAVEUSDT')]
YEARS = [2022, 2023, 2024, 2025, 2026]
YT = {y: int(datetime.datetime(y, 1, 1, tzinfo=datetime.UTC).timestamp()) for y in YEARS + [2027]}
H = 192
BIG = 10 ** 9
WARM = 800

FE, OUT = {}, {}
for s in COINS:
    c = SP.CTX[s]
    T1, B1, F = c['T1'], c['B1'], c['F']
    cl = B1[:, 3]
    FE[s] = dict(T1=T1, atr=F['atr'], rsi6=PB._rsi(cl, 6), rsi48=PB._rsi(cl, 48), rsi14=F['rsi14'],
                 rng=F['rng'], ret24=F['ret24'], volreg=F['volreg'], btc24=F['btc24'], slope=F['slope'])
    t15, a15, pos = c['t15'], c['a15'], c['pos']
    slip = SLIP.get(s, 0.0003)
    res = {True: {}, False: {}}
    for i in range(WARM, len(T1)):
        atr = F['atr'][i]
        if not np.isfinite(atr) or atr <= 0:
            continue
        close = int(T1[i]) + 3600
        j = pos.get(close)
        if j is None or j + H > len(t15):
            continue
        o, h, l, cc = (a15[j:j + H, x] for x in range(4))
        for lg in (True, False):
            e = o[0] * (1 + slip) if lg else o[0] * (1 - slip)
            sf = 3.0 * atr / e
            if sf > 0.10:
                continue
            TP, SL = (e + atr, e - 3 * atr) if lg else (e - atr, e + 3 * atr)
            hs, ht = (l <= SL, h >= TP) if lg else (h >= SL, l <= TP)
            js = int(np.argmax(hs)) if hs.any() else BIG
            jt = int(np.argmax(ht)) if ht.any() else BIG
            if js <= jt and js < BIG:
                jj, f = js, (min(SL, o[js]) * (1 - slip) if lg else max(SL, o[js]) * (1 + slip))
            elif jt < BIG:
                jj, f = jt, (TP * (1 - slip) if lg else TP * (1 + slip))
            else:
                jj, f = H - 1, (cc[-1] * (1 - slip) if lg else cc[-1] * (1 + slip))
            ret = ((f / e - 1) if lg else (1 - f / e)) - FEE
            res[lg][i] = (close, int(t15[j + jj]) + 900, ret / sf, sf)
    OUT[s] = res
    print('  %-9s исходы посчитаны: LONG %d, SHORT %d' % (s, len(res[True]), len(res[False])), flush=True)

NAMES = ['rsi6', 'rsi48', 'rsi14', 'rng', 'ret24', 'volreg', 'btc24', 'slope']
Q = {}
for n in NAMES:
    v = np.concatenate([FE[s][n][WARM:] for s in COINS])
    v = v[np.isfinite(v)]
    Q[n] = {q: float(np.quantile(v, q)) for q in (0.05, 0.1, 0.2, 0.8, 0.9, 0.95)}

CONDS = {True: [], False: []}
for n, qs in (('rsi6', (0.05, 0.1, 0.2)), ('rsi48', (0.1, 0.2)), ('rsi14', (0.1, 0.2)), ('rng', (0.05, 0.1, 0.2)),
              ('ret24', (0.05, 0.1, 0.2)), ('slope', (0.1, 0.2)), ('btc24', (0.1, 0.2))):
    for q in qs:
        CONDS[True].append((n, '<=', Q[n][q], q))
        CONDS[False].append((n, '>=', Q[n][1 - q], 1 - q))
for q in (0.8, 0.9):
    CONDS[True].append(('volreg', '>=', Q['volreg'][q], q))
    CONDS[False].append(('volreg', '>=', Q['volreg'][q], q))


def mask(s, cond):
    n, op, thr, q = cond
    x = FE[s][n]
    with np.errstate(invalid='ignore'):
        return (x <= thr) if op == '<=' else (x >= thr)


MASKS = {(lg, k): {s: mask(s, cond) for s in COINS} for lg in (True, False) for k, cond in enumerate(CONDS[lg])}


def book(lg, idx_by_coin, side_out):
    rows = []
    for s in COINS:
        busy = 0
        res = OUT[s][side_out]
        for i in idx_by_coin[s]:
            r = res.get(int(i))
            if r is None or r[0] < busy:
                continue
            rows.append((r[0], r[1], r[2], r[3], s))
            busy = r[1]
    rows.sort()
    return rows


def stats(rows):
    if len(rows) < 100:
        return None
    v = np.array([x[2] for x in rows])
    yrs = []
    for y in YEARS:
        vy = [x[2] for x in rows if YT[y] <= x[0] < YT[y + 1]]
        yrs.append(float(np.mean(vy)) if len(vy) >= 20 else float('nan'))
    years = (rows[-1][0] - rows[0][0]) / (365.25 * 86400)
    return dict(n=len(v), tpy=len(v) / years, wr=float(np.mean(v > 0)), avg=float(v.mean()), yrs=yrs)


NULL = {}
for lg in (True, False):
    allidx = {s: np.array(sorted(OUT[s][lg])) for s in COINS}
    st = stats(book(lg, allidx, lg))
    NULL[lg] = st
    print('  нуль %s: %4.0f сделок/год ВР %.1f%% ср %+.4f по годам %s'
          % ('LONG' if lg else 'SHORT', st['tpy'], 100 * st['wr'], st['avg'],
             ' '.join('%+.3f' % y for y in st['yrs'])), flush=True)

ALL = {}
for lg in (True, False):
    K = len(CONDS[lg])
    for a, b in itertools.combinations(range(K), 2):
        if CONDS[lg][a][0] == CONDS[lg][b][0]:
            continue
        idx = {s: np.flatnonzero(MASKS[(lg, a)][s] & MASKS[(lg, b)][s]) for s in COINS}
        rows = book(lg, idx, lg)
        st = stats(rows)
        if st is None:
            continue
        mir = stats(book(lg, idx, not lg))
        ALL[(lg, a, b)] = (st, rows, mir)
print('  пар с 100+ сделками: %d' % len(ALL), flush=True)


def label(lg, a, b):
    ca, cb = CONDS[lg][a], CONDS[lg][b]
    return '%s %s%s%.3g(q%.2f) + %s%s%.3g(q%.2f)' % ('LONG' if lg else 'SHORT', ca[0], ca[1], ca[2], ca[3], cb[0], cb[1], cb[2], cb[3])


for lg in (True, False):
    keys = [k for k in ALL if k[0] == lg and ALL[k][0]['tpy'] >= 60]
    keys.sort(key=lambda k: -ALL[k][0]['avg'])
    print('', flush=True)
    print('  ===== %s: лучшие пары по всей истории (60+ сделок/год) =====' % ('LONG' if lg else 'SHORT'), flush=True)
    for k in keys[:15]:
        st, rows, mir = ALL[k]
        pos = sum(1 for y in st['yrs'] if np.isfinite(y) and y > 0)
        print('    %-62s %4.0f/год ВР %.1f%% ср %+.4f | годы %s (%d/5) | зеркало %s'
              % (label(*k), st['tpy'], 100 * st['wr'], st['avg'], ' '.join('%+.3f' % y if np.isfinite(y) else '  -   ' for y in st['yrs']),
                 pos, ('%+.4f' % mir['avg']) if mir else '-'), flush=True)
    print('  -- слепой отбор %s: пара берётся, если на 4 годах ср R > нуль+0.05, 150+ сделок, плюс в 3+ годах --' % ('LONG' if lg else 'SHORT'), flush=True)
    for hy in YEARS:
        lo, hi = YT[hy], YT[hy + 1]
        base = NULL[lg]['avg']
        chosen = []
        for k in [k for k in ALL if k[0] == lg]:
            st, rows, mir = ALL[k]
            tr = [x[2] for x in rows if not (lo <= x[0] < hi)]
            if len(tr) < 150:
                continue
            ty = [np.mean([x[2] for x in rows if YT[y] <= x[0] < YT[y + 1]]) for y in YEARS
                  if y != hy and sum(1 for x in rows if YT[y] <= x[0] < YT[y + 1]) >= 20]
            if np.mean(tr) > base + 0.05 and sum(1 for m in ty if m > 0) >= 3:
                chosen.append(k)
        held = [np.mean([x[2] for x in ALL[k][1] if lo <= x[0] < hi]) for k in chosen
                if sum(1 for x in ALL[k][1] if lo <= x[0] < hi) >= 20]
        nul = NULL[lg]['yrs'][YEARS.index(hy)]
        print('    %d выбрано пар %3d | слепой год: ср R выбранных %s (нуль %+.4f), в плюсе %d из %d'
              % (hy, len(chosen), ('%+.4f' % np.mean(held)) if held else '  -  ', nul,
                 sum(1 for m in held if m > 0), len(held)), flush=True)

pickle.dump({k: (v[0], v[1]) for k, v in ALL.items()}, open('reversal_pairs.pkl', 'wb'))
pickle.dump(CONDS, open('reversal_conds.pkl', 'wb'))
