"""Does the pullback rule need BTC's daily filter, or does the filter just cost it trades?

The search on the blocked pool keeps returning rsi48 >= 56.6 - the pullback rule's own first
condition - which suggests the rule pays on hours the BTC gate refuses. If so, the gate is not
protecting this rule, it is taxing it.

Four variants, each measured on its own and then as money on top of the bank, which keeps its own
gate untouched in every case:

    как сейчас     - the rule inside the full gate, BTC filter included
    без BTC-фильтра - the regime interval still required, BTC's daily close ignored
    только сбитые  - ONLY the hours the BTC filter refuses, so the two halves can be compared
    без режима     - no regime interval either; the rule stands alone

Blind years for each, because "more trades at a similar edge" is exactly how a rule quietly rots.
"""
import bisect
import collections
import csv
import datetime
import sys

import numpy as np

sys.path.insert(0, 'C:/Users/Lenovo/Desktop/Торговля/crypto-bot')
import struct_params as SP
import feat2
import gauntlet2 as G
import stop_width as SW
import sim_w
from src import pullback_bank as PB

meas = collections.defaultdict(list)
for r in csv.DictReader(open('book_samples.csv')):
    meas[r['coin'] + 'USDT'].append(float(r['cost109']))
SP.SLIP_REAL.update({k: float(np.median(v)) for k, v in meas.items()})
G.FC.clear()

C = [('rsi48', '>=', 56.3761), ('rsi6', '<=', 33.7947)]
BIG = 10 ** 9
ALL = [c for c in G.ALL if c != 'AAVEUSDT']
YT = {y: int(datetime.datetime(y, 1, 1, tzinfo=datetime.UTC).timestamp()) for y in range(2022, 2028)}


def signals(coin, mode):
    c = SP.CTX[coin]
    T1, f = G.feats(coin)
    bsm = c['sma'].setdefault(50, PB._sma(c['bdc'], 50))
    out = {}
    for i in range(PB.MIN_HOURS, len(T1)):
        close = int(T1[i]) + PB.HOUR
        j = bisect.bisect_right(c['starts'], close) - 1
        inside = j >= 0 and c['iv'][j][0] <= close < c['iv'][j][1] and c['iv'][j][2] == 'LONG'
        d = bisect.bisect_right(c['bdt'], close - 86400) - 1
        blocked = d >= 49 and c['bdc'][d] < bsm[d]
        if mode == 'как сейчас' and not (inside and not blocked):
            continue
        if mode == 'без BTC-фильтра' and not inside:
            continue
        if mode == 'только сбитые' and not (inside and blocked):
            continue
        if mode == 'без режима' and blocked:
            continue
        if not np.isfinite(c['F']['atr'][i]):
            continue
        ok = True
        for nm, op, thr in C:
            x = f[nm][i]
            if not np.isfinite(x) or (x < thr if op == '>=' else x > thr):
                ok = False
                break
        if ok:
            out[close] = (float(c['B1'][i, 3]), float(c['F']['atr'][i]))
    return out


def trades(mode, coins=None):
    rows = []
    for s in (coins or ALL):
        sig = signals(s, mode)
        c = SP.CTX[s]
        slip = SP.SLIP_REAL.get(s, 0.0002)
        t15, A15, pos = c['t15'], c['a15'], c['pos']
        busy = 0
        for close in sorted(sig):
            if close < busy:
                continue
            lim, atr = sig[close]
            i = pos.get(close)
            if i is None or i + PB.HOLD_BARS > len(t15):
                continue
            o, h, l, cc = (A15[i:i + PB.HOLD_BARS, x] for x in range(4))
            if o[0] <= lim:
                e, touch = o[0] * (1 + slip), False
            elif l[0] <= lim:
                e, touch = lim * (1 + slip), True
            else:
                continue
            TP, SL = e + 1.0 * atr, e - 3.0 * atr
            hs, ht = l <= SL, h >= TP
            if touch:
                ht = ht.copy()
                ht[0] = False
            js = int(np.argmax(hs)) if hs.any() else BIG
            jt = int(np.argmax(ht)) if ht.any() else BIG
            if js <= jt and js < BIG:
                jj, f_ = js, min(SL, o[js]) * (1 - slip)
            elif jt < BIG:
                jj, f_ = jt, TP * (1 - slip)
            else:
                jj, f_ = PB.HOLD_BARS - 1, cc[-1] * (1 - slip)
            ret = (f_ / e - 1) - 0.0004
            rows.append((close, int(t15[i + jj]) + 900, ret / (3.0 * atr / e), 3.0 * atr / e, s))
            busy = int(t15[i + jj]) + 900
    rows.sort()
    return rows


bank = SW.run(3.0, 1.0, coins=ALL, with_meta=True)


def account(cand):
    ev = [(a, b, R, sf, s, 0) for a, b, R, sf, s in bank] + \
         [(a, b, R, sf, s, 1) for a, b, R, sf, s in cand]
    ev.sort(key=lambda x: (x[0], x[5]))
    busy, out = {}, []
    for a, b, R, sf, s, t in ev:
        if busy.get(s, 0) > a:
            continue
        busy[s] = b
        out.append((a, b, R, sf, s, 1.0))
    out.sort()
    return out


m0 = sim_w.money_at_dd([(a, b, R, sf, s, 1.0) for a, b, R, sf, s in bank], 0.12)[1]
print('  банк один (без AAVE, измеренные издержки): $%.0f' % m0, flush=True)
print('', flush=True)
for mode in ('как сейчас', 'без BTC-фильтра', 'только сбитые', 'без режима'):
    tr = trades(mode)
    v = np.array([r[2] for r in tr]) if tr else np.zeros(0)
    if not len(v):
        print('  %-16s сделок нет' % mode, flush=True)
        continue
    mons = collections.Counter(datetime.datetime.fromtimestamp(r[0], datetime.UTC).strftime('%y-%m')
                               for r in tr)
    acc = account(tr)
    k, m = sim_w.money_at_dd(acc, 0.12)
    yr = []
    for y in range(2022, 2027):
        sub = [r[2] for r in tr if YT[y] <= r[0] < YT[y + 1]]
        yr.append('%+.3f' % np.mean(sub) if len(sub) >= 12 else '  -  ')
    print('  %-16s n%4d ВР%5.1f%% ср%+.4f мес%3d | счёт $%6.0f (%+3.0f%%) | по годам %s'
          % (mode, len(v), 100 * np.mean(v > 0), v.mean(), len(mons), m, 100 * (m / m0 - 1),
             ' '.join(yr)), flush=True)
