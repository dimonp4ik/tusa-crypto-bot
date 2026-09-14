"""The hours the gate throws away, which no search in this project has ever seen.

Every candidate so far was forced inside the bank's own preconditions: the hour must fall inside a
regime interval, the side must be the one the regime names, and a long must additionally survive
BTC's daily close being above its SMA(50). That is not a small filter, and nothing has ever asked
what is on the other side of it.

Three separate exclusions, recorded separately because they are different claims:

    вне режима    - the hour falls in no regime interval at all
    против режима - inside an interval, taken the way the regime says NOT to
    сбит BTC-SMA  - a long the daily BTC filter refuses

Same geometry, same costs, same 48-hour limit as everything else, so the numbers are comparable to
the existing table. Costs are the measured ones.
"""
import bisect
import collections
import csv
import datetime
import pickle
import sys

import numpy as np

sys.path.insert(0, 'C:/Users/Lenovo/Desktop/Торговля/crypto-bot')
import struct_params as SP
import feat2
from src import pullback_bank as PB

meas = collections.defaultdict(list)
for r in csv.DictReader(open('book_samples.csv')):
    meas[r['coin'] + 'USDT'].append(float(r['cost109']))
SP.SLIP_REAL.update({k: float(np.median(v)) for k, v in meas.items()})

BIG = 10 ** 9
COST = 0.0004
COINS = list(SP.COINS)
SL_MULT, TP_MULT = 3.0, 1.0


def outcome(c, slip, k, atr_v, lg):
    t15, A15 = c['t15'], c['a15']
    lim = None
    o, h, l, cc = (A15[k:k + PB.HOLD_BARS, x] for x in range(4))
    e = o[0] * (1 + slip) if lg else o[0] * (1 - slip)
    if lg:
        TP, SL = e + TP_MULT * atr_v, e - SL_MULT * atr_v
        hs, ht = l <= SL, h >= TP
    else:
        TP, SL = e - TP_MULT * atr_v, e + SL_MULT * atr_v
        hs, ht = h >= SL, l <= TP
    js = int(np.argmax(hs)) if hs.any() else BIG
    jt = int(np.argmax(ht)) if ht.any() else BIG
    if js <= jt and js < BIG:
        f_ = (min(SL, o[js]) * (1 - slip)) if lg else (max(SL, o[js]) * (1 + slip))
    elif jt < BIG:
        f_ = TP * (1 - slip) if lg else TP * (1 + slip)
    else:
        f_ = cc[-1] * (1 - slip) if lg else cc[-1] * (1 + slip)
    ret = ((f_ / e - 1) if lg else (1 - f_ / e)) - COST
    return ret / (SL_MULT * atr_v / e)


rows = []
tally = collections.Counter()
br = feat2.breadth(COINS)
for s in COINS:
    c = SP.CTX[s]
    T1, f = feat2.build(s)
    f['breadth'] = np.array([br.get(int(t), (np.nan, np.nan))[0] for t in T1])
    f['breadth_med'] = np.array([br.get(int(t), (np.nan, np.nan))[1] for t in T1])
    slip = SP.SLIP_REAL.get(s, 0.0002)
    t15, A15, pos = c['t15'], c['a15'], c['pos']
    starts, iv = c['starts'], c['iv']
    bsm = c['sma'].setdefault(50, PB._sma(c['bdc'], 50))
    atr = c['F']['atr']
    names = sorted(f)
    for i in range(PB.MIN_HOURS, len(T1)):
        close = int(T1[i]) + PB.HOUR
        a = atr[i]
        k = pos.get(close)
        if k is None or not np.isfinite(a) or k + PB.HOLD_BARS > len(t15):
            continue
        j = bisect.bisect_right(starts, close) - 1
        inside = j >= 0 and iv[j][0] <= close < iv[j][1]
        side = iv[j][2] if inside else None
        fv = {n: float(f[n][i]) for n in names}
        if not inside:
            tally['вне режима'] += 2
            for lg in (True, False):
                rows.append((close, s, lg, 'вне режима', outcome(c, slip, k, a, lg), fv))
            continue
        want = side == 'LONG'
        d = bisect.bisect_right(c['bdt'], close - 86400) - 1
        blocked = want and d >= 49 and c['bdc'][d] < bsm[d]
        # the side the regime refuses - always outside the searched space
        tally['против режима'] += 1
        rows.append((close, s, not want, 'против режима', outcome(c, slip, k, a, not want), fv))
        if blocked:
            tally['сбит BTC-SMA'] += 1
            rows.append((close, s, want, 'сбит BTC-SMA', outcome(c, slip, k, a, want), fv))
    print('  %-10s готово, строк пока %d' % (s, len(rows)), flush=True)

print('  всего строк %d: %s' % (len(rows), dict(tally)), flush=True)
pickle.dump(rows, open('outside_cache.pkl', 'wb'), protocol=4)
print('  сохранено', flush=True)

print('', flush=True)
print('  === что вообще даёт каждая отброшенная группа, без всяких условий ===', flush=True)
by = collections.defaultdict(list)
for close, s, lg, why, R, fv in rows:
    by[(why, lg)].append(R)
for (why, lg), v in sorted(by.items()):
    v = np.array(v)
    print('    %-16s %-5s n%7d ВР%5.1f%% ср%+.4fR' % (why, 'LONG' if lg else 'SHORT', len(v),
                                                      100 * np.mean(v > 0), v.mean()), flush=True)
