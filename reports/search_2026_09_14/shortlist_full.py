"""All 130 five-blind-year survivors through the gauntlet, not just the top 22.

The first pass took the 22 with the best worst-year and found five that cleared the gauntlet, of
which one made money and then died on its threshold grid. Ranking by worst blind year is itself a
choice, and a family can have a modest worst year and still be the one that survives execution. The
other 108 have never been looked at.

Same bar as before, and it is the execution bar rather than the statistical one: at least 35% of
eligible hours must become trades, at least 30 distinct months, no month over 25%, the mirror must
not also pay, and the bank must not already be taking the hours.
"""
import collections
import datetime
import pickle
import sys

import numpy as np

sys.path.insert(0, 'C:/Users/Lenovo/Desktop/Торговля/crypto-bot')
import struct_params as SP
import gauntlet2 as G
import stop_width as SW

YEARS = [2022, 2023, 2024, 2025, 2026]
YT = {y: int(datetime.datetime(y, 1, 1, tzinfo=datetime.UTC).timestamp()) for y in YEARS + [2027]}
MIN_TRAIN, MIN_HOLD, NQ = 150, 25, 14

rows = pickle.load(open('wide_cache.pkl', 'rb'))
names = sorted(rows[0][5])
ts = np.array([r[0] for r in rows], dtype=np.int64)
coin = np.array([r[2] for r in rows])
lg = np.array([r[3] for r in rows], dtype=bool)
R = np.array([r[4] for r in rows], dtype=np.float64)
X = {n: np.array([r[5][n] for r in rows], dtype=np.float64) for n in names}
del rows
year = np.zeros(len(ts), dtype=np.int32)
for y in YEARS:
    year[(ts >= YT[y]) & (ts < YT[y + 1])] = y
SIDE = {True: lg, False: ~lg}

surv = [d for d in pickle.load(open('loyo_survivors.pkl', 'rb')) if d['folds'] == 5]
surv.sort(key=lambda d: -d['worst'])
print('  прошедших все пять слепых лет: %d' % len(surv), flush=True)


def conds_for(side, n):
    col = X[n]
    ok = SIDE[side] & np.isfinite(col)
    out = {}
    for q in np.unique(np.round(np.quantile(col[ok], np.linspace(0.12, 0.88, NQ)), 6)):
        out[('>=', q)] = ok & (col >= q)
        out[('<=', q)] = ok & (col <= q)
    return out


def repick(side, fa, fb):
    ca, cb = conds_for(side, fa), conds_for(side, fb)
    picks = []
    for y in YEARS:
        tr, ho = year != y, year == y
        best, bk = None, None
        for ka, ma in ca.items():
            for kb, mb in cb.items():
                m = ma & mb & tr
                if m.sum() < MIN_TRAIN:
                    continue
                v = float(R[m].mean())
                if best is None or v > best:
                    best, bk = v, (ka, kb)
        if bk is None:
            continue
        if (ca[bk[0]] & cb[bk[1]] & ho).sum() < MIN_HOLD:
            continue
        picks.append(bk)
    if not picks:
        return None
    oa = collections.Counter(p[0][0] for p in picks).most_common(1)[0][0]
    ob = collections.Counter(p[1][0] for p in picks).most_common(1)[0][0]
    ta = float(np.median([p[0][1] for p in picks if p[0][0] == oa]))
    tb = float(np.median([p[1][1] for p in picks if p[1][0] == ob]))
    return (fa, oa, ta), (fb, ob, tb)


bank = SW.run(3.0, 1.0, coins=G.ALL, with_meta=True)
BK = {(a, s) for a, b, Rr, sf, s in bank}
PBK = {(r[0], r[4]) for r in G.simulate([('rsi48', '>=', 56.3761), ('rsi6', '<=', 33.7947)], True)[0]}
print('  банк %d, откат %d' % (len(BK), len(PBK)), flush=True)

seen_keys, passed = [], []
for k, d in enumerate(surv):
    side = d['side'] == 'LONG'
    r = repick(side, d['a'], d['b'])
    if r is None:
        continue
    A, B = r
    cc = [A, B]
    trades, hours = G.simulate(cc, side)
    if len(trades) < 100:
        continue
    keys = {(x[0], x[4]) for x in trades}
    dup = any(len(keys & k2) / max(min(len(keys), len(k2)), 1) > 0.6 for _, k2 in seen_keys)
    if dup:
        continue
    v = np.array([x[2] for x in trades])
    mons = collections.Counter(
        datetime.datetime.fromtimestamp(x[0], datetime.UTC).strftime('%y-%m') for x in trades)
    conv = len(trades) / max(hours, 1)
    worst = max(mons.values()) / len(trades)
    ob = len(keys & BK) / len(keys)
    op = len(keys & PBK) / len(keys)
    mir, _ = G.simulate(cc, not side)
    mv = np.array([x[2] for x in mir]) if mir else np.zeros(0)
    bad = []
    if conv < 0.35:
        bad.append('состояние')
    if len(mons) < 30:
        bad.append('мало месяцев')
    if worst > 0.25:
        bad.append('сбито в месяц')
    if len(mv) and mv.mean() > 0.05:
        bad.append('зеркало плюс')
    if ob > 0.3:
        bad.append('банк берёт')
    if op > 0.5:
        bad.append('это откат')
    lbl = '%s %s%s%.4f + %s%s%.4f' % (d['side'], A[0], A[1], A[2], B[0], B[1], B[2])
    seen_keys.append((lbl, keys))
    if not bad:
        passed.append((lbl, side, cc, trades))
        print('  ПРОШЛО %-46s n%4d ВР%5.1f%% ср%+.4f конв%3.0f%% мес%3d банк%3.0f%% откат%3.0f%%'
              % (lbl, len(v), 100 * np.mean(v > 0), v.mean(), 100 * conv, len(mons),
                 100 * ob, 100 * op), flush=True)
    if k % 20 == 0:
        print('    ... разобрано %d из %d, различных %d, прошло %d'
              % (k + 1, len(surv), len(seen_keys), len(passed)), flush=True)

print('', flush=True)
print('  различных семейств %d, прошло конвейер %d' % (len(seen_keys), len(passed)), flush=True)
pickle.dump([(l, s, c) for l, s, c, _ in passed], open('gauntlet_pass_full.pkl', 'wb'))
