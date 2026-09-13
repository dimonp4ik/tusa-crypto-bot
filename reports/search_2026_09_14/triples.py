"""Does a third condition help, or does it only look like it helps?

The honest way to ask is not "find the best third condition and admire it" - that is what produced
every retraction today. It is: choose the third condition on four years, then measure what it did to
the fifth. Five times, once per held-out year. If a third condition is real, the held-out delta
against the plain pair is positive in most of the five. If triples are just a finer sieve for noise,
the held-out delta will hover around zero or go negative while the training delta looks splendid.

The gap between the training delta and the held-out delta is the measurement that matters. It is the
price of the extra condition, in R per trade, and it is charged whether or not it is noticed.
"""
import datetime
import pickle
import sys

import numpy as np

sys.path.insert(0, 'C:/Users/Lenovo/Desktop/Торговля/crypto-bot')

YEARS = [2022, 2023, 2024, 2025, 2026]
YT = {y: int(datetime.datetime(y, 1, 1, tzinfo=datetime.UTC).timestamp()) for y in YEARS + [2027]}
MIN_TRAIN, MIN_HOLD = 150, 20

rows = pickle.load(open('wide_cache.pkl', 'rb'))
names = sorted(rows[0][5])
ts = np.array([r[0] for r in rows], dtype=np.int64)
lg = np.array([r[3] for r in rows], dtype=bool)
R = np.array([r[4] for r in rows], dtype=float)
F = {n: np.array([r[5][n] for r in rows], dtype=float) for n in names}
del rows
year = np.zeros(len(ts), dtype=int)
for y in YEARS:
    year[(ts >= YT[y]) & (ts < YT[y + 1])] = y

BASE = [
    ('ОТКАТ', True, [('rsi48', '>=', 56.3761), ('rsi6', '<=', 33.7947)]),
    ('КАПИТУЛЯЦИЯ', True, [('breadth', '<=', 0.0625), ('vol720', '>=', 1.4462)]),
]


def mk(nm, op, thr, side):
    c = F[nm]
    return (lg == side) & np.isfinite(c) & ((c >= thr) if op == '>=' else (c <= thr))


for fam, side, conds in BASE:
    base = mk(*conds[0], side)
    for c in conds[1:]:
        base &= mk(*c, side)
    used = {c[0] for c in conds}
    THIRD = []
    for nm in names:
        if nm in used or nm == 'hour':
            continue
        col = F[nm][lg == side]
        col = col[np.isfinite(col)]
        if len(col) < 20000:
            continue
        for q in np.linspace(0.15, 0.85, 15):
            t = float(np.quantile(col, q))
            THIRD.append((nm, '>=', t))
            THIRD.append((nm, '<=', t))
    print('', flush=True)
    print('  ===== %s: пара даёт %+.4fR на n%d; третьих условий на пробу %d ====='
          % (fam, R[base].mean(), base.sum(), len(THIRD)), flush=True)
    M3 = {}
    for k in THIRD:
        m = base & mk(*k, side)
        if m.sum() >= MIN_TRAIN:
            M3[k] = m
    print('  из них с достаточным n: %d' % len(M3), flush=True)

    dtr, dho = [], []
    for hold in YEARS:
        tr = year != hold
        ho = year == hold
        nb = int((base & ho).sum())
        if nb < MIN_HOLD:
            print('    %d  выброшенный год слишком мал (n%d)' % (hold, nb), flush=True)
            continue
        bt = R[base & tr].mean()
        bh = R[base & ho].mean()
        best, bk = None, None
        for k, m in M3.items():
            mt = m & tr
            if mt.sum() < MIN_TRAIN:
                continue
            v = float(R[mt].mean())
            if best is None or v > best:
                best, bk = v, k
        mh = M3[bk] & ho
        n = int(mh.sum())
        if n < MIN_HOLD:
            print('    %d  выбрано %s%s%.4f, обуч%+.4f (пара%+.4f) | в невиданном году всего n%d'
                  % (hold, bk[0], bk[1], bk[2], best, bt, n), flush=True)
            continue
        d_tr, d_ho = best - bt, R[mh].mean() - bh
        dtr.append(d_tr)
        dho.append(d_ho)
        print('    %d  выбрано %-10s%s%-9.4f | обучение: тройка%+.4f пара%+.4f (+%.4f) '
              '| НЕВИДАННЫЙ: тройка%+.4f пара%+.4f (%+.4f) n%d'
              % (hold, bk[0], bk[1], bk[2], best, bt, d_tr, R[mh].mean(), bh, d_ho, n), flush=True)
    if dtr:
        print('    ИТОГ: прибавка на обучении %+.4fR, в невиданном году %+.4fR — '
              'цена третьего условия %+.4fR; лучше пары в %d годах из %d'
              % (np.mean(dtr), np.mean(dho), np.mean(dho) - np.mean(dtr),
                 sum(1 for x in dho if x > 0), len(dho)), flush=True)
