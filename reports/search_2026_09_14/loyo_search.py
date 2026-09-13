"""The blind-year test as the search itself, not as a check afterwards.

Every retraction today has the same shape: a number chosen by looking at all five years, then
defended with tests that also looked at all five years. The leave-one-year-out re-pick was the only
thing that separated the two surviving families from the two that fail - and it was applied to four
candidates, after the fact, out of 26,866.

Here it IS the filter. For every pair of features, for each of the five years in turn: both
thresholds are re-chosen on the other four years, and the discarded year is then looked at once. A
family is only interesting if it pays in EVERY blind year it has enough trades in. Nothing is ranked
by a number that saw the year it is judged on.

This is the search that should have been run first.
"""
import collections
import datetime
import pickle
import sys
import time

import numpy as np

sys.path.insert(0, 'C:/Users/Lenovo/Desktop/Торговля/crypto-bot')

YEARS = [2022, 2023, 2024, 2025, 2026]
YT = {y: int(datetime.datetime(y, 1, 1, tzinfo=datetime.UTC).timestamp()) for y in YEARS + [2027]}
MIN_TRAIN, MIN_HOLD, MIN_FOLDS = 150, 25, 3
NQ = 14

rows = pickle.load(open('wide_cache.pkl', 'rb'))
names = sorted(rows[0][5])
ts = np.array([r[0] for r in rows], dtype=np.int64)
lg = np.array([r[3] for r in rows], dtype=bool)
R = np.array([r[4] for r in rows], dtype=np.float64)
X = {n: np.array([r[5][n] for r in rows], dtype=np.float64) for n in names}
del rows
year = np.zeros(len(ts), dtype=np.int32)
for y in YEARS:
    year[(ts >= YT[y]) & (ts < YT[y + 1])] = y
print('  строк %d, признаков %d' % (len(ts), len(names)), flush=True)

pairs = pickle.load(open('wide_pairs.pkl', 'rb'))
fam = collections.OrderedDict()
for d in pairs:
    a, b = d['a'].split()[0], d['b'].split()[0]
    key = (d['side'] == 'LONG', tuple(sorted((a, b))))
    fam.setdefault(key, 0)
    fam[key] += 1
print('  различных семейств (пара признаков + сторона): %d' % len(fam), flush=True)

# per-side masks for every candidate condition, built once
SIDE = {True: lg, False: ~lg}
COND = {}
for side in (True, False):
    sm = SIDE[side]
    for n in names:
        if n == 'hour':
            continue
        col = X[n]
        ok = sm & np.isfinite(col)
        if ok.sum() < 5000:
            continue
        qs = np.quantile(col[ok], np.linspace(0.12, 0.88, NQ))
        for q in np.unique(np.round(qs, 6)):
            COND[(side, n, '>=', q)] = ok & (col >= q)
            COND[(side, n, '<=', q)] = ok & (col <= q)
print('  масок условий: %d' % len(COND), flush=True)

FOLD_TR = {y: (year != y) for y in YEARS}
FOLD_HO = {y: (year == y) for y in YEARS}
out = []
t0 = time.time()
for k, (side, (fa, fb)) in enumerate(fam):
    ca = [c for c in COND if c[0] == side and c[1] == fa]
    cb = [c for c in COND if c[0] == side and c[1] == fb]
    if not ca or not cb:
        continue
    folds, hold_vals, picks = 0, [], []
    ok_all = True
    for y in YEARS:
        tr, ho = FOLD_TR[y], FOLD_HO[y]
        best, bm, bk = None, None, None
        for i in ca:
            mi = COND[i]
            for j in cb:
                m = mi & COND[j]
                mt = m & tr
                n = int(mt.sum())
                if n < MIN_TRAIN:
                    continue
                v = float(R[mt].mean())
                if best is None or v > best:
                    best, bm, bk = v, m, (i, j)
        if bm is None:
            continue
        mh = bm & ho
        n = int(mh.sum())
        if n < MIN_HOLD:
            continue
        folds += 1
        hv = float(R[mh].mean())
        hold_vals.append(hv)
        picks.append((bk[0][2], bk[0][3], bk[1][2], bk[1][3]))
        if hv <= 0:
            ok_all = False
    if folds < MIN_FOLDS or not ok_all:
        continue
    base = R[SIDE[side]].mean()
    # how far the re-picked thresholds wander, as a share of their own spread
    sa = np.std([p[1] for p in picks]) / (abs(np.mean([p[1] for p in picks])) + 1e-9)
    sb = np.std([p[3] for p in picks]) / (abs(np.mean([p[3] for p in picks])) + 1e-9)
    out.append(dict(side='LONG' if side else 'SHORT', a=fa, b=fb, folds=folds,
                    hold=float(np.mean(hold_vals)), worst=float(min(hold_vals)),
                    edge=float(np.mean(hold_vals)) - base, wander=float(max(sa, sb)),
                    vals=[round(v, 4) for v in hold_vals]))
    if len(out) % 5 == 0:
        print('    ... проверено %d семейств из %d, выжило %d, %.0f сек'
              % (k + 1, len(fam), len(out), time.time() - t0), flush=True)

print('  выжило семейств: %d из %d (за %.0f сек)' % (len(out), len(fam), time.time() - t0),
      flush=True)
out.sort(key=lambda d: -d['edge'])
print('', flush=True)
print('  прошли ВСЕ слепые годы:', flush=True)
for d in out[:40]:
    print('    %-5s %-9s + %-9s лет%d  худший%+.4f  средний%+.4f  эдж%+.4f  разброс порога %4.2f  %s'
          % (d['side'], d['a'], d['b'], d['folds'], d['worst'], d['hold'], d['edge'],
             d['wander'], d['vals']), flush=True)
pickle.dump(out, open('loyo_survivors.pkl', 'wb'))
