"""The blind-year search, run on the hours the gate throws away.

The pool baselines are worse than the traded one - -0.0284 to -0.0794 against -0.0118 - so the gate
is doing real work and anything found here has to clear a lower bar to look good and a harder one to
BE good: it must beat its own pool's null by enough to still be positive in absolute terms, because
a rule that is merely less bad than -0.079 still loses money.

Same protocol as the search that produced today's only survivor: thresholds re-chosen on four years,
the fifth looked at once, every blind year must pay. The three pools are searched separately, since
they are three different claims about what the gate is wrong about.
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
MIN_TRAIN, MIN_HOLD, NQ = 200, 30, 12
POOL = sys.argv[1] if len(sys.argv) > 1 else 'сбит BTC-SMA'

rows = pickle.load(open('outside_cache.pkl', 'rb'))
rows = [r for r in rows if r[3] == POOL]
print('  пул %r: строк %d' % (POOL, len(rows)), flush=True)
names = sorted(rows[0][5])
ts = np.array([r[0] for r in rows], dtype=np.int64)
lg = np.array([r[2] for r in rows], dtype=bool)
R = np.array([r[4] for r in rows], dtype=np.float64)
X = {n: np.array([r[5][n] for r in rows], dtype=np.float64) for n in names}
del rows
year = np.zeros(len(ts), dtype=np.int32)
for y in YEARS:
    year[(ts >= YT[y]) & (ts < YT[y + 1])] = y

SIDE = {True: lg, False: ~lg}
COND = collections.defaultdict(dict)
for side in (True, False):
    if SIDE[side].sum() < 5000:
        continue
    base = R[SIDE[side]].mean()
    print('  сторона %-5s n%7d ноль %+.4f' % ('LONG' if side else 'SHORT', SIDE[side].sum(), base),
          flush=True)
    for n in names:
        if n == 'hour':
            continue
        col = X[n]
        ok = SIDE[side] & np.isfinite(col)
        if ok.sum() < 5000:
            continue
        for q in np.unique(np.round(np.quantile(col[ok], np.linspace(0.12, 0.88, NQ)), 6)):
            COND[side][(n, '>=', q)] = ok & (col >= q)
            COND[side][(n, '<=', q)] = ok & (col <= q)
print('  масок: %s' % {('LONG' if k else 'SHORT'): len(v) for k, v in COND.items()}, flush=True)

out = []
t0 = time.time()
for side in COND:
    feats = sorted({k[0] for k in COND[side]})
    base = R[SIDE[side]].mean()
    for ia in range(len(feats)):
        for ib in range(ia + 1, len(feats)):
            ca = [k for k in COND[side] if k[0] == feats[ia]]
            cb = [k for k in COND[side] if k[0] == feats[ib]]
            holds, picks, ok_all, folds = [], [], True, 0
            for y in YEARS:
                tr, ho = year != y, year == y
                best, bm, bk = None, None, None
                for ka in ca:
                    ma = COND[side][ka]
                    for kb in cb:
                        m = ma & COND[side][kb]
                        mt = m & tr
                        if mt.sum() < MIN_TRAIN:
                            continue
                        v = float(R[mt].mean())
                        if best is None or v > best:
                            best, bm, bk = v, m, (ka, kb)
                if bm is None:
                    continue
                mh = bm & ho
                if mh.sum() < MIN_HOLD:
                    continue
                folds += 1
                hv = float(R[mh].mean())
                holds.append(hv)
                picks.append(bk)
                if hv <= 0:
                    ok_all = False
            if folds < 3 or not ok_all:
                continue
            out.append(dict(side='LONG' if side else 'SHORT', a=feats[ia], b=feats[ib],
                            folds=folds, hold=float(np.mean(holds)), worst=float(min(holds)),
                            edge=float(np.mean(holds)) - base,
                            vals=[round(v, 4) for v in holds],
                            pick=collections.Counter(str(p) for p in picks).most_common(1)[0][0]))
        if ia % 6 == 0:
            print('    ... %s признак %d/%d, выжило %d, %.0f сек'
                  % ('LONG' if side else 'SHORT', ia, len(feats), len(out), time.time() - t0),
                  flush=True)

print('  выжило: %d (за %.0f сек)' % (len(out), time.time() - t0), flush=True)
out.sort(key=lambda d: -d['worst'])
for d in out[:30]:
    print('    %-5s %-12s + %-12s лет%d худший%+.4f средний%+.4f эдж%+.4f %s | %s'
          % (d['side'], d['a'], d['b'], d['folds'], d['worst'], d['hold'], d['edge'], d['vals'],
             d['pick']), flush=True)
pickle.dump(out, open('outside_%s.pkl' % POOL.replace(' ', '_'), 'wb'))
