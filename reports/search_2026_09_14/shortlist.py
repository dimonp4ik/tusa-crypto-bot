"""From 247 survivors to the few that are actually different from each other.

The survivor list is not 247 findings. Nearly every short family contains vol168 and nearly every
long one contains btc168, which is what one mechanism looks like when a search is allowed to name it
in a hundred ways. Two things separate the list: how many blind years a family survived (five is
worth much more than three, because three means the family had too few trades in two of the years),
and then whether its trades are the same trades.

Thresholds are the MEDIAN of the five blind re-picks. That is a number no single year chose, which
is the whole point; the search's own maximising pick is what produced today's retractions.
"""
import collections
import datetime
import pickle
import sys

import numpy as np

sys.path.insert(0, 'C:/Users/Lenovo/Desktop/Торговля/crypto-bot')

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

surv = pickle.load(open('loyo_survivors.pkl', 'rb'))
five = [d for d in surv if d['folds'] == 5]
print('  выживших всего %d, из них прошедших ВСЕ ПЯТЬ слепых лет: %d' % (len(surv), len(five)),
      flush=True)
five.sort(key=lambda d: -d['worst'])

SIDE = {True: lg, False: ~lg}


def conds_for(side, n):
    sm = SIDE[side]
    col = X[n]
    ok = sm & np.isfinite(col)
    qs = np.quantile(col[ok], np.linspace(0.12, 0.88, NQ))
    out = {}
    for q in np.unique(np.round(qs, 6)):
        out[('>=', q)] = ok & (col >= q)
        out[('<=', q)] = ok & (col <= q)
    return out


def repick(side, fa, fb):
    """The five blind re-picks, and the median of them."""
    ca, cb = conds_for(side, fa), conds_for(side, fb)
    picks, holds = [], []
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
        mh = ca[bk[0]] & cb[bk[1]] & ho
        if mh.sum() < MIN_HOLD:
            continue
        picks.append(bk)
        holds.append(float(R[mh].mean()))
    if not picks:
        return None
    oa = collections.Counter(p[0][0] for p in picks).most_common(1)[0][0]
    ob = collections.Counter(p[1][0] for p in picks).most_common(1)[0][0]
    ta = float(np.median([p[0][1] for p in picks if p[0][0] == oa]))
    tb = float(np.median([p[1][1] for p in picks if p[1][0] == ob]))
    return (fa, oa, ta), (fb, ob, tb), holds


print('', flush=True)
print('  === медианные пороги слепых переподборов, и прореживание по пересечению ===', flush=True)
chosen = []
seen_keys = []
for d in five[:22]:
    side = d['side'] == 'LONG'
    r = repick(side, d['a'], d['b'])
    if r is None:
        continue
    A, B, holds = r
    m = SIDE[side].copy()
    for nm, op, t in (A, B):
        c = X[nm]
        m &= np.isfinite(c) & ((c >= t) if op == '>=' else (c <= t))
    keys = {(int(ts[i]), coin[i]) for i in np.flatnonzero(m)}
    dup = None
    for nm2, k2 in seen_keys:
        j = len(keys & k2) / max(min(len(keys), len(k2)), 1)
        if j > 0.6:
            dup = (nm2, j)
            break
    lbl = '%s %s%s%.4f + %s%s%.4f' % (d['side'], A[0], A[1], A[2], B[0], B[1], B[2])
    v = R[m]
    print('    %-52s n%5d ср%+.4f | слепые годы %s%s'
          % (lbl, len(v), v.mean(), [round(x, 3) for x in holds],
             '' if dup is None else '  <- те же сделки, что %s (%.0f%%)' % (dup[0], 100 * dup[1])),
          flush=True)
    if dup is None:
        seen_keys.append((lbl, keys))
        chosen.append((lbl, side, [A, B]))

print('', flush=True)
print('  различных семейств после прореживания: %d' % len(chosen), flush=True)
pickle.dump(chosen, open('shortlist.pkl', 'wb'))
for lbl, side, conds in chosen:
    print('    %s' % lbl, flush=True)
