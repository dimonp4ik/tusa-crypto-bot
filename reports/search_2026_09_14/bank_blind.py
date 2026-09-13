"""The blind year turned on the bank's own five rules.

Every number in the deployed rule set was chosen by looking at all five years, which is exactly the
procedure that produced today's retractions. The rules have been audited before - thirteen fitted
numbers, twelve of which stood - but never this way: re-pick the rule's own threshold and its own
hour window on four years, then look at the fifth once.

Two questions. Does each rule pay in the year it did not choose? And where does the blind re-pick
put the threshold - near the deployed number, or somewhere else entirely?

A rule that fails this is not necessarily wrong. It may simply have too few trades in one year to
be measured. The reading that matters is whether the deployed number lands inside the spread of the
blind re-picks, because a number outside that spread was chosen by the years it is judged on.
"""
import collections
import datetime
import pickle
import sys

import numpy as np

sys.path.insert(0, 'C:/Users/Lenovo/Desktop/Торговля/crypto-bot')

YEARS = [2022, 2023, 2024, 2025, 2026]
YT = {y: int(datetime.datetime(y, 1, 1, tzinfo=datetime.UTC).timestamp()) for y in YEARS + [2027]}
MIN_TRAIN, MIN_HOLD = 120, 20

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
hour = X['hour'] if 'hour' in X else np.array(
    [datetime.datetime.fromtimestamp(int(t), datetime.UTC).hour for t in ts], dtype=float)
print('  признаки в кэше: %s' % ', '.join(names), flush=True)

BANK = [
    ('btc_pump_evening', True, ('btc24', '>=', 0.04896), ('hour', (20, 23))),
    ('coin_run_evening', True, ('ret24', '>=', 3.864), ('hour', (20, 23))),
    ('btc_pump_night', True, ('btc24', '>=', 0.04896), ('hour', (0, 3))),
    ('short_btc_up_morning', False, ('btc24', '>=', 0.01701), ('hour', (8, 11))),
    ('short_pop_in_downtrend', False, ('rsi2', '>=', 71.02), ('rsi14', '<=', 35.22)),
]

WINDOWS = []
for w in (3, 4, 6):
    for s in range(24):
        WINDOWS.append((s, (s + w - 1) % 24))


def hmask(win):
    a, b = win
    return (hour >= a) & (hour <= b) if a <= b else ((hour >= a) | (hour <= b))


def tmask(nm, op, t, side):
    c = X[nm]
    return (lg == side) & np.isfinite(c) & ((c >= t) if op == '>=' else (c <= t))


for nm, side, A, B in BANK:
    print('', flush=True)
    if A[0] not in X or (B[0] != 'hour' and B[0] not in X):
        print('  ===== %s: признака %s нет в кэше — не проверить ====='
              % (nm, A[0] if A[0] not in X else B[0]), flush=True)
        continue
    print('  ===== %s (%s) =====' % (nm, 'LONG' if side else 'SHORT'), flush=True)
    col = X[A[0]][lg == side]
    col = col[np.isfinite(col)]
    grid = np.unique(np.round(np.quantile(col, np.linspace(0.05, 0.95, 25)), 6))
    grid = np.unique(np.round(np.append(grid, A[2]), 6))
    MA = {t: tmask(A[0], A[1], t, side) for t in grid}
    if B[0] == 'hour':
        MB = {w: hmask(w) for w in WINDOWS}
        dep_b = B[1]
    else:
        col2 = X[B[0]][lg == side]
        col2 = col2[np.isfinite(col2)]
        g2 = np.unique(np.round(np.quantile(col2, np.linspace(0.05, 0.95, 25)), 6))
        g2 = np.unique(np.round(np.append(g2, B[2]), 6))
        MB = {t: tmask(B[0], B[1], t, side) for t in g2}
        dep_b = B[2]
    dep = MA[round(A[2], 6)] & MB[dep_b]
    picks_a, picks_b, holds, deps = [], [], [], []
    for y in YEARS:
        tr, ho = year != y, year == y
        best, bk = None, None
        for ta, ma in MA.items():
            for tb, mb in MB.items():
                m = ma & mb & tr
                if m.sum() < MIN_TRAIN:
                    continue
                v = float(R[m].mean())
                if best is None or v > best:
                    best, bk = v, (ta, tb)
        if bk is None:
            print('    %d  на четырёх годах ничего не выбралось' % y, flush=True)
            continue
        mh = MA[bk[0]] & MB[bk[1]] & ho
        dh = dep & ho
        if mh.sum() < MIN_HOLD or dh.sum() < MIN_HOLD:
            print('    %d  выбрано %s%s%-9.4g + %-10s — в слепом году мало (n%d, у рабочего n%d)'
                  % (y, A[0], A[1], bk[0], str(bk[1]), mh.sum(), dh.sum()), flush=True)
            continue
        picks_a.append(bk[0])
        picks_b.append(bk[1])
        holds.append(float(R[mh].mean()))
        deps.append(float(R[dh].mean()))
        print('    %d  слепой переподбор %s%s%-9.4g + %-10s | СЛЕПОЙ ГОД: переподбор%+.4f (n%d) '
              '| РАБОЧЕЕ ПРАВИЛО%+.4f (n%d)'
              % (y, A[0], A[1], bk[0], str(bk[1]), R[mh].mean(), mh.sum(), R[dh].mean(), dh.sum()),
              flush=True)
    if picks_a:
        lo, hi = min(picks_a), max(picks_a)
        inside = lo <= A[2] <= hi
        print('    рабочий порог %s = %.5g; слепые переподборы от %.5g до %.5g -> %s'
              % (A[0], A[2], lo, hi, 'ВНУТРИ разброса' if inside else 'ВНЕ разброса'), flush=True)
        print('    окно/второе условие: рабочее %s, слепые %s'
              % (str(dep_b), collections.Counter(str(x) for x in picks_b).most_common()), flush=True)
        print('    рабочее правило в слепых годах: %s, плюсовых %d из %d'
              % ([round(x, 4) for x in deps], sum(1 for x in deps if x > 0), len(deps)), flush=True)
