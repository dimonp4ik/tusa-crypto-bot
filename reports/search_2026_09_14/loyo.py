"""The test none of these rules has had: thresholds chosen without seeing the year they are judged on.

Every number so far was produced by a search that read all five years. A rule can look stable across
years and still be an artefact, because the years were used to pick it. The honest version is to
throw one year away, re-run the choice on the other four with a grid three times finer than the
search used, and then look at the discarded year once.

Two things are read off it. Whether the held-out year pays - and whether the re-picked thresholds
land anywhere near the ones in use. A rule whose thresholds jump around when a year is removed is
fitted to the years, whatever its held-out return happens to be.
"""
import datetime
import pickle
import sys

import numpy as np

sys.path.insert(0, 'C:/Users/Lenovo/Desktop/Торговля/crypto-bot')

YEARS = [2022, 2023, 2024, 2025, 2026]
YT = {y: int(datetime.datetime(y, 1, 1, tzinfo=datetime.UTC).timestamp()) for y in YEARS + [2027]}
MIN_N, MIN_YEAR_N = 200, 15

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
print('  строк %d' % len(ts), flush=True)

FAM = [
    ('ОТКАТ', True, ('rsi48', '>=', 56.3761), ('rsi6', '<=', 33.7947)),
    ('ПАДЕНИЕ', True, ('ret24', '<=', -1.5434), ('vol720', '>=', 1.4462)),
    ('КАПИТУЛЯЦИЯ', True, ('breadth', '<=', 0.0625), ('vol720', '>=', 1.4462)),
    ('РАСХОЖДЕНИЕ', False, ('btc168', '<=', -0.0558), ('pos48', '>=', 0.8794)),
]


def grid(nm, op, cur, side):
    """A grid three times finer than the search's, centred on the quantile band it searched."""
    col = F[nm][lg == side]
    col = col[np.isfinite(col)]
    qs = np.linspace(0.10, 0.95, 36)
    g = sorted(set(np.round(np.quantile(col, qs), 6).tolist() + [round(cur, 6)]))
    return g


def mask_of(nm, op, thr, side):
    c = F[nm]
    return (lg == side) & np.isfinite(c) & ((c >= thr) if op == '>=' else (c <= thr))


for fam, side, A, B in FAM:
    print('', flush=True)
    print('  ===== %s (%s), в работе %s%s%.4f + %s%s%.4f ====='
          % (fam, 'LONG' if side else 'SHORT', A[0], A[1], A[2], B[0], B[1], B[2]), flush=True)
    gA, gB = grid(A[0], A[1], A[2], side), grid(B[0], B[1], B[2], side)
    mA = {t: mask_of(A[0], A[1], t, side) for t in gA}
    mB = {t: mask_of(B[0], B[1], t, side) for t in gB}
    base = R[lg == side].mean()
    for hold in YEARS:
        tr = year != hold
        best, bt = None, None
        for ta in gA:
            for tb in gB:
                m = mA[ta] & mB[tb] & tr
                n = int(m.sum())
                if n < MIN_N:
                    continue
                # must pay in every training year it appears in, not just on average
                yrs = [R[m & (year == y)].mean() for y in YEARS if y != hold
                       and int((m & (year == y)).sum()) >= MIN_YEAR_N]
                if len(yrs) < 3 or min(yrs) <= 0:
                    continue
                v = float(R[m].mean())
                if best is None or v > best:
                    best, bt = v, (ta, tb)
        if bt is None:
            print('    %d  на четырёх годах ничего не выбралось' % hold, flush=True)
            continue
        ta, tb = bt
        out = mA[ta] & mB[tb] & (year == hold)
        n = int(out.sum())
        s = ('n%4d ср%+.4fR ВР%5.1f%%' % (n, R[out].mean(), 100 * np.mean(R[out] > 0))
             if n >= MIN_YEAR_N else 'n%d - мало' % n)
        print('    %d  переподбор %s%s%-9.4f + %s%s%-9.4f обуч%+.4f | НЕВИДАННЫЙ ГОД %s'
              % (hold, A[0], A[1], ta, B[0], B[1], tb, best, s), flush=True)
