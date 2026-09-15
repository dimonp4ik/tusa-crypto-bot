"""Writes shortgeo.py: re-measure the stop and take of both short rules on the final system.

The shorts carry most of the money (removing them costs 80% of the account) and most of the drawdown,
and adding short trades was just rejected - breadth works only on the long side. What is left is their
geometry. short_pop_in_downtrend was moved to stop 2 / take 1.5 back when the system had no boost and
no breadth source; short_btc_up_morning still runs at stop 3 / take 1. Both are re-measured here on the
final system (boost n >= 10 + breadth over 2 hours), one rule at a time, the other rule untouched.

Grid: stop 1.5 / 2.0 / 2.5 / 3.0 ATR against take 0.75 / 1.0 / 1.5 / 2.0 ATR, hold unchanged.
Three books, full fill, money at 1.4% and at 12% drawdown, per-year against the final system; ten paired
85% fills for the two best cells of each rule; rolling choice by money at 12% drawdown.
"""
import ast
import io

s = io.open('rollbreadth.py', encoding='utf-8').read()
head = s[:s.index("CELLS = [")]

TAIL = r'''
FULL = BASE_KEYS | {'r2_4'}
SHORTS = [('r%d' % k, r.get('name', 'r%d' % k)) for k, r in enumerate(RULES) if r.get('side', 'LONG') == 'SHORT']
SLS = (1.5, 2.0, 2.5, 3.0)
TPS = (0.75, 1.0, 1.5, 2.0)
BASE_GEO = dict(GEO)


def with_geo(key, sl, tp, hold=48):
    GEO.clear(); GEO.update(BASE_GEO)
    GEO[key] = (sl, tp, hold)


for book_name, coins in BOOKS3:
    H = {s: coin_hours(s) for s in coins}
    base_sig = make_sig_raw(coins)
    extra = roll_entries(coins, H, 'r2_4', 4, 2, 4, 0)
    sig_all = sorted(base_sig + extra, key=lambda x: (x[0], x[1]))
    GEO.clear(); GEO.update(BASE_GEO)
    full = boost(book(sig_all, FULL)[0])
    feq, fdd = fix2(full); fm = sim_w.money_at_dd(full, 0.12)[1]; fpy = pyr(full)
    print('', flush=True)
    print('  ===== %s: ИТОГОВАЯ $%.0f %.1f%% DD12 $%.0f =====' % (book_name, feq, 100 * fdd, fm), flush=True)
    RES = {'итоговая': full}
    for key, name in SHORTS:
        cur = BASE_GEO.get(key, (3.0, 1.0, 48))
        print('    ПРАВИЛО %s (сейчас стоп %.1f / тейк %.2f): ячейка = $ при 1.4%% / просадка / $ при DD12; * = выше итоговой по обеим мерам'
              % (name, cur[0], cur[1]), flush=True)
        print('      стоп\\тейк ' + ''.join('%-26s' % ('%.2f' % t) for t in TPS), flush=True)
        for sl in SLS:
            cells = []
            for tp in TPS:
                with_geo(key, sl, tp)
                rows = boost(book(sig_all, FULL)[0])
                eq, dd = fix2(rows)
                m = sim_w.money_at_dd(rows, 0.12)[1]
                RES['%s стоп%.1f тейк%.2f' % (name, sl, tp)] = rows
                cells.append('%s$%5.0f %4.1f%% $%6.0f  ' % ('*' if (eq > feq and m > fm) else ' ', eq, 100 * dd, m))
            print('      %-9.1f %s' % (sl, ''.join(cells)), flush=True)
        GEO.clear(); GEO.update(BASE_GEO)
    cand = [k for k in RES if k != 'итоговая']
    top = sorted(cand, key=lambda k: -sim_w.money_at_dd(RES[k], 0.12)[1])[:3]
    print('    ЗАЛИВКА 85%% против ИТОГОВОЙ: %s' % ', '.join(top), flush=True)
    for lbl in top:
        name, sls, tps = lbl.split()
        sl, tp = float(sls[4:]), float(tps[4:])
        key = next(k for k, n in SHORTS if n == name)
        A, B = [], []
        for seed in range(1, 11):
            rng = np.random.default_rng(seed)
            sub = [x for x in sig_all if rng.random() < 0.85]
            GEO.clear(); GEO.update(BASE_GEO)
            r0 = boost(book(sub, FULL)[0])
            with_geo(key, sl, tp)
            r1 = boost(book(sub, FULL)[0])
            e0, d0 = fix2(r0); e1, d1 = fix2(r1)
            A.append((e0, d0, sim_w.money_at_dd(r0, 0.12)[1], pyr(r0))); B.append((e1, d1, sim_w.money_at_dd(r1, 0.12)[1], pyr(r1)))
        GEO.clear(); GEO.update(BASE_GEO)
        med = lambda X, i: float(np.median([x[i] for x in X]))
        yw = ' '.join('%d/10' % sum(1 for x, y in zip(A, B) if y[3][j] > x[3][j]) for j in range(5))
        print('      %-28s итоговая $%.0f %.1f%% DD12 $%.0f | вариант $%.0f %.1f%% (макс %.1f%%) DD12 $%.0f | лучше $ %d/10, DD12 %d/10 | годы %s'
              % (lbl, med(A, 0), 100 * med(A, 1), med(A, 2), med(B, 0), 100 * med(B, 1), 100 * max(x[1] for x in B), med(B, 2),
                 sum(1 for x, y in zip(A, B) if y[0] > x[0]), sum(1 for x, y in zip(A, B) if y[2] > x[2]), yw), flush=True)
    print('    СКОЛЬЗЯЩИЙ ВЫБОР: ячейка по $ при DD12 на 4 годах, замер 5-го при 1.4% против ИТОГОВОЙ', flush=True)
    wins = 0
    for y in range(2022, 2027):
        lo, hi = YT[y], YT[y + 1]
        best, bk = -1, None
        for lbl, rows in RES.items():
            mm = sim_w.money_at_dd([x for x in rows if not (lo <= x[0] < hi)], 0.12)[1]
            if mm > best:
                best, bk = mm, lbl
        mp = sim_w.simulate([x for x in RES[bk] if lo <= x[0] < hi], 0.014)['eq']
        mb = sim_w.simulate([x for x in full if lo <= x[0] < hi], 0.014)['eq']
        wins += int(mp > mb)
        print('      %d выбрано «%s» | $%.0f против $%.0f %s' % (y, bk, mp, mb, 'лучше' if mp > mb else ('так же' if mp == mb else 'хуже')), flush=True)
    print('      лучше итоговой в %d из 5 лет' % wins, flush=True)
'''

src = head + TAIL
ast.parse(src)
io.open('shortgeo.py', 'w', encoding='utf-8').write(src)
print('shortgeo.py готов, синтаксис ок')
