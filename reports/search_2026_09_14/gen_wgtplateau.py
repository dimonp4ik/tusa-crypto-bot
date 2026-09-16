"""Writes wgtplateau.py: is x1.25 for the breadth trades a plateau or a lucky point?

Proposal item 10 sizes the breadth source's trades x1.25 - the only weighting that held both measures on
all three books under the 85% fill. x1.50 already failed on book 2. A single good cell between a
do-nothing 1.0 and a failing 1.5 would be a fit, so the multiplier is swept: 1.10, 1.15, 1.25, 1.35,
1.40, with ten paired 85% draws per cell on all three books, both measures, per-year wins, and the
rolling choice among the cells and the final system.
"""
import ast
import io

s = io.open('srcweight.py', encoding='utf-8').read()
head = s[:s.index("VARS = (")]

TAIL = r'''
MULTS = (1.10, 1.15, 1.25, 1.35, 1.40)

for book_name, coins in BOOKS3:
    H = {s: coin_hours(s) for s in coins}
    base_sig = make_sig_raw(coins)
    extra = roll_entries(coins, H, 'r2_4', 4, 2, 4, 0)
    sig_all = sorted(base_sig + extra, key=lambda x: (x[0], x[1]))
    GEO.clear(); GEO.update(BASE_GEO)
    rows, keys = book_keyed(sig_all, FULL)
    full = weigh_src(rows, keys, set(), 1.0)
    feq, fdd = fix2(full); fm = sim_w.money_at_dd(full, 0.12)[1]
    n_w = sum(1 for k in keys if k == 'r2_4')
    print('', flush=True)
    print('  ===== %s: ИТОГОВАЯ $%.0f %.1f%% DD12 $%.0f, сделок ширины %d =====' % (book_name, feq, 100 * fdd, fm, n_w), flush=True)
    RES = {'итоговая': full}
    print('    ПОЛНАЯ ЗАЛИВКА:', flush=True)
    for m in MULTS:
        wr = weigh_src(rows, keys, {'r2_4'}, m)
        RES['x%.2f' % m] = wr
        eq, dd = fix2(wr)
        mm = sim_w.money_at_dd(wr, 0.12)[1]
        print('      x%.2f  $%6.0f %4.1f%% DD12 $%6.0f %s' % (m, eq, 100 * dd, mm, '+' if (eq > feq and mm > fm) else ' '), flush=True)
    A = []
    B = {m: [] for m in MULTS}
    for seed in range(1, 11):
        rng = np.random.default_rng(seed)
        sub = [x for x in sig_all if rng.random() < 0.85]
        r0, k0 = book_keyed(sub, FULL)
        a0 = weigh_src(r0, k0, set(), 1.0)
        e0, d0 = fix2(a0)
        A.append((e0, d0, sim_w.money_at_dd(a0, 0.12)[1], pyr(a0)))
        for m in MULTS:
            b1 = weigh_src(r0, k0, {'r2_4'}, m)
            e1, d1 = fix2(b1)
            B[m].append((e1, d1, sim_w.money_at_dd(b1, 0.12)[1], pyr(b1)))
    med = lambda X, i: float(np.median([x[i] for x in X]))
    print('    ЗАЛИВКА 85%%, 10 парных розыгрышей: итоговая $%.0f %.1f%% (макс %.1f%%) DD12 $%.0f'
          % (med(A, 0), 100 * med(A, 1), 100 * max(x[1] for x in A), med(A, 2)), flush=True)
    for m in MULTS:
        X = B[m]
        yw = ' '.join('%d/10' % sum(1 for x, y in zip(A, X) if y[3][j] > x[3][j]) for j in range(5))
        w1 = sum(1 for x, y in zip(A, X) if y[0] > x[0])
        w2 = sum(1 for x, y in zip(A, X) if y[2] > x[2])
        print('      x%.2f  $%6.0f %4.1f%% (макс %4.1f%%) DD12 $%6.0f | лучше $ %2d/10, DD12 %2d/10 %s | годы %s'
              % (m, med(X, 0), 100 * med(X, 1), 100 * max(x[1] for x in X), med(X, 2), w1, w2,
                 '+' if (w1 >= 8 and w2 >= 8) else ' ', yw), flush=True)
    print('    СКОЛЬЗЯЩИЙ ВЫБОР: множитель по $ при DD12 на 4 годах, замер 5-го при 1.4% против ИТОГОВОЙ', flush=True)
    wins = 0
    for y in range(2022, 2027):
        lo, hi = YT[y], YT[y + 1]
        best, bk = -1, None
        for lbl, rr in RES.items():
            mm = sim_w.money_at_dd([x for x in rr if not (lo <= x[0] < hi)], 0.12)[1]
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
io.open('wgtplateau.py', 'w', encoding='utf-8').write(src)
print('wgtplateau.py готов, синтаксис ок')
