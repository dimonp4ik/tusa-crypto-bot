"""Writes srcweight2.py: the source weights under the 85% fill, every variant on every book.

srcweight.py measured four weighting variants at full fill but only fill-tested the two best per book.
The one that passed both measures on all three books was the smallest: breadth trades x1.25. The larger
ones (pullback x1.15, breadth+pullback x1.25) earn more money but fall below the final system at equal
drawdown on book 2 - the shape of a leverage gain. Only the fill decides.

Here all four variants are fill-tested on all three books: ten paired 85% draws, the same subset for
the final system and every variant, median money at 1.4%, median and worst drawdown, money at 12%
drawdown, seeds won on each measure, per-year wins; then the rolling choice among the variants and the
final system by money at 12% drawdown.
"""
import ast
import io

s = io.open('srcweight.py', encoding='utf-8').read()
head = s[:s.index("VARS = (")]

TAIL = r'''
VARS = (('вес ширины x1.25', {'r2_4'}, 1.25), ('вес ширины x1.50', {'r2_4'}, 1.5),
        ('вес откатов x1.15', PULL_KEYS, 1.15), ('вес ширины и откатов x1.25', PULL_KEYS | {'r2_4'}, 1.25))

for book_name, coins in BOOKS3:
    H = {s: coin_hours(s) for s in coins}
    base_sig = make_sig_raw(coins)
    extra = roll_entries(coins, H, 'r2_4', 4, 2, 4, 0)
    sig_all = sorted(base_sig + extra, key=lambda x: (x[0], x[1]))
    GEO.clear(); GEO.update(BASE_GEO)
    rows, keys = book_keyed(sig_all, FULL)
    full = weigh_src(rows, keys, set(), 1.0)
    feq, fdd = fix2(full); fm = sim_w.money_at_dd(full, 0.12)[1]
    print('', flush=True)
    print('  ===== %s: ИТОГОВАЯ $%.0f %.1f%% DD12 $%.0f =====' % (book_name, feq, 100 * fdd, fm), flush=True)
    RES = {'итоговая': full}
    for lbl, want, mult in VARS:
        RES[lbl] = weigh_src(rows, keys, want, mult)
    A = []
    B = {lbl: [] for lbl, _, _ in VARS}
    for seed in range(1, 11):
        rng = np.random.default_rng(seed)
        sub = [x for x in sig_all if rng.random() < 0.85]
        r0, k0 = book_keyed(sub, FULL)
        a0 = weigh_src(r0, k0, set(), 1.0)
        e0, d0 = fix2(a0)
        A.append((e0, d0, sim_w.money_at_dd(a0, 0.12)[1], pyr(a0)))
        for lbl, want, mult in VARS:
            b1 = weigh_src(r0, k0, want, mult)
            e1, d1 = fix2(b1)
            B[lbl].append((e1, d1, sim_w.money_at_dd(b1, 0.12)[1], pyr(b1)))
    med = lambda X, i: float(np.median([x[i] for x in X]))
    print('    ЗАЛИВКА 85%%, 10 парных розыгрышей: итоговая $%.0f %.1f%% (макс %.1f%%) DD12 $%.0f'
          % (med(A, 0), 100 * med(A, 1), 100 * max(x[1] for x in A), med(A, 2)), flush=True)
    for lbl, _, _ in VARS:
        X = B[lbl]
        yw = ' '.join('%d/10' % sum(1 for x, y in zip(A, X) if y[3][j] > x[3][j]) for j in range(5))
        ok = sum(1 for x, y in zip(A, X) if y[0] > x[0]), sum(1 for x, y in zip(A, X) if y[2] > x[2])
        print('      %-28s $%6.0f %4.1f%% (макс %4.1f%%) DD12 $%6.0f | лучше $ %2d/10, DD12 %2d/10 %s | годы %s'
              % (lbl, med(X, 0), 100 * med(X, 1), 100 * max(x[1] for x in X), med(X, 2), ok[0], ok[1],
                 '+' if (ok[0] >= 8 and ok[1] >= 8) else ' ', yw), flush=True)
    print('    СКОЛЬЗЯЩИЙ ВЫБОР: по $ при DD12 на 4 годах, замер 5-го при 1.4% против ИТОГОВОЙ', flush=True)
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
io.open('srcweight2.py', 'w', encoding='utf-8').write(src)
print('srcweight2.py готов, синтаксис ок')
