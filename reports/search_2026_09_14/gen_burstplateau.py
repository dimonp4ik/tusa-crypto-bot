"""Writes burstplateau.py: is 'x1.25 when 7+ trades open together' a plateau or a peak?

burstsize.py found one row above the system on both books, both measures, 5 years of 5, and above the
shuffled control. Three things decide whether it is proposed:

1. Plateau: burst threshold n >= 4..10 against boost x1.15 / x1.25 / x1.35 / x1.5, money at 1.4% and
   at 12% drawdown on both books. A lone good cell is a fit; a region is a mechanism.
2. The 85% fill: ten random fills, the same subset for base and rule, and n counted on the trades that
   actually opened in that fill - live can only count what it opens.
3. Rolling choice: the (n, boost) cell - or no boost - picked on four years by money at 12% drawdown,
   then measured on the fifth at 1.4% against the system.
"""
import ast
import io

s = io.open('burst.py', encoding='utf-8').read()
head = s[:s.index("for bi, (book_name, coins) in enumerate(BOOKS):")]

TAIL = r'''
NS = (4, 5, 6, 7, 8, 10)
BOOSTS = (1.15, 1.25, 1.35, 1.5)


def weighted(rows, nmin, boost):
    cnt = collections.Counter(x[0] for x in rows)
    return [x[:5] + ((boost if cnt[x[0]] >= nmin else 1.0),) for x in rows]


for bi, (book_name, coins) in enumerate(BOOKS):
    sig = make_sig_raw(coins)
    rows, _ = book(sig, BASE_KEYS)
    beq, bdd = fix2(rows)
    bm = sim_w.money_at_dd(rows, 0.12)[1]
    print('', flush=True)
    print('  ===== %s: система $%.0f просадка %.1f%% DD12 $%.0f =====' % (book_name, beq, 100 * bdd, bm), flush=True)
    print('    ПЛАТО: ячейка = $ при 1.4%% / просадка / $ при DD12 (база $%.0f / %.1f%% / $%.0f)' % (beq, 100 * bdd, bm), flush=True)
    print('    n\\x   ' + ''.join('%-28s' % ('x%.2f' % b) for b in BOOSTS), flush=True)
    for nmin in NS:
        cells = []
        for b in BOOSTS:
            wr = weighted(rows, nmin, b)
            eq, dd = fix2(wr)
            m = sim_w.money_at_dd(wr, 0.12)[1]
            mark = '*' if (eq > beq and m > bm) else ' '
            cells.append('%s$%5.0f %4.1f%% $%6.0f      ' % (mark, eq, 100 * dd, m))
        print('    %-5s %s' % ('>=%d' % nmin, ''.join(cells)), flush=True)
    print('    (* = выше системы по обеим мерам)', flush=True)

    print('    ЗАЛИВКА 85%: 10 розыгрышей, n по открытым в розыгрыше сделкам', flush=True)
    for nmin, b in ((7, 1.25), (6, 1.25), (8, 1.25), (7, 1.15), (7, 1.35)):
        A, B = [], []
        for seed in range(1, 11):
            rng = np.random.default_rng(seed)
            sub = [x for x in sig if rng.random() < 0.85]
            r0, _ = book(sub, BASE_KEYS)
            r1 = weighted(r0, nmin, b)
            e0, d0 = fix2(r0); e1, d1 = fix2(r1)
            A.append((e0, d0, sim_w.money_at_dd(r0, 0.12)[1], pyr(r0)))
            B.append((e1, d1, sim_w.money_at_dd(r1, 0.12)[1], pyr(r1)))
        med = lambda X, k: float(np.median([x[k] for x in X]))
        yw = ' '.join('%d/10' % sum(1 for a, c in zip(A, B) if c[3][k] > a[3][k]) for k in range(5))
        print('      n>=%d x%.2f: база $%.0f %.1f%% DD12 $%.0f | правило $%.0f %.1f%% (макс %.1f%%) DD12 $%.0f | лучше $ %d/10, DD12 %d/10 | по годам %s'
              % (nmin, b, med(A, 0), 100 * med(A, 1), med(A, 2), med(B, 0), 100 * med(B, 1), 100 * max(x[1] for x in B), med(B, 2),
                 sum(1 for a, c in zip(A, B) if c[0] > a[0]), sum(1 for a, c in zip(A, B) if c[2] > a[2]), yw), flush=True)

    print('    СКОЛЬЗЯЩИЙ ВЫБОР: ячейка (или без буста) по $ при DD12 на 4 годах, замер 5-го при 1.4%', flush=True)
    wins = 0
    for y in range(2022, 2027):
        lo, hi = YT[y], YT[y + 1]
        tr = [x for x in rows if not (lo <= x[0] < hi)]
        te = [x for x in rows if lo <= x[0] < hi]
        best, bk = sim_w.money_at_dd(tr, 0.12)[1], None
        for nmin in NS:
            for b in BOOSTS:
                m = sim_w.money_at_dd(weighted(tr, nmin, b), 0.12)[1]
                if m > best:
                    best, bk = m, (nmin, b)
        mb = sim_w.simulate(te, 0.014)['eq']
        mp = sim_w.simulate(weighted(te, *bk), 0.014)['eq'] if bk else mb
        wins += int(mp > mb)
        print('      %d выбрано %s | $%.0f против $%.0f %s' % (y, ('n>=%d x%.2f' % bk) if bk else 'без буста', mp, mb,
                                                            'лучше' if mp > mb else ('так же' if mp == mb else 'хуже')), flush=True)
    print('      лучше системы в %d из 5 лет' % wins, flush=True)
'''

src = head + TAIL
ast.parse(src)
io.open('burstplateau.py', 'w', encoding='utf-8').write(src)
print('burstplateau.py готов, синтаксис ок')
