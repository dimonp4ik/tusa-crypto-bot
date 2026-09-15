"""Writes burstfill.py: the high-threshold burst boost under the 85% fill, on three books.

burstplateau.py: x1.25 for trades opening in bursts of 7+ holds every check on book 1 but on book 2
under the 85% fill it buys money with drawdown (12.3%, max 14.2%, DD12 flat). One region never raised
the drawdown on either book at full fill: n >= 10. With 85% fill the bursts shrink, so n >= 8 and 9
are read too.

A third book is added - all coins except BILL and AAVE, XLM kept - because it is the closest to the
live list once AAVE is removed. Books 1 and 2 stay as the protocol.

Per book: full-fill numbers per cell, then ten paired 85% fills (the same subset for base and every
cell, n counted on the trades that opened), with median money, drawdown, worst drawdown, money at 12%
drawdown, seeds won on each measure, and per-year wins.
"""
import ast
import io

s = io.open('burstplateau.py', encoding='utf-8').read()
head = s[:s.index("for bi, (book_name, coins) in enumerate(BOOKS):")]

TAIL = r'''
BOOKS3 = BOOKS + (('КНИГА 3 (без AAVE, с XLM)', [c for c in SP.COINS if c not in ('BILLUSDT', 'AAVEUSDT')]),)
CELLS = ((8, 1.25), (9, 1.25), (10, 1.25), (10, 1.5), (7, 1.15))

for book_name, coins in BOOKS3:
    sig = make_sig_raw(coins)
    rows, _ = book(sig, BASE_KEYS)
    beq, bdd = fix2(rows)
    bm = sim_w.money_at_dd(rows, 0.12)[1]
    print('', flush=True)
    print('  ===== %s: система $%.0f просадка %.1f%% DD12 $%.0f =====' % (book_name, beq, 100 * bdd, bm), flush=True)
    for nmin, b in CELLS:
        wr = weighted(rows, nmin, b)
        eq, dd = fix2(wr)
        m = sim_w.money_at_dd(wr, 0.12)[1]
        print('    100%% n>=%-2d x%.2f  $%6.0f %4.1f%% | DD12 $%6.0f %s' % (nmin, b, eq, 100 * dd, m, '*' if (eq > beq and m > bm) else ''), flush=True)
    A = []
    B = {c: [] for c in CELLS}
    for seed in range(1, 11):
        rng = np.random.default_rng(seed)
        sub = [x for x in sig if rng.random() < 0.85]
        r0, _ = book(sub, BASE_KEYS)
        e0, d0 = fix2(r0)
        A.append((e0, d0, sim_w.money_at_dd(r0, 0.12)[1], pyr(r0)))
        for c in CELLS:
            r1 = weighted(r0, *c)
            e1, d1 = fix2(r1)
            B[c].append((e1, d1, sim_w.money_at_dd(r1, 0.12)[1], pyr(r1)))
    med = lambda X, k: float(np.median([x[k] for x in X]))
    print('    85%%: база $%.0f %.1f%% (макс %.1f%%) DD12 $%.0f' % (med(A, 0), 100 * med(A, 1), 100 * max(x[1] for x in A), med(A, 2)), flush=True)
    for c in CELLS:
        X = B[c]
        yw = ' '.join('%d/10' % sum(1 for a, x in zip(A, X) if x[3][k] > a[3][k]) for k in range(5))
        print('    85%% n>=%-2d x%.2f  $%6.0f %4.1f%% (макс %4.1f%%) DD12 $%6.0f | лучше $ %2d/10, DD12 %2d/10, просадка не хуже %2d/10 | по годам %s'
              % (c[0], c[1], med(X, 0), 100 * med(X, 1), 100 * max(x[1] for x in X), med(X, 2),
                 sum(1 for a, x in zip(A, X) if x[0] > a[0]), sum(1 for a, x in zip(A, X) if x[2] > a[2]),
                 sum(1 for a, x in zip(A, X) if x[1] <= a[1] + 1e-9), yw), flush=True)
'''

src = head + TAIL
ast.parse(src)
io.open('burstfill.py', 'w', encoding='utf-8').write(src)
print('burstfill.py готов, синтаксис ок')
