"""Writes wideplateau.py: is the breadth source's 50/45 a region or a peak?

Proposal item 9 takes coins with rsi48 >= 50 and rsi6 <= 45 when the strict 1h pullback fires on 4+
coins. 50/45 was one of four threshold pairs tried - a lone good cell would be a fit. The plateau:
rsi48 in 46..54, rsi6 in 40..50 and no rsi6 limit at all, K = 4, stacked on the boost (n >= 10) and
compared with the boost alone. Three books, money at 1.4% and at 12% drawdown at full fill; ten paired
85% fills for 50/45 and its neighbours.
"""
import ast
import io

s = io.open('combo.py', encoding='utf-8').read()
head = s[:s.index("VARS = (")]

TAIL = r'''
R48 = (46.0, 48.0, 50.0, 52.0, 54.0)
R6 = (40.0, 42.5, 45.0, 47.5, 50.0, 100.0)
NEIGH = ((50.0, 45.0), (48.0, 45.0), (52.0, 45.0), (50.0, 42.5), (50.0, 47.5), (50.0, 100.0))


def cell_key(a, b):
    return 'c%d_%d' % (int(a * 10), int(b * 10))


for book_name, coins in BOOKS3:
    H = {s: coin_hours(s) for s in coins}
    base_sig = make_sig_raw(coins)
    extra = []
    for a in R48:
        for b in R6:
            extra += wide_entries(coins, H, cell_key(a, b), 4, a, b, 4, 0)
    sig_all = sorted(base_sig + extra, key=lambda x: (x[0], x[1]))
    ref = boost(book(sig_all, BASE_KEYS)[0])
    req, rdd = fix2(ref)
    rm = sim_w.money_at_dd(ref, 0.12)[1]
    print('', flush=True)
    print('  ===== %s: только буст $%.0f %.1f%% DD12 $%.0f =====' % (book_name, req, 100 * rdd, rm), flush=True)
    print('    ячейка = $ при 1.4%% / просадка / $ при DD12 / сделок источника в книге; * = выше «только буст» по обеим мерам', flush=True)
    print('    rsi48\\rsi6 ' + ''.join('%-30s' % ('<=%s' % ('%.1f' % b if b < 100 else 'любой')) for b in R6), flush=True)
    for a in R48:
        cells = []
        for b in R6:
            k = cell_key(a, b)
            rows, cnt = book(sig_all, BASE_KEYS | {k})
            rows = boost(rows)
            eq, dd = fix2(rows)
            m = sim_w.money_at_dd(rows, 0.12)[1]
            cells.append('%s$%5.0f %4.1f%% $%6.0f %3d   ' % ('*' if (eq > req and m > rm) else ' ', eq, 100 * dd, m, cnt[k]))
        print('    >=%-8.0f %s' % (a, ''.join(cells)), flush=True)
    print('    ЗАЛИВКА 85%: соседи 50/45 против «только буст», 10 парных розыгрышей', flush=True)
    for a, b in NEIGH:
        k = cell_key(a, b)
        A, B = [], []
        for seed in range(1, 11):
            rng = np.random.default_rng(seed)
            sub = [x for x in sig_all if rng.random() < 0.85]
            r0 = boost(book(sub, BASE_KEYS)[0])
            r1 = boost(book(sub, BASE_KEYS | {k})[0])
            e0, d0 = fix2(r0); e1, d1 = fix2(r1)
            A.append((e0, d0, sim_w.money_at_dd(r0, 0.12)[1]))
            B.append((e1, d1, sim_w.money_at_dd(r1, 0.12)[1]))
        med = lambda X, i: float(np.median([x[i] for x in X]))
        print('      %s/%s: буст $%.0f %.1f%% DD12 $%.0f | + источник $%.0f %.1f%% (макс %.1f%%) DD12 $%.0f | лучше $ %d/10, DD12 %d/10'
              % (('%.0f' % a), ('%.1f' % b if b < 100 else 'любой'), med(A, 0), 100 * med(A, 1), med(A, 2), med(B, 0), 100 * med(B, 1),
                 100 * max(x[1] for x in B), med(B, 2), sum(1 for x, y in zip(A, B) if y[0] > x[0]), sum(1 for x, y in zip(A, B) if y[2] > x[2])), flush=True)
'''

src = head + TAIL
ast.parse(src)
io.open('wideplateau.py', 'w', encoding='utf-8').write(src)
print('wideplateau.py готов, синтаксис ок')
