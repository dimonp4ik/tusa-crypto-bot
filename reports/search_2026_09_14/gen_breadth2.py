"""Writes breadth2.py: is breadth the edge, and where does book 2's extra drawdown come from?

breadth.py: when the strict 1h pullback fires on 3+ coins in the same hour, coins with relaxed
thresholds win 88-92% at +0.18..+0.23R, and the same thresholds without breadth lose money. On book 1
the source adds +25% at the same drawdown; on book 2 it deepens the drawdown. The week-earlier control
was only run at K >= 5, where there are almost no trades.

1. Controls at the working thresholds: breadth read 1 day, 3 days and 7 days earlier (same hour of
   day, same distribution, unrelated moment), K = 3 and 4, cells 53/38, 50/45 and 52/33.79.
2. Three books: the two protocol books and the live-like book (no BILL, no AAVE, XLM kept).
3. The 85% fill, ten paired draws (breadth is market information and is read before the fill).
4. Per coin on each book for 50/45 K>=3: wide trades in the book, win rate, mean R, sum R - which
   coins bring the drawdown on book 2.
"""
import ast
import io

s = io.open('breadth.py', encoding='utf-8').read()
head = s[:s.index("RELAX = ")]

TAIL = r'''
DAY = 86400
BOOKS3 = (('КНИГА 1 (без XLM, AAVE)', [c for c in SP.COINS if c not in ('BILLUSDT', 'XLMUSDT', 'AAVEUSDT')]),
          ('КНИГА 2 (все 15 монет)', [c for c in SP.COINS if c != 'BILLUSDT']),
          ('КНИГА 3 (без AAVE, с XLM)', [c for c in SP.COINS if c not in ('BILLUSDT', 'AAVEUSDT')]))
CELLS = ((53.0, 38.0, '53/38'), (50.0, 45.0, '50/45'), (52.0, 33.7947, '52/33.8'))
SHIFTS = ((0, 'настоящая'), (DAY, 'КОНТРОЛЬ -1 день'), (3 * DAY, 'КОНТРОЛЬ -3 дня'), (7 * DAY, 'КОНТРОЛЬ -7 дней'))

for book_name, coins in BOOKS3:
    H = {s: coin_hours(s) for s in coins}
    base_sig = make_sig_raw(coins)
    base_rows, _ = book(base_sig, BASE_KEYS)
    beq, bdd = fix2(base_rows)
    bm = sim_w.money_at_dd(base_rows, 0.12)[1]
    bpy = pyr(base_rows)
    base_keys_at = {(x[0], x[3]) for x in base_sig}
    print('', flush=True)
    print('  ===== %s: система $%.0f просадка %.1f%% DD12 $%.0f =====' % (book_name, beq, 100 * bdd, bm), flush=True)
    for lo48, hi6, cl in CELLS:
        for K in (3, 4):
            for shift, sl in SHIFTS:
                extra = wide_entries(coins, H, 'wide', 3, lo48, hi6, K, shift)
                alone, _ = book(sorted(extra, key=lambda x: (x[0], x[1])), {'wide'})
                va = np.array([x[2] for x in alone]) if alone else np.zeros(0)
                rows, cnt = book(sorted(base_sig + extra, key=lambda x: (x[0], x[1])), BASE_KEYS | {'wide'})
                eq, dd = fix2(rows)
                m = sim_w.money_at_dd(rows, 0.12)[1]
                py = pyr(rows)
                print('    %-8s K>=%d %-18s сам %4d сд ВР %5.1f%% ср %s | в книге %4d | $%6.0f %4.1f%% | DD12 $%6.0f %s | лучше по годам %d/5 (%s)'
                      % (cl, K, sl, len(va), 100 * np.mean(va > 0) if len(va) else 0, ('%+.3f' % va.mean()) if len(va) else '  -   ',
                         cnt['wide'], eq, 100 * dd, m, '*' if (eq > beq and m > bm) else ' ',
                         sum(1 for a, b in zip(bpy, py) if b > a), ' '.join('%+.0f' % (b - a) for a, b in zip(bpy, py))), flush=True)

    print('    ЗАЛИВКА 85%: 10 розыгрышей, одинаковые подмножества сигналов', flush=True)
    for lo48, hi6, cl, K in ((53.0, 38.0, '53/38', 3), (53.0, 38.0, '53/38', 4), (50.0, 45.0, '50/45', 3), (50.0, 45.0, '50/45', 4)):
        extra = wide_entries(coins, H, 'wide', 3, lo48, hi6, K, 0)
        full = sorted(base_sig + extra, key=lambda x: (x[0], x[1]))
        A, B = [], []
        for seed in range(1, 11):
            rng = np.random.default_rng(seed)
            sub = [x for x in full if rng.random() < 0.85]
            r0, _ = book(sub, BASE_KEYS)
            r1, _ = book(sub, BASE_KEYS | {'wide'})
            e0, d0 = fix2(r0); e1, d1 = fix2(r1)
            A.append((e0, d0, sim_w.money_at_dd(r0, 0.12)[1], pyr(r0)))
            B.append((e1, d1, sim_w.money_at_dd(r1, 0.12)[1], pyr(r1)))
        med = lambda X, k: float(np.median([x[k] for x in X]))
        yw = ' '.join('%d/10' % sum(1 for a, x in zip(A, B) if x[3][k] > a[3][k]) for k in range(5))
        print('      %-6s K>=%d: база $%.0f %.1f%% DD12 $%.0f | правило $%.0f %.1f%% (макс %.1f%%) DD12 $%.0f | лучше $ %d/10, DD12 %d/10 | по годам %s'
              % (cl, K, med(A, 0), 100 * med(A, 1), med(A, 2), med(B, 0), 100 * med(B, 1), 100 * max(x[1] for x in B), med(B, 2),
                 sum(1 for a, x in zip(A, B) if x[0] > a[0]), sum(1 for a, x in zip(A, B) if x[2] > a[2]), yw), flush=True)

    print('    ПО МОНЕТАМ (50/45, K>=3): сделки ширины, попавшие в книгу', flush=True)
    extra = wide_entries(coins, H, 'wide', 3, 50.0, 45.0, 3, 0)
    wide_at = {(x[0], x[3]) for x in extra} - base_keys_at
    rows, _ = book(sorted(base_sig + extra, key=lambda x: (x[0], x[1])), BASE_KEYS | {'wide'})
    per = collections.defaultdict(list)
    for x in rows:
        if (x[0], x[4]) in wide_at:
            per[x[4]].append(x[2])
    for sym in sorted(per, key=lambda k: sum(per[k])):
        v = np.array(per[sym])
        print('      %-10s %3d сд ВР %5.1f%% ср %+.3f сумма %+.2fR' % (sym, len(v), 100 * np.mean(v > 0), v.mean(), v.sum()), flush=True)
'''

src = head + TAIL
ast.parse(src)
io.open('breadth2.py', 'w', encoding='utf-8').write(src)
print('breadth2.py готов, синтаксис ок')
