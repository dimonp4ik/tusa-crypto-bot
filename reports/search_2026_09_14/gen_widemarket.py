"""Writes widemarket.py: on a market-wide dip, how much of the market to buy?

wideplateau.py: the breadth source is a plateau - every rsi48 46..54 × rsi6 40..50 cell beats the boost,
and the rsi6 limit does not matter. The edge is the breadth condition itself. So the question widens:
when the strict 1h pullback fires on K coins, is it right to buy every coin that is not falling apart -
lower rsi48, or no strength filter at all - and does a higher K make that safe?

Cells: rsi48 >= none / 40 / 44 / 46 / 50, rsi6 any, K = 4 / 5 / 6, stacked on the boost (n >= 10),
compared with the accepted rule (50/45 K4) and with the boost alone. Controls for the widest cell
(no rsi48, K4) and for 44/any K4: breadth read 3 and 7 days earlier. Three books, full fill; ten paired
85% fills for the best few against the accepted rule.
"""
import ast
import io

s = io.open('combo.py', encoding='utf-8').read()
head = s[:s.index("VARS = (")]

TAIL = r'''
DAY = 86400
R48 = ((0.0, 'без'), (40.0, '40'), (44.0, '44'), (46.0, '46'), (50.0, '50'))
KS = (4, 5, 6)
FILLC = ('m0_4', 'm44_4', 'm46_4', 'm0_5', 'm44_5')


def mk(a, K, sh=0):
    return 'm%d_%d%s' % (int(a), K, ('s%d' % (sh // DAY)) if sh else '')


for book_name, coins in BOOKS3:
    H = {s: coin_hours(s) for s in coins}
    base_sig = make_sig_raw(coins)
    extra = wide_entries(coins, H, 'acc', 4, 50.0, 45.0, 4, 0)
    for a, _ in R48:
        for K in KS:
            extra += wide_entries(coins, H, mk(a, K), 4, a, 100.0, K, 0)
    for a in (0.0, 44.0):
        for sh in (3 * DAY, 7 * DAY):
            extra += wide_entries(coins, H, mk(a, 4, sh), 4, a, 100.0, 4, sh)
    sig_all = sorted(base_sig + extra, key=lambda x: (x[0], x[1]))
    bo = boost(book(sig_all, BASE_KEYS)[0])
    beq, bdd = fix2(bo); bm = sim_w.money_at_dd(bo, 0.12)[1]
    acc_rows, acnt = book(sig_all, BASE_KEYS | {'acc'})
    acc = boost(acc_rows)
    aeq, add = fix2(acc); am = sim_w.money_at_dd(acc, 0.12)[1]
    print('', flush=True)
    print('  ===== %s: только буст $%.0f %.1f%% DD12 $%.0f | ПРИНЯТОЕ 50/45 K4 $%.0f %.1f%% DD12 $%.0f (%d сд) ====='
          % (book_name, beq, 100 * bdd, bm, aeq, 100 * add, am, acnt['acc']), flush=True)
    print('    ячейка = $ при 1.4% / просадка / $ при DD12 / сделок источника / ВР и ср R источника; + = выше ПРИНЯТОГО по обеим мерам', flush=True)
    for a, al in R48:
        for K in KS:
            k = mk(a, K)
            rows, cnt = book(sig_all, BASE_KEYS | {k})
            alone, _ = book(sorted([x for x in extra if x[2] == k], key=lambda x: (x[0], x[1])), {k})
            va = np.array([x[2] for x in alone]) if alone else np.zeros(1)
            rows = boost(rows)
            eq, dd = fix2(rows)
            m = sim_w.money_at_dd(rows, 0.12)[1]
            print('    rsi48>=%-4s K>=%d %s $%6.0f %4.1f%% DD12 $%6.0f | %3d сд | сам ВР %5.1f%% ср %+.3f'
                  % (al, K, '+' if (eq > aeq and m > am) else ' ', eq, 100 * dd, m, cnt[k], 100 * np.mean(va > 0), va.mean()), flush=True)
    for a, al in ((0.0, 'без'), (44.0, '44')):
        for sh in (3 * DAY, 7 * DAY):
            k = mk(a, 4, sh)
            rows, cnt = book(sig_all, BASE_KEYS | {k})
            rows = boost(rows)
            eq, dd = fix2(rows)
            m = sim_w.money_at_dd(rows, 0.12)[1]
            print('    КОНТРОЛЬ rsi48>=%-4s K>=4 ширина -%d дн  $%6.0f %4.1f%% DD12 $%6.0f | %3d сд' % (al, sh // DAY, eq, 100 * dd, m, cnt[k]), flush=True)
    print('    ЗАЛИВКА 85% против ПРИНЯТОГО, 10 парных розыгрышей', flush=True)
    for k in FILLC:
        A, B = [], []
        for seed in range(1, 11):
            rng = np.random.default_rng(seed)
            sub = [x for x in sig_all if rng.random() < 0.85]
            r0 = boost(book(sub, BASE_KEYS | {'acc'})[0])
            r1 = boost(book(sub, BASE_KEYS | {k})[0])
            e0, d0 = fix2(r0); e1, d1 = fix2(r1)
            A.append((e0, d0, sim_w.money_at_dd(r0, 0.12)[1], pyr(r0)))
            B.append((e1, d1, sim_w.money_at_dd(r1, 0.12)[1], pyr(r1)))
        med = lambda X, i: float(np.median([x[i] for x in X]))
        yw = ' '.join('%d/10' % sum(1 for x, y in zip(A, B) if y[3][j] > x[3][j]) for j in range(5))
        print('      %-6s: принятое $%.0f %.1f%% DD12 $%.0f | вариант $%.0f %.1f%% (макс %.1f%%) DD12 $%.0f | лучше $ %d/10, DD12 %d/10 | годы %s'
              % (k, med(A, 0), 100 * med(A, 1), med(A, 2), med(B, 0), 100 * med(B, 1), 100 * max(x[1] for x in B), med(B, 2),
                 sum(1 for x, y in zip(A, B) if y[0] > x[0]), sum(1 for x, y in zip(A, B) if y[2] > x[2]), yw), flush=True)
'''

src = head + TAIL
ast.parse(src)
io.open('widemarket.py', 'w', encoding='utf-8').write(src)
print('widemarket.py готов, синтаксис ок')
