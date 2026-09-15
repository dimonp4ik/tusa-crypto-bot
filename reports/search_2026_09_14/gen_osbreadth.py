"""Writes osbreadth.py: breadth measured by how many coins are oversold, not by the strict rule.

The accepted breadth source counts coins where the strict 1h pullback fires (rsi48 >= 56.38 and
rsi6 <= 33.79). That count misses fast market-wide dips where many coins get oversold on rsi6 but some
of them already have rsi48 below 56 - the dip is wide, the strict rule just does not see it.

Oversold breadth at an hourly close = coins (BTC filter passing) with rsi6 <= T6. When it is >= K, buy
coins with rsi48 >= 50 that did not fire strictly. Stacked on the accepted system (boost n >= 10 +
wide 50/45 K4), lower priority than the accepted source, one position per coin.
Cells: T6 = 30 with K = 4 / 6 / 8, T6 = 35 with K = 5 / 7 / 9. Controls for T6 35 K7 and T6 30 K6: the
count read 3 and 7 days earlier. Three books, full fill; ten paired 85% fills for the best two cells.
"""
import ast
import io

s = io.open('combo.py', encoding='utf-8').read()
head = s[:s.index("VARS = (")]

TAIL = r'''
DAY = 86400
ACC = BASE_KEYS | {'w50'}


def os_entries(coins, H, key, prio, lo48, T6, K, shift):
    breadth = collections.Counter()
    for s in coins:
        for close, (a, b, atr, ok) in H[s].items():
            if ok and b <= T6:
                breadth[close] += 1
    out = []
    for s in coins:
        c = SP.CTX[s]
        t15, a15, pos = c['t15'], c['a15'], c['pos']
        slip = SLIP.get(s, 0.0003)
        for close, (a, b, atr, ok) in H[s].items():
            if not ok or atr <= 0 or (a >= 56.3761 and b <= 33.7947) or a < lo48:
                continue
            if breadth.get(close - shift, 0) < K:
                continue
            j = pos.get(close)
            if j is None or j + MAXH > len(t15):
                continue
            o, h, l, cc = (a15[j:j + MAXH, x] for x in range(4))
            e = o[0] * (1 + slip)
            fav, adv = (h - e) / atr, (e - l) / atr
            out.append((close, prio, key, s, e, atr, fav, adv, np.maximum.accumulate(fav), np.maximum.accumulate(adv),
                        (o - e) / atr, (cc - e) / atr, t15[j:j + MAXH], slip, float('nan'), 0))
    return out


CELLS = [(30.0, 4), (30.0, 6), (30.0, 8), (35.0, 5), (35.0, 7), (35.0, 9)]
CTRL = [(35.0, 7, 3 * DAY), (35.0, 7, 7 * DAY), (30.0, 6, 3 * DAY), (30.0, 6, 7 * DAY)]


def ck(T6, K, sh=0):
    return 'o%d_%d%s' % (int(T6), K, ('s%d' % (sh // DAY)) if sh else '')


for book_name, coins in BOOKS3:
    H = {s: coin_hours(s) for s in coins}
    base_sig = make_sig_raw(coins)
    extra = wide_entries(coins, H, 'w50', 4, 50.0, 45.0, 4, 0)
    for T6, K in CELLS:
        extra += os_entries(coins, H, ck(T6, K), 6, 50.0, T6, K, 0)
    for T6, K, sh in CTRL:
        extra += os_entries(coins, H, ck(T6, K, sh), 6, 50.0, T6, K, sh)
    sig_all = sorted(base_sig + extra, key=lambda x: (x[0], x[1]))
    acc = boost(book(sig_all, ACC)[0])
    aeq, add = fix2(acc); am = sim_w.money_at_dd(acc, 0.12)[1]
    apy = pyr(acc)
    print('', flush=True)
    print('  ===== %s: ПРИНЯТОЕ (буст + ширина 50/45 K4) $%.0f %.1f%% DD12 $%.0f =====' % (book_name, aeq, 100 * add, am), flush=True)
    RES = {}
    for T6, K, sh in [(t, k, 0) for t, k in CELLS] + CTRL:
        k = ck(T6, K, sh)
        rows, cnt = book(sig_all, ACC | {k})
        alone, _ = book(sorted([x for x in extra if x[2] == k], key=lambda x: (x[0], x[1])), {k})
        va = np.array([x[2] for x in alone]) if alone else np.zeros(1)
        rows = boost(rows)
        eq, dd = fix2(rows)
        m = sim_w.money_at_dd(rows, 0.12)[1]
        py = pyr(rows)
        if not sh:
            RES[k] = (eq, m)
        print('    %s rsi6<=%d на K>=%d %s | в книге %3d сд, сам %4d сд ВР %5.1f%% ср %+.3f | $%6.0f %4.1f%% DD12 $%6.0f %s | лучше принятого по годам %d/5 (%s)'
              % ('КОНТРОЛЬ' if sh else '        ', int(T6), K, ('ширина -%dд' % (sh // DAY)) if sh else '          ', cnt[k], len(alone),
                 100 * np.mean(va > 0), va.mean(), eq, 100 * dd, m, '+' if (eq > aeq and m > am) else ' ',
                 sum(1 for a, b in zip(apy, py) if b > a), ' '.join('%+.0f' % (b - a) for a, b in zip(apy, py))), flush=True)
    top = sorted(RES, key=lambda k: -RES[k][1])[:2]
    print('    ЗАЛИВКА 85%% против ПРИНЯТОГО, 10 парных розыгрышей: %s' % ', '.join(top), flush=True)
    for k in top:
        A, B = [], []
        for seed in range(1, 11):
            rng = np.random.default_rng(seed)
            sub = [x for x in sig_all if rng.random() < 0.85]
            r0 = boost(book(sub, ACC)[0])
            r1 = boost(book(sub, ACC | {k})[0])
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
io.open('osbreadth.py', 'w', encoding='utf-8').write(src)
print('osbreadth.py готов, синтаксис ок')
