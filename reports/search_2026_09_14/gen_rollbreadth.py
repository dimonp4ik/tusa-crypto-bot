"""Writes rollbreadth.py: breadth accumulated over the last few hours.

The accepted breadth source needs the strict 1h pullback on 4+ coins at the SAME hourly close. A market
dip can unfold over two or three hours - two coins pull back this hour, two the next - and the same-hour
count never reaches 4 although the dip is as wide.

Rolling breadth at close t = distinct coins whose strict 1h pullback fired at any close in (t - W h, t].
W = 1 is the accepted rule. W = 2 / 3 with K = 4 / 5 / 6 replace it (relaxed 50/45 entries at t while the
rolling breadth >= K), stacked on the boost (n >= 10). Controls for W2 K5 and W3 K6: breadth read 3 and 7
days earlier. Three books, full fill both measures; ten paired 85% fills for the best two cells against
the accepted rule; rolling choice by money at 12% drawdown.
"""
import ast
import io

s = io.open('combo.py', encoding='utf-8').read()
head = s[:s.index("VARS = (")]

TAIL = r'''
DAY = 86400


def roll_entries(coins, H, key, prio, W, K, shift, lo48=50.0, hi6=45.0):
    fired = collections.defaultdict(set)
    for s in coins:
        for close, (a, b, atr, ok) in H[s].items():
            if ok and a >= 56.3761 and b <= 33.7947:
                fired[close].add(s)
    closes = sorted(fired)
    breadth = {}
    for t in set(c for s in coins for c in H[s]):
        u = set()
        for w in range(W):
            u |= fired.get(t - shift - w * 3600, set())
        breadth[t] = len(u)
    out = []
    for s in coins:
        c = SP.CTX[s]
        t15, a15, pos = c['t15'], c['a15'], c['pos']
        slip = SLIP.get(s, 0.0003)
        for close, (a, b, atr, ok) in H[s].items():
            if not ok or atr <= 0 or (a >= 56.3761 and b <= 33.7947) or not (a >= lo48 and b <= hi6):
                continue
            if breadth.get(close, 0) < K:
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


CELLS = [(1, 4), (2, 4), (2, 5), (2, 6), (3, 4), (3, 5), (3, 6)]
CTRL = [(2, 5, 3 * DAY), (2, 5, 7 * DAY), (3, 6, 3 * DAY), (3, 6, 7 * DAY)]


def rk(W, K, sh=0):
    return 'r%d_%d%s' % (W, K, ('s%d' % (sh // DAY)) if sh else '')


for book_name, coins in BOOKS3:
    H = {s: coin_hours(s) for s in coins}
    base_sig = make_sig_raw(coins)
    extra = []
    for W, K in CELLS:
        extra += roll_entries(coins, H, rk(W, K), 4, W, K, 0)
    for W, K, sh in CTRL:
        extra += roll_entries(coins, H, rk(W, K, sh), 4, W, K, sh)
    sig_all = sorted(base_sig + extra, key=lambda x: (x[0], x[1]))
    acc = boost(book(sig_all, BASE_KEYS | {rk(1, 4)})[0])
    aeq, add = fix2(acc); am = sim_w.money_at_dd(acc, 0.12)[1]
    apy = pyr(acc)
    print('', flush=True)
    print('  ===== %s: ПРИНЯТОЕ (W1 K4) $%.0f %.1f%% DD12 $%.0f =====' % (book_name, aeq, 100 * add, am), flush=True)
    RES = {rk(1, 4): acc}
    for W, K, sh in [(w, k, 0) for w, k in CELLS] + CTRL:
        k = rk(W, K, sh)
        rows, cnt = book(sig_all, BASE_KEYS | {k})
        alone, _ = book(sorted([x for x in extra if x[2] == k], key=lambda x: (x[0], x[1])), {k})
        va = np.array([x[2] for x in alone]) if alone else np.zeros(1)
        rows = boost(rows)
        eq, dd = fix2(rows)
        m = sim_w.money_at_dd(rows, 0.12)[1]
        py = pyr(rows)
        if not sh:
            RES[k] = rows
        print('    %s окно %dч K>=%d %-12s | в книге %3d сд, сам ВР %5.1f%% ср %+.3f | $%6.0f %4.1f%% DD12 $%6.0f %s | по годам против принятого %d/5 (%s)'
              % ('КОНТРОЛЬ' if sh else '        ', W, K, ('ширина -%dд' % (sh // DAY)) if sh else '', cnt[k], 100 * np.mean(va > 0), va.mean(),
                 eq, 100 * dd, m, '+' if (eq > aeq and m > am) else ' ', sum(1 for a, b in zip(apy, py) if b > a),
                 ' '.join('%+.0f' % (b - a) for a, b in zip(apy, py))), flush=True)
    cand = [k for k in RES if k != rk(1, 4)]
    top = sorted(cand, key=lambda k: -sim_w.money_at_dd(RES[k], 0.12)[1])[:2]
    print('    ЗАЛИВКА 85%% против ПРИНЯТОГО: %s' % ', '.join(top), flush=True)
    for k in top:
        A, B = [], []
        for seed in range(1, 11):
            rng = np.random.default_rng(seed)
            sub = [x for x in sig_all if rng.random() < 0.85]
            r0 = boost(book(sub, BASE_KEYS | {rk(1, 4)})[0]); r1 = boost(book(sub, BASE_KEYS | {k})[0])
            e0, d0 = fix2(r0); e1, d1 = fix2(r1)
            A.append((e0, d0, sim_w.money_at_dd(r0, 0.12)[1], pyr(r0))); B.append((e1, d1, sim_w.money_at_dd(r1, 0.12)[1], pyr(r1)))
        med = lambda X, i: float(np.median([x[i] for x in X]))
        yw = ' '.join('%d/10' % sum(1 for x, y in zip(A, B) if y[3][j] > x[3][j]) for j in range(5))
        print('      %-6s: принятое $%.0f %.1f%% DD12 $%.0f | вариант $%.0f %.1f%% (макс %.1f%%) DD12 $%.0f | лучше $ %d/10, DD12 %d/10 | годы %s'
              % (k, med(A, 0), 100 * med(A, 1), med(A, 2), med(B, 0), 100 * med(B, 1), 100 * max(x[1] for x in B), med(B, 2),
                 sum(1 for x, y in zip(A, B) if y[0] > x[0]), sum(1 for x, y in zip(A, B) if y[2] > x[2]), yw), flush=True)
    print('    СКОЛЬЗЯЩИЙ ВЫБОР: по $ при DD12 на 4 годах, замер 5-го при 1.4% против ПРИНЯТОГО', flush=True)
    wins = 0
    for y in range(2022, 2027):
        lo, hi = YT[y], YT[y + 1]
        best, bk = -1, None
        for k, rows in RES.items():
            mm = sim_w.money_at_dd([x for x in rows if not (lo <= x[0] < hi)], 0.12)[1]
            if mm > best:
                best, bk = mm, k
        mp = sim_w.simulate([x for x in RES[bk] if lo <= x[0] < hi], 0.014)['eq']
        mb = sim_w.simulate([x for x in acc if lo <= x[0] < hi], 0.014)['eq']
        wins += int(mp > mb)
        print('      %d выбрано «%s» | $%.0f против $%.0f %s' % (y, bk, mp, mb, 'лучше' if mp > mb else ('так же' if mp == mb else 'хуже')), flush=True)
    print('      лучше принятого в %d из 5 лет' % wins, flush=True)
'''

src = head + TAIL
ast.parse(src)
io.open('rollbreadth.py', 'w', encoding='utf-8').write(src)
print('rollbreadth.py готов, синтаксис ок')
