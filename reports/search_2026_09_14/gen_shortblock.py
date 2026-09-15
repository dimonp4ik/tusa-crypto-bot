"""Writes shortblock.py: do not short into a market-wide pullback.

The breadth work showed a pullback that fires on many coins at once is a market dip that recovers -
the system's best long trades. The system also runs short rules (short_pop_in_downtrend on 2/1.5).
A short opened in or right after such a dip sells into the recovery.

Gate: a SHORT signal at close t is skipped when the strict 1h pullback breadth reached K at any hourly
close in [t - W hours, t]. K = 2 / 3 / 4, W = 0 / 3 / 6 / 12. Stacked on the accepted system (boost n>=10
+ wide 50/45 K4). Control: the same number of short signals skipped at random (median of 5 draws).
Three books, money at 1.4% and at 12% drawdown, per-year versus the accepted system, the blocked
shorts' own win rate and mean R, ten paired 85% fills for the best two cells, rolling choice.
"""
import ast
import io

s = io.open('combo.py', encoding='utf-8').read()
head = s[:s.index("VARS = (")]

TAIL = r'''
ACC = BASE_KEYS | {'w50'}
SHORT_KEYS = {'r%d' % k for k, r in enumerate(RULES) if r.get('side', 'LONG') == 'SHORT'}
KS = (2, 3, 4)
WS = (0, 3, 6, 12)


def strict_breadth(coins, H):
    b = collections.Counter()
    for s in coins:
        for close, (a, r6, atr, ok) in H[s].items():
            if ok and a >= 56.3761 and r6 <= 33.7947:
                b[close] += 1
    return b


def blocked_idx(sig, breadth, K, W):
    hot = np.array(sorted(t for t, n in breadth.items() if n >= K), dtype=np.int64)
    out = set()
    for i, x in enumerate(sig):
        if x[2] not in SHORT_KEYS:
            continue
        t = x[0]
        j = np.searchsorted(hot, t, side='right')
        if j > 0 and hot[j - 1] >= t - W * 3600:
            out.add(i)
    return out


for book_name, coins in BOOKS3:
    H = {s: coin_hours(s) for s in coins}
    base_sig = make_sig_raw(coins)
    extra = wide_entries(coins, H, 'w50', 4, 50.0, 45.0, 4, 0)
    sig_all = sorted(base_sig + extra, key=lambda x: (x[0], x[1]))
    br = strict_breadth(coins, H)
    acc = boost(book(sig_all, ACC)[0])
    aeq, add = fix2(acc); am = sim_w.money_at_dd(acc, 0.12)[1]
    apy = pyr(acc)
    R_of = {(x[0], x[4]): x[2] for x in acc}
    shorts = [i for i, x in enumerate(sig_all) if x[2] in SHORT_KEYS]
    print('', flush=True)
    print('  ===== %s: ПРИНЯТОЕ $%.0f %.1f%% DD12 $%.0f, шорт-сигналов %d =====' % (book_name, aeq, 100 * add, am, len(shorts)), flush=True)
    RES = {'принятое': acc}
    for K in KS:
        for W in WS:
            bl = blocked_idx(sig_all, br, K, W)
            if not bl:
                print('    K>=%d за %2dч: ни одного шорта не попало' % (K, W), flush=True)
                continue
            sub = [x for i, x in enumerate(sig_all) if i not in bl]
            rows = boost(book(sub, ACC)[0])
            eq, dd = fix2(rows)
            m = sim_w.money_at_dd(rows, 0.12)[1]
            py = pyr(rows)
            bR = [R_of[(sig_all[i][0], sig_all[i][3])] for i in bl if (sig_all[i][0], sig_all[i][3]) in R_of]
            lbl = 'K>=%d за %dч' % (K, W)
            RES[lbl] = rows
            ce, cm = [], []
            for seed in range(5):
                rng = np.random.default_rng(seed)
                pick = set(rng.choice(shorts, size=len(bl), replace=False).tolist())
                cr = boost(book([x for i, x in enumerate(sig_all) if i not in pick], ACC)[0])
                ce.append(fix2(cr)[0]); cm.append(sim_w.money_at_dd(cr, 0.12)[1])
            print('    %-12s убрано %3d шорт-сигналов (в книге было %3d, их ВР %s ср %s) | $%6.0f %4.1f%% DD12 $%6.0f %s | лучше по годам %d/5 (%s) | КОНТРОЛЬ случайные $%.0f DD12 $%.0f'
                  % (lbl, len(bl), len(bR), ('%.1f%%' % (100 * np.mean(np.array(bR) > 0))) if bR else '  -  ', ('%+.3f' % np.mean(bR)) if bR else '  -  ',
                     eq, 100 * dd, m, '+' if (eq > aeq and m > am) else ' ', sum(1 for a, b in zip(apy, py) if b > a),
                     ' '.join('%+.0f' % (b - a) for a, b in zip(apy, py)), np.median(ce), np.median(cm)), flush=True)
    cand = [k for k in RES if k != 'принятое']
    top = sorted(cand, key=lambda k: -sim_w.money_at_dd(RES[k], 0.12)[1])[:2]
    print('    ЗАЛИВКА 85%% против ПРИНЯТОГО: %s' % ', '.join(top), flush=True)
    for lbl in top:
        K, W = int(lbl.split('>=')[1].split()[0]), int(lbl.split('за ')[1].rstrip('ч'))
        bl = blocked_idx(sig_all, br, K, W)
        A, B = [], []
        for seed in range(1, 11):
            rng = np.random.default_rng(seed)
            keep = [rng.random() < 0.85 for _ in sig_all]
            sub0 = [x for i, x in enumerate(sig_all) if keep[i]]
            sub1 = [x for i, x in enumerate(sig_all) if keep[i] and i not in bl]
            r0 = boost(book(sub0, ACC)[0]); r1 = boost(book(sub1, ACC)[0])
            e0, d0 = fix2(r0); e1, d1 = fix2(r1)
            A.append((e0, d0, sim_w.money_at_dd(r0, 0.12)[1], pyr(r0)))
            B.append((e1, d1, sim_w.money_at_dd(r1, 0.12)[1], pyr(r1)))
        med = lambda X, i: float(np.median([x[i] for x in X]))
        yw = ' '.join('%d/10' % sum(1 for x, y in zip(A, B) if y[3][j] > x[3][j]) for j in range(5))
        print('      %-12s: принятое $%.0f %.1f%% DD12 $%.0f | вариант $%.0f %.1f%% (макс %.1f%%) DD12 $%.0f | лучше $ %d/10, DD12 %d/10 | годы %s'
              % (lbl, med(A, 0), 100 * med(A, 1), med(A, 2), med(B, 0), 100 * med(B, 1), 100 * max(x[1] for x in B), med(B, 2),
                 sum(1 for x, y in zip(A, B) if y[0] > x[0]), sum(1 for x, y in zip(A, B) if y[2] > x[2]), yw), flush=True)
    print('    СКОЛЬЗЯЩИЙ ВЫБОР: по $ при DD12 на 4 годах, замер 5-го при 1.4% против ПРИНЯТОГО', flush=True)
    wins = 0
    for y in range(2022, 2027):
        lo, hi = YT[y], YT[y + 1]
        best, bk = -1, None
        for lbl, rows in RES.items():
            mm = sim_w.money_at_dd([x for x in rows if not (lo <= x[0] < hi)], 0.12)[1]
            if mm > best:
                best, bk = mm, lbl
        mp = sim_w.simulate([x for x in RES[bk] if lo <= x[0] < hi], 0.014)['eq']
        mb = sim_w.simulate([x for x in acc if lo <= x[0] < hi], 0.014)['eq']
        wins += int(mp > mb)
        print('      %d выбрано «%s» | $%.0f против $%.0f %s' % (y, bk, mp, mb, 'лучше' if mp > mb else ('так же' if mp == mb else 'хуже')), flush=True)
    print('      лучше принятого в %d из 5 лет' % wins, flush=True)
'''

src = head + TAIL
ast.parse(src)
io.open('shortblock.py', 'w', encoding='utf-8').write(src)
print('shortblock.py готов, синтаксис ок')
