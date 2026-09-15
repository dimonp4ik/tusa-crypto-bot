"""Writes shortburst.py: a lower boost threshold for short bursts.

sidediag.py: short trades opening together with 7-9 other shorts already win 86-91% at +0.16..+0.22R on
every book, while long 7-9 bursts are weaker (+0.06..+0.14R). The accepted boost counts all trades and
needs n >= 10. Shorts and longs never share a bar (the BTC filter separates them), so a side-specific
threshold is well defined.

Variants against the accepted rule (x1.25 at n >= 10, all trades):
    shorts at n >= 7 / 8, longs stay at n >= 10
    shorts at n >= 7 with x1.25 and x1.5
    control: longs at n >= 7, shorts at n >= 10 (the long 7-9 bursts are the weak ones)
Three books, full fill both measures, per-year versus accepted, ten paired 85% fills, rolling choice.
"""
import ast
import io

s = io.open('sidediag.py', encoding='utf-8').read()
head = s[:s.index("MODES = (")]

TAIL = r'''
def weigh2(rows, short, nl, ns, ml=1.25, ms=1.25):
    cnt = collections.Counter(x[0] for x in rows)
    return [x[:5] + (((ms if cnt[x[0]] >= ns else 1.0) if sh else (ml if cnt[x[0]] >= nl else 1.0)),) for x, sh in zip(rows, short)]


VARS2 = (('ПРИНЯТОЕ: все при n>=10 x1.25', 10, 10, 1.25, 1.25),
         ('шорты n>=7, лонги n>=10', 10, 7, 1.25, 1.25),
         ('шорты n>=8, лонги n>=10', 10, 8, 1.25, 1.25),
         ('шорты n>=7 x1.5, лонги n>=10 x1.25', 10, 7, 1.25, 1.5),
         ('КОНТРОЛЬ лонги n>=7, шорты n>=10', 7, 10, 1.25, 1.25))

for book_name, coins in BOOKS3:
    H = {s: coin_hours(s) for s in coins}
    base_sig = make_sig_raw(coins)
    extra = wide_entries(coins, H, 'w50', 4, 50.0, 45.0, 4, 0)
    sig_all = sorted(base_sig + extra, key=lambda x: (x[0], x[1]))
    rows, short = book_side(sig_all, ACC)
    print('', flush=True)
    print('  ===== %s =====' % book_name, flush=True)
    RES = {}
    acc = weigh2(rows, short, 10, 10)
    aeq, add = fix2(acc); am = sim_w.money_at_dd(acc, 0.12)[1]; apy = pyr(acc)
    for lbl, nl, ns, ml, ms in VARS2:
        wr = weigh2(rows, short, nl, ns, ml, ms)
        RES[lbl] = wr
        eq, dd = fix2(wr)
        m = sim_w.money_at_dd(wr, 0.12)[1]
        py = pyr(wr)
        print('    100%% %-38s буст на %4d | $%6.0f %4.1f%% DD12 $%6.0f %s | по годам против принятого %d/5 (%s)'
              % (lbl, sum(1 for x in wr if x[5] > 1), eq, 100 * dd, m, '+' if (eq > aeq and m > am) else ' ',
                 sum(1 for a, b in zip(apy, py) if b > a), ' '.join('%+.0f' % (b - a) for a, b in zip(apy, py))), flush=True)
    print('    ЗАЛИВКА 85% против ПРИНЯТОГО, 10 парных розыгрышей', flush=True)
    for lbl, nl, ns, ml, ms in VARS2[1:]:
        A, B = [], []
        for seed in range(1, 11):
            rng = np.random.default_rng(seed)
            sub = [x for x in sig_all if rng.random() < 0.85]
            r, sh = book_side(sub, ACC)
            r0, r1 = weigh2(r, sh, 10, 10), weigh2(r, sh, nl, ns, ml, ms)
            e0, d0 = fix2(r0); e1, d1 = fix2(r1)
            A.append((e0, d0, sim_w.money_at_dd(r0, 0.12)[1], pyr(r0))); B.append((e1, d1, sim_w.money_at_dd(r1, 0.12)[1], pyr(r1)))
        med = lambda X, i: float(np.median([x[i] for x in X]))
        yw = ' '.join('%d/10' % sum(1 for x, y in zip(A, B) if y[3][j] > x[3][j]) for j in range(5))
        print('      %-38s принятое $%.0f %.1f%% DD12 $%.0f | вариант $%.0f %.1f%% (макс %.1f%%) DD12 $%.0f | лучше $ %d/10, DD12 %d/10 | годы %s'
              % (lbl, med(A, 0), 100 * med(A, 1), med(A, 2), med(B, 0), 100 * med(B, 1), 100 * max(x[1] for x in B), med(B, 2),
                 sum(1 for x, y in zip(A, B) if y[0] > x[0]), sum(1 for x, y in zip(A, B) if y[2] > x[2]), yw), flush=True)
    print('    СКОЛЬЗЯЩИЙ ВЫБОР (без контроля): по $ при DD12 на 4 годах, замер 5-го при 1.4% против ПРИНЯТОГО', flush=True)
    wins = 0
    for y in range(2022, 2027):
        lo, hi = YT[y], YT[y + 1]
        best, bk = -1, None
        for lbl, wr in RES.items():
            if 'КОНТРОЛЬ' in lbl:
                continue
            mm = sim_w.money_at_dd([x for x in wr if not (lo <= x[0] < hi)], 0.12)[1]
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
io.open('shortburst.py', 'w', encoding='utf-8').write(src)
print('shortburst.py готов, синтаксис ок')
