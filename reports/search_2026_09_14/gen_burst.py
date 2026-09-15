"""Writes burst.py: trades that open together, in the same closing bar.

The drawdown is made by trades opened before a sell-off, not by entries during it. The system checks
every coin at the same bar close, so when the market dips inside strength the pullback rule fires on
many coins in the same hour: one market bet placed n times. That n is known at the moment of entry.

Part 1 - diagnosis: trades grouped by how many trades of the book opened at the same timestamp
(1, 2-3, 4-6, 7+): share, win rate, mean R, per-year mean R, and the share of groups where every trade
was stopped.
Part 2 - sizing by the burst: weight 1/sqrt(n) and 1/n (and a floor of 0.5), against the same weights
shuffled across trades (control), at 1.4% and at 12% drawdown, both books, per-year, blind choice on
book 1. A size cut is only kept if the cut subset is weak in the hostile years, and it has to pass the
fixed-risk measure, not only equal drawdown.
"""
import ast
import io

s = io.open('finer.py', encoding='utf-8').read()
head = s[:s.index("BASE_KEYS = {")]

TAIL = r'''
BASE_KEYS = {'r0', 'r1', 'r2', 'r3', 'r4', 'o1h', 'o2h'}
BOOKS = (('КНИГА 1 (без XLM, AAVE)', [c for c in SP.COINS if c not in ('BILLUSDT', 'XLMUSDT', 'AAVEUSDT')]),
         ('КНИГА 2 (все 15 монет)', [c for c in SP.COINS if c != 'BILLUSDT']))


def fix2(rows):
    r = sim_w.simulate(rows, 0.014)
    return r['eq'], abs(L.dd_of(r['curve']))


def group(n):
    return '1' if n == 1 else ('2-3' if n <= 3 else ('4-6' if n <= 6 else '7+'))


for bi, (book_name, coins) in enumerate(BOOKS):
    sig = make_sig_raw(coins)
    rows, _ = book(sig, BASE_KEYS)
    cnt = collections.Counter(x[0] for x in rows)
    N = [cnt[x[0]] for x in rows]
    beq, bdd = fix2(rows)
    bdd12 = sim_w.money_at_dd(rows, 0.12)[1]
    bpy = pyr(rows)
    print('', flush=True)
    print('  ===== %s: система $%.0f просадка %.1f%% DD12 $%.0f, сделок %d =====' % (book_name, beq, 100 * bdd, bdd12, len(rows)), flush=True)
    print('    пачка  доля сделок  ВР     ср R    по годам ср R                             пачек все в стопе', flush=True)
    for g in ('1', '2-3', '4-6', '7+'):
        idx = [k for k, n in enumerate(N) if group(n) == g]
        if not idx:
            continue
        v = np.array([rows[k][2] for k in idx])
        yrs = []
        for y in range(2022, 2027):
            vy = [rows[k][2] for k in idx if YT[y] <= rows[k][0] < YT[y + 1]]
            yrs.append('%+.3f' % np.mean(vy) if len(vy) >= 10 else '  -   ')
        bursts = collections.defaultdict(list)
        for k in idx:
            bursts[rows[k][0]].append(rows[k][2])
        allstop = [all(r < 0 for r in b) for b in bursts.values() if len(b) >= 2]
        print('    %-5s  %5.1f%%      %4.1f%% %+.4f  %s   %s'
              % (g, 100 * len(idx) / len(rows), 100 * np.mean(v > 0), v.mean(), ' '.join(yrs),
                 ('%.1f%% из %d' % (100 * np.mean(allstop), len(allstop))) if allstop else '-'), flush=True)
    RES = {'система (сейчас)': rows}
    for lbl, wf in (('1/sqrt(n)', lambda n: 1 / np.sqrt(n)), ('1/n', lambda n: 1.0 / n),
                    ('1/sqrt(n), не меньше 0.5', lambda n: max(0.5, 1 / np.sqrt(n))),
                    ('x0.5 при n>=4', lambda n: 0.5 if n >= 4 else 1.0)):
        W = np.array([wf(n) for n in N])
        W = W / W.mean() * 1.0 if False else W
        wr = [x[:5] + (float(w),) for x, w in zip(rows, W)]
        eq, dd = fix2(wr)
        m = sim_w.money_at_dd(wr, 0.12)[1]
        py = pyr(wr)
        RES[lbl] = wr
        ceq, cm = [], []
        for seed in range(5):
            rng = np.random.default_rng(seed)
            Wp = rng.permutation(W)
            cr = [x[:5] + (float(w),) for x, w in zip(rows, Wp)]
            ceq.append(fix2(cr)[0]); cm.append(sim_w.money_at_dd(cr, 0.12)[1])
        print('    размер %-26s $%6.0f %4.1f%% | DD12 $%6.0f | лучше по годам %d/5 (%s) | КОНТРОЛЬ перемешанные веса $%.0f DD12 $%.0f'
              % (lbl, eq, 100 * dd, m, sum(1 for a, b in zip(bpy, py) if b > a), ' '.join('%+.0f' % (b - a) for a, b in zip(bpy, py)),
                 np.median(ceq), np.median(cm)), flush=True)
    if bi == 0:
        print('    -- слепой выбор (мера: $ при 1.4%): 4 года выбор, 5-й замер --', flush=True)
        wins = 0
        for y in range(2022, 2027):
            lo, hi = YT[y], YT[y + 1]
            best, bk = -1, None
            for lbl, rr in RES.items():
                mm = sim_w.simulate([x for x in rr if not (lo <= x[0] < hi)], 0.014)['eq']
                if mm > best:
                    best, bk = mm, lbl
            mp = sim_w.simulate([x for x in RES[bk] if lo <= x[0] < hi], 0.014)['eq']
            mb = sim_w.simulate([x for x in rows if lo <= x[0] < hi], 0.014)['eq']
            wins += int(mp > mb)
            print('      %d выбрано «%s» | $%.0f против $%.0f %s'
                  % (y, bk, mp, mb, 'лучше' if mp > mb else ('так же' if mp == mb else 'хуже')), flush=True)
        print('      лучше нынешней системы в %d из 5 лет' % wins, flush=True)
'''

src = head + TAIL
ast.parse(src)
io.open('burst.py', 'w', encoding='utf-8').write(src)
print('burst.py готов, синтаксис ок')
