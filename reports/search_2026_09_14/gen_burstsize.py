"""Writes burstsize.py: size up the widest bursts, size down the 2-3 bursts.

burst.py found the trades that open together with 7+ others are the best of the book (win rate 87-88%,
+0.16..+0.18R, almost never stopped as a group) and 2-3 bursts the weakest (+0.06..+0.10R, 12-13% of
groups fully stopped). Cutting the bursts lost everywhere. The reverse is a size rule, so the
discipline for size rules applies: a boost has to come from a subset that is strong in the hostile
years, it must pass the fixed-risk measure AND equal drawdown, and it must beat the same weights
shuffled across trades.
"""
import ast
import io

s = io.open('burst.py', encoding='utf-8').read()
head = s[:s.index("for bi, (book_name, coins) in enumerate(BOOKS):")]

TAIL = r'''
WEIGHTS = (('x1.5 при n>=7', lambda n: 1.5 if n >= 7 else 1.0),
           ('x1.25 при n>=7', lambda n: 1.25 if n >= 7 else 1.0),
           ('x0.6 при n=2-3', lambda n: 0.6 if 2 <= n <= 3 else 1.0),
           ('x0.75 при n=2-3', lambda n: 0.75 if 2 <= n <= 3 else 1.0),
           ('x1.5 при n>=7 и x0.6 при n=2-3', lambda n: 1.5 if n >= 7 else (0.6 if 2 <= n <= 3 else 1.0)),
           ('x1.25 при n>=4, x0.75 при n=2-3', lambda n: 1.25 if n >= 4 else (0.75 if 2 <= n <= 3 else 1.0)))

for bi, (book_name, coins) in enumerate(BOOKS):
    sig = make_sig_raw(coins)
    rows, _ = book(sig, BASE_KEYS)
    cnt = collections.Counter(x[0] for x in rows)
    N = [cnt[x[0]] for x in rows]
    beq, bdd = fix2(rows)
    bdd12 = sim_w.money_at_dd(rows, 0.12)[1]
    bpy = pyr(rows)
    print('', flush=True)
    print('  ===== %s: система $%.0f просадка %.1f%% DD12 $%.0f =====' % (book_name, beq, 100 * bdd, bdd12), flush=True)
    RES = {'система (сейчас)': rows}
    for lbl, wf in WEIGHTS:
        W = np.array([wf(n) for n in N])
        wr = [x[:5] + (float(w),) for x, w in zip(rows, W)]
        eq, dd = fix2(wr)
        m = sim_w.money_at_dd(wr, 0.12)[1]
        py = pyr(wr)
        RES[lbl] = wr
        ceq, cm, cdd = [], [], []
        for seed in range(5):
            rng = np.random.default_rng(seed)
            cr = [x[:5] + (float(w),) for x, w in zip(rows, rng.permutation(W))]
            e2, d2 = fix2(cr)
            ceq.append(e2); cdd.append(d2); cm.append(sim_w.money_at_dd(cr, 0.12)[1])
        print('    %-34s $%6.0f %4.1f%% | DD12 $%6.0f | лучше по годам %d/5 (%s) | КОНТРОЛЬ перемешанные $%.0f %.1f%% DD12 $%.0f'
              % (lbl, eq, 100 * dd, m, sum(1 for a, b in zip(bpy, py) if b > a), ' '.join('%+.0f' % (b - a) for a, b in zip(bpy, py)),
                 np.median(ceq), 100 * np.median(cdd), np.median(cm)), flush=True)
    if bi == 0:
        print('    -- слепой выбор (мера: $ при DD12, чтобы не выбирать плечо): 4 года выбор, 5-й замер при 1.4% --', flush=True)
        wins = 0
        for y in range(2022, 2027):
            lo, hi = YT[y], YT[y + 1]
            best, bk = -1, None
            for lbl, rr in RES.items():
                mm = sim_w.money_at_dd([x for x in rr if not (lo <= x[0] < hi)], 0.12)[1]
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
io.open('burstsize.py', 'w', encoding='utf-8').write(src)
print('burstsize.py готов, синтаксис ок')
