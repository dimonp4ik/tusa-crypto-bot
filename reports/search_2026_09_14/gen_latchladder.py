"""Writes latchladder.py: is the ceiling the strategy or the latch?

The risk ladder says 1.4% risk is the peak: at 2% the 15% drawdown latch fires in 8-10 draws out of 10
and the account earns less, at 5% it earns nothing. That ceiling is set by a guard, not by the edge, so
the guard is measured: latch at 15% (live), 20%, 25% and switched off, crossed with risk 1.4 / 2.0 /
2.5 / 3.0 / 4.0%.

This is a risk decision, not a strategy change - nothing here is proposed. The table answers one
question honestly: what the owner would be buying, and with how much drawdown, if the latch were
loosened. Per book and cell: money from $120, percent a month, median and worst drawdown, and how many
draws hit the latch.
"""
import ast
import io

s = io.open('srcweight.py', encoding='utf-8').read()
head = s[:s.index("VARS = (")]

TAIL = r'''
RISKS = (0.014, 0.020, 0.025, 0.030, 0.040)
LATCHES = (0.15, 0.20, 0.25, None)


def sim_latch(trades, target, latch):
    """sim_w.simulate with a different (or no) drawdown latch."""
    old = sim_w.MAX_DD
    try:
        sim_w.MAX_DD = latch if latch is not None else 10.0
        return sim_w.simulate(trades, target)
    finally:
        sim_w.MAX_DD = old


for book_name, coins in BOOKS3:
    H = {s: coin_hours(s) for s in coins}
    base_sig = make_sig_raw(coins)
    extra = roll_entries(coins, H, 'r2_4', 4, 2, 4, 0)
    sig_all = sorted(base_sig + extra, key=lambda x: (x[0], x[1]))
    GEO.clear(); GEO.update(BASE_GEO)
    subs = []
    for seed in range(1, 11):
        rng = np.random.default_rng(seed)
        sub = [x for x in sig_all if rng.random() < 0.85]
        r0, k0 = book_keyed(sub, FULL)
        subs.append(weigh_src(r0, k0, {'r2_4'}, 1.25))
    months = float(np.median([(rows[-1][1] - rows[0][0]) / (365.25 * 86400 / 12) for rows in subs]))
    print('', flush=True)
    print('  ===== %s: итоговая система, заливка 85%%, медиана 10 розыгрышей =====' % book_name, flush=True)
    print('    защёлка   риск    $ из 120    в месяц   просадка (худшая)   защёлка сработала', flush=True)
    for latch in LATCHES:
        for rk in RISKS:
            res = []
            for rows in subs:
                r = sim_latch(rows, rk, latch)
                res.append((r['eq'], abs(L.dd_of(r['curve'])), bool(r['paused_at'])))
            med = lambda i: float(np.median([x[i] for x in res]))
            eq = med(0)
            print('    %-9s %4.1f%%   $%7.0f   %+6.2f%%   %5.1f%% (%5.1f%%)      %d из 10'
                  % (('%.0f%%' % (100 * latch)) if latch else 'выключена', 100 * rk, eq,
                     100 * ((eq / 120) ** (1 / months) - 1), 100 * med(1), 100 * max(x[1] for x in res),
                     sum(1 for x in res if x[2])), flush=True)
        print('', flush=True)
'''

src = head + TAIL
ast.parse(src)
io.open('latchladder.py', 'w', encoding='utf-8').write(src)
print('latchladder.py готов, синтаксис ок')
