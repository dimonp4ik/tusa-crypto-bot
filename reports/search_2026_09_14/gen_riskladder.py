"""Writes riskladder.py: the risk ladder of the FINAL system, with the live latch on.

The owner's target is 30% a month. The old ladder was measured on the bank alone, before the pullback
sources, the breadth source and the size rules, and it said 30% needs ~7% risk at ~45% drawdown. The
system has changed, so the ladder is re-measured honestly: risk 1.0 / 1.4 / 2.0 / 2.5 / 3.0 / 4.0 /
5.0%, the final system (items 8, 9, 10), 85% fill, ten draws each, with every live guard in place -
including the 15% drawdown latch that switches the account off for good.

Per book and risk: median money from $120, percent a month, times a year, median and worst drawdown,
and in how many of the ten draws the latch fired. The latch is the point of the table: above some risk
the account does not earn more, it dies.
"""
import ast
import io

s = io.open('srcweight.py', encoding='utf-8').read()
head = s[:s.index("VARS = (")]

TAIL = r'''
RISKS = (0.010, 0.014, 0.020, 0.025, 0.030, 0.040, 0.050)

for book_name, coins in BOOKS3:
    H = {s: coin_hours(s) for s in coins}
    base_sig = make_sig_raw(coins)
    extra = roll_entries(coins, H, 'r2_4', 4, 2, 4, 0)
    sig_all = sorted(base_sig + extra, key=lambda x: (x[0], x[1]))
    GEO.clear(); GEO.update(BASE_GEO)
    print('', flush=True)
    print('  ===== %s: итоговая система, заливка 85%%, медиана 10 розыгрышей =====' % book_name, flush=True)
    print('    риск    $ из 120    в месяц   за год   просадка (худшая)   защёлка сработала', flush=True)
    subs = []
    for seed in range(1, 11):
        rng = np.random.default_rng(seed)
        sub = [x for x in sig_all if rng.random() < 0.85]
        r0, k0 = book_keyed(sub, FULL)
        subs.append(weigh_src(r0, k0, {'r2_4'}, 1.25))
    for rk in RISKS:
        res = []
        for rows in subs:
            r = sim_w.simulate(rows, rk)
            months = (rows[-1][1] - rows[0][0]) / (365.25 * 86400 / 12)
            res.append((r['eq'], (r['eq'] / 120) ** (1 / months) - 1, (r['eq'] / 120) ** (12 / months),
                        abs(L.dd_of(r['curve'])), bool(r['paused_at'])))
        med = lambda i: float(np.median([x[i] for x in res]))
        print('    %4.1f%%   $%7.0f   %+6.2f%%   x%5.2f   %5.1f%% (%5.1f%%)      %d из 10'
              % (100 * rk, med(0), 100 * med(1), med(2), 100 * med(3), 100 * max(x[3] for x in res),
                 sum(1 for x in res if x[4])), flush=True)
    live = []
    for seed in range(1, 11):
        rng = np.random.default_rng(seed)
        sub = [x for x in sig_all if rng.random() < 0.85]
        rl, _ = book_keyed(sub, BASE_KEYS)
        rows = [x[:5] + (1.0,) for x in rl]
        r = sim_w.simulate(rows, 0.014)
        months = (rows[-1][1] - rows[0][0]) / (365.25 * 86400 / 12)
        live.append((r['eq'], (r['eq'] / 120) ** (1 / months) - 1, abs(L.dd_of(r['curve']))))
    print('    для сравнения живая система при 1.4%%: $%.0f, %+.2f%%/мес, просадка %.1f%%'
          % (np.median([x[0] for x in live]), 100 * np.median([x[1] for x in live]), 100 * np.median([x[2] for x in live])), flush=True)
'''

src = head + TAIL
ast.parse(src)
io.open('riskladder.py', 'w', encoding='utf-8').write(src)
print('riskladder.py готов, синтаксис ок')
