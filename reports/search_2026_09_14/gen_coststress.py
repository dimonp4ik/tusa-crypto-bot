"""Writes coststress.py: what happens to the proposal if the real costs are worse than measured.

Every number in these reports rests on the costs measured from book_frozen.csv plus a 0.0004 fee. The
live market can be worse: a thinner book, a fast hour, a wider spread. The proposal adds trades on
coins with a shallower pullback (item 9), and those could be more cost-sensitive than the core - if so,
the proposal degrades faster than the system it replaces and the edge is partly an artefact of the cost
assumption.

Scenarios: measured costs (base), slippage x1.5, slippage x2, fee x2, and slippage x1.5 with fee x2.
For each book and scenario, both systems are rebuilt from scratch with the scaled costs: the live
system (bank + pullback 1h/2h + shorts) and the final system (plus items 8, 9, 10). Reported: money at
1.4%, drawdown, money at 12% drawdown, trades, win rate - and the gap between the two systems, which is
what actually matters: the proposal must keep its lead under worse costs, not merely stay positive.
"""
import ast
import io

s = io.open('srcweight.py', encoding='utf-8').read()
head = s[:s.index("VARS = (")]

TAIL = r'''
SLIP0 = dict(SLIP)
FEE0 = FEE
SCEN = (('измеренные издержки', 1.0, 1.0), ('проскальзывание x1.5', 1.5, 1.0), ('проскальзывание x2', 2.0, 1.0),
        ('комиссия x2', 1.0, 2.0), ('проскальзывание x1.5 и комиссия x2', 1.5, 2.0))

for book_name, coins in BOOKS3:
    H = {s: coin_hours(s) for s in coins}
    print('', flush=True)
    print('  ===== %s =====' % book_name, flush=True)
    print('    сценарий                          живая: $ / просадка / DD12          итоговая: $ / просадка / DD12      разрыв', flush=True)
    for lbl, ks, kf in SCEN:
        SLIP.clear()
        SLIP.update({k: v * ks for k, v in SLIP0.items()})
        globals()['FEE'] = FEE0 * kf
        GEO.clear(); GEO.update(BASE_GEO)
        base_sig = make_sig_raw(coins)
        extra = roll_entries(coins, H, 'r2_4', 4, 2, 4, 0)
        sig_all = sorted(base_sig + extra, key=lambda x: (x[0], x[1]))
        rl, kl = book_keyed(sig_all, BASE_KEYS)
        live = [x[:5] + (1.0,) for x in rl]
        rf, kf2 = book_keyed(sig_all, FULL)
        fin = weigh_src(rf, kf2, {'r2_4'}, 1.25)
        le, ld = fix2(live); lm = sim_w.money_at_dd(live, 0.12)[1]
        fe, fd = fix2(fin); fm = sim_w.money_at_dd(fin, 0.12)[1]
        vl = np.array([x[2] for x in live]); vf = np.array([x[2] for x in fin])
        print('    %-33s $%6.0f %4.1f%% $%6.0f (%4d сд, ВР %4.1f%%)   $%6.0f %4.1f%% $%6.0f (%4d сд, ВР %4.1f%%)   %+5.0f%% / %+5.0f%%'
              % (lbl, le, 100 * ld, lm, len(vl), 100 * np.mean(vl > 0), fe, 100 * fd, fm, len(vf), 100 * np.mean(vf > 0),
                 100 * (fe / le - 1) if le > 0 else float('nan'), 100 * (fm / lm - 1) if lm > 0 else float('nan')), flush=True)
    SLIP.clear(); SLIP.update(SLIP0)
    globals()['FEE'] = FEE0
'''

src = head + TAIL
ast.parse(src)
io.open('coststress.py', 'w', encoding='utf-8').write(src)
print('coststress.py готов, синтаксис ок')
