"""Writes fillstress.py: what if fewer signals actually get filled than 85%?

Every headline number uses an 85% fill - the share measured on real X-Perp candles. Live can be worse:
a missed candle, a symbol muted by the spread gate, a slow scan. The breadth source (item 9) has only
52-87 trades over 4.3 years, so a thinner fill could wipe out its contribution while the core keeps
trading - that would make the proposal's edge fragile in exactly the way that matters.

Fills 95 / 85 / 75 / 65%, ten paired draws each (the same subset for both systems), three books. For
each: median money at 1.4%, median drawdown, money at 12% drawdown, trades, breadth trades that
survived the fill, and the gap between the final and the live system on both measures.
"""
import ast
import io

s = io.open('srcweight.py', encoding='utf-8').read()
head = s[:s.index("VARS = (")]

TAIL = r'''
FILLS = (0.95, 0.85, 0.75, 0.65)

for book_name, coins in BOOKS3:
    H = {s: coin_hours(s) for s in coins}
    base_sig = make_sig_raw(coins)
    extra = roll_entries(coins, H, 'r2_4', 4, 2, 4, 0)
    sig_all = sorted(base_sig + extra, key=lambda x: (x[0], x[1]))
    GEO.clear(); GEO.update(BASE_GEO)
    print('', flush=True)
    print('  ===== %s =====' % book_name, flush=True)
    print('    заливка   живая: $ / просадка / DD12        итоговая: $ / просадка / DD12    сделок ширины   разрыв', flush=True)
    for f in FILLS:
        LV, FN, NW = [], [], []
        for seed in range(1, 11):
            rng = np.random.default_rng(seed)
            sub = [x for x in sig_all if rng.random() < f]
            rl, _ = book_keyed(sub, BASE_KEYS)
            rf, kf = book_keyed(sub, FULL)
            live = [x[:5] + (1.0,) for x in rl]
            fin = weigh_src(rf, kf, {'r2_4'}, 1.25)
            e0, d0 = fix2(live); e1, d1 = fix2(fin)
            LV.append((e0, d0, sim_w.money_at_dd(live, 0.12)[1], len(live)))
            FN.append((e1, d1, sim_w.money_at_dd(fin, 0.12)[1], len(fin)))
            NW.append(sum(1 for k in kf if k == 'r2_4'))
        med = lambda X, i: float(np.median([x[i] for x in X]))
        print('    %3d%%      $%6.0f %4.1f%% $%6.0f (%4.0f сд)   $%6.0f %4.1f%% $%6.0f (%4.0f сд)   %3.0f          %+5.0f%% / %+5.0f%%'
              % (100 * f, med(LV, 0), 100 * med(LV, 1), med(LV, 2), med(LV, 3),
                 med(FN, 0), 100 * med(FN, 1), med(FN, 2), med(FN, 3), np.median(NW),
                 100 * (med(FN, 0) / med(LV, 0) - 1), 100 * (med(FN, 2) / med(LV, 2) - 1)), flush=True)
'''

src = head + TAIL
ast.parse(src)
io.open('fillstress.py', 'w', encoding='utf-8').write(src)
print('fillstress.py готов, синтаксис ок')
