"""Is stop 2.5 / target 1.3 better, or better only in the configuration it was found in?

Earlier today target 1.3 was measured as WORSE than 1.0 - $2,499 against $2,595 - with the gated
rule, assumed costs and AAVE in the book. Now, with the gateless rule, measured costs and the spread
gate, it is better. A sign that flips with configuration is exactly what killed the coin screen this
afternoon, and the rule established there applies here: measure on more than one book, and let
disagreement bury the finding.

Four configurations: the rule gated or not, the coins filtered by the spread gate or not. Plus the
cost tables, assumed and measured. If 2.5/1.3 wins everywhere it is real geometry; if it wins only
where it was found, it is another coin screen.
"""
import collections
import csv
import sys

import numpy as np

sys.path.insert(0, 'C:/Users/Lenovo/Desktop/Торговля/crypto-bot')
import struct_params as SP
import gauntlet2 as G
import stop_width as SW
import sim_w
import nogate_attack as NA
from src import pullback_bank as PB

OLD = dict(SP.SLIP_REAL)
meas = collections.defaultdict(list)
spread = collections.defaultdict(list)
for r in csv.DictReader(open('book_samples.csv')):
    meas[r['coin'] + 'USDT'].append(float(r['cost109']))
    spread[r['coin'] + 'USDT'].append(float(r['spread']))
MEASURED = dict(OLD)
MEASURED.update({k: float(np.median(v)) for k, v in meas.items()})
MED = {k: float(np.median(v)) for k, v in spread.items()}

C = [('rsi48', '>=', 56.3761), ('rsi6', '<=', 33.7947)]
BIG = 10 ** 9
GEOM = [(3.0, 1.0), (2.5, 1.0), (2.5, 1.3), (3.0, 1.3), (2.0, 1.3), (2.5, 1.6)]


def rule_trades(mode, sl, tp, coins):
    rows = []
    for s in coins:
        sig = NA.sigs(s, mode, C)
        c = SP.CTX[s]
        slip = SP.SLIP_REAL.get(s, 0.0002)
        t15, A15, pos = c['t15'], c['a15'], c['pos']
        busy = 0
        for close in sorted(sig):
            if close < busy:
                continue
            lim, atr = sig[close]
            i = pos.get(close)
            if i is None or i + PB.HOLD_BARS > len(t15):
                continue
            o, h, l, cc = (A15[i:i + PB.HOLD_BARS, x] for x in range(4))
            if o[0] <= lim:
                e, touch = o[0] * (1 + slip), False
            elif l[0] <= lim:
                e, touch = lim * (1 + slip), True
            else:
                continue
            TP, SL = e + tp * atr, e - sl * atr
            hs, ht = l <= SL, h >= TP
            if touch:
                ht = ht.copy()
                ht[0] = False
            js = int(np.argmax(hs)) if hs.any() else BIG
            jt = int(np.argmax(ht)) if ht.any() else BIG
            if js <= jt and js < BIG:
                jj, f_ = js, min(SL, o[js]) * (1 - slip)
            elif jt < BIG:
                jj, f_ = jt, TP * (1 - slip)
            else:
                jj, f_ = PB.HOLD_BARS - 1, cc[-1] * (1 - slip)
            ret = (f_ / e - 1) - 0.0004
            rows.append((close, int(t15[i + jj]) + 900, ret / (sl * atr / e), sl * atr / e, s))
            busy = int(t15[i + jj]) + 900
    rows.sort()
    return rows


def money(bank, cand):
    ev = [(a, b, R, sf, s, 0) for a, b, R, sf, s in bank]
    ev += [(a, b, R, sf, s, 1) for a, b, R, sf, s in cand]
    ev.sort(key=lambda x: (x[0], x[5]))
    busy, out = {}, []
    for a, b, R, sf, s, k in ev:
        if busy.get(s, 0) > a:
            continue
        busy[s] = b
        out.append((a, b, R, sf, s, 1.0))
    out.sort()
    return sim_w.money_at_dd(out, 0.12)[1]


print('  геометрия правила: %s' % ', '.join('%.1f/%.1f' % g for g in GEOM), flush=True)
print('', flush=True)
rows_out = []
for cost_lbl, table in (('заложенные', OLD), ('измеренные', MEASURED)):
    SP.SLIP_REAL.clear()
    SP.SLIP_REAL.update(table)
    G.FC.clear()
    for coin_lbl, filt in (('все монеты', False), ('после отсечки спреда', True)):
        coins = [s for s in G.ALL if not (filt and MED.get(s, 0) > 0.0005)]
        bank = SW.run(3.0, 1.0, coins=coins, with_meta=True)
        for mode_lbl, mode in (('в гейте', 'гейт'), ('без гейта', 'без режима')):
            cells = []
            for sl, tp in GEOM:
                cells.append(money(bank, rule_trades(mode, sl, tp, coins)))
            best = max(range(len(GEOM)), key=lambda i: cells[i])
            print('  %-11s %-21s %-10s %s | лучшая %.1f/%.1f'
                  % (cost_lbl, coin_lbl, mode_lbl,
                     ' '.join('$%5.0f%s' % (c, '*' if i == best else ' ')
                              for i, c in enumerate(cells)),
                     GEOM[best][0], GEOM[best][1]), flush=True)
            rows_out.append((cost_lbl, coin_lbl, mode_lbl, cells))

print('', flush=True)
print('  === сколько раз каждая геометрия оказалась лучшей из 8 конфигураций ===', flush=True)
cnt = collections.Counter()
rank = collections.defaultdict(list)
for _, _, _, cells in rows_out:
    order = sorted(range(len(GEOM)), key=lambda i: -cells[i])
    cnt[GEOM[order[0]]] += 1
    for place, i in enumerate(order):
        rank[GEOM[i]].append(place + 1)
for g in GEOM:
    print('    %.1f/%.1f  лучшая %d раз, среднее место %.1f, худшее место %d'
          % (g[0], g[1], cnt[g], float(np.mean(rank[g])), max(rank[g])), flush=True)
