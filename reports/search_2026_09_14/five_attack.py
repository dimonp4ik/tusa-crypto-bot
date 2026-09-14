"""The five that add money, attacked the way the pullback rule was.

Adding money on the whole sample is where the last candidate got to before its threshold grid killed
it: the edge lived in a single column with a cliff on either side, because its two conditions nearly
contradict each other and its trades were a sliver on the boundary. That test is run first here,
since it is the cheapest and it is what actually decides.

Then the rest: each condition alone, the inverse, the mirror, the years, the coins.
"""
import collections
import datetime
import sys

import numpy as np

sys.path.insert(0, 'C:/Users/Lenovo/Desktop/Торговля/crypto-bot')
import struct_params as SP
import gauntlet2 as G
import full_money as FM

FLIP = {'>=': '<=', '<=': '>='}
YT = FM.YT
COINS = FM.COINS

CASES = [
    ('SHORT падение при узком диапазоне', False,
     [('ret24', '<=', -4.4872), ('vol_rel', '<=', 0.4697)]),
    ('LONG рывок при слабом рынке', True,
     [('breadth_med', '<=', -1.8818), ('ret3', '>=', 1.2746)]),
    ('LONG падение при волатильности', True,
     [('ret24', '<=', -1.2173), ('vol168', '>=', 1.5867)]),
    ('SHORT перепроданность в ровном тренде', False,
     [('ema_slope', '>=', -0.5145), ('rsi6', '<=', 25.1292)]),
    ('SHORT рывок при слабом RSI', False,
     [('ret6', '>=', 0.5852), ('rsi14', '<=', 34.0162)]),
]


def st(conds, side, lab):
    rows, _ = G.simulate(conds, side, coins=COINS)
    v = np.array([r[2] for r in rows]) if rows else np.zeros(0)
    if not len(v):
        print('    %-36s сделок нет' % lab, flush=True)
        return None
    print('    %-36s n%5d ВР%5.1f%% ср%+.4fR' % (lab, len(v), 100 * np.mean(v > 0), v.mean()),
          flush=True)
    return rows


for name, side, C in CASES:
    print('', flush=True)
    print('  ================ %s ================' % name, flush=True)
    a_name, a_op, a_thr = C[0]
    b_name, b_op, b_thr = C[1]

    def grid_vals(nm, thr, side_):
        col = np.array([G.feats(s)[1][nm] for s in COINS[:1]][0])
        vals = []
        for s in COINS:
            vals.append(G.feats(s)[1][nm])
        allv = np.concatenate(vals)
        allv = allv[np.isfinite(allv)]
        qs = np.quantile(allv, np.linspace(0.03, 0.97, 40))
        near = sorted(qs, key=lambda x: abs(x - thr))[:5]
        out = sorted(set([float(x) for x in near] + [float(thr)]))
        return out

    GA = grid_vals(a_name, a_thr, side)
    GB = grid_vals(b_name, b_thr, side)
    print('    сетка порогов (среднее R / сделок), рабочая клетка помечена *', flush=True)
    print('      %-10s %s' % (a_name + '\\' + b_name,
                              ' '.join('%13.4g' % b for b in GB)), flush=True)
    for a in GA:
        cells = []
        for b in GB:
            rr, _ = G.simulate([(a_name, a_op, a), (b_name, b_op, b)], side, coins=COINS)
            v = np.array([x[2] for x in rr]) if rr else np.zeros(0)
            mark = '*' if (abs(a - a_thr) < 1e-9 and abs(b - b_thr) < 1e-9) else ' '
            cells.append(('%+.4f/%4d%s' % (v.mean(), len(v), mark)) if len(v) else '      -      ')
        print('      %10.4g %s' % (a, ' '.join(cells)), flush=True)

    print('    атаки:', flush=True)
    base = st(C, side, 'ЦЕЛИКОМ')
    for c in C:
        st([c], side, 'только %s%s%.4f' % c)
    st([(n, FLIP[o], t) for n, o, t in C], side, 'ИНВЕРСИЯ (должна терять)')
    st(C, not side, 'ЗЕРКАЛО (должно терять)')
    if base:
        print('    по годам:', flush=True)
        line = []
        for y in range(2022, 2027):
            sub = [r[2] for r in base if YT[y] <= r[0] < YT[y + 1]]
            line.append('%d %s' % (y, ('%+.3f(%d)' % (np.mean(sub), len(sub))) if len(sub) >= 10
                                   else 'мало'))
        print('      %s' % '  '.join(line), flush=True)
        per = collections.defaultdict(list)
        for r in base:
            per[r[4]].append(r[2])
        pos = sum(1 for s in per if np.mean(per[s]) > 0)
        print('    по монетам: плюсовых %d из %d' % (pos, len(per)), flush=True)
