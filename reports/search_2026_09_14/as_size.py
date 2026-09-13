"""The rules taken apart: as size on the bank's own trades, not as entries of their own.

Capitulation is the most robust signal found - its breadth threshold reproduced in all five blind
re-picks - and it still makes the account poorer, because a new entry costs a slot and arrives
alongside the bank's own losses. A condition can be worth knowing without being worth entering on.

So the same conditions are applied a second way: when they hold at the moment of a bank trade, the
bank trade is sized up; the entry set does not change at all. That is the one comparison the project
can actually prove, because the trade set is identical and only the money moves.

Both directions are tried: sizing up where the condition holds, and sizing down where it does not.
"""
import collections
import datetime
import sys

import numpy as np

sys.path.insert(0, 'C:/Users/Lenovo/Desktop/Торговля/crypto-bot')
import gauntlet2 as G
import feat2
import struct_params as SP
import stop_width as SW
import margin_cap as MC
import live_rules_sim as L
import sim_w

bank = SW.run(3.0, 1.0, coins=G.ALL, with_meta=True)
CAND = [
    ('капитуляция', [('breadth', '<=', 0.0625), ('vol720', '>=', 1.4462)]),
    ('широта одна', [('breadth', '<=', 0.0625)]),
    ('широта <=0.25', [('breadth', '<=', 0.25)]),
    ('широта >=0.75', [('breadth', '>=', 0.75)]),
    ('откат', [('rsi48', '>=', 56.3761), ('rsi6', '<=', 33.7947)]),
    ('rsi6 низкий', [('rsi6', '<=', 33.7947)]),
    ('vol720 высокий', [('vol720', '>=', 1.4462)]),
]

# feature values at each bank trade's own hour
FV = {}
for s in G.ALL:
    T1, f = G.feats(s)
    idx = {int(t) + 3600: i for i, t in enumerate(T1)}
    FV[s] = (idx, f)


def holds(conds, a, s):
    idx, f = FV[s]
    i = idx.get(a)
    if i is None:
        return None
    for nm, op, thr in conds:
        x = f[nm][i]
        if not np.isfinite(x):
            return None
        if (x < thr) if op == '>=' else (x > thr):
            return False
    return True


money_at_dd = sim_w.money_at_dd

base = [(a, b, R, sf, s, 1.0) for a, b, R, sf, s in bank]
k0, m0 = money_at_dd(base)
print('  банк как есть: сделок %d, риск %.3f%%, $%.0f' % (len(base), 100 * k0, m0), flush=True)
print('', flush=True)

for nm, conds in CAND:
    flag = {}
    for a, b, R, sf, s in bank:
        flag[(a, s)] = holds(conds, a, s)
    hit = [x for x in bank if flag[(x[0], x[4])] is True]
    miss = [x for x in bank if flag[(x[0], x[4])] is False]
    unk = len(bank) - len(hit) - len(miss)
    if len(hit) < 60:
        print('  %-16s сработало всего %d сделок банка — мало' % (nm, len(hit)), flush=True)
        continue
    vh = np.array([x[2] for x in hit])
    vm = np.array([x[2] for x in miss])
    print('  %-16s банка под условием %4d ср%+.4f ВР%5.1f%% | вне %4d ср%+.4f ВР%5.1f%% | нет данных %d'
          % (nm, len(hit), vh.mean(), 100 * np.mean(vh > 0), len(miss), vm.mean(),
             100 * np.mean(vm > 0), unk), flush=True)
    for mult in (1.25, 1.5, 2.0, 0.75, 0.5, 0.0):
        rows = [(a, b, R, sf, s, mult if flag[(a, s)] is True else 1.0)
                for a, b, R, sf, s in bank]
        k, m = money_at_dd(rows)
        print('      размер x%.2f под условием: риск %.3f%%  $%6.0f  (%+.0f%%)'
              % (mult, 100 * k, m, 100 * (m / m0 - 1)), flush=True)

    # year by year, to see whether the split is one year's accident
    YT = {y: int(datetime.datetime(y, 1, 1, tzinfo=datetime.UTC).timestamp())
          for y in range(2022, 2028)}
    line = []
    for y in range(2022, 2027):
        h = [x[2] for x in hit if YT[y] <= x[0] < YT[y + 1]]
        m_ = [x[2] for x in miss if YT[y] <= x[0] < YT[y + 1]]
        if len(h) >= 12 and len(m_) >= 12:
            line.append('%d %+.3f/%+.3f' % (y, np.mean(h), np.mean(m_)))
    print('      по годам (под условием / вне): %s' % '  '.join(line), flush=True)
