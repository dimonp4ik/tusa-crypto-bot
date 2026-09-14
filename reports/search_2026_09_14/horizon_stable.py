"""Does the four-day strength horizon beat the two-day one on more than one book?

Per-trade edge peaks at 48 hours and money peaks at 96, because the longer horizon admits twice the
trades at a slightly smaller edge. That is the kind of finding that died twice today - the coin
screen and the stop geometry both won in the configuration they were found in and lost elsewhere -
so it is measured across the same eight books before anything is said about it.

Configurations: the rule gated or not, coins filtered by the spread gate or not, costs assumed or
measured. Horizons 36 to 144, with the fast side and the selectivity held fixed.
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
import slow_horizon as SH

OLD = dict(SP.SLIP_REAL)
meas = collections.defaultdict(list)
spread = collections.defaultdict(list)
for r in csv.DictReader(open('book_samples.csv')):
    meas[r['coin'] + 'USDT'].append(float(r['cost109']))
    spread[r['coin'] + 'USDT'].append(float(r['spread']))
MEASURED = dict(OLD)
MEASURED.update({k: float(np.median(v)) for k, v in meas.items()})
MED = {k: float(np.median(v)) for k, v in spread.items()}

# slow_horizon only added the extra RSIs for the spread-filtered coins; the "all coins"
# configuration needs them on every coin too.
for _s in G.ALL:
    _T1, _f = G.feats(_s)
    _close = np.asarray(SP.CTX[_s]['B1'][:, 3], dtype=float)
    for _n in (12, 18, 24, 36, 48, 60, 72, 96, 144):
        _k = 'rsiX%d' % _n
        if _k not in _f:
            _f[_k] = SH.rsi(_close, _n)

HOR = [36, 48, 60, 72, 96, 144]
B = float(np.quantile(SH.Q['f'], SH.qf))
THR = {n: float(np.quantile(SH.Q[n], SH.qs)) for n in HOR}
print('  быстрый rsi6 <= %.2f; пороги медленного: %s'
      % (B, ', '.join('%d:%.2f' % (n, THR[n]) for n in HOR)), flush=True)
print('', flush=True)


def money(coins, mode, n):
    bank = SW.run(3.0, 1.0, coins=coins, with_meta=True)
    tr = NA.trades(mode, [('rsiX%d' % n, '>=', THR[n]), ('rsi6', '<=', B)], coins=coins)
    ev = [(x[0], x[1], x[2], x[3], x[4], 0) for x in bank]
    ev += [(x[0], x[1], x[2], x[3], x[4], 1) for x in tr]
    ev.sort(key=lambda z: (z[0], z[5]))
    busy, out = {}, []
    for p, q, R, sf, s, k in ev:
        if busy.get(s, 0) > p:
            continue
        busy[s] = q
        out.append((p, q, R, sf, s, 1.0))
    out.sort()
    return sim_w.money_at_dd(out, 0.12)[1], len(tr)


rows = []
for cost_lbl, table in (('заложенные', OLD), ('измеренные', MEASURED)):
    SP.SLIP_REAL.clear()
    SP.SLIP_REAL.update(table)
    for coin_lbl, filt in (('все монеты', False), ('после отсечки', True)):
        coins = [s for s in G.ALL if not (filt and MED.get(s, 0) > 0.0005)]
        for mode_lbl, mode in (('в гейте', 'гейт'), ('без гейта', 'без режима')):
            cells = []
            for n in HOR:
                m, cnt = money(coins, mode, n)
                cells.append(m)
            best = max(range(len(HOR)), key=lambda i: cells[i])
            print('  %-11s %-14s %-10s %s | лучший %dч'
                  % (cost_lbl, coin_lbl, mode_lbl,
                     ' '.join('$%5.0f%s' % (c, '*' if i == best else ' ')
                              for i, c in enumerate(cells)), HOR[best]), flush=True)
            rows.append(cells)

print('  %-11s %-14s %-10s %s' % ('', '', 'горизонт', ' '.join('%6dч' % n for n in HOR)),
      flush=True)
print('', flush=True)
print('  === место каждого горизонта по восьми книгам ===', flush=True)
rank = collections.defaultdict(list)
wins = collections.Counter()
for cells in rows:
    order = sorted(range(len(HOR)), key=lambda i: -cells[i])
    wins[HOR[order[0]]] += 1
    for place, i in enumerate(order):
        rank[HOR[i]].append(place + 1)
for n in HOR:
    print('    %3dч  лучший %d раз, среднее место %.1f, худшее %d'
          % (n, wins[n], float(np.mean(rank[n])), max(rank[n])), flush=True)
