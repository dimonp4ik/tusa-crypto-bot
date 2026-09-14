"""What is the best way to say "strong"? A search driven by the mechanism, not by the data.

The rule is now understood: buy oversold inside strength, where strength is a two-day RSI held high
and oversold is a six-hour RSI pushed low. Removing the strength condition costs 0.19R, so that half
is carrying the rule. But RSI is only one way to say strong. The feature set has several others -
return over 48, 96 and 168 hours, distance from the 50-period EMA, the EMA's slope, position in the
recent range, strength relative to the market, and BTC's own move.

Substituting them one at a time, at the same selectivity and with the oversold half untouched, asks
a mechanical question rather than mining: if the mechanism is real, several of these should work,
and the best one is a better instrument for the same idea. If only rsi48 works, the "mechanism" was
a description of one lucky pair.

Every candidate is measured on four books, since two findings died today for lack of that.
"""
import collections
import csv
import datetime
import sys

import numpy as np

sys.path.insert(0, 'C:/Users/Lenovo/Desktop/Торговля/crypto-bot')
import struct_params as SP
import gauntlet2 as G
import stop_width as SW
import sim_w
import nogate_attack as NA

sp = collections.defaultdict(list)
cost = collections.defaultdict(list)
for r in csv.DictReader(open('book_frozen.csv')):
    sp[r['coin'] + 'USDT'].append(float(r['spread']))
    cost[r['coin'] + 'USDT'].append(float(r['cost109']))
SP.SLIP_REAL.update({k: float(np.median(v)) for k, v in cost.items()})
G.FC.clear()
MED = {k: float(np.median(v)) for k, v in sp.items()}
ALL = list(G.ALL)
YT = {y: int(datetime.datetime(y, 1, 1, tzinfo=datetime.UTC).timestamp()) for y in range(2022, 2028)}

FAST = ('rsi6', '<=', 33.7947)
PROXIES = ['rsi48', 'ret48', 'ret96', 'ret168', 'dist_e50', 'ema_slope', 'ema_gap',
           'pos48', 'pos96', 'rs24', 'btc48' if 'btc48' in G.feats(ALL[0])[1] else 'btc72']

vals = collections.defaultdict(list)
for s in ALL:
    _, f = G.feats(s)
    for n in set(PROXIES + ['rsi48']):
        if n in f:
            v = f[n]
            vals[n].append(v[np.isfinite(v)])
Q = {k: np.concatenate(v) for k, v in vals.items() if v}
q_strength = float((Q['rsi48'] <= 56.3761).mean())
print('  условие силы берётся на квантиле %.3f (верхние %.0f%% значений)'
      % (q_strength, 100 * (1 - q_strength)), flush=True)
print('  быстрое условие неизменно: rsi6 <= 33.79', flush=True)
print('', flush=True)

BOOKS = [('все 15', ALL, 'без режима'),
         ('без AAVE и XLM', [s for s in ALL if s not in ('AAVEUSDT', 'XLMUSDT')], 'без режима'),
         ('все 15, в гейте', ALL, 'гейт'),
         ('без AAVE и XLM, в гейте',
          [s for s in ALL if s not in ('AAVEUSDT', 'XLMUSDT')], 'гейт')]
BANKS = {}
for lbl, coins, mode in BOOKS:
    key = tuple(coins)
    if key not in BANKS:
        BANKS[key] = SW.run(3.0, 1.0, coins=coins, with_meta=True)


def money(coins, mode, conds):
    bank = BANKS[tuple(coins)]
    cand = NA.trades(mode, conds, coins=coins)
    ev = [(x[0], x[1], x[2], x[3], x[4], 0) for x in bank]
    ev += [(x[0], x[1], x[2], x[3], x[4], 1) for x in cand]
    ev.sort(key=lambda z: (z[0], z[5]))
    busy, out = {}, []
    for p, q, R, sf, s, k in ev:
        if busy.get(s, 0) > p:
            continue
        busy[s] = q
        out.append((p, q, R, sf, s, 1.0))
    out.sort()
    return sim_w.money_at_dd(out, 0.12)[1], cand


print('  признак силы   порог      сделок    ВР      ср R    по годам                       деньги по 4 книгам',
      flush=True)
rows = []
for nm in PROXIES:
    if nm not in Q:
        continue
    thr = float(np.quantile(Q[nm], q_strength))
    conds = [(nm, '>=', thr), FAST]
    cells, first = [], None
    for lbl, coins, mode in BOOKS:
        m, cand = money(coins, mode, conds)
        cells.append(m)
        if first is None:
            first = cand
    if not first or len(first) < 80:
        print('  %-12s %8.4f  сделок %d — мало' % (nm, thr, len(first) if first else 0), flush=True)
        continue
    v = np.array([x[2] for x in first])
    yr = []
    for y in range(2022, 2027):
        sub = [x[2] for x in first if YT[y] <= x[0] < YT[y + 1]]
        yr.append('%+.3f' % np.mean(sub) if len(sub) >= 10 else '  -  ')
    mark = '  <- рабочее' if nm == 'rsi48' else ''
    print('  %-12s %8.4f  %6d %6.1f%% %+7.4f  %s  %s%s'
          % (nm, thr, len(v), 100 * np.mean(v > 0), v.mean(), ' '.join(yr),
             ' '.join('$%5.0f' % c for c in cells), mark), flush=True)
    rows.append((nm, cells, v.mean()))

print('', flush=True)
print('  === сколько книг каждый признак выигрывает у rsi48 ===', flush=True)
base = dict((nm, cells) for nm, cells, _ in rows)['rsi48']
for nm, cells, mu in sorted(rows, key=lambda r: -sum(c > b for c, b in zip(r[1], base))):
    if nm == 'rsi48':
        continue
    w = sum(1 for c, b in zip(cells, base) if c > b)
    print('    %-12s лучше в %d книгах из 4  (%s)'
          % (nm, w, ' '.join('%+5.0f' % (c - b) for c, b in zip(cells, base))), flush=True)
