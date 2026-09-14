"""How much of the headline depends on one coin sitting on the threshold?

The book sample grew while today's scripts were running, DOT's median spread crossed 0.05% from
below, the spread-gated coin set went from thirteen coins to twelve, and the money moved from $2,906
to $5,257. A number that swings by three quarters when one borderline coin leaves is not a number.

The cost table is frozen now (book_frozen.csv, the complete 26-hour sample). This measures the
account across every coin set the threshold could plausibly produce, so the range is visible instead
of being reported as a point.
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
C = [('rsi48', '>=', 56.3761), ('rsi6', '<=', 33.7947)]
YT = {y: int(datetime.datetime(y, 1, 1, tzinfo=datetime.UTC).timestamp()) for y in range(2022, 2028)}

print('  замороженная таблица, медианы спреда:', flush=True)
for s in sorted(MED, key=lambda x: -MED[x]):
    print('    %-6s %.6f' % (s.replace('USDT', ''), MED[s]), flush=True)


def money(coins, with_rule):
    bank = SW.run(3.0, 1.0, coins=coins, with_meta=True)
    cand = NA.trades('без режима', C, coins=coins) if with_rule else []
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
    m = sim_w.money_at_dd(out, 0.12)[1]
    return m, len(out), len(cand)


ALL = list(G.ALL)
print('', flush=True)
print('  === счёт при разных порогах отсечки спреда ===', flush=True)
print('  порог     монет  выброшены                     банк    банк+откат  во сколько раз', flush=True)
for lim in (0.0003, 0.0004, 0.0005, 0.0006, 0.0008, 0.0012, 0.0020, 0.0050):
    coins = [s for s in ALL if MED.get(s, 0) <= lim]
    if len(coins) < 6:
        continue
    out = [s.replace('USDT', '') for s in ALL if s not in coins]
    m0, n0, _ = money(coins, False)
    m1, n1, nc = money(coins, True)
    print('  %.4f    %2d    %-28s $%6.0f    $%6.0f      %.2f'
          % (lim, len(coins), ','.join(out) or '—', m0, m1, m1 / m0), flush=True)

print('', flush=True)
print('  === и без всякой отсечки, просто выбивая монеты по одной ===', flush=True)
m0, _, _ = money(ALL, False)
m1, _, _ = money(ALL, True)
print('    все 15: банк $%.0f, банк+откат $%.0f (x%.2f)' % (m0, m1, m1 / m0), flush=True)
for drop in ('AAVEUSDT', 'XLMUSDT', 'DOTUSDT'):
    coins = [s for s in ALL if s != drop]
    a, _, _ = money(coins, False)
    b, _, _ = money(coins, True)
    print('    без %-5s банк $%6.0f, банк+откат $%6.0f (x%.2f)'
          % (drop.replace('USDT', ''), a, b, b / a), flush=True)
coins = [s for s in ALL if s not in ('AAVEUSDT', 'XLMUSDT')]
a, _, _ = money(coins, False)
b, _, _ = money(coins, True)
print('    без AAVE и XLM:      банк $%6.0f, банк+откат $%6.0f (x%.2f)' % (a, b, b / a), flush=True)
coins = [s for s in ALL if s not in ('AAVEUSDT', 'XLMUSDT', 'DOTUSDT')]
a, _, _ = money(coins, False)
b, _, _ = money(coins, True)
print('    без AAVE, XLM и DOT: банк $%6.0f, банк+откат $%6.0f (x%.2f)' % (a, b, b / a), flush=True)
