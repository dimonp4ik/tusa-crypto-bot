"""What the model's $2,734 becomes once the things it does not model are taken off.

Three known haircuts, applied one at a time and then together:

  заливка   the venue fills 84-86% of signals; the model fills 100%. Which signals miss is not
            predictable, so it is drawn at random with several seeds and reported as a range -
            refusals change the path of the equity curve and a single seed means nothing.
  минимум   at $120 the position can fall below the exchange minimum contract size, so early
            trades are skipped. margin_cap already models this; here it is isolated.
  начало    starting later - the deposit compounds, so the result depends heavily on which years
            are included. Each year measured as its own account says how much.

None of this covers the biggest unknown, which is whether the next four years look like the last
four.
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
import live_rules_sim as L
import nogate_attack as NA

cost = collections.defaultdict(list)
for r in csv.DictReader(open('book_frozen.csv')):
    cost[r['coin'] + 'USDT'].append(float(r['cost109']))
SP.SLIP_REAL.update({k: float(np.median(v)) for k, v in cost.items()})
G.FC.clear()
NOBAD = [s for s in G.ALL if s not in ('AAVEUSDT', 'XLMUSDT')]
C = [('rsi48', '>=', 56.3761), ('rsi6', '<=', 33.7947)]
RISK = 0.014
YT = {y: int(datetime.datetime(y, 1, 1, tzinfo=datetime.UTC).timestamp()) for y in range(2022, 2028)}

bank = SW.run(3.0, 1.0, coins=NOBAD, with_meta=True)
cand = NA.trades('без режима', C, coins=NOBAD)
ev = [(x[0], x[1], x[2], x[3], x[4], 0) for x in bank] + \
     [(x[0], x[1], x[2], x[3], x[4], 1) for x in cand]
ev.sort(key=lambda z: (z[0], z[5]))


def build(fill=1.0, seed=1):
    rng = np.random.default_rng(seed)
    busy, out = {}, []
    for p, q, R, sf, s, k in ev:
        if busy.get(s, 0) > p:
            continue
        if fill < 1.0 and rng.random() > fill:
            continue                      # the venue did not fill this one
        busy[s] = q
        out.append((p, q, R, sf, s, 1.0))
    out.sort()
    return out


full = build()
r = sim_w.simulate(full, RISK)
print('  модель как есть: %d сделок, $%.0f, просадка %.1f%%'
      % (len(full), r['eq'], 100 * abs(L.dd_of(r['curve']))), flush=True)
print('  из них пропущено по марже (позиция меньше минимума биржи): %d' % r['skipped_margin'],
      flush=True)
print('', flush=True)
print('  === заливка 85%%, десять розыгрышей ===', flush=True)
res = []
for seed in range(1, 11):
    rows = build(0.85, seed)
    rr = sim_w.simulate(rows, RISK)
    res.append((rr['eq'], abs(L.dd_of(rr['curve'])), len(rows), bool(rr['paused_at'])))
for i, (eq, dd, n, lat) in enumerate(res, 1):
    print('    розыгрыш %2d: %4d сделок  $%6.0f  просадка %5.1f%%  %s'
          % (i, n, eq, 100 * dd, 'ЗАЩЁЛКА' if lat else ''), flush=True)
eqs = np.array([x[0] for x in res])
print('    итог: медиана $%.0f, от $%.0f до $%.0f, защёлок %d из 10'
      % (np.median(eqs), eqs.min(), eqs.max(), sum(1 for x in res if x[3])), flush=True)

print('', flush=True)
print('  === каждый год как отдельный счёт с $120 (риск 1.40%%) ===', flush=True)
for y in range(2022, 2027):
    sub = [x for x in full if YT[y] <= x[0] < YT[y + 1]]
    if len(sub) < 50:
        continue
    rr = sim_w.simulate(sub, RISK)
    v = np.array([x[2] for x in sub])
    print('    %d  %4d сделок ВР%5.1f%%  $120 -> $%5.0f (x%.1f)  просадка %5.1f%%'
          % (y, len(sub), 100 * np.mean(v > 0), rr['eq'], rr['eq'] / 120,
             100 * abs(L.dd_of(rr['curve']))), flush=True)

print('', flush=True)
print('  === и то же самое, если начать позже ===', flush=True)
for start in (2022, 2023, 2024, 2025):
    sub = [x for x in full if x[0] >= YT[start]]
    rr = sim_w.simulate(sub, RISK)
    months = (sub[-1][1] - sub[0][0]) / (365.25 * 86400 / 12)
    print('    старт %d: %4d сделок, $120 -> $%6.0f за %.1f мес (%+.2f%%/мес), просадка %5.1f%% %s'
          % (start, len(sub), rr['eq'], months,
             100 * ((rr['eq'] / 120) ** (1 / months) - 1),
             100 * abs(L.dd_of(rr['curve'])), 'ЗАЩЁЛКА' if rr['paused_at'] else ''), flush=True)
