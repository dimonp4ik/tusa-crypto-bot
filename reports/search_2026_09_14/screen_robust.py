"""Is "drop the coin with the worst trailing win rate" a rule, or one lucky setting?

It reproduces the hindsight removal of AAVE exactly, without looking forward, which is the kind of
result that has been wrong four times today. A rule survives if it keeps working when its arbitrary
choices are moved: how often it recomputes, how much history it reads, how many trades a coin needs
before it can be judged, and how many coins it drops.

If only one cell of that grid works, it is AAVE found by a longer route.
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

meas = collections.defaultdict(list)
for r in csv.DictReader(open('book_samples.csv')):
    meas[r['coin'] + 'USDT'].append(float(r['cost109']))
SP.SLIP_REAL.update({k: float(np.median(v)) for k, v in meas.items()})
G.FC.clear()

C = [('rsi48', '>=', 56.3761), ('rsi6', '<=', 33.7947)]
ALL = list(G.ALL)
bank = SW.run(3.0, 1.0, coins=ALL, with_meta=True)
cand, _ = G.simulate(C, True, coins=ALL)
EV = [(a, b, R, sf, s, 'bank') for a, b, R, sf, s in bank]
EV += [(a, b, R, sf, s, 'cand') for a, b, R, sf, s in cand]
EV.sort(key=lambda x: (x[0], 0 if x[5] == 'bank' else 1))
first, last = min(x[0] for x in EV), max(x[0] for x in EV)
LO = first + 365 * 86400


def build(drop_at, step):
    busy, out = {}, []
    for a, b, R, sf, s, tag in EV:
        k = first + ((a - first) // step) * step
        if s in drop_at.get(k, ()):
            continue
        if busy.get(s, 0) > a:
            continue
        busy[s] = b
        out.append((a, b, R, sf, s, 1.0))
    out.sort()
    return out


def run(step, lookback, min_n, k_drop):
    drop_at = {}
    t = first
    while t < last:
        lo_h = t - lookback if lookback else 0
        per = collections.defaultdict(list)
        for a, b, R, sf, s, tag in EV:
            if lo_h <= a < t:
                per[s].append(R)
        sc = [(-float(np.mean(np.array(v) > 0)), s) for s, v in per.items() if len(v) >= min_n]
        sc.sort(reverse=True)
        drop_at[t] = {s for _, s in sc[:k_drop]}
        t += step
    rows = build(drop_at, step)
    m = sim_w.money_at_dd([r for r in rows if r[0] >= LO], 0.12)[1]
    who = collections.Counter()
    tot = 0
    for t, ss in drop_at.items():
        if t < LO:
            continue
        tot += 1
        for s in ss:
            who[s] += 1
    return m, who, tot


base_rows = build({}, 182 * 86400)
m0 = sim_w.money_at_dd([r for r in base_rows if r[0] >= LO], 0.12)[1]
print('  опора (все монеты): $%.0f' % m0, flush=True)
print('', flush=True)
STEPS = [(91 * 86400, 'квартал'), (182 * 86400, 'полгода'), (365 * 86400, 'год')]
LOOKS = [(0, 'вся история'), (365 * 86400, 'год назад'), (182 * 86400, 'полгода назад')]
for step, sl in STEPS:
    for look, ll in LOOKS:
        cells = []
        for min_n in (25, 40, 60):
            m, who, tot = run(step, look, min_n, 1)
            cells.append('n>=%d: $%5.0f (%+4.0f%%)' % (min_n, m, 100 * (m / m0 - 1)))
        m, who, tot = run(step, look, 40, 1)
        print('    пересчёт %-8s окно %-14s %s | кого рубит: %s'
              % (sl, ll, '  '.join(cells),
                 ', '.join('%s %d/%d' % (s.replace('USDT', ''), n, tot)
                           for s, n in who.most_common(3))), flush=True)

print('', flush=True)
print('  === сколько монет выбрасывать (пересчёт раз в полгода, вся история, n>=40) ===',
      flush=True)
for k in (0, 1, 2, 3, 4):
    m, who, tot = run(182 * 86400, 0, 40, k)
    print('    k=%d  $%5.0f (%+4.0f%%) | %s'
          % (k, m, 100 * (m / m0 - 1),
             ', '.join('%s %d/%d' % (s.replace('USDT', ''), n, tot) for s, n in who.most_common(5))),
          flush=True)

print('', flush=True)
print('  === то же правило, но выбрасывать ЛУЧШУЮ монету (должно терять) ===', flush=True)


def run_best(step, k_drop):
    drop_at = {}
    t = first
    while t < last:
        per = collections.defaultdict(list)
        for a, b, R, sf, s, tag in EV:
            if a < t:
                per[s].append(R)
        sc = [(float(np.mean(np.array(v) > 0)), s) for s, v in per.items() if len(v) >= 40]
        sc.sort(reverse=True)
        drop_at[t] = {s for _, s in sc[:k_drop]}
        t += step
    rows = build(drop_at, step)
    return sim_w.money_at_dd([r for r in rows if r[0] >= LO], 0.12)[1]


for k in (1, 2):
    m = run_best(182 * 86400, k)
    print('    выбросить %d ЛУЧШИХ: $%5.0f (%+4.0f%%)' % (k, m, 100 * (m / m0 - 1)), flush=True)
