"""Can a coin be dropped before it costs money, rather than after?

AAVE was found by looking at all five years, which is the procedure this session has spent the day
distrusting. But it left a fingerprint that does not need the answer to compute: 21.1% of its trades
end within 0.3R of flat against 7.3% for the rest. A coin that does not travel inside the holding
window is measurable from its own past.

So the rule is made causal. At each half-year boundary, every coin is ranked on what its trades did
BEFORE that date - stuck share, win rate, mean R - the worst are dropped, and the next half-year is
traded without them. Nothing looks forward. The comparison is against holding all coins and against
dropping AAVE by hindsight, which is the best any rule of this kind could do.

Fill costs are the measured ones throughout.
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
SLICE = 182 * 86400
WARM = 365 * 86400


def build(drop_at):
    """drop_at: ts -> set of coins banned from that slice onward. One position per coin."""
    busy, out = {}, []
    for a, b, R, sf, s, tag in EV:
        k = first + ((a - first) // SLICE) * SLICE
        if s in drop_at.get(k, ()):  # coin is out for this slice
            continue
        if busy.get(s, 0) > a:
            continue
        busy[s] = b
        out.append((a, b, R, sf, s, 1.0))
    out.sort()
    return out


def screen(metric, k_drop, min_n=40):
    """At each boundary, rank coins on data strictly BEFORE it and drop the worst k."""
    drop_at = {}
    t = first
    while t < last:
        hist = [x for x in EV if x[0] < t]
        per = collections.defaultdict(list)
        for a, b, R, sf, s, tag in hist:
            per[s].append(R)
        scored = []
        for s, v in per.items():
            if len(v) < min_n:
                continue
            v = np.array(v)
            if metric == 'зависшие':
                scored.append((float(np.mean(np.abs(v) < 0.3)), s))
            elif metric == 'винрейт':
                scored.append((-float(np.mean(v > 0)), s))
            elif metric == 'средний R':
                scored.append((-float(v.mean()), s))
            elif metric == 'полные стопы':
                scored.append((float(np.mean(v <= -0.9)), s))
        scored.sort(reverse=True)
        drop_at[t] = {s for _, s in scored[:k_drop]}
        t += SLICE
    return drop_at


def money(rows, lo=None):
    sub = [r for r in rows if lo is None or r[0] >= lo] if lo else rows
    return sim_w.money_at_dd(sub, 0.12)[1]


LO = first + WARM
base = build({})
m0 = money(base, LO)
hind = build({t: {'AAVEUSDT'} for t in
              [first + i * SLICE for i in range(20)]})
mh = money(hind, LO)
print('  опора (все монеты, после года разогрева): $%.0f' % m0, flush=True)
print('  задним числом без AAVE: $%.0f (%+.0f%%)' % (mh, 100 * (mh / m0 - 1)), flush=True)
print('', flush=True)
print('  === причинный отбор: ранг по прошлому, выбор до начала полугодия ===', flush=True)
for metric in ('зависшие', 'винрейт', 'средний R', 'полные стопы'):
    line = []
    for k in (1, 2, 3):
        d = screen(metric, k)
        m = money(build(d), LO)
        picked = collections.Counter()
        for t, ss in d.items():
            for s in ss:
                picked[s] += 1
        line.append('k=%d $%5.0f (%+4.0f%%)' % (k, m, 100 * (m / m0 - 1)))
    d1 = screen(metric, 1)
    who = collections.Counter()
    for t, ss in sorted(d1.items()):
        for s in ss:
            who[s] += 1
    print('    %-14s %s | кого выбрасывал при k=1: %s'
          % (metric, '  '.join(line),
             ', '.join('%s x%d' % (s.replace('USDT', ''), n) for s, n in who.most_common(4))),
          flush=True)

print('', flush=True)
print('  === сколько раз AAVE попадала в выброшенные, по полугодиям ===', flush=True)
for metric in ('зависшие', 'винрейт', 'средний R'):
    d = screen(metric, 2)
    seq = []
    for t in sorted(d):
        if t < LO:
            continue
        seq.append('%s%s' % (datetime.datetime.fromtimestamp(t, datetime.UTC).strftime('%y-%m'),
                             '+' if 'AAVEUSDT' in d[t] else '-'))
    print('    %-12s %s' % (metric, ' '.join(seq)), flush=True)
