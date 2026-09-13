"""The five that passed, as money on top of what already stands.

Per-trade edge is not the test. Capitulation had +0.1401R and made the account poorer, because a new
entry costs a slot and arrives beside the bank's own losses. Three of these five are SHORTS that the
bank never takes, and the account is almost entirely long, so if diversification is worth anything
in this system it will show up here as drawdown rather than as return.

Equal drawdown of 12%, added one at a time and then together, with the walk-forward on both axes.
"""
import datetime
import itertools
import pickle
import sys

import numpy as np

sys.path.insert(0, 'C:/Users/Lenovo/Desktop/Торговля/crypto-bot')
import gauntlet2 as G
import struct_params as SP
import stop_width as SW
import sim_w

PASS = pickle.load(open('gauntlet_pass.pkl', 'rb'))
SHORT = {lbl: lbl.split()[0] + ' ' + lbl.split()[1].split('>')[0].split('<')[0] for lbl, _, _ in PASS}
bank = SW.run(3.0, 1.0, coins=G.ALL, with_meta=True)
S = {'ОТКАТ': G.simulate([('rsi48', '>=', 56.3761), ('rsi6', '<=', 33.7947)], True)[0]}
NAMES = ['ОТКАТ']
for lbl, side, conds in PASS:
    S[lbl] = G.simulate(conds, side)[0]
    NAMES.append(lbl)
    print('  %-46s %d сделок' % (lbl, len(S[lbl])), flush=True)


def merge(names):
    ev = [(a, b, R, sf, s, 'bank') for a, b, R, sf, s in bank]
    for nm in names:
        ev += [(a, b, R, sf, s, nm) for a, b, R, sf, s in S[nm]]
    pri = {'bank': 0}
    pri.update({nm: i + 1 for i, nm in enumerate(names)})
    ev.sort(key=lambda x: (x[0], pri[x[5]]))
    busy, out = {}, []
    for a, b, R, sf, s, tag in ev:
        if busy.get(s, 0) > a:
            continue
        busy[s] = b
        out.append((a, b, R, sf, s, tag))
    out.sort()
    return out


def money(rows, want=0.12):
    return sim_w.money_at_dd([(a, b, R, sf, s, 1.0) for a, b, R, sf, s, _ in rows], want)


print('', flush=True)
base = merge([])
k0, m0 = money(base)
print('  банк один: %d сделок, риск %.3f%%, $%.0f' % (len(base), 100 * k0, m0), flush=True)
pb = merge(['ОТКАТ'])
k1, m1 = money(pb)
print('  банк+откат: %d сделок, риск %.3f%%, $%.0f (+%.0f%%)'
      % (len(pb), 100 * k1, m1, 100 * (m1 / m0 - 1)), flush=True)
print('', flush=True)
print('  === каждое поверх банка+отката ===', flush=True)
solo = {}
for lbl, _, _ in PASS:
    rows = merge(['ОТКАТ', lbl])
    k, m = money(rows)
    solo[lbl] = m
    print('    %-46s %4d сделок риск %.3f%% $%6.0f (%+.0f%% к банку+откату)'
          % (lbl, len(rows), 100 * k, m, 100 * (m / m1 - 1)), flush=True)

order = sorted(solo, key=lambda x: -solo[x])
print('', flush=True)
print('  === нарастающим итогом, от лучшего ===', flush=True)
cur = ['ОТКАТ']
best_rows = pb
for lbl in order:
    cur.append(lbl)
    rows = merge(cur)
    k, m = money(rows)
    print('    +%-45s %4d сделок риск %.3f%% $%6.0f (%+.0f%% к банку)'
          % (lbl, len(rows), 100 * k, m, 100 * (m / m0 - 1)), flush=True)
    best_rows = rows

print('', flush=True)
print('  === скользящий прогон полного набора против банка ===', flush=True)
b3 = [(a, b, R) for a, b, R, sf, s, t in base]
c3 = [(a, b, R) for a, b, R, sf, s, t in best_rows]
first, last = min(r[0] for r in b3), max(r[0] for r in b3)
t, wr, wd, seen = first, 0, 0, 0
while t < last:
    a, b = t, t + 182 * 86400
    s0, s1 = SP.stats(b3, lo=a, hi=b), SP.stats(c3, lo=a, hi=b)
    if s0 and s1:
        seen += 1
        wr += int(s1['r_mo'] > s0['r_mo'])
        wd += int(s1['dd'] > s0['dd'])
        print('    %s  R/мес банк%+6.2f набор%+6.2f | просадка банк%+6.1f набор%+6.1f'
              % (datetime.datetime.fromtimestamp(a, datetime.UTC).strftime('%y-%m'),
                 s0['r_mo'], s1['r_mo'], s0['dd'], s1['dd']), flush=True)
    t = b
print('    набор лучше: R/мес %d/%d, просадка %d/%d' % (wr, seen, wd, seen), flush=True)

print('', flush=True)
print('  === сколько шортов в наборе и что они делают с просадкой ===', flush=True)
tags = {}
for a, b, R, sf, s, tg in best_rows:
    tags[tg] = tags.get(tg, 0) + 1
print('    состав: %s' % ' | '.join('%s:%d' % kv for kv in sorted(tags.items(), key=lambda kv: -kv[1])),
      flush=True)
for want in (0.08, 0.10, 0.12, 0.15):
    kb, mb = money(base, want)
    kc, mc = money(best_rows, want)
    print('    при просадке %2.0f%%: банк риск %.3f%% $%6.0f | набор риск %.3f%% $%6.0f (x%.2f)'
          % (100 * want, 100 * kb, mb, 100 * kc, mc, mc / mb), flush=True)
