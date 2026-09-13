"""The system built from thresholds that survived not seeing the year they were judged on.

The search picked its numbers by maximising over all five years. The leave-one-year-out re-pick
picks them five more times, each time blind to one year, and the median of those five is a number
no single year can have chosen. Where the two disagree, the consensus is the one to trust - and they
do disagree: the search wanted vol720 >= 1.4462, every honest re-pick wanted 1.67 or more.

Two rules survived that test in every held-out year. This measures the account built from those two
and nothing else, against the search's own picks, at equal drawdown.
"""
import collections
import datetime
import sys

import numpy as np

sys.path.insert(0, 'C:/Users/Lenovo/Desktop/Торговля/crypto-bot')
import gauntlet2 as G
import struct_params as SP
import stop_width as SW
import margin_cap as MC
import live_rules_sim as L

SETS = {
    'ОТКАТ-поиск': ([('rsi48', '>=', 56.3761), ('rsi6', '<=', 33.7947)], True),
    'ОТКАТ-согл': ([('rsi48', '>=', 57.1155), ('rsi6', '<=', 31.6895)], True),
    'КАПИТ-поиск': ([('breadth', '<=', 0.0625), ('vol720', '>=', 1.4462)], True),
    'КАПИТ-согл': ([('breadth', '<=', 0.0625), ('vol720', '>=', 1.6710)], True),
    'ПАДЕНИЕ-согл': ([('ret24', '<=', -1.7963), ('vol720', '>=', 1.8966)], True),
    'РАСХОЖД-согл': ([('btc168', '<=', -0.0734), ('pos48', '>=', 0.8794)], False),
}
S = {}
for nm, (c, lg) in SETS.items():
    S[nm] = G.describe(nm, c, lg)

print('', flush=True)
print('  === пересечения ===', flush=True)
K = {nm: {(r[0], r[4]) for r in S[nm]} for nm in S}
bank = SW.run(3.0, 1.0, coins=G.ALL, with_meta=True)
K['БАНК'] = {(a, s) for a, b, R, sf, s in bank}
for a in ('ОТКАТ-согл', 'КАПИТ-согл'):
    for b in ('БАНК', 'ОТКАТ-согл', 'КАПИТ-согл'):
        if a < b:
            print('    %-13s x %-13s %3d' % (a, b, len(K[a] & K[b])), flush=True)


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


def money_at_dd(rows, want=0.12):
    MC.trades = [(a, b, R, sf, s) for a, b, R, sf, s, _ in rows]
    lo, hi = 0.002, 0.030
    for _ in range(26):
        mid = (lo + hi) / 2
        r = L.simulate(target=mid)
        if bool(r['paused_at']) or abs(L.dd_of(r['curve'])) > want:
            hi = mid
        else:
            lo = mid
    r = L.simulate(target=lo)
    return lo, r['eq'], abs(L.dd_of(r['curve']))


VAR = [('банк', []),
       ('+откат-поиск', ['ОТКАТ-поиск']),
       ('+откат-согл', ['ОТКАТ-согл']),
       ('+капит-согл', ['КАПИТ-согл']),
       ('откат+капит согл', ['ОТКАТ-согл', 'КАПИТ-согл']),
       ('они же по поиску', ['ОТКАТ-поиск', 'КАПИТ-поиск']),
       ('+падение,+расх', ['ОТКАТ-согл', 'КАПИТ-согл', 'ПАДЕНИЕ-согл', 'РАСХОЖД-согл'])]

print('', flush=True)
print('  === при равной просадке 12%% ===', flush=True)
KEEP = {}
for nm, names in VAR:
    rows = merge(names)
    KEEP[nm] = rows
    k, eq, dd = money_at_dd(rows)
    v = np.array([r[2] for r in rows])
    print('    %-18s сделок %4d ср%+.4fR  риск %.3f%%  $%7.0f  просадка %5.1f%%'
          % (nm, len(rows), v.mean(), 100 * k, eq, 100 * dd), flush=True)

print('', flush=True)
print('  === скользящий прогон против банка (9 полугодий) ===', flush=True)
b3 = [(a, b, R) for a, b, R, sf, s, t in KEEP['банк']]
first, last = min(r[0] for r in b3), max(r[0] for r in b3)
for nm, _ in VAR[1:]:
    c3 = [(a, b, R) for a, b, R, sf, s, t in KEEP[nm]]
    t, wr, wd, seen = first, 0, 0, 0
    while t < last:
        a, b = t, t + 182 * 86400
        s0, s1 = SP.stats(b3, lo=a, hi=b), SP.stats(c3, lo=a, hi=b)
        if s0 and s1:
            seen += 1
            wr += int(s1['r_mo'] > s0['r_mo'])
            wd += int(s1['dd'] > s0['dd'])
        t = b
    print('    %-18s R/мес %d/%d  просадка %d/%d' % (nm, wr, seen, wd, seen), flush=True)

print('', flush=True)
print('  === помесячный разброс связки откат+капитуляция ===', flush=True)
rows = KEEP['откат+капит согл']
mon = collections.Counter(datetime.datetime.fromtimestamp(r[0], datetime.UTC).strftime('%y-%m')
                          for r in rows if r[5] != 'bank')
n = sum(mon.values())
print('    новых сделок %d в %d месяцах, максимум в одном месяце %d (%.0f%%)'
      % (n, len(mon), max(mon.values()), 100 * max(mon.values()) / n), flush=True)
