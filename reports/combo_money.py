"""Four rules competing for the same slots, as one account.

Each family looks good alone, but they cannot all be taken: one position per coin, and the bank has
first claim on any hour it wants. Two of the three share vol720, so they may be the same trades
wearing different names. This measures what actually reaches the account.

The null here is the pool itself: every hour the regime gate allows, taken with the same geometry
and the same costs. That is what a rule has to beat to have said anything.
"""
import datetime
import itertools
import sys

import numpy as np

sys.path.insert(0, 'C:/Users/Lenovo/Desktop/Торговля/crypto-bot')
import gauntlet2 as G
import struct_params as SP
import stop_width as SW
import margin_cap as MC
import live_rules_sim as L

RULES = [
    ('ОТКАТ', [('rsi48', '>=', 56.3761), ('rsi6', '<=', 33.7947)], True),
    ('ПАДЕНИЕ', [('ret24', '<=', -1.5434), ('vol720', '>=', 1.4462)], True),
    ('КАПИТУЛЯЦИЯ', [('breadth', '<=', 0.0625), ('vol720', '>=', 1.4462)], True),
    ('РАСХОЖДЕНИЕ', [('btc168', '<=', -0.0558), ('pos48', '>=', 0.8794)], False),
]

S = {}
for nm, conds, lg in RULES:
    S[nm], _ = G.simulate(conds, lg)
    print('  %-12s %d сделок' % (nm, len(S[nm])), flush=True)

print('', flush=True)
print('  === нулевая модель: весь разрешённый гейтом поток той же геометрией ===', flush=True)
for side, lab in ((True, 'LONG'), (False, 'SHORT')):
    rows, hours = G.simulate([], side)
    v = np.array([r[2] for r in rows])
    print('    %-5s n%5d ВР%5.1f%% ср%+.4fR  <- это и есть ноль' % (lab, len(v), 100 * np.mean(v > 0), v.mean()),
          flush=True)

print('', flush=True)
print('  === пересечение семейств между собой (сделок общих) ===', flush=True)
K = {nm: {(r[0], r[4]) for r in S[nm]} for nm in S}
for a, b in itertools.combinations(K, 2):
    print('    %-12s x %-12s %3d' % (a, b, len(K[a] & K[b])), flush=True)

bank = SW.run(3.0, 1.0, coins=G.ALL, with_meta=True)


def merge(names):
    """Bank first claim, then the named rules in order; one position per coin at a time."""
    ev = [(a, b, R, sf, s, 'bank') for a, b, R, sf, s in bank]
    for k, nm in enumerate(names):
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


VAR = [('банк', []),
       ('+откат', ['ОТКАТ']),
       ('+падение', ['ПАДЕНИЕ']),
       ('+капитуляция', ['КАПИТУЛЯЦИЯ']),
       ('+расхождение', ['РАСХОЖДЕНИЕ']),
       ('откат+падение', ['ОТКАТ', 'ПАДЕНИЕ']),
       ('откат+расхожд', ['ОТКАТ', 'РАСХОЖДЕНИЕ']),
       ('все четыре', ['ОТКАТ', 'ПАДЕНИЕ', 'КАПИТУЛЯЦИЯ', 'РАСХОЖДЕНИЕ'])]

print('', flush=True)
print('  вариант         сделок  ср R      1.40%%: $ / просадка    1.55%%: $ / просадка', flush=True)
KEEP = {}
for nm, names in VAR:
    rows = merge(names)
    KEEP[nm] = rows
    v = np.array([r[2] for r in rows])
    MC.trades = [(a, b, R, sf, s) for a, b, R, sf, s, _ in rows]
    out = []
    for risk in (0.014, 0.0155):
        r = L.simulate(target=risk)
        out.append('$%7.0f /%5.1f%% %s' % (r['eq'], 100 * L.dd_of(r['curve']),
                                           'ЗАЩ' if r['paused_at'] else '   '))
    print('  %-15s %5d  %+.4f   %s   %s' % (nm, len(v), v.mean(), out[0], out[1]), flush=True)


def month(x):
    return datetime.datetime.fromtimestamp(int(x), datetime.UTC).strftime('%y-%m')


print('', flush=True)
print('  === скользящий прогон: каждое полугодие против БАНКА ===', flush=True)
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
    print('    %-15s R/мес %d/%d   просадка %d/%d' % (nm, wr, seen, wd, seen), flush=True)
