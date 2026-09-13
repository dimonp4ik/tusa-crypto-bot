"""The one episode that sets the ceiling.

The account latches at a 15% drawdown and never resumes, so the deepest single pit decides the
largest risk the account may carry - and therefore all the money. Everything else in the five years
is irrelevant to that ceiling. Worth knowing exactly what is in the pit: when, which rules, which
coins, and whether the losses are simultaneous or merely consecutive.

Comparison is at equal drawdown rather than equal risk: for each variant, the risk that produces a
12% worst drawdown, and the money it makes there. That removes the cliff from the comparison
entirely.
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

RULES = {
    'ОТКАТ': ([('rsi48', '>=', 56.3761), ('rsi6', '<=', 33.7947)], True),
    'ПАДЕНИЕ': ([('ret24', '<=', -1.5434), ('vol720', '>=', 1.4462)], True),
    'КАПИТУЛЯЦИЯ': ([('breadth', '<=', 0.0625), ('vol720', '>=', 1.4462)], True),
    'РАСХОЖДЕНИЕ': ([('btc168', '<=', -0.0558), ('pos48', '>=', 0.8794)], False),
}
S = {nm: G.simulate(c, lg)[0] for nm, (c, lg) in RULES.items()}
bank = SW.run(3.0, 1.0, coins=G.ALL, with_meta=True)


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


VAR = [('банк', []), ('+откат', ['ОТКАТ']),
       ('откат+расхожд', ['ОТКАТ', 'РАСХОЖДЕНИЕ']),
       ('три (без кап.)', ['ОТКАТ', 'ПАДЕНИЕ', 'РАСХОЖДЕНИЕ']),
       ('все четыре', ['ОТКАТ', 'ПАДЕНИЕ', 'КАПИТУЛЯЦИЯ', 'РАСХОЖДЕНИЕ'])]
KEEP = {nm: merge(n) for nm, n in VAR}


def at_risk(rows, risk):
    MC.trades = [(a, b, R, sf, s) for a, b, R, sf, s, _ in rows]
    r = L.simulate(target=risk)
    return r['eq'], L.dd_of(r['curve']), bool(r['paused_at'])


print('  === при РАВНОЙ просадке 12%%: какой риск и сколько денег ===', flush=True)
for nm, _ in VAR:
    rows = KEEP[nm]
    lo, hi = 0.004, 0.030
    for _ in range(24):
        mid = (lo + hi) / 2
        eq, dd, lat = at_risk(rows, mid)
        if lat or dd > 0.12:
            hi = mid
        else:
            lo = mid
    eq, dd, lat = at_risk(rows, lo)
    print('    %-15s риск %.3f%%  $%7.0f  просадка %5.1f%%  сделок %d'
          % (nm, 100 * lo, eq, 100 * dd, len(rows)), flush=True)

print('', flush=True)
print('  === самая глубокая яма при 1.35%% ===', flush=True)
for nm, _ in VAR:
    rows = KEEP[nm]
    MC.trades = [(a, b, R, sf, s) for a, b, R, sf, s, _ in rows]
    r = L.simulate(target=0.0135)
    cur = np.array([e for t, e in r['curve']], dtype=float)
    tsx = np.array([t for t, e in r['curve']], dtype=np.int64)
    peak = np.maximum.accumulate(cur)
    dd = cur / peak - 1
    j = int(np.argmin(dd))
    i = int(np.flatnonzero(cur[:j + 1] == peak[j])[-1])
    f = lambda k: datetime.datetime.fromtimestamp(int(tsx[k]), datetime.UTC).strftime('%y-%m-%d')
    seg = [x for x in rows if tsx[i] <= x[0] <= tsx[j]]
    byt = collections.Counter(x[5] for x in seg)
    bys = collections.Counter(x[4] for x in seg if x[2] < 0)
    losses = [x for x in seg if x[2] < 0]
    print('    %-15s %5.1f%%  %s..%s (%2d дн)  сделок %3d, убыточных %3d  правила %s'
          % (nm, 100 * dd[j], f(i), f(j), (tsx[j] - tsx[i]) // 86400, len(seg), len(losses),
             ' '.join('%s:%d' % kv for kv in byt.most_common())), flush=True)
    if nm == 'все четыре':
        print('      монеты в минусе: %s'
              % ' '.join('%s:%d' % (s.replace('USDT', ''), c) for s, c in bys.most_common(8)),
              flush=True)
        st = sorted(losses, key=lambda x: x[0])
        print('      первый убыток %s, последний %s'
              % (datetime.datetime.fromtimestamp(st[0][0], datetime.UTC).strftime('%y-%m-%d %H:%M'),
                 datetime.datetime.fromtimestamp(st[-1][0], datetime.UTC).strftime('%y-%m-%d %H:%M')),
              flush=True)
        open_at = []
        for x in st:
            k = sum(1 for y in seg if y[0] <= x[0] < y[1])
            open_at.append(k)
        print('      одновременно открытых позиций в момент убытка: медиана %.0f, максимум %d'
              % (float(np.median(open_at)), max(open_at)), flush=True)

print('', flush=True)
print('  === скользящий прогон каждого варианта против банка ===', flush=True)
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
    print('    %-15s R/мес %d/%d  просадка %d/%d' % (nm, wr, seen, wd, seen), flush=True)
