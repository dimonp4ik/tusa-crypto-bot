"""Was the coin screen's +13% just the warm-up year being excluded?

The screen earned its place by reproducing the hindsight removal of AAVE without lookahead. In the
full assembly it loses money and wins one half-year of nine. The difference between the two runs is
a year of warm-up: the earlier comparison threw away the first year, during which the screen ranks
coins on too little history and cuts more or less at random.

If that is the whole story, the screen is not wrong - it just cannot be used before it has data, and
the honest version of it waits. If the screen loses even with the warm-up, the +13% was an artefact
of what the comparison excluded.
"""
import collections
import datetime
import sys

import numpy as np

sys.path.insert(0, 'C:/Users/Lenovo/Desktop/Торговля/crypto-bot')
import assemble as A
import sim_w

YT = A.YT


def money_from(rows, lo):
    sub = [r for r in rows if r[0] >= lo]
    return sim_w.money_at_dd(sub, 0.12)[1], len(sub)


base_all = A.build()
first = min(r[0] for r in base_all)
print('  === один и тот же отбор, при разном разогреве ===', flush=True)
print('  разогрев   банк один      + отбор монеты     разница', flush=True)
for warm_days in (0, 182, 365, 547, 730):
    lo = first + warm_days * 86400
    m0, n0 = money_from(base_all, lo)
    m1, n1 = money_from(A.build(screen=True), lo)
    print('    %4d дн  $%6.0f (%4d)  $%6.0f (%4d)   %+5.0f%%'
          % (warm_days, m0, n0, m1, n1, 100 * (m1 / m0 - 1)), flush=True)

print('', flush=True)
print('  === и то же самое, но отбор НЕ РАБОТАЕТ пока не набрано n сделок на монету ===',
      flush=True)


def build_delayed(min_n, warm_days):
    """The screen only cuts once every surviving coin has at least min_n trades behind it."""
    ev = []
    for s in A.ALL:
        ev += [(a, b, R, sf, s, 0) for a, b, R, sf, s2 in A.BANK[s]]
    ev.sort()
    firstt = min(x[0] for x in ev)
    drop, t = {}, firstt
    last = max(x[0] for x in ev)
    while t < last:
        per = collections.defaultdict(list)
        for a, b, R, sf, s, k in ev:
            if a < t:
                per[s].append(R)
        ready = [s for s in per if len(per[s]) >= min_n]
        if len(ready) >= len(A.ALL) - 2:
            sc = sorted(((-float(np.mean(np.array(per[s]) > 0)), s) for s in ready), reverse=True)
            drop[t] = {sc[0][1]}
        else:
            drop[t] = set()
        t += A.SLICE
    busy, out = {}, []
    for a, b, R, sf, s, k in ev:
        key = firstt + ((a - firstt) // A.SLICE) * A.SLICE
        if s in drop.get(key, ()):
            continue
        if busy.get(s, 0) > a:
            continue
        busy[s] = b
        out.append((a, b, R, sf, s, 1.0))
    out.sort()
    return out, drop


for min_n in (40, 80, 120):
    rows, drop = build_delayed(min_n, 0)
    m0, _ = money_from(base_all, 0)
    m1, n1 = money_from(rows, 0)
    who = collections.Counter()
    for t, ss in drop.items():
        for s in ss:
            who[s] += 1
    print('    порог %3d сделок: $%6.0f против $%6.0f (%+5.0f%%) | рубил: %s'
          % (min_n, m1, m0, 100 * (m1 / m0 - 1),
             ', '.join('%s x%d' % (s.replace('USDT', ''), n) for s, n in who.most_common(3))
             or 'никого'), flush=True)

print('', flush=True)
print('  === кого отбор рубит в каждом полугодии (порог 40) ===', flush=True)
rows, drop = build_delayed(40, 0)
for t in sorted(drop):
    ss = drop[t]
    print('    %s  %s' % (datetime.datetime.fromtimestamp(t, datetime.UTC).strftime('%y-%m'),
                          ', '.join(s.replace('USDT', '') for s in ss) or '— (мало данных)'),
          flush=True)
