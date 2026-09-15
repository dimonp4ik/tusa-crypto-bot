"""Is a wider reference stop better, or better only where it was found?

Turning the shrink off costs more than half the money, so the guard is real. Its level is another
matter: 0.0394 is what runs live, and 0.05 to 0.08 returns 27% more at equal drawdown. That is the
shape of a finding that has died twice today for lack of a second book, so it gets four.

What the reference does, mechanically: it sets the base position size and shrinks any trade whose own
stop is wider than it. Raising it means shrinking fewer trades, which raises risk - and the
equal-drawdown search then lowers the base risk to compensate. Whether that trade is worth taking is
what the money says.
"""
import collections, csv, datetime, sys
import numpy as np
sys.path.insert(0, 'C:/Users/Lenovo/Desktop/Торговля/crypto-bot')
import struct_params as SP
import gauntlet2 as G
import live_rules_sim as L
import exec_model as EM
import guards as GD

YT = EM.YT
BOOKS = [
    ('без AAVE и XLM, банк+откат', EM.book(EM.NOBAD, EM.SIG_ALL, 'рынок')[0]),
    ('все 15, банк+откат',          EM.book(EM.ALL,   EM.SIG_ALL, 'рынок')[0]),
    ('без AAVE и XLM, банк один',   EM.book(EM.NOBAD, EM.SIG_BANK, 'рынок')[0]),
    ('все 15, банк один',           EM.book(EM.ALL,   EM.SIG_BANK, 'рынок')[0]),
]
REFS = [0.03, 0.0394, 0.045, 0.05, 0.06, 0.08]

def at_dd(trades, want=0.12, **kw):
    lo, hi = 0.0005, 0.060
    for _ in range(26):
        mid = (lo + hi) / 2
        r = GD.sim(trades, mid, **kw)
        if bool(r['paused_at']) or abs(L.dd_of(r['curve'])) > want:
            hi = mid
        else:
            lo = mid
    return lo, GD.sim(trades, lo, **kw)

print('  опорный стоп: %s' % ' '.join('%7.4f' % r for r in REFS), flush=True)
rows = []
for lbl, tr in BOOKS:
    cells = []
    for ref in REFS:
        k, r = at_dd(tr, stop_ref=ref)
        cells.append(r['eq'])
    best = max(range(len(REFS)), key=lambda i: cells[i])
    print('  %-28s %s | лучший %.4f' % (lbl, ' '.join('%7.0f%s' % (c, '*' if i == best else ' ')
                                                       for i, c in enumerate(cells)), REFS[best]),
          flush=True)
    rows.append(cells)

print('', flush=True)
print('  === место каждого значения по четырём книгам ===', flush=True)
rank = collections.defaultdict(list); wins = collections.Counter()
for cells in rows:
    order = sorted(range(len(REFS)), key=lambda i: -cells[i])
    wins[REFS[order[0]]] += 1
    for place, i in enumerate(order):
        rank[REFS[i]].append(place + 1)
for r in REFS:
    print('    %.4f  лучший %d раз, среднее место %.1f, худшее %d'
          % (r, wins[r], float(np.mean(rank[r])), max(rank[r])), flush=True)

print('', flush=True)
print('  === и по годам, для 0.0394 против лучшего ===', flush=True)
tr = BOOKS[0][1]
for ref in (0.0394, 0.05, 0.06):
    per = []
    for y in range(2022, 2027):
        sub = [x for x in tr if YT[y] <= x[0] < YT[y+1]]
        if len(sub) < 50:
            per.append(float('nan')); continue
        k, r = at_dd(sub, stop_ref=ref)
        per.append(r['eq'])
    print('    %.4f: %s' % (ref, ' '.join('%4.0f' % x if np.isfinite(x) else '   -' for x in per)),
          flush=True)
