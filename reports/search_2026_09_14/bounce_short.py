"""The pullback rule's conceptual mirror, which has never been tested.

Every "mirror" test so far took the SAME conditions and flipped the side - rsi48 high and rsi6 low,
sold short. That is a control, not a hypothesis. The hypothesis is the reflection of the mechanism:
if a dip inside medium-term strength is worth buying, then a bounce inside medium-term weakness
should be worth selling.

    rsi48 <= X   (weak over two days)   AND   rsi6 >= Y   (bouncing right now)   ->  SHORT

If this works, the account gets a short-side rule of its own kind rather than a borrowed one - and
this project has established that shorts do not diversify, so it would have to earn its place on
money alone, not on balance.

Grid first, then the blind year, then the attacks. Gateless, since the gate costs the long version
money and the same question applies here.
"""
import collections
import datetime
import sys

import numpy as np

sys.path.insert(0, 'C:/Users/Lenovo/Desktop/Торговля/crypto-bot')
import struct_params as SP
import gauntlet2 as G
import stop_width as SW
import sim_w
import nogate_attack as NA
import full_money as FM

COINS = FM.COINS
YT = FM.YT
A_GRID = [30.0, 34.0, 38.0, 42.0, 46.0, 50.0]
B_GRID = [55.0, 60.0, 65.0, 70.0, 75.0, 80.0]


def trades(a, b, mode='без режима'):
    return NA.trades(mode, [('rsi48', '<=', a), ('rsi6', '>=', b)], lg=False, coins=COINS)


print('  === сетка порогов, SHORT: rsi48 низкий И rsi6 высокий ===', flush=True)
print('    rsi48 \\ rsi6 %s' % ' '.join('%13.0f' % b for b in B_GRID), flush=True)
CACHE = {}
for a in A_GRID:
    cells = []
    for b in B_GRID:
        tr = trades(a, b)
        CACHE[(a, b)] = tr
        v = np.array([x[2] for x in tr]) if tr else np.zeros(0)
        cells.append('%+.4f/%4d' % (v.mean(), len(v)) if len(v) else '      -      ')
    print('    %12.0f %s' % (a, ' '.join(cells)), flush=True)

best = max((k for k in CACHE if len(CACHE[k]) >= 120), key=lambda k: np.mean([x[2] for x in CACHE[k]]))
print('', flush=True)
print('  лучшая клетка с n>=120: rsi48<=%.0f rsi6>=%.0f, n%d, ср%+.4f'
      % (best[0], best[1], len(CACHE[best]), np.mean([x[2] for x in CACHE[best]])), flush=True)

print('', flush=True)
print('  === выбор клетки по четырём годам, замер на пятом ===', flush=True)
ok = 0
tot = 0
for hold in range(2022, 2027):
    lo, hi = YT[hold], YT[hold + 1]
    bk, bv = None, None
    for k, tr in CACHE.items():
        sub = [x[2] for x in tr if not (lo <= x[0] < hi)]
        if len(sub) < 120:
            continue
        v = float(np.mean(sub))
        if bv is None or v > bv:
            bv, bk = v, k
    ho = [x[2] for x in CACHE[bk] if lo <= x[0] < hi]
    if len(ho) < 12:
        print('    %d  выбрано rsi48<=%.0f rsi6>=%.0f — в слепом году мало (n%d)'
              % (hold, bk[0], bk[1], len(ho)), flush=True)
        continue
    tot += 1
    ok += int(np.mean(ho) > 0)
    print('    %d  выбрано rsi48<=%.0f rsi6>=%.0f | слепой год%+.4f (n%d)'
          % (hold, bk[0], bk[1], np.mean(ho), len(ho)), flush=True)
print('    слепых лет в плюсе: %d из %d' % (ok, tot), flush=True)

print('', flush=True)
print('  === атаки на лучшую клетку ===', flush=True)
a, b = best


def st(conds, lg, lab):
    tr = NA.trades('без режима', conds, lg=lg, coins=COINS)
    v = np.array([x[2] for x in tr]) if tr else np.zeros(0)
    if not len(v):
        print('    %-38s сделок нет' % lab, flush=True)
        return None
    print('    %-38s n%5d ВР%5.1f%% ср%+.4fR' % (lab, len(v), 100 * np.mean(v > 0), v.mean()),
          flush=True)
    return tr


st([], False, 'весь поток SHORT без гейта (ноль)')
base = st([('rsi48', '<=', a), ('rsi6', '>=', b)], False, 'ЦЕЛИКОМ')
st([('rsi48', '<=', a)], False, 'только rsi48<=%.0f' % a)
st([('rsi6', '>=', b)], False, 'только rsi6>=%.0f' % b)
st([('rsi48', '>=', a), ('rsi6', '<=', b)], False, 'ИНВЕРСИЯ (должна терять)')
st([('rsi48', '<=', a), ('rsi6', '>=', b)], True, 'ЗЕРКАЛО LONG (должно терять)')

if base:
    print('    по годам:', flush=True)
    line = []
    for y in range(2022, 2027):
        sub = [x[2] for x in base if YT[y] <= x[0] < YT[y + 1]]
        line.append('%d %s' % (y, '%+.3f(%d)' % (np.mean(sub), len(sub)) if len(sub) >= 10 else 'мало'))
    print('      %s' % '  '.join(line), flush=True)
    per = collections.defaultdict(list)
    for x in base:
        per[x[4]].append(x[2])
    print('    монет в плюсе %d из %d' % (sum(1 for s in per if np.mean(per[s]) > 0), len(per)),
          flush=True)
    mons = collections.Counter(
        datetime.datetime.fromtimestamp(x[0], datetime.UTC).strftime('%y-%m') for x in base)
    print('    месяцев %d, максимум в одном %.0f%%'
          % (len(mons), 100 * max(mons.values()) / len(base)), flush=True)

    print('', flush=True)
    print('  === деньги поверх банка + отката ===', flush=True)
    bank = SW.run(3.0, 1.0, coins=COINS, with_meta=True)
    pull = NA.trades('без режима', [('rsi48', '>=', 56.3761), ('rsi6', '<=', 33.7947)], coins=COINS)

    def money(extra):
        ev = [(x[0], x[1], x[2], x[3], x[4], 0) for x in bank]
        ev += [(x[0], x[1], x[2], x[3], x[4], 1) for x in pull]
        ev += [(x[0], x[1], x[2], x[3], x[4], 2) for x in extra]
        ev.sort(key=lambda z: (z[0], z[5]))
        busy, out = {}, []
        for p, q, R, sf, s, k in ev:
            if busy.get(s, 0) > p:
                continue
            busy[s] = q
            out.append((p, q, R, sf, s, 1.0))
        out.sort()
        m = sim_w.money_at_dd(out, 0.12)[1]
        per_ = [sim_w.money_at_dd([r for r in out if YT[y] <= r[0] < YT[y + 1]], 0.12)[1]
                for y in range(2022, 2027)]
        return m, per_

    m0, p0 = money([])
    m1, p1 = money(base)
    print('    без шорта   $%6.0f | по годам %s' % (m0, ' '.join('%3.0f' % x for x in p0)),
          flush=True)
    print('    с шортом    $%6.0f | по годам %s | лучше в %d годах из 5'
          % (m1, ' '.join('%3.0f' % x for x in p1),
             sum(1 for x, y in zip(p0, p1) if y > x)), flush=True)
