"""Replace the cliff with a slope: shrink the position as the drawdown deepens.

Today the account trades at full size down to -14.9% and switches off forever at -15%. That cliff is
what made every comparison in this session unstable - a variant landing a tenth of a percent on the
wrong side of it lost 80% of its money for reasons that had nothing to do with its trades.

A proportional guard does the same job continuously: the deeper the account is below its own peak,
the smaller the next position. It cannot be argued into existence - de-risking mechanically reduces
drawdown, so of course the drawdown falls. The question is whether it BUYS anything: at a fixed
worst drawdown, each variant is allowed its own base risk, and the one that ends with more money has
earned it.

Four shapes, and the latch left in place underneath as a backstop in every one of them.
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
import margin_cap as MC
import live_rules_sim as L

meas = collections.defaultdict(list)
for r in csv.DictReader(open('book_samples.csv')):
    meas[r['coin'] + 'USDT'].append(float(r['cost109']))
SP.SLIP_REAL.update({k: float(np.median(v)) for k, v in meas.items()})
G.FC.clear()

DEPOSIT, LEV, REF = L.DEPOSIT, L.LEV, L.REF
MAX_DD, MAX_DAILY, USABLE = L.MAX_DD, L.MAX_DAILY, MC.USABLE


def simulate(trades, target, shape=None, latch=True):
    """shape(dd_fraction) -> size multiplier in [0,1]; dd_fraction is how far below peak, 0..1."""
    margin_frac = target / (LEV * REF)
    eq = peak = DEPOSIT
    open_pos = []
    day, day_start, day_paused = None, DEPOSIT, False
    paused, paused_at = False, None
    curve, taken = [], 0
    for a, b, R, sf, s in trades:
        while open_pos and open_pos[0][0] <= a:
            t_close, m, mpr, rr = open_pos.pop(0)
            eq += mpr * rr
            peak = max(peak, eq)
            curve.append((t_close, eq))
        d = L.day_of(a)
        if d != day:
            day, day_start, day_paused = d, eq, False
        dd = max(0.0, 1 - eq / peak)
        if latch and not paused and dd >= MAX_DD:
            paused, paused_at = True, a
        if not day_paused and eq <= day_start * (1 - MAX_DAILY):
            day_paused = True
        if paused or day_paused:
            continue
        w = 1.0 if shape is None else float(shape(dd))
        if w <= 0:
            continue
        margin = margin_frac * eq
        if sf > REF:
            margin *= REF / sf
        margin *= w
        used = sum(p[1] for p in open_pos)
        if used + margin > eq * USABLE:
            continue
        taken += 1
        open_pos.append((b, margin, margin * LEV * sf, R))
        open_pos.sort(key=lambda p: p[0])
    for t_close, m, mpr, rr in open_pos:
        eq += mpr * rr
        curve.append((t_close, eq))
    return dict(eq=eq, curve=curve, paused_at=paused_at, taken=taken)


def at_dd(trades, shape, want=0.12, latch=True):
    lo, hi = 0.0005, 0.060
    for _ in range(28):
        mid = (lo + hi) / 2
        r = simulate(trades, mid, shape, latch)
        if bool(r['paused_at']) or abs(L.dd_of(r['curve'])) > want:
            hi = mid
        else:
            lo = mid
    r = simulate(trades, lo, shape, latch)
    return lo, r['eq'], abs(L.dd_of(r['curve'])), r['taken']


C = [('rsi48', '>=', 56.3761), ('rsi6', '<=', 33.7947)]
ALL = [c for c in G.ALL if c != 'AAVEUSDT']
bank = SW.run(3.0, 1.0, coins=ALL, with_meta=True)
cand, _ = G.simulate(C, True, coins=ALL)
ev = [(a, b, R, sf, s, 0) for a, b, R, sf, s in bank] + \
     [(a, b, R, sf, s, 1) for a, b, R, sf, s in cand]
ev.sort(key=lambda x: (x[0], x[5]))
busy, TR = {}, []
for a, b, R, sf, s, t in ev:
    if busy.get(s, 0) > a:
        continue
    busy[s] = b
    TR.append((a, b, R, sf, s))
print('  банк+откат без AAVE, измеренные издержки: %d сделок' % len(TR), flush=True)

SHAPES = [
    ('нет (как сейчас)', None),
    ('линейно до нуля', lambda d: max(0.0, 1 - d / MAX_DD)),
    ('линейно до половины', lambda d: 1 - 0.5 * min(1.0, d / MAX_DD)),
    ('квадрат', lambda d: max(0.0, 1 - d / MAX_DD) ** 2),
    ('корень', lambda d: max(0.0, 1 - d / MAX_DD) ** 0.5),
    ('ступень 5%->0.75, 10%->0.5', lambda d: 1.0 if d < 0.05 else (0.75 if d < 0.10 else 0.5)),
    ('ступень 7%->0.5', lambda d: 1.0 if d < 0.07 else 0.5),
    ('плавно от 5%', lambda d: 1.0 if d < 0.05 else max(0.0, 1 - (d - 0.05) / (MAX_DD - 0.05))),
]
print('', flush=True)
print('  === при равной худшей просадке 12%% ===', flush=True)
res = {}
for lbl, sh in SHAPES:
    k, m, dd, n = at_dd(TR, sh)
    res[lbl] = m
    print('    %-28s риск %.3f%%  $%6.0f  просадка %5.1f%%  сделок %d'
          % (lbl, 100 * k, m, 100 * dd, n), flush=True)

print('', flush=True)
print('  === и при других уровнях просадки ===', flush=True)
print('    %-28s %s' % ('форма', ' '.join('%9s' % ('%.0f%%' % (100 * w))
                                          for w in (0.08, 0.10, 0.12, 0.15, 0.20))), flush=True)
for lbl, sh in SHAPES:
    cells = []
    for want in (0.08, 0.10, 0.12, 0.15, 0.20):
        k, m, dd, n = at_dd(TR, sh, want)
        cells.append('%9.0f' % m)
    print('    %-28s %s' % (lbl, ' '.join(cells)), flush=True)

print('', flush=True)
print('  === что делает защёлка, если под ней есть наклон ===', flush=True)
for lbl, sh in SHAPES[:5]:
    k1, m1, d1, _ = at_dd(TR, sh, 0.12, latch=True)
    k2, m2, d2, _ = at_dd(TR, sh, 0.12, latch=False)
    print('    %-28s с защёлкой $%6.0f | без защёлки $%6.0f' % (lbl, m1, m2), flush=True)
