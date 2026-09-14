"""Is the mechanism about horizons, or about the numbers 48 and 6?

The pullback rule pairs a two-day RSI held high with a six-hour RSI pushed low. If that describes a
real thing - a dip inside strength - then neighbouring horizon pairs should show it too, weaker or
stronger but the same sign. If only rsi48 with rsi6 works and its neighbours are flat, the pair was
selected rather than discovered, and 26,866 survivors is more than enough to select one.

Four RSI horizons exist in the feature set: rsi2, rsi6, rsi14, rsi48. Every ordered pair where the
slow one is held high and the fast one pushed low is measured on the same grid, gateless, with the
working thresholds' quantiles carried across so that the comparison is like for like.
"""
import collections
import datetime
import sys

import numpy as np

sys.path.insert(0, 'C:/Users/Lenovo/Desktop/Торговля/crypto-bot')
import gauntlet2 as G
import nogate_attack as NA
import full_money as FM

COINS = FM.COINS
YT = FM.YT
RSI = ['rsi2', 'rsi6', 'rsi14', 'rsi48']
ORDER = {'rsi2': 0, 'rsi6': 1, 'rsi14': 2, 'rsi48': 3}

# the quantiles the working thresholds sit at, so every pair is compared at the same selectivity
vals = collections.defaultdict(list)
for s in COINS:
    _, f = G.feats(s)
    for n in RSI:
        v = f[n]
        vals[n].append(v[np.isfinite(v)])
Q = {n: np.concatenate(v) for n, v in vals.items()}
q_slow = float((Q['rsi48'] <= 56.3761).mean())
q_fast = float((Q['rsi6'] <= 33.7947).mean())
print('  рабочие пороги стоят на квантилях: медленный %.3f (сверху %.1f%%), быстрый %.3f'
      % (q_slow, 100 * (1 - q_slow), q_fast), flush=True)
print('', flush=True)

print('  пара (медленный высокий + быстрый низкий), на тех же квантилях:', flush=True)
print('  %-22s %6s %7s %9s %9s   по годам' % ('пара', 'сделок', 'ВР', 'ср R', 'месяцев'), flush=True)
res = {}
for slow in RSI:
    for fast in RSI:
        if ORDER[fast] >= ORDER[slow]:
            continue
        a = float(np.quantile(Q[slow], q_slow))
        b = float(np.quantile(Q[fast], q_fast))
        tr = NA.trades('без режима', [(slow, '>=', a), (fast, '<=', b)], coins=COINS)
        if len(tr) < 40:
            print('  %-22s сделок %d — мало' % ('%s>=%.1f + %s<=%.1f' % (slow, a, fast, b), len(tr)),
                  flush=True)
            continue
        v = np.array([x[2] for x in tr])
        mons = collections.Counter(
            datetime.datetime.fromtimestamp(x[0], datetime.UTC).strftime('%y-%m') for x in tr)
        yr = []
        for y in range(2022, 2027):
            sub = [x[2] for x in tr if YT[y] <= x[0] < YT[y + 1]]
            yr.append('%+.3f' % np.mean(sub) if len(sub) >= 10 else '  -  ')
        mark = '  <- рабочее' if (slow == 'rsi48' and fast == 'rsi6') else ''
        print('  %-22s %6d %6.1f%% %+9.4f %9d   %s%s'
              % ('%s>=%.1f+%s<=%.1f' % (slow, a, fast, b), len(v), 100 * np.mean(v > 0), v.mean(),
                 len(mons), ' '.join(yr), mark), flush=True)
        res[(slow, fast)] = v.mean()

print('', flush=True)
print('  === и то же самое при вдвое более мягких порогах (проверка, что дело в горизонтах) ===',
      flush=True)
for slow in RSI:
    for fast in RSI:
        if ORDER[fast] >= ORDER[slow]:
            continue
        a = float(np.quantile(Q[slow], (q_slow + 1) / 2 if q_slow > 0.5 else q_slow * 0.5))
        a = float(np.quantile(Q[slow], min(0.99, q_slow * 0.7 + 0.3)))
        b = float(np.quantile(Q[fast], min(0.99, q_fast * 2)))
        tr = NA.trades('без режима', [(slow, '>=', a), (fast, '<=', b)], coins=COINS)
        if len(tr) < 40:
            continue
        v = np.array([x[2] for x in tr])
        print('  %-26s %6d сделок ВР%5.1f%% ср%+.4f'
              % ('%s>=%.1f + %s<=%.1f' % (slow, a, fast, b), len(v), 100 * np.mean(v > 0),
                 v.mean()), flush=True)

print('', flush=True)
print('  === а если МЕДЛЕННЫЙ низкий и БЫСТРЫЙ низкий (просто слабость) ===', flush=True)
for slow, fast in (('rsi48', 'rsi6'), ('rsi14', 'rsi6'), ('rsi48', 'rsi2')):
    a = float(np.quantile(Q[slow], 1 - q_slow))
    b = float(np.quantile(Q[fast], q_fast))
    tr = NA.trades('без режима', [(slow, '<=', a), (fast, '<=', b)], coins=COINS)
    if len(tr) < 40:
        continue
    v = np.array([x[2] for x in tr])
    print('  %-26s %6d сделок ВР%5.1f%% ср%+.4f'
          % ('%s<=%.1f + %s<=%.1f' % (slow, a, fast, b), len(v), 100 * np.mean(v > 0), v.mean()),
          flush=True)
