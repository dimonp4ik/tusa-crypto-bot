"""The fourteen distinct blind-year survivors, through the gauntlet that killed everything else.

Surviving five blind years says the condition is related to the outcome. It says nothing about
whether the account can take the trade: a condition that persists collapses under one-position-per-
coin, a condition that fires in bursts cannot be spread across an account, a condition that pays
both ways has found range rather than direction, and a condition the bank already trades adds
nothing.

The bar, unchanged from the rest of the session: at least 30 distinct months, no month over 25% of
the trades, the mirror must lose, and the overlap with what is already taken must be small.
"""
import collections
import datetime
import pickle
import sys

import numpy as np

sys.path.insert(0, 'C:/Users/Lenovo/Desktop/Торговля/crypto-bot')
import gauntlet2 as G
import stop_width as SW

chosen = pickle.load(open('shortlist.pkl', 'rb'))
bank = SW.run(3.0, 1.0, coins=G.ALL, with_meta=True)
BK = {(a, s) for a, b, R, sf, s in bank}
PB, _ = G.simulate([('rsi48', '>=', 56.3761), ('rsi6', '<=', 33.7947)], True)
PBK = {(r[0], r[4]) for r in PB}
print('  банк %d, откат %d' % (len(BK), len(PBK)), flush=True)
print('', flush=True)

keep = []
for lbl, side, conds in chosen:
    cc = [(nm, op, t) for nm, op, t in conds]
    rows, hours = G.simulate(cc, side)
    if not rows:
        print('  %-46s сделок нет' % lbl, flush=True)
        continue
    v = np.array([r[2] for r in rows])
    mons = collections.Counter(
        datetime.datetime.fromtimestamp(r[0], datetime.UTC).strftime('%y-%m') for r in rows)
    conv = len(rows) / max(hours, 1)
    worst = max(mons.values()) / len(rows)
    keys = {(r[0], r[4]) for r in rows}
    ob, op_ = len(keys & BK) / len(keys), len(keys & PBK) / len(keys)
    mir, _ = G.simulate(cc, not side)
    mv = np.array([r[2] for r in mir]) if mir else np.zeros(0)
    verdict = []
    if conv < 0.35:
        verdict.append('СОСТОЯНИЕ')
    if len(mons) < 30:
        verdict.append('мало месяцев')
    if worst > 0.25:
        verdict.append('сбито в месяц')
    if len(mv) and mv.mean() > 0.05:
        verdict.append('ЗЕРКАЛО ТОЖЕ ПЛЮС')
    if ob > 0.3:
        verdict.append('банк уже берёт')
    print('  %-46s n%4d ВР%5.1f%% ср%+.4f конв%3.0f%% мес%3d макс%3.0f%% банк%3.0f%% откат%3.0f%% '
          'зеркало%+.4f  %s'
          % (lbl, len(v), 100 * np.mean(v > 0), v.mean(), 100 * conv, len(mons), 100 * worst,
             100 * ob, 100 * op_, mv.mean() if len(mv) else 0,
             ' / '.join(verdict) if verdict else 'ПРОШЛО'), flush=True)
    if not verdict:
        keep.append((lbl, side, cc, rows))

print('', flush=True)
print('  прошло конвейер: %d' % len(keep), flush=True)
pickle.dump([(l, s, c) for l, s, c, _ in keep], open('gauntlet_pass.pkl', 'wb'))
if len(keep) > 1:
    print('  пересечение прошедших между собой:', flush=True)
    for i in range(len(keep)):
        for j in range(i + 1, len(keep)):
            a = {(r[0], r[4]) for r in keep[i][3]}
            b = {(r[0], r[4]) for r in keep[j][3]}
            print('    %-42s x %-42s %3d' % (keep[i][0], keep[j][0], len(a & b)), flush=True)
