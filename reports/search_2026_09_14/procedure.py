"""The last question: would the procedure have found this rule without seeing the test year?

Everything so far validated the RULE - its thresholds re-picked blind, its mechanism taken apart, its
money measured two ways. What was never tested is the PROCEDURE that produced it. The pullback rule
was chosen from 26,866 survivors by someone who could see all five years. A method that only works
when it can see the answer is not a method.

So the whole selection is run inside each fold. For each year in turn: every distinct family is
scored on the other four years by what it does to the account, the best one is taken, and the
discarded year is then looked at once. Five folds.

Two readings. Whether the blindly chosen rule helps in the year it did not see - that is whether the
method works at all. And how often the method picks the pullback rule itself - that is whether this
particular rule was the obvious answer or a lucky one.
"""
import collections
import csv
import datetime
import pickle
import sys

import numpy as np

sys.path.insert(0, 'C:/Users/Lenovo/Desktop/Торговля/crypto-bot')
import struct_params as SP
import gauntlet2 as G
import stop_width as SW
import sim_w
import nogate_attack as NA

cost = collections.defaultdict(list)
for r in csv.DictReader(open('book_frozen.csv')):
    cost[r['coin'] + 'USDT'].append(float(r['cost109']))
SP.SLIP_REAL.update({k: float(np.median(v)) for k, v in cost.items()})
G.FC.clear()
COINS = [s for s in G.ALL if s not in ('AAVEUSDT', 'XLMUSDT')]
YEARS = [2022, 2023, 2024, 2025, 2026]
YT = {y: int(datetime.datetime(y, 1, 1, tzinfo=datetime.UTC).timestamp()) for y in YEARS + [2027]}
PULL = ('ОТКАТ rsi48>=56.38 + rsi6<=33.79', True,
        [('rsi48', '>=', 56.3761), ('rsi6', '<=', 33.7947)])

CAND = [(l, s, c) for l, s, c in pickle.load(open('gauntlet_pass_full.pkl', 'rb'))] + [PULL]
print('  кандидатов в отборе: %d (включая сам откат)' % len(CAND), flush=True)

bank = SW.run(3.0, 1.0, coins=COINS, with_meta=True)
TR = {}
for lbl, side, conds in CAND:
    tr = NA.trades('без режима', conds, lg=side, coins=COINS)
    if len(tr) >= 60:
        TR[lbl] = [(x[0], x[1], x[2], x[3], x[4]) for x in tr]
print('  с достаточным числом сделок: %d' % len(TR), flush=True)


def acc(name=None):
    ev = [(x[0], x[1], x[2], x[3], x[4], 0) for x in bank]
    if name:
        ev += [(x[0], x[1], x[2], x[3], x[4], 1) for x in TR[name]]
    ev.sort(key=lambda z: (z[0], z[5]))
    busy, out = {}, []
    for p, q, R, sf, s, k in ev:
        if busy.get(s, 0) > p:
            continue
        busy[s] = q
        out.append((p, q, R, sf, s, 1.0))
    out.sort()
    return out


CACHE = {None: acc()}
for name in TR:
    CACHE[name] = acc(name)


def money(rows, keep):
    sub = [r for r in rows if keep(r[0])]
    if len(sub) < 80:
        return 0.0
    return sim_w.money_at_dd(sub, 0.12)[1]


print('', flush=True)
print('  === отбор правила на четырёх годах, замер на пятом ===', flush=True)
wins = seen = 0
picked = collections.Counter()
for hold in YEARS:
    lo, hi = YT[hold], YT[hold + 1]
    train = lambda t: not (lo <= t < hi)
    blind = lambda t: lo <= t < hi
    base_train = money(CACHE[None], train)
    best, pick = base_train, None
    for name in TR:
        m = money(CACHE[name], train)
        if m > best:
            best, pick = m, name
    base_blind = money(CACHE[None], blind)
    got_blind = money(CACHE[pick], blind) if pick else base_blind
    if base_blind <= 0 or got_blind <= 0:
        print('    %d  выбрано %s | слепой год не измерить' % (hold, pick), flush=True)
        continue
    seen += 1
    wins += int(got_blind > base_blind)
    if pick:
        picked[pick] += 1
    print('    %d  выбрано: %-40s' % (hold, (pick or 'ничего')[:40]), flush=True)
    print('        обучение $%6.0f (банк один $%6.0f) | СЛЕПОЙ ГОД $%5.0f против банка $%5.0f  %s'
          % (best, base_train, got_blind, base_blind,
             'лучше' if got_blind > base_blind else 'ХУЖЕ'), flush=True)

print('', flush=True)
print('  слепых лет, где выбранное вслепую правило лучше банка: %d из %d' % (wins, seen), flush=True)
print('  что выбиралось:', flush=True)
for name, c in picked.most_common():
    print('    %d/%d  %s' % (c, seen, name), flush=True)

print('', flush=True)
print('  === а сам откат, в каждом слепом году (он выбран НЕ вслепую) ===', flush=True)
for hold in YEARS:
    lo, hi = YT[hold], YT[hold + 1]
    blind = lambda t: lo <= t < hi
    b = money(CACHE[None], blind)
    p = money(CACHE[PULL[0]], blind)
    if b <= 0 or p <= 0:
        continue
    print('    %d  банк $%5.0f -> с откатом $%5.0f  (%+.0f%%)' % (hold, b, p, 100 * (p / b - 1)),
          flush=True)
