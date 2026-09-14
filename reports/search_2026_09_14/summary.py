"""Win rate, profit and drawdown for the system as it stands, in one place.

The three numbers the user asks for are not independent: win rate is bought with money through the
geometry, and profit is bounded by the drawdown the account may carry before the latch fires. So
each is reported at a fixed setting of the others rather than at its own best.

Two books - all fifteen coins, and without the two the live spread gate refuses - because the
difference between them is larger than most of the effects measured today.
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
import sim_w
import live_rules_sim as L
import nogate_attack as NA

cost = collections.defaultdict(list)
for r in csv.DictReader(open('book_frozen.csv')):
    cost[r['coin'] + 'USDT'].append(float(r['cost109']))
SP.SLIP_REAL.update({k: float(np.median(v)) for k, v in cost.items()})
G.FC.clear()
ALL = list(G.ALL)
NOBAD = [s for s in ALL if s not in ('AAVEUSDT', 'XLMUSDT')]
C = [('rsi48', '>=', 56.3761), ('rsi6', '<=', 33.7947)]


def account(coins, with_rule):
    bank = SW.run(3.0, 1.0, coins=coins, with_meta=True)
    cand = NA.trades('без режима', C, coins=coins) if with_rule else []
    ev = [(x[0], x[1], x[2], x[3], x[4], 0) for x in bank]
    ev += [(x[0], x[1], x[2], x[3], x[4], 1) for x in cand]
    ev.sort(key=lambda z: (z[0], z[5]))
    busy, out = {}, []
    for p, q, R, sf, s, k in ev:
        if busy.get(s, 0) > p:
            continue
        busy[s] = q
        out.append((p, q, R, sf, s, 1.0, k))
    out.sort()
    return out


def line(lbl, rows, risk):
    plain = [(a, b, R, sf, s, w) for a, b, R, sf, s, w, k in rows]
    r = sim_w.simulate(plain, risk)
    v = np.array([x[2] for x in rows])
    dd = abs(L.dd_of(r['curve']))
    months = (rows[-1][1] - rows[0][0]) / (365.25 * 86400 / 12)
    gain = r['eq'] / L.DEPOSIT
    print('  %-26s %5d сд. ВР %4.1f%% | $%7.0f из $%.0f (x%5.1f, %+5.2f%%/мес) | просадка %5.1f%% %s'
          % (lbl, len(v), 100 * np.mean(v > 0), r['eq'], L.DEPOSIT, gain,
             100 * (gain ** (1 / months) - 1), 100 * dd,
             'ЗАЩЁЛКА' if r['paused_at'] else ''), flush=True)
    return r['eq'], dd


print('  === при риске 1.40%% на сделку (нынешняя настройка предложения) ===', flush=True)
for cl, coins in (('все 15 монет', ALL), ('без AAVE и XLM', NOBAD)):
    for wl, wr in (('банк', False), ('банк+откат', True)):
        line('%s, %s' % (cl, wl), account(coins, wr), 0.014)
print('', flush=True)
print('  === каждому свой риск, чтобы просадка была ровно 12%% ===', flush=True)
for cl, coins in (('все 15 монет', ALL), ('без AAVE и XLM', NOBAD)):
    for wl, wr in (('банк', False), ('банк+откат', True)):
        rows = account(coins, wr)
        plain = [(a, b, R, sf, s, w) for a, b, R, sf, s, w, k in rows]
        k, m = sim_w.money_at_dd(plain, 0.12)
        line('%s, %s (риск %.2f%%)' % (cl, wl, 100 * k), rows, k)
print('', flush=True)
print('  === лестница риска для предложения (без AAVE и XLM, банк+откат) ===', flush=True)
rows = account(NOBAD, True)
for risk in (0.008, 0.010, 0.0125, 0.014, 0.0155, 0.017, 0.019):
    line('риск %.2f%%' % (100 * risk), rows, risk)
print('', flush=True)
print('  === винрейт покупается геометрией: что он стоит (банк+откат, без AAVE и XLM) ===',
      flush=True)
print('  геометрия правила   ВР правила   ВР счёта   деньги при равной просадке 12%', flush=True)
