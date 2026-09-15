"""30% a month: what the market offers, and what each lever costs.

Monthly return = trades x edge per trade x risk per trade. Today: ~47 trades a month, +0.10R, 1.4%.
The target needs that product roughly four times larger. Three things are measured:

  1. the market itself - how far coins actually move in a day, and what perfect hindsight would make
  2. the risk lever - the same book at higher risk, latch OFF, so the true drawdown is visible
  3. the frequency lever is tested separately (target30_freq.py)
"""
import collections, datetime, sys
import numpy as np
sys.path.insert(0, 'C:/Users/Lenovo/Desktop/Торговля/crypto-bot')
import struct_params as SP
import exec_model as EM
import sim_w
import live_rules_sim as L

COINS = [s for s in EM.ALL if s not in ('XLMUSDT', 'AAVEUSDT')]

print('  === 1. сколько рынок двигается за день (последний год) ===', flush=True)
print('  монета   средний дневной размах   лучший лонг за день задним числом', flush=True)
tot_range, tot_best = [], []
for s in COINS:
    c = SP.CTX[s]
    t, A = c['t15'], c['a15']
    last = t[-1]
    m = t >= last - 365 * 86400
    t, A = t[m], A[m]
    day = t // 86400
    rng, best = [], []
    for d in np.unique(day):
        k = day == d
        if k.sum() < 80:
            continue
        lo, hi = A[k, 2], A[k, 1]
        rng.append(hi.max() / lo.min() - 1)
        # best long: buy at a low, sell at a later high, inside the day
        cummin = np.minimum.accumulate(lo)
        best.append((hi / cummin - 1).max())
    tot_range.append(np.mean(rng)); tot_best.append(np.mean(best))
    print('    %-6s      %5.2f%%                    %5.2f%%' % (s.replace('USDT', ''), 100*np.mean(rng), 100*np.mean(best)), flush=True)
print('    в среднем: размах %.2f%% в день, идеальный лонг %.2f%% в день' % (100*np.mean(tot_range), 100*np.mean(tot_best)), flush=True)

print('', flush=True)
print('  === 2. та же книга при большем риске, защёлка ВЫКЛЮЧЕНА ===', flush=True)
rows = EM.book(COINS, EM.SIG_ALL, 'рынок')[0]
months = (rows[-1][1] - rows[0][0]) / (365.25 * 86400 / 12)
print('  сделок %d за %.0f мес (%.0f в месяц)' % (len(rows), months, len(rows) / months), flush=True)
print('  риск на сделку   в месяц    итог из $120        худшая просадка', flush=True)
for risk in (0.014, 0.02, 0.03, 0.04, 0.05, 0.07, 0.10):
    r = sim_w.simulate(rows, risk, latch=False)
    eq = r['eq']
    mo = (eq / 120) ** (1 / months) - 1 if eq > 0 else -1
    print('    %5.1f%%        %+6.2f%%   $%14.0f      %5.1f%%'
          % (100 * risk, 100 * mo, eq, 100 * abs(L.dd_of(r['curve']))), flush=True)
