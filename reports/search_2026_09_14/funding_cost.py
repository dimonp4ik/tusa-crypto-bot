"""Funding was never in the bank model. Price it on the real trade book.

Perpetuals pay funding every 8 hours (00, 08, 16 UTC); a position open across a boundary pays or
receives rate x notional. The book holds positions up to 48 hours and is mostly long, and the rates
in the sampled months are mostly positive, so longs pay.

History covers only 2026-05-25..08-26. Two estimates: actual rates on trades inside that window, and
each coin's mean rate applied to every boundary crossed over the full history. Every trade is
treated as LONG, which overstates the cost - bank shorts would receive funding when rates are
positive - so this is an upper bound.
"""
import json, sys
import numpy as np
sys.path.insert(0, 'C:/Users/Lenovo/Desktop/Торговля/crypto-bot')
import exec_model as EM
import sim_w
import live_rules_sim as L

fh = json.load(open('C:/Users/Lenovo/Desktop/Торговля/crypto-bot/funding_history.json', encoding='utf-8'))
RATES = {s: {int(t): float(r) for t, r in v} for s, v in fh.items()}
MEAN = {s: float(np.mean(list(v.values()))) for s, v in RATES.items()}
lo_cov = min(min(v) for v in RATES.values())
hi_cov = max(max(v) for v in RATES.values())
STEP = 8 * 3600

COINS = [s for s in EM.ALL if s not in ('XLMUSDT', 'AAVEUSDT')]
rows = EM.book(COINS, EM.SIG_ALL, 'рынок')[0]


def boundaries(a, b):
    first = (a // STEP + 1) * STEP
    return list(range(first, b + 1, STEP))


act_R, est_R, adj = [], [], []
for a, b, R, sf, s, w in rows:
    bs = boundaries(a, b)
    est = sum(MEAN.get(s, 0.0) for _ in bs)
    est_R.append(est / sf)
    if lo_cov <= a and b <= hi_cov:
        act = sum(RATES.get(s, {}).get(t, MEAN.get(s, 0.0)) for t in bs)
        act_R.append(act / sf)
    adj.append((a, b, R - est / sf, sf, s, w))

base = np.array([x[2] for x in rows])
est_R = np.array(est_R)
print('  сделок %d, средний результат %+.4fR' % (len(rows), base.mean()))
print('  выплат funding на сделку в среднем: %.2f' % np.mean([len(boundaries(a, b)) for a, b, *_ in rows]))
print('  цена funding по средним ставкам, вся история: %+.5fR на сделку (%.1f%% эджа)'
      % (est_R.mean(), 100 * est_R.mean() / base.mean()))
if act_R:
    print('  по реальным ставкам на %d сделках покрытого окна: %+.5fR на сделку' % (len(act_R), np.mean(act_R)))
r0 = sim_w.simulate(rows, 0.014)
r1 = sim_w.simulate(adj, 0.014)
print('  деньги при 1.4%%: без funding $%.0f, с funding $%.0f (%+.1f%%), просадка %.1f%% -> %.1f%%'
      % (r0['eq'], r1['eq'], 100 * (r1['eq'] / r0['eq'] - 1),
         100 * abs(L.dd_of(r0['curve'])), 100 * abs(L.dd_of(r1['curve']))))
