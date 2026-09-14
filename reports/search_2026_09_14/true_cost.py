"""The whole account on the fill costs that were actually measured, not the ones assumed.

A day of order-book sampling at ten-minute intervals gives the real cost of entering each coin at
the sizes this account uses. Against the constants the model has been running on:

    XLM   assumed 0.00106, measured 0.001496   understated by 41%
    DOT   assumed 0.00026, measured 0.000666   understated 2.6x
    NEAR  assumed 0.00021, measured 0.000427   understated 2x
    TAO   assumed 0.00021, measured 0.000422   understated 2x

Costs are subtracted twice per trade, on the way in and on the way out, so an understatement of a
few basis points per side is not decoration. Everything is re-measured on the measured numbers: the
per-coin table, the drop-one-coin control, and the money.

The sample is one day, so it says what the book looks like now rather than what it looked like in
2022. That cuts one way only - it cannot make a coin look worse than it really is today.
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

rows = list(csv.DictReader(open('book_samples.csv')))
meas = collections.defaultdict(list)
for r in rows:
    meas[r['coin'] + 'USDT'].append(float(r['cost109']))
MEASURED = {k: float(np.median(v)) for k, v in meas.items()}
print('  измеренные издержки против заложенных:', flush=True)
for s in sorted(MEASURED, key=lambda x: -MEASURED[x]):
    old = SP.SLIP_REAL.get(s, 0.0002)
    print('    %-6s заложено %.5f  измерено %.6f  x%.2f'
          % (s.replace('USDT', ''), old, MEASURED[s], MEASURED[s] / max(old, 1e-9)), flush=True)

C = [('rsi48', '>=', 56.3761), ('rsi6', '<=', 33.7947)]
OLD = dict(SP.SLIP_REAL)
YT = {y: int(datetime.datetime(y, 1, 1, tzinfo=datetime.UTC).timestamp()) for y in range(2022, 2028)}


def account(coins, with_pb):
    bank = SW.run(3.0, 1.0, coins=coins, with_meta=True)
    ev = [(a, b, R, sf, s, 'bank') for a, b, R, sf, s in bank]
    if with_pb:
        cand, _ = G.simulate(C, True, coins=coins)
        ev += [(a, b, R, sf, s, 'cand') for a, b, R, sf, s in cand]
    ev.sort(key=lambda x: (x[0], 0 if x[5] == 'bank' else 1))
    busy, out = {}, []
    for a, b, R, sf, s, tag in ev:
        if busy.get(s, 0) > a:
            continue
        busy[s] = b
        out.append((a, b, R, sf, s, 1.0))
    out.sort()
    return out


ALL = list(G.ALL)
for lbl, table in (('ЗАЛОЖЕННЫЕ издержки', OLD), ('ИЗМЕРЕННЫЕ издержки', MEASURED)):
    SP.SLIP_REAL.clear()
    SP.SLIP_REAL.update(table)
    G.FC.clear()
    SW.CACHE.clear() if hasattr(SW, 'CACHE') else None
    print('', flush=True)
    print('  ===== %s =====' % lbl, flush=True)
    b = account(ALL, False)
    p = account(ALL, True)
    per = collections.defaultdict(list)
    for a, bb, R, sf, s, w in b:
        per[s].append(R)
    print('    по монетам:', flush=True)
    for s in sorted(per, key=lambda x: -np.mean(per[x])):
        v = np.array(per[s])
        print('      %-6s n%4d ВР%5.1f%% ср%+.4f' % (s.replace('USDT', ''), len(v),
                                                     100 * np.mean(v > 0), v.mean()), flush=True)
    mb = sim_w.money_at_dd(b, 0.12)[1]
    mp = sim_w.money_at_dd(p, 0.12)[1]
    noa = [c for c in ALL if c != 'AAVEUSDT']
    mb2 = sim_w.money_at_dd(account(noa, False), 0.12)[1]
    mp2 = sim_w.money_at_dd(account(noa, True), 0.12)[1]
    noax = [c for c in noa if c != 'XLMUSDT']
    mb3 = sim_w.money_at_dd(account(noax, False), 0.12)[1]
    mp3 = sim_w.money_at_dd(account(noax, True), 0.12)[1]
    print('    банк $%.0f | банк+откат $%.0f' % (mb, mp), flush=True)
    print('    без AAVE: банк $%.0f | банк+откат $%.0f' % (mb2, mp2), flush=True)
    print('    без AAVE и XLM: банк $%.0f | банк+откат $%.0f' % (mb3, mp3), flush=True)
