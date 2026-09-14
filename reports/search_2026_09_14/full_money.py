"""Thirty-four gauntlet survivors, each priced on top of the configuration that already stands.

Passing the gauntlet means the trades are takeable, not that they are worth taking. Last time one of
five added money and then died on its threshold grid. The test that matters is what each family does
to the account it would actually join: the bank, the live spread gate, and the gateless pullback
rule.

Anything that adds less than 5% is not worth a config change and is not reported as a candidate.
Anything that adds more goes on to the same attacks the pullback rule survived.
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

meas = collections.defaultdict(list)
spread = collections.defaultdict(list)
for r in csv.DictReader(open('book_samples.csv')):
    meas[r['coin'] + 'USDT'].append(float(r['cost109']))
    spread[r['coin'] + 'USDT'].append(float(r['spread']))
SP.SLIP_REAL.update({k: float(np.median(v)) for k, v in meas.items()})
G.FC.clear()
MED = {k: float(np.median(v)) for k, v in spread.items()}
COINS = [s for s in G.ALL if MED.get(s, 0) <= 0.0005]
print('  монет после отсечки спреда: %d' % len(COINS), flush=True)

C = [('rsi48', '>=', 56.3761), ('rsi6', '<=', 33.7947)]
YT = {y: int(datetime.datetime(y, 1, 1, tzinfo=datetime.UTC).timestamp()) for y in range(2022, 2028)}
SLICE = 182 * 86400

bank = SW.run(3.0, 1.0, coins=COINS, with_meta=True)
pull = NA.trades('без режима', C, coins=COINS)
PASS = pickle.load(open('gauntlet_pass_full.pkl', 'rb'))
print('  опора: банк %d + откат %d; кандидатов %d' % (len(bank), len(pull), len(PASS)), flush=True)


def merge(extra):
    ev = [(a, b, R, sf, s, 0) for a, b, R, sf, s in bank]
    ev += [(a, b, R, sf, s, 1) for a, b, R, sf, s in pull]
    ev += [(a, b, R, sf, s, 2) for a, b, R, sf, s in extra]
    ev.sort(key=lambda x: (x[0], x[5]))
    busy, out = {}, []
    for a, b, R, sf, s, k in ev:
        if busy.get(s, 0) > a:
            continue
        busy[s] = b
        out.append((a, b, R, sf, s, 1.0))
    out.sort()
    return out


REF = merge([])
m0 = sim_w.money_at_dd(REF, 0.12)[1]
p0 = [sim_w.money_at_dd([r for r in REF if YT[y] <= r[0] < YT[y + 1]], 0.12)[1]
      for y in range(2022, 2027)]
print('  опора: $%.0f | по годам %s' % (m0, ' '.join('%3.0f' % x for x in p0)), flush=True)
print('', flush=True)

res = []
for lbl, side, conds in PASS:
    tr, _ = G.simulate(conds, side, coins=COINS)
    if len(tr) < 60:
        continue
    rows = merge([(x[0], x[1], x[2], x[3], x[4]) for x in tr])
    m = sim_w.money_at_dd(rows, 0.12)[1]
    per = [sim_w.money_at_dd([r for r in rows if YT[y] <= r[0] < YT[y + 1]], 0.12)[1]
           for y in range(2022, 2027)]
    better = sum(1 for a, b in zip(p0, per) if b > a)
    res.append((m, lbl, len(tr), better, per))
res.sort(reverse=True)
print('  кандидат                                           своих  деньги  прибавка  лет лучше',
      flush=True)
for m, lbl, n, better, per in res:
    print('  %-50s %4d  $%6.0f  %+6.1f%%    %d/5  | %s'
          % (lbl, n, m, 100 * (m / m0 - 1), better, ' '.join('%3.0f' % x for x in per)), flush=True)

print('', flush=True)
good = [x for x in res if x[0] / m0 - 1 > 0.05 and x[3] >= 4]
print('  прибавка больше 5%% И лучше опоры хотя бы в 4 годах из 5: %d' % len(good), flush=True)
for m, lbl, n, better, per in good:
    print('    %s  $%.0f (%+.0f%%), лет %d/5' % (lbl, m, 100 * (m / m0 - 1), better), flush=True)
pickle.dump([x[1] for x in good], open('money_pass.pkl', 'wb'))
