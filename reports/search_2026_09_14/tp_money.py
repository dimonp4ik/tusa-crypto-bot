"""The target, measured in money rather than in R.

Mean R cannot be compared across stop widths: R is the return divided by the trade's own stop, so a
tighter stop is a larger position at the same risk - a leverage knob, and the project has been
fooled by exactly that before. At a FIXED stop the target axis is honest in R, but even there the
win rate moves, and the account does not spend win rate.

So: the stop stays at 3.0 where the bank put it, the target sweeps, and the answer is the money the
whole account makes at equal drawdown - with the win rate printed beside it, because the user asked
for both and they pull in opposite directions.
"""
import collections
import datetime
import sys

import numpy as np

sys.path.insert(0, 'C:/Users/Lenovo/Desktop/Торговля/crypto-bot')
import gauntlet2 as G
import struct_params as SP
import stop_width as SW
import margin_cap as MC
import live_rules_sim as L

RULES = [
    ('ОТКАТ', [('rsi48', '>=', 56.3761), ('rsi6', '<=', 33.7947)], True),
    ('КАПИТУЛЯЦИЯ', [('breadth', '<=', 0.0625), ('vol720', '>=', 1.4462)], True),
]
TPS = [1.0, 1.3, 1.6, 2.0, 2.5]
bank = SW.run(3.0, 1.0, coins=G.ALL, with_meta=True)
SIG = {(fam, tp): G.simulate(c, lg, tp=tp, sl=3.0)[0]
       for fam, c, lg in RULES for tp in TPS}


def merge(pairs):
    ev = [(a, b, R, sf, s, 'bank') for a, b, R, sf, s in bank]
    for i, (fam, tp) in enumerate(pairs):
        ev += [(a, b, R, sf, s, fam) for a, b, R, sf, s in SIG[(fam, tp)]]
    pri = {'bank': 0}
    pri.update({fam: i + 1 for i, (fam, tp) in enumerate(pairs)})
    ev.sort(key=lambda x: (x[0], pri[x[5]]))
    busy, out = {}, []
    for a, b, R, sf, s, tag in ev:
        if busy.get(s, 0) > a:
            continue
        busy[s] = b
        out.append((a, b, R, sf, s, tag))
    out.sort()
    return out


def money_at_dd(rows, want=0.12):
    MC.trades = [(a, b, R, sf, s) for a, b, R, sf, s, _ in rows]
    lo, hi = 0.002, 0.030
    for _ in range(26):
        mid = (lo + hi) / 2
        r = L.simulate(target=mid)
        if bool(r['paused_at']) or abs(L.dd_of(r['curve'])) > want:
            hi = mid
        else:
            lo = mid
    r = L.simulate(target=lo)
    return lo, r['eq']


print('  === тейк каждого правила поодиночке, поверх банка, при равной просадке 12%% ===',
      flush=True)
bk = merge([])
k0, m0 = money_at_dd(bk)
v0 = np.array([r[2] for r in bk])
print('    банк один: сделок %d ВР %.1f%% риск %.3f%% $%.0f'
      % (len(bk), 100 * np.mean(v0 > 0), 100 * k0, m0), flush=True)
for fam, _, _ in RULES:
    for tp in TPS:
        rows = merge([(fam, tp)])
        k, m = money_at_dd(rows)
        own = np.array([r[2] for r in SIG[(fam, tp)]])
        allv = np.array([r[2] for r in rows])
        print('    %-12s тейк %.1f: своих %3d ВР %.0f%% ср%+.4f | счёт: ВР %.1f%% риск %.3f%% '
              '$%6.0f (%+.0f%% к банку)'
              % (fam, tp, len(own), 100 * np.mean(own > 0), own.mean(),
                 100 * np.mean(allv > 0), 100 * k, m, 100 * (m / m0 - 1)), flush=True)

print('', flush=True)
print('  === оба вместе ===', flush=True)
BEST = {}
for a in TPS:
    for b in TPS:
        rows = merge([('ОТКАТ', a), ('КАПИТУЛЯЦИЯ', b)])
        k, m = money_at_dd(rows)
        BEST[(a, b)] = (m, k, rows)
print('    откат\капит %s' % ' '.join('%8.1f' % t for t in TPS), flush=True)
for a in TPS:
    print('    %10.1f  %s' % (a, ' '.join('%8.0f' % BEST[(a, b)][0] for b in TPS)), flush=True)

ba = max(BEST, key=lambda k: BEST[k][0])
m, k, rows = BEST[ba]
v = np.array([r[2] for r in rows])
print('    лучшая клетка: откат %.1f, капитуляция %.1f -> $%.0f при риске %.3f%%, '
      'ВР счёта %.1f%%, сделок %d' % (ba[0], ba[1], m, 100 * k, 100 * np.mean(v > 0), len(rows)),
      flush=True)

print('', flush=True)
print('  === та же клетка, но выбранная по четырём годам и замеренная на пятом ===', flush=True)
YT = {y: int(datetime.datetime(y, 1, 1, tzinfo=datetime.UTC).timestamp())
      for y in range(2022, 2028)}
wins = tot = 0
for hold in range(2022, 2027):
    lo_, hi_ = YT[hold], YT[hold + 1]
    best, bk2 = None, None
    for key, (m_, k_, rows_) in BEST.items():
        tr = [r[2] for r in rows_ if not (lo_ <= r[0] < hi_)]
        v_ = float(np.mean(tr))
        if best is None or v_ > best:
            best, bk2 = v_, key
    ho = [r[2] for r in BEST[bk2][2] if lo_ <= r[0] < hi_]
    cur = [r[2] for r in BEST[(1.0, 1.0)][2] if lo_ <= r[0] < hi_]
    tot += 1
    wins += int(np.mean(ho) > np.mean(cur))
    print('    %d  выбрано откат %.1f капит %.1f | НЕВИДАННЫЙ: %+.4f против нынешних 1.0/1.0 %+.4f'
          % (hold, bk2[0], bk2[1], np.mean(ho), np.mean(cur)), flush=True)
print('    шире тейк лучше нынешнего в %d годах из %d' % (wins, tot), flush=True)

print('', flush=True)
print('  === скользящий прогон лучшей клетки против банка ===', flush=True)
b3 = [(a, b, R) for a, b, R, sf, s, t in bk]
c3 = [(a, b, R) for a, b, R, sf, s, t in rows]
first, last = min(r[0] for r in b3), max(r[0] for r in b3)
t, wr, wd, seen = first, 0, 0, 0
while t < last:
    a, b = t, t + 182 * 86400
    s0, s1 = SP.stats(b3, lo=a, hi=b), SP.stats(c3, lo=a, hi=b)
    if s0 and s1:
        seen += 1
        wr += int(s1['r_mo'] > s0['r_mo'])
        wd += int(s1['dd'] > s0['dd'])
        print('      %s  банк%+6.2f связка%+6.2f | просадка банк%+6.1f связка%+6.1f'
              % (datetime.datetime.fromtimestamp(a, datetime.UTC).strftime('%y-%m'),
                 s0['r_mo'], s1['r_mo'], s0['dd'], s1['dd']), flush=True)
    t = b
print('    связка лучше: R/мес %d/%d, просадка %d/%d' % (wr, seen, wd, seen), flush=True)
