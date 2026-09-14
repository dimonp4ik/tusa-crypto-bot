"""Do rejected candidates work in pairs, when neither works alone?

Everything was added to the account one rule at a time and judged on what it did alone. That misses
the case the user is asking about: two rules that each fall short on their own but take different
trades at different times, so together they fill the account's empty slots without doubling its
losses.

The danger is obvious - there are hundreds of pairs and the best of them will look good by
construction. So this is only the first half. It measures every pair directly, and the second half
(combos_blind.py) re-selects the whole combination on four years and reads the fifth once.

Base is what stands: bank, measured costs, no AAVE or XLM, plus the gateless pullback rule.
"""
import collections
import csv
import datetime
import itertools
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
YT = {y: int(datetime.datetime(y, 1, 1, tzinfo=datetime.UTC).timestamp()) for y in range(2022, 2028)}
PB = [('rsi48', '>=', 56.3761), ('rsi6', '<=', 33.7947)]

# every candidate that ever reached the money test today, whatever happened to it afterwards
EXTRA = [
    ('ПАДЕНИЕ', True, [('ret24', '<=', -1.5434), ('vol720', '>=', 1.4462)]),
    ('КАПИТУЛЯЦИЯ', True, [('breadth', '<=', 0.0625), ('vol720', '>=', 1.4462)]),
    ('РАСХОЖДЕНИЕ', False, [('btc168', '<=', -0.0558), ('pos48', '>=', 0.8794)]),
    ('ЭЙФОРИЯ', False, [('breadth', '>=', 1.0), ('vol168', '>=', 1.3947)]),
    ('ОТКАТ-72ч', True, [('rsiX72', '>=', 55.28), ('rsi6', '<=', 33.7947)]),
]
PASS = pickle.load(open('gauntlet_pass_full.pkl', 'rb'))
CAND = [(l, s, c) for l, s, c in PASS] + EXTRA
print('  кандидатов: %d' % len(CAND), flush=True)


def _rsi(close, n):
    d = np.diff(close, prepend=close[0])
    up = np.where(d > 0, d, 0.0)
    dn = np.where(d < 0, -d, 0.0)
    au = np.full(len(close), np.nan)
    ad = np.full(len(close), np.nan)
    au[n] = up[1:n + 1].mean()
    ad[n] = dn[1:n + 1].mean()
    for i in range(n + 1, len(close)):
        au[i] = (au[i - 1] * (n - 1) + up[i]) / n
        ad[i] = (ad[i - 1] * (n - 1) + dn[i]) / n
    return 100 - 100 / (1 + au / np.where(ad == 0, 1e-12, ad))


for _s in COINS:
    _T1, _f = G.feats(_s)
    if 'rsiX72' not in _f:
        _f['rsiX72'] = _rsi(np.asarray(SP.CTX[_s]['B1'][:, 3], dtype=float), 72)

bank = SW.run(3.0, 1.0, coins=COINS, with_meta=True)
pull = NA.trades('без режима', PB, coins=COINS)
TR = {}
for lbl, side, conds in CAND:
    tr = NA.trades('без режима', conds, lg=side, coins=COINS)
    if len(tr) >= 60:
        TR[lbl] = [(x[0], x[1], x[2], x[3], x[4]) for x in tr]
print('  с достаточным числом сделок: %d' % len(TR), flush=True)


def acc(extra_lists):
    ev = [(x[0], x[1], x[2], x[3], x[4], 0) for x in bank]
    ev += [(x[0], x[1], x[2], x[3], x[4], 1) for x in pull]
    for i, lst in enumerate(extra_lists):
        ev += [(x[0], x[1], x[2], x[3], x[4], 2 + i) for x in lst]
    ev.sort(key=lambda z: (z[0], z[5]))
    busy, out = {}, []
    for p, q, R, sf, s, k in ev:
        if busy.get(s, 0) > p:
            continue
        busy[s] = q
        out.append((p, q, R, sf, s, 1.0))
    out.sort()
    return out


def money(rows, want=0.12):
    return sim_w.money_at_dd(rows, want)[1]


M0 = money(acc([]))
print('  опора (банк + откат): $%.0f' % M0, flush=True)

solo = {}
for lbl, lst in TR.items():
    solo[lbl] = money(acc([lst]))
order = sorted(solo, key=lambda x: -solo[x])
print('', flush=True)
print('  === поодиночке (напоминание) ===', flush=True)
for lbl in order[:12]:
    print('    %-46s $%6.0f (%+.0f%%)' % (lbl, solo[lbl], 100 * (solo[lbl] / M0 - 1)), flush=True)

print('', flush=True)
print('  === ПАРЫ: только те, где ОБА поодиночке не дотянули до +5%% ===', flush=True)
weak = [l for l in TR if solo[l] < M0 * 1.05]
print('  слабых поодиночке: %d, пар из них: %d'
      % (len(weak), len(weak) * (len(weak) - 1) // 2), flush=True)
res = []
for a, b in itertools.combinations(weak, 2):
    m = money(acc([TR[a], TR[b]]))
    res.append((m, a, b))
res.sort(reverse=True)
for m, a, b in res[:15]:
    best_solo = max(solo[a], solo[b])
    print('    $%6.0f (%+5.0f%% к опоре, %+5.0f%% к лучшему из двоих)  %s  +  %s'
          % (m, 100 * (m / M0 - 1), 100 * (m / best_solo - 1), a, b), flush=True)
print('    ... всего пар лучше опоры на 5%%+: %d из %d'
      % (sum(1 for m, _, _ in res if m > M0 * 1.05), len(res)), flush=True)

print('', flush=True)
print('  === и пары среди ВСЕХ кандидатов, для сравнения ===', flush=True)
res2 = []
for a, b in itertools.combinations(list(TR), 2):
    m = money(acc([TR[a], TR[b]]))
    res2.append((m, a, b))
res2.sort(reverse=True)
for m, a, b in res2[:10]:
    print('    $%6.0f (%+5.0f%%)  %s  +  %s' % (m, 100 * (m / M0 - 1), a, b), flush=True)
pickle.dump([(m, a, b) for m, a, b in res2], open('pairs_money.pkl', 'wb'))
