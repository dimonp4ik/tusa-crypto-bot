"""The combination itself chosen blind: assembled on four years, read on the fifth.

Measuring every pair finds the best pair, and with seven hundred pairs the best one is worth nothing
on its own. The question that matters is different: if a person had built the best combination they
could from four years of data, would it have helped them in the year they had not seen?

So the whole selection procedure is run inside each fold. Greedy forward selection - add whichever
candidate most improves the money, repeat - on the four training years, then the chosen set is
applied to the held-out year once. Five folds, five held-out years.

If combinations carry real information, the blind years should be better than the base more often
than not, and the chosen sets should have something in common. If it is selection noise, the blind
years will hover around the base and each fold will choose a different set.
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
PB = [('rsi48', '>=', 56.3761), ('rsi6', '<=', 33.7947)]
MAX_ADD = 4


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

EXTRA = [
    ('ПАДЕНИЕ', True, [('ret24', '<=', -1.5434), ('vol720', '>=', 1.4462)]),
    ('КАПИТУЛЯЦИЯ', True, [('breadth', '<=', 0.0625), ('vol720', '>=', 1.4462)]),
    ('РАСХОЖДЕНИЕ', False, [('btc168', '<=', -0.0558), ('pos48', '>=', 0.8794)]),
    ('ЭЙФОРИЯ', False, [('breadth', '>=', 1.0), ('vol168', '>=', 1.3947)]),
    ('ОТКАТ-72ч', True, [('rsiX72', '>=', 55.28), ('rsi6', '<=', 33.7947)]),
]
CAND = [(l, s, c) for l, s, c in pickle.load(open('gauntlet_pass_full.pkl', 'rb'))] + EXTRA
bank = SW.run(3.0, 1.0, coins=COINS, with_meta=True)
pull = NA.trades('без режима', PB, coins=COINS)
TR = {}
for lbl, side, conds in CAND:
    tr = NA.trades('без режима', conds, lg=side, coins=COINS)
    if len(tr) >= 60:
        TR[lbl] = [(x[0], x[1], x[2], x[3], x[4]) for x in tr]
NAMES = sorted(TR)
print('  кандидатов с достаточным числом сделок: %d' % len(NAMES), flush=True)


def acc(names):
    ev = [(x[0], x[1], x[2], x[3], x[4], 0) for x in bank]
    ev += [(x[0], x[1], x[2], x[3], x[4], 1) for x in pull]
    for i, nm in enumerate(names):
        ev += [(x[0], x[1], x[2], x[3], x[4], 2 + i) for x in TR[nm]]
    ev.sort(key=lambda z: (z[0], z[5]))
    busy, out = {}, []
    for p, q, R, sf, s, k in ev:
        if busy.get(s, 0) > p:
            continue
        busy[s] = q
        out.append((p, q, R, sf, s, 1.0))
    out.sort()
    return out


def money(rows, keep):
    sub = [r for r in rows if keep(r[0])]
    if len(sub) < 80:
        return 0.0
    return sim_w.money_at_dd(sub, 0.12)[1]


print('', flush=True)
print('  === жадный отбор на четырёх годах, замер на пятом ===', flush=True)
wins = 0
seen = 0
chosen_all = collections.Counter()
for hold in YEARS:
    lo, hi = YT[hold], YT[hold + 1]
    train = lambda t: not (lo <= t < hi)
    blind = lambda t: lo <= t < hi
    cur = []
    best_train = money(acc(cur), train)
    while len(cur) < MAX_ADD:
        gain, pick = 0.0, None
        for nm in NAMES:
            if nm in cur:
                continue
            m = money(acc(cur + [nm]), train)
            if m > best_train + gain:
                gain, pick = m - best_train, nm
        if pick is None:
            break
        cur.append(pick)
        best_train += gain
    base_blind = money(acc([]), blind)
    got_blind = money(acc(cur), blind)
    if base_blind <= 0 or got_blind <= 0:
        print('    %d  выбрано %s | слепой год не измерить' % (hold, cur), flush=True)
        continue
    seen += 1
    wins += int(got_blind > base_blind)
    for nm in cur:
        chosen_all[nm] += 1
    print('    %d  обучение $%6.0f (опора $%6.0f) | СЛЕПОЙ ГОД $%5.0f против опоры $%5.0f  %s'
          % (hold, best_train, money(acc([]), train), got_blind, base_blind,
             'лучше' if got_blind > base_blind else 'ХУЖЕ'), flush=True)
    print('        выбрано: %s' % ' | '.join(cur), flush=True)

print('', flush=True)
print('  слепых лет, где комбинация лучше опоры: %d из %d' % (wins, seen), flush=True)
print('  сколько раз каждый кандидат попадал в выбранный набор:', flush=True)
for nm, c in chosen_all.most_common():
    print('    %2d/%d  %s' % (c, seen, nm), flush=True)
