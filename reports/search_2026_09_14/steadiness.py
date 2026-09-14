"""A falsifiable prediction: if the mechanism is steadiness, a different steadiness measure must work.

Every magnitude-based measure of strength fails in the rule's strength slot - return over 48, 96 and
168 hours, EMA slope and gap, range position, relative strength, BTC's own move - while every RSI
horizon from 36 to 144 hours works. The difference between them is what they measure: return says
how FAR price travelled, RSI says how STEADILY, since it is the ratio of average gains to average
losses.

If that reading is right, a steadiness measure with no relationship to RSI must also work. Three are
tried:

    up_share   the fraction of the last N hourly bars that closed up
    dd_share   how little of the window was spent below its own running peak
    path_ratio net move divided by total travelled distance - a straight line scores 1

If these pay and the magnitude measures do not, the mechanism is named correctly. If they do not
pay, "steadiness" is a story told about rsi48 and the honest description is that rsi48 works and
nobody knows why.
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
import nogate_attack as NA

sp = collections.defaultdict(list)
cost = collections.defaultdict(list)
for r in csv.DictReader(open('book_frozen.csv')):
    sp[r['coin'] + 'USDT'].append(float(r['spread']))
    cost[r['coin'] + 'USDT'].append(float(r['cost109']))
SP.SLIP_REAL.update({k: float(np.median(v)) for k, v in cost.items()})
G.FC.clear()
MED = {k: float(np.median(v)) for k, v in sp.items()}
ALL = list(G.ALL)
YT = {y: int(datetime.datetime(y, 1, 1, tzinfo=datetime.UTC).timestamp()) for y in range(2022, 2028)}
N = 48


def build(close):
    n = len(close)
    d = np.diff(close, prepend=close[0])
    up = (d > 0).astype(float)
    up_share = np.full(n, np.nan)
    path = np.full(n, np.nan)
    dd_share = np.full(n, np.nan)
    cs_up = np.cumsum(up)
    cs_abs = np.cumsum(np.abs(d))
    for i in range(N, n):
        up_share[i] = (cs_up[i] - cs_up[i - N]) / N
        travelled = cs_abs[i] - cs_abs[i - N]
        path[i] = (close[i] - close[i - N]) / travelled if travelled > 0 else np.nan
        w = close[i - N:i + 1]
        peak = np.maximum.accumulate(w)
        dd_share[i] = float(np.mean(w >= peak * 0.995))
    return up_share, path, dd_share


for s in ALL:
    _, f = G.feats(s)
    if 'up_share' in f:
        continue
    close = np.asarray(SP.CTX[s]['B1'][:, 3], dtype=float)
    a, b, c = build(close)
    f['up_share'], f['path_ratio'], f['dd_share'] = a, b, c
print('  меры ровности посчитаны на окне %d часов' % N, flush=True)

vals = collections.defaultdict(list)
for s in ALL:
    _, f = G.feats(s)
    for n in ('rsi48', 'up_share', 'path_ratio', 'dd_share', 'ret48'):
        v = f[n]
        vals[n].append(v[np.isfinite(v)])
Q = {k: np.concatenate(v) for k, v in vals.items()}
q = float((Q['rsi48'] <= 56.3761).mean())
FAST = ('rsi6', '<=', 33.7947)
print('  квантиль условия силы %.3f' % q, flush=True)
print('', flush=True)

BOOKS = [('все 15', ALL, 'без режима'),
         ('без AAVE и XLM', [s for s in ALL if s not in ('AAVEUSDT', 'XLMUSDT')], 'без режима'),
         ('все 15, в гейте', ALL, 'гейт'),
         ('без AAVE и XLM, в гейте',
          [s for s in ALL if s not in ('AAVEUSDT', 'XLMUSDT')], 'гейт')]
BANKS = {}
for lbl, coins, mode in BOOKS:
    BANKS.setdefault(tuple(coins), SW.run(3.0, 1.0, coins=coins, with_meta=True))


def money(coins, mode, conds):
    bank = BANKS[tuple(coins)]
    cand = NA.trades(mode, conds, coins=coins)
    ev = [(x[0], x[1], x[2], x[3], x[4], 0) for x in bank]
    ev += [(x[0], x[1], x[2], x[3], x[4], 1) for x in cand]
    ev.sort(key=lambda z: (z[0], z[5]))
    busy, out = {}, []
    for p, qq, R, sf, s, k in ev:
        if busy.get(s, 0) > p:
            continue
        busy[s] = qq
        out.append((p, qq, R, sf, s, 1.0))
    out.sort()
    return sim_w.money_at_dd(out, 0.12)[1], cand


print('  мера          порог    сделок    ВР      ср R    по годам                       деньги',
      flush=True)
base_cells = None
for nm in ('rsi48', 'up_share', 'path_ratio', 'dd_share', 'ret48'):
    thr = float(np.quantile(Q[nm], q))
    conds = [(nm, '>=', thr), FAST]
    cells, first = [], None
    for lbl, coins, mode in BOOKS:
        m, cand = money(coins, mode, conds)
        cells.append(m)
        if first is None:
            first = cand
    if nm == 'rsi48':
        base_cells = cells
    if not first or len(first) < 60:
        print('  %-12s %7.4f  сделок %d — мало' % (nm, thr, len(first) if first else 0), flush=True)
        continue
    v = np.array([x[2] for x in first])
    yr = []
    for y in range(2022, 2027):
        sub = [x[2] for x in first if YT[y] <= x[0] < YT[y + 1]]
        yr.append('%+.3f' % np.mean(sub) if len(sub) >= 10 else '  -  ')
    w = sum(1 for c, b in zip(cells, base_cells)) if base_cells else 0
    print('  %-12s %7.4f  %6d %6.1f%% %+7.4f  %s  %s'
          % (nm, thr, len(v), 100 * np.mean(v > 0), v.mean(), ' '.join(yr),
             ' '.join('$%5.0f' % c for c in cells)), flush=True)

print('', flush=True)
print('  === и мера ровности ВМЕСТЕ с rsi48 (проверка, добавляет ли она что-то) ===', flush=True)
for nm in ('up_share', 'path_ratio', 'dd_share'):
    thr = float(np.quantile(Q[nm], 0.5))
    conds = [('rsi48', '>=', 56.3761), (nm, '>=', thr), FAST]
    cells = []
    first = None
    for lbl, coins, mode in BOOKS:
        m, cand = money(coins, mode, conds)
        cells.append(m)
        if first is None:
            first = cand
    v = np.array([x[2] for x in first]) if first else np.zeros(0)
    if not len(v):
        continue
    print('  rsi48 + %-10s>=%.4f  %5d сделок ср%+.4f | %s | лучше базовой в %d книгах из 4'
          % (nm, thr, len(v), v.mean(), ' '.join('$%5.0f' % c for c in cells),
             sum(1 for c, b in zip(cells, base_cells) if c > b)), flush=True)
