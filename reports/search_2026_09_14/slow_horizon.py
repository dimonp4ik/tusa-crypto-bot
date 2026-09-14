"""Is it the two-day horizon, or is it the number 48?

The strength side of the rule must be rsi48: with rsi14 in that slot the edge collapses to +0.004.
Only one medium horizon exists in the feature set, so "48 is required" and "48 was selected" cannot
be told apart. Adding neighbours settles it. If 24, 36, 72 and 96 also work and 48 is merely the
best of a smooth run, the mechanism is about the horizon; if 48 is a spike among them, the pair was
selected from 26,866 survivors and nothing more.

RSI is computed here the same way the feature file does it, over hourly bars, so the numbers are
comparable to the ones already in use.
"""
import collections
import datetime
import sys

import numpy as np

sys.path.insert(0, 'C:/Users/Lenovo/Desktop/Торговля/crypto-bot')
import struct_params as SP
import gauntlet2 as G
import feat2
import nogate_attack as NA
import full_money as FM
import sim_w
import stop_width as SW
from src import pullback_bank as PB

COINS = FM.COINS
YT = FM.YT
HORIZONS = [12, 18, 24, 36, 48, 60, 72, 96, 144]


def rsi(close, n):
    d = np.diff(close, prepend=close[0])
    up = np.where(d > 0, d, 0.0)
    dn = np.where(d < 0, -d, 0.0)
    au = np.full(len(close), np.nan)
    ad = np.full(len(close), np.nan)
    if len(close) <= n:
        return np.full(len(close), np.nan)
    au[n] = up[1:n + 1].mean()
    ad[n] = dn[1:n + 1].mean()
    for i in range(n + 1, len(close)):
        au[i] = (au[i - 1] * (n - 1) + up[i]) / n
        ad[i] = (ad[i - 1] * (n - 1) + dn[i]) / n
    rs = au / np.where(ad == 0, 1e-12, ad)
    return 100 - 100 / (1 + rs)


# add the extra horizons to the cached feature tables the signal generator reads
for s in COINS:
    T1, f = G.feats(s)
    c = SP.CTX[s]
    close = np.asarray(c['B1'][:, 3], dtype=float)
    for n in HORIZONS:
        key = 'rsiX%d' % n
        if key not in f:
            f[key] = rsi(close, n)
print('  горизонты добавлены: %s' % ', '.join('rsiX%d' % n for n in HORIZONS), flush=True)

# match the working rule's selectivity: same quantiles for both conditions
allv = collections.defaultdict(list)
for s in COINS:
    _, f = G.feats(s)
    for n in HORIZONS:
        v = f['rsiX%d' % n]
        allv[n].append(v[np.isfinite(v)])
    v6 = f['rsi6']
    allv['f'].append(v6[np.isfinite(v6)])
Q = {k: np.concatenate(v) for k, v in allv.items()}
qs = float((np.concatenate([G.feats(s)[1]['rsi48'][np.isfinite(G.feats(s)[1]['rsi48'])]
                            for s in COINS]) <= 56.3761).mean())
qf = float((Q['f'] <= 33.7947).mean())
b = float(np.quantile(Q['f'], qf))
print('  быстрый: rsi6 <= %.2f (квантиль %.3f); медленный берётся на квантиле %.3f'
      % (b, qf, qs), flush=True)
print('', flush=True)
print('  медленный  порог   сделок    ВР      ср R   мес   по годам', flush=True)
bank = SW.run(3.0, 1.0, coins=COINS, with_meta=True)
pull_ref = None
for n in HORIZONS:
    a = float(np.quantile(Q[n], qs))
    tr = NA.trades('без режима', [('rsiX%d' % n, '>=', a), ('rsi6', '<=', b)], coins=COINS)
    if len(tr) < 60:
        print('  rsiX%-6d %6.2f  сделок %d — мало' % (n, a, len(tr)), flush=True)
        continue
    v = np.array([x[2] for x in tr])
    mons = collections.Counter(
        datetime.datetime.fromtimestamp(x[0], datetime.UTC).strftime('%y-%m') for x in tr)
    yr = []
    for y in range(2022, 2027):
        sub = [x[2] for x in tr if YT[y] <= x[0] < YT[y + 1]]
        yr.append('%+.3f' % np.mean(sub) if len(sub) >= 10 else '  -  ')
    ev = [(x[0], x[1], x[2], x[3], x[4], 0) for x in bank]
    ev += [(x[0], x[1], x[2], x[3], x[4], 1) for x in tr]
    ev.sort(key=lambda z: (z[0], z[5]))
    busy, out = {}, []
    for p, q, R, sf, s, k in ev:
        if busy.get(s, 0) > p:
            continue
        busy[s] = q
        out.append((p, q, R, sf, s, 1.0))
    out.sort()
    m = sim_w.money_at_dd(out, 0.12)[1]
    print('  rsiX%-6d %6.2f  %6d %6.1f%% %+9.4f %5d   %s  $%6.0f'
          % (n, a, len(v), 100 * np.mean(v > 0), v.mean(), len(mons), ' '.join(yr), m), flush=True)
