"""The three-day strength horizon, attacked from scratch.

Changing the slow side from 48 hours to 72 changes which trades the rule takes, so nothing it passed
before carries. It arrives with one thing already in its favour that today's two dead candidates
never had: it beats the incumbent in all eight books, not just the one it was found in.

Everything again - the null, each condition alone, the inverse, the mirror, the years, the coins,
the spread across months, the threshold grid, and the blind-year re-pick of both thresholds.
"""
import collections
import datetime
import sys

import numpy as np

sys.path.insert(0, 'C:/Users/Lenovo/Desktop/Торговля/crypto-bot')
import struct_params as SP
import gauntlet2 as G
import stop_width as SW
import sim_w
import nogate_attack as NA
import slow_horizon as SH
import full_money as FM

COINS = FM.COINS
YT = FM.YT
FLIP = {'>=': '<=', '<=': '>='}
A, B = 55.28, 33.7947          # rsiX72 at the same quantile the working rule uses
C = [('rsiX72', '>=', A), ('rsi6', '<=', B)]


def st(conds, lg, lab):
    tr = NA.trades('без режима', conds, lg=lg, coins=COINS)
    v = np.array([x[2] for x in tr]) if tr else np.zeros(0)
    if not len(v):
        print('    %-40s сделок нет' % lab, flush=True)
        return None
    print('    %-40s n%5d ВР%5.1f%% ср%+.4fR' % (lab, len(v), 100 * np.mean(v > 0), v.mean()),
          flush=True)
    return tr


print('  === атаки на 72 часа ===', flush=True)
st([], True, 'весь поток LONG без гейта (ноль)')
base = st(C, True, 'ЦЕЛИКОМ rsiX72>=%.2f + rsi6<=%.2f' % (A, B))
st([C[0]], True, 'только rsiX72>=%.2f' % A)
st([C[1]], True, 'только rsi6<=%.2f' % B)
st([(n, FLIP[o], t) for n, o, t in C], True, 'ИНВЕРСИЯ (должна терять)')
st(C, False, 'ЗЕРКАЛО SHORT (должно терять)')

print('', flush=True)
print('  по годам:', flush=True)
for y in range(2022, 2027):
    sub = [x[2] for x in base if YT[y] <= x[0] < YT[y + 1]]
    print('    %d n%4d ВР%5.1f%% ср%+.4f' % (y, len(sub), 100 * np.mean(np.array(sub) > 0),
                                             np.mean(sub)), flush=True)

print('', flush=True)
print('  по монетам:', flush=True)
per = collections.defaultdict(list)
for x in base:
    per[x[4]].append(x[2])
for s, v in sorted(per.items(), key=lambda kv: -np.mean(kv[1])):
    print('    %-6s n%4d ВР%5.1f%% ср%+.4f' % (s.replace('USDT', ''), len(v),
                                               100 * np.mean(np.array(v) > 0), np.mean(v)),
          flush=True)
mons = collections.Counter(
    datetime.datetime.fromtimestamp(x[0], datetime.UTC).strftime('%y-%m') for x in base)
print('    месяцев %d, максимум в одном %d (%.0f%%)'
      % (len(mons), max(mons.values()), 100 * max(mons.values()) / len(base)), flush=True)

print('', flush=True)
print('  === сетка порогов (среднее R / сделок) ===', flush=True)
GA = [49.0, 52.0, 55.28, 58.0, 61.0, 64.0]
GB = [24.0, 29.0, 33.7947, 38.0, 43.0, 48.0]
CACHE = {}
print('    rsiX72 \\ rsi6 %s' % ' '.join('%13.1f' % b for b in GB), flush=True)
for a in GA:
    cells = []
    for b in GB:
        tr = NA.trades('без режима', [('rsiX72', '>=', a), ('rsi6', '<=', b)], coins=COINS)
        CACHE[(a, b)] = tr
        v = np.array([x[2] for x in tr]) if tr else np.zeros(0)
        cells.append('%+.4f/%4d' % (v.mean(), len(v)) if len(v) else '      -      ')
    print('    %13.2f %s' % (a, ' '.join(cells)), flush=True)

print('', flush=True)
print('  === слепой переподбор обоих порогов ===', flush=True)
ok = tot = 0
for hold in range(2022, 2027):
    lo, hi = YT[hold], YT[hold + 1]
    bk, bv = None, None
    for k, tr in CACHE.items():
        sub = [x[2] for x in tr if not (lo <= x[0] < hi)]
        if len(sub) < 200:
            continue
        v = float(np.mean(sub))
        if bv is None or v > bv:
            bv, bk = v, k
    ho = [x[2] for x in CACHE[bk] if lo <= x[0] < hi]
    cur = [x[2] for x in CACHE[(55.28, 33.7947)] if lo <= x[0] < hi]
    if len(ho) < 15:
        print('    %d  выбрано %.1f/%.1f — мало (n%d)' % (hold, bk[0], bk[1], len(ho)), flush=True)
        continue
    tot += 1
    ok += int(np.mean(ho) > 0)
    print('    %d  выбрано rsiX72>=%.1f rsi6<=%.1f | слепой год: выбор%+.4f (n%d), рабочее%+.4f (n%d)'
          % (hold, bk[0], bk[1], np.mean(ho), len(ho), np.mean(cur), len(cur)), flush=True)
print('    слепых лет в плюсе: %d из %d' % (ok, tot), flush=True)

print('', flush=True)
print('  === деньги и скользящий прогон ===', flush=True)
bank = SW.run(3.0, 1.0, coins=COINS, with_meta=True)


def acc(cand):
    ev = [(x[0], x[1], x[2], x[3], x[4], 0) for x in bank]
    ev += [(x[0], x[1], x[2], x[3], x[4], 1) for x in cand]
    ev.sort(key=lambda z: (z[0], z[5]))
    busy, out = {}, []
    for p, q, R, sf, s, k in ev:
        if busy.get(s, 0) > p:
            continue
        busy[s] = q
        out.append((p, q, R, sf, s, 1.0))
    out.sort()
    return out


ref = acc([])
r48 = acc(NA.trades('без режима', [('rsi48', '>=', 56.3761), ('rsi6', '<=', 33.7947)], coins=COINS))
r72 = acc(base)
for lbl, rows in (('банк один', ref), ('откат 48ч', r48), ('откат 72ч', r72)):
    m = sim_w.money_at_dd(rows, 0.12)[1]
    per_ = [sim_w.money_at_dd([r for r in rows if YT[y] <= r[0] < YT[y + 1]], 0.12)[1]
            for y in range(2022, 2027)]
    b3 = [(a, b, R) for a, b, R, sf, s, w in ref]
    c3 = [(a, b, R) for a, b, R, sf, s, w in rows]
    t, wr, wd, seen = min(r[0] for r in b3), 0, 0, 0
    last = max(r[0] for r in b3)
    while t < last:
        s0, s1 = SP.stats(b3, lo=t, hi=t + 182 * 86400), SP.stats(c3, lo=t, hi=t + 182 * 86400)
        if s0 and s1:
            seen += 1
            wr += int(s1['r_mo'] > s0['r_mo'])
            wd += int(s1['dd'] > s0['dd'])
        t += 182 * 86400
    print('    %-12s %4d сделок $%6.0f | по годам %s | R %d/%d просадка %d/%d'
          % (lbl, len(rows), m, ' '.join('%3.0f' % x for x in per_), wr, seen, wd, seen), flush=True)
