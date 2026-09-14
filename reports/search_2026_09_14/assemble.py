"""Everything that survived today, assembled into one account and measured end to end.

The pieces were each validated alone. Together they interact - a coin removed frees slots, a rule
without its gate takes hours the bank wanted, a spread gate refuses entries that would have blocked
a coin for two days. So the assembly is measured as one thing, one piece at a time, in the order a
person would actually apply them:

    1. measured fill costs               (a correction, not a choice)
    2. drop the worst coin by trailing win rate, recomputed each half-year   (causal, no lookahead)
    3. the pullback rule inside the bank's gate                              (yesterday's proposal)
    4. the pullback rule with the gate removed                               (today's)

Each step is judged on money at equal drawdown, on each year as its own account, and on the nine
forward half-years against the bank. A step that fails any of the three does not go in.
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

meas = collections.defaultdict(list)
spread = collections.defaultdict(list)
for r in csv.DictReader(open('book_samples.csv')):
    meas[r['coin'] + 'USDT'].append(float(r['cost109']))
    spread[r['coin'] + 'USDT'].append(float(r['spread']))
SP.SLIP_REAL.update({k: float(np.median(v)) for k, v in meas.items()})
G.FC.clear()
MED_SPREAD = {k: float(np.median(v)) for k, v in spread.items()}

C = [('rsi48', '>=', 56.3761), ('rsi6', '<=', 33.7947)]
ALL = list(G.ALL)
YT = {y: int(datetime.datetime(y, 1, 1, tzinfo=datetime.UTC).timestamp()) for y in range(2022, 2028)}
SLICE = 182 * 86400

BANK = {s: SW.run(3.0, 1.0, coins=[s], with_meta=True) for s in ALL}
CAND = {mode: {s: NA.trades(mode, C, coins=[s]) for s in ALL} for mode in ('гейт', 'без режима')}
print('  готово: банк %d сделок, откат в гейте %d, без гейта %d'
      % (sum(len(v) for v in BANK.values()),
         sum(len(v) for v in CAND['гейт'].values()),
         sum(len(v) for v in CAND['без режима'].values())), flush=True)


def build(mode=None, screen=False, spread_gate=False):
    coins = [s for s in ALL if not (spread_gate and MED_SPREAD.get(s, 0) > 0.0005)]
    ev = []
    for s in coins:
        ev += [(a, b, R, sf, s, 0) for a, b, R, sf, s2 in BANK[s]]
        if mode:
            ev += [(a, b, R, sf, s, 1) for a, b, R, sf, s2 in CAND[mode][s]]
    ev.sort(key=lambda x: (x[0], x[5]))
    if not ev:
        return []
    first = min(x[0] for x in ev)
    drop = {}
    if screen:
        t = first
        last = max(x[0] for x in ev)
        while t < last:
            per = collections.defaultdict(list)
            for a, b, R, sf, s, k in ev:
                if a < t:
                    per[s].append(R)
            sc = [(-float(np.mean(np.array(v) > 0)), s) for s, v in per.items() if len(v) >= 40]
            sc.sort(reverse=True)
            drop[t] = {s for _, s in sc[:1]}
            t += SLICE
    busy, out = {}, []
    for a, b, R, sf, s, k in ev:
        if screen:
            key = first + ((a - first) // SLICE) * SLICE
            if s in drop.get(key, ()):
                continue
        if busy.get(s, 0) > a:
            continue
        busy[s] = b
        out.append((a, b, R, sf, s, 1.0))
    out.sort()
    return out


def report(lbl, rows, ref=None):
    k, m = sim_w.money_at_dd(rows, 0.12)
    per = []
    for y in range(2022, 2027):
        sub = [r for r in rows if YT[y] <= r[0] < YT[y + 1]]
        per.append(sim_w.money_at_dd(sub, 0.12)[1] if len(sub) >= 50 else float('nan'))
    wf = ''
    if ref is not None:
        b3 = [(a, b, R) for a, b, R, sf, s, w in ref]
        c3 = [(a, b, R) for a, b, R, sf, s, w in rows]
        t, wr, wd, seen = min(r[0] for r in b3), 0, 0, 0
        last = max(r[0] for r in b3)
        while t < last:
            s0, s1 = SP.stats(b3, lo=t, hi=t + SLICE), SP.stats(c3, lo=t, hi=t + SLICE)
            if s0 and s1:
                seen += 1
                wr += int(s1['r_mo'] > s0['r_mo'])
                wd += int(s1['dd'] > s0['dd'])
            t += SLICE
        wf = ' | прогон R %d/%d просадка %d/%d' % (wr, seen, wd, seen)
    print('  %-42s %4d сд. риск %.3f%% $%6.0f | %s%s'
          % (lbl, len(rows), 100 * k, m,
             ' '.join('%3.0f' % x if np.isfinite(x) else '  -' for x in per), wf), flush=True)
    return m, per


print('', flush=True)
print('  вариант                                   сделок  риск    деньги | по годам 22 23 24 25 26',
      flush=True)
base = build()
m0, p0 = report('1. банк, измеренные издержки', base)
m1, p1 = report('2. + отсечка спреда (нет XLM, AAVE)', build(spread_gate=True), base)
m2, p2 = report('3. + откат в гейте', build('гейт', spread_gate=True), base)
m3, p3 = report('4. + откат БЕЗ гейта', build('без режима', spread_gate=True), base)
m4, p4 = m3, p3
print('', flush=True)
print('  для сравнения, отвергнутый отбор монеты:', flush=True)
report('   банк + отбор', build(screen=True), base)
report('   полная сборка + отбор', build('без режима', screen=True, spread_gate=True), base)

print('', flush=True)
print('  === каждый шаг против предыдущего, по годам ===', flush=True)
names = ['банк', '+отсечка спреда', '+откат в гейте', '+откат без гейта']
prev = p0
for nm, cur in zip(names[1:], [p1, p2, p3]):
    better = sum(1 for a, b in zip(prev, cur) if np.isfinite(a) and np.isfinite(b) and b > a)
    tot = sum(1 for a, b in zip(prev, cur) if np.isfinite(a) and np.isfinite(b))
    print('    %-20s лучше предыдущего в %d годах из %d  %s'
          % (nm, better, tot, ' '.join('%+4.0f' % (b - a) if np.isfinite(a) and np.isfinite(b)
                                       else '   -' for a, b in zip(prev, cur))), flush=True)
    prev = cur

print('', flush=True)
print('  === итог против банка ===', flush=True)
print('    банк %.0f -> полная сборка %.0f, во сколько раз: %.2f' % (m0, m3, m3 / m0), flush=True)
print('    по годам банк:   %s' % ' '.join('%3.0f' % x for x in p0), flush=True)
print('    по годам сборка: %s' % ' '.join('%3.0f' % x for x in p3), flush=True)
