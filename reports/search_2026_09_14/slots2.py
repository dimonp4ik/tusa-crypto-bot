"""Two positions on one coin: the only clean finding from the slot test, checked properly.

The previous run produced two numbers that were artefacts of the harness and are retracted:
evicting an open position erased its history rather than closing it, and shortening the lock freed
the capital early while still collecting the 48-hour outcome. Both read the future.

What survives needs no such trick. Today a coin holds one position; the second signal on a busy coin
is dropped, and 121 signals are dropped that way over four years. Their average is +0.1180R against
+0.1040R for the ones taken, so the lock is discarding the better half. Letting the coin hold two
positions takes them, with real entries and real exits and nothing rewritten.

The cost is concentration: two simultaneous positions on one coin move together, and this account is
already known to lose in clusters. So the drawdown is the test, not the money - measured on two
books, year by year, and over the nine forward half-years.
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
import live_rules_sim as L
import nogate_attack as NA

cost = collections.defaultdict(list)
for r in csv.DictReader(open('book_frozen.csv')):
    cost[r['coin'] + 'USDT'].append(float(r['cost109']))
SP.SLIP_REAL.update({k: float(np.median(v)) for k, v in cost.items()})
G.FC.clear()
ALL = list(G.ALL)
NOBAD = [s for s in ALL if s not in ('AAVEUSDT', 'XLMUSDT')]
YT = {y: int(datetime.datetime(y, 1, 1, tzinfo=datetime.UTC).timestamp()) for y in range(2022, 2028)}
PB = [('rsi48', '>=', 56.3761), ('rsi6', '<=', 33.7947)]
SLICE = 182 * 86400


def events(coins, with_rule=True):
    bank = SW.run(3.0, 1.0, coins=coins, with_meta=True)
    ev = [(x[0], x[1], x[2], x[3], x[4], 'банк') for x in bank]
    if with_rule:
        ev += [(x[0], x[1], x[2], x[3], x[4], 'откат')
               for x in NA.trades('без режима', PB, coins=coins)]
    ev.sort()
    return ev


def run(ev, per_coin=1, same_rule_only=False, min_gap_h=0):
    """A coin may hold `per_coin` positions at once. Nothing is erased or shortened."""
    live = collections.defaultdict(list)
    out, dropped = [], []
    for a, b, R, sf, s, tag in ev:
        live[s] = [p for p in live[s] if p[0] > a]
        if len(live[s]) >= per_coin:
            dropped.append((a, R, tag))
            continue
        if same_rule_only and live[s] and any(p[1] == tag for p in live[s]):
            dropped.append((a, R, tag))
            continue
        if min_gap_h and live[s] and any(a - p[2] < min_gap_h * 3600 for p in live[s]):
            dropped.append((a, R, tag))
            continue
        live[s].append((b, tag, a))
        out.append((a, b, R, sf, s, 1.0))
    out.sort()
    return out, dropped


def report(lbl, rows, ref=None):
    k, m = sim_w.money_at_dd(rows, 0.12)
    v = np.array([x[2] for x in rows])
    per = []
    for y in range(2022, 2027):
        sub = [x for x in rows if YT[y] <= x[0] < YT[y + 1]]
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
        wf = ' | R %d/%d просадка %d/%d' % (wr, seen, wd, seen)
    print('  %-30s %4d сд. ВР%5.1f%% риск %.3f%% $%6.0f | %s%s'
          % (lbl, len(v), 100 * np.mean(v > 0), 100 * k, m,
             ' '.join('%3.0f' % x if np.isfinite(x) else '  -' for x in per), wf), flush=True)
    return m


for cl, coins in (('без AAVE и XLM', NOBAD), ('все 15 монет', ALL)):
    ev = events(coins)
    print('', flush=True)
    print('  ===== книга: %s =====' % cl, flush=True)
    base, drop1 = run(ev, 1)
    m1 = report('1 позиция на монету', base)
    two, drop2 = run(ev, 2)
    report('2 позиции на монету', two, base)
    report('2, но только от РАЗНЫХ правил', run(ev, 2, same_rule_only=True)[0], base)
    for gap in (4, 12, 24):
        report('2, вторая не раньше чем через %2dч' % gap,
               run(ev, 2, min_gap_h=gap)[0], base)
    v = np.array([r for _, r, _ in drop1])
    print('    отброшено замком %d сигналов, их ср%+.4fR; из них банка %d, отката %d'
          % (len(drop1), v.mean(), sum(1 for _, _, t in drop1 if t == 'банк'),
             sum(1 for _, _, t in drop1 if t == 'откат')), flush=True)

print('', flush=True)
print('  === и то же самое БЕЗ правила отката, чтобы отделить одно от другого ===', flush=True)
ev = events(NOBAD, with_rule=False)
base = run(ev, 1)[0]
report('банк один, 1 позиция', base)
report('банк один, 2 позиции', run(ev, 2)[0], base)
