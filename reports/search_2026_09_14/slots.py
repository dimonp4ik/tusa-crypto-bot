"""The account is limited by slots, not by signals. So how are the slots being spent?

The pair test said something structural: adding a second rule subtracts, because it occupies a coin
for up to 48 hours and displaces what the bank would have taken there. Yet only about four of
thirteen coins are busy at any time, so the account is not full - the binding constraint is the
per-coin lock, and the lock is resolved by arrival time. Whoever signals first gets the coin,
regardless of which signal is better.

Nobody has ever asked what that costs. Four questions, in order of how much they could matter:

  1. how often do signals actually collide on a busy coin, and what is lost
  2. does resolving collisions by expected quality instead of arrival help
  3. does letting a coin hold two positions at once help, and what it does to the drawdown
  4. does a shorter lock - releasing the coin early - help

Expected quality has to be causal: it can only use what is known at the signal's own hour. The
rule's identity and its historical average are known; the trade's outcome is not.
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
COINS = [s for s in G.ALL if s not in ('AAVEUSDT', 'XLMUSDT')]
YT = {y: int(datetime.datetime(y, 1, 1, tzinfo=datetime.UTC).timestamp()) for y in range(2022, 2028)}
PB = [('rsi48', '>=', 56.3761), ('rsi6', '<=', 33.7947)]

bank = SW.run(3.0, 1.0, coins=COINS, with_meta=True)
pull = NA.trades('без режима', PB, coins=COINS)
EV = [(x[0], x[1], x[2], x[3], x[4], 'банк') for x in bank] + \
     [(x[0], x[1], x[2], x[3], x[4], 'откат') for x in pull]
EV.sort()
print('  сигналов всего: банк %d, откат %d' % (len(bank), len(pull)), flush=True)


def run(policy='время', per_coin=1, hold_cap=None):
    """policy: 'время' first-come, 'качество' better expected R wins, 'откат' rule has priority."""
    PRIOR = {'банк': 0.0843, 'откат': 0.1521}       # historical averages, known before the trade
    live = collections.defaultdict(list)            # coin -> list of (exit_ts, prior, index)
    out, lost = [], []
    for a, b, R, sf, s, tag in EV:
        bb = min(b, a + hold_cap) if hold_cap else b
        live[s] = [p for p in live[s] if p[0] > a]
        if len(live[s]) < per_coin:
            live[s].append((bb, PRIOR[tag], len(out)))
            out.append((a, bb, R, sf, s, 1.0, tag))
            continue
        if policy == 'время':
            lost.append((a, R, tag))
            continue
        if policy == 'откат':
            if tag != 'откат':
                lost.append((a, R, tag))
                continue
            worst = min(live[s], key=lambda p: p[1])
            if worst[1] >= PRIOR[tag]:
                lost.append((a, R, tag))
                continue
        else:
            worst = min(live[s], key=lambda p: p[1])
            if worst[1] >= PRIOR[tag]:
                lost.append((a, R, tag))
                continue
        # evict the weaker open position: it never happened, and this one takes the coin
        live[s].remove(worst)
        out[worst[2]] = None
        live[s].append((bb, PRIOR[tag], len(out)))
        out.append((a, bb, R, sf, s, 1.0, tag))
    out = [x for x in out if x is not None]
    out.sort()
    return out, lost


base, lost = run()
print('  взято %d, вытеснено %d' % (len(base), len(lost)), flush=True)
v_lost = np.array([r for _, r, _ in lost]) if lost else np.zeros(0)
v_took = np.array([x[2] for x in base])
print('  вытесненные сделки: ср%+.4fR ВР%5.1f%%  |  взятые: ср%+.4fR ВР%5.1f%%'
      % (v_lost.mean(), 100 * np.mean(v_lost > 0), v_took.mean(), 100 * np.mean(v_took > 0)),
      flush=True)
by = collections.Counter(t for _, _, t in lost)
print('  из вытесненных: банк %d, откат %d' % (by['банк'], by['откат']), flush=True)


def report(lbl, rows):
    plain = [(a, b, R, sf, s, w) for a, b, R, sf, s, w, t in rows]
    k, m = sim_w.money_at_dd(plain, 0.12)
    v = np.array([x[2] for x in rows])
    per = []
    for y in range(2022, 2027):
        sub = [x for x in plain if YT[y] <= x[0] < YT[y + 1]]
        per.append(sim_w.money_at_dd(sub, 0.12)[1] if len(sub) >= 50 else float('nan'))
    print('  %-34s %4d сд. ВР%5.1f%% ср%+.4f риск %.3f%% $%6.0f | %s'
          % (lbl, len(v), 100 * np.mean(v > 0), v.mean(), 100 * k, m,
             ' '.join('%3.0f' % x if np.isfinite(x) else '  -' for x in per)), flush=True)
    return m


print('', flush=True)
print('  === кто получает монету, когда сигналы сталкиваются ===', flush=True)
m0 = report('по времени (как сейчас)', base)
report('по ожидаемому качеству', run('качество')[0])
report('приоритет откату', run('откат')[0])

print('', flush=True)
print('  === сколько позиций держать на монете ===', flush=True)
for pc in (1, 2, 3):
    report('до %d позиций на монету' % pc, run('время', per_coin=pc)[0])

print('', flush=True)
print('  === отпускать монету раньше (замок короче удержания) ===', flush=True)
for cap in (6, 12, 24, 36):
    report('замок %dч (сделка живёт 48ч)' % cap, run('время', hold_cap=cap * 3600)[0])

print('', flush=True)
print('  === сколько монет занято одновременно ===', flush=True)
busy_at = []
ev2 = sorted([(x[0], 1) for x in base] + [(x[1], -1) for x in base])
cur = 0
for t, d in ev2:
    cur += d
    busy_at.append(cur)
ba = np.array(busy_at)
print('    медиана %.0f из %d монет, 90%% времени не больше %.0f, максимум %d'
      % (np.median(ba), len(COINS), np.quantile(ba, 0.9), ba.max()), flush=True)
