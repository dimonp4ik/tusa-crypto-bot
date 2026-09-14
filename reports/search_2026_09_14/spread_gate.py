"""The live spread gate, which the model has never applied.

src/execution_guard.py refuses any entry whose book spread exceeds PULLBACK_LIVE_MAX_SPREAD, 0.05%.
A day of sampling says three coins sit permanently above it - BILL at 0.47%, XLM at 0.26%, AAVE at
0.12% - and two more straddle it. So the live bot would refuse trades the model happily takes, on
coins that carry 162 and 209 trades in the book.

Two readings, because the sample is one day:

  жёстко  - a coin whose MEDIAN spread exceeds the limit is removed entirely
  по доле - each trade is refused with the coin's own measured refusal rate, drawn with a fixed
            seed, which is closer to the truth for the coins that straddle the limit

Neither is the live truth. Together they bracket it, and the bracket is what matters: if the account
survives the hard version, the gate is priced.
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

LIMIT = 0.0005
sp = collections.defaultdict(list)
cost = collections.defaultdict(list)
for r in csv.DictReader(open('book_samples.csv')):
    sp[r['coin'] + 'USDT'].append(float(r['spread']))
    cost[r['coin'] + 'USDT'].append(float(r['cost109']))
SP.SLIP_REAL.update({k: float(np.median(v)) for k, v in cost.items()})
G.FC.clear()

PASS = {k: float(np.mean(np.array(v) <= LIMIT)) for k, v in sp.items()}
MED = {k: float(np.median(v)) for k, v in sp.items()}
print('  монета   медиана спреда  доля замеров ниже порога 0.05%', flush=True)
for s in sorted(MED, key=lambda x: -MED[x]):
    mark = ''
    if MED[s] > LIMIT:
        mark = '  <- живой бот отказывает почти всегда'
    elif PASS[s] < 0.9:
        mark = '  <- отказывает иногда'
    print('    %-6s %.6f      %5.1f%%%s' % (s.replace('USDT', ''), MED[s], 100 * PASS[s], mark),
          flush=True)

ALL = list(G.ALL)
C = [('rsi48', '>=', 56.3761), ('rsi6', '<=', 33.7947)]
YT = {y: int(datetime.datetime(y, 1, 1, tzinfo=datetime.UTC).timestamp()) for y in range(2022, 2028)}


def account(coins, mode, mode_pass=None, seed=7):
    """mode: None = bank alone, 'гейт' = plus the gated rule, 'без режима' = plus the gateless one."""
    bank = SW.run(3.0, 1.0, coins=coins, with_meta=True)
    cand = NA.trades(mode, C, coins=coins) if mode else []
    ev = [(a, b, R, sf, s, 0) for a, b, R, sf, s in bank] + \
         [(a, b, R, sf, s, 1) for a, b, R, sf, s in cand]
    ev.sort(key=lambda x: (x[0], x[5]))
    rng = np.random.default_rng(seed)
    busy, out = {}, []
    for a, b, R, sf, s, t in ev:
        if busy.get(s, 0) > a:
            continue
        if mode_pass is not None:
            p = mode_pass.get(s, 1.0)
            if p < 1.0 and rng.random() > p:
                continue          # the live spread gate refused this entry
        busy[s] = b
        out.append((a, b, R, sf, s, 1.0))
    out.sort()
    return out


def show(lbl, rows):
    k, m = sim_w.money_at_dd(rows, 0.12)
    per = []
    for y in range(2022, 2027):
        sub = [r for r in rows if YT[y] <= r[0] < YT[y + 1]]
        per.append('%d $%3.0f' % (y, sim_w.money_at_dd(sub, 0.12)[1]) if len(sub) >= 50
                   else '%d   -' % y)
    print('  %-40s сделок %4d риск %.3f%% $%6.0f | %s'
          % (lbl, len(rows), 100 * k, m, ' '.join(per)), flush=True)
    return m


print('', flush=True)
print('  === без отсечки (как считалось до сих пор) ===', flush=True)
show('банк ОДИН', account(ALL, None))
show('банк + откат в гейте', account(ALL, 'гейт'))
show('банк + откат без гейта', account(ALL, 'без режима'))

HARD = [s for s in ALL if MED.get(s, 0) <= LIMIT]
print('', flush=True)
print('  === жёстко: убраны монеты с медианным спредом выше порога (%s) ==='
      % ', '.join(s.replace('USDT', '') for s in ALL if s not in HARD), flush=True)
show('банк ОДИН', account(HARD, None))
show('банк + откат в гейте', account(HARD, 'гейт'))
show('банк + откат без гейта', account(HARD, 'без режима'))

print('', flush=True)
print('  === по доле отказов, три зерна ===', flush=True)
for seed in (7, 11, 23, 31, 47, 59):
    show('банк + откат без гейта (зерно %d)' % seed, account(ALL, 'без режима', mode_pass=PASS, seed=seed))

print('', flush=True)
print('  === и то и другое: монеты выше порога убраны, остальным доля отказов ===', flush=True)
for seed in (7, 11, 23, 31, 47, 59):
    show('банк + откат без гейта (зерно %d)' % seed,
         account(HARD, 'без режима', mode_pass=PASS, seed=seed))
