"""Is the rule fading? The only question left that changes the decision.

Its yearly contribution runs +9%, +11%, +44%, +26%, +13% - no monotone decline, but 2026 sits below
the middle years and 2026 is the data closest to now. A rule that worked until recently and stopped
is the worst thing to switch on.

Month by month over the whole history, and then the last twelve months in detail: the rule's own
trades, and what the account does with and without it.
"""
import collections, csv, datetime, sys
import numpy as np
sys.path.insert(0, 'C:/Users/Lenovo/Desktop/Торговля/crypto-bot')
import struct_params as SP
import gauntlet2 as G
import stop_width as SW
import sim_w
import nogate_attack as NA

cost = collections.defaultdict(list)
for r in csv.DictReader(open('book_frozen.csv')):
    cost[r['coin']+'USDT'].append(float(r['cost109']))
SP.SLIP_REAL.update({k: float(np.median(v)) for k, v in cost.items()})
G.FC.clear()
COINS = [s for s in G.ALL if s not in ('AAVEUSDT', 'XLMUSDT')]
RULE = [('rsi48','>=',56.3761), ('rsi6','<=',33.7947)]
tr = NA.trades('без режима', RULE, coins=COINS)
print('  сделок правила: %d' % len(tr), flush=True)
mon = collections.defaultdict(list)
for x in tr:
    mon[datetime.datetime.fromtimestamp(x[0], datetime.UTC).strftime('%y-%m')].append(x[2])
ks = sorted(mon)
print('', flush=True)
print('  === помесячно: сделок / средний R (пусто = меньше 3 сделок) ===', flush=True)
line = []
for k in ks:
    v = mon[k]
    line.append('%s %2d %+.2f' % (k, len(v), np.mean(v)) if len(v) >= 3 else '%s %2d   .  ' % (k, len(v)))
for i in range(0, len(line), 4):
    print('    ' + ' | '.join(line[i:i+4]), flush=True)

print('', flush=True)
print('  === по полугодиям: правило против своего же среднего ===', flush=True)
half = collections.defaultdict(list)
for x in tr:
    d = datetime.datetime.fromtimestamp(x[0], datetime.UTC)
    half['%d-%s' % (d.year, 'H1' if d.month <= 6 else 'H2')].append(x[2])
allm = np.mean([x[2] for x in tr])
for k in sorted(half):
    v = np.array(half[k])
    if len(v) < 10:
        continue
    print('    %-8s n%3d ВР%5.1f%% ср%+.4f  %s' % (k, len(v), 100*np.mean(v>0), v.mean(),
          'выше среднего' if v.mean() > allm else 'ниже'), flush=True)
print('    среднее за всё: %+.4f' % allm, flush=True)

print('', flush=True)
print('  === последние 12 месяцев: счёт с правилом и без ===', flush=True)
bank = SW.run(3.0, 1.0, coins=COINS, with_meta=True)
def acc(withrule):
    ev = [(x[0],x[1],x[2],x[3],x[4],0) for x in bank]
    if withrule:
        ev += [(x[0],x[1],x[2],x[3],x[4],1) for x in tr]
    ev.sort(key=lambda z:(z[0],z[5])); busy, out = {}, []
    for p,q,R,sf,s,k in ev:
        if busy.get(s,0) > p: continue
        busy[s] = q; out.append((p,q,R,sf,s,1.0))
    out.sort(); return out
A, B = acc(False), acc(True)
last = max(x[0] for x in A)
for months in (12, 9, 6, 3):
    lo = last - months*30*86400
    a = [x for x in A if x[0] >= lo]; b = [x for x in B if x[0] >= lo]
    if len(a) < 40: continue
    ra, rb = sim_w.simulate(a, 0.014), sim_w.simulate(b, 0.014)
    va = np.array([x[2] for x in a]); vb = np.array([x[2] for x in b])
    print('    последние %2d мес: банк %3d сд ср%+.4f $%5.1f | с правилом %3d сд ср%+.4f $%5.1f  (%+.0f%%)'
          % (months, len(va), va.mean(), ra['eq']-120, len(vb), vb.mean(), rb['eq']-120,
             100*((rb['eq']-120)/(ra['eq']-120)-1) if ra['eq'] > 120 else float('nan')), flush=True)
