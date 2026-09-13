"""Geometry for the new rules, and every coin one by one.

The 3.0 ATR stop and 1.0 ATR target were chosen for the bank's own entries. The two new rules enter
on a different thing - a pullback inside strength, and a capitulation - so there is no reason their
best geometry is the bank's. This sweeps it, with the held-out year protocol, because a geometry
sweep is exactly the kind of search that produced today's three retractions.

Then the per-coin table the account actually lives on: R per trade and win rate for the bank and for
each new rule, coin by coin, so a coin that is carried by one rule and wrecked by another is visible
rather than averaged away.
"""
import collections
import sys

import datetime
import numpy as np

sys.path.insert(0, 'C:/Users/Lenovo/Desktop/Торговля/crypto-bot')
import gauntlet2 as G
import stop_width as SW

RULES = [
    ('ОТКАТ', [('rsi48', '>=', 56.3761), ('rsi6', '<=', 33.7947)], True),
    ('КАПИТУЛЯЦИЯ', [('breadth', '<=', 0.0625), ('vol720', '>=', 1.4462)], True),
]
YEARS = [2022, 2023, 2024, 2025, 2026]
YT = {y: int(datetime.datetime(y, 1, 1, tzinfo=datetime.UTC).timestamp()) for y in YEARS + [2027]}
SLS = [2.0, 2.5, 3.0, 3.5, 4.0]
TPS = [0.6, 0.8, 1.0, 1.3, 1.6, 2.0, 2.5]

for fam, conds, lg in RULES:
    print('', flush=True)
    print('  ===== %s: геометрия =====' % fam, flush=True)
    CUR = {}
    for sl in SLS:
        for tp in TPS:
            CUR[(sl, tp)] = G.simulate(conds, lg, tp=tp, sl=sl)[0]
    print('  стоп\тейк %s' % ' '.join('%7.1f' % t for t in TPS), flush=True)
    for sl in SLS:
        cells = []
        for tp in TPS:
            v = np.array([r[2] for r in CUR[(sl, tp)]])
            cells.append('%7.4f' % v.mean() if len(v) else '      -')
        print('  %8.1f  %s' % (sl, ' '.join(cells)), flush=True)
    print('  сделок:', flush=True)
    for sl in SLS:
        print('  %8.1f  %s' % (sl, ' '.join('%7d' % len(CUR[(sl, t)]) for t in TPS)), flush=True)

    print('  выбор по четырём годам, замер на пятом (в скобках — что выбрано):', flush=True)
    wins = tot = 0
    for hold in YEARS:
        a, b = YT[hold], YT[hold + 1]
        best, bk = None, None
        for k, rows in CUR.items():
            tr = [r[2] for r in rows if not (a <= r[0] < b)]
            if len(tr) < 150:
                continue
            v = float(np.mean(tr))
            if best is None or v > best:
                best, bk = v, k
        ho = [r[2] for r in CUR[bk] if a <= r[0] < b]
        cur = [r[2] for r in CUR[(3.0, 1.0)] if a <= r[0] < b]
        if len(ho) < 15 or len(cur) < 15:
            print('    %d  выбрано стоп%.1f тейк%.1f — в невиданном году мало (n%d)'
                  % (hold, bk[0], bk[1], len(ho)), flush=True)
            continue
        tot += 1
        wins += int(np.mean(ho) > np.mean(cur))
        print('    %d  выбрано стоп%.1f тейк%.1f | НЕВИДАННЫЙ: выбранная%+.4f (n%d) '
              'против нынешней 3.0/1.0 %+.4f (n%d)'
              % (hold, bk[0], bk[1], np.mean(ho), len(ho), np.mean(cur), len(cur)), flush=True)
    if tot:
        print('    выбранная геометрия лучше нынешней в %d годах из %d' % (wins, tot), flush=True)

print('', flush=True)
print('  ===== разбор по монетам =====', flush=True)
bank = SW.run(3.0, 1.0, coins=G.ALL, with_meta=True)
per = {'БАНК': collections.defaultdict(list)}
for a, b, R, sf, s in bank:
    per['БАНК'][s].append(R)
for fam, conds, lg in RULES:
    per[fam] = collections.defaultdict(list)
    for r in G.simulate(conds, lg)[0]:
        per[fam][r[4]].append(r[2])
coins = sorted(G.ALL, key=lambda s: -np.mean(per['БАНК'].get(s, [0])))
print('  монета     банк               откат              капитуляция', flush=True)
for s in coins:
    cells = []
    for fam in ('БАНК', 'ОТКАТ', 'КАПИТУЛЯЦИЯ'):
        v = per[fam].get(s, [])
        cells.append('n%4d %+.4f %3.0f%%' % (len(v), np.mean(v), 100 * np.mean(np.array(v) > 0))
                     if len(v) >= 3 else 'n%-4d   -        ' % len(v))
    print('  %-10s %s' % (s.replace('USDT', ''), '  '.join(cells)), flush=True)
