"""Break the short rule before believing it, exactly as the pullback rule was broken.

It is better than bank+pullback in all five years and in seven of nine forward half-years, which is
the strongest incremental evidence anything has produced today. That is a reason to attack it, not
to accept it: the pullback rule earned belief by failing to break, and nothing less will do here.
"""
import collections
import datetime
import sys

import numpy as np

sys.path.insert(0, 'C:/Users/Lenovo/Desktop/Торговля/crypto-bot')
import gauntlet2 as G

C = [('pos24', '<=', 0.1649), ('rsi48', '>=', 46.2810)]
FLIP = {'>=': '<=', '<=': '>='}


def stat(rows, lab):
    v = np.array([r[2] for r in rows]) if rows else np.zeros(0)
    if not len(v):
        print('    %-34s сделок нет' % lab, flush=True)
        return
    print('    %-34s n%5d ВР%5.1f%% ср%+.4fR' % (lab, len(v), 100 * np.mean(v > 0), v.mean()),
          flush=True)


rows, hours = G.simulate(C, False)
stat(rows, 'ЦЕЛИКОМ')
for c in C:
    stat(G.simulate([c], False)[0], 'только %s%s%.4f' % c)
stat(G.simulate([(n, FLIP[o], t) for n, o, t in C], False)[0], 'ИНВЕРСИЯ обоих (должна терять)')
stat(G.simulate(C, True)[0], 'ЗЕРКАЛО LONG (должно терять)')

print('', flush=True)
print('  по годам:', flush=True)
for y in range(2022, 2027):
    a = int(datetime.datetime(y, 1, 1, tzinfo=datetime.UTC).timestamp())
    b = int(datetime.datetime(y + 1, 1, 1, tzinfo=datetime.UTC).timestamp())
    sub = [r for r in rows if a <= r[0] < b]
    if len(sub) < 10:
        print('    %d n%d — мало' % (y, len(sub)), flush=True)
        continue
    v = np.array([r[2] for r in sub])
    print('    %d n%3d ВР%5.1f%% ср%+.4fR' % (y, len(v), 100 * np.mean(v > 0), v.mean()), flush=True)

print('', flush=True)
print('  по монетам:', flush=True)
per = collections.defaultdict(list)
for r in rows:
    per[r[4]].append(r[2])
line = sorted(per.items(), key=lambda kv: -np.mean(kv[1]))
print('    плюсовых %d из %d' % (sum(1 for s in per if np.mean(per[s]) > 0), len(per)), flush=True)
for s, v in line:
    print('      %-6s n%3d ср%+.4f ВР%3.0f%%' % (s.replace('USDT', ''), len(v), np.mean(v),
                                                 100 * np.mean(np.array(v) > 0)), flush=True)

print('', flush=True)
print('  сетка порогов вокруг выбранной клетки (среднее R / число сделок):', flush=True)
PS = [0.10, 0.13, 0.1649, 0.20, 0.24, 0.28]
RS = [40.0, 43.0, 46.2810, 49.0, 52.0, 55.0]
print('    pos24\rsi48 %s' % ' '.join('%13.1f' % r for r in RS), flush=True)
for p in PS:
    cells = []
    for r in RS:
        rr, _ = G.simulate([('pos24', '<=', p), ('rsi48', '>=', r)], False)
        v = np.array([x[2] for x in rr]) if rr else np.zeros(0)
        cells.append('%+.4f/%4d' % (v.mean(), len(v)) if len(v) else '        -    ')
    print('    %11.4f %s' % (p, ' '.join(cells)), flush=True)
