"""Attack the four new families the way the pullback rule was attacked.

A number at the top of a list of 26,866 survivors is not evidence. What convinced me about the
pullback rule was that it failed to survive being taken apart: each condition alone was worthless,
the inverse lost money, the mirror lost money, and no single year or coin carried it.

The same four attacks, plus the one that decides whether a rule is worth anything at all - how much
of it the bank and the pullback rule are already taking.
"""
import collections
import datetime
import sys

import numpy as np

sys.path.insert(0, 'C:/Users/Lenovo/Desktop/Торговля/crypto-bot')
import gauntlet2 as G

FLIP = {'>=': '<=', '<=': '>='}
YEARS = [2022, 2023, 2024, 2025, 2026]

CAND = [
    ('ПАДЕНИЕ', [('ret24', '<=', -1.5434), ('vol720', '>=', 1.4462)], True),
    ('КАПИТУЛЯЦИЯ', [('breadth', '<=', 0.0625), ('vol720', '>=', 1.4462)], True),
    ('ЭЙФОРИЯ', [('breadth', '>=', 1.0), ('vol168', '>=', 1.3947)], False),
    ('РАСХОЖДЕНИЕ', [('btc168', '<=', -0.0558), ('pos48', '>=', 0.8794)], False),
]

# what is already being taken
base = {}
for nm, conds, lg in [('ОТКАТ', [('rsi48', '>=', 56.3761), ('rsi6', '<=', 33.7947)], True)]:
    rows, _ = G.simulate(conds, lg)
    base[nm] = {(r[0], r[4]) for r in rows}
import stop_width as SW
bank = SW.run(3.0, 1.0, with_meta=True)
base['БАНК'] = {(a, s) for a, b, R, sf, s in bank}
print('  банк %d сделок, откат %d' % (len(base['БАНК']), len(base['ОТКАТ'])), flush=True)


def stats(rows):
    v = np.array([r[2] for r in rows]) if rows else np.zeros(0)
    if not len(v):
        return 'сделок нет'
    return 'n%5d ВР%5.1f%% ср%+.4fR' % (len(v), 100 * np.mean(v > 0), v.mean())


for name, conds, lg in CAND:
    print('', flush=True)
    print('  ================ %s (%s) ================' % (name, 'LONG' if lg else 'SHORT'),
          flush=True)
    rows, hours = G.simulate(conds, lg)
    print('    ЦЕЛИКОМ                 %s' % stats(rows), flush=True)
    for c in conds:
        r1, _ = G.simulate([c], lg)
        print('    только %-22s %s' % ('%s%s%.4f' % c, stats(r1)), flush=True)
    inv = [(n, FLIP[o], t) for n, o, t in conds]
    ri, _ = G.simulate(inv, lg)
    print('    ИНВЕРСИЯ обоих условий  %s   <- должна терять' % stats(ri), flush=True)
    rm, _ = G.simulate(conds, not lg)
    print('    ЗЕРКАЛО (другая сторона)%s   <- должно терять' % stats(rm), flush=True)

    print('    по годам:', flush=True)
    for y in YEARS:
        a = int(datetime.datetime(y, 1, 1, tzinfo=datetime.UTC).timestamp())
        b = int(datetime.datetime(y + 1, 1, 1, tzinfo=datetime.UTC).timestamp())
        sub = [r for r in rows if a <= r[0] < b]
        if len(sub) < 15:
            continue
        v = np.array([r[2] for r in sub])
        print('      %d n%4d ВР%5.1f%% ср%+.4fR' % (y, len(v), 100 * np.mean(v > 0), v.mean()),
              flush=True)

    print('    по монетам:', flush=True)
    per = collections.defaultdict(list)
    for r in rows:
        per[r[4]].append(r[2])
    pos = sum(1 for s in per if np.mean(per[s]) > 0)
    line = sorted(per.items(), key=lambda kv: -np.mean(kv[1]))
    print('      плюсовых монет %d из %d | лучшие %s | худшие %s'
          % (pos, len(per),
             ', '.join('%s%+.3f(%d)' % (s.replace('USDT', ''), np.mean(v), len(v))
                       for s, v in line[:3]),
             ', '.join('%s%+.3f(%d)' % (s.replace('USDT', ''), np.mean(v), len(v))
                       for s, v in line[-3:])), flush=True)

    keys = {(r[0], r[4]) for r in rows}
    for other in ('БАНК', 'ОТКАТ'):
        ov = keys & base[other]
        print('    пересечение с %-7s %d из %d' % (other, len(ov), len(keys)), flush=True)
