"""Remove AAVE and the account gains 51%. Remove any coin and what happens?

That is the whole question. A single coin out of fifteen cannot plausibly cost a third of the money
unless it is genuinely bad - or unless removing ANY coin helps, in which case the gain is about
slots and leverage rather than about AAVE. The recorded rule in this project is that a coin may be
cut only on weakness in a hostile window, never on its average, so the year-by-year money is here
too.

Every coin is dropped in turn, for the bank alone and for bank+pullback, at equal drawdown.
"""
import collections
import datetime
import sys

import numpy as np

sys.path.insert(0, 'C:/Users/Lenovo/Desktop/Торговля/crypto-bot')
import gauntlet2 as G
import stop_width as SW
import sim_w

C = [('rsi48', '>=', 56.3761), ('rsi6', '<=', 33.7947)]
YT = {y: int(datetime.datetime(y, 1, 1, tzinfo=datetime.UTC).timestamp()) for y in range(2022, 2028)}


def account(coins, with_pb):
    bank = SW.run(3.0, 1.0, coins=coins, with_meta=True)
    ev = [(a, b, R, sf, s, 'bank') for a, b, R, sf, s in bank]
    if with_pb:
        cand, _ = G.simulate(C, True, coins=coins)
        ev += [(a, b, R, sf, s, 'cand') for a, b, R, sf, s in cand]
    ev.sort(key=lambda x: (x[0], 0 if x[5] == 'bank' else 1))
    busy, out = {}, []
    for a, b, R, sf, s, tag in ev:
        if busy.get(s, 0) > a:
            continue
        busy[s] = b
        out.append((a, b, R, sf, s, 1.0))
    out.sort()
    return out


def money(rows, want=0.12):
    return sim_w.money_at_dd(rows, want)[1]


ALL = list(G.ALL)
base_bank = account(ALL, False)
m_bank = money(base_bank)
base_pb = account(ALL, True)
m_pb = money(base_pb)
print('  опора: банк $%.0f, банк+откат $%.0f (монет %d)' % (m_bank, m_pb, len(ALL)), flush=True)

per = collections.defaultdict(list)
for a, b, R, sf, s, w in base_bank:
    per[s].append(R)

print('', flush=True)
print('  === выбить каждую монету по очереди ===', flush=True)
res = []
for drop in ALL:
    coins = [c for c in ALL if c != drop]
    rb = account(coins, False)
    rp = account(coins, True)
    mb, mp = money(rb), money(rp)
    v = per.get(drop, [0])
    res.append((drop, mb, mp, len(v), float(np.mean(v))))
for drop, mb, mp, n, mu in sorted(res, key=lambda x: -x[1]):
    print('    без %-6s банк $%6.0f (%+5.0f%%) | банк+откат $%6.0f (%+5.0f%%) | её сделок в банке %3d ср%+.4f'
          % (drop.replace('USDT', ''), mb, 100 * (mb / m_bank - 1), mp, 100 * (mp / m_pb - 1), n, mu),
          flush=True)

print('', flush=True)
print('  === деньги по годам: с AAVE и без ===', flush=True)
noaave = [c for c in ALL if c != 'AAVEUSDT']
for lbl, rows in (('банк со всеми', base_bank), ('банк без AAVE', account(noaave, False)),
                  ('банк+откат со всеми', base_pb), ('банк+откат без AAVE', account(noaave, True))):
    cells = []
    for y in range(2022, 2027):
        sub = [r for r in rows if YT[y] <= r[0] < YT[y + 1]]
        cells.append('%d $%3.0f' % (y, money(sub)) if len(sub) >= 50 else '%d    -' % y)
    print('    %-22s %4d сделок | %s' % (lbl, len(rows), '  '.join(cells)), flush=True)

print('', flush=True)
print('  === AAVE во ВРАЖДЕБНОМ окне: её худшие полугодия против общих ===', flush=True)
import struct_params as SP
aa = [(a, b, R) for a, b, R, sf, s, w in base_bank if s == 'AAVEUSDT']
ot = [(a, b, R) for a, b, R, sf, s, w in base_bank if s != 'AAVEUSDT']
first, last = min(r[0] for r in base_bank), max(r[0] for r in base_bank)
t, worse = first, 0
seen = 0
while t < last:
    a, b = t, t + 182 * 86400
    sa = [r[2] for r in aa if a <= r[0] < b]
    so = [r[2] for r in ot if a <= r[0] < b]
    if len(sa) >= 10 and len(so) >= 50:
        seen += 1
        worse += int(np.mean(sa) < np.mean(so))
        print('    %s  AAVE n%3d ср%+.4f | остальные ср%+.4f  %s'
              % (datetime.datetime.fromtimestamp(a, datetime.UTC).strftime('%y-%m'),
                 len(sa), np.mean(sa), np.mean(so),
                 'ХУЖЕ' if np.mean(sa) < np.mean(so) else 'лучше'), flush=True)
    t = b
print('    AAVE хуже остальных в %d полугодиях из %d' % (worse, seen), flush=True)
