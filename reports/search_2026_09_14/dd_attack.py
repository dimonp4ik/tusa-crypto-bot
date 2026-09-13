"""Attack the drawdown itself, which is what sets every other number.

The account latches at 15% and never restarts, so the deepest pit decides the largest risk that may
be carried, and therefore all of the money. The pit is a cluster - 22 losses in 58 days with a
median of 4 positions open at each one - so the two obvious levers are a cap on how many positions
may be open at once, and a pause after a run of losses.

Both were tried on the bank alone months ago and neither survived. They are worth retrying now
because the trade set is different: four rules with near-zero overlap fill slots the bank left
empty, so a cap now bites on diversity rather than on the bank's own best trades.

Comparison is at EQUAL DRAWDOWN: for each variant, the risk whose worst drawdown is 12%, and the
money made there. That takes the latch cliff out of the comparison - the previous table was
accidentally measuring the cliff edge, because dd_of returns a negative number.
"""
import collections
import sys

import numpy as np

sys.path.insert(0, 'C:/Users/Lenovo/Desktop/Торговля/crypto-bot')
import gauntlet2 as G
import stop_width as SW
import margin_cap as MC
import live_rules_sim as L

RULES = {
    'ОТКАТ': ([('rsi48', '>=', 56.3761), ('rsi6', '<=', 33.7947)], True),
    'ПАДЕНИЕ': ([('ret24', '<=', -1.5434), ('vol720', '>=', 1.4462)], True),
    'КАПИТУЛЯЦИЯ': ([('breadth', '<=', 0.0625), ('vol720', '>=', 1.4462)], True),
    'РАСХОЖДЕНИЕ': ([('btc168', '<=', -0.0558), ('pos48', '>=', 0.8794)], False),
}
S = {nm: G.simulate(c, lg)[0] for nm, (c, lg) in RULES.items()}
bank = SW.run(3.0, 1.0, coins=G.ALL, with_meta=True)
ALL4 = ['ОТКАТ', 'ПАДЕНИЕ', 'КАПИТУЛЯЦИЯ', 'РАСХОЖДЕНИЕ']


def merge(names, cap=None, loss_pause=None):
    """cap: max positions open at once, enforced chronologically at entry.
       loss_pause: (k, hours) - after k losing exits in a row, take nothing for `hours`."""
    ev = [(a, b, R, sf, s, 'bank') for a, b, R, sf, s in bank]
    for nm in names:
        ev += [(a, b, R, sf, s, nm) for a, b, R, sf, s in S[nm]]
    pri = {'bank': 0}
    pri.update({nm: i + 1 for i, nm in enumerate(names)})
    ev.sort(key=lambda x: (x[0], pri[x[5]]))
    busy, out, live = {}, [], []
    streak, until = 0, 0
    for a, b, R, sf, s, tag in ev:
        while live and live[0][0] <= a:
            _, rr = live.pop(0)
            if loss_pause:
                streak = streak + 1 if rr < 0 else 0
                if streak >= loss_pause[0]:
                    until, streak = max(until, a + loss_pause[1] * 3600), 0
        if busy.get(s, 0) > a:
            continue
        if cap is not None and len(live) >= cap:
            continue
        if loss_pause and a < until:
            continue
        busy[s] = b
        out.append((a, b, R, sf, s, tag))
        live.append((b, R))
        live.sort(key=lambda p: p[0])
    out.sort()
    return out


def money_at_dd(rows, want=0.12):
    """Risk whose worst drawdown is `want`; returns (risk, money, drawdown, trades)."""
    MC.trades = [(a, b, R, sf, s) for a, b, R, sf, s, _ in rows]
    lo, hi = 0.002, 0.030
    for _ in range(26):
        mid = (lo + hi) / 2
        r = L.simulate(target=mid)
        bad = bool(r['paused_at']) or abs(L.dd_of(r['curve'])) > want
        if bad:
            hi = mid
        else:
            lo = mid
    r = L.simulate(target=lo)
    return lo, r['eq'], abs(L.dd_of(r['curve'])), len(rows)


print('  === при РАВНОЙ просадке 12%: риск и деньги ===', flush=True)
BASE = {}
for nm, names in (('банк', []), ('+откат', ['ОТКАТ']),
                  ('откат+расхожд', ['ОТКАТ', 'РАСХОЖДЕНИЕ']),
                  ('три (без кап.)', ['ОТКАТ', 'ПАДЕНИЕ', 'РАСХОЖДЕНИЕ']),
                  ('все четыре', ALL4)):
    rows = merge(names)
    BASE[nm] = rows
    k, eq, dd, n = money_at_dd(rows)
    print('    %-15s риск %.3f%%  $%7.0f  просадка %5.1f%%  сделок %d'
          % (nm, 100 * k, eq, 100 * dd, n), flush=True)

print('', flush=True)
print('  === потолок одновременных позиций (все четыре, при равной просадке 12%) ===', flush=True)
for cap in (None, 10, 8, 6, 5, 4, 3):
    rows = merge(ALL4, cap=cap)
    k, eq, dd, n = money_at_dd(rows)
    print('    потолок %-4s сделок %4d  риск %.3f%%  $%7.0f  просадка %5.1f%%'
          % ('нет' if cap is None else cap, n, 100 * k, eq, 100 * dd), flush=True)

print('', flush=True)
print('  === пауза после серии убытков (все четыре) ===', flush=True)
for lp in (None, (3, 24), (3, 48), (4, 24), (4, 48), (5, 24), (5, 72), (6, 48)):
    rows = merge(ALL4, loss_pause=lp)
    k, eq, dd, n = money_at_dd(rows)
    print('    %-14s сделок %4d  риск %.3f%%  $%7.0f  просадка %5.1f%%'
          % ('нет' if lp is None else '%d убытков/%dч' % lp, n, 100 * k, eq, 100 * dd), flush=True)

print('', flush=True)
print('  === и то и другое ===', flush=True)
for cap in (6, 5, 4):
    for lp in ((4, 48), (5, 48)):
        rows = merge(ALL4, cap=cap, loss_pause=lp)
        k, eq, dd, n = money_at_dd(rows)
        print('    потолок %d + %d убытков/%dч: сделок %4d риск %.3f%% $%7.0f просадка %5.1f%%'
              % (cap, lp[0], lp[1], n, 100 * k, eq, 100 * dd), flush=True)
