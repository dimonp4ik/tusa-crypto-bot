"""Is AAVE special, or does the test simply flatter whichever coin is at the bottom?

Being worse than the rest in nine half-years of nine is only impressive if no other coin does the
same. And the mechanism matters: a coin that loses because its slippage constant is large is an
execution problem, not a selection one, and removing it would be removing the measurement rather
than the loss.
"""
import collections
import datetime
import sys

import numpy as np

sys.path.insert(0, 'C:/Users/Lenovo/Desktop/Торговля/crypto-bot')
import gauntlet2 as G
import struct_params as SP
import stop_width as SW

bank = SW.run(3.0, 1.0, coins=G.ALL, with_meta=True)
print('  издержки заливки по монетам (SLIP_REAL):', flush=True)
for s in sorted(G.ALL, key=lambda x: -SP.SLIP_REAL.get(x, 0.0002)):
    print('    %-6s %.5f' % (s.replace('USDT', ''), SP.SLIP_REAL.get(s, 0.0002)), flush=True)

print('', flush=True)
print('  === «хуже остальных» по полугодиям — для КАЖДОЙ монеты ===', flush=True)
first, last = min(r[0] for r in bank), max(r[0] for r in bank)
rank = []
for s in G.ALL:
    mine = [(a, R) for a, b, R, sf, c in bank if c == s]
    rest = [(a, R) for a, b, R, sf, c in bank if c != s]
    t, worse, seen = first, 0, 0
    while t < last:
        a, b = t, t + 182 * 86400
        sa = [r[1] for r in mine if a <= r[0] < b]
        so = [r[1] for r in rest if a <= r[0] < b]
        if len(sa) >= 8 and len(so) >= 50:
            seen += 1
            worse += int(np.mean(sa) < np.mean(so))
        t = b
    v = [R for a, R in mine]
    rank.append((s, worse, seen, len(v), float(np.mean(v))))
for s, worse, seen, n, mu in sorted(rank, key=lambda x: -(x[1] / max(x[2], 1))):
    print('    %-6s хуже остальных в %d из %d полугодий | сделок %3d ср%+.4f'
          % (s.replace('USDT', ''), worse, seen, n, mu), flush=True)

print('', flush=True)
print('  === AAVE: где она теряет — стопы, тейки или зависания ===', flush=True)
for lbl, sel in (('AAVE', lambda c: c == 'AAVEUSDT'), ('остальные', lambda c: c != 'AAVEUSDT')):
    v = np.array([R for a, b, R, sf, c in bank if sel(c)])
    wins = v[v > 0]
    loss = v[v <= 0]
    print('    %-10s n%4d ВР%5.1f%% | выигрыш ср%+.4f медиана%+.4f | убыток ср%+.4f медиана%+.4f'
          % (lbl, len(v), 100 * len(wins) / len(v), wins.mean(), np.median(wins),
             loss.mean(), np.median(loss)), flush=True)
    print('                 доля полных стопов (R<=-0.9): %.1f%% | доля зависших (|R|<0.3): %.1f%%'
          % (100 * np.mean(v <= -0.9), 100 * np.mean(np.abs(v) < 0.3)), flush=True)

print('', flush=True)
print('  === по каждому правилу банка отдельно: AAVE против остальных ===', flush=True)
sig_by_rule = collections.defaultdict(lambda: collections.defaultdict(list))
for s in G.ALL:
    sg = SP.bank_signals([dict(r, tp=1.0) for r in SP.BASE], s, 50)
    for close, (ri, lim, atr) in sg.items():
        sig_by_rule[ri][s].append(close)
lookup = {}
for a, b, R, sf, c in bank:
    lookup[(a, c)] = R
for ri, rule in enumerate(SP.BASE):
    aa, oo = [], []
    for s, stamps in sig_by_rule[ri].items():
        for t in stamps:
            if (t, s) in lookup:
                (aa if s == 'AAVEUSDT' else oo).append(lookup[(t, s)])
    if len(aa) >= 10:
        print('    %-24s AAVE n%3d ср%+.4f | остальные n%4d ср%+.4f'
              % (rule.get('name', ri), len(aa), np.mean(aa), len(oo), np.mean(oo)), flush=True)
