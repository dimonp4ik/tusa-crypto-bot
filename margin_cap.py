"""What raising the risk actually does to a $120 account, once margin is allowed to say no.

The earlier projection - $224 after a year at 1.5% risk, $401 at 3%, $863 at 5% - came from
compounding the R stream. That arithmetic silently assumes every signal is taken. It is not true
above 1.5%: at the busiest moment the bank wants sixteen positions at once, needing $111 of margin
at 1.5% risk, $223 at 3% and $371 at 5%, against a $120 deposit.

An account that runs out of margin does not earn less of the same thing - it takes a different,
unmeasured subset of the trades, and which ones it skips is decided by arrival order rather than by
quality. So the honest projection has to refuse the signals it cannot fund.

Simulated here with the deposit compounding, so the cap loosens as the account grows, and with the
skip rule the live bot would actually follow: a signal arrives, if the margin is not there it is
gone. Reported against the same run with no cap, which is what the old numbers assumed.
"""
import collections
import datetime
import sys

import numpy as np

sys.path.insert(0, 'C:/Users/Lenovo/Desktop/Торговля/crypto-bot')
import struct_params as SP

DEPOSIT = 120.0
LEVERAGE = 10.0
USABLE = 0.95          # keep a little free so the exchange does not reject the order
RISKS = (0.015, 0.03, 0.05)
BASE_COINS = list(SP.COINS)

trades = []
for s in BASE_COINS:
    SP.COINS = [s]
    try:
        part = SP.portfolio(SP.BASE, sma_n=50)
    finally:
        SP.COINS = BASE_COINS
    sig = SP.bank_signals(SP.BASE, s, 50)
    for a, b, R in part:
        ri, lim, atr = sig[a]
        trades.append((a, b, R, SP.BASE[ri]['sl'] * atr / lim, s))
trades.sort()
print('  сделок: %d, с %s по %s'
      % (len(trades),
         datetime.datetime.fromtimestamp(trades[0][0], datetime.UTC).strftime('%Y-%m'),
         datetime.datetime.fromtimestamp(trades[-1][0], datetime.UTC).strftime('%Y-%m')),
      flush=True)


def simulate(risk, cap=True):
    """Chronological walk. Positions hold margin until they close; a signal that does not fit
    is skipped. Equity compounds on realised results only."""
    eq = DEPOSIT
    open_pos = []              # (close_ts, margin, risk_dollars, R)
    taken = skipped = 0
    curve = []
    for a, b, R, sf, s in trades:
        while open_pos and open_pos[0][0] <= a:
            _, m, rd, rr = open_pos.pop(0)
            eq += rd * rr
            curve.append((_, eq))
        used = sum(p[1] for p in open_pos)
        rd = risk * eq
        notional = rd / sf
        margin = notional / LEVERAGE
        if cap and used + margin > eq * USABLE:
            skipped += 1
            continue
        taken += 1
        open_pos.append((b, margin, rd, R))
        open_pos.sort(key=lambda p: p[0])
    for _, m, rd, rr in open_pos:
        eq += rd * rr
        curve.append((_, eq))
    return eq, taken, skipped, curve


print('', flush=True)
print('  риск   вариант        сделок взято  пропущено   итог из $120   в месяц', flush=True)
months = (trades[-1][1] - trades[0][0]) / (365.25 * 86400 / 12)
for r in RISKS:
    for cap, name in ((False, 'без учёта маржи'), (True, 'с учётом маржи ')):
        eq, taken, skipped, _ = simulate(r, cap)
        growth = (eq / DEPOSIT) ** (1 / months) - 1
        print('  %4.1f%%  %s  %6d     %6d      $%9.0f   %+6.2f%%'
              % (100 * r, name, taken, skipped, eq, 100 * growth), flush=True)

print('', flush=True)
print('  === первый год (депозит ещё маленький) ===', flush=True)
cut = trades[0][0] + 365 * 86400
sub = [t for t in trades if t[0] < cut]
saved = trades[:]
for r in RISKS:
    globals()['trades'] = sub
    e1, t1, s1, _ = simulate(r, True)
    e0, t0, s0, _ = simulate(r, False)
    globals()['trades'] = saved
    print('    риск %4.1f%%: с маржой $%.0f (взято %d, пропущено %d) | без учёта $%.0f'
          % (100 * r, e1, t1, s1, e0), flush=True)
