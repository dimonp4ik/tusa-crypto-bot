"""What the live bank would actually have done, with its own rules rather than the backtest's.

Three things in src/pullback_live.py and config.py are not in any number reported so far.

  * Sizing. pullback_live.py:254 shrinks the margin only when the stop is WIDER than
    PULLBACK_STOP_REF (3.94%); a tighter stop keeps the full margin. So risk per trade is not
    constant - it is capped at the reference and falls proportionally for tight stops. Every
    figure in the reports assumes equal risk per trade.
  * A latched drawdown pause at 15% of peak equity (PULLBACK_LIVE_MAX_DRAWDOWN). The worst
    drawdown measured at 1.5% risk is -16.4%. It is latched on purpose: "recovering equity must
    not silently restart it" - so the bank would stop, permanently, inside a drawdown it is
    expected to have.
  * A daily loss pause at 3% (PULLBACK_LIVE_MAX_DAILY_LOSS). At 1.5% risk that is two stops in a
    day, and stops cluster - 43% of them land within six hours of another.

Simulated here together, on the same historical path, so the three can be separated: sizing alone,
sizing plus the daily pause, and everything including the latch.
"""
import collections
import datetime
import sys

import numpy as np

sys.path.insert(0, 'C:/Users/Lenovo/Desktop/Торговля/crypto-bot')
import margin_cap as MC

DEPOSIT = MC.DEPOSIT
LEV = MC.LEVERAGE
REF = 0.0394                 # PULLBACK_STOP_REF
MAX_DAILY = 0.03             # PULLBACK_LIVE_MAX_DAILY_LOSS
MAX_DD = 0.15                # PULLBACK_LIVE_MAX_DRAWDOWN
TARGET = 0.015               # the owner's risk setting


def day_of(ts):
    return datetime.datetime.fromtimestamp(int(ts), datetime.UTC).strftime('%Y-%m-%d')


def simulate(target=TARGET, sizing='live', daily=True, latch=True, margin_cap=True):
    """sizing='equal_risk': margin is set so EVERY trade risks `target` - what the reports assume.
       sizing='live'      : the deployed rule - a fixed margin, shrunk only when the stop is wider
                            than the reference, so risk is capped at `target` and is lower for
                            tighter stops.
       sizing='equal_margin': the same fixed margin with no shrink at all, for reference."""
    margin_frac = target / (LEV * REF)
    eq = DEPOSIT
    peak = DEPOSIT
    open_pos = []
    day, day_start, day_paused = None, DEPOSIT, False
    paused = False
    taken = skipped_margin = skipped_daily = 0
    curve = []
    paused_at = None
    for a, b, R, sf, s in MC.trades:
        while open_pos and open_pos[0][0] <= a:
            t_close, m, money_per_R, rr = open_pos.pop(0)
            eq += money_per_R * rr
            peak = max(peak, eq)
            curve.append((t_close, eq))
        d = day_of(a)
        if d != day:
            day, day_start, day_paused = d, eq, False
        if latch and not paused and eq <= peak * (1 - MAX_DD):
            paused, paused_at = True, a
        if daily and not day_paused and eq <= day_start * (1 - MAX_DAILY):
            day_paused = True
        if paused or (daily and day_paused):
            skipped_daily += 1
            continue
        if sizing == 'equal_risk':
            margin = target * eq / (LEV * sf)
        else:
            margin = margin_frac * eq
            if sizing == 'live' and sf > REF:
                margin *= REF / sf
        notional = margin * LEV
        used = sum(p[1] for p in open_pos)
        if margin_cap and used + margin > eq * MC.USABLE:
            skipped_margin += 1
            continue
        taken += 1
        open_pos.append((b, margin, notional * sf, R))
        open_pos.sort(key=lambda p: p[0])
    for t_close, m, money_per_R, rr in open_pos:
        eq += money_per_R * rr
        curve.append((t_close, eq))
    return dict(eq=eq, taken=taken, skipped_margin=skipped_margin, skipped_pause=skipped_daily,
                curve=curve, paused_at=paused_at)


def dd_of(curve):
    if not curve:
        return 0.0
    e = np.array([c[1] for c in curve], dtype=float)
    return float((e / np.maximum.accumulate(e) - 1).min())


print('  всего сделок: %d' % len(MC.trades), flush=True)
print('', flush=True)
VAR = [
    ('как в отчётах: равный риск, без отсечек',
     dict(sizing='equal_risk', daily=False, latch=False)),
    ('живой размер (stop_ref), без отсечек', dict(sizing='live', daily=False, latch=False)),
    ('равная маржа без урезания (для сравнения)',
     dict(sizing='equal_margin', daily=False, latch=False)),
    ('живой размер + дневной лимит 3%', dict(sizing='live', daily=True, latch=False)),
    ('живой размер + оба лимита (как в коде)', dict(sizing='live', daily=True, latch=True)),
    ('равный риск + оба лимита', dict(sizing='equal_risk', daily=True, latch=True)),
]
print('  вариант                                   взято  проп.маржа  проп.пауза     итог   просадка',
      flush=True)
for tag, kw in VAR:
    r = simulate(**kw)
    print('  %-40s %6d      %6d      %6d  $%7.0f    %5.1f%%'
          % (tag, r['taken'], r['skipped_margin'], r['skipped_pause'], r['eq'],
             100 * dd_of(r['curve'])), flush=True)
    if r['paused_at']:
        print('       защёлка сработала %s и больше не снималась'
              % datetime.datetime.fromtimestamp(int(r['paused_at']), datetime.UTC).strftime('%Y-%m-%d'),
              flush=True)

print('', flush=True)
print('  === то же при разных ставках риска, со всеми живыми правилами ===', flush=True)
print('  риск     взято  проп.пауза     итог   просадка   защёлка', flush=True)
for t in (0.005, 0.0075, 0.01, 0.0125, 0.015, 0.02, 0.03):
    r = simulate(target=t)
    pa = (datetime.datetime.fromtimestamp(int(r['paused_at']), datetime.UTC).strftime('%Y-%m')
          if r['paused_at'] else 'не сработала')
    print('  %5.2f%%  %6d      %6d  $%7.0f    %5.1f%%   %s'
          % (100 * t, r['taken'], r['skipped_pause'], r['eq'], 100 * dd_of(r['curve']), pa),
          flush=True)
