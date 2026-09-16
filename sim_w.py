"""The live guards with a size multiplier that actually multiplies size.

The previous attempt scaled R up and the stop fraction down by the same factor. Profit in this model
is margin x leverage x stop_fraction x R, so those two cancel exactly and the knob did nothing
except where the stop_ref shrink happened to fire. That is the third knob in this project found to
be inert after being read as a result, so this one carries a self-test: at weight 2 on every trade,
the money must move, and at weight 1 it must reproduce the untouched simulation to the cent.
"""
import sys

import numpy as np

sys.path.insert(0, 'C:/Users/Lenovo/Desktop/Торговля/crypto-bot')
import margin_cap as MC
import live_rules_sim as L

DEPOSIT, LEV, REF = L.DEPOSIT, L.LEV, L.REF
MAX_DD, MAX_DAILY, USABLE = L.MAX_DD, L.MAX_DAILY, MC.USABLE


def simulate(trades, target, daily=True, latch=True, margin_cap=True):
    """trades: (entry, exit, R, stop_frac, coin, weight). The weight scales the position, and
    therefore both the money risked and the money made, exactly as a size rule would."""
    margin_frac = target / (LEV * REF)
    eq = peak = DEPOSIT
    open_pos = []
    day, day_start, day_paused = None, DEPOSIT, False
    paused, paused_at = False, None
    taken = skipped_margin = skipped_pause = 0
    curve = []
    for a, b, R, sf, s, w in trades:
        while open_pos and open_pos[0][0] <= a:
            t_close, m, money_per_R, rr = open_pos.pop(0)
            eq += money_per_R * rr
            peak = max(peak, eq)
            curve.append((t_close, eq))
        d = L.day_of(a)
        if d != day:
            day, day_start, day_paused = d, eq, False
        if latch and not paused and eq <= peak * (1 - MAX_DD):
            paused, paused_at = True, a
        if daily and not day_paused and eq <= day_start * (1 - MAX_DAILY):
            day_paused = True
        if paused or (daily and day_paused):
            skipped_pause += 1
            continue
        margin = margin_frac * eq
        if sf > REF:
            margin *= REF / sf
        margin *= w
        used = sum(p[1] for p in open_pos)
        if margin_cap and used + margin > eq * USABLE:
            skipped_margin += 1
            continue
        taken += 1
        open_pos.append((b, margin, margin * LEV * sf, R))
        open_pos.sort(key=lambda p: p[0])
    for t_close, m, money_per_R, rr in open_pos:
        eq += money_per_R * rr
        curve.append((t_close, eq))
    return dict(eq=eq, taken=taken, skipped_margin=skipped_margin, skipped_pause=skipped_pause,
                curve=curve, paused_at=paused_at)


def money_at_dd(trades, want=0.12):
    lo, hi = 0.0005, 0.030
    for _ in range(26):
        mid = (lo + hi) / 2
        r = simulate(trades, mid)
        if bool(r['paused_at']) or abs(L.dd_of(r['curve'])) > want:
            hi = mid
        else:
            lo = mid
    r = simulate(trades, lo)
    return lo, r['eq']


if __name__ == '__main__':
    import gauntlet2 as G
    import stop_width as SW
    bank = SW.run(3.0, 1.0, coins=G.ALL, with_meta=True)
    w1 = [(a, b, R, sf, s, 1.0) for a, b, R, sf, s in bank]
    w2 = [(a, b, R, sf, s, 2.0) for a, b, R, sf, s in bank]
    MC.trades = [(a, b, R, sf, s) for a, b, R, sf, s in bank]
    ref = L.simulate(target=0.0125)
    mine = simulate(w1, 0.0125)
    dbl = simulate(w2, 0.0125)
    print('  САМОПРОВЕРКА', flush=True)
    print('    вес 1 против нетронутой симуляции: $%.2f против $%.2f  %s'
          % (mine['eq'], ref['eq'], 'совпало' if abs(mine['eq'] - ref['eq']) < 0.01 else 'РАСХОЖДЕНИЕ'),
          flush=True)
    print('    вес 2 на всех сделках: $%.2f, взято %d, отказов по марже %d  %s'
          % (dbl['eq'], dbl['taken'], dbl['skipped_margin'],
             'вес работает' if abs(dbl['eq'] - ref['eq']) > 1 else 'ВЕС НЕ РАБОТАЕТ'), flush=True)
    print('    вес 2 то же, что риск вдвое: $%.2f против $%.2f'
          % (dbl['eq'], simulate(w1, 0.025)['eq']), flush=True)
