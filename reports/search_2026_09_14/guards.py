"""The live risk layer, never audited: the daily pause, the latch, and the reference stop.

Three numbers sit in the live path and none has been measured for what it is worth.

    PULLBACK_LIVE_MAX_DAILY    3%   stop trading for the rest of the day after this fall
    PULLBACK_LIVE_MAX_DRAWDOWN 15%  latch: stop forever, never resume
    PULLBACK_STOP_REF       0.0394  shrink the position when the trade's stop is wider than this

They are guards, so the question is not whether they make money - it is what they cost when they
never fire and what they save when they do. Each is moved on its own, on the settled execution
model, with the other two left at their live values.

The latch is the one that binds everything: it is what makes a risk setting above 1.7% collapse the
account, so its level decides the ceiling on every other number in this project.
"""
import collections
import csv
import datetime
import sys

import numpy as np

sys.path.insert(0, 'C:/Users/Lenovo/Desktop/Торговля/crypto-bot')
import struct_params as SP
import gauntlet2 as G
import live_rules_sim as L
import margin_cap as MC
import sim_w
import exec_model as EM

NOBAD, ALL, YT = EM.NOBAD, EM.ALL, EM.YT
DEPOSIT, LEV, REF = L.DEPOSIT, L.LEV, L.REF
USABLE = MC.USABLE
print('  живые значения: дневная пауза %.1f%%, защёлка %.0f%%, опорный стоп %.4f, плечо %g'
      % (100 * L.MAX_DAILY, 100 * L.MAX_DD, REF, LEV), flush=True)

TR = EM.book(NOBAD, EM.SIG_ALL, 'рынок')[0]
print('  книга: %d сделок (банк + откат, без AAVE и XLM, вход по рынку)' % len(TR), flush=True)


def sim(trades, target, max_daily=None, max_dd=None, stop_ref=None, shrink=True,
        sizing='live'):
    """The live guards, each one switchable. stop_ref=None disables the shrink."""
    # None means "leave at the live value"; 0 means OFF. The first version used None for both
    # meanings, so every "disabled" row silently re-ran the live setting.
    md = L.MAX_DAILY if max_daily is None else max_daily
    dd_lim = L.MAX_DD if max_dd is None else max_dd
    # stop_ref does two jobs in the live code: it sets the base position size AND it is the
    # threshold above which a wide stop shrinks the position. Turning it "off" can only mean the
    # second one - the base size still needs a reference.
    ref = REF if stop_ref is None else stop_ref
    margin_frac = target / (LEV * ref) if sizing == 'live' else None
    eq = peak = DEPOSIT
    open_pos = []
    day, day_start, day_paused = None, DEPOSIT, False
    paused, paused_at = False, None
    curve, taken, skipped = [], 0, 0
    for a, b, R, sf, s, w in trades:
        while open_pos and open_pos[0][0] <= a:
            t_close, m, mpr, rr = open_pos.pop(0)
            eq += mpr * rr
            peak = max(peak, eq)
            curve.append((t_close, eq))
        d = L.day_of(a)
        if d != day:
            day, day_start, day_paused = d, eq, False
        if dd_lim and not paused and eq <= peak * (1 - dd_lim):
            paused, paused_at = True, a
        if md and not day_paused and eq <= day_start * (1 - md):
            day_paused = True
        if paused or day_paused:
            skipped += 1
            continue
        if sizing == 'equal_risk':
            margin = target * eq / (LEV * sf)
        else:
            margin = margin_frac * eq
            if shrink and ref and sf > ref:
                margin *= ref / sf
        used = sum(p[1] for p in open_pos)
        if used + margin > eq * USABLE:
            skipped += 1
            continue
        taken += 1
        open_pos.append((b, margin, margin * LEV * sf, R))
        open_pos.sort(key=lambda p: p[0])
    for t_close, m, mpr, rr in open_pos:
        eq += mpr * rr
        curve.append((t_close, eq))
    return dict(eq=eq, curve=curve, paused_at=paused_at, taken=taken, skipped=skipped)


def at_dd(want=0.12, **kw):
    lo, hi = 0.0005, 0.040
    for _ in range(26):
        mid = (lo + hi) / 2
        r = sim(TR, mid, **kw)
        if bool(r['paused_at']) or abs(L.dd_of(r['curve'])) > want:
            hi = mid
        else:
            lo = mid
    r = sim(TR, lo, **kw)
    return lo, r


def show(lbl, **kw):
    k, r = at_dd(**kw)
    print('  %-38s риск %.3f%%  $%6.0f  просадка %5.1f%%  взято %4d, отказано %3d'
          % (lbl, 100 * k, r['eq'], 100 * abs(L.dd_of(r['curve'])), r['taken'], r['skipped']),
          flush=True)
    return r['eq']


print('', flush=True)
print('  === дневная пауза (при равной просадке 12%) ===', flush=True)
base = show('как в коде: 3%')
for md in (0, 0.02, 0.04, 0.05, 0.08):
    show('дневная пауза %s' % ('ВЫКЛЮЧЕНА' if md == 0 else '%.0f%%' % (100 * md)), max_daily=md)

print('', flush=True)
print('  === опорный стоп (урезание позиции при широком стопе) ===', flush=True)
show('урезание ВЫКЛЮЧЕНО (размер как при 0.0394)', shrink=False)
for sr in (0.02, 0.03, 0.0394, 0.05, 0.08, 0.15):
    show('опорный стоп %.4f' % sr, stop_ref=sr)

print('', flush=True)
print('  === защёлка: где обрыв и что за ним ===', flush=True)
for dd_lim in (0.10, 0.12, 0.15, 0.20, 0.25, 0):
    lbl = 'защёлка %s' % ('ВЫКЛЮЧЕНА' if dd_lim == 0 else '%.0f%%' % (100 * dd_lim))
    best_eq, best_k = 0, 0
    for k in np.arange(0.006, 0.060, 0.001):
        r = sim(TR, float(k), max_dd=dd_lim)
        if r['paused_at']:
            continue
        if r['eq'] > best_eq:
            best_eq, best_k = r['eq'], float(k)
    r = sim(TR, best_k, max_dd=dd_lim)
    print('  %-22s лучший риск %.1f%%  $%7.0f  просадка %5.1f%%'
          % (lbl, 100 * best_k, best_eq, 100 * abs(L.dd_of(r['curve']))), flush=True)
