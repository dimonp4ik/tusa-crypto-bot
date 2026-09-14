"""Why does the gateless variant make more money and win fewer half-years?

A pullback trade holds its coin for up to 48 hours, so a bank signal arriving inside that window is
silently dropped. The gateless rule takes a third more trades, so it should displace more of the
bank - and the bank's own trades are the ones carrying R per month, which is the axis where the
gateless variant slips from 7 of 9 to 6 of 9.

This counts the displacement directly, and then tries the obvious remedies:

    приоритет банка    - a pullback trade is skipped if the bank has a signal on that coin within
                         the next N hours. Not implementable live (it reads the future), but it
                         bounds how much of the gap displacement explains.
    короче удержание   - the pullback rule exits sooner, freeing the coin
    только свободные   - the pullback rule takes a coin only if the bank has not traded it recently
"""
import collections
import csv
import datetime
import sys

import numpy as np

sys.path.insert(0, 'C:/Users/Lenovo/Desktop/Торговля/crypto-bot')
import struct_params as SP
import sim_w
import assemble as A

ALL = [s for s in A.ALL if A.MED_SPREAD.get(s, 0) <= 0.0005]
SLICE = A.SLICE


def build(mode, hold_cap=None, avoid_h=0):
    """hold_cap: seconds - the rule's trade is cut short to free the coin.
       avoid_h: skip a rule trade if a bank signal lands within this many hours (future-reading)."""
    bank, cand = [], []
    for s in ALL:
        bank += [(a, b, R, sf, s) for a, b, R, sf, s2 in A.BANK[s]]
        if mode:
            cand += [(a, b, R, sf, s) for a, b, R, sf, s2 in A.CAND[mode][s]]
    bank.sort()
    bank_by_coin = collections.defaultdict(list)
    for a, b, R, sf, s in bank:
        bank_by_coin[s].append(a)
    ev = [(a, b, R, sf, s, 0) for a, b, R, sf, s in bank]
    for a, b, R, sf, s in cand:
        if avoid_h:
            nxt = [x for x in bank_by_coin[s] if a < x <= a + avoid_h * 3600]
            if nxt:
                continue
        bb = min(b, a + hold_cap) if hold_cap else b
        ev.append((a, bb, R, sf, s, 1))
    ev.sort(key=lambda x: (x[0], x[5]))
    busy, out, dropped = {}, [], 0
    for a, b, R, sf, s, k in ev:
        if busy.get(s, 0) > a:
            if k == 0:
                dropped += 1
            continue
        busy[s] = b
        out.append((a, b, R, sf, s, 1.0, k))
    out.sort()
    return out, dropped


def show(lbl, rows, dropped, ref_rows):
    plain = [(a, b, R, sf, s, w) for a, b, R, sf, s, w, k in rows]
    k, m = sim_w.money_at_dd(plain, 0.12)
    nb = sum(1 for r in rows if r[6] == 0)
    nc = len(rows) - nb
    b3 = [(a, b, R) for a, b, R, sf, s, w, kk in ref_rows]
    c3 = [(a, b, R) for a, b, R, sf, s, w, kk in rows]
    t, wr, wd, seen = min(r[0] for r in b3), 0, 0, 0
    last = max(r[0] for r in b3)
    while t < last:
        s0, s1 = SP.stats(b3, lo=t, hi=t + SLICE), SP.stats(c3, lo=t, hi=t + SLICE)
        if s0 and s1:
            seen += 1
            wr += int(s1['r_mo'] > s0['r_mo'])
            wd += int(s1['dd'] > s0['dd'])
        t += SLICE
    print('  %-38s банка %4d, правила %3d, вытеснено банка %3d | $%6.0f | R %d/%d просадка %d/%d'
          % (lbl, nb, nc, dropped, m, wr, seen, wd, seen), flush=True)
    return m


ref, d0 = build(None)
print('  банк один: %d сделок, вытеснено самим собой %d' % (len(ref), d0), flush=True)
print('', flush=True)
show('банк один', ref, d0, ref)
show('+ откат в гейте', *build('гейт'), ref)
show('+ откат без гейта', *build('без режима'), ref)
print('', flush=True)
print('  === лечение вытеснения ===', flush=True)
for h in (12, 24, 48):
    show('без гейта, уступать банку за %2dч' % h, *build('без режима', avoid_h=h), ref)
for cap in (6, 12, 24):
    show('без гейта, удержание правила %2dч' % cap,
         *build('без режима', hold_cap=cap * 3600), ref)
print('', flush=True)
for cap in (6, 12, 24):
    show('в гейте, удержание правила %2dч' % cap, *build('гейт', hold_cap=cap * 3600), ref)
