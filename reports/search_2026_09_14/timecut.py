"""Release the coin early by actually closing the trade, not by pretending.

The earlier version of this freed the coin after N hours while still collecting the 48-hour outcome,
which reads the future and is retracted. Here the trade really ends at N hours: the exit is priced on
the coin's own bars at that moment and pays the exit cost.

Two effects pull against each other. Closing early gives up whatever the trade would have done in
the hours that remain, and this project has already measured that expectation decays with time but
stays positive for the first several hours. Against that, the coin is free sooner, so the 121
signals a year currently discarded can be taken.

A time stop was tested and rejected for this bot once before, but for a different purpose - as a way
to cut losers. Here the purpose is to free the slot, so the question is new even if the knob is not.
"""
import collections, csv, datetime, sys
import numpy as np
sys.path.insert(0, 'C:/Users/Lenovo/Desktop/Торговля/crypto-bot')
import struct_params as SP
import gauntlet2 as G
import stop_width as SW
import sim_w
import nogate_attack as NA
import swap as SWAP

ALL, NOBAD, YT = SWAP.ALL, SWAP.NOBAD, SWAP.YT
RULE = SWAP.RULE
SLICE = SWAP.SLICE

def build(coins, cut_h=None):
    bank = SW.run(3.0, 1.0, coins=coins, with_meta=True)
    ev = [(x[0], x[1], x[2], x[3], x[4], 'банк') for x in bank]
    ev += [(x[0], x[1], x[2], x[3], x[4], 'откат')
           for x in NA.trades('без режима', RULE, coins=coins)]
    ev.sort()
    busy, out, dropped, cut = {}, [], 0, 0
    for a, b, R, sf, s, tag in ev:
        if busy.get(s, 0) > a:
            dropped += 1
            continue
        bb, RR = b, R
        if cut_h and b - a > cut_h * 3600:
            r_e = SWAP.early_exit_R(s, a, a + cut_h * 3600, None, sf)
            if r_e is not None:
                bb, RR, = a + cut_h * 3600, r_e
                cut += 1
        busy[s] = bb
        out.append((a, bb, RR, sf, s, 1.0))
    out.sort()
    return out, dropped, cut

def report(lbl, rows, ref=None):
    k, m = sim_w.money_at_dd(rows, 0.12)
    v = np.array([x[2] for x in rows])
    per = []
    for y in range(2022, 2027):
        sub = [x for x in rows if YT[y] <= x[0] < YT[y+1]]
        per.append(sim_w.money_at_dd(sub, 0.12)[1] if len(sub) >= 50 else float('nan'))
    wf = ''
    if ref is not None:
        b3 = [(a,b,R) for a,b,R,sf,s,w in ref]; c3 = [(a,b,R) for a,b,R,sf,s,w in rows]
        t, wr, wd, seen = min(r[0] for r in b3), 0, 0, 0
        last = max(r[0] for r in b3)
        while t < last:
            s0, s1 = SP.stats(b3, lo=t, hi=t+SLICE), SP.stats(c3, lo=t, hi=t+SLICE)
            if s0 and s1:
                seen += 1; wr += int(s1['r_mo'] > s0['r_mo']); wd += int(s1['dd'] > s0['dd'])
            t += SLICE
        wf = ' | R %d/%d просадка %d/%d' % (wr, seen, wd, seen)
    print('  %-30s %4d сд. ВР%5.1f%% ср%+.4f риск %.3f%% $%6.0f | %s%s'
          % (lbl, len(v), 100*np.mean(v>0), v.mean(), 100*k, m,
             ' '.join('%3.0f' % x if np.isfinite(x) else '  -' for x in per), wf), flush=True)
    return m

for cl, coins in (('без AAVE и XLM', NOBAD), ('все 15 монет', ALL)):
    print('', flush=True); print('  ===== книга: %s =====' % cl, flush=True)
    base, d0, _ = build(coins)
    report('без обрезки (48ч)', base)
    for cut in (6, 12, 18, 24, 36):
        rows, d, n = build(coins, cut)
        m = report('закрывать через %2dч' % cut, rows, base)
        print('        обрезано сделок %d, отброшено сигналов %d (было %d)' % (n, d, d0), flush=True)
