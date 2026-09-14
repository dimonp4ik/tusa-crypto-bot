"""A shorter hold, simulated properly: the stop still works inside the shorter window.

The previous attempt priced the early exit by reading the close at the cut moment and ignoring the
stop, so a trade that breached its stop at hour three and recovered by hour twelve was booked at the
recovered price. That is not a shorter hold - it is trading without a stop for twelve hours, and its
$22,412 is retracted.

Here the whole book is re-simulated with the hold reduced. Every bar between entry and the cut is
walked exactly as before: the stop fires if touched, the target fires if touched, and only a trade
that reaches the cut still open is closed at the market. Both the bank and the rule get the same
hold, since the coin is released by whichever is holding it.

Two things move against each other and the test is which wins: a shorter hold gives up whatever the
trade would have earned in the hours it no longer runs, and frees the coin for the 121 signals a
year the lock currently discards.
"""
import bisect
import collections
import csv
import datetime
import sys

import numpy as np

sys.path.insert(0, 'C:/Users/Lenovo/Desktop/Торговля/crypto-bot')
import struct_params as SP
import gauntlet2 as G
import sim_w
import nogate_attack as NA
from src import pullback_bank as PB

cost = collections.defaultdict(list)
for r in csv.DictReader(open('book_frozen.csv')):
    cost[r['coin'] + 'USDT'].append(float(r['cost109']))
SP.SLIP_REAL.update({k: float(np.median(v)) for k, v in cost.items()})
G.FC.clear()
ALL = list(G.ALL)
NOBAD = [s for s in ALL if s not in ('AAVEUSDT', 'XLMUSDT')]
YT = {y: int(datetime.datetime(y, 1, 1, tzinfo=datetime.UTC).timestamp()) for y in range(2022, 2028)}
RULE = [('rsi48', '>=', 56.3761), ('rsi6', '<=', 33.7947)]
SLICE = 182 * 86400
BIG = 10 ** 9
COST = 0.0004


def signals(coin):
    """Both sources of signals for one coin: (ts, limit, atr, side, tag)."""
    out = {}
    for close, (ri, lim, atr) in SP.bank_signals(
            [dict(r, tp=1.0) for r in SP.BASE], coin, 50).items():
        out[close] = (lim, atr, SP.BASE[ri].get('side', 'LONG') == 'LONG', 'банк')
    for close, (lim, atr) in NA.sigs(coin, 'без режима', RULE).items():
        out.setdefault(close, (lim, atr, True, 'откат'))
    return out


SIG = {s: signals(s) for s in ALL}
print('  сигналов собрано: %d' % sum(len(v) for v in SIG.values()), flush=True)


def book(coins, hold_bars):
    rows, dropped = [], 0
    for s in coins:
        c = SP.CTX[s]
        slip = SP.SLIP_REAL.get(s, 0.0002)
        t15, A15, pos = c['t15'], c['a15'], c['pos']
        busy = 0
        for close in sorted(SIG[s]):
            if close < busy:
                dropped += 1
                continue
            lim, atr, lg, tag = SIG[s][close]
            i = pos.get(close)
            if i is None or i + hold_bars > len(t15):
                continue
            o, h, l, cc = (A15[i:i + hold_bars, x] for x in range(4))
            if lg:
                if o[0] <= lim:
                    e, touch = o[0] * (1 + slip), False
                elif l[0] <= lim:
                    e, touch = lim * (1 + slip), True
                else:
                    continue
                TP, SL = e + 1.0 * atr, e - 3.0 * atr
                hs, ht = l <= SL, h >= TP
            else:
                if o[0] >= lim:
                    e, touch = o[0] * (1 - slip), False
                elif h[0] >= lim:
                    e, touch = lim * (1 - slip), True
                else:
                    continue
                TP, SL = e - 1.0 * atr, e + 3.0 * atr
                hs, ht = h >= SL, l <= TP
            if touch:
                ht = ht.copy()
                ht[0] = False
            js = int(np.argmax(hs)) if hs.any() else BIG
            jt = int(np.argmax(ht)) if ht.any() else BIG
            if js <= jt and js < BIG:
                jj = js
                f_ = (min(SL, o[js]) * (1 - slip)) if lg else (max(SL, o[js]) * (1 + slip))
            elif jt < BIG:
                jj, f_ = jt, (TP * (1 - slip) if lg else TP * (1 + slip))
            else:
                jj = hold_bars - 1
                f_ = cc[-1] * (1 - slip) if lg else cc[-1] * (1 + slip)
            ret = ((f_ / e - 1) if lg else (1 - f_ / e)) - COST
            end = int(t15[i + jj]) + 900
            rows.append((close, end, ret / (3.0 * atr / e), 3.0 * atr / e, s, 1.0))
            busy = end
    rows.sort()
    return rows, dropped


def report(lbl, rows, ref=None):
    k, m = sim_w.money_at_dd(rows, 0.12)
    v = np.array([x[2] for x in rows])
    per = []
    for y in range(2022, 2027):
        sub = [x for x in rows if YT[y] <= x[0] < YT[y + 1]]
        per.append(sim_w.money_at_dd(sub, 0.12)[1] if len(sub) >= 50 else float('nan'))
    wf = ''
    if ref is not None:
        b3 = [(a, b, R) for a, b, R, sf, s, w in ref]
        c3 = [(a, b, R) for a, b, R, sf, s, w in rows]
        t, wr, wd, seen = min(r[0] for r in b3), 0, 0, 0
        last = max(r[0] for r in b3)
        while t < last:
            s0, s1 = SP.stats(b3, lo=t, hi=t + SLICE), SP.stats(c3, lo=t, hi=t + SLICE)
            if s0 and s1:
                seen += 1
                wr += int(s1['r_mo'] > s0['r_mo'])
                wd += int(s1['dd'] > s0['dd'])
            t += SLICE
        wf = ' | R %d/%d просадка %d/%d' % (wr, seen, wd, seen)
    print('  %-24s %4d сд. ВР%5.1f%% ср%+.4f риск %.3f%% $%6.0f | %s%s'
          % (lbl, len(v), 100 * np.mean(v > 0), v.mean(), 100 * k, m,
             ' '.join('%3.0f' % x if np.isfinite(x) else '  -' for x in per), wf), flush=True)
    return m


for cl, coins in (('без AAVE и XLM', NOBAD), ('все 15 монет', ALL)):
    print('', flush=True)
    print('  ===== книга: %s =====' % cl, flush=True)
    base, d0 = book(coins, 192)
    report('удержание 48ч (сейчас)', base)
    for h in (6, 12, 18, 24, 36):
        rows, d = book(coins, h * 4)
        report('удержание %2dч' % h, rows, base)
        print('        отброшено замком %d (было %d)' % (d, d0), flush=True)
