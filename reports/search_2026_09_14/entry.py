"""How the trade is entered, which has never been varied.

A signal fires at the close of an hour and the bot places its limit at that close. The next fifteen
minutes decide: if price opens at or below the level it fills at the open, if it trades down to the
level it fills there, and if it runs away it never fills at all. Those runaways are simply lost, and
nobody has counted them.

Four ways in, all implementable:

    лимит на закрытии часа   what happens now
    по рынку                 take the next open, whatever it is - never miss, always pay up
    лимит, ждать дольше      keep the order alive for several bars instead of one
    лимит чуть выше          place it above the close so it fills more often, at a worse price

The stop and target are measured from the actual fill in every case, so a worse entry carries a
worse geometry with it rather than being hidden.
"""
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
HOLD = 192


def signals(coin):
    out = {}
    for close, (ri, lim, atr) in SP.bank_signals(
            [dict(r, tp=1.0) for r in SP.BASE], coin, 50).items():
        out[close] = (lim, atr, SP.BASE[ri].get('side', 'LONG') == 'LONG')
    for close, (lim, atr) in NA.sigs(coin, 'без режима', RULE).items():
        out.setdefault(close, (lim, atr, True))
    return out


SIG = {s: signals(s) for s in ALL}
print('  сигналов: %d' % sum(len(v) for v in SIG.values()), flush=True)


def book(coins, how='лимит', wait=1, offset=0.0):
    """how: 'лимит' wait for the level, 'рынок' take the next open."""
    rows, missed, taken = [], 0, 0
    for s in coins:
        c = SP.CTX[s]
        slip = SP.SLIP_REAL.get(s, 0.0002)
        t15, A15, pos = c['t15'], c['a15'], c['pos']
        busy = 0
        for close in sorted(SIG[s]):
            if close < busy:
                continue
            lim0, atr, lg = SIG[s][close]
            lim = lim0 * (1 + offset) if lg else lim0 * (1 - offset)
            i = pos.get(close)
            if i is None or i + HOLD > len(t15):
                continue
            e = None
            if how == 'рынок':
                e = A15[i, 0] * (1 + slip) if lg else A15[i, 0] * (1 - slip)
                k0 = i
            else:
                for k in range(i, min(i + wait, len(t15))):
                    o_, h_, l_ = A15[k, 0], A15[k, 1], A15[k, 2]
                    if lg:
                        if o_ <= lim:
                            e, k0 = o_ * (1 + slip), k
                            break
                        if l_ <= lim:
                            e, k0 = lim * (1 + slip), k
                            break
                    else:
                        if o_ >= lim:
                            e, k0 = o_ * (1 - slip), k
                            break
                        if h_ >= lim:
                            e, k0 = lim * (1 - slip), k
                            break
                if e is None:
                    missed += 1
                    continue
            if k0 + HOLD > len(t15):
                continue
            o, h, l, cc = (A15[k0:k0 + HOLD, x] for x in range(4))
            touch = not (o[0] * (1 + slip) == e or o[0] * (1 - slip) == e)
            if lg:
                TP, SL = e + 1.0 * atr, e - 3.0 * atr
                hs, ht = l <= SL, h >= TP
            else:
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
                jj = HOLD - 1
                f_ = cc[-1] * (1 - slip) if lg else cc[-1] * (1 + slip)
            ret = ((f_ / e - 1) if lg else (1 - f_ / e)) - COST
            end = int(t15[k0 + jj]) + 900
            rows.append((close, end, ret / (3.0 * atr / e), 3.0 * atr / e, s, 1.0))
            taken += 1
            busy = end
    rows.sort()
    return rows, missed


def report(lbl, rows, missed, ref=None):
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
    print('  %-28s %4d сд. (не влезло %4d) ВР%5.1f%% ср%+.4f $%6.0f | %s%s'
          % (lbl, len(v), missed, 100 * np.mean(v > 0), v.mean(), m,
             ' '.join('%3.0f' % x if np.isfinite(x) else '  -' for x in per), wf), flush=True)
    return m


for cl, coins in (('без AAVE и XLM', NOBAD), ('все 15 монет', ALL)):
    print('', flush=True)
    print('  ===== книга: %s =====' % cl, flush=True)
    base, mb = book(coins, 'лимит', 1)
    report('лимит, 1 бар (сейчас)', base, mb)
    report('по рынку сразу', *book(coins, 'рынок'), base)
    for w in (2, 4, 8):
        report('лимит, ждать %d баров' % w, *book(coins, 'лимит', w), base)
    for off in (0.0005, 0.0015, 0.003):
        report('лимит выше на %.2f%%' % (100 * off), *book(coins, 'лимит', 1, off), base)
