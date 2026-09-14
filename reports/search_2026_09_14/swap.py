"""Close the weaker position to take the better signal - the implementable version.

Two positions on one coin is worth +33% in the model and cannot be done: OKX runs this account in
net mode, so two longs on one instrument merge into one and cannot carry two stops. The loss it
points at is real all the same - 121 signals a year are discarded because the coin is busy, and they
average +0.1180R against +0.1040R for the ones taken.

The implementable form is a swap: when a better signal arrives on a busy coin, close what is open at
that moment and enter the new one. One position per instrument throughout, which the exchange
supports.

The earlier attempt at this was wrong and its number is retracted: it deleted the open trade instead
of closing it, so the position's profit or loss between opening and the swap simply vanished. Here
the early exit is priced on the actual 15-minute bars, and the swap pays the exit cost and the new
entry cost like any other trade.

"Better" must be causal. The only thing known at the moment of the swap is which rule fired, and
what that rule has historically returned - not what this trade will do.
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
import stop_width as SW
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
COST = 0.0004
PRIOR = {'банк': 0.0843, 'откат': 0.1521}


def early_exit_R(coin, entry_ts, exit_ts, R_full, sf, lg=True):
    """What the trade was worth if closed at exit_ts instead of running to its own end.

    Reconstructs the entry price from the coin's own bars and reads the close at the swap moment.
    The stop distance is unchanged, so R stays comparable.
    """
    c = SP.CTX[coin]
    t15, A15, pos = c['t15'], c['a15'], c['pos']
    i = pos.get(entry_ts)
    if i is None:
        return None
    j = bisect.bisect_right(t15, exit_ts) - 1
    if j <= i or j >= len(t15):
        return None
    slip = SP.SLIP_REAL.get(coin, 0.0002)
    e = A15[i, 0] * (1 + slip) if lg else A15[i, 0] * (1 - slip)
    px = A15[j, 3]
    f_ = px * (1 - slip) if lg else px * (1 + slip)
    ret = ((f_ / e - 1) if lg else (1 - f_ / e)) - COST
    return ret / sf


def build(coins, mode='нет'):
    """mode: 'нет' = first come keeps the coin; 'обмен' = a better-prior signal swaps it out."""
    bank = SW.run(3.0, 1.0, coins=coins, with_meta=True)
    ev = [(x[0], x[1], x[2], x[3], x[4], 'банк') for x in bank]
    ev += [(x[0], x[1], x[2], x[3], x[4], 'откат')
           for x in NA.trades('без режима', RULE, coins=coins)]
    ev.sort()
    open_pos = {}          # coin -> (exit_ts, tag, index, entry_ts, sf)
    out, swaps, dropped = [], 0, 0
    for a, b, R, sf, s, tag in ev:
        cur = open_pos.get(s)
        if cur and cur[0] <= a:
            cur = None
            open_pos.pop(s, None)
        if cur is not None:
            if mode != 'обмен' or PRIOR[tag] <= PRIOR[cur[1]]:
                dropped += 1
                continue
            r_early = early_exit_R(s, cur[3], a, None, cur[4])
            if r_early is None:
                dropped += 1
                continue
            out[cur[2]] = (cur[3], a, r_early, cur[4], s, 1.0)   # closed at the swap moment
            swaps += 1
        open_pos[s] = (b, tag, len(out), a, sf)
        out.append((a, b, R, sf, s, 1.0))
    out.sort()
    return out, swaps, dropped


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
    print('  %-32s %4d сд. ВР%5.1f%% ср%+.4f риск %.3f%% $%6.0f | %s%s'
          % (lbl, len(v), 100 * np.mean(v > 0), v.mean(), 100 * k, m,
             ' '.join('%3.0f' % x if np.isfinite(x) else '  -' for x in per), wf), flush=True)
    return m


for cl, coins in (('без AAVE и XLM', NOBAD), ('все 15 монет', ALL)):
    print('', flush=True)
    print('  ===== книга: %s =====' % cl, flush=True)
    base, _, d0 = build(coins, 'нет')
    m0 = report('как сейчас (кто раньше, того монета)', base)
    sw, n_sw, d1 = build(coins, 'обмен')
    report('обмен на лучший сигнал', sw, base)
    print('    обменов %d, всё ещё отброшено %d (было %d)' % (n_sw, d1, d0), flush=True)
    cut = [x for x in sw if x[1] - x[0] < 47 * 3600 and x[2] is not None]
    early = [x for x in sw if x not in base]
    print('    сделки, закрытые досрочно ради обмена: %d' % n_sw, flush=True)
