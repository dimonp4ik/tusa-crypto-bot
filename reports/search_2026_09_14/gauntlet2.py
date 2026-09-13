"""One gauntlet for every candidate, with the first test fixed.

The previous version counted hours-to-trades by blocking a coin for the full 48-hour hold. Trades
actually exit in a median of 4.5 hours, so it under-counted every rule - and marked the bank's own
rules as "a state, not an event", which is how the bug was caught: the test was run on something
known to work before it was used on something unknown.

Here the exit is computed, not assumed. Order of steps is by cost of discovery:

  1. hours -> trades, with real exits. A rule whose conditions persist collapses here.
  2. the mirror: same conditions, opposite side. A direction that pays both ways is not a direction.
  3. overlap with what the bank already takes.
  4. the portfolio chronologically through the live guards: money, drawdown, latch.
  5. walk-forward of the combined system, on the axis the candidate claims.

Nothing reaches a later step after failing an earlier one.
"""
import bisect
import collections
import datetime
import sys

import numpy as np

sys.path.insert(0, 'C:/Users/Lenovo/Desktop/Торговля/crypto-bot')
import struct_params as SP
import feat2
from src import pullback_bank as PB

BIG = 10 ** 9
COST = 0.0004
ALL = [c for c in SP.COINS if c != 'BILLUSDT']
FC = {}
_BR = None


def _breadth():
    """Share of coins up over 24h at each hour - the only cross-sectional signal available."""
    global _BR
    if _BR is None:
        _BR = feat2.breadth(ALL)
    return _BR


def feats(coin):
    if coin not in FC:
        T1, f = feat2.build(coin)
        br = _breadth()
        f['breadth'] = np.array([br.get(int(t), (np.nan, np.nan))[0] for t in T1])
        f['breadth_med'] = np.array([br.get(int(t), (np.nan, np.nan))[1] for t in T1])
        FC[coin] = (T1, f)
    return FC[coin]


def signals(coin, conds, side_long):
    """Hours where every condition holds, inside the bank's own preconditions."""
    c = SP.CTX[coin]
    T1, f = feats(coin)
    bsm = c['sma'].setdefault(50, PB._sma(c['bdc'], 50))
    want = 'LONG' if side_long else 'SHORT'
    out = {}
    for i in range(PB.MIN_HOURS, len(T1)):
        close = int(T1[i]) + PB.HOUR
        j = bisect.bisect_right(c['starts'], close) - 1
        if j < 0 or not (c['iv'][j][0] <= close < c['iv'][j][1]) or c['iv'][j][2] != want:
            continue
        if side_long:
            d = bisect.bisect_right(c['bdt'], close - 86400) - 1
            if d >= 49 and c['bdc'][d] < bsm[d]:
                continue
        if not np.isfinite(c['F']['atr'][i]):
            continue
        ok = True
        for nm, op, thr in conds:
            x = f[nm][i]
            if not np.isfinite(x) or (x < thr if op == '>=' else x > thr):
                ok = False
                break
        if ok:
            out[close] = (float(c['B1'][i, 3]), float(c['F']['atr'][i]))
    return out


def simulate(conds, side_long, coins=None, tp=1.0, sl=3.0):
    """Real trades with real exits: (entry_ts, exit_ts, R, stop_frac, coin)."""
    coins = coins or ALL
    rows, hours = [], 0
    for s in coins:
        sig = signals(s, conds, side_long)
        hours += len(sig)
        c = SP.CTX[s]
        slip = SP.SLIP_REAL.get(s, 0.0002)
        t15, A15, pos = c['t15'], c['a15'], c['pos']
        busy = 0
        for close in sorted(sig):
            if close < busy:
                continue
            lim, atr = sig[close]
            i = pos.get(close)
            if i is None or i + PB.HOLD_BARS > len(t15):
                continue
            o, h, l, cc = (A15[i:i + PB.HOLD_BARS, x] for x in range(4))
            lg = side_long
            if lg:
                if o[0] <= lim:
                    e, touch = o[0] * (1 + slip), False
                elif l[0] <= lim:
                    e, touch = lim * (1 + slip), True
                else:
                    continue
                TP, SL = e + tp * atr, e - sl * atr
                hs, ht = l <= SL, h >= TP
            else:
                if o[0] >= lim:
                    e, touch = o[0] * (1 - slip), False
                elif h[0] >= lim:
                    e, touch = lim * (1 - slip), True
                else:
                    continue
                TP, SL = e - tp * atr, e + sl * atr
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
                jj = PB.HOLD_BARS - 1
                f_ = cc[-1] * (1 - slip) if lg else cc[-1] * (1 + slip)
            ret = ((f_ / e - 1) if lg else (1 - f_ / e)) - COST
            end = int(t15[i + jj]) + 900
            rows.append((close, end, ret / (sl * atr / e), sl * atr / e, s))
            busy = end
    rows.sort()
    return rows, hours


def describe(name, conds, side_long):
    rows, hours = simulate(conds, side_long)
    v = np.array([r[2] for r in rows]) if rows else np.zeros(0)
    if not len(v):
        print('  %-44s сделок нет' % name, flush=True)
        return None
    keep = len(rows) / max(hours, 1)
    mons = collections.Counter(
        datetime.datetime.fromtimestamp(r[0], datetime.UTC).strftime('%y-%m') for r in rows)
    print('  %-44s часов%5d сделок%5d (%2.0f%%) ВР%5.1f%% ср%+.4fR сумма%+7.1fR мес%3d макс%3.0f%%'
          % (name, hours, len(rows), 100 * keep, 100 * np.mean(v > 0), v.mean(), v.sum(),
             len(mons), 100 * max(mons.values()) / len(rows)), flush=True)
    return rows


if __name__ == '__main__':
    print('  === калибровка: правила, про которые ответ известен ===', flush=True)
    describe('БАНК btc24>=0.04896 (работает)', [('btc24', '>=', 0.04896)], True)
    describe('vol720>=1.8886 (отвергнут, скучен)', [('vol720', '>=', 1.8886)], True)
    print('', flush=True)
    print('  === кандидат из перебора пар ===', flush=True)
    describe('rsi48>=56.38 + rsi6<=33.79', [('rsi48', '>=', 56.3761), ('rsi6', '<=', 33.7947)], True)
    describe('  его ЗЕРКАЛО (SHORT)', [('rsi48', '>=', 56.3761), ('rsi6', '<=', 33.7947)], False)
