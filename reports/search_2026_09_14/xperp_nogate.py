"""The gateless pullback rule executed on the venue's own candles, not the feed's.

The model reads levels and touches from the same SWAP feed, which makes a fill look certain whenever
the level was touched on that tape. The bank has already been caught by this once: on real X-Perp
candles only 79% of its trades filled at all. The rule is worth nothing if it only fills on the tape
it was measured on.

Six coins have X-Perp history (ADA, AVAX, BTC, ETH, SOL, XRP) covering about six months, so this is
a sample, not a verdict. The signals are generated exactly as before on the feed - that is what the
live bot does, it decides from the feed - and then the ENTRY, the stop, the target and the exit are
all resolved on the X-Perp tape.

Three numbers matter: how many signals fill at all, what the filled ones return, and whether the
rule's edge survives the subset that fills.
"""
import bisect
import collections
import datetime
import pickle
import sys

import numpy as np

sys.path.insert(0, 'C:/Users/Lenovo/Desktop/Торговля/crypto-bot')
import struct_params as SP
import gauntlet2 as G
import nogate_attack as NA
from src import pullback_bank as PB

CACHE = 'C:/Users/Lenovo/Desktop/Торговля/crypto-bot/reports/audit_2026_09_08/xperp_cache'
COINS = ['ADAUSDT', 'AVAXUSDT', 'BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'XRPUSDT']
C = [('rsi48', '>=', 56.3761), ('rsi6', '<=', 33.7947)]
BIG = 10 ** 9
COST = 0.0004

XP = {}
for s in COINS:
    d = pickle.load(open('%s/%s_15min_18000.pkl' % (CACHE, s), 'rb'))
    t = np.array(d['time'], dtype=np.int64)
    A = np.column_stack([np.array(d[k], dtype=float) for k in ('open', 'high', 'low', 'close')])
    XP[s] = (t, A, {int(x): i for i, x in enumerate(t)})
    f = lambda x: datetime.datetime.fromtimestamp(int(x), datetime.UTC).strftime('%Y-%m-%d')
    print('  %-8s свечей X-Perp %5d  %s .. %s' % (s, len(t), f(t[0]), f(t[-1])), flush=True)


def run(mode, tape):
    """tape: 'feed' or 'xperp'. Signals always come from the feed, as the live bot decides."""
    rows, missed, seen = [], 0, 0
    for s in COINS:
        sig = NA.sigs(s, mode, C)
        c = SP.CTX[s]
        slip = SP.SLIP_REAL.get(s, 0.0002)
        if tape == 'xperp':
            t15, A15, pos = XP[s]
            lo, hi = int(t15[0]), int(t15[-1])
        else:
            t15, A15, pos = c['t15'], c['a15'], c['pos']
            lo, hi = int(XP[s][0][0]), int(XP[s][0][-1])
        busy = 0
        for close in sorted(sig):
            if not (lo <= close <= hi):
                continue
            if close < busy:
                continue
            seen += 1
            lim, atr = sig[close]
            i = pos.get(close)
            if i is None or i + PB.HOLD_BARS > len(t15):
                continue
            o, h, l, cc = (A15[i:i + PB.HOLD_BARS, x] for x in range(4))
            if o[0] <= lim:
                e, touch = o[0] * (1 + slip), False
            elif l[0] <= lim:
                e, touch = lim * (1 + slip), True
            else:
                missed += 1
                continue
            TP, SL = e + 1.0 * atr, e - 3.0 * atr
            hs, ht = l <= SL, h >= TP
            if touch:
                ht = ht.copy()
                ht[0] = False
            js = int(np.argmax(hs)) if hs.any() else BIG
            jt = int(np.argmax(ht)) if ht.any() else BIG
            if js <= jt and js < BIG:
                jj, f_ = js, min(SL, o[js]) * (1 - slip)
            elif jt < BIG:
                jj, f_ = jt, TP * (1 - slip)
            else:
                jj, f_ = PB.HOLD_BARS - 1, cc[-1] * (1 - slip)
            ret = (f_ / e - 1) - COST
            rows.append((close, int(t15[i + jj]) + 900, ret / (3.0 * atr / e), s))
            busy = int(t15[i + jj]) + 900
    return rows, missed, seen


print('', flush=True)
for mode in ('без режима', 'гейт'):
    out = {}
    for tape in ('feed', 'xperp'):
        rows, missed, seen = run(mode, tape)
        v = np.array([r[2] for r in rows]) if rows else np.zeros(0)
        out[tape] = {(r[0], r[3]) for r in rows}
        print('  %-12s лента %-6s сигналов %3d, залилось %3d (%3.0f%%), ВР%5.1f%% ср%+.4fR сумма%+6.1fR'
              % (mode, tape, seen, len(v), 100 * len(v) / max(seen, 1),
                 100 * np.mean(v > 0) if len(v) else 0, v.mean() if len(v) else 0,
                 v.sum() if len(v) else 0), flush=True)
    both = out['feed'] & out['xperp']
    print('               общих сделок на обеих лентах: %d' % len(both), flush=True)

print('', flush=True)
print('  === только те сделки, что залились на ОБЕИХ лентах ===', flush=True)
for mode in ('без режима', 'гейт'):
    rf, _, _ = run(mode, 'feed')
    rx, _, _ = run(mode, 'xperp')
    kf = {(r[0], r[3]): r[2] for r in rf}
    kx = {(r[0], r[3]): r[2] for r in rx}
    keys = sorted(set(kf) & set(kx))
    if not keys:
        continue
    a = np.array([kf[k] for k in keys])
    b = np.array([kx[k] for k in keys])
    print('  %-12s n%3d | лента фида ср%+.4f ВР%5.1f%% | лента X-Perp ср%+.4f ВР%5.1f%% | разница%+.4f'
          % (mode, len(keys), a.mean(), 100 * np.mean(a > 0), b.mean(), 100 * np.mean(b > 0),
             b.mean() - a.mean()), flush=True)
