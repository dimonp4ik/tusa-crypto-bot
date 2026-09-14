"""Does the venue's tape cost the pullback rule more than it costs the bank?

On fifteen coins the gateless rule loses 0.056R per trade when its stops and targets are resolved on
X-Perp candles rather than the feed's. That is either a property of the rule or a haircut the whole
book pays, and the difference decides whether it matters. The bank runs on the same coins, the same
five months and the same two tapes.

Note what this window is: April to August 2026, the only period X-Perp has history for. Twenty-odd
trades for the rule. It cannot price anything. It can only say whether the rule is treated worse
than the bank is.
"""
import collections
import datetime
import pickle
import sys

import numpy as np

sys.path.insert(0, 'C:/Users/Lenovo/Desktop/Торговля/crypto-bot')
import struct_params as SP
import gauntlet2 as G
import nogate_attack as NA
import xperp_nogate as XN
from src import pullback_bank as PB

BIG = 10 ** 9
COST = 0.0004
C = [('rsi48', '>=', 56.3761), ('rsi6', '<=', 33.7947)]


def run_bank(tape):
    rows, seen = [], 0
    for s in XN.COINS:
        c = SP.CTX[s]
        slip = SP.SLIP_REAL.get(s, 0.0002)
        sig = SP.bank_signals([dict(r, tp=1.0) for r in SP.BASE], s, 50)
        if tape == 'xperp':
            t15, A15, pos = XN.XP[s]
        else:
            t15, A15, pos = c['t15'], c['a15'], c['pos']
        lo, hi = int(XN.XP[s][0][0]), int(XN.XP[s][0][-1])
        busy = 0
        for close in sorted(sig):
            if not (lo <= close <= hi) or close < busy:
                continue
            seen += 1
            ri, lim, atr = sig[close]
            lg = SP.BASE[ri].get('side', 'LONG') == 'LONG'
            i = pos.get(close)
            if i is None or i + PB.HOLD_BARS > len(t15):
                continue
            o, h, l, cc = (A15[i:i + PB.HOLD_BARS, x] for x in range(4))
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
                jj = PB.HOLD_BARS - 1
                f_ = cc[-1] * (1 - slip) if lg else cc[-1] * (1 + slip)
            ret = ((f_ / e - 1) if lg else (1 - f_ / e)) - COST
            rows.append((close, int(t15[i + jj]) + 900, ret / (3.0 * atr / e), s))
            busy = int(t15[i + jj]) + 900
    return rows, seen


print('  окно: %s .. %s, монет %d'
      % (datetime.datetime.fromtimestamp(int(min(XN.XP[s][0][0] for s in XN.COINS)),
                                         datetime.UTC).strftime('%Y-%m-%d'),
         datetime.datetime.fromtimestamp(int(max(XN.XP[s][0][-1] for s in XN.COINS)),
                                         datetime.UTC).strftime('%Y-%m-%d'), len(XN.COINS)),
      flush=True)
print('', flush=True)

res = {}
for lbl, fn in (('БАНК', run_bank), ('ОТКАТ без гейта', lambda t: XN.run('без режима', t)[:2])):
    out = {}
    for tape in ('feed', 'xperp'):
        r = fn(tape)
        rows = r[0]
        v = np.array([x[2] for x in rows]) if rows else np.zeros(0)
        out[tape] = {(x[0], x[3]): x[2] for x in rows}
        print('  %-16s лента %-6s сделок %3d ВР%5.1f%% ср%+.4fR'
              % (lbl, tape, len(v), 100 * np.mean(v > 0) if len(v) else 0,
                 v.mean() if len(v) else 0), flush=True)
    keys = sorted(set(out['feed']) & set(out['xperp']))
    a = np.array([out['feed'][k] for k in keys])
    b = np.array([out['xperp'][k] for k in keys])
    d = b - a
    se = d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else float('nan')
    print('  %-16s общих %3d | фид ср%+.4f | X-Perp ср%+.4f | разница %+.4f ± %.4f (%.1f сигмы)'
          % (lbl, len(keys), a.mean(), b.mean(), d.mean(), se,
             abs(d.mean() / se) if se and np.isfinite(se) else 0), flush=True)
    res[lbl] = d
    stops_f = np.mean(a <= -0.9)
    stops_x = np.mean(b <= -0.9)
    print('                   полных стопов: фид %.0f%%, X-Perp %.0f%%'
          % (100 * stops_f, 100 * stops_x), flush=True)
    print('', flush=True)

if len(res) == 2:
    A, B = res['БАНК'], res['ОТКАТ без гейта']
    print('  разница разниц: банк %+.4f, откат %+.4f -> откату хуже на %+.4fR'
          % (A.mean(), B.mean(), B.mean() - A.mean()), flush=True)
