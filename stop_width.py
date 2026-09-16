"""The stop is pierced by a median of 7% of its own depth. Should it stand further out?

This is not the threshold sweep that failed earlier - that one moved the take and the stop as a
pair, searching. This one follows a measured mechanism: 58.8% of stops are shallow pokes, and the
same trades with no stop at all would have been worth +32.5R more, 17% of everything the bank makes.

Risk stays constant: R is the return divided by THIS trade's own stop distance, so a wider stop is
a smaller position, not a bigger bet. That is the honest comparison - otherwise widening the stop
just adds leverage and every number goes up for the wrong reason.

Chosen on 2022-2024 by ratio, read once on 2025-2026. Real per-coin book costs.
"""
import collections
import datetime
import sys

import numpy as np

sys.path.insert(0, 'C:/Users/Lenovo/Desktop/Торговля/crypto-bot')
import struct_params as SP
from src import pullback_bank as PB

FIT_END = int(datetime.datetime(2025, 1, 1, tzinfo=datetime.UTC).timestamp())
YEARS = [2022, 2023, 2024, 2025, 2026]
YT = {y: int(datetime.datetime(y, 1, 1, tzinfo=datetime.UTC).timestamp()) for y in YEARS + [2027]}
BIG = 10 ** 9
COINS = list(SP.COINS)


def run(sl_mult, tp_mult=0.75, coins=None, with_meta=False, keep_tp=False):
    rules = [dict(r, sl=sl_mult, tp=(r['tp'] if keep_tp else tp_mult))
             for r in SP.BASE]
    rows = []
    for s in (COINS if coins is None else coins):
        c = SP.CTX[s]
        slip = SP.SLIP_REAL.get(s, 0.0002)
        t15, A15, pos = c['t15'], c['a15'], c['pos']
        sig = SP.bank_signals(rules, s, 50)
        busy = 0
        for close in sorted(sig):
            if close < busy:
                continue
            i = pos.get(close)
            if i is None or i + PB.HOLD_BARS > len(t15):
                continue
            ri, lim, atr = sig[close]
            r = rules[ri]
            lg = r.get('side', 'LONG') == 'LONG'
            o, h, l, cc = (A15[i:i + PB.HOLD_BARS, x] for x in range(4))
            if lg:
                if o[0] <= lim:
                    e, touch = o[0] * (1 + slip), False
                elif l[0] <= lim:
                    e, touch = lim * (1 + slip), True
                else:
                    continue
                TP, SL = e + r['tp'] * atr, e - r['sl'] * atr
                hs, ht = l <= SL, h >= TP
            else:
                if o[0] >= lim:
                    e, touch = o[0] * (1 - slip), False
                elif h[0] >= lim:
                    e, touch = lim * (1 - slip), True
                else:
                    continue
                TP, SL = e - r['tp'] * atr, e + r['sl'] * atr
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
            ret = ((f_ / e - 1) if lg else (1 - f_ / e)) - 0.0004
            if with_meta:
                rows.append((close, int(t15[i + jj]) + 900,
                             ret / (r['sl'] * atr / e), r['sl'] * atr / e, s))
            else:
                rows.append((close, int(t15[i + jj]) + 900, ret / (r['sl'] * atr / e)))
            busy = int(t15[i + jj]) + 900
    return sorted(rows)


print('  стоп     ОБУЧЕНИЕ 22-24              ЭКЗАМЕН 25-26            стопов', flush=True)
res = {}
for sl in (2.5, 2.75, 3.0, 3.25, 3.5, 4.0, 4.5, 5.0):
    rows = run(sl)
    res[sl] = rows
    f = SP.stats(rows, hi=FIT_END)
    e = SP.stats(rows, lo=FIT_END)
    all_ = SP.stats(rows)
    sr = 100 * np.mean(np.array([r[2] for r in rows]) <= -0.5)
    star = ' <<< в коде' if sl == 3.0 else ''
    print('  %4.2f  ВР%3.0f%% ср%+.4f отн%5.2f    ВР%3.0f%% ср%+.4f отн%5.2f    %4.1f%%%s'
          % (sl, 100 * f['wr'], f['avg'], f['ratio'], 100 * e['wr'], e['avg'], e['ratio'],
             sr, star), flush=True)

best = max(res, key=lambda k: SP.stats(res[k], hi=FIT_END)['ratio'])
print('', flush=True)
print('  выбор по обучению: стоп %.2f ATR (отн %.2f)'
      % (best, SP.stats(res[best], hi=FIT_END)['ratio']), flush=True)
e_best = SP.stats(res[best], lo=FIT_END)
e_base = SP.stats(res[3.0], lo=FIT_END)
print('  ЭКЗАМЕН при нём: ср%+.4fR R/мес%+5.2f DD%+7.1f отн%5.2f'
      % (e_best['avg'], e_best['r_mo'], e_best['dd'], e_best['ratio']), flush=True)
print('  экзамен базы   : ср%+.4fR R/мес%+5.2f DD%+7.1f отн%5.2f'
      % (e_base['avg'], e_base['r_mo'], e_base['dd'], e_base['ratio']), flush=True)
print('', flush=True)
print('  по годам (отношение):', flush=True)
print('  стоп  ' + ''.join('%8d' % y for y in YEARS), flush=True)
for sl in sorted(res):
    cells = ''
    for y in YEARS:
        st = SP.stats(res[sl], lo=YT[y], hi=YT[y + 1])
        cells += ('%8.2f' % st['ratio']) if st else '       -'
    print('  %4.2f  %s' % (sl, cells), flush=True)
