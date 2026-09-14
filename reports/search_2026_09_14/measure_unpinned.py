"""Measure SEIUSDT and LABUSDT, which the live bot would trade and the model never tested.

The exclusion is not about data: SEI has 9.8 months of 15-minute history and LAB has 10.2, while
ZEC - which IS in the model - has 10.0. The line that drops them gives no reason.

Their book cost is unknown, so the answer is bracketed: the cheapest fill any coin in the set gets,
and the most expensive (BILLUSDT's 0.00386, which is what a thin book costs here). If a coin only
pays at the cheap end, it does not pay.
"""
import collections
import datetime
import sys

import numpy as np

sys.path.insert(0, 'C:/Users/Lenovo/Desktop/Торговля/crypto-bot')
import struct_params as SP
import tm_tag as T
import gauntlet2 as G
import sim_w
from src import pullback_bank as PB

EXTRA = ['SEIUSDT', 'LABUSDT']
for s in EXTRA:
    t15, a15 = T.data[s]['T'], T.data[s]['A']
    T1, B1 = PB.build_bars(t15, a15, SP.HOUR)
    F = PB.features(T1, B1, T.bt, T.ba)
    iv = PB.trend_regime(t15, a15, *PB.btc_daily(T.bt, T.ba))
    bdt, bdc = PB.btc_daily(T.bt, T.ba)
    bdc = np.asarray(bdc, dtype=float)
    SP.CTX[s] = dict(t15=t15, a15=a15, T1=T1, B1=B1, F=F, iv=iv, starts=[x[0] for x in iv],
                     bdt=bdt, bdc=bdc, pos=dict(zip(t15.tolist(), range(len(t15)))),
                     sma={n: PB._sma(bdc, n) for n in (20, 30, 50, 80, 120, 200)})
print('  контекст расширен до %d монет' % len(SP.CTX), flush=True)

C = [('rsi48', '>=', 56.3761), ('rsi6', '<=', 33.7947)]
BIG = 10 ** 9


def trades(coin, rules=None, cand=None, slip=None):
    c = SP.CTX[coin]
    sl = SP.SLIP_REAL.get(coin, 0.0002) if slip is None else slip
    t15, A15, pos = c['t15'], c['a15'], c['pos']
    sig = {}
    if rules is not None:
        for close, (ri, lim, atr) in SP.bank_signals([dict(r, tp=1.0) for r in rules],
                                                     coin, 50).items():
            sig[close] = (lim, atr, rules[ri].get('side', 'LONG') == 'LONG', 'bank')
    if cand is not None:
        for close, (lim, atr) in G.signals(coin, cand, True).items():
            sig.setdefault(close, (lim, atr, True, 'cand'))
    out, busy = [], 0
    for close in sorted(sig):
        if close < busy:
            continue
        lim, atr, lgs, tag = sig[close]
        i = pos.get(close)
        if i is None or i + PB.HOLD_BARS > len(t15):
            continue
        o, h, l, cc = (A15[i:i + PB.HOLD_BARS, x] for x in range(4))
        if lgs:
            if o[0] <= lim:
                e, touch = o[0] * (1 + sl), False
            elif l[0] <= lim:
                e, touch = lim * (1 + sl), True
            else:
                continue
            TP, SL = e + 1.0 * atr, e - 3.0 * atr
            hs, ht = l <= SL, h >= TP
        else:
            if o[0] >= lim:
                e, touch = o[0] * (1 - sl), False
            elif h[0] >= lim:
                e, touch = lim * (1 - sl), True
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
            f_ = (min(SL, o[js]) * (1 - sl)) if lgs else (max(SL, o[js]) * (1 + sl))
        elif jt < BIG:
            jj, f_ = jt, (TP * (1 - sl) if lgs else TP * (1 + sl))
        else:
            jj = PB.HOLD_BARS - 1
            f_ = cc[-1] * (1 - sl) if lgs else cc[-1] * (1 + sl)
        ret = ((f_ / e - 1) if lgs else (1 - f_ / e)) - SP.COST
        out.append((close, int(t15[i + jj]) + 900, ret / (3.0 * atr / e), 3.0 * atr / e, coin, 1.0))
        busy = int(t15[i + jj]) + 900
    return out


print('', flush=True)
print('  === SEI и LAB под правилами банка, в вилке издержек ===', flush=True)
for s in EXTRA:
    for slip, lbl in ((0.00005, 'дешёвая заливка'), (0.00077, 'как у AAVE'), (0.00386, 'как у BILL')):
        tr = trades(s, rules=SP.BASE, slip=slip)
        if not tr:
            print('    %-8s %-18s сделок нет' % (s, lbl), flush=True)
            continue
        v = np.array([x[2] for x in tr])
        print('    %-8s %-18s n%4d ВР%5.1f%% ср%+.4fR сумма%+7.1fR'
              % (s.replace('USDT', ''), lbl, len(v), 100 * np.mean(v > 0), v.mean(), v.sum()),
              flush=True)
    tr = trades(s, cand=C, slip=0.00077)
    v = np.array([x[2] for x in tr]) if tr else np.zeros(0)
    print('    %-8s ОТКАТ (как у AAVE)  n%4d %s'
          % (s.replace('USDT', ''), len(v),
             'ВР%5.1f%% ср%+.4fR' % (100 * np.mean(v > 0), v.mean()) if len(v) else ''), flush=True)

print('', flush=True)
print('  === для сравнения: те же 10 месяцев у монет, которые В модели ===', flush=True)
lo = min(SP.CTX['SEIUSDT']['t15'][0], SP.CTX['LABUSDT']['t15'][0])
for s in ('ZECUSDT', 'HYPEUSDT', 'SOLUSDT', 'ADAUSDT', 'AAVEUSDT'):
    tr = [x for x in trades(s, rules=SP.BASE) if x[0] >= lo]
    if not tr:
        continue
    v = np.array([x[2] for x in tr])
    print('    %-8s n%4d ВР%5.1f%% ср%+.4fR' % (s.replace('USDT', ''), len(v),
                                                100 * np.mean(v > 0), v.mean()), flush=True)

print('', flush=True)
print('  === счёт: 15 монет против 17 (те же 10 месяцев, чтобы сравнение было честным) ===',
      flush=True)


def account(coins, with_pb, slip_extra=0.00077):
    rows = []
    for s in coins:
        sl = slip_extra if s in EXTRA else None
        rows += trades(s, rules=SP.BASE, cand=C if with_pb else None, slip=sl)
    rows.sort()
    return [r for r in rows if r[0] >= lo]


BASE15 = [c for c in SP.COINS if c != 'BILLUSDT']
for lbl, coins in (('15 монет модели', BASE15),
                   ('17 (плюс SEI и LAB)', BASE15 + EXTRA),
                   ('14 (без AAVE)', [c for c in BASE15 if c != 'AAVEUSDT']),
                   ('16 (без AAVE, плюс SEI и LAB)',
                    [c for c in BASE15 if c != 'AAVEUSDT'] + EXTRA)):
    for with_pb in (False, True):
        rows = account(coins, with_pb)
        v = np.array([r[2] for r in rows])
        k, m = sim_w.money_at_dd(rows, 0.12)
        print('    %-32s %-12s n%4d ср%+.4f риск %.3f%% $%6.0f'
              % (lbl, 'банк+откат' if with_pb else 'банк', len(v), v.mean(), 100 * k, m),
              flush=True)
