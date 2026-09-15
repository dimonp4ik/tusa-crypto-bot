"""Is the fast-bar collapse a cost problem or a missing pattern?

On 15m bars the pullback rule trades three times as often and earns nothing. Two explanations predict
different things. If costs eat it, the rule with zero costs keeps a strong edge on fast bars, and a
cheaper execution (a resting limit entry paying maker fee, no entry slippage) recovers part of it.
If the pattern simply does not exist on fast bars, zero costs do not rescue it either.

Three cost levels on each bar size:
  ноль     no slippage, no fee - the pattern's raw edge
  мейкер   limit entry at the signal close price (fills only if touched), maker fee in, taker out,
           measured slippage on the exit leg only
  тейкер   market entry and exit, measured slippage both legs, 4bp fees (today's live model)
"""
import collections, csv, sys
import numpy as np
sys.path.insert(0, 'C:/Users/Lenovo/Desktop/Торговля/crypto-bot')
import struct_params as SP
import sim_w
import live_rules_sim as L
from src import pullback_bank as PB

cost = collections.defaultdict(list)
for r in csv.DictReader(open('book_frozen.csv')):
    cost[r['coin'] + 'USDT'].append(float(r['cost109']))
SLIP = {k: float(np.median(v)) for k, v in cost.items()}
COINS = [s for s in SP.COINS if s not in ('BILLUSDT', 'XLMUSDT', 'AAVEUSDT')]
BIG = 10 ** 9


def run(sec, mode):
    rows, missed = [], 0
    for s in COINS:
        c = SP.CTX[s]
        t15, a15, pos = c['t15'], c['a15'], c['pos']
        T, B = PB.build_bars(t15, a15, sec)
        cl = B[:, 3]
        r48, r6, atr = PB._rsi(cl, 48), PB._rsi(cl, 6), PB._atr(B)
        bdt, bdc = c['bdt'], np.asarray(c['bdc'], dtype=float)
        sma = PB._sma(bdc, 50)
        ms = SLIP.get(s, 0.0003)
        hold = 48 * sec // 900
        busy = 0
        for i in range(200, len(T)):
            if not (r48[i] >= 56.3761 and r6[i] <= 33.7947) or not np.isfinite(atr[i]):
                continue
            close = int(T[i]) + sec
            if close < busy:
                continue
            d = np.searchsorted(bdt, close - 86400, side='right') - 1
            if d >= 49 and bdc[d] < sma[d]:
                continue
            j = pos.get(close)
            if j is None or j + hold > len(t15):
                continue
            o, h, l, cc = (a15[j:j + hold, x] for x in range(4))
            if mode == 'ноль':
                e, ex_slip, fee, first = o[0], 0.0, 0.0, 0
            elif mode == 'тейкер':
                e, ex_slip, fee, first = o[0] * (1 + ms), ms, 0.0004, 0
            else:  # мейкер: rest a limit at the bar close for one 15m bar
                lim = cl[i]
                if o[0] <= lim:
                    e = o[0]
                elif l[0] <= lim:
                    e = lim
                else:
                    missed += 1
                    continue
                ex_slip, fee, first = ms, 0.0003, 0
            TP, SL = e + atr[i], e - 3 * atr[i]
            hs, ht = l <= SL, h >= TP
            js = int(np.argmax(hs)) if hs.any() else BIG
            jt = int(np.argmax(ht)) if ht.any() else BIG
            if js <= jt and js < BIG:
                jj, f = js, min(SL, o[js]) * (1 - ex_slip)
            elif jt < BIG:
                jj, f = jt, TP * (1 - ex_slip)
            else:
                jj, f = hold - 1, cc[-1] * (1 - ex_slip)
            sf = 3 * atr[i] / e
            end = int(t15[j + jj]) + 900
            rows.append((close, end, (f / e - 1 - fee) / sf, sf, s, 1.0))
            busy = end
    rows.sort()
    return rows, missed


print('  свеча  издержки   сделок/год   ВР      ср R    цена сделки в R   $ при 1.4%   просадка', flush=True)
for sec in (3600, 1800, 900):
    for mode in ('ноль', 'мейкер', 'тейкер'):
        rows, missed = run(sec, mode)
        v = np.array([x[2] for x in rows])
        years = (rows[-1][0] - rows[0][0]) / (365.25 * 86400)
        sf = np.median([x[3] for x in rows])
        c_r = {'ноль': 0.0, 'мейкер': 0.0003 + np.median(list(SLIP.values())),
               'тейкер': 0.0004 + 2 * np.median(list(SLIP.values()))}[mode] / sf
        r = sim_w.simulate(rows, 0.014)
        print('  %3dм   %-8s   %6.0f       %4.1f%%  %+.4f     %.3f          $%7.0f   %5.1f%%%s%s'
              % (sec // 60, mode, len(v) / years, 100 * np.mean(v > 0), v.mean(), c_r, r['eq'],
                 100 * abs(L.dd_of(r['curve'])), '  ЗАЩ' if r['paused_at'] else '',
                 ('  (не залилось %d)' % missed) if missed else ''), flush=True)
    print('', flush=True)
