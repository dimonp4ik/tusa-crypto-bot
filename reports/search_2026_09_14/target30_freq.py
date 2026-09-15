"""The frequency lever: the same pullback rule on faster bars.

The rule is defined in bar counts - RSI(48) at least 56.38, RSI(6) at most 33.79, stop 3 ATR(14),
take 1 ATR, hold 48 bars. On 1h bars that is a 48-hour context and a 6-hour dip. On 30m bars the
same counts mean a 24-hour context and a 3-hour dip, on 15m bars 12 hours and 90 minutes. If the
pattern is a property of how markets move rather than of the hour, the faster versions should keep
their edge and trade two to four times as often.

The trap for faster trading is cost: fill cost is a fixed share of price, while the stop shrinks
with the bar, so every extra trade pays a larger fraction of its risk in cost. Costs here are the
measured ones on both legs, entry at the next 15m open after the bar closes (market, as live).
"""
import collections, csv, datetime, sys
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
FEE = 0.0004
COINS = [s for s in SP.COINS if s not in ('BILLUSDT', 'XLMUSDT', 'AAVEUSDT')]
BIG = 10 ** 9


def run(sec, rsi_hi=56.3761, rsi_lo=33.7947):
    rows = []
    for s in COINS:
        c = SP.CTX[s]
        t15, a15, pos = c['t15'], c['a15'], c['pos']
        T, B = PB.build_bars(t15, a15, sec)
        if len(T) < 400:
            continue
        cl = B[:, 3]
        r48, r6 = PB._rsi(cl, 48), PB._rsi(cl, 6)
        atr = PB._atr(B)
        bdt, bdc = c['bdt'], np.asarray(c['bdc'], dtype=float)
        sma = PB._sma(bdc, 50)
        slip = SLIP.get(s, 0.0003)
        hold = 48 * sec // 900
        busy = 0
        for i in range(200, len(T)):
            if not (r48[i] >= rsi_hi and r6[i] <= rsi_lo) or not np.isfinite(atr[i]):
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
            e = o[0] * (1 + slip)
            TP, SL = e + atr[i], e - 3 * atr[i]
            hs, ht = l <= SL, h >= TP
            js = int(np.argmax(hs)) if hs.any() else BIG
            jt = int(np.argmax(ht)) if ht.any() else BIG
            if js <= jt and js < BIG:
                jj, f = js, min(SL, o[js]) * (1 - slip)
            elif jt < BIG:
                jj, f = jt, TP * (1 - slip)
            else:
                jj, f = hold - 1, cc[-1] * (1 - slip)
            sf = 3 * atr[i] / e
            ret = f / e - 1 - FEE
            end = int(t15[j + jj]) + 900
            rows.append((close, end, ret / sf, sf, s, 1.0))
            busy = end
    rows.sort()
    return rows


print('  свеча    контекст/откат/удержание   сделок в год   ВР     ср R    стоп медиана   $ при 1.4%%   просадка', flush=True)
for sec, lbl in ((3600, '48ч / 6ч / 48ч'), (1800, '24ч / 3ч / 24ч'), (900, '12ч / 1.5ч / 12ч')):
    rows = run(sec)
    if not rows:
        continue
    v = np.array([x[2] for x in rows])
    years = (rows[-1][0] - rows[0][0]) / (365.25 * 86400)
    r = sim_w.simulate(rows, 0.014)
    sf = np.median([x[3] for x in rows])
    print('  %4dм    %-18s        %6.0f       %4.1f%%  %+.4f    %.4f      $%8.0f   %5.1f%%%s'
          % (sec // 60, lbl, len(v) / years, 100 * np.mean(v > 0), v.mean(), sf, r['eq'],
             100 * abs(L.dd_of(r['curve'])), '  ЗАЩЁЛКА' if r['paused_at'] else ''), flush=True)
