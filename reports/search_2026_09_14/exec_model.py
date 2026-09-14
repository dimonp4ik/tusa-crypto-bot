"""One explicit execution model, written down so the numbers stop moving.

Three conventions have been in play today and all three are wrong in different places.

  мой прежний    slippage charged on entry AND exit, 4 bps of fees on every fill, including
                 passive ones that pay neither
  модуль банка   one round-trip fee, no slippage on the exit at all - so a market stop is filled
                 at the bar's open for free, which it is not
  здесь          each leg charged what that leg actually is

The legs, as the live code places them. Entry is a market order (place_market_entry, ordType
"market"), so it is a taker: it pays the measured book cost and the taker fee. The protection is an
OCO: the take is a resting limit and pays the maker fee with no slippage, the stop triggers a market
order and pays both. The time exit is a market order.

The limit-entry variant is kept for comparison only, because the research module claims the edge
lives there - but the live bot does not place limit entries, so it is not what the account will do.
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

MEAS = collections.defaultdict(list)
for r in csv.DictReader(open('book_frozen.csv')):
    MEAS[r['coin'] + 'USDT'].append(float(r['cost109']))
SP.SLIP_REAL.update({k: float(np.median(v)) for k, v in MEAS.items()})
G.FC.clear()
ALL = list(G.ALL)
NOBAD = [s for s in ALL if s not in ('AAVEUSDT', 'XLMUSDT')]
YT = {y: int(datetime.datetime(y, 1, 1, tzinfo=datetime.UTC).timestamp()) for y in range(2022, 2028)}
RULE = [('rsi48', '>=', 56.3761), ('rsi6', '<=', 33.7947)]
BIG, HOLD = 10 ** 9, 192
FEE_TAKER, FEE_MAKER = 0.0002, 0.0001      # per leg
SLICE = 182 * 86400


def signals(coin, with_rule=True):
    out = {}
    for close, (ri, lim, atr) in SP.bank_signals(
            [dict(r, tp=1.0) for r in SP.BASE], coin, 50).items():
        out[close] = (lim, atr, SP.BASE[ri].get('side', 'LONG') == 'LONG')
    if with_rule:
        for close, (lim, atr) in NA.sigs(coin, 'без режима', RULE).items():
            out.setdefault(close, (lim, atr, True))
    return out


SIG_ALL = {s: signals(s, True) for s in ALL}
SIG_BANK = {s: signals(s, False) for s in ALL}


def book(coins, sig, how='рынок'):
    rows, missed = [], 0
    for s in coins:
        c = SP.CTX[s]
        slip = SP.SLIP_REAL.get(s, 0.0002)
        t15, A15, pos = c['t15'], c['a15'], c['pos']
        busy = 0
        for close in sorted(sig[s]):
            if close < busy:
                continue
            lim, atr, lg = sig[s][close]
            i = pos.get(close)
            if i is None or i + HOLD > len(t15):
                continue
            o0, h0, l0 = A15[i, 0], A15[i, 1], A15[i, 2]
            passive = False
            if how == 'рынок':
                e = o0 * (1 + slip) if lg else o0 * (1 - slip)
                fee_in = FEE_TAKER
            else:
                if (o0 <= lim) if lg else (o0 >= lim):
                    e, fee_in = (o0 * (1 + slip) if lg else o0 * (1 - slip)), FEE_TAKER
                elif (l0 <= lim) if lg else (h0 >= lim):
                    e, fee_in, passive = lim, FEE_MAKER, True
                else:
                    missed += 1
                    continue
            o, h, l, cc = (A15[i:i + HOLD, x] for x in range(4))
            if lg:
                TP, SL = e + 1.0 * atr, e - 3.0 * atr
                hs, ht = l <= SL, h >= TP
            else:
                TP, SL = e - 1.0 * atr, e + 3.0 * atr
                hs, ht = h >= SL, l <= TP
            if passive:
                ht = ht.copy()
                ht[0] = False
            js = int(np.argmax(hs)) if hs.any() else BIG
            jt = int(np.argmax(ht)) if ht.any() else BIG
            if js <= jt and js < BIG:                      # stop: market, slips and pays taker
                jj = js
                base = min(SL, o[js]) if lg else max(SL, o[js])
                f_ = base * (1 - slip) if lg else base * (1 + slip)
                fee_out = FEE_TAKER
            elif jt < BIG:                                  # take: resting limit, maker, no slip
                jj, f_, fee_out = jt, TP, FEE_MAKER
            else:                                           # time exit: market
                jj = HOLD - 1
                f_ = cc[-1] * (1 - slip) if lg else cc[-1] * (1 + slip)
                fee_out = FEE_TAKER
            ret = ((f_ / e - 1) if lg else (1 - f_ / e)) - fee_in - fee_out
            end = int(t15[i + jj]) + 900
            rows.append((close, end, ret / (3.0 * atr / e), 3.0 * atr / e, s, 1.0))
            busy = end
    rows.sort()
    return rows, missed


def show(lbl, rows, missed, ref=None):
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
    print('  %-34s %4d сд. ВР%5.1f%% ср%+.4f риск %.3f%% $%6.0f | %s%s'
          % (lbl, len(v), 100 * np.mean(v > 0), v.mean(), 100 * k, m,
             ' '.join('%3.0f' % x if np.isfinite(x) else '  -' for x in per), wf), flush=True)
    return m


print('  комиссия: тейкер %.2f б.п. за ногу, мейкер %.2f б.п.; проскальзывание измеренное'
      % (10000 * FEE_TAKER, 10000 * FEE_MAKER), flush=True)
for cl, coins in (('без AAVE и XLM', NOBAD), ('все 15 монет', ALL)):
    print('', flush=True)
    print('  ===== %s =====' % cl, flush=True)
    b = show('БАНК ОДИН, вход по рынку (как живёт)', *book(coins, SIG_BANK, 'рынок'))
    show('БАНК ОДИН, вход лимитом (не живёт)', *book(coins, SIG_BANK, 'лимит'))
    bank_rows = book(coins, SIG_BANK, 'рынок')[0]
    d = show('банк+откат, по рынку (как живёт)', *book(coins, SIG_ALL, 'рынок'), bank_rows)
    show('банк+откат, лимитом (не живёт)', *book(coins, SIG_ALL, 'лимит'), bank_rows)
    print('        вклад отката при живом входе: x%.2f' % (d / b), flush=True)
