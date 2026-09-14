"""Market entry against limit entry, with the two fills charged what they actually cost.

The first attempt charged both the same, which is wrong in both directions at once. The bank's own
module separates them:

    TAKER_SLIP   = 0.0004   filled through the limit or at market: 4 bps of slippage
    TAKER_COST   = 0.0004   and 4 bps of fees
    PASSIVE_COST = 0.0002   filled AT the limit as the maker: 2 bps, and no slippage

So a passive fill was being charged twice what it costs and given slippage a maker does not pay,
while a market fill was charged measured slippage that is smaller than the module assumes for most
coins. Both errors favour the market entry, which is exactly what the first test concluded.

That matters because the module's own docstring makes the opposite claim outright: "the edge lives
here - the same filters with a market entry are ~0". One of the two is wrong, and the way to find
out is to charge each fill honestly.

Two cost models, because the right one is not obvious: the module's flat assumption, and the
measured order book with the maker/taker split kept.
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

MEAS = collections.defaultdict(list)
for r in csv.DictReader(open('book_frozen.csv')):
    MEAS[r['coin'] + 'USDT'].append(float(r['cost109']))
MEASURED = {k: float(np.median(v)) for k, v in MEAS.items()}
SP.SLIP_REAL.update(MEASURED)
G.FC.clear()
ALL = list(G.ALL)
NOBAD = [s for s in ALL if s not in ('AAVEUSDT', 'XLMUSDT')]
YT = {y: int(datetime.datetime(y, 1, 1, tzinfo=datetime.UTC).timestamp()) for y in range(2022, 2028)}
RULE = [('rsi48', '>=', 56.3761), ('rsi6', '<=', 33.7947)]
BIG = 10 ** 9
HOLD = 192
TAKER_SLIP, TAKER_COST, PASSIVE_COST = 0.0004, 0.0004, 0.0002


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
print('  сигналов: с откатом %d, банк один %d'
      % (sum(len(v) for v in SIG_ALL.values()), sum(len(v) for v in SIG_BANK.values())), flush=True)


def book(coins, sig, how='лимит', costs='модуль'):
    rows, missed = [], 0
    for s in coins:
        c = SP.CTX[s]
        meas = SP.SLIP_REAL.get(s, 0.0002)
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
                slip = TAKER_SLIP if costs == 'модуль' else meas
                e = o0 * (1 + slip) if lg else o0 * (1 - slip)
                cost = TAKER_COST
            else:
                if (o0 <= lim) if lg else (o0 >= lim):
                    slip = TAKER_SLIP if costs == 'модуль' else meas
                    e = o0 * (1 + slip) if lg else o0 * (1 - slip)
                    cost = TAKER_COST
                elif (l0 <= lim) if lg else (h0 >= lim):
                    e, cost, passive = lim, PASSIVE_COST, True
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
                ht[0] = False          # the bar's extreme may precede the fill
            js = int(np.argmax(hs)) if hs.any() else BIG
            jt = int(np.argmax(ht)) if ht.any() else BIG
            if js <= jt and js < BIG:
                jj = js
                f_ = min(SL, o[js]) if lg else max(SL, o[js])
            elif jt < BIG:
                jj, f_ = jt, TP
            else:
                jj, f_ = HOLD - 1, cc[-1]
            ret = ((f_ / e - 1) if lg else (1 - f_ / e)) - cost
            end = int(t15[i + jj]) + 900
            rows.append((close, end, ret / (3.0 * atr / e), 3.0 * atr / e, s, 1.0))
            busy = end
    rows.sort()
    return rows, missed


def show(lbl, rows, missed):
    k, m = sim_w.money_at_dd(rows, 0.12)
    v = np.array([x[2] for x in rows])
    per = []
    for y in range(2022, 2027):
        sub = [x for x in rows if YT[y] <= x[0] < YT[y + 1]]
        per.append(sim_w.money_at_dd(sub, 0.12)[1] if len(sub) >= 50 else float('nan'))
    print('  %-36s %4d сд. (потеряно %3d) ВР%5.1f%% ср%+.4f $%6.0f | %s'
          % (lbl, len(v), missed, 100 * np.mean(v > 0), v.mean(), m,
             ' '.join('%3.0f' % x if np.isfinite(x) else '  -' for x in per)), flush=True)
    return m


for costs in ('модуль', 'измеренные'):
    print('', flush=True)
    print('  ################ издержки: %s ################' % costs, flush=True)
    for cl, coins in (('без AAVE и XLM', NOBAD), ('все 15 монет', ALL)):
        print('  ===== %s =====' % cl, flush=True)
        a = show('БАНК ОДИН, лимит', *book(coins, SIG_BANK, 'лимит', costs))
        b = show('БАНК ОДИН, по рынку', *book(coins, SIG_BANK, 'рынок', costs))
        print('        рынок против лимита: %+.0f%%' % (100 * (b / a - 1)), flush=True)
        c = show('банк+откат, лимит', *book(coins, SIG_ALL, 'лимит', costs))
        d = show('банк+откат, по рынку', *book(coins, SIG_ALL, 'рынок', costs))
        print('        рынок против лимита: %+.0f%% | вклад отката: x%.2f (лимит) x%.2f (рынок)'
              % (100 * (d / c - 1), c / a, d / b), flush=True)
