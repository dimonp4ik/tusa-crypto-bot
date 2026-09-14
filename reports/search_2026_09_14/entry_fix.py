"""The model enters on a limit; the live bot sends a market order. How much has that hidden?

src/pullback_live.py line 279 calls place_market_entry, and okx_trader.py sets ordType "market".
The bot wakes at the hourly close and buys at whatever the book offers. The model instead places a
limit at that same hourly close, fills only if the next fifteen minutes trade back to it, and
discards the signal otherwise.

So the model has been simulating an entry the bot does not use. It errs conservatively - it throws
away the signals where price left immediately, which are the ones that were right - but it is still
wrong, and every number measured today rests on it.

This isolates the correction: the bank alone, the bank with the rule, both books, both entry models.
Nothing else changes.
"""
import collections, csv, datetime, sys
import numpy as np
sys.path.insert(0, 'C:/Users/Lenovo/Desktop/Торговля/crypto-bot')
import struct_params as SP
import gauntlet2 as G
import sim_w
import entry as E

ALL, NOBAD, YT = E.ALL, E.NOBAD, E.YT


def bank_only(coins, how):
    """Same machinery, but only the bank's own signals."""
    saved = {}
    for s in coins:
        saved[s] = E.SIG[s]
    for s in coins:
        E.SIG[s] = {t: v for t, v in saved[s].items()
                    if t in SP.bank_signals([dict(r, tp=1.0) for r in SP.BASE], s, 50)}
    rows, missed = E.book(coins, how)
    for s in coins:
        E.SIG[s] = saved[s]
    return rows, missed


def show(lbl, rows, missed):
    k, m = sim_w.money_at_dd(rows, 0.12)
    v = np.array([x[2] for x in rows])
    per = [sim_w.money_at_dd([x for x in rows if YT[y] <= x[0] < YT[y+1]], 0.12)[1]
           if len([x for x in rows if YT[y] <= x[0] < YT[y+1]]) >= 50 else float('nan')
           for y in range(2022, 2027)]
    print('  %-38s %4d сд. (потеряно %3d) ВР%5.1f%% ср%+.4f $%6.0f | %s'
          % (lbl, len(v), missed, 100*np.mean(v>0), v.mean(), m,
             ' '.join('%3.0f' % x if np.isfinite(x) else '  -' for x in per)), flush=True)
    return m


for cl, coins in (('без AAVE и XLM', NOBAD), ('все 15 монет', ALL)):
    print('', flush=True); print('  ===== %s =====' % cl, flush=True)
    a = show('БАНК ОДИН, лимит (модель)', *bank_only(coins, 'лимит'))
    b = show('БАНК ОДИН, по рынку (как живёт бот)', *bank_only(coins, 'рынок'))
    print('      разница: %+.0f%%' % (100*(b/a-1)), flush=True)
    c = show('банк+откат, лимит (модель)', *E.book(coins, 'лимит'))
    d = show('банк+откат, по рынку (как живёт бот)', *E.book(coins, 'рынок'))
    print('      разница: %+.0f%% | вклад отката при рыночном входе: x%.2f'
          % (100*(d/c-1), d/b), flush=True)
