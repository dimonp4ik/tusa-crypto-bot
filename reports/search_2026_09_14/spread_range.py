"""Two days of book data disagree about which coins the spread gate refuses. Bound the answer.

Day one put BILL, XLM and AAVE above the 0.05% line. Day two put DOT and AVAX there as well - AVAX
crossed outright (+189%) and DOT nearly doubled. The live gate does not use a median: it reads the
spread at each signal and refuses that entry, so coins near the line are refused part of the time
and the coin-removal model is only an approximation.

The costs themselves are stable (correlation 0.997 between the days), so only the membership is in
question. Bounding it: the account with the three coins that are refused on both days, and with all
five that are refused on either.
"""
import collections, csv, datetime, sys
import numpy as np
sys.path.insert(0, 'C:/Users/Lenovo/Desktop/Торговля/crypto-bot')
import struct_params as SP
import gauntlet2 as G
import sim_w
import exec_model as EM

YT = EM.YT
ALL = EM.ALL
SETS = [
    ('все 15 монет',                    ALL),
    ('без BILL/XLM/AAVE (обе сутки)',    [s for s in ALL if s not in ('XLMUSDT', 'AAVEUSDT')]),
    ('и без DOT (вторые сутки)',         [s for s in ALL if s not in ('XLMUSDT', 'AAVEUSDT', 'DOTUSDT')]),
    ('и без AVAX тоже (вторые сутки)',   [s for s in ALL if s not in ('XLMUSDT', 'AAVEUSDT', 'DOTUSDT', 'AVAXUSDT')]),
]
print('  вариант                            банк один   банк+откат   вклад   по годам (банк+откат)',
      flush=True)
for lbl, coins in SETS:
    b = EM.book(coins, EM.SIG_BANK, 'рынок')[0]
    d = EM.book(coins, EM.SIG_ALL, 'рынок')[0]
    mb = sim_w.money_at_dd(b, 0.12)[1]
    md = sim_w.money_at_dd(d, 0.12)[1]
    per = []
    for y in range(2022, 2027):
        sub = [x for x in d if YT[y] <= x[0] < YT[y+1]]
        per.append(sim_w.money_at_dd(sub, 0.12)[1] if len(sub) >= 50 else float('nan'))
    print('  %-34s $%6.0f     $%6.0f    x%.2f   %s'
          % (lbl, mb, md, md / mb,
             ' '.join('%3.0f' % x if np.isfinite(x) else '  -' for x in per)), flush=True)

print('', flush=True)
print('  === и то же с отсечкой стопа 0.10 (защита от ликвидации) ===', flush=True)
for lbl, coins in SETS:
    b = [x for x in EM.book(coins, EM.SIG_BANK, 'рынок')[0] if x[3] <= 0.10]
    d = [x for x in EM.book(coins, EM.SIG_ALL, 'рынок')[0] if x[3] <= 0.10]
    mb = sim_w.money_at_dd(b, 0.12)[1]
    md = sim_w.money_at_dd(d, 0.12)[1]
    print('  %-34s $%6.0f     $%6.0f    x%.2f' % (lbl, mb, md, md / mb), flush=True)
