"""Two runs of the same measurement gave different answers. Find out why before trusting anything.

slow_horizon printed a 96-hour threshold of 54.61 and 964 trades in one run and 54.64 and 890 in the
next; the bank alone was $1560 and then $1623. Same code, same data files. Either some state leaks
between imports or something is not deterministic, and every cross-script comparison made today is
suspect until it is known which.
"""
import sys

import numpy as np

sys.path.insert(0, 'C:/Users/Lenovo/Desktop/Торговля/crypto-bot')
import struct_params as SP
import gauntlet2 as G
import stop_width as SW

print('  что лежит в SP.SLIP_REAL СРАЗУ после импорта struct_params:', flush=True)
for k in ('BTCUSDT', 'SOLUSDT', 'ZECUSDT', 'AAVEUSDT'):
    print('    %-10s %.6f' % (k, SP.SLIP_REAL.get(k, float('nan'))), flush=True)
print('  монет: G.ALL=%d, SP.COINS=%d' % (len(G.ALL), len(SP.COINS)), flush=True)

bank1 = SW.run(3.0, 1.0, coins=G.ALL, with_meta=True)
bank2 = SW.run(3.0, 1.0, coins=G.ALL, with_meta=True)
print('  SW.run дважды подряд: %d и %d сделок, сумма R %.6f и %.6f'
      % (len(bank1), len(bank2), sum(x[2] for x in bank1), sum(x[2] for x in bank2)), flush=True)

import full_money as FM
print('', flush=True)
print('  ПОСЛЕ импорта full_money:', flush=True)
for k in ('BTCUSDT', 'SOLUSDT', 'ZECUSDT', 'AAVEUSDT'):
    print('    %-10s %.6f' % (k, SP.SLIP_REAL.get(k, float('nan'))), flush=True)
bank3 = SW.run(3.0, 1.0, coins=G.ALL, with_meta=True)
print('  SW.run снова: %d сделок, сумма R %.6f  %s'
      % (len(bank3), sum(x[2] for x in bank3),
         'ТА ЖЕ (кэш! издержки не применились)' if abs(sum(x[2] for x in bank3) - sum(x[2] for x in bank1)) < 1e-9
         else 'другая (издержки применились)'), flush=True)

print('', flush=True)
print('  есть ли кэш в stop_width:', flush=True)
import inspect
src = inspect.getsource(SW)
for line in src.split('\n'):
    if 'cache' in line.lower() or 'CACHE' in line:
        print('    %s' % line.strip()[:100], flush=True)

print('', flush=True)
print('  кэш признаков gauntlet2: %d монет загружено' % len(G.FC), flush=True)
_, f = G.feats('SOLUSDT')
print('  признаков у SOL: %d, из них rsiX: %d'
      % (len(f), sum(1 for k in f if k.startswith('rsiX'))), flush=True)
