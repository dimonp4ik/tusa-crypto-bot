"""The two live symbols the model has never measured.

The live universe is eighteen symbols. The model builds its context from sixteen, with SEIUSDT and
LABUSDT excluded by a line in the file that gives no reason, and BILLUSDT already on the way out.
That means the deployed bot would trade two coins under rules that were never tested on them - and
this project has already measured what that costs once: the unvalidated half of the live universe
returned 55.1% and -8.71R against 68.4% and +22.05R for the validated half.

So first, is there enough history to measure them at all, and second, what do they do.
"""
import datetime
import sys

import numpy as np

sys.path.insert(0, 'C:/Users/Lenovo/Desktop/Торговля/crypto-bot')
import tm_tag as T
import mrlib as M

print('  в M.PIN: %d -> %s' % (len(M.PIN), ', '.join(M.PIN)), flush=True)
print('', flush=True)
for s in ('SEIUSDT', 'LABUSDT', 'BILLUSDT', 'HYPEUSDT', 'ZECUSDT'):
    if s not in T.data:
        print('  %-10s данных в T.data НЕТ' % s, flush=True)
        continue
    t = T.data[s]['T']
    f = lambda x: datetime.datetime.fromtimestamp(int(x), datetime.UTC).strftime('%Y-%m-%d')
    print('  %-10s свечей 15м %6d  %s .. %s  (%.1f мес)'
          % (s, len(t), f(t[0]), f(t[-1]), (t[-1] - t[0]) / (30.44 * 86400)), flush=True)
