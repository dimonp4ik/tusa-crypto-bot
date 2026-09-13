"""The other two families from the pair search, through the same gauntlet.

The pullback family came first because it had the best spread and the cleanest mechanism. Two others
appeared repeatedly at the top and have not been tested:

  * the dip in a volatile market - a coin that has just fallen while volatility is elevated
  * market breadth, a signal the bank has never had: capitulation (almost nothing is rising) taken
    long, euphoria (everything is rising) taken short

Breadth is the only cross-sectional feature available and it is the one most likely to be genuinely
new information rather than a rearrangement of the same price history.
"""
import sys

import numpy as np

sys.path.insert(0, 'C:/Users/Lenovo/Desktop/Торговля/crypto-bot')
import gauntlet2 as G

CASES = [
    ('ОТКАТ rsi48>=56.38 + rsi6<=33.79', [('rsi48', '>=', 56.3761), ('rsi6', '<=', 33.7947)], True),
    ('ПАДЕНИЕ ret24<=-1.54 + vol720>=1.45', [('ret24', '<=', -1.5434), ('vol720', '>=', 1.4462)], True),
    ('ПАДЕНИЕ ret12<=-1.82 + vol720>=1.80', [('ret12', '<=', -1.8214), ('vol720', '>=', 1.8040)], True),
    ('ПАДЕНИЕ pos48<=0.20 + vol720>=1.45', [('pos48', '<=', 0.1983), ('vol720', '>=', 1.4462)], True),
    ('КАПИТУЛЯЦИЯ breadth<=0.06 + vol720>=1.45',
     [('breadth', '<=', 0.0625), ('vol720', '>=', 1.4462)], True),
    ('ЭЙФОРИЯ breadth>=1.0 + vol168>=1.39 SHORT',
     [('breadth', '>=', 1.0), ('vol168', '>=', 1.3947)], False),
    ('ЭЙФОРИЯ breadth>=0.71 + vol168>=1.69 SHORT',
     [('breadth', '>=', 0.7143), ('vol168', '>=', 1.6890)], False),
    ('РАСХОЖДЕНИЕ btc72<=-0.04 + pos48>=0.78 SHORT',
     [('btc72', '<=', -0.0395), ('pos48', '>=', 0.7764)], False),
    ('РАСХОЖДЕНИЕ btc168<=-0.06 + pos48>=0.88 SHORT',
     [('btc168', '<=', -0.0558), ('pos48', '>=', 0.8794)], False),
]
for name, conds, lg in CASES:
    G.describe(name, conds, lg)
