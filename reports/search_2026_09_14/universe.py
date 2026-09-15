"""Forty coins of prepared data, sixteen in use. What does the rest of the universe do?

The account holds two coins of thirteen at a typical moment and the per-coin lock is what discards
signals, so the obvious lever is more coins. Bars are already prepared for forty symbols, and the
live venue lists 179 instruments, so the constraint was never data - the model's coin list simply
never grew.

Each extra coin is built exactly like the existing ones, then measured three ways: what its own
trades look like, how much history it actually has, and what the account does when it is added.

Two things have to be watched. A coin with a year of history contributes to recent years only, so
the comparison has to be made over a window all of them cover. And cost: these are thinner books,
and the book sampler never measured them, so their fill cost is assumed at the worst measured value
rather than the best.
"""
import bisect
import collections
import csv
import datetime
import json
import sys
import urllib.request

import numpy as np

sys.path.insert(0, 'C:/Users/Lenovo/Desktop/Торговля/crypto-bot')
import mrlib as M
import struct_params as SP
import gauntlet2 as G
import sim_w
import nogate_attack as NA
from src import pullback_bank as PB

UA = {'User-Agent': 'Mozilla/5.0'}
req = urllib.request.Request(
    'https://www.okx.com/api/v5/public/instruments?instType=FUTURES', headers=UA)
body = json.loads(urllib.request.urlopen(req, timeout=25).read().decode())
XPERP = {x['instId'].split('-')[0] for x in body.get('data', [])
         if '_UM_XPERP-' in x.get('instId', '') and x.get('state') == 'live'}
print('  живых инструментов X-Perp: %d' % len(XPERP), flush=True)

cost = collections.defaultdict(list)
for r in csv.DictReader(open('book_frozen.csv')):
    cost[r['coin'] + 'USDT'].append(float(r['cost109']))
MEASURED = {k: float(np.median(v)) for k, v in cost.items()}
WORST = max(v for k, v in MEASURED.items() if k != 'BILLUSDT')
SP.SLIP_REAL.update(MEASURED)
G.FC.clear()
print('  худшая измеренная стоимость входа (без BILL): %.6f — её и назначаю неизмеренным'
      % WORST, flush=True)

IN_MODEL = set(SP.COINS)
EXTRA = [s for s in M.U if s not in IN_MODEL and s.replace('USDT', '') in XPERP]
print('  монет в модели %d, кандидатов с данными И инструментом: %d'
      % (len(IN_MODEL), len(EXTRA)), flush=True)

bdt, bdc = PB.btc_daily(SP.CTX['BTCUSDT']['bdt'], SP.CTX['BTCUSDT']['bdc']) \
    if False else (SP.CTX['BTCUSDT']['bdt'], SP.CTX['BTCUSDT']['bdc'])
built, skipped = [], []
for s in EXTRA:
    try:
        d = np.load(M.D + s + '.npz')
        t15, a15 = d['T'], d['A']
        if len(t15) < 8000:
            skipped.append((s, 'мало баров %d' % len(t15)))
            continue
        T1, B1 = PB.build_bars(t15, a15, SP.HOUR)
        F = PB.features(T1, B1, SP.CTX['BTCUSDT']['t15'], SP.CTX['BTCUSDT']['a15'])
        iv = PB.trend_regime(t15, a15, bdt, bdc)
        SP.CTX[s] = dict(t15=t15, a15=a15, T1=T1, B1=B1, F=F, iv=iv,
                         starts=[x[0] for x in iv], bdt=bdt, bdc=np.asarray(bdc, dtype=float),
                         pos=dict(zip(t15.tolist(), range(len(t15)))),
                         sma={n: PB._sma(np.asarray(bdc, dtype=float), n)
                              for n in (20, 30, 50, 80, 120, 200)})
        SP.SLIP_REAL.setdefault(s, WORST)
        built.append(s)
    except Exception as e:
        skipped.append((s, str(e)[:50]))
print('  построено: %d — %s' % (len(built), ', '.join(x.replace('USDT', '') for x in built)),
      flush=True)
for s, why in skipped:
    print('    пропущено %-8s %s' % (s.replace('USDT', ''), why), flush=True)

f = lambda x: datetime.datetime.fromtimestamp(int(x), datetime.UTC).strftime('%Y-%m')
print('', flush=True)
print('  монета   баров   история        сделок банка  сделок отката', flush=True)
RULE = [('rsi48', '>=', 56.3761), ('rsi6', '<=', 33.7947)]
stats = {}
for s in built:
    t = SP.CTX[s]['t15']
    try:
        nb = len(SP.bank_signals([dict(r, tp=1.0) for r in SP.BASE], s, 50))
        nr = len(NA.sigs(s, 'без режима', RULE))
    except Exception as e:
        print('    %-8s ошибка сигналов: %s' % (s.replace('USDT', ''), str(e)[:40]), flush=True)
        continue
    stats[s] = (len(t), f(t[0]), f(t[-1]), nb, nr)
    print('  %-8s %6d  %s..%s %8d %12d'
          % (s.replace('USDT', ''), len(t), f(t[0]), f(t[-1]), nb, nr), flush=True)

import pickle
pickle.dump(built, open('extra_coins.pkl', 'wb'))
print('', flush=True)
print('  сохранено в extra_coins.pkl', flush=True)
