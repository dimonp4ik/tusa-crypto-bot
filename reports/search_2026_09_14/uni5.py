"""Only five of the twenty pass the venue's spread gate. Measure those, honestly costed.

The first pass assigned every new coin the worst fill cost ever measured (0.001540) and they returned
-0.0280R. A snapshot of the books says that assumption was not harsh but accurate for fifteen of
them - their spreads run from 0.06% to 0.44%, and the live gate refuses anything above 0.05%. So the
universe does not expand because of liquidity, not because the rules fail.

Five do pass: DOGE, WLD, BNB, PUMP, LTC. They get their own snapshot spread as the fill cost, which
is still one reading rather than a day of sampling, and the account is measured with them.
"""
import collections, datetime, pickle, sys
import numpy as np
sys.path.insert(0, 'C:/Users/Lenovo/Desktop/Торговля/crypto-bot')
import struct_params as SP
import gauntlet2 as G
import sim_w
import live_rules_sim as L
import universe as UNI
import exec_model as EM

# snapshot spreads; the walk cost for a $109 order is close to half the spread on a healthy book
SNAP = {'DOGEUSDT': 0.000120, 'WLDUSDT': 0.000267, 'BNBUSDT': 0.000139,
        'PUMPUSDT': 0.000276, 'LTCUSDT': 0.000189}
for s, sp in SNAP.items():
    SP.SLIP_REAL[s] = sp / 2
G.FC.clear()
PASS5 = list(SNAP)
BASE13 = [s for s in SP.COINS if s not in ('BILLUSDT', 'AAVEUSDT', 'XLMUSDT')]
LO = int(datetime.datetime(2024, 1, 15, tzinfo=datetime.UTC).timestamp())
print('  опора %d монет, добавляем %d прошедших отсечку' % (len(BASE13), len(PASS5)), flush=True)

def book(coins):
    sig = {s: EM.signals(s, True) for s in coins}
    saved = EM.SIG_ALL
    EM.SIG_ALL = sig
    rows = EM.book(coins, sig, 'рынок')[0]
    EM.SIG_ALL = saved
    return [r for r in rows if r[0] >= LO]

A = book(BASE13)
B = book(BASE13 + PASS5)
print('  сделок: опора %d, с пятью %d (+%.0f%%)'
      % (len(A), len(B), 100 * (len(B) / len(A) - 1)), flush=True)
print('', flush=True)
print('  === при фиксированном риске ===', flush=True)
for risk in (0.010, 0.0125, 0.014, 0.0155, 0.017):
    ra, rb = sim_w.simulate(A, risk), sim_w.simulate(B, risk)
    print('    %.2f%%  опора $%6.0f / %4.1f%%   +5 монет $%6.0f / %4.1f%%   %+5.0f%%'
          % (100 * risk, ra['eq'], 100 * abs(L.dd_of(ra['curve'])),
             rb['eq'], 100 * abs(L.dd_of(rb['curve'])), 100 * (rb['eq'] / ra['eq'] - 1)), flush=True)
print('', flush=True)
for lbl, rows in (('опора 13', A), ('13 + 5 монет', B)):
    k, m = sim_w.money_at_dd(rows, 0.12)
    v = np.array([x[2] for x in rows])
    print('  при равной просадке: %-14s %4d сд. ВР%5.1f%% ср%+.4f риск %.3f%% $%6.0f'
          % (lbl, len(v), 100 * np.mean(v > 0), v.mean(), 100 * k, m), flush=True)

print('', flush=True)
print('  === каждая из пяти по отдельности, поверх опоры ===', flush=True)
base_fix = sim_w.simulate(A, 0.014)['eq']
for s in PASS5:
    rows = book(BASE13 + [s])
    own = [x for x in rows if x[4] == s]
    a = np.array([x[2] for x in own]) if own else np.zeros(0)
    r = sim_w.simulate(rows, 0.014)
    print('    +%-6s своих n%3d ВР%5.1f%% ср%+.4f | счёт $%6.0f (%+4.0f%%) просадка %4.1f%%'
          % (s.replace('USDT', ''), len(a), 100 * np.mean(a > 0) if len(a) else 0,
             a.mean() if len(a) else 0, r['eq'], 100 * (r['eq'] / base_fix - 1),
             100 * abs(L.dd_of(r['curve']))), flush=True)
