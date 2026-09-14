"""The pullback rule's stop, measured in money, which is the only way it can be measured.

The rule inherited 3.0 ATR from the bank and nobody checked whether that is its own optimum. It
cannot be checked in R: R is the return divided by the trade's own stop, so a tighter stop is
mechanically a bigger position at the same risk, and the whole column reads as an improvement that
is really leverage. The account at equal drawdown has no such problem - it sizes each variant to the
same worst drawdown and reports what is left.

The bank keeps 3.0/1.0 throughout. Only the rule's geometry moves.
"""
import collections
import csv
import datetime
import sys

import numpy as np

sys.path.insert(0, 'C:/Users/Lenovo/Desktop/Торговля/crypto-bot')
import struct_params as SP
import gauntlet2 as G
import stop_width as SW
import sim_w
import nogate_attack as NA
from src import pullback_bank as PB

meas = collections.defaultdict(list)
spread = collections.defaultdict(list)
for r in csv.DictReader(open('book_samples.csv')):
    meas[r['coin'] + 'USDT'].append(float(r['cost109']))
    spread[r['coin'] + 'USDT'].append(float(r['spread']))
SP.SLIP_REAL.update({k: float(np.median(v)) for k, v in meas.items()})
G.FC.clear()
MED = {k: float(np.median(v)) for k, v in spread.items()}
COINS = [s for s in G.ALL if MED.get(s, 0) <= 0.0005]

C = [('rsi48', '>=', 56.3761), ('rsi6', '<=', 33.7947)]
YT = {y: int(datetime.datetime(y, 1, 1, tzinfo=datetime.UTC).timestamp()) for y in range(2022, 2028)}
SLICE = 182 * 86400
BIG = 10 ** 9
bank = SW.run(3.0, 1.0, coins=COINS, with_meta=True)


def rule_trades(sl, tp):
    """Gateless pullback with its own geometry; the bank's is untouched."""
    rows = []
    for s in COINS:
        sig = NA.sigs(s, 'без режима', C)
        c = SP.CTX[s]
        slip = SP.SLIP_REAL.get(s, 0.0002)
        t15, A15, pos = c['t15'], c['a15'], c['pos']
        busy = 0
        for close in sorted(sig):
            if close < busy:
                continue
            lim, atr = sig[close]
            i = pos.get(close)
            if i is None or i + PB.HOLD_BARS > len(t15):
                continue
            o, h, l, cc = (A15[i:i + PB.HOLD_BARS, x] for x in range(4))
            if o[0] <= lim:
                e, touch = o[0] * (1 + slip), False
            elif l[0] <= lim:
                e, touch = lim * (1 + slip), True
            else:
                continue
            TP, SL = e + tp * atr, e - sl * atr
            hs, ht = l <= SL, h >= TP
            if touch:
                ht = ht.copy()
                ht[0] = False
            js = int(np.argmax(hs)) if hs.any() else BIG
            jt = int(np.argmax(ht)) if ht.any() else BIG
            if js <= jt and js < BIG:
                jj, f_ = js, min(SL, o[js]) * (1 - slip)
            elif jt < BIG:
                jj, f_ = jt, TP * (1 - slip)
            else:
                jj, f_ = PB.HOLD_BARS - 1, cc[-1] * (1 - slip)
            ret = (f_ / e - 1) - 0.0004
            rows.append((close, int(t15[i + jj]) + 900, ret / (sl * atr / e), sl * atr / e, s))
            busy = int(t15[i + jj]) + 900
    rows.sort()
    return rows


def account(cand):
    ev = [(a, b, R, sf, s, 0) for a, b, R, sf, s in bank]
    ev += [(a, b, R, sf, s, 1) for a, b, R, sf, s in cand]
    ev.sort(key=lambda x: (x[0], x[5]))
    busy, out = {}, []
    for a, b, R, sf, s, k in ev:
        if busy.get(s, 0) > a:
            continue
        busy[s] = b
        out.append((a, b, R, sf, s, 1.0))
    out.sort()
    return out


SLS = [2.0, 2.5, 3.0, 3.5, 4.0, 5.0]
TPS = [0.8, 1.0, 1.3]
ref = account([])
m0 = sim_w.money_at_dd(ref, 0.12)[1]
print('  опора: банк один (после отсечки спреда) $%.0f' % m0, flush=True)
print('', flush=True)
print('  === деньги счёта при равной просадке 12%%; винрейт правила в скобках ===', flush=True)
print('  стоп\\тейк %s' % '  '.join('%18.1f' % t for t in TPS), flush=True)
CACHE = {}
for sl in SLS:
    cells = []
    for tp in TPS:
        cand = rule_trades(sl, tp)
        CACHE[(sl, tp)] = cand
        rows = account(cand)
        m = sim_w.money_at_dd(rows, 0.12)[1]
        v = np.array([r[2] for r in cand])
        cells.append('$%6.0f n%3d ВР%3.0f%%' % (m, len(v), 100 * np.mean(v > 0)))
    print('  %8.1f  %s' % (sl, '  '.join(cells)), flush=True)

print('', flush=True)
print('  === выбор геометрии по четырём годам, замер на пятом ===', flush=True)
wins = tot = 0
for hold in range(2022, 2027):
    lo, hi = YT[hold], YT[hold + 1]
    best, bk = None, None
    for k, cand in CACHE.items():
        tr = [r[2] for r in cand if not (lo <= r[0] < hi)]
        if len(tr) < 150:
            continue
        v = float(np.mean(tr))
        if best is None or v > best:
            best, bk = v, k
    ho = [r[2] for r in CACHE[bk] if lo <= r[0] < hi]
    cur = [r[2] for r in CACHE[(3.0, 1.0)] if lo <= r[0] < hi]
    if len(ho) < 12 or len(cur) < 12:
        print('    %d  выбрано стоп%.1f тейк%.1f — мало сделок' % (hold, bk[0], bk[1]), flush=True)
        continue
    tot += 1
    wins += int(np.mean(ho) > np.mean(cur))
    print('    %d  выбрано стоп%.1f тейк%.1f | невиданный год: выбор%+.4f (n%d), рабочее 3.0/1.0%+.4f (n%d)'
          % (hold, bk[0], bk[1], np.mean(ho), len(ho), np.mean(cur), len(cur)), flush=True)
print('    выбранная геометрия лучше рабочей в %d годах из %d (в R, не в деньгах)' % (wins, tot),
      flush=True)
