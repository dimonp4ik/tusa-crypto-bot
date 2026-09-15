"""Slower bars, and the same rule stacked across timeframes as separate trade sources.

The fast-bar test showed the pullback pattern weakens below 1h even with zero costs. The opposite
direction was never measured: 2h and 4h bars, where RSI(48)/RSI(6) mean 4 and 8 days of context and
a 12h or 24h dip. Hold is capped at 72 hours (the owner's maximum), stop 3 / take 1 ATR of the bar's
own ATR, stops beyond 10% dropped (liquidation at 10x).

Different bar sizes fire at different moments, so a 1h, 2h and 4h version of one rule are partly
different trades. Stacked in one book with the bank - bank first, then 1h, 2h, 4h - they are the
cheapest possible combination: nothing new is fitted, the thresholds are the ones already found.
Money under the LIVE sizing (stop_ref), which beat equal-risk sizing by 57%.
"""
import collections, csv, datetime, sys
import numpy as np
sys.path.insert(0, 'C:/Users/Lenovo/Desktop/Торговля/crypto-bot')
import struct_params as SP
import sim_w
import live_rules_sim as L
from src import pullback_bank as PB

cost = collections.defaultdict(list)
for r in csv.DictReader(open('book_frozen.csv')):
    cost[r['coin'] + 'USDT'].append(float(r['cost109']))
SLIP = {k: float(np.median(v)) for k, v in cost.items()}
FEE = 0.0004
COINS = [s for s in SP.COINS if s != 'BILLUSDT']   # SECOND BOOK: all fifteen coins
BIG = 10 ** 9
YT = {y: int(datetime.datetime(y, 1, 1, tzinfo=datetime.UTC).timestamp()) for y in range(2022, 2028)}


def sim_exit(c, j, lg, atr, slip, sl, tp, hold):
    t15, a15 = c['t15'], c['a15']
    if j + hold > len(t15):
        return None
    o, h, l, cc = (a15[j:j + hold, x] for x in range(4))
    e = o[0] * (1 + slip) if lg else o[0] * (1 - slip)
    sf = sl * atr / e
    if sf > 0.10:
        return None
    TP_, SL_ = (e + tp * atr, e - sl * atr) if lg else (e - tp * atr, e + sl * atr)
    hs, ht = (l <= SL_, h >= TP_) if lg else (h >= SL_, l <= TP_)
    js = int(np.argmax(hs)) if hs.any() else BIG
    jt = int(np.argmax(ht)) if ht.any() else BIG
    if js <= jt and js < BIG:
        jj, f = js, (min(SL_, o[js]) * (1 - slip) if lg else max(SL_, o[js]) * (1 + slip))
    elif jt < BIG:
        jj, f = jt, (TP_ * (1 - slip) if lg else TP_ * (1 + slip))
    else:
        jj, f = hold - 1, (cc[-1] * (1 - slip) if lg else cc[-1] * (1 + slip))
    ret = ((f / e - 1) if lg else (1 - f / e)) - FEE
    return int(t15[j + jj]) + 900, ret / sf, sf


def pull_signals(s, sec):
    c = SP.CTX[s]
    T, B = PB.build_bars(c['t15'], c['a15'], sec)
    cl = B[:, 3]
    r48, r6, atr = PB._rsi(cl, 48), PB._rsi(cl, 6), PB._atr(B)
    bdt, bdc = c['bdt'], np.asarray(c['bdc'], dtype=float)
    sma = PB._sma(bdc, 50)
    out = []
    for i in range(200, len(T)):
        if r48[i] >= 56.3761 and r6[i] <= 33.7947 and np.isfinite(atr[i]):
            close = int(T[i]) + sec
            d = np.searchsorted(bdt, close - 86400, side='right') - 1
            if d >= 49 and bdc[d] < sma[d]:
                continue
            out.append((close, float(atr[i])))
    return out


SOURCES = {}   # name -> list of (close, coin, long, atr, sl, tp, hold)
bank = []
for s in COINS:
    for t, v in SP.bank_signals([dict(r, tp=1.0) for r in SP.BASE], s, 50).items():
        bank.append((t, s, SP.BASE[v[0]].get('side', 'LONG') == 'LONG', v[2], 3.0, 1.0, 192))
SOURCES['банк 1ч'] = bank
for sec, nm in ((3600, 'откат 1ч'), (7200, 'откат 2ч'), (14400, 'откат 4ч')):
    hold = min(48 * sec, 72 * 3600) // 900
    SOURCES[nm] = [(t, s, True, a, 3.0, 1.0, hold) for s in COINS for t, a in pull_signals(s, sec)]
print('  сигналов: %s' % ', '.join('%s %d' % (k, len(v)) for k, v in SOURCES.items()), flush=True)


def build(names):
    ev = []
    for pr, nm in enumerate(names):
        ev += [(t, pr, s, lg, a, sl, tp, hd) for t, s, lg, a, sl, tp, hd in SOURCES[nm]]
    ev.sort()
    busy, out = {}, []
    for t, pr, s, lg, a, sl, tp, hd in ev:
        if busy.get(s, 0) > t:
            continue
        c = SP.CTX[s]
        j = c['pos'].get(t)
        if j is None:
            continue
        r = sim_exit(c, j, lg, a, SLIP.get(s, 0.0003), sl, tp, hd)
        if r is None:
            continue
        end, R, sf = r
        out.append((t, end, R, sf, s, 1.0, pr))
        busy[s] = end
    out.sort()
    return out


def report(lbl, rows):
    plain = [x[:6] for x in rows]
    v = np.array([x[2] for x in rows])
    years = (rows[-1][0] - rows[0][0]) / (365.25 * 86400)
    r = sim_w.simulate(plain, 0.014)
    k, m = sim_w.money_at_dd(plain, 0.12)
    per = []
    for y in range(2022, 2027):
        sub = [x for x in plain if YT[y] <= x[0] < YT[y + 1]]
        per.append(sim_w.simulate(sub, 0.014)['eq'] if len(sub) >= 50 else float('nan'))
    print('  %-34s %5.0f сд/год ВР %4.1f%% ср%+.3f | 1.4%%: $%6.0f просадка %4.1f%%%s | DD12%%: $%6.0f (%.2f%%) | годы %s'
          % (lbl, len(v) / years, 100 * np.mean(v > 0), v.mean(), r['eq'], 100 * abs(L.dd_of(r['curve'])),
             ' ЗАЩ' if r['paused_at'] else '', m, 100 * k,
             ' '.join('%4.0f' % x if np.isfinite(x) else '   -' for x in per)), flush=True)



print('', flush=True)
print('  === ВТОРАЯ КНИГА (все 15 монет): стек таймфреймов ===', flush=True)
PER = {}
for names in (['банк 1ч'], ['банк 1ч', 'откат 1ч'], ['банк 1ч', 'откат 1ч', 'откат 2ч'],
              ['банк 1ч', 'откат 1ч', 'откат 2ч', 'откат 4ч']):
    rows = build(names)
    report(' + '.join(n.replace('откат ', 'о') for n in names), rows)
    plain = [x[:6] for x in rows]
    PER[len(names)] = [sim_w.simulate([x for x in plain if YT[y] <= x[0] < YT[y + 1]], 0.014)['eq']
                       for y in range(2022, 2027)]
b, c = PER[2], PER[3]
print('  +о2ч к банк+о1ч: лучше в %d из 5 лет (%s)'
      % (sum(1 for x, y in zip(b, c) if y > x), ' '.join('%+.0f' % (y - x) for x, y in zip(b, c))), flush=True)
