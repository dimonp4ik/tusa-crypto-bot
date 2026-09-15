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
COINS = [s for s in SP.COINS if s not in ('BILLUSDT', 'XLMUSDT', 'AAVEUSDT')]
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



"""Structure events from the old SMC bot, as signals on 1h bars.

The SMC bot read the market through events the bank never sees: a sweep of a recent low that closes
back above it (stops harvested, then reversal), a break of structure, a fair value gap, an engulfing
candle. Each is tested raw, with the BTC SMA(50) filter on longs, and with a strength filter (the
pullback rule's rsi48 >= 56.38 for longs, <= 43.62 for shorts). Market entry at the next 15m open,
stop 3 / take 1 ATR, 48h, measured costs, stops beyond 10% dropped, one position per coin.
Null: every hour taken, per side. Mirror: the same event traded the other way.
"""


def hourly(s):
    c = SP.CTX[s]
    B = c['B1']
    return c['T1'], B[:, 0], B[:, 1], B[:, 2], B[:, 3], c['F']


def rolling_prev(x, n, fn):
    out = np.full(len(x), np.nan)
    for i in range(n, len(x)):
        out[i] = fn(x[i - n:i])
    return out


EVENTS = {}
for s in COINS:
    T, o, h, l, cl, F = hourly(s)
    ev = {}
    for n in (12, 24, 48):
        pl, ph = rolling_prev(l, n, np.min), rolling_prev(h, n, np.max)
        ev['снятие_мин_%d' % n] = (l < pl) & (cl > pl), True
        ev['снятие_макс_%d' % n] = (h > ph) & (cl < ph), False
        ev['слом_вверх_%d' % n] = cl > ph, True
        ev['слом_вниз_%d' % n] = cl < pl, False
    gu = np.zeros(len(cl), bool); gd = np.zeros(len(cl), bool)
    gu[2:] = l[2:] > h[:-2]; gd[2:] = h[2:] < l[:-2]
    ev['разрыв_вверх'] = gu, True
    ev['разрыв_вниз'] = gd, False
    eb = np.zeros(len(cl), bool); es = np.zeros(len(cl), bool)
    eb[1:] = (cl[1:] > o[1:]) & (cl[:-1] < o[:-1]) & (o[1:] <= cl[:-1]) & (cl[1:] >= o[:-1])
    es[1:] = (cl[1:] < o[1:]) & (cl[:-1] > o[:-1]) & (o[1:] >= cl[:-1]) & (cl[1:] <= o[:-1])
    ev['поглощение_вверх'] = eb, True
    ev['поглощение_вниз'] = es, False
    EVENTS[s] = (T, ev, F)
NAMES = sorted(EVENTS[COINS[0]][1])
RSI48 = {}
for s in COINS:
    c = SP.CTX[s]
    RSI48[s] = PB._rsi(c['B1'][:, 3], 48)


def sources_for(name, mode, flip=False):
    rows = []
    for s in COINS:
        c = SP.CTX[s]
        T, ev, F = EVENTS[s]
        mask, lg = ev[name]
        if flip:
            lg = not lg
        bdt, bdc = c['bdt'], np.asarray(c['bdc'], dtype=float)
        sma = PB._sma(bdc, 50)
        r48 = RSI48[s]
        for i in np.flatnonzero(mask):
            if i < 200 or not np.isfinite(F['atr'][i]):
                continue
            close = int(T[i]) + 3600
            if mode in ('btc', 'сила') and lg:
                d = np.searchsorted(bdt, close - 86400, side='right') - 1
                if d >= 49 and bdc[d] < sma[d]:
                    continue
            if mode == 'сила' and ((lg and not r48[i] >= 56.3761) or (not lg and not r48[i] <= 43.6239)):
                continue
            rows.append((close, s, lg, float(F['atr'][i]), 3.0, 1.0, 192))
    return rows


def run(src):
    globals()['SOURCES'] = {'x': src}
    rows = build(['x'])
    if len(rows) < 60:
        return None
    v = np.array([x[2] for x in rows])
    yrs = []
    for y in range(2022, 2027):
        vy = [x[2] for x in rows if YT[y] <= x[0] < YT[y + 1]]
        yrs.append(np.mean(vy) if len(vy) >= 15 else np.nan)
    months = (rows[-1][1] - rows[0][0]) / (365.25 * 86400 / 12)
    return dict(n=len(v), tpy=len(v) / (months / 12), wr=float(np.mean(v > 0)), avg=float(v.mean()),
                yrs=yrs, conv=len(v) / max(len(src), 1))


print('  === нулевая модель: каждый час, 3/1/48ч ===', flush=True)
for lg in (True, False):
    src = []
    for s in COINS:
        c = SP.CTX[s]
        T, F = c['T1'], c['F']
        for i in range(200, len(T), 3):
            if np.isfinite(F['atr'][i]):
                src.append((int(T[i]) + 3600, s, lg, float(F['atr'][i]), 3.0, 1.0, 192))
    r = run(src)
    print('    %-5s сделок/год %4.0f ВР %.1f%% ср %+.4f' % ('LONG' if lg else 'SHORT', r['tpy'], 100 * r['wr'], r['avg']), flush=True)

print('', flush=True)
print('  событие               фильтр  сделок/год  конв   ВР     ср R    по годам                          плюс лет | зеркало ср R', flush=True)
for name in NAMES:
    for mode in ('сырое', 'btc', 'сила'):
        r = run(sources_for(name, mode))
        if r is None:
            continue
        m = run(sources_for(name, mode, flip=True))
        pos = sum(1 for y in r['yrs'] if np.isfinite(y) and y > 0)
        seen = sum(1 for y in r['yrs'] if np.isfinite(y))
        print('  %-20s  %-5s  %6.0f     %3.0f%%  %4.1f%%  %+.4f  %s  %d/%d | %s'
              % (name, mode, r['tpy'], 100 * r['conv'], 100 * r['wr'], r['avg'],
                 ' '.join('%+.3f' % y if np.isfinite(y) else '  -   ' for y in r['yrs']), pos, seen,
                 ('%+.4f' % m['avg']) if m else '  -'), flush=True)
