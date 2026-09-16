"""The three decisions in the bank that are not thresholds, and were never re-picked blind.

    btc_sma = 50      longs are blocked while BTC's last daily close is under its SMA(50)
    HOLD_BARS = 192   the time exit, 48 hours
    rule order        first match wins, so the order of the five rules decides which one
                      claims an hour where two of them agree

The first two are numbers somebody chose. The third is not even written down as a choice - it is
the order the rules happen to sit in the file, and it silently decides the take and stop of every
overlapping signal.

Same protocol as the thresholds: score on 2022-2024, pick the winner there, read 2025-2026 once.
Real per-coin book costs throughout.
"""
import bisect
import collections
import datetime
import itertools
import sys

import numpy as np

sys.path.insert(0, 'C:/Users/Lenovo/Desktop/Торговля/crypto-bot')
import mrlib as M
import tm_tag as T
from src import pullback_bank as PB

SLIP_REAL = {
    'BTCUSDT': 0.00000, 'ETHUSDT': 0.00000, 'SOLUSDT': 0.00005, 'ADAUSDT': 0.00033,
    'LINKUSDT': 0.00029, 'AVAXUSDT': 0.00027, 'XRPUSDT': 0.00004, 'DOTUSDT': 0.00026,
    'XLMUSDT': 0.00106, 'SUIUSDT': 0.00014, 'HYPEUSDT': 0.00006, 'ZECUSDT': 0.00002,
    'AAVEUSDT': 0.00077, 'TAOUSDT': 0.00021, 'NEARUSDT': 0.00021, 'BILLUSDT': 0.00386,
}
COST = 0.0004
COINS = [c for c in M.PIN if c not in ('SEIUSDT', 'LABUSDT')]
HOUR = PB.HOUR
BIG = 10 ** 9
FIT_END = int(datetime.datetime(2025, 1, 1, tzinfo=datetime.UTC).timestamp())
BASE = PB.RULE_SETS['strict3+short2_wide']

CTX = {}
for s in COINS:
    t15, a15 = T.data[s]['T'], T.data[s]['A']
    T1, B1 = PB.build_bars(t15, a15, HOUR)
    F = PB.features(T1, B1, T.bt, T.ba)
    iv = PB.trend_regime(t15, a15, *PB.btc_daily(T.bt, T.ba))
    bdt, bdc = PB.btc_daily(T.bt, T.ba)
    bdc = np.asarray(bdc, dtype=float)
    CTX[s] = dict(t15=t15, a15=a15, T1=T1, B1=B1, F=F, iv=iv, starts=[x[0] for x in iv],
                  bdt=bdt, bdc=bdc, pos=dict(zip(t15.tolist(), range(len(t15)))),
                  sma={n: PB._sma(bdc, n) for n in (20, 30, 50, 80, 120, 200)})
print('  контекст: %d монет' % len(CTX), flush=True)


def bank_signals(rules, coin, sma_n):
    c = CTX[coin]
    out = {}
    T1, F = c['T1'], c['F']
    if sma_n and sma_n not in c['sma']:
        c['sma'][sma_n] = PB._sma(c['bdc'], sma_n)
    bsm = c['sma'][sma_n] if sma_n else None
    for i in range(PB.MIN_HOURS, len(T1)):
        close = int(T1[i]) + HOUR
        j = bisect.bisect_right(c['starts'], close) - 1
        if j < 0 or not (c['iv'][j][0] <= close < c['iv'][j][1]) or np.isnan(F['atr'][i]):
            continue
        side = c['iv'][j][2]
        if side == 'LONG' and bsm is not None:
            d = bisect.bisect_right(c['bdt'], close - 86400) - 1
            if d >= sma_n - 1 and c['bdc'][d] < bsm[d]:
                continue
        for k, r in enumerate(rules):
            if r.get('side', 'LONG') == side and PB._match(r, F, i):
                out[close] = (k, float(c['B1'][i, 3]), float(F['atr'][i]))
                break
    return out


def portfolio(rules, sma_n=50, hold=192):
    rows = []
    for s in COINS:
        c = CTX[s]
        slip = SLIP_REAL.get(s, 0.0002)
        t15, A15, pos = c['t15'], c['a15'], c['pos']
        sig = bank_signals(rules, s, sma_n)
        busy = 0
        for close in sorted(sig):
            if close < busy:
                continue
            i = pos.get(close)
            if i is None or i + hold > len(t15):
                continue
            ri, lim, atr = sig[close]
            r = rules[ri]
            lg = r.get('side', 'LONG') == 'LONG'
            o, h, l, cc = (A15[i:i + hold, x] for x in range(4))
            if lg:
                if o[0] <= lim:
                    e, touch = o[0] * (1 + slip), False
                elif l[0] <= lim:
                    e, touch = lim * (1 + slip), True
                else:
                    continue
                TP, SL = e + r['tp'] * atr, e - r['sl'] * atr
                hs, ht = l <= SL, h >= TP
            else:
                if o[0] >= lim:
                    e, touch = o[0] * (1 - slip), False
                elif h[0] >= lim:
                    e, touch = lim * (1 - slip), True
                else:
                    continue
                TP, SL = e - r['tp'] * atr, e + r['sl'] * atr
                hs, ht = h >= SL, l <= TP
            if touch:
                ht = ht.copy()
                ht[0] = False
            js = int(np.argmax(hs)) if hs.any() else BIG
            jt = int(np.argmax(ht)) if ht.any() else BIG
            if js <= jt and js < BIG:
                jj = js
                f_ = (min(SL, o[js]) * (1 - slip)) if lg else (max(SL, o[js]) * (1 + slip))
            elif jt < BIG:
                jj, f_ = jt, (TP * (1 - slip) if lg else TP * (1 + slip))
            else:
                jj = hold - 1
                f_ = cc[-1] * (1 - slip) if lg else cc[-1] * (1 + slip)
            ret = ((f_ / e - 1) if lg else (1 - f_ / e)) - COST
            rows.append((close, int(t15[i + jj]) + 900, ret / (r['sl'] * atr / e)))
            busy = int(t15[i + jj]) + 900
    return sorted(rows)


def stats(rows, lo=None, hi=None):
    v = [x for x in rows if (lo is None or x[0] >= lo) and (hi is None or x[0] < hi)]
    if len(v) < 40:
        return None
    v = sorted(v, key=lambda z: z[1])
    a = np.array([x[2] for x in v])
    eq = np.cumsum(a)
    dd = float((eq - np.maximum.accumulate(np.r_[0, eq])[1:]).min())
    mo = collections.defaultdict(float)
    for x in v:
        mo[datetime.datetime.fromtimestamp(int(x[1]), datetime.UTC).strftime('%y-%m')] += x[2]
    mv = np.array(list(mo.values()))
    return dict(n=len(a), wr=float(np.mean(a > 0)), avg=float(a.mean()),
                r_mo=float(mv.mean()), dd=dd, ratio=float(mv.mean() / max(-dd, 1e-9)),
                pos=int((mv > 0).sum()), nm=len(mv))


def show(st):
    if st is None:
        return 'мало сделок'
    return ('n%5d ВР%4.0f%% ср%+.4fR R/мес%+5.2f DD%+7.1f отн%5.2f (мес +%d/%d)'
            % (st['n'], 100 * st['wr'], st['avg'], st['r_mo'], st['dd'], st['ratio'],
               st['pos'], st['nm']))


if __name__ == '__main__':
    base = portfolio(BASE)
    bf, be = stats(base, hi=FIT_END), stats(base, lo=FIT_END)
    print('', flush=True)
    print('  БАЗА  обучение: %s' % show(bf), flush=True)
    print('        экзамен : %s' % show(be), flush=True)

    print('', flush=True)
    print('  === фильтр BTC по дневной SMA (в коде 50; 0 = фильтра нет) ===', flush=True)
    best = None
    seen = {}
    for n in (20, 30, 50, 80, 120, 200):
        rows = portfolio(BASE, sma_n=n)
        sf, se = stats(rows, hi=FIT_END), stats(rows, lo=FIT_END)
        seen[n] = (sf, se)
        if sf and (best is None or sf['ratio'] > best[1]):
            best = (n, sf['ratio'])
        print('    SMA%4d  обучение: %s' % (n, show(sf)), flush=True)
    print('    -> выбор по обучению: SMA%d (отн %.2f)' % best, flush=True)
    print('    -> ЭКЗАМЕН при нём: %s' % show(seen[best[0]][1]), flush=True)
    print('    -> экзамен базы   : %s' % show(be), flush=True)

    print('', flush=True)
    print('  === время выхода (в коде 192 бара = 48ч; максимум по требованию 3 суток) ===', flush=True)
    best = None
    seen = {}
    for hb in (48, 96, 144, 192, 240, 288):
        rows = portfolio(BASE, hold=hb)
        sf, se = stats(rows, hi=FIT_END), stats(rows, lo=FIT_END)
        seen[hb] = (sf, se)
        if sf and (best is None or sf['ratio'] > best[1]):
            best = (hb, sf['ratio'])
        print('    %3dч  обучение: %s' % (hb // 4, show(sf)), flush=True)
    print('    -> выбор по обучению: %dч (отн %.2f)' % (best[0] // 4, best[1]), flush=True)
    print('    -> ЭКЗАМЕН при нём: %s' % show(seen[best[0]][1]), flush=True)
    print('    -> экзамен базы   : %s' % show(be), flush=True)

    print('', flush=True)
    print('  === порядок правил (первое совпавшее забирает час) ===', flush=True)
    longs = [r for r in BASE if r.get('side', 'LONG') == 'LONG']
    shorts = [r for r in BASE if r.get('side') == 'SHORT']
    best = None
    seen = {}
    for pl in itertools.permutations(range(len(longs))):
        for ps in itertools.permutations(range(len(shorts))):
            order = [longs[i] for i in pl] + [shorts[i] for i in ps]
            rows = portfolio(order)
            sf, se = stats(rows, hi=FIT_END), stats(rows, lo=FIT_END)
            key = (pl, ps)
            seen[key] = (sf, se)
            if sf and (best is None or sf['ratio'] > best[1]):
                best = (key, sf['ratio'])
            tag = '+'.join(longs[i]['name'][:9] for i in pl) + ' | ' + \
                  '+'.join(shorts[i]['name'][12:20] for i in ps)
            star = ' <<< в коде' if pl == (0, 1, 2) and ps == (0, 1) else ''
            print('    %-46s обучение отн %.2f  (ср%+.4f DD%+6.1f)%s'
                  % (tag, sf['ratio'], sf['avg'], sf['dd'], star), flush=True)
    print('    -> выбор по обучению: %s (отн %.2f)' % (best[0], best[1]), flush=True)
    print('    -> ЭКЗАМЕН при нём: %s' % show(seen[best[0]][1]), flush=True)
    print('    -> экзамен базы   : %s' % show(be), flush=True)
