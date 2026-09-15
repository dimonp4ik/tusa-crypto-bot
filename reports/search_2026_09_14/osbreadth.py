"""Each of the bank's five rules on its own stop, take and hold.

Geometry was only ever searched for the whole book at once. The five rules fire on different things -
BTC pumps in the evening, coin runs, the night pump, a morning short, a pop in a downtrend - and there
is no reason they share an optimum. One rule at a time is varied (stop 2/3/4 ATR, take 0.75/1/1.5/2,
hold 24/48/72h) while the other four and both pullback sources stay at 3/1/48h.

Book: bank 1h + pullback 1h + pullback 2h, coins without XLM and AAVE, market entry at next 15m open,
measured costs, stops beyond 10% dropped, one position per coin, bank first. Money under the LIVE
stop_ref sizing at a 12% drawdown and at a fixed 1.4%. Each rule's geometry is then chosen blind on
four years and read on the fifth.
"""
import collections, csv, datetime, itertools, sys
import numpy as np
sys.path.insert(0, 'C:/Users/Lenovo/Desktop/Торговля/crypto-bot')
import struct_params as SP
import gauntlet2 as G
import nogate_attack as NA
import sim_w
import live_rules_sim as L
from src import pullback_bank as PB

cost = collections.defaultdict(list)
for r in csv.DictReader(open('book_frozen.csv')):
    cost[r['coin'] + 'USDT'].append(float(r['cost109']))
SLIP = {k: float(np.median(v)) for k, v in cost.items()}
SP.SLIP_REAL.update(SLIP); G.FC.clear()
FEE = 0.0004
COINS = [s for s in SP.COINS if s not in ('BILLUSDT', 'XLMUSDT', 'AAVEUSDT')]
PULL = [('rsi48', '>=', 56.3761), ('rsi6', '<=', 33.7947)]
MAXH = 288
YT = {y: int(datetime.datetime(y, 1, 1, tzinfo=datetime.UTC).timestamp()) for y in range(2022, 2028)}
RULES = [dict(r, tp=1.0) for r in SP.BASE]


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



BASE_HOLD = {'о1ч': 48, 'о2ч': 72}
BASE_G = (3.0, 1.0, 48)



GEO = {'r4': (2.0, 1.5, 48)}


def make_sig_raw(coins):
    sig = []
    for s in coins:
        c = SP.CTX[s]
        t15, a15, pos = c['t15'], c['a15'], c['pos']
        slip = SLIP.get(s, 0.0003)
        items = [(t, 0, 'r%d' % v[0], RULES[v[0]].get('side', 'LONG') == 'LONG', v[2])
                 for t, v in SP.bank_signals(RULES, s, 50).items()]
        items += [(t, 1, 'o1h', True, a) for t, a in pull_signals(s, 3600)]
        items += [(t, 2, 'o2h', True, a) for t, a in pull_signals(s, 7200)]
        T1, F = c['T1'], c['F']
        for close, prio, key, lg, atr in items:
            j = pos.get(close)
            if j is None or j + MAXH > len(t15) or not np.isfinite(atr) or atr <= 0:
                continue
            o, h, l, cc = (a15[j:j + MAXH, x] for x in range(4))
            e = o[0] * (1 + slip) if lg else o[0] * (1 - slip)
            fav = (h - e) / atr if lg else (e - l) / atr
            adv = (e - l) / atr if lg else (h - e) / atr
            i1 = int(np.searchsorted(T1, close - 3600))
            vr = float(F['volreg'][i1]) if i1 < len(T1) else float('nan')
            hour = datetime.datetime.fromtimestamp(close, datetime.UTC).hour
            sig.append((close, prio, key, s, e, atr, fav, adv, np.maximum.accumulate(fav),
                        np.maximum.accumulate(adv), (o - e) / atr if lg else (e - o) / atr,
                        (cc - e) / atr if lg else (e - cc) / atr, t15[j:j + MAXH], slip, vr, hour))
    sig.sort(key=lambda x: (x[0], x[1]))
    return sig



HOLDS = {'o1h': 48, 'o2h': 72}


def pull_off(s, sec, off):
    c = SP.CTX[s]
    T, B = PB.build_bars(np.asarray(c['t15'], dtype=np.int64) - off, c['a15'], sec)
    cl = B[:, 3]
    r48, r6, atr = PB._rsi(cl, 48), PB._rsi(cl, 6), PB._atr(B)
    bdt, bdc = c['bdt'], np.asarray(c['bdc'], dtype=float)
    sma = PB._sma(bdc, 50)
    out = []
    for i in range(200, len(T)):
        if r48[i] >= 56.3761 and r6[i] <= 33.7947 and np.isfinite(atr[i]):
            close = int(T[i]) + sec + off
            d = np.searchsorted(bdt, close - 86400, side='right') - 1
            if d >= 49 and bdc[d] < sma[d]:
                continue
            out.append((close, float(atr[i])))
    return out


def long_entries(coins, key, prio, sec, off):
    out = []
    for s in coins:
        c = SP.CTX[s]
        t15, a15, pos = c['t15'], c['a15'], c['pos']
        slip = SLIP.get(s, 0.0003)
        for close, atr in pull_off(s, sec, off):
            j = pos.get(close)
            if j is None or j + MAXH > len(t15) or atr <= 0:
                continue
            o, h, l, cc = (a15[j:j + MAXH, x] for x in range(4))
            e = o[0] * (1 + slip)
            fav = (h - e) / atr
            adv = (e - l) / atr
            out.append((close, prio, key, s, e, atr, fav, adv, np.maximum.accumulate(fav),
                        np.maximum.accumulate(adv), (o - e) / atr, (cc - e) / atr, t15[j:j + MAXH], slip,
                        float('nan'), 0))
    return out


def book(sig, keys):
    busy, out, cnt = {}, [], collections.Counter()
    for close, prio, key, s, e, atr, fav, adv, Mu, Md, on, cn, tt, slip, vr, hour in sig:
        if key not in keys or busy.get(s, 0) > close:
            continue
        sl, tp, hh = GEO.get(key, (3.0, 1.0, HOLDS.get(key, 48)))
        H = hh * 4
        sf = sl * atr / e
        if sf > 0.10:
            continue
        js = int(np.searchsorted(Md[:H], sl, side='left'))
        jt = int(np.searchsorted(Mu[:H], tp, side='left'))
        if js < H and js <= jt:
            jj, x = js, min(-sl, on[js])
        elif jt < H:
            jj, x = jt, tp
        else:
            jj, x = H - 1, cn[H - 1]
        end = int(tt[jj]) + 900
        out.append((close, end, (x * atr / e - slip - FEE) / sf, sf, s, 1.0))
        cnt[key] += 1
        busy[s] = end
    out.sort()
    return out, cnt


def fix(r6):
    r = sim_w.simulate(r6, 0.014)
    return r['eq'], abs(L.dd_of(r['curve']))


def pyr(r6):
    return [sim_w.simulate([x for x in r6 if YT[y] <= x[0] < YT[y + 1]], 0.014)['eq'] for y in range(2022, 2027)]


def tpy(rows):
    return len(rows) / ((rows[-1][0] - rows[0][0]) / (365.25 * 86400))



BASE_KEYS = {'r0', 'r1', 'r2', 'r3', 'r4', 'o1h', 'o2h'}
WEEK = 7 * 86400


def coin_hours(s):
    c = SP.CTX[s]
    T, B = PB.build_bars(c['t15'], c['a15'], 3600)
    cl = B[:, 3]
    r48, r6, atr = PB._rsi(cl, 48), PB._rsi(cl, 6), PB._atr(B)
    bdt, bdc = c['bdt'], np.asarray(c['bdc'], dtype=float)
    sma = PB._sma(bdc, 50)
    out = {}
    for i in range(200, len(T)):
        if not np.isfinite(atr[i]) or not np.isfinite(r48[i]) or not np.isfinite(r6[i]):
            continue
        close = int(T[i]) + 3600
        d = np.searchsorted(bdt, close - 86400, side='right') - 1
        ok = not (d >= 49 and bdc[d] < sma[d])
        out[close] = (float(r48[i]), float(r6[i]), float(atr[i]), ok)
    return out


def wide_entries(coins, H, key, prio, lo48, hi6, K, shift):
    breadth = collections.Counter()
    for s in coins:
        for close, (a, b, atr, ok) in H[s].items():
            if ok and a >= 56.3761 and b <= 33.7947:
                breadth[close] += 1
    out = []
    for s in coins:
        c = SP.CTX[s]
        t15, a15, pos = c['t15'], c['a15'], c['pos']
        slip = SLIP.get(s, 0.0003)
        for close, (a, b, atr, ok) in H[s].items():
            if not ok or atr <= 0:
                continue
            if a >= 56.3761 and b <= 33.7947:
                continue
            if not (a >= lo48 and b <= hi6):
                continue
            if K is not None and breadth.get(close - shift, 0) < K:
                continue
            j = pos.get(close)
            if j is None or j + MAXH > len(t15):
                continue
            o, h, l, cc = (a15[j:j + MAXH, x] for x in range(4))
            e = o[0] * (1 + slip)
            fav, adv = (h - e) / atr, (e - l) / atr
            out.append((close, prio, key, s, e, atr, fav, adv, np.maximum.accumulate(fav), np.maximum.accumulate(adv),
                        (o - e) / atr, (cc - e) / atr, t15[j:j + MAXH], slip, float('nan'), 0))
    return out


def fix2(rows):
    r = sim_w.simulate(rows, 0.014)
    return r['eq'], abs(L.dd_of(r['curve']))



BOOKS3 = (('КНИГА 1 (без XLM, AAVE)', [c for c in SP.COINS if c not in ('BILLUSDT', 'XLMUSDT', 'AAVEUSDT')]),
          ('КНИГА 2 (все 15 монет)', [c for c in SP.COINS if c != 'BILLUSDT']),
          ('КНИГА 3 (без AAVE, с XLM)', [c for c in SP.COINS if c not in ('BILLUSDT', 'AAVEUSDT')]))


def boost(rows, nmin=10, mult=1.25):
    cnt = collections.Counter(x[0] for x in rows)
    return [x[:5] + ((mult if cnt[x[0]] >= nmin else 1.0),) for x in rows]



DAY = 86400
ACC = BASE_KEYS | {'w50'}


def os_entries(coins, H, key, prio, lo48, T6, K, shift):
    breadth = collections.Counter()
    for s in coins:
        for close, (a, b, atr, ok) in H[s].items():
            if ok and b <= T6:
                breadth[close] += 1
    out = []
    for s in coins:
        c = SP.CTX[s]
        t15, a15, pos = c['t15'], c['a15'], c['pos']
        slip = SLIP.get(s, 0.0003)
        for close, (a, b, atr, ok) in H[s].items():
            if not ok or atr <= 0 or (a >= 56.3761 and b <= 33.7947) or a < lo48:
                continue
            if breadth.get(close - shift, 0) < K:
                continue
            j = pos.get(close)
            if j is None or j + MAXH > len(t15):
                continue
            o, h, l, cc = (a15[j:j + MAXH, x] for x in range(4))
            e = o[0] * (1 + slip)
            fav, adv = (h - e) / atr, (e - l) / atr
            out.append((close, prio, key, s, e, atr, fav, adv, np.maximum.accumulate(fav), np.maximum.accumulate(adv),
                        (o - e) / atr, (cc - e) / atr, t15[j:j + MAXH], slip, float('nan'), 0))
    return out


CELLS = [(30.0, 4), (30.0, 6), (30.0, 8), (35.0, 5), (35.0, 7), (35.0, 9)]
CTRL = [(35.0, 7, 3 * DAY), (35.0, 7, 7 * DAY), (30.0, 6, 3 * DAY), (30.0, 6, 7 * DAY)]


def ck(T6, K, sh=0):
    return 'o%d_%d%s' % (int(T6), K, ('s%d' % (sh // DAY)) if sh else '')


for book_name, coins in BOOKS3:
    H = {s: coin_hours(s) for s in coins}
    base_sig = make_sig_raw(coins)
    extra = wide_entries(coins, H, 'w50', 4, 50.0, 45.0, 4, 0)
    for T6, K in CELLS:
        extra += os_entries(coins, H, ck(T6, K), 6, 50.0, T6, K, 0)
    for T6, K, sh in CTRL:
        extra += os_entries(coins, H, ck(T6, K, sh), 6, 50.0, T6, K, sh)
    sig_all = sorted(base_sig + extra, key=lambda x: (x[0], x[1]))
    acc = boost(book(sig_all, ACC)[0])
    aeq, add = fix2(acc); am = sim_w.money_at_dd(acc, 0.12)[1]
    apy = pyr(acc)
    print('', flush=True)
    print('  ===== %s: ПРИНЯТОЕ (буст + ширина 50/45 K4) $%.0f %.1f%% DD12 $%.0f =====' % (book_name, aeq, 100 * add, am), flush=True)
    RES = {}
    for T6, K, sh in [(t, k, 0) for t, k in CELLS] + CTRL:
        k = ck(T6, K, sh)
        rows, cnt = book(sig_all, ACC | {k})
        alone, _ = book(sorted([x for x in extra if x[2] == k], key=lambda x: (x[0], x[1])), {k})
        va = np.array([x[2] for x in alone]) if alone else np.zeros(1)
        rows = boost(rows)
        eq, dd = fix2(rows)
        m = sim_w.money_at_dd(rows, 0.12)[1]
        py = pyr(rows)
        if not sh:
            RES[k] = (eq, m)
        print('    %s rsi6<=%d на K>=%d %s | в книге %3d сд, сам %4d сд ВР %5.1f%% ср %+.3f | $%6.0f %4.1f%% DD12 $%6.0f %s | лучше принятого по годам %d/5 (%s)'
              % ('КОНТРОЛЬ' if sh else '        ', int(T6), K, ('ширина -%dд' % (sh // DAY)) if sh else '          ', cnt[k], len(alone),
                 100 * np.mean(va > 0), va.mean(), eq, 100 * dd, m, '+' if (eq > aeq and m > am) else ' ',
                 sum(1 for a, b in zip(apy, py) if b > a), ' '.join('%+.0f' % (b - a) for a, b in zip(apy, py))), flush=True)
    top = sorted(RES, key=lambda k: -RES[k][1])[:2]
    print('    ЗАЛИВКА 85%% против ПРИНЯТОГО, 10 парных розыгрышей: %s' % ', '.join(top), flush=True)
    for k in top:
        A, B = [], []
        for seed in range(1, 11):
            rng = np.random.default_rng(seed)
            sub = [x for x in sig_all if rng.random() < 0.85]
            r0 = boost(book(sub, ACC)[0])
            r1 = boost(book(sub, ACC | {k})[0])
            e0, d0 = fix2(r0); e1, d1 = fix2(r1)
            A.append((e0, d0, sim_w.money_at_dd(r0, 0.12)[1], pyr(r0)))
            B.append((e1, d1, sim_w.money_at_dd(r1, 0.12)[1], pyr(r1)))
        med = lambda X, i: float(np.median([x[i] for x in X]))
        yw = ' '.join('%d/10' % sum(1 for x, y in zip(A, B) if y[3][j] > x[3][j]) for j in range(5))
        print('      %-6s: принятое $%.0f %.1f%% DD12 $%.0f | вариант $%.0f %.1f%% (макс %.1f%%) DD12 $%.0f | лучше $ %d/10, DD12 %d/10 | годы %s'
              % (k, med(A, 0), 100 * med(A, 1), med(A, 2), med(B, 0), 100 * med(B, 1), 100 * max(x[1] for x in B), med(B, 2),
                 sum(1 for x, y in zip(A, B) if y[0] > x[0]), sum(1 for x, y in zip(A, B) if y[2] > x[2]), yw), flush=True)
