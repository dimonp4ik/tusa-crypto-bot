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
HOLDS.update({'w50': 48, 'w2': 72, 'wU': 48})
LONG_RULES = {k for k, r in enumerate(RULES) if r.get('side', 'LONG') == 'LONG'}


def coin_bars(s, sec):
    c = SP.CTX[s]
    T, B = PB.build_bars(c['t15'], c['a15'], sec)
    cl = B[:, 3]
    r48, r6, atr = PB._rsi(cl, 48), PB._rsi(cl, 6), PB._atr(B)
    bdt, bdc = c['bdt'], np.asarray(c['bdc'], dtype=float)
    sma = PB._sma(bdc, 50)
    out = {}
    for i in range(200, len(T)):
        if not (np.isfinite(atr[i]) and np.isfinite(r48[i]) and np.isfinite(r6[i])):
            continue
        close = int(T[i]) + sec
        d = np.searchsorted(bdt, close - 86400, side='right') - 1
        out[close] = (float(r48[i]), float(r6[i]), float(atr[i]), not (d >= 49 and bdc[d] < sma[d]))
    return out


def union_entries(coins, H, key, prio, lo48, hi6, K, shift):
    breadth = collections.Counter()
    bank_long = collections.defaultdict(set)
    for s in coins:
        for t, v in SP.bank_signals(RULES, s, 50).items():
            if v[0] in LONG_RULES:
                bank_long[t].add(s)
        for close, (a, b, atr, ok) in H[s].items():
            if ok and a >= 56.3761 and b <= 33.7947:
                bank_long[close].add(s)
    for t, ss in bank_long.items():
        breadth[t] = len(ss)
    out = []
    for s in coins:
        c = SP.CTX[s]
        t15, a15, pos = c['t15'], c['a15'], c['pos']
        slip = SLIP.get(s, 0.0003)
        for close, (a, b, atr, ok) in H[s].items():
            if not ok or atr <= 0 or (a >= 56.3761 and b <= 33.7947) or not (a >= lo48 and b <= hi6):
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


BEST = BASE_KEYS | {'w50'}
VARS = [('система (сейчас)', BASE_KEYS, False),
        ('ЛУЧШЕЕ: буст + ширина 50/45 K4', BEST, True),
        ('+ ширина 2ч K3', BEST | {'w2_3'}, True),
        ('+ ширина 2ч K4', BEST | {'w2_4'}, True),
        ('  КОНТРОЛЬ 2ч K3 -3 дня', BEST | {'w2_3s3'}, True),
        ('  КОНТРОЛЬ 2ч K3 -7 дней', BEST | {'w2_3s7'}, True),
        ('широкий счёт 1ч+банк K5 вместо 1ч K4', BASE_KEYS | {'wU5'}, True),
        ('широкий счёт 1ч+банк K6 вместо 1ч K4', BASE_KEYS | {'wU6'}, True),
        ('  КОНТРОЛЬ широкий K5 -3 дня', BASE_KEYS | {'wU5s3'}, True),
        ('  КОНТРОЛЬ широкий K5 -7 дней', BASE_KEYS | {'wU5s7'}, True)]
HOLDS.update({'w2_3': 72, 'w2_4': 72, 'w2_3s3': 72, 'w2_3s7': 72})


def build2(sig_all, keys, do_boost):
    rows, cnt = book(sig_all, keys)
    return (boost(rows) if do_boost else rows), cnt


for book_name, coins in BOOKS3:
    H1 = {s: coin_hours(s) for s in coins}
    H2 = {s: coin_bars(s, 7200) for s in coins}
    base_sig = make_sig_raw(coins)
    extra = wide_entries(coins, H1, 'w50', 4, 50.0, 45.0, 4, 0)
    for key, K, sh in (('w2_3', 3, 0), ('w2_4', 4, 0), ('w2_3s3', 3, 3 * DAY), ('w2_3s7', 3, 7 * DAY)):
        extra += wide_entries(coins, H2, key, 5, 50.0, 45.0, K, sh)
    for key, K, sh in (('wU5', 5, 0), ('wU6', 6, 0), ('wU5s3', 5, 3 * DAY), ('wU5s7', 5, 7 * DAY)):
        extra += union_entries(coins, H1, key, 4, 50.0, 45.0, K, sh)
    sig_all = sorted(base_sig + extra, key=lambda x: (x[0], x[1]))
    print('', flush=True)
    print('  ===== %s =====' % book_name, flush=True)
    RES = {}
    for lbl, keys, bo in VARS:
        rows, cnt = build2(sig_all, keys, bo)
        RES[lbl] = rows
        eq, dd = fix2(rows)
        m = sim_w.money_at_dd(rows, 0.12)[1]
        v = np.array([x[2] for x in rows])
        yrs = (rows[-1][0] - rows[0][0]) / (365.25 * 86400)
        added = ', '.join('%s %d' % (k, n) for k, n in sorted(cnt.items()) if k not in BASE_KEYS)
        print('    100%% %-40s %4.0f сд/год ВР %4.1f%% | $%6.0f %4.1f%% | DD12 $%6.0f | %s' % (lbl, len(v) / yrs, 100 * np.mean(v > 0), eq, 100 * dd, m, added), flush=True)
    R = {lbl: [] for lbl, _, _ in VARS}
    for seed in range(1, 11):
        rng = np.random.default_rng(seed)
        sub = [x for x in sig_all if rng.random() < 0.85]
        for lbl, keys, bo in VARS:
            rows, _ = build2(sub, keys, bo)
            e, d = fix2(rows)
            R[lbl].append(dict(eq=e, dd=d, m=sim_w.money_at_dd(rows, 0.12)[1], yrs=pyr(rows)))
    ref = R['ЛУЧШЕЕ: буст + ширина 50/45 K4']
    med = lambda X, k: float(np.median([x[k] for x in X]))
    print('    85%% заливки против ЛУЧШЕГО (медиана 10 розыгрышей):', flush=True)
    for lbl, _, _ in VARS:
        X = R[lbl]
        yw = ' '.join('%d/10' % sum(1 for a, x in zip(ref, X) if x['yrs'][k] > a['yrs'][k]) for k in range(5))
        print('      %-40s $%6.0f %4.1f%% (макс %4.1f%%) DD12 $%6.0f | лучше ЛУЧШЕГО $ %2d/10 DD12 %2d/10 | по годам %s'
              % (lbl, med(X, 'eq'), 100 * med(X, 'dd'), 100 * max(x['dd'] for x in X), med(X, 'm'),
                 sum(1 for a, x in zip(ref, X) if x['eq'] > a['eq']), sum(1 for a, x in zip(ref, X) if x['m'] > a['m']), yw), flush=True)
    print('    СКОЛЬЗЯЩИЙ ВЫБОР (без контролей): по $ при DD12 на 4 годах, замер 5-го при 1.4% против ЛУЧШЕГО', flush=True)
    wins = 0
    for y in range(2022, 2027):
        lo, hi = YT[y], YT[y + 1]
        best, bk = -1, None
        for lbl, rows in RES.items():
            if 'КОНТРОЛЬ' in lbl:
                continue
            mm = sim_w.money_at_dd([x for x in rows if not (lo <= x[0] < hi)], 0.12)[1]
            if mm > best:
                best, bk = mm, lbl
        mp = sim_w.simulate([x for x in RES[bk] if lo <= x[0] < hi], 0.014)['eq']
        mb = sim_w.simulate([x for x in RES['ЛУЧШЕЕ: буст + ширина 50/45 K4'] if lo <= x[0] < hi], 0.014)['eq']
        wins += int(mp > mb)
        print('      %d выбрано «%s» | $%.0f против $%.0f %s' % (y, bk, mp, mb, 'лучше' if mp > mb else ('так же' if mp == mb else 'хуже')), flush=True)
    print('      лучше ЛУЧШЕГО в %d из 5 лет' % wins, flush=True)
