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


def roll_entries(coins, H, key, prio, W, K, shift, lo48=50.0, hi6=45.0):
    fired = collections.defaultdict(set)
    for s in coins:
        for close, (a, b, atr, ok) in H[s].items():
            if ok and a >= 56.3761 and b <= 33.7947:
                fired[close].add(s)
    closes = sorted(fired)
    breadth = {}
    for t in set(c for s in coins for c in H[s]):
        u = set()
        for w in range(W):
            u |= fired.get(t - shift - w * 3600, set())
        breadth[t] = len(u)
    out = []
    for s in coins:
        c = SP.CTX[s]
        t15, a15, pos = c['t15'], c['a15'], c['pos']
        slip = SLIP.get(s, 0.0003)
        for close, (a, b, atr, ok) in H[s].items():
            if not ok or atr <= 0 or (a >= 56.3761 and b <= 33.7947) or not (a >= lo48 and b <= hi6):
                continue
            if breadth.get(close, 0) < K:
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



import bisect
import datetime as DT
FULL = BASE_KEYS | {'r2_4'}
SHORT_KEYS = {'r%d' % k for k, r in enumerate(RULES) if r.get('side', 'LONG') == 'SHORT'}


def book_keyed(sig, keys):
    busy, out = {}, []
    for close, prio, key, s, e, atr, fav, adv, Mu, Md, on, cn, tt, slip, vr, hour in sig:
        if key not in keys or busy.get(s, 0) > close:
            continue
        sl, tp, hh = GEO.get(key, (3.0, 1.0, HOLDS.get(key, 48)))
        Hh = hh * 4
        sf = sl * atr / e
        if sf > 0.10:
            continue
        js = int(np.searchsorted(Md[:Hh], sl, side='left'))
        jt = int(np.searchsorted(Mu[:Hh], tp, side='left'))
        if js < Hh and js <= jt:
            jj, x = js, min(-sl, on[js])
        elif jt < Hh:
            jj, x = jt, tp
        else:
            jj, x = Hh - 1, cn[Hh - 1]
        end = int(tt[jj]) + 900
        out.append((close, end, (x * atr / e - slip - FEE) / sf, sf, s, key not in SHORT_KEYS))
        busy[s] = end
    out.sort()
    return out


rng = np.random.default_rng(5)
for book_name, coins in BOOKS3:
    H = {s: coin_hours(s) for s in coins}
    base_sig = make_sig_raw(coins)
    extra = roll_entries(coins, H, 'r2_4', 4, 2, 4, 0)
    sig_all = sorted(base_sig + extra, key=lambda x: (x[0], x[1]))
    rows = book_keyed(sig_all, FULL)
    by_coin = collections.defaultdict(list)
    for close, end, R, sf, s, lg in rows:
        by_coin[s].append((close, end, lg, R))
    breadth = collections.Counter()
    for s in coins:
        for close, (a, b, atr, ok) in H[s].items():
            if ok and a >= 56.3761 and b <= 33.7947:
                breadth[close] += 1
    roll_b = {}
    for t in set(c for s in coins for c in H[s]):
        roll_b[t] = breadth.get(t, 0) + breadth.get(t - 3600, 0)
    print('', flush=True)
    print('  ===== %s: сделок %d =====' % (book_name, len(rows)), flush=True)
    MOVES, BASE_H, FEAT = [], [], {}
    for s in coins:
        c = SP.CTX[s]
        t15, a15, T1, F = c['t15'], c['a15'], c['T1'], c['F']
        cl = c['B1'][:, 3]
        FEAT[s] = dict(T1=T1, volreg=F['volreg'], rsi14=F['rsi14'], rsi48=PB._rsi(cl, 48), rsi6=PB._rsi(cl, 6),
                       btc24=F['btc24'], ret24=F['ret24'], rng=F['rng'], iv=c['iv'], starts=c['starts'])
        day = t15 // 86400
        bounds = np.r_[0, np.flatnonzero(np.diff(day)) + 1, len(t15)]
        daily = []
        for a, b in zip(bounds[:-1], bounds[1:]):
            if b - a < 80:
                continue
            lo, hi = a15[a:b, 2], a15[a:b, 1]
            gain = hi / np.minimum.accumulate(lo) - 1
            k = int(np.argmax(gain))
            st = int(np.argmin(lo[:k + 1]))
            drop = 1 - lo / np.maximum.accumulate(hi)
            k2 = int(np.argmax(drop))
            st2 = int(np.argmin(-hi[:k2 + 1]))
            daily.append((int(t15[a + st]), int(t15[a + k]), float(gain[k]), int(t15[a + st2]), int(t15[a + k2]), float(drop[k2])))
        if not daily:
            continue
        up_thr, dn_thr = np.quantile([d[2] for d in daily], 0.8), np.quantile([d[5] for d in daily], 0.8)
        tr = sorted(by_coin[s])
        for st, pk, g, st2, pk2, dr in daily:
            for lg, a0, a1, size, thr in ((True, st, pk, g, up_thr), (False, st2, pk2, dr, dn_thr)):
                if size < thr or a1 <= a0:
                    continue
                over = [x for x in tr if x[2] == lg and x[0] < a1 and x[1] > a0]
                near = [x for x in tr if x[2] == lg and abs(x[0] - a0) <= 6 * 3600]
                MOVES.append(dict(coin=s, long=lg, start=a0, size=size, caught=bool(over), near=bool(near),
                                  R=float(np.mean([x[3] for x in over])) if over else float('nan')))
        idx = rng.choice(np.arange(800, len(T1)), size=min(300, max(1, len(T1) - 800)), replace=False)
        BASE_H += [(s, int(i)) for i in idx]

    def feats_at(s, ts):
        f = FEAT[s]
        i = int(np.searchsorted(f['T1'], ts - 3600, side='right')) - 1
        if i < 800 or i >= len(f['T1']):
            return None
        close = int(f['T1'][i]) + 3600
        j = bisect.bisect_right(f['starts'], close) - 1
        reg = f['iv'][j][2] if j >= 0 and f['iv'][j][0] <= close < f['iv'][j][1] else 'вне'
        return dict(volreg=f['volreg'][i], rsi48=f['rsi48'][i], rsi6=f['rsi6'][i], rsi14=f['rsi14'][i],
                    btc24=f['btc24'][i], ret24=f['ret24'][i], rng=f['rng'][i], reg=reg, close=close,
                    hour=DT.datetime.fromtimestamp(ts, DT.UTC).hour)

    for lg, name in ((True, 'ХОДЫ ВВЕРХ'), (False, 'ХОДЫ ВНИЗ')):
        mv = [m for m in MOVES if m['long'] == lg]
        if not mv:
            continue
        caught = [m for m in mv if m['caught']]
        print('    %s: крупных %d, средний размер %.1f%% | поймано %.1f%%, вход в 6ч от начала %.1f%%, ср R поймавших %+.3f'
              % (name, len(mv), 100 * np.mean([m['size'] for m in mv]), 100 * len(caught) / len(mv),
                 100 * sum(1 for m in mv if m['near']) / len(mv), np.nanmean([m['R'] for m in caught]) if caught else float('nan')), flush=True)
        yrs = collections.defaultdict(lambda: [0, 0])
        for m in mv:
            y = DT.datetime.fromtimestamp(m['start'], DT.UTC).year
            yrs[y][0] += 1; yrs[y][1] += int(m['caught'])
        print('      по годам: %s' % '  '.join('%d %.0f%%' % (y, 100 * v[1] / v[0]) for y, v in sorted(yrs.items())), flush=True)
        groups = {'пропущенные': [feats_at(m['coin'], m['start']) for m in mv if not m['caught']],
                  'пойманные': [feats_at(m['coin'], m['start']) for m in caught],
                  'обычные часы': [feats_at(s, int(FEAT[s]['T1'][i]) + 3601) for s, i in BASE_H]}
        groups = {k: [x for x in v if x] for k, v in groups.items()}
        print('      признак за час до хода       пропущенные          пойманные            обычные часы', flush=True)
        for fn in ('volreg', 'rsi48', 'rsi6', 'rsi14', 'btc24', 'ret24', 'rng'):
            cells = []
            for k in ('пропущенные', 'пойманные', 'обычные часы'):
                v = np.array([x[fn] for x in groups[k] if np.isfinite(x[fn])])
                cells.append('%7.3g [%6.3g..%6.3g]' % (np.median(v), np.quantile(v, .25), np.quantile(v, .75)) if len(v) else '     -')
            print('        %-8s                  %s' % (fn, ' '.join(cells)), flush=True)
        for k in ('пропущенные', 'пойманные'):
            cnt = collections.Counter(x['reg'] for x in groups[k])
            tot = sum(cnt.values())
            print('        режим %-12s %s' % (k, ', '.join('%s %.0f%%' % (r, 100 * n / tot) for r, n in cnt.most_common())), flush=True)
        if lg:
            miss = [m for m in mv if not m['caught']]
            b = [roll_b.get(feats_at(m['coin'], m['start'])['close'], 0) for m in miss if feats_at(m['coin'], m['start'])]
            b = np.array(b)
            print('        ширина в час начала пропущенного хода: 0 монет %.0f%%, 1-3 монеты %.0f%%, 4+ %.0f%%'
                  % (100 * np.mean(b == 0), 100 * np.mean((b >= 1) & (b <= 3)), 100 * np.mean(b >= 4)), flush=True)
            soft = [m for m in miss if (lambda f: f and f['rsi48'] >= 50 and f['rsi6'] <= 45)(feats_at(m['coin'], m['start']))]
            print('        из пропущенных прошли бы мягкие пороги (rsi48>=50, rsi6<=45): %.0f%% (%d из %d)'
                  % (100 * len(soft) / max(1, len(miss)), len(soft), len(miss)), flush=True)
