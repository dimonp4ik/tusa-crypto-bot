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



FULL = BASE_KEYS | {'r2_4'}
SHORT_KEYS = ['r%d' % k for k, r in enumerate(RULES) if r.get('side', 'LONG') == 'SHORT']
BASE_GEO = dict(GEO)
PULL_KEYS = {'o1h', 'o2h'}


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
        out.append(((close, end, (x * atr / e - slip - FEE) / sf, sf, s, 1.0), key))
        busy[s] = end
    out.sort(key=lambda z: z[0])
    return [z[0] for z in out], [z[1] for z in out]


def weigh_src(rows, keys, want, mult):
    cnt = collections.Counter(x[0] for x in rows)
    out = []
    for x, k in zip(rows, keys):
        w = 1.25 if cnt[x[0]] >= 10 else 1.0
        if k in want:
            w *= mult
        out.append(x[:5] + (w,))
    return out


def weigh_rand(rows, n_target, mult, seed):
    rng = np.random.default_rng(seed)
    idx = set(rng.choice(len(rows), size=min(n_target, len(rows)), replace=False).tolist())
    cnt = collections.Counter(x[0] for x in rows)
    return [x[:5] + ((1.25 if cnt[x[0]] >= 10 else 1.0) * (mult if i in idx else 1.0),) for i, x in enumerate(rows)]



REFS = ((0.0394, 'как сейчас 0.0394'), (0.045, 'ширине 0.045'), (0.050, 'ширине 0.050'),
        (0.060, 'ширине 0.060'), (None, 'ширине без stop_ref'))


def sim_ref(trades, target, refs):
    """sim_w.simulate with a per-trade stop_ref (refs: list of thresholds, None = no shrink)."""
    eq = peak = sim_w.DEPOSIT
    open_pos = []
    day, day_start, day_paused, paused, paused_at = None, sim_w.DEPOSIT, False, False, None
    curve = []
    for (a, b, R, sf, s, w), ref in zip(trades, refs):
        while open_pos and open_pos[0][0] <= a:
            t_close, m, mpr, rr = open_pos.pop(0)
            eq += mpr * rr; peak = max(peak, eq); curve.append((t_close, eq))
        d = L.day_of(a)
        if d != day:
            day, day_start, day_paused = d, eq, False
        if not paused and eq <= peak * (1 - sim_w.MAX_DD):
            paused, paused_at = True, a
        if not day_paused and eq <= day_start * (1 - sim_w.MAX_DAILY):
            day_paused = True
        if paused or day_paused:
            continue
        margin = (target / (sim_w.LEV * sim_w.REF)) * eq
        if ref is not None and sf > ref:
            margin *= ref / sf
        margin *= w
        if sum(p[1] for p in open_pos) + margin > eq * sim_w.USABLE:
            continue
        open_pos.append((b, margin, margin * sim_w.LEV * sf, R))
        open_pos.sort(key=lambda p: p[0])
    for t_close, m, mpr, rr in open_pos:
        eq += mpr * rr; curve.append((t_close, eq))
    curve.sort()
    return dict(eq=eq, curve=curve, paused_at=paused_at)


def money_at_dd_ref(rows, refs, want=0.12):
    lo, hi = 0.0005, 0.030
    for _ in range(22):
        mid = (lo + hi) / 2
        r = sim_ref(rows, mid, refs)
        if bool(r['paused_at']) or abs(L.dd_of(r['curve'])) > want:
            hi = mid
        else:
            lo = mid
    return sim_ref(rows, lo, refs)['eq']



import bisect
WIDE_REF = 0.060
SHORT_IDX = {k: r for k, r in enumerate(RULES) if r.get('side', 'LONG') == 'SHORT'}
POP = next(k for k, r in SHORT_IDX.items() if r.get('name') == 'short_pop_in_downtrend')
MOR = next(k for k, r in SHORT_IDX.items() if r.get('name') == 'short_btc_up_morning')
POP_A = (65.0, 68.0, 71.02, 74.0, 77.0)
POP_B = (30.0, 33.0, 35.22, 38.0, 41.0)
MOR_C = (0.010, 0.015, 0.01701, 0.020, 0.025)
MOR_H = ((7, 11), (8, 11), (8, 12), (9, 11))


def short_candidates(coins):
    """Every hour in the SHORT regime that could pass the loosest thresholds, with its feature values."""
    out = []
    for s in coins:
        c = SP.CTX[s]
        T1, F, iv, starts = c['T1'], c['F'], c['iv'], c['starts']
        t15, a15, pos = c['t15'], c['a15'], c['pos']
        slip = SLIP.get(s, 0.0003)
        for i in range(PB.MIN_HOURS, len(T1)):
            atr = F['atr'][i]
            if not np.isfinite(atr) or atr <= 0:
                continue
            close = int(T1[i]) + 3600
            j = bisect.bisect_right(starts, close) - 1
            if j < 0 or not (iv[j][0] <= close < iv[j][1]) or iv[j][2] != 'SHORT':
                continue
            r2, r14, b24 = float(F['rsi2'][i]), float(F['rsi14'][i]), float(F['btc24'][i])
            hour = int((close // 3600) % 24)
            pop_ok = r2 >= 65.0 and r14 <= 41.0
            mor_ok = b24 >= 0.010 and 7 <= hour <= 12
            if not (pop_ok or mor_ok):
                continue
            k = pos.get(close)
            if k is None or k + MAXH > len(t15):
                continue
            o, h, l, cc = (a15[k:k + MAXH, x] for x in range(4))
            e = o[0] * (1 - slip)
            fav, adv = (e - l) / atr, (h - e) / atr
            row = (close, 0, None, s, e, atr, fav, adv, np.maximum.accumulate(fav), np.maximum.accumulate(adv),
                   (e - o) / atr, (e - cc) / atr, t15[k:k + MAXH], slip, float('nan'), hour)
            out.append((row, r2, r14, b24, hour))
    return out


def short_entries(cands, a, b, cthr, hlo, hhi):
    out = []
    for row, r2, r14, b24, hour in cands:
        if r2 >= a and r14 <= b:
            out.append(row[:2] + ('r%d' % POP,) + row[3:])
        elif b24 >= cthr and hlo <= hour <= hhi:
            out.append(row[:2] + ('r%d' % MOR,) + row[3:])
    return out


for book_name, coins in BOOKS3:
    H = {s: coin_hours(s) for s in coins}
    full_sig = make_sig_raw(coins)
    longs = [x for x in full_sig if x[2] not in ('r%d' % POP, 'r%d' % MOR)]
    extra = roll_entries(coins, H, 'r2_4', 4, 2, 4, 0)
    cands = short_candidates(coins)
    GEO.clear(); GEO.update(BASE_GEO)
    print('', flush=True)
    print('  ===== %s: кандидатов шорта %d =====' % (book_name, len(cands)), flush=True)

    def run(a, b, cthr, hlo, hhi):
        sig = sorted(longs + extra + short_entries(cands, a, b, cthr, hlo, hhi), key=lambda x: (x[0], x[1]))
        rf, kf = book_keyed(sig, FULL)
        fin = weigh_src(rf, kf, {'r2_4'}, 1.25)
        refs = [(WIDE_REF if k == 'r2_4' else sim_w.REF) for k in kf]
        r = sim_ref(fin, 0.014, refs)
        return r['eq'], abs(L.dd_of(r['curve'])), money_at_dd_ref(fin, refs), len(fin)

    beq, bdd, bm, bn = run(71.02, 35.22, 0.01701, 8, 11)
    print('    живые пороги: $%.0f %.1f%% DD12 $%.0f (%d сделок) — контроль сборки' % (beq, 100 * bdd, bm, bn), flush=True)
    print('    ПРАВИЛО short_pop (rsi2 >= A и rsi14 <= B), утреннее правило на живых порогах:', flush=True)
    print('      A\\B    ' + ''.join('%-22s' % ('<=%.2f' % b) for b in POP_B), flush=True)
    for a in POP_A:
        cells = []
        for b in POP_B:
            eq, dd, m, n = run(a, b, 0.01701, 8, 11)
            cells.append('%s$%6.0f %4.1f%% $%6.0f ' % ('*' if (eq > beq and m > bm) else (' ' if (a, b) != (71.02, 35.22) else '='), eq, 100 * dd, m))
        print('      >=%-5.2f %s' % (a, ''.join(cells)), flush=True)
    print('    ПРАВИЛО short_btc_up_morning (btc24 >= C, часы), short_pop на живых порогах:', flush=True)
    for hlo, hhi in MOR_H:
        cells = []
        for cthr in MOR_C:
            eq, dd, m, n = run(71.02, 35.22, cthr, hlo, hhi)
            cells.append('%s$%6.0f %4.1f%% $%6.0f ' % ('*' if (eq > beq and m > bm) else (' ' if (cthr, hlo, hhi) != (0.01701, 8, 11) else '='), eq, 100 * dd, m))
        print('      часы %d-%d  %s' % (hlo, hhi, ''.join(cells)), flush=True)
    print('      (столбцы C: %s)' % ', '.join('%.3f' % c for c in MOR_C), flush=True)
