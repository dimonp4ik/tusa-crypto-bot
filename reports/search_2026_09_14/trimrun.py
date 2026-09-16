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



import datetime as DT
WIDE_REF = 0.060
MULTS = (1.00, 0.75, 0.50, 0.25)
YEARS = (2022, 2023, 2024, 2025, 2026)
TRIM = 'r1'


def yr(ts):
    return DT.datetime.fromtimestamp(ts, DT.UTC).year


def weigh_trim(rows, keys, mult, pick=None):
    cnt = collections.Counter(x[0] for x in rows)
    out = []
    for i, (x, k) in enumerate(zip(rows, keys)):
        w = 1.25 if cnt[x[0]] >= 10 else 1.0
        if k == 'r2_4':
            w *= 1.25
        if (pick is None and k == TRIM) or (pick is not None and i in pick):
            w *= mult
        out.append(x[:5] + (w,))
    return out


def meas(fin, refs, keep=None):
    if keep is not None:
        ix = [i for i, x in enumerate(fin) if keep(x[0])]
        fin = [fin[i] for i in ix]; refs = [refs[i] for i in ix]
    r = sim_ref(fin, 0.014, refs)
    return r['eq'], abs(L.dd_of(r['curve'])), money_at_dd_ref(fin, refs)


print('  УРЕЗАНИЕ РАЗМЕРА САМОГО СЛАБОГО ПРАВИЛА (разгон монеты вечером)', flush=True)
for book_name, coins in BOOKS3:
    H = {s: coin_hours(s) for s in coins}
    sig_all = sorted(make_sig_raw(coins) + roll_entries(coins, H, 'r2_4', 4, 2, 4, 0),
                     key=lambda x: (x[0], x[1]))
    GEO.clear(); GEO.update(BASE_GEO)
    rows, keys = book_keyed(sig_all, FULL)
    refs = [(WIDE_REF if k == 'r2_4' else sim_w.REF) for k in keys]
    idx = [i for i, k in enumerate(keys) if k == TRIM]
    print('', flush=True)
    print('  %s — сделок правила %d из %d' % (book_name, len(idx), len(rows)), flush=True)
    print('    ПО ГОДАМ (это правило): %s' % ', '.join(
        '%d: %d сд. ср%+.3f' % (y, sum(1 for i in idx if yr(rows[i][0]) == y),
                                float(np.mean([rows[i][2] for i in idx if yr(rows[i][0]) == y] or [0])))
        for y in YEARS), flush=True)
    base = weigh_trim(rows, keys, 1.0)
    e0, d0, m0 = meas(base, refs)
    print('    КОНТРОЛЬ без урезания: $%.0f просадка %.1f%% при равной просадке $%.0f' % (e0, 100 * d0, m0), flush=True)
    for mult in MULTS[1:]:
        fin = weigh_trim(rows, keys, mult)
        e, d, m = meas(fin, refs)
        ce, cm = [], []
        for seed in range(5):
            rng = np.random.default_rng(seed)
            other = [i for i in range(len(rows)) if keys[i] != TRIM]
            pick = set(rng.choice(other, size=min(len(idx), len(other)), replace=False).tolist())
            cf = weigh_trim(rows, keys, mult, pick=pick)
            x, _, z = meas(cf, refs)
            ce.append(x); cm.append(z)
        print('    x%.2f: $%6.0f (%+5.1f%%) просадка %.1f%% | при равной просадке $%6.0f (%+5.1f%%) | КОНТРОЛЬ случайные $%.0f (%+.1f%%) DD12 $%.0f (%+.1f%%)'
              % (mult, e, 100 * (e / e0 - 1), 100 * d, m, 100 * (m / m0 - 1),
                 np.median(ce), 100 * (np.median(ce) / e0 - 1), np.median(cm), 100 * (np.median(cm) / m0 - 1)), flush=True)

    print('    СЛЕПОЙ ВЫБОР множителя по четырём годам, замер на пятом', flush=True)
    wins = 0
    for y in YEARS:
        sc = []
        for mult in MULTS:
            fin = weigh_trim(rows, keys, mult)
            e, _, m = meas(fin, refs, keep=lambda t: yr(t) != y)
            sc.append((min(e / e0, m / m0), mult) if mult == 1.0 else (0, mult))
        sc = []
        eb, _, mb = meas(base, refs, keep=lambda t: yr(t) != y)
        for mult in MULTS:
            fin = weigh_trim(rows, keys, mult)
            e, _, m = meas(fin, refs, keep=lambda t: yr(t) != y)
            sc.append((min(e / eb, m / mb), mult))
        sc.sort(reverse=True)
        pick = sc[0][1]
        fin = weigh_trim(rows, keys, pick)
        e, _, m = meas(fin, refs, keep=lambda t: yr(t) == y)
        eb, _, mb = meas(base, refs, keep=lambda t: yr(t) == y)
        ok = (e > eb and m > mb)
        wins += ok
        print('      %d: выбрано x%.2f | на этом году %+5.1f%% денег, %+5.1f%% при равной просадке %s'
              % (y, pick, 100 * (e / eb - 1), 100 * (m / mb - 1), 'лучше' if ok else ''), flush=True)
    print('      лет, где выбранный вслепую множитель лучше базы по ОБЕИМ мерам: %d из %d' % (wins, len(YEARS)), flush=True)
