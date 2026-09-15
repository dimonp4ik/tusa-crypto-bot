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
BOOKS = (('КНИГА 1 (без XLM, AAVE)', [c for c in SP.COINS if c not in ('BILLUSDT', 'XLMUSDT', 'AAVEUSDT')]),
         ('КНИГА 2 (все 15 монет)', [c for c in SP.COINS if c != 'BILLUSDT']))


def fix2(rows):
    r = sim_w.simulate(rows, 0.014)
    return r['eq'], abs(L.dd_of(r['curve']))


def group(n):
    return '1' if n == 1 else ('2-3' if n <= 3 else ('4-6' if n <= 6 else '7+'))



SHORT_KEYS = {'r%d' % k for k, r in enumerate(RULES) if r.get('side', 'LONG') == 'SHORT'}


def book_t(sig, keys, wide, tp_w, hold_w):
    """wide: set of (close, coin) that get the wide exit."""
    busy, out = {}, []
    for close, prio, key, s, e, atr, fav, adv, Mu, Md, on, cn, tt, slip, vr, hour in sig:
        if key not in keys or busy.get(s, 0) > close:
            continue
        sl, tp, hh = GEO.get(key, (3.0, 1.0, HOLDS.get(key, 48)))
        if (close, s) in wide and key not in SHORT_KEYS:
            tp, hh = tp_w, hold_w
        H = min(hh * 4, len(Mu))
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
        busy[s] = end
    out.sort()
    return out


KS = (5, 7, 10)
EXITS = ((1.25, 48), (1.5, 48), (2.0, 48), (1.5, 72), (2.0, 72))

for bi, (book_name, coins) in enumerate(BOOKS):
    sig = make_sig_raw(coins)
    rows, _ = book(sig, BASE_KEYS)
    cnt = collections.Counter(x[0] for x in rows)
    long_rows = [(x[0], x[4]) for x in rows]
    key_at = {(x[0], x[3]): x[2] for x in sig}
    long_rows = [k for k in long_rows if key_at.get(k) not in SHORT_KEYS]
    beq, bdd = fix2(rows)
    bm = sim_w.money_at_dd(rows, 0.12)[1]
    bpy = pyr(rows)
    print('', flush=True)
    print('  ===== %s: система $%.0f просадка %.1f%% DD12 $%.0f, лонгов %d =====' % (book_name, beq, 100 * bdd, bm, len(long_rows)), flush=True)
    RES = {'система (сейчас)': rows}
    for K in KS:
        wide = {k for k in long_rows if cnt[k[0]] >= K}
        rest = [k for k in long_rows if cnt[k[0]] < K]
        for tp_w, hold_w in EXITS:
            rr = book_t(sig, BASE_KEYS, wide, tp_w, hold_w)
            eq, dd = fix2(rr)
            m = sim_w.money_at_dd(rr, 0.12)[1]
            py = pyr(rr)
            v = np.array([x[2] for x in rr if (x[0], x[4]) in wide])
            lbl = 'n>=%d тейк %.2f %dч' % (K, tp_w, hold_w)
            RES[lbl] = rr
            ce, cm = [], []
            for seed in range(5):
                rng = np.random.default_rng(seed)
                pick = {rest[i] for i in rng.choice(len(rest), size=min(len(wide), len(rest)), replace=False)}
                cr = book_t(sig, BASE_KEYS, pick, tp_w, hold_w)
                ce.append(fix2(cr)[0]); cm.append(sim_w.money_at_dd(cr, 0.12)[1])
            print('    %-24s пачечных %4d ВР %5.1f%% ср %+.3f | $%6.0f %4.1f%% | DD12 $%6.0f %s | лучше по годам %d/5 (%s) | КОНТРОЛЬ случайные $%.0f DD12 $%.0f'
                  % (lbl, len(v), 100 * np.mean(v > 0) if len(v) else 0, v.mean() if len(v) else 0, eq, 100 * dd, m,
                     '*' if (eq > beq and m > bm) else ' ', sum(1 for a, b in zip(bpy, py) if b > a),
                     ' '.join('%+.0f' % (b - a) for a, b in zip(bpy, py)), np.median(ce), np.median(cm)), flush=True)
    if bi == 0:
        print('    -- слепой выбор (мера: $ при 1.4%): 4 года выбор, 5-й замер --', flush=True)
        wins = 0
        for y in range(2022, 2027):
            lo, hi = YT[y], YT[y + 1]
            best, bk = -1, None
            for lbl, rr in RES.items():
                mm = sim_w.simulate([x for x in rr if not (lo <= x[0] < hi)], 0.014)['eq']
                if mm > best:
                    best, bk = mm, lbl
            mp = sim_w.simulate([x for x in RES[bk] if lo <= x[0] < hi], 0.014)['eq']
            mb = sim_w.simulate([x for x in rows if lo <= x[0] < hi], 0.014)['eq']
            wins += int(mp > mb)
            print('      %d выбрано «%s» | $%.0f против $%.0f %s'
                  % (y, bk, mp, mb, 'лучше' if mp > mb else ('так же' if mp == mb else 'хуже')), flush=True)
        print('      лучше нынешней системы в %d из 5 лет' % wins, flush=True)
