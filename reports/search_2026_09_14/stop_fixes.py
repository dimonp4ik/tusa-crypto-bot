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


def exit_full(Md, Mu, on, cn, sl, tp, H):
    js = int(np.searchsorted(Md[:H], sl, side='left'))
    jt = int(np.searchsorted(Mu[:H], tp, side='left'))
    if js < H and js <= jt:
        return js, min(-sl, on[js]), js
    if jt < H:
        return jt, tp, js
    return H - 1, cn[H - 1], js


def book_p(sig, p=None, be=False):
    """p: take half off at p*TP (None = no partial). be: after the partial, the rest's stop moves to entry."""
    busy, out = {}, []
    for close, prio, key, s, e, atr, fav, adv, Mu, Md, on, cn, tt, slip, vr, hour in sig:
        if busy.get(s, 0) > close:
            continue
        sl, tp, hh = GEO.get(key, (3.0, 1.0, HOLDS.get(key, 48)))
        H = hh * 4
        sf = sl * atr / e
        if sf > 0.10:
            continue
        jj, x, js = exit_full(Md, Mu, on, cn, sl, tp, H)
        xx, end_j = x, jj
        if p is not None:
            jp = int(np.searchsorted(Mu[:H], p * tp, side='left'))
            if jp < H and jp < js:                      # partial fills strictly before any stop
                x1, j1 = p * tp, jp
                if be:
                    rest = adv[jp + 1:H] >= 0.0
                    kb = jp + 1 + int(np.argmax(rest)) if rest.any() else H
                    jt = int(np.searchsorted(Mu[:H], tp, side='left'))
                    stops = []
                    if js < H:
                        stops.append((js, min(-sl, on[js])))
                    if kb < H:
                        stops.append((kb, min(0.0, on[kb])))
                    if stops:
                        stop_j, stop_x = min(stops)
                    else:
                        stop_j, stop_x = H, None
                    if stop_j < H and stop_j <= jt:
                        x2, j2 = stop_x, stop_j
                    elif jt < H:
                        x2, j2 = tp, jt
                    else:
                        x2, j2 = cn[H - 1], H - 1
                else:
                    x2, j2 = x, jj
                xx, end_j = 0.5 * x1 + 0.5 * x2, max(j1, j2)
        end = int(tt[end_j]) + 900
        out.append((close, end, (xx * atr / e - slip - FEE) / sf, sf, s, 1.0, key, vr, hour))
        busy[s] = end
    return out


def p6(rows, wfun=None):
    return [(a, b, R, sf, s, (wfun(vr, hr) if wfun else 1.0)) for a, b, R, sf, s, w, k, vr, hr in rows]


def fix(r6):
    r = sim_w.simulate(r6, 0.014)
    return r['eq'], abs(L.dd_of(r['curve']))


def pyr(r6):
    return [sim_w.simulate([x for x in r6 if YT[y] <= x[0] < YT[y + 1]], 0.014)['eq'] for y in range(2022, 2027)]


BOOKS = (('КНИГА 1 (без XLM, AAVE)', [c for c in SP.COINS if c not in ('BILLUSDT', 'XLMUSDT', 'AAVEUSDT')]),
         ('КНИГА 2 (все 15 монет)', [c for c in SP.COINS if c != 'BILLUSDT']))
BASE_LBL = 'без частичной (сейчас)'
PARTS = [(BASE_LBL, None, False), ('половина на 50% пути', 0.5, False),
         ('половина на 75% пути', 0.75, False), ('половина на 50% + безубыток', 0.5, True),
         ('половина на 75% + безубыток', 0.75, True)]
for bi, (book_name, coins) in enumerate(BOOKS):
    sig = make_sig_raw(coins)
    print('', flush=True)
    print('  ===== %s: ЧАСТИЧНАЯ ФИКСАЦИЯ =====' % book_name, flush=True)
    RES = {}
    base = None
    for lbl, p, be in PARTS:
        r6 = p6(book_p(sig, p, be))
        RES[lbl] = r6
        eq, dd = fix(r6)
        k, m = sim_w.money_at_dd(r6, 0.12)
        v = np.array([x[2] for x in r6])
        py = pyr(r6)
        if base is None:
            base = py
        print('    %-30s | ВР %4.1f%% ср %+.4f | 1.4%%: $%6.0f %4.1f%% | DD12%%: $%6.0f | лучше по годам %d/5 (%s)'
              % (lbl, 100 * np.mean(v > 0), v.mean(), eq, 100 * dd, m,
                 sum(1 for a, b in zip(base, py) if b > a), ' '.join('%+.0f' % (b - a) for a, b in zip(base, py))), flush=True)
    if bi == 0:
        print('    -- слепой выбор (мера: $ при 1.4%): 4 года выбор, 5-й замер --', flush=True)
        wins = seen = 0
        for y in range(2022, 2027):
            lo, hi = YT[y], YT[y + 1]
            best, bk = -1, None
            for lbl, r6 in RES.items():
                mm = sim_w.simulate([x for x in r6 if not (lo <= x[0] < hi)], 0.014)['eq']
                if mm > best:
                    best, bk = mm, lbl
            mp = sim_w.simulate([x for x in RES[bk] if lo <= x[0] < hi], 0.014)['eq']
            mb = sim_w.simulate([x for x in RES[BASE_LBL] if lo <= x[0] < hi], 0.014)['eq']
            seen += 1
            wins += int(mp > mb)
            print('      %d выбрано «%s» | $%.0f против $%.0f %s'
                  % (y, bk, mp, mb, 'лучше' if mp > mb else ('так же' if mp == mb else 'хуже')), flush=True)
        print('      лучше нынешнего в %d из %d лет' % (wins, seen), flush=True)

    print('  ===== %s: РАЗМЕР В ТИХОМ РЫНКЕ И В 12-15 UTC =====' % book_name, flush=True)
    rows = book_p(sig, None, False)
    vrs = np.array([x[7] for x in rows if np.isfinite(x[7])])
    q1 = float(np.quantile(vrs, 0.2))
    bpy = pyr(p6(rows))
    WF = [('сейчас', None),
          ('тихий рынок (volreg Q1) x0.5', lambda vr, hr: 0.5 if np.isfinite(vr) and vr <= q1 else 1.0),
          ('часы 12-15 x0.5', lambda vr, hr: 0.5 if 12 <= hr <= 15 else 1.0),
          ('оба x0.5', lambda vr, hr: 0.5 if (np.isfinite(vr) and vr <= q1) or 12 <= hr <= 15 else 1.0)]
    for lbl, wf in WF:
        r6 = p6(rows, wf)
        eq, dd = fix(r6)
        k, m = sim_w.money_at_dd(r6, 0.12)
        py = pyr(r6)
        print('    %-30s | 1.4%%: $%6.0f %4.1f%% | DD12%%: $%6.0f | лучше по годам %d/5 (%s)'
              % (lbl, eq, 100 * dd, m, sum(1 for a, b in zip(bpy, py) if b > a),
                 ' '.join('%+.0f' % (b - a) for a, b in zip(bpy, py))), flush=True)
