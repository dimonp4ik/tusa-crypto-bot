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
EXTRA = [('o30', 3, 1800, 0), ('o45', 4, 2700, 0), ('o1h15', 5, 3600, 900), ('o1h30', 6, 3600, 1800),
         ('o1h45', 7, 3600, 2700)]
MC, LL = sim_w.MC, sim_w.L


def book_k(sig, keys):
    busy, out = {}, []
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
        out.append((close, end, (x * atr / e - slip - FEE) / sf, sf, s, 1.0, key not in BASE_KEYS))
        busy[s] = end
    out.sort()
    return out


def sim_g(trades, target, gate=None):
    margin_frac = target / (sim_w.LEV * sim_w.REF)
    eq = peak = sim_w.DEPOSIT
    open_pos = []
    day, day_start, day_paused = None, sim_w.DEPOSIT, False
    paused, paused_at = False, None
    curve = []
    gated = 0
    for a, b, R, sf, s, w, extra in trades:
        while open_pos and open_pos[0][0] <= a:
            t_close, m, money_per_R, rr = open_pos.pop(0)
            eq += money_per_R * rr
            peak = max(peak, eq)
            curve.append((t_close, eq))
        d = LL.day_of(a)
        if d != day:
            day, day_start, day_paused = d, eq, False
        if not paused and eq <= peak * (1 - sim_w.MAX_DD):
            paused, paused_at = True, a
        if not day_paused and eq <= day_start * (1 - sim_w.MAX_DAILY):
            day_paused = True
        if paused or day_paused:
            continue
        if extra and gate is not None:
            kind, v = gate
            if (kind == 'dd' and eq <= peak * (1 - v)) or (kind == 'open' and len(open_pos) >= v):
                gated += 1
                continue
        margin = margin_frac * eq
        if sf > sim_w.REF:
            margin *= sim_w.REF / sf
        margin *= w
        used = sum(p[1] for p in open_pos)
        if used + margin > eq * sim_w.USABLE:
            continue
        open_pos.append((b, margin, margin * sim_w.LEV * sf, R))
        open_pos.sort(key=lambda p: p[0])
    for t_close, m, money_per_R, rr in open_pos:
        eq += money_per_R * rr
        curve.append((t_close, eq))
    return dict(eq=eq, curve=curve, paused_at=paused_at, gated=gated)


def dd12_g(trades, gate, want=0.12):
    lo, hi = 0.0005, 0.030
    for _ in range(20):
        mid = (lo + hi) / 2
        r = sim_g(trades, mid, gate)
        if bool(r['paused_at']) or abs(LL.dd_of(r['curve'])) > want:
            hi = mid
        else:
            lo = mid
    return sim_g(trades, lo, gate)['eq']


def yrs_g(rows, gate):
    return [sim_g([x for x in rows if YT[y] <= x[0] < YT[y + 1]], 0.014, gate)['eq'] for y in range(2022, 2027)]


SOURCES = (('45м', {'o45'}), ('1ч сдвиг 30м', {'o1h30'}), ('1ч сдвиги 15/30/45', {'o1h15', 'o1h30', 'o1h45'}),
           ('30м+45м+1ч сдвиги', {'o30', 'o45', 'o1h15', 'o1h30', 'o1h45'}))
GATES = (('без ограничителя', None), ('dd 3%', ('dd', 0.03)), ('dd 5%', ('dd', 0.05)), ('dd 7%', ('dd', 0.07)),
         ('open 3', ('open', 3)), ('open 5', ('open', 5)), ('open 8', ('open', 8)))
BOOKS = (('КНИГА 1 (без XLM, AAVE)', [c for c in SP.COINS if c not in ('BILLUSDT', 'XLMUSDT', 'AAVEUSDT')]),
         ('КНИГА 2 (все 15 монет)', [c for c in SP.COINS if c != 'BILLUSDT']))

for bi, (book_name, coins) in enumerate(BOOKS):
    sig = make_sig_raw(coins)
    for key, prio, sec, off in EXTRA:
        sig += long_entries(coins, key, prio, sec, off)
    sig.sort(key=lambda x: (x[0], x[1]))
    base_rows = book_k(sig, BASE_KEYS)
    if bi == 0:
        ref = sim_w.simulate([x[:6] for x in base_rows], 0.014)['eq']
        mine = sim_g(base_rows, 0.014)['eq']
        print('  САМОПРОВЕРКА: копия симулятора $%.2f против sim_w $%.2f %s'
              % (mine, ref, 'совпало' if abs(mine - ref) < 0.01 else 'РАСХОЖДЕНИЕ'), flush=True)
    b_eq = sim_g(base_rows, 0.014)
    base_y = yrs_g(base_rows, None)
    print('', flush=True)
    print('  ===== %s: система $%.0f просадка %.1f%% DD12 $%.0f =====' % (book_name, b_eq['eq'], 100 * abs(LL.dd_of(b_eq['curve'])), dd12_g(base_rows, None)), flush=True)
    RES = {'система (сейчас)': (base_rows, None)}
    for sname, skeys in SOURCES:
        rows = book_k(sig, BASE_KEYS | skeys)
        for gname, gate in GATES:
            r = sim_g(rows, 0.014, gate)
            py = yrs_g(rows, gate)
            lbl = '+ %s, %s' % (sname, gname)
            RES[lbl] = (rows, gate)
            print('    %-40s $%6.0f %4.1f%% | DD12 $%6.0f | отсечено доп %4d | лучше по годам %d/5 (%s)'
                  % (lbl, r['eq'], 100 * abs(LL.dd_of(r['curve'])), dd12_g(rows, gate), r['gated'],
                     sum(1 for a, b in zip(base_y, py) if b > a), ' '.join('%+.0f' % (b - a) for a, b in zip(base_y, py))), flush=True)
    if bi == 0:
        print('    -- слепой выбор (мера: $ при 1.4%): 4 года выбор, 5-й замер --', flush=True)
        wins = 0
        for y in range(2022, 2027):
            lo, hi = YT[y], YT[y + 1]
            best, bk = -1, None
            for lbl, (rows, gate) in RES.items():
                mm = sim_g([x for x in rows if not (lo <= x[0] < hi)], 0.014, gate)['eq']
                if mm > best:
                    best, bk = mm, lbl
            rows, gate = RES[bk]
            mp = sim_g([x for x in rows if lo <= x[0] < hi], 0.014, gate)['eq']
            mb = sim_g([x for x in base_rows if lo <= x[0] < hi], 0.014)['eq']
            wins += int(mp > mb)
            print('      %d выбрано «%s» | $%.0f против $%.0f %s'
                  % (y, bk, mp, mb, 'лучше' if mp > mb else ('так же' if mp == mb else 'хуже')), flush=True)
        print('      лучше нынешней системы в %d из 5 лет' % wins, flush=True)
