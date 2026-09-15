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


def entry_row(s, close, key, prio, atr, jj, c, slip):
    t15, a15 = c['t15'], c['a15']
    if jj + MAXH > len(t15):
        return None
    o, h, l, cc = (a15[jj:jj + MAXH, x] for x in range(4))
    e = o[0] * (1 + slip)
    fav = (h - e) / atr
    adv = (e - l) / atr
    return (int(t15[jj]), prio, key, s, e, atr, fav, adv, np.maximum.accumulate(fav), np.maximum.accumulate(adv),
            (o - e) / atr, (cc - e) / atr, t15[jj:jj + MAXH], slip, float('nan'), 0)


def pull_confirm(coins, W, delays=None, seed=0):
    rng = np.random.default_rng(seed)
    out, dl, nsig = [], [], 0
    for s in coins:
        c = SP.CTX[s]
        t15, a15, pos = c['t15'], c['a15'], c['pos']
        slip = SLIP.get(s, 0.0003)
        for key, prio, sec in (('o1h', 1, 3600), ('o2h', 2, 7200)):
            for close, atr in pull_signals(s, sec):
                j = pos.get(close)
                if j is None or j < 4 or atr <= 0:
                    continue
                nsig += 1
                if delays is None:
                    k = None
                    for kk in range(W):
                        b = j + kk
                        if b >= len(t15):
                            break
                        if a15[b, 3] > a15[b - 4:b, 1].max():
                            k = kk
                            break
                    if k is None:
                        continue
                    dl.append(k)
                else:
                    if rng.random() >= len(delays) / max(1, delays_n[0]):
                        continue
                    k = int(rng.choice(delays))
                r = entry_row(s, close, key, prio, atr, j + k + 1, c, slip)
                if r is not None:
                    out.append(r)
    return out, dl, nsig


def fix2(rows):
    r = sim_w.simulate(rows, 0.014)
    return r['eq'], abs(L.dd_of(r['curve']))


BOOKS = (('КНИГА 1 (без XLM, AAVE)', [c for c in SP.COINS if c not in ('BILLUSDT', 'XLMUSDT', 'AAVEUSDT')]),
         ('КНИГА 2 (все 15 монет)', [c for c in SP.COINS if c != 'BILLUSDT']))
delays_n = [1]

for bi, (book_name, coins) in enumerate(BOOKS):
    raw = make_sig_raw(coins)
    bank = [x for x in raw if x[2] not in ('o1h', 'o2h')]
    base_rows, _ = book(sorted(raw, key=lambda x: (x[0], x[1])), BASE_KEYS)
    beq, bdd = fix2(base_rows)
    bpy = pyr(base_rows)
    print('', flush=True)
    print('  ===== %s: система $%.0f просадка %.1f%% DD12 $%.0f =====' % (book_name, beq, 100 * bdd, sim_w.money_at_dd(base_rows, 0.12)[1]), flush=True)
    RES = {'система (сейчас)': base_rows}

    def show(lbl, pull):
        sig = sorted(bank + pull, key=lambda x: (x[0], x[1]))
        rows, cnt = book(sig, BASE_KEYS)
        eq, dd = fix2(rows)
        m = sim_w.money_at_dd(rows, 0.12)[1]
        py = pyr(rows)
        pr = [x[2] for x, kk in zip(rows, [None] * len(rows))]
        print('    %-36s откатов в книге %4d | $%6.0f %4.1f%% | DD12 $%6.0f | лучше по годам %d/5 (%s)'
              % (lbl, cnt['o1h'] + cnt['o2h'], eq, 100 * dd, m, sum(1 for a, b in zip(bpy, py) if b > a),
                 ' '.join('%+.0f' % (b - a) for a, b in zip(bpy, py))), flush=True)
        return rows

    for W in (4, 8, 16, 32):
        pull, dl, nsig = pull_confirm(coins, W)
        v = np.array([x[2] for x in book(sorted(pull, key=lambda x: (x[0], x[1])), {'o1h', 'o2h'})[0]])
        print('    окно %2d баров (%.0fч): подтвердилось %d из %d сигналов (%.0f%%), медиана задержки %.0f баров, откат сам по себе ВР %.1f%% ср %+.4f'
              % (W, W / 4, len(dl), nsig, 100 * len(dl) / max(1, nsig), np.median(dl) if dl else 0, 100 * np.mean(v > 0), v.mean()), flush=True)
        RES['подтверждение %dч' % (W // 4)] = show('  подтверждение за %.0fч' % (W / 4), pull)
        delays_n[0] = nsig
        eqs, ms = [], []
        for seed in range(3):
            pc, _, _ = pull_confirm(coins, W, delays=dl, seed=seed)
            sig = sorted(bank + pc, key=lambda x: (x[0], x[1]))
            rows, _ = book(sig, BASE_KEYS)
            eqs.append(fix2(rows)[0]); ms.append(sim_w.money_at_dd(rows, 0.12)[1])
        print('      КОНТРОЛЬ случайная задержка той же доли и длины: $%.0f, DD12 $%.0f (медиана 3)' % (np.median(eqs), np.median(ms)), flush=True)
    if bi == 0:
        print('    -- слепой выбор (мера: $ при 1.4%): 4 года выбор, 5-й замер --', flush=True)
        wins = 0
        for y in range(2022, 2027):
            lo, hi = YT[y], YT[y + 1]
            best, bk = -1, None
            for lbl, rows in RES.items():
                mm = sim_w.simulate([x for x in rows if not (lo <= x[0] < hi)], 0.014)['eq']
                if mm > best:
                    best, bk = mm, lbl
            mp = sim_w.simulate([x for x in RES[bk] if lo <= x[0] < hi], 0.014)['eq']
            mb = sim_w.simulate([x for x in base_rows if lo <= x[0] < hi], 0.014)['eq']
            wins += int(mp > mb)
            print('      %d выбрано «%s» | $%.0f против $%.0f %s'
                  % (y, bk, mp, mb, 'лучше' if mp > mb else ('так же' if mp == mb else 'хуже')), flush=True)
        print('      лучше нынешней системы в %d из 5 лет' % wins, flush=True)
