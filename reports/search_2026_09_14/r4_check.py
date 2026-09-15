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


def make_sig(coins):
    sig = []
    for s in coins:
        c = SP.CTX[s]
        t15, a15, pos = c['t15'], c['a15'], c['pos']
        slip = SLIP.get(s, 0.0003)
        items = [(t, 0, 'r%d' % v[0], RULES[v[0]].get('side', 'LONG') == 'LONG', v[2])
                 for t, v in SP.bank_signals(RULES, s, 50).items()]
        items += [(t, 1, 'о1ч', True, a) for t, a in pull_signals(s, 3600)]
        items += [(t, 2, 'о2ч', True, a) for t, a in pull_signals(s, 7200)]
        for close, prio, key, lg, atr in items:
            j = pos.get(close)
            if j is None or j + MAXH > len(t15) or not np.isfinite(atr) or atr <= 0:
                continue
            o, h, l, cc = (a15[j:j + MAXH, x] for x in range(4))
            e = o[0] * (1 + slip) if lg else o[0] * (1 - slip)
            fav = (h - e) / atr if lg else (e - l) / atr
            adv = (e - l) / atr if lg else (h - e) / atr
            sig.append((close, prio, key, s, e, atr, np.maximum.accumulate(fav), np.maximum.accumulate(adv),
                        (o - e) / atr if lg else (e - o) / atr, (cc - e) / atr if lg else (e - cc) / atr,
                        t15[j:j + MAXH], slip))
    sig.sort(key=lambda x: (x[0], x[1]))
    return sig


def book(sig, geo):
    busy, out = {}, []
    for close, prio, key, s, e, atr, Mu, Md, on, cn, tt, slip in sig:
        if busy.get(s, 0) > close:
            continue
        sl, tp, hh = geo.get(key, (3.0, 1.0, BASE_HOLD.get(key, 48)))
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
        out.append((close, end, (x * atr / e - slip - FEE) / sf, sf, s, 1.0, key))
        busy[s] = end
    return out


def plain(rows):
    return [x[:6] for x in rows]


def m_dd(rows):
    return sim_w.money_at_dd(plain(rows), 0.12)[1] if len(rows) >= 80 else float('nan')


def m_fix(rows):
    r = sim_w.simulate(plain(rows), 0.014)
    return r['eq'], abs(L.dd_of(r['curve']))


def per_year(rows):
    return [sim_w.simulate([x for x in plain(rows) if YT[y] <= x[0] < YT[y + 1]], 0.014)['eq'] for y in range(2022, 2027)]


BOOKS = (('КНИГА 1 (без XLM, AAVE)', [c for c in SP.COINS if c not in ('BILLUSDT', 'XLMUSDT', 'AAVEUSDT')]),
         ('КНИГА 2 (все 15 монет)', [c for c in SP.COINS if c != 'BILLUSDT']))
CANDS = [BASE_G, (2.0, 1.5, 48), (2.0, 1.0, 48), (2.0, 2.0, 48), (2.5, 1.5, 48)]

for book_name, coins in BOOKS:
    sig = make_sig(coins)
    print('', flush=True)
    print('  ===== %s: r4 short_pop_in_downtrend =====' % book_name, flush=True)
    base = book(sig, {})
    b_py = per_year(base)
    for g in CANDS:
        rows = book(sig, {'r4': g})
        mf, dd = m_fix(rows)
        md = m_dd(rows)
        py = per_year(rows)
        own = np.array([x[2] for x in rows if x[6] == 'r4'])
        print('    r4 %s | 1.4%%: $%6.0f %4.1f%% | DD12%%: $%6.0f | свои сделки r4: %4d ВР %4.1f%% ср %+.3f | лучше по годам %d/5 (%s)'
              % (g, mf, 100 * dd, md, len(own), 100 * np.mean(own > 0) if len(own) else 0,
                 own.mean() if len(own) else 0, sum(1 for a, b in zip(b_py, py) if b > a),
                 ' '.join('%+.0f' % (b - a) for a, b in zip(b_py, py))), flush=True)

# plateau and stop-only blind check on book 1
sig = make_sig(BOOKS[0][1])
print('', flush=True)
print('  ===== площадка r4 на книге 1: 1.4%% $ (просадка) =====', flush=True)
TPS = (1.0, 1.25, 1.5, 1.75, 2.0)
print('    стоп \ тейк  ' + ''.join('%16.2f' % t for t in TPS) + '     (удержание 48ч)', flush=True)
for sl in (1.5, 1.75, 2.0, 2.25, 2.5, 3.0):
    cells = []
    for tp in TPS:
        mf, dd = m_fix(book(sig, {'r4': (sl, tp, 48)}))
        cells.append('$%6.0f (%4.1f%%)' % (mf, 100 * dd))
    print('    %5.2f        %s' % (sl, ' '.join(cells)), flush=True)
print('    удержание при стопе 2 / тейке 1.5:', flush=True)
for hh in (24, 36, 48, 60, 72):
    mf, dd = m_fix(book(sig, {'r4': (2.0, 1.5, hh)}))
    print('      %2dч  $%6.0f (%4.1f%%)' % (hh, mf, 100 * dd), flush=True)

print('', flush=True)
print('  ===== слепой выбор СТОПА r4 при тейке 1.5 и 48ч (книга 1), мера 1.4%% =====', flush=True)
STOPS = (1.5, 2.0, 2.5, 3.0)
RES = {sl: book(sig, {'r4': (sl, 1.5, 48)}) for sl in STOPS}
BASEROWS = book(sig, {})
wins = seen = 0
for y in range(2022, 2027):
    lo, hi = YT[y], YT[y + 1]
    best, bk = -1, None
    for sl, rows in RES.items():
        m = sim_w.simulate([x for x in plain(rows) if not (lo <= x[0] < hi)], 0.014)['eq']
        if m > best:
            best, bk = m, sl
    mp = sim_w.simulate([x for x in plain(RES[bk]) if lo <= x[0] < hi], 0.014)['eq']
    mb = sim_w.simulate([x for x in plain(BASEROWS) if lo <= x[0] < hi], 0.014)['eq']
    seen += 1; wins += int(mp > mb)
    print('    %d выбран стоп %.1f | слепой год $%.0f против нынешней $%.0f %s'
          % (y, bk, mp, mb, 'лучше' if mp > mb else ('так же' if mp == mb else 'хуже')), flush=True)
print('    лучше нынешней в %d из %d лет' % (wins, seen), flush=True)
