"""Writes pullshort.py: the true mirror of the pullback rule, as a SHORT source.

The system's best rule buys a short-term dip inside strength (rsi48 >= 56.38 and rsi6 <= 33.79). Its
mirror was only ever measured as the same thresholds traded the other way, which is not a mirror. The
real mirror sells a short-term pop inside weakness: rsi48 <= 43.62 and rsi6 >= 66.21, with the BTC
filter inverted (no shorts while BTC's daily close is above its SMA50), no regime requirement.

Measured as its own book, with each condition alone as a control, with mirror thresholds and with
quantile-matched thresholds, on 1h and 2h bars, and stacked onto the current system on both coin books
with a blind choice. Machinery from stop_fixes.py: measured costs, market entry, stops beyond 10%
dropped, one position per coin, live sizing.
"""
import ast
import io

s = io.open('stop_fixes.py', encoding='utf-8').read()
head = s[:s.index("HOLDS = {'o1h'")]

TAIL = r'''
HOLDS = {'o1h': 48, 'o2h': 72}


def short_signals(s, sec, hi48, lo6, btc_filter):
    c = SP.CTX[s]
    T, B = PB.build_bars(c['t15'], c['a15'], sec)
    cl = B[:, 3]
    r48, r6, atr = PB._rsi(cl, 48), PB._rsi(cl, 6), PB._atr(B)
    bdt, bdc = c['bdt'], np.asarray(c['bdc'], dtype=float)
    sma = PB._sma(bdc, 50)
    out = []
    for i in range(200, len(T)):
        if not np.isfinite(atr[i]):
            continue
        if hi48 is not None and not r48[i] <= hi48:
            continue
        if lo6 is not None and not r6[i] >= lo6:
            continue
        close = int(T[i]) + sec
        if btc_filter:
            d = np.searchsorted(bdt, close - 86400, side='right') - 1
            if d >= 49 and bdc[d] > sma[d]:
                continue
        out.append((close, float(atr[i])))
    return out


def short_entries(coins, key, prio, sec, hi48, lo6, btc_filter):
    out = []
    for s in coins:
        c = SP.CTX[s]
        t15, a15, pos = c['t15'], c['a15'], c['pos']
        slip = SLIP.get(s, 0.0003)
        for close, atr in short_signals(s, sec, hi48, lo6, btc_filter):
            j = pos.get(close)
            if j is None or j + MAXH > len(t15) or atr <= 0:
                continue
            o, h, l, cc = (a15[j:j + MAXH, x] for x in range(4))
            e = o[0] * (1 - slip)
            fav = (e - l) / atr
            adv = (h - e) / atr
            out.append((close, prio, key, s, e, atr, fav, adv, np.maximum.accumulate(fav),
                        np.maximum.accumulate(adv), (e - o) / atr, (e - cc) / atr, t15[j:j + MAXH], slip,
                        float('nan'), 0))
    return out


def book(sig, keys):
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
        out.append((close, end, (x * atr / e - slip - FEE) / sf, sf, s, 1.0))
        busy[s] = end
    out.sort()
    return out


def fix(r6):
    r = sim_w.simulate(r6, 0.014)
    return r['eq'], abs(L.dd_of(r['curve']))


def pyr(r6):
    return [sim_w.simulate([x for x in r6 if YT[y] <= x[0] < YT[y + 1]], 0.014)['eq'] for y in range(2022, 2027)]


def tpy(rows):
    if len(rows) < 2:
        return 0.0
    return len(rows) / ((rows[-1][0] - rows[0][0]) / (365.25 * 86400))


# quantile-matched thresholds: the long rule's tail shares, mirrored
r48_all, r6_all = [], []
for sym in [c for c in SP.COINS if c != 'BILLUSDT']:
    cl = SP.CTX[sym]['B1'][:, 3]
    a, b = PB._rsi(cl, 48)[800:], PB._rsi(cl, 6)[800:]
    r48_all.append(a[np.isfinite(a)]); r6_all.append(b[np.isfinite(b)])
r48_all, r6_all = np.concatenate(r48_all), np.concatenate(r6_all)
share_hi = float(np.mean(r48_all >= 56.3761))
share_lo = float(np.mean(r6_all <= 33.7947))
Q48 = float(np.quantile(r48_all, share_hi))
Q6 = float(np.quantile(r6_all, 1 - share_lo))
print('  доли хвостов длинного правила: rsi48>=56.38 %.1f%%, rsi6<=33.79 %.1f%%' % (100 * share_hi, 100 * share_lo), flush=True)
print('  зеркальные пороги: rsi48<=43.62 и rsi6>=66.21 | по квантилям: rsi48<=%.2f и rsi6>=%.2f' % (Q48, Q6), flush=True)

BASE_KEYS = {'r0', 'r1', 'r2', 'r3', 'r4', 'o1h', 'o2h'}
EXTRA = [('psM1', 3, 3600, 43.6239, 66.2053, True),
         ('psM1n', 4, 3600, 43.6239, 66.2053, False),
         ('psM2', 5, 7200, 43.6239, 66.2053, True),
         ('psQ1', 6, 3600, None, None, True),
         ('psA48', 7, 3600, 43.6239, None, True),
         ('psA6', 8, 3600, None, 66.2053, True)]
HOLDS.update({'psM1': 48, 'psM1n': 48, 'psM2': 72, 'psQ1': 48, 'psA48': 48, 'psA6': 48})
BOOKS = (('КНИГА 1 (без XLM, AAVE)', [c for c in SP.COINS if c not in ('BILLUSDT', 'XLMUSDT', 'AAVEUSDT')]),
         ('КНИГА 2 (все 15 монет)', [c for c in SP.COINS if c != 'BILLUSDT']))
VARIANTS = [('система (сейчас)', BASE_KEYS),
            ('+ шорт-откат 1ч (зеркало, BTC)', BASE_KEYS | {'psM1'}),
            ('+ шорт-откат 1ч (зеркало, без BTC)', BASE_KEYS | {'psM1n'}),
            ('+ шорт-откат 1ч + 2ч (зеркало, BTC)', BASE_KEYS | {'psM1', 'psM2'}),
            ('+ шорт-откат 1ч (квантили, BTC)', BASE_KEYS | {'psQ1'})]

for bi, (book_name, coins) in enumerate(BOOKS):
    sig = make_sig_raw(coins)
    for key, prio, sec, h48, l6, btc in EXTRA:
        if key == 'psQ1':
            h48, l6 = Q48, Q6
        sig += short_entries(coins, key, prio, sec, h48, l6, btc)
    sig.sort(key=lambda x: (x[0], x[1]))
    print('', flush=True)
    print('  ===== %s: шорт-откат сам по себе =====' % book_name, flush=True)
    for key, lbl in (('psM1', 'зеркало 1ч, BTC'), ('psM1n', 'зеркало 1ч, без BTC'), ('psM2', 'зеркало 2ч, BTC'),
                     ('psQ1', 'квантили 1ч, BTC'), ('psA48', 'КОНТРОЛЬ только rsi48<=43.62'), ('psA6', 'КОНТРОЛЬ только rsi6>=66.21')):
        rows = book(sig, {key})
        if len(rows) < 30:
            print('    %-30s сделок мало (%d)' % (lbl, len(rows)), flush=True)
            continue
        v = np.array([x[2] for x in rows])
        yrs = []
        for y in range(2022, 2027):
            vy = [x[2] for x in rows if YT[y] <= x[0] < YT[y + 1]]
            yrs.append('%+.3f' % np.mean(vy) if len(vy) >= 15 else '  -   ')
        print('    %-30s %4.0f сд/год ВР %4.1f%% ср %+.4f | по годам %s' % (lbl, tpy(rows), 100 * np.mean(v > 0), v.mean(), ' '.join(yrs)), flush=True)
    print('  ===== %s: в стеке с системой =====' % book_name, flush=True)
    RES = {}
    base = None
    for lbl, keys in VARIANTS:
        rows = book(sig, keys)
        RES[lbl] = rows
        eq, dd = fix(rows)
        k, m = sim_w.money_at_dd(rows, 0.12)
        v = np.array([x[2] for x in rows])
        py = pyr(rows)
        if base is None:
            base = py
        print('    %-36s %4.0f сд/год ВР %4.1f%% | 1.4%%: $%6.0f %4.1f%% | DD12%%: $%6.0f | лучше по годам %d/5 (%s)'
              % (lbl, tpy(rows), 100 * np.mean(v > 0), eq, 100 * dd, m,
                 sum(1 for a, b in zip(base, py) if b > a), ' '.join('%+.0f' % (b - a) for a, b in zip(base, py))), flush=True)
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
            mb = sim_w.simulate([x for x in RES['система (сейчас)'] if lo <= x[0] < hi], 0.014)['eq']
            wins += int(mp > mb)
            print('      %d выбрано «%s» | $%.0f против $%.0f %s'
                  % (y, bk, mp, mb, 'лучше' if mp > mb else ('так же' if mp == mb else 'хуже')), flush=True)
        print('      лучше нынешней системы в %d из 5 лет' % wins, flush=True)
'''

src = head + TAIL
ast.parse(src)
io.open('pullshort.py', 'w', encoding='utf-8').write(src)
print('pullshort.py готов, синтаксис ок')
