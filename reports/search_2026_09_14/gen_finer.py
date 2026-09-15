"""Writes finer.py: the pullback rule read more often - finer bars and shifted grids.

The pullback rule is checked once per hour on bars that start on the hour, and once per two hours on
the even-hour grid. A dip inside strength that forms and recovers between two checks is never seen:
the 1h bar that started at :30 can satisfy the rule while the :00 bars on either side do not. Adding
the 2h source already paid +11%, so the rule gains from being read on more grids.

Sources tried, each stacked onto the current system: 30m bars, 45m bars, 1h bars shifted by 15/30/45
minutes, 2h bars shifted by 1h. Live could run all of these - they only need closed 15m bars. Priority
stays bank > 1h > 2h > new. Everything else as stop_fixes.py: measured costs, market entry at the next
15m open, stops beyond 10% dropped, one position per coin, live sizing, both books, both measures,
blind choice on book 1.
"""
import ast
import io

s = io.open('stop_fixes.py', encoding='utf-8').read()
head = s[:s.index("HOLDS = {'o1h'")]

TAIL = r'''
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
         ('o1h45', 7, 3600, 2700), ('o2h60', 8, 7200, 3600)]
HOLDS.update({'o30': 48, 'o45': 48, 'o1h15': 48, 'o1h30': 48, 'o1h45': 48, 'o2h60': 72})
BOOKS = (('КНИГА 1 (без XLM, AAVE)', [c for c in SP.COINS if c not in ('BILLUSDT', 'XLMUSDT', 'AAVEUSDT')]),
         ('КНИГА 2 (все 15 монет)', [c for c in SP.COINS if c != 'BILLUSDT']))
VARIANTS = [('система (сейчас)', BASE_KEYS),
            ('+ откат 30м', BASE_KEYS | {'o30'}),
            ('+ откат 45м', BASE_KEYS | {'o45'}),
            ('+ 1ч со сдвигом 30м', BASE_KEYS | {'o1h30'}),
            ('+ 1ч со сдвигами 15/30/45м', BASE_KEYS | {'o1h15', 'o1h30', 'o1h45'}),
            ('+ 2ч со сдвигом 1ч', BASE_KEYS | {'o2h60'}),
            ('+ 1ч сдвиг 30м + 2ч сдвиг 1ч', BASE_KEYS | {'o1h30', 'o2h60'}),
            ('+ всё', BASE_KEYS | {'o30', 'o45', 'o1h15', 'o1h30', 'o1h45', 'o2h60'})]

for bi, (book_name, coins) in enumerate(BOOKS):
    sig = make_sig_raw(coins)
    for key, prio, sec, off in EXTRA:
        sig += long_entries(coins, key, prio, sec, off)
    sig.sort(key=lambda x: (x[0], x[1]))
    print('', flush=True)
    print('  ===== %s: новые источники сами по себе =====' % book_name, flush=True)
    for key, prio, sec, off in EXTRA:
        rows, _ = book(sig, {key})
        v = np.array([x[2] for x in rows])
        yrs = []
        for y in range(2022, 2027):
            vy = [x[2] for x in rows if YT[y] <= x[0] < YT[y + 1]]
            yrs.append('%+.3f' % np.mean(vy) if len(vy) >= 15 else '  -   ')
        print('    %-8s %4.0f сд/год ВР %4.1f%% ср %+.4f | по годам %s' % (key, tpy(rows), 100 * np.mean(v > 0), v.mean(), ' '.join(yrs)), flush=True)
    print('  ===== %s: в стеке с системой =====' % book_name, flush=True)
    RES = {}
    base = None
    for lbl, keys in VARIANTS:
        rows, cnt = book(sig, keys)
        RES[lbl] = rows
        eq, dd = fix(rows)
        k, m = sim_w.money_at_dd(rows, 0.12)
        v = np.array([x[2] for x in rows])
        py = pyr(rows)
        if base is None:
            base = py
        extra = ', '.join('%s %d' % (kk, n) for kk, n in sorted(cnt.items()) if kk not in BASE_KEYS)
        print('    %-30s %4.0f сд/год ВР %4.1f%% | 1.4%%: $%6.0f %4.1f%% | DD12%%: $%6.0f | лучше по годам %d/5 (%s) %s'
              % (lbl, tpy(rows), 100 * np.mean(v > 0), eq, 100 * dd, m,
                 sum(1 for a, b in zip(base, py) if b > a), ' '.join('%+.0f' % (b - a) for a, b in zip(base, py)), extra), flush=True)
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
io.open('finer.py', 'w', encoding='utf-8').write(src)
print('finer.py готов, синтаксис ок')
