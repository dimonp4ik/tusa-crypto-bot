"""Writes breadth.py: when the whole market pulls back, take the coins that nearly qualified.

burst.py: trades opened together with 7+ others win 87-88% at +0.16..+0.18R - a pullback across many
coins at once is a market-wide dip inside strength, and it recovers. The strict pullback rule
(rsi48 >= 56.38 and rsi6 <= 33.79) only takes the coins that dipped deepest. When breadth is high,
coins with a shallower dip ride the same recovery - more moves caught from the best kind of moment.

Breadth at an hourly close = number of the book's coins where the strict 1h rule (with its BTC filter)
fires at that close. New source 'wide': a coin that does NOT fire strictly, but meets relaxed
thresholds, while breadth >= K. Entry at the next 15m open, stop 3 / take 1 ATR, 48h, one position per
coin, priority after bank and the strict sources.

Controls: the same relaxed entries with no breadth requirement, and with breadth read one week
earlier (same distribution, unrelated moment). Both books, money at 1.4% and at 12% drawdown,
per-year versus the current system, blind choice on book 1.
"""
import ast
import io

s = io.open('finer.py', encoding='utf-8').read()
head = s[:s.index("BASE_KEYS = {")]

TAIL = r'''
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


RELAX = ((52.0, 33.7947, 'rsi48>=52'), (56.3761, 40.0, 'rsi6<=40'), (53.0, 38.0, 'rsi48>=53 и rsi6<=38'), (50.0, 45.0, 'rsi48>=50 и rsi6<=45'))
BOOKS = (('КНИГА 1 (без XLM, AAVE)', [c for c in SP.COINS if c not in ('BILLUSDT', 'XLMUSDT', 'AAVEUSDT')]),
         ('КНИГА 2 (все 15 монет)', [c for c in SP.COINS if c != 'BILLUSDT']))

for bi, (book_name, coins) in enumerate(BOOKS):
    H = {s: coin_hours(s) for s in coins}
    base_sig = make_sig_raw(coins)
    base_rows, _ = book(base_sig, BASE_KEYS)
    beq, bdd = fix2(base_rows)
    bpy = pyr(base_rows)
    print('', flush=True)
    print('  ===== %s: система $%.0f просадка %.1f%% DD12 $%.0f =====' % (book_name, beq, 100 * bdd, sim_w.money_at_dd(base_rows, 0.12)[1]), flush=True)
    RES = {'система (сейчас)': base_rows}
    for lo48, hi6, rl in RELAX:
        for K, shift, kl in ((3, 0, 'ширина>=3'), (5, 0, 'ширина>=5'), (7, 0, 'ширина>=7'),
                             (None, 0, 'КОНТРОЛЬ без ширины'), (5, WEEK, 'КОНТРОЛЬ ширина>=5 неделей раньше')):
            extra = wide_entries(coins, H, 'wide', 3, lo48, hi6, K, shift)
            sig = sorted(base_sig + extra, key=lambda x: (x[0], x[1]))
            rows, cnt = book(sig, BASE_KEYS | {'wide'})
            alone, _ = book(sorted(extra, key=lambda x: (x[0], x[1])), {'wide'})
            va = np.array([x[2] for x in alone]) if alone else np.zeros(0)
            eq, dd = fix2(rows)
            m = sim_w.money_at_dd(rows, 0.12)[1]
            py = pyr(rows)
            lbl = '%s, %s' % (rl, kl)
            if not kl.startswith('КОНТРОЛЬ'):
                RES[lbl] = rows
            print('    %-46s сам %4d сд ВР %4.1f%% ср %s | в книге %4d | $%6.0f %4.1f%% | DD12 $%6.0f | лучше по годам %d/5 (%s)'
                  % (lbl, len(va), 100 * np.mean(va > 0) if len(va) else 0, ('%+.3f' % va.mean()) if len(va) else '  -  ',
                     cnt['wide'], eq, 100 * dd, m, sum(1 for a, b in zip(bpy, py) if b > a),
                     ' '.join('%+.0f' % (b - a) for a, b in zip(bpy, py))), flush=True)
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
            mb = sim_w.simulate([x for x in base_rows if lo <= x[0] < hi], 0.014)['eq']
            wins += int(mp > mb)
            print('      %d выбрано «%s» | $%.0f против $%.0f %s'
                  % (y, bk, mp, mb, 'лучше' if mp > mb else ('так же' if mp == mb else 'хуже')), flush=True)
        print('      лучше нынешней системы в %d из 5 лет' % wins, flush=True)
'''

src = head + TAIL
ast.parse(src)
io.open('breadth.py', 'w', encoding='utf-8').write(src)
print('breadth.py готов, синтаксис ок')
