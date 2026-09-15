"""Writes struct_stop.py: the old SMC bot's stop placement, tried on the bank and the pullback rule.

The SMC bot put its stop beyond the last swing low; the bank puts it a fixed 3 ATR away. The stop
analysis found 29% of stops are followed by the price reaching the target inside the same window,
which is what a stop sitting where the market sweeps liquidity would look like. So the stop is
placed beyond the lowest low (highest high for shorts) of the last N hourly bars plus a 0.25 ATR
buffer, with a floor, and in one family never closer than today's 3 ATR.

short_pop_in_downtrend keeps its own validated 2/1.5 geometry. Everything else as in stop_fixes.py:
measured costs, market entry, stops beyond 10% dropped, one position per coin, live sizing.
"""
import ast
import io

s = io.open('stop_fixes.py', encoding='utf-8').read()
head = s[:s.index("HOLDS = {'o1h'")]

TAIL = r'''
HOLDS = {'o1h': 48, 'o2h': 72}
SHORT_KEYS = {'r%d' % k for k, r in enumerate(RULES) if r.get('side', 'LONG') == 'SHORT'}
KEEP_OWN = {'r4'}


def book_s(sig, N=None, floor=1.0, buf=0.25, at_least=None):
    busy, out, sls = {}, [], []
    for close, prio, key, s, e, atr, fav, adv, Mu, Md, on, cn, tt, slip, vr, hour in sig:
        if busy.get(s, 0) > close:
            continue
        sl, tp, hh = GEO.get(key, (3.0, 1.0, HOLDS.get(key, 48)))
        H = hh * 4
        if N is not None and key not in KEEP_OWN:
            c = SP.CTX[s]
            i1 = int(np.searchsorted(c['T1'], close - 3600))
            if i1 < N or i1 >= len(c['T1']):
                continue
            B1 = c['B1']
            if key not in SHORT_KEYS:
                d = (e - float(np.min(B1[i1 - N + 1:i1 + 1, 2]))) / atr
            else:
                d = (float(np.max(B1[i1 - N + 1:i1 + 1, 1])) - e) / atr
            sl = max(floor, d + buf)
            if at_least is not None:
                sl = max(sl, at_least)
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
        sls.append(sl)
        busy[s] = end
    out.sort()
    return out, sls


def fix(r6):
    r = sim_w.simulate(r6, 0.014)
    return r['eq'], abs(L.dd_of(r['curve']))


def pyr(r6):
    return [sim_w.simulate([x for x in r6 if YT[y] <= x[0] < YT[y + 1]], 0.014)['eq'] for y in range(2022, 2027)]


BOOKS = (('КНИГА 1 (без XLM, AAVE)', [c for c in SP.COINS if c not in ('BILLUSDT', 'XLMUSDT', 'AAVEUSDT')]),
         ('КНИГА 2 (все 15 монет)', [c for c in SP.COINS if c != 'BILLUSDT']))
BASE_LBL = 'сейчас (3 ATR)'
VARS = [(BASE_LBL, dict(N=None)),
        ('структура 12ч, мин 1 ATR', dict(N=12, floor=1.0)),
        ('структура 24ч, мин 1 ATR', dict(N=24, floor=1.0)),
        ('структура 48ч, мин 1 ATR', dict(N=48, floor=1.0)),
        ('структура 24ч, мин 2 ATR', dict(N=24, floor=2.0)),
        ('структура 24ч, не ближе 3 ATR', dict(N=24, floor=1.0, at_least=3.0)),
        ('структура 48ч, не ближе 3 ATR', dict(N=48, floor=1.0, at_least=3.0))]
for bi, (book_name, coins) in enumerate(BOOKS):
    sig = make_sig_raw(coins)
    print('', flush=True)
    print('  ===== %s: СТОП ОТ СТРУКТУРЫ =====' % book_name, flush=True)
    RES = {}
    base = None
    for lbl, kw in VARS:
        rows, sls = book_s(sig, **kw)
        RES[lbl] = rows
        eq, dd = fix(rows)
        k, m = sim_w.money_at_dd(rows, 0.12)
        v = np.array([x[2] for x in rows])
        py = pyr(rows)
        if base is None:
            base = py
        years = (rows[-1][0] - rows[0][0]) / (365.25 * 86400)
        print('    %-30s | %4.0f сд/год ВР %4.1f%% ср %+.4f стоп медиана %.2f ATR | 1.4%%: $%6.0f %4.1f%% | DD12%%: $%6.0f | лучше по годам %d/5 (%s)'
              % (lbl, len(v) / years, 100 * np.mean(v > 0), v.mean(), float(np.median(sls)), eq, 100 * dd, m,
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
            mb = sim_w.simulate([x for x in RES[BASE_LBL] if lo <= x[0] < hi], 0.014)['eq']
            wins += int(mp > mb)
            print('      %d выбрано «%s» | $%.0f против $%.0f %s'
                  % (y, bk, mp, mb, 'лучше' if mp > mb else ('так же' if mp == mb else 'хуже')), flush=True)
        print('      лучше нынешнего в %d из 5 лет' % wins, flush=True)
'''

src = head + TAIL
ast.parse(src)
io.open('struct_stop.py', 'w', encoding='utf-8').write(src)
print('struct_stop.py готов, синтаксис ок')
