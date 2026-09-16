"""Writes pullthr.py: are the pullback rule's own thresholds a plateau or a cliff?

The short sweep just showed rsi14 <= 35.22 sits one step from a collapse. The pullback rule
(rsi48 >= 56.3761 and rsi6 <= 33.7947) is the single biggest addition in the proposal and its thresholds
were mined once and never re-measured on the current system. They also drive item 9: breadth counts the
coins where the STRICT rule fires, so moving them moves the breadth source too - both are rebuilt per
cell here, while the relaxed thresholds of item 9 stay at 50/45.

Grid: rsi48 >= 52 / 54 / 56.3761 / 58 / 60 against rsi6 <= 28 / 31 / 33.7947 / 36 / 39, on the final
system (items 8-11), three books, money at 1.4% and at 12% drawdown. The live cell is marked '=' and
must reproduce the known numbers ($7145 / 10.6% / $12190 / 2571 trades on book 1) - if it does not, the
grid is void, as the first short sweep was.
"""
import ast
import io

s = io.open('widestopref.py', encoding='utf-8').read()
head = s[:s.index("for bi, (book_name, coins) in enumerate(BOOKS3):")]

TAIL = r'''
WIDE_REF = 0.060
A48 = (52.0, 54.0, 56.3761, 58.0, 60.0)
B6 = (28.0, 31.0, 33.7947, 36.0, 39.0)
LOOSE_A, LOOSE_B = 50.0, 45.0        # мягкие пороги пункта 9 не двигаются


def bar_ctx(s, sec):
    """Признаки каждого закрытого бара: (close, rsi48, rsi6, atr, фильтр BTC пройден)."""
    c = SP.CTX[s]
    T, B = PB.build_bars(c['t15'], c['a15'], sec)
    cl = B[:, 3]
    r48, r6, atr = PB._rsi(cl, 48), PB._rsi(cl, 6), PB._atr(B)
    bdt, bdc = c['bdt'], np.asarray(c['bdc'], dtype=float)
    sma = PB._sma(bdc, 50)
    out = []
    for i in range(200, len(T)):
        if not (np.isfinite(atr[i]) and np.isfinite(r48[i]) and np.isfinite(r6[i])):
            continue
        close = int(T[i]) + sec
        d = np.searchsorted(bdt, close - 86400, side='right') - 1
        ok = not (d >= 49 and bdc[d] < sma[d])
        out.append((close, float(r48[i]), float(r6[i]), float(atr[i]), ok))
    return out


def entry_row(s, close, atr, key, prio, long_side=True):
    c = SP.CTX[s]
    t15, a15, pos = c['t15'], c['a15'], c['pos']
    j = pos.get(close)
    if j is None or j + MAXH > len(t15) or atr <= 0:
        return None
    slip = SLIP.get(s, 0.0003)
    o, h, l, cc = (a15[j:j + MAXH, x] for x in range(4))
    e = o[0] * (1 + slip)
    fav, adv = (h - e) / atr, (e - l) / atr
    return (close, prio, key, s, e, atr, fav, adv, np.maximum.accumulate(fav), np.maximum.accumulate(adv),
            (o - e) / atr, (cc - e) / atr, t15[j:j + MAXH], slip, float('nan'), 0)


def build(coins, C1, C2, a48, b6):
    """Строгий откат на 1ч и 2ч + широкий откат (ширина за текущий и прошлый час) при данных порогах."""
    out = []
    fired = collections.defaultdict(set)
    for s in coins:
        for close, r48, r6, atr, ok in C1[s]:
            if ok and r48 >= a48 and r6 <= b6:
                fired[close].add(s)
                r = entry_row(s, close, atr, 'o1h', 1)
                if r:
                    out.append(r)
        for close, r48, r6, atr, ok in C2[s]:
            if ok and r48 >= a48 and r6 <= b6:
                r = entry_row(s, close, atr, 'o2h', 2)
                if r:
                    out.append(r)
    for s in coins:
        for close, r48, r6, atr, ok in C1[s]:
            if not ok or (r48 >= a48 and r6 <= b6):
                continue
            if not (r48 >= LOOSE_A and r6 <= LOOSE_B):
                continue
            if len(fired.get(close, set()) | fired.get(close - 3600, set())) < 4:
                continue
            r = entry_row(s, close, atr, 'r2_4', 4)
            if r:
                out.append(r)
    return out


for book_name, coins in BOOKS3:
    bank = [x for x in make_sig_raw(coins) if x[2] not in ('o1h', 'o2h')]
    C1 = {s: bar_ctx(s, 3600) for s in coins}
    C2 = {s: bar_ctx(s, 7200) for s in coins}
    GEO.clear(); GEO.update(BASE_GEO)

    def run(a48, b6):
        sig = sorted(bank + build(coins, C1, C2, a48, b6), key=lambda x: (x[0], x[1]))
        rf, kf = book_keyed(sig, FULL)
        fin = weigh_src(rf, kf, {'r2_4'}, 1.25)
        refs = [(WIDE_REF if k == 'r2_4' else sim_w.REF) for k in kf]
        r = sim_ref(fin, 0.014, refs)
        return r['eq'], abs(L.dd_of(r['curve'])), money_at_dd_ref(fin, refs), len(fin)

    beq, bdd, bm, bn = run(56.3761, 33.7947)
    print('', flush=True)
    print('  ===== %s =====' % book_name, flush=True)
    print('    живые пороги: $%.0f %.1f%% DD12 $%.0f (%d сделок) — контроль сборки' % (beq, 100 * bdd, bm, bn), flush=True)
    print('    rsi48\\rsi6 ' + ''.join('%-22s' % ('<=%.2f' % b) for b in B6), flush=True)
    for a in A48:
        cells = []
        for b in B6:
            eq, dd, m, n = run(a, b)
            mark = '=' if (abs(a - 56.3761) < 1e-6 and abs(b - 33.7947) < 1e-6) else ('*' if (eq > beq and m > bm) else ' ')
            cells.append('%s$%6.0f %4.1f%% $%6.0f ' % (mark, eq, 100 * dd, m))
        print('    >=%-7.2f %s' % (a, ''.join(cells)), flush=True)
'''

src = head + TAIL
ast.parse(src)
io.open('pullthr.py', 'w', encoding='utf-8').write(src)
print('pullthr.py готов, синтаксис ок')
