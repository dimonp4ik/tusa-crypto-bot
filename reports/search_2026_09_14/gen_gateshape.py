"""Writes gateshape.py: the 4h regime gate is load-bearing - is it the right SHAPE?

bankgate settled that the bank dies without its gate (-98%) and that no blind year prefers anything else.
So the gate carries the strategy. But its shape was never chosen: it is binary and exactly co-terminous
with the 4h position - the bank trades while trend4h holds, and stops the same second it exits.

Nothing says that is the best window. Two plausible mis-shapes:
  * the move usually outlives the 4h stop, so the hours just AFTER an exit may still be tradable;
  * a trend is not uniform - the first hours after an open differ from the tail.

Variants (all keep everything else identical, BTC SMA50 still blocks longs):
  as now | +4h, +8h, +12h of grace after each exit | only the first 12h / 24h after an open |
  only after the first 12h
Three books, both measures, and the shape chosen blind on four years then read on the fifth.
"""
import ast
import io

s = io.open('widestopref.py', encoding='utf-8').read()
head = s[:s.index("for bi, (book_name, coins) in enumerate(BOOKS3):")]

TAIL = r'''
import bisect
import datetime as DT
WIDE_REF = 0.060
YEARS = (2022, 2023, 2024, 2025, 2026)
H = 3600
SHAPES = (('как сейчас', 0, 0, 10 ** 9),
          ('продление +4ч', 4 * H, 0, 10 ** 9),
          ('продление +8ч', 8 * H, 0, 10 ** 9),
          ('продление +12ч', 12 * H, 0, 10 ** 9),
          ('только первые 12ч', 0, 0, 12 * H),
          ('только первые 24ч', 0, 0, 24 * H),
          ('только после 12ч', 0, 12 * H, 10 ** 9))


def yr(ts):
    return DT.datetime.fromtimestamp(ts, DT.UTC).year


def bank_shape(coin, grace, lo, hi, sma_n=50):
    c = SP.CTX[coin]
    T1, F, out = c['T1'], c['F'], {}
    bsm = c['sma'][sma_n]
    for i in range(PB.MIN_HOURS, len(T1)):
        close = int(T1[i]) + SP.HOUR
        if np.isnan(F['atr'][i]):
            continue
        j = bisect.bisect_right(c['starts'], close) - 1
        if j < 0:
            continue
        a, b, side = c['iv'][j][0], c['iv'][j][1], c['iv'][j][2]
        age = close - a
        if not (a <= close < b + grace) or not (lo <= age < hi):
            continue
        for k, r in enumerate(RULES):
            if r.get('side', 'LONG') != side or not PB._match(r, F, i):
                continue
            if side == 'LONG':
                d = bisect.bisect_right(c['bdt'], close - 86400) - 1
                if d >= sma_n - 1 and c['bdc'][d] < bsm[d]:
                    break
            out[close] = (k, float(c['B1'][i, 3]), float(F['atr'][i]))
            break
    return out


def sig_for(coins, shape):
    _, grace, lo, hi = shape
    sig = []
    for s in coins:
        c = SP.CTX[s]
        t15, a15, pos = c['t15'], c['a15'], c['pos']
        slip = SLIP.get(s, 0.0003)
        items = [(t, 0, 'r%d' % v[0], RULES[v[0]].get('side', 'LONG') == 'LONG', v[2])
                 for t, v in bank_shape(s, grace, lo, hi).items()]
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
            hour = DT.datetime.fromtimestamp(close, DT.UTC).hour
            sig.append((close, prio, key, s, e, atr, fav, adv, np.maximum.accumulate(fav),
                        np.maximum.accumulate(adv), (o - e) / atr if lg else (e - o) / atr,
                        (cc - e) / atr if lg else (e - cc) / atr, t15[j:j + MAXH], slip, vr, hour))
    sig.sort(key=lambda x: (x[0], x[1]))
    return sig


def prep(sig, coins, HH):
    allsig = sorted(sig + roll_entries(coins, HH, 'r2_4', 4, 2, 4, 0), key=lambda x: (x[0], x[1]))
    GEO.clear(); GEO.update(BASE_GEO)
    rows, keys = book_keyed(allsig, FULL)
    fin = weigh_src(rows, keys, {'r2_4'}, 1.25)
    refs = [(WIDE_REF if k == 'r2_4' else sim_w.REF) for k in keys]
    return fin, refs


def meas(fin, refs, keep=None):
    if keep is not None:
        ix = [i for i, x in enumerate(fin) if keep(x[0])]
        fin = [fin[i] for i in ix]; refs = [refs[i] for i in ix]
    r = sim_ref(fin, 0.014, refs)
    return r['eq'], abs(L.dd_of(r['curve'])), money_at_dd_ref(fin, refs), len(fin)


print('  ФОРМА ГЕЙТА ПО РЕЖИМУ 4ч', flush=True)
for book_name, coins in BOOKS3:
    HH = {s: coin_hours(s) for s in coins}
    P = [prep(sig_for(coins, sh), coins, HH) for sh in SHAPES]
    e0, d0, m0, n0 = meas(*P[0])
    print('', flush=True)
    print('  %s' % book_name, flush=True)
    for i, sh in enumerate(SHAPES):
        e, d, m, n = meas(*P[i])
        print('    %-20s сделок %4d | $%7.0f (%+6.1f%%) просадка %.1f%% | при равной просадке $%7.0f (%+6.1f%%)'
              % (sh[0], n, e, 100 * (e / e0 - 1), 100 * d, m, 100 * (m / m0 - 1)), flush=True)
    print('    СЛЕПОЙ ВЫБОР формы по четырём годам, замер на пятом', flush=True)
    wins = 0
    for y in YEARS:
        eb, _, mb, _ = meas(*P[0], keep=lambda t: yr(t) != y)
        sc = []
        for i in range(len(SHAPES)):
            e, _, m, _ = meas(*P[i], keep=lambda t: yr(t) != y)
            sc.append((min(e / eb, m / mb), i))
        sc.sort(reverse=True)
        pick = sc[0][1]
        e, _, m, _ = meas(*P[pick], keep=lambda t: yr(t) == y)
        eb, _, mb, _ = meas(*P[0], keep=lambda t: yr(t) == y)
        ok = (e > eb and m > mb)
        wins += ok
        print('      %d: выбрана форма «%s» | на этом году %+6.1f%% денег, %+6.1f%% при равной просадке %s'
              % (y, SHAPES[pick][0], 100 * (e / eb - 1), 100 * (m / mb - 1), 'лучше' if ok else ''), flush=True)
    print('      лет, где выбранная вслепую форма лучше нынешней по ОБЕИМ мерам: %d из %d' % (wins, len(YEARS)), flush=True)
'''

src = head + TAIL
ast.parse(src)
io.open('gateshape.py', 'w', encoding='utf-8').write(src)
print('gateshape.py gotov, sintaksis ok')
