"""Writes bankgate.py: does the BANK need the 4h regime gate?

The bank's five rules only fire while the paper 4h trend strategy holds a position, and only in its
direction - regime_update() calls T4.advance(gate=True) and bank_signals refuses any hour outside an
interval or with the wrong side. For the PULLBACK rule that same gate was measured and dropped: it cost
25% (item 1 of the proposal). For the bank itself nobody has ever asked. Five rules tuned inside a gate
may simply have learnt to live in it.

Three variants, three books, both measures, full fill:
  A  as now         - inside an interval AND the rule's side equals the regime's side
  B  side free      - inside an interval, any rule may fire (its own side is used)
  C  no gate        - the rule fires whenever its conditions match; BTC SMA50 still blocks longs

Then the only test that matters for a change of this size: choose the variant blind on four years and
read the fifth, for each of the five years.
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
VARIANTS = ('A как сейчас', 'B сторона свободна', 'C без гейта')


def yr(ts):
    return DT.datetime.fromtimestamp(ts, DT.UTC).year


def bank_var(coin, mode, sma_n=50):
    """mode 0 = as now, 1 = inside interval any side, 2 = no regime at all."""
    c = SP.CTX[coin]
    T1, F, out = c['T1'], c['F'], {}
    bsm = c['sma'][sma_n]
    for i in range(PB.MIN_HOURS, len(T1)):
        close = int(T1[i]) + SP.HOUR
        if np.isnan(F['atr'][i]):
            continue
        side = None
        if mode < 2:
            j = bisect.bisect_right(c['starts'], close) - 1
            if j < 0 or not (c['iv'][j][0] <= close < c['iv'][j][1]):
                continue
            side = c['iv'][j][2]
        for k, r in enumerate(RULES):
            rs = r.get('side', 'LONG')
            if mode == 0 and rs != side:
                continue
            if not PB._match(r, F, i):
                continue
            if rs == 'LONG':
                d = bisect.bisect_right(c['bdt'], close - 86400) - 1
                if d >= sma_n - 1 and c['bdc'][d] < bsm[d]:
                    break
            out[close] = (k, float(c['B1'][i, 3]), float(F['atr'][i]))
            break
    return out


def sig_for(coins, mode):
    sig = []
    for s in coins:
        c = SP.CTX[s]
        t15, a15, pos = c['t15'], c['a15'], c['pos']
        slip = SLIP.get(s, 0.0003)
        items = [(t, 0, 'r%d' % v[0], RULES[v[0]].get('side', 'LONG') == 'LONG', v[2])
                 for t, v in bank_var(s, mode).items()]
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


def prep(sig, coins, H):
    extra = roll_entries(coins, H, 'r2_4', 4, 2, 4, 0)
    allsig = sorted(sig + extra, key=lambda x: (x[0], x[1]))
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


print('  НУЖЕН ЛИ БАНКУ ГЕЙТ ПО РЕЖИМУ 4ч', flush=True)
for book_name, coins in BOOKS3:
    H = {s: coin_hours(s) for s in coins}
    P = [prep(sig_for(coins, m), coins, H) for m in range(3)]
    e0, d0, m0, n0 = meas(*P[0])
    print('', flush=True)
    print('  %s' % book_name, flush=True)
    for m, lbl in enumerate(VARIANTS):
        e, d, mm, n = meas(*P[m])
        print('    %-20s сделок %4d | $%7.0f (%+6.1f%%) просадка %.1f%% | при равной просадке $%7.0f (%+6.1f%%)'
              % (lbl, n, e, 100 * (e / e0 - 1), 100 * d, mm, 100 * (mm / m0 - 1)), flush=True)
    print('    СЛЕПОЙ ВЫБОР варианта по четырём годам, замер на пятом', flush=True)
    wins = 0
    for y in YEARS:
        sc = []
        eb, _, mb, _ = meas(*P[0], keep=lambda t: yr(t) != y)
        for m in range(3):
            e, _, mm, _ = meas(*P[m], keep=lambda t: yr(t) != y)
            sc.append((min(e / eb, mm / mb), m))
        sc.sort(reverse=True)
        pick = sc[0][1]
        e, _, mm, _ = meas(*P[pick], keep=lambda t: yr(t) == y)
        eb, _, mb, _ = meas(*P[0], keep=lambda t: yr(t) == y)
        ok = (e > eb and mm > mb)
        wins += ok
        print('      %d: выбран вариант %s | на этом году %+6.1f%% денег, %+6.1f%% при равной просадке %s'
              % (y, VARIANTS[pick], 100 * (e / eb - 1), 100 * (mm / mb - 1), 'лучше' if ok else ''), flush=True)
    print('      лет, где выбранный вслепую вариант лучше нынешнего по ОБЕИМ мерам: %d из %d' % (wins, len(YEARS)), flush=True)
'''

src = head + TAIL
ast.parse(src)
io.open('bankgate.py', 'w', encoding='utf-8').write(src)
print('bankgate.py gotov, sintaksis ok')
