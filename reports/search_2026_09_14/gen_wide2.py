"""Writes wide2.py: extend the breadth source to more grids and a wider count.

combo.py accepted "boost x1.25 at n>=10 + wide 50/45 when the strict 1h pullback fires on 4+ coins".
The same mechanism may live in two more places:

  w2   - the 2h grid: when the strict 2h pullback fires on K coins at the same 2h close, take coins
         with relaxed 2h thresholds (50/45), hold 72h like the strict 2h source.
  wU   - a wider breadth count: strict 1h pullback OR any bank LONG rule firing at the hour, relaxed
         1h entries 50/45 when that count >= K.

Everything is stacked on the new best (boost + wide 50/45 K4) and the boost is re-counted on the
final rows. Controls: breadth read 3 and 7 days earlier. Three books, full fill and ten paired 85%
fills, both measures, per-year wins, rolling choice by money at 12% drawdown.
"""
import ast
import io

s = io.open('combo.py', encoding='utf-8').read()
head = s[:s.index("VARS = (")]

TAIL = r'''
DAY = 86400
HOLDS.update({'w50': 48, 'w2': 72, 'wU': 48})
LONG_RULES = {k for k, r in enumerate(RULES) if r.get('side', 'LONG') == 'LONG'}


def coin_bars(s, sec):
    c = SP.CTX[s]
    T, B = PB.build_bars(c['t15'], c['a15'], sec)
    cl = B[:, 3]
    r48, r6, atr = PB._rsi(cl, 48), PB._rsi(cl, 6), PB._atr(B)
    bdt, bdc = c['bdt'], np.asarray(c['bdc'], dtype=float)
    sma = PB._sma(bdc, 50)
    out = {}
    for i in range(200, len(T)):
        if not (np.isfinite(atr[i]) and np.isfinite(r48[i]) and np.isfinite(r6[i])):
            continue
        close = int(T[i]) + sec
        d = np.searchsorted(bdt, close - 86400, side='right') - 1
        out[close] = (float(r48[i]), float(r6[i]), float(atr[i]), not (d >= 49 and bdc[d] < sma[d]))
    return out


def union_entries(coins, H, key, prio, lo48, hi6, K, shift):
    breadth = collections.Counter()
    bank_long = collections.defaultdict(set)
    for s in coins:
        for t, v in SP.bank_signals(RULES, s, 50).items():
            if v[0] in LONG_RULES:
                bank_long[t].add(s)
        for close, (a, b, atr, ok) in H[s].items():
            if ok and a >= 56.3761 and b <= 33.7947:
                bank_long[close].add(s)
    for t, ss in bank_long.items():
        breadth[t] = len(ss)
    out = []
    for s in coins:
        c = SP.CTX[s]
        t15, a15, pos = c['t15'], c['a15'], c['pos']
        slip = SLIP.get(s, 0.0003)
        for close, (a, b, atr, ok) in H[s].items():
            if not ok or atr <= 0 or (a >= 56.3761 and b <= 33.7947) or not (a >= lo48 and b <= hi6):
                continue
            if breadth.get(close - shift, 0) < K:
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


BEST = BASE_KEYS | {'w50'}
VARS = [('система (сейчас)', BASE_KEYS, False),
        ('ЛУЧШЕЕ: буст + ширина 50/45 K4', BEST, True),
        ('+ ширина 2ч K3', BEST | {'w2_3'}, True),
        ('+ ширина 2ч K4', BEST | {'w2_4'}, True),
        ('  КОНТРОЛЬ 2ч K3 -3 дня', BEST | {'w2_3s3'}, True),
        ('  КОНТРОЛЬ 2ч K3 -7 дней', BEST | {'w2_3s7'}, True),
        ('широкий счёт 1ч+банк K5 вместо 1ч K4', BASE_KEYS | {'wU5'}, True),
        ('широкий счёт 1ч+банк K6 вместо 1ч K4', BASE_KEYS | {'wU6'}, True),
        ('  КОНТРОЛЬ широкий K5 -3 дня', BASE_KEYS | {'wU5s3'}, True),
        ('  КОНТРОЛЬ широкий K5 -7 дней', BASE_KEYS | {'wU5s7'}, True)]
HOLDS.update({'w2_3': 72, 'w2_4': 72, 'w2_3s3': 72, 'w2_3s7': 72})


def build2(sig_all, keys, do_boost):
    rows, cnt = book(sig_all, keys)
    return (boost(rows) if do_boost else rows), cnt


for book_name, coins in BOOKS3:
    H1 = {s: coin_hours(s) for s in coins}
    H2 = {s: coin_bars(s, 7200) for s in coins}
    base_sig = make_sig_raw(coins)
    extra = wide_entries(coins, H1, 'w50', 4, 50.0, 45.0, 4, 0)
    for key, K, sh in (('w2_3', 3, 0), ('w2_4', 4, 0), ('w2_3s3', 3, 3 * DAY), ('w2_3s7', 3, 7 * DAY)):
        extra += wide_entries(coins, H2, key, 5, 50.0, 45.0, K, sh)
    for key, K, sh in (('wU5', 5, 0), ('wU6', 6, 0), ('wU5s3', 5, 3 * DAY), ('wU5s7', 5, 7 * DAY)):
        extra += union_entries(coins, H1, key, 4, 50.0, 45.0, K, sh)
    sig_all = sorted(base_sig + extra, key=lambda x: (x[0], x[1]))
    print('', flush=True)
    print('  ===== %s =====' % book_name, flush=True)
    RES = {}
    for lbl, keys, bo in VARS:
        rows, cnt = build2(sig_all, keys, bo)
        RES[lbl] = rows
        eq, dd = fix2(rows)
        m = sim_w.money_at_dd(rows, 0.12)[1]
        v = np.array([x[2] for x in rows])
        yrs = (rows[-1][0] - rows[0][0]) / (365.25 * 86400)
        added = ', '.join('%s %d' % (k, n) for k, n in sorted(cnt.items()) if k not in BASE_KEYS)
        print('    100%% %-40s %4.0f сд/год ВР %4.1f%% | $%6.0f %4.1f%% | DD12 $%6.0f | %s' % (lbl, len(v) / yrs, 100 * np.mean(v > 0), eq, 100 * dd, m, added), flush=True)
    R = {lbl: [] for lbl, _, _ in VARS}
    for seed in range(1, 11):
        rng = np.random.default_rng(seed)
        sub = [x for x in sig_all if rng.random() < 0.85]
        for lbl, keys, bo in VARS:
            rows, _ = build2(sub, keys, bo)
            e, d = fix2(rows)
            R[lbl].append(dict(eq=e, dd=d, m=sim_w.money_at_dd(rows, 0.12)[1], yrs=pyr(rows)))
    ref = R['ЛУЧШЕЕ: буст + ширина 50/45 K4']
    med = lambda X, k: float(np.median([x[k] for x in X]))
    print('    85%% заливки против ЛУЧШЕГО (медиана 10 розыгрышей):', flush=True)
    for lbl, _, _ in VARS:
        X = R[lbl]
        yw = ' '.join('%d/10' % sum(1 for a, x in zip(ref, X) if x['yrs'][k] > a['yrs'][k]) for k in range(5))
        print('      %-40s $%6.0f %4.1f%% (макс %4.1f%%) DD12 $%6.0f | лучше ЛУЧШЕГО $ %2d/10 DD12 %2d/10 | по годам %s'
              % (lbl, med(X, 'eq'), 100 * med(X, 'dd'), 100 * max(x['dd'] for x in X), med(X, 'm'),
                 sum(1 for a, x in zip(ref, X) if x['eq'] > a['eq']), sum(1 for a, x in zip(ref, X) if x['m'] > a['m']), yw), flush=True)
    print('    СКОЛЬЗЯЩИЙ ВЫБОР (без контролей): по $ при DD12 на 4 годах, замер 5-го при 1.4% против ЛУЧШЕГО', flush=True)
    wins = 0
    for y in range(2022, 2027):
        lo, hi = YT[y], YT[y + 1]
        best, bk = -1, None
        for lbl, rows in RES.items():
            if 'КОНТРОЛЬ' in lbl:
                continue
            mm = sim_w.money_at_dd([x for x in rows if not (lo <= x[0] < hi)], 0.12)[1]
            if mm > best:
                best, bk = mm, lbl
        mp = sim_w.simulate([x for x in RES[bk] if lo <= x[0] < hi], 0.014)['eq']
        mb = sim_w.simulate([x for x in RES['ЛУЧШЕЕ: буст + ширина 50/45 K4'] if lo <= x[0] < hi], 0.014)['eq']
        wins += int(mp > mb)
        print('      %d выбрано «%s» | $%.0f против $%.0f %s' % (y, bk, mp, mb, 'лучше' if mp > mb else ('так же' if mp == mb else 'хуже')), flush=True)
    print('      лучше ЛУЧШЕГО в %d из 5 лет' % wins, flush=True)
'''

src = head + TAIL
ast.parse(src)
io.open('wide2.py', 'w', encoding='utf-8').write(src)
print('wide2.py готов, синтаксис ок')
