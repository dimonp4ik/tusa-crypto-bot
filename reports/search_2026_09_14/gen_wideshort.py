"""Writes wideshort.py: the mirror of item 9 on the short side.

Three facts point the same way. Short bursts are the best short trades (7+ shorts in a bar: win rate
86-92%, +0.16..+0.25R). The ablation says shorts carry most of the money - removing them costs 80% of
the account - and also most of the drawdown. And on the long side, buying coins with a shallower
pullback while the strict rule fires across the market was worth +14..+21%.

So: when the strict short rules (short_btc_up_morning: btc24 >= 0.01701 and hour in 8..11;
short_pop_in_downtrend: rsi2 >= 71.02 and rsi14 <= 35.22) fire on K or more distinct coins over the
current and previous hourly close, also short coins that are in the SHORT regime and meet relaxed
thresholds (rsi2 >= A, rsi14 <= B) but did not fire strictly.

Cells: A in 60 / 65, B in 40 / 45 / 50, K = 3 / 4, geometry as the pullback (stop 3 / take 1) and as
short_pop (stop 2 / take 1.5). Control: the breadth read 3 days earlier. Stacked on the final system
(boost n >= 10 + breadth over 2 hours); three books; full fill both measures; ten paired 85% fills for
the best two cells; rolling choice by money at 12% drawdown.
"""
import ast
import io

s = io.open('rollbreadth.py', encoding='utf-8').read()
head = s[:s.index("CELLS = [")]

TAIL = r'''
import bisect
DAY = 86400
FULL = BASE_KEYS | {'r2_4'}
SHORT_RULES = [(k, r) for k, r in enumerate(RULES) if r.get('side', 'LONG') == 'SHORT']


def short_ctx(s):
    """Per hourly close: (in SHORT regime, strict short fired, rsi2, rsi14, atr)."""
    c = SP.CTX[s]
    T1, F, iv, starts = c['T1'], c['F'], c['iv'], c['starts']
    out = {}
    for i in range(PB.MIN_HOURS, len(T1)):
        if not np.isfinite(F['atr'][i]):
            continue
        close = int(T1[i]) + 3600
        j = bisect.bisect_right(starts, close) - 1
        if j < 0 or not (iv[j][0] <= close < iv[j][1]) or iv[j][2] != 'SHORT':
            continue
        strict = any(PB._match(r, F, i) for _, r in SHORT_RULES)
        out[close] = (strict, float(F['rsi2'][i]), float(F['rsi14'][i]), float(F['atr'][i]))
    return out


def wide_short(coins, S, key, prio, A, B, K, shift, W=2):
    fired = collections.defaultdict(set)
    for s in coins:
        for close, (strict, r2, r14, atr) in S[s].items():
            if strict:
                fired[close].add(s)
    out = []
    for s in coins:
        c = SP.CTX[s]
        t15, a15, pos = c['t15'], c['a15'], c['pos']
        slip = SLIP.get(s, 0.0003)
        for close, (strict, r2, r14, atr) in S[s].items():
            if strict or atr <= 0 or not (r2 >= A and r14 <= B):
                continue
            u = set()
            for w in range(W):
                u |= fired.get(close - shift - w * 3600, set())
            if len(u) < K:
                continue
            j = pos.get(close)
            if j is None or j + MAXH > len(t15):
                continue
            o, h, l, cc = (a15[j:j + MAXH, x] for x in range(4))
            e = o[0] * (1 - slip)
            fav, adv = (e - l) / atr, (h - e) / atr
            out.append((close, prio, key, s, e, atr, fav, adv, np.maximum.accumulate(fav), np.maximum.accumulate(adv),
                        (e - o) / atr, (e - cc) / atr, t15[j:j + MAXH], slip, float('nan'), 0))
    return out


CELLS = [(60.0, 45.0, 3), (60.0, 45.0, 4), (65.0, 40.0, 3), (65.0, 40.0, 4), (65.0, 45.0, 3), (60.0, 50.0, 3)]
GEOS = ((3.0, 1.0, 'g31'), (2.0, 1.5, 'g215'))


def wk(A, B, K, g, sh=0):
    return 'w%d_%d_%d%s%s' % (int(A), int(B), K, g, ('s%d' % (sh // DAY)) if sh else '')


for book_name, coins in BOOKS3:
    H = {s: coin_hours(s) for s in coins}
    S = {s: short_ctx(s) for s in coins}
    base_sig = make_sig_raw(coins)
    extra = roll_entries(coins, H, 'r2_4', 4, 2, 4, 0)
    ws = []
    for A, B, K in CELLS:
        for sl, tp, g in GEOS:
            key = wk(A, B, K, g)
            GEO[key] = (sl, tp, 48)
            ws += wide_short(coins, S, key, 5, A, B, K, 0)
    for A, B, K in ((60.0, 45.0, 3), (65.0, 40.0, 3)):
        key = wk(A, B, K, 'g215', 3 * DAY)
        GEO[key] = (2.0, 1.5, 48)
        ws += wide_short(coins, S, key, 5, A, B, K, 3 * DAY)
    sig_all = sorted(base_sig + extra + ws, key=lambda x: (x[0], x[1]))
    full = boost(book(sig_all, FULL)[0])
    feq, fdd = fix2(full); fm = sim_w.money_at_dd(full, 0.12)[1]; fpy = pyr(full)
    print('', flush=True)
    print('  ===== %s: ИТОГОВАЯ $%.0f %.1f%% DD12 $%.0f =====' % (book_name, feq, 100 * fdd, fm), flush=True)
    RES = {'итоговая': full}
    for A, B, K in CELLS:
        for sl, tp, g in GEOS:
            key = wk(A, B, K, g)
            rows, cnt = book(sig_all, FULL | {key})
            alone, _ = book(sorted([x for x in ws if x[2] == key], key=lambda x: (x[0], x[1])), {key})
            va = np.array([x[2] for x in alone]) if alone else np.zeros(1)
            rows = boost(rows)
            eq, dd = fix2(rows)
            m = sim_w.money_at_dd(rows, 0.12)[1]
            py = pyr(rows)
            RES['rsi2>=%d rsi14<=%d K%d %s' % (A, B, K, 'стоп3/тейк1' if g == 'g31' else 'стоп2/тейк1.5')] = rows
            print('    rsi2>=%2d rsi14<=%2d K>=%d %-12s | в книге %3d сд, сам ВР %5.1f%% ср %+.3f | $%6.0f %4.1f%% DD12 $%6.0f %s | по годам %d/5 (%s)'
                  % (A, B, K, 'стоп3/тейк1' if g == 'g31' else 'стоп2/тейк1.5', cnt[key], 100 * np.mean(va > 0), va.mean(),
                     eq, 100 * dd, m, '+' if (eq > feq and m > fm) else ' ', sum(1 for a, b in zip(fpy, py) if b > a),
                     ' '.join('%+.0f' % (b - a) for a, b in zip(fpy, py))), flush=True)
    for A, B, K in ((60.0, 45.0, 3), (65.0, 40.0, 3)):
        key = wk(A, B, K, 'g215', 3 * DAY)
        rows, cnt = book(sig_all, FULL | {key})
        rows = boost(rows)
        eq, dd = fix2(rows)
        m = sim_w.money_at_dd(rows, 0.12)[1]
        print('    КОНТРОЛЬ rsi2>=%2d rsi14<=%2d K>=%d ширина -3 дня | в книге %3d сд | $%6.0f %4.1f%% DD12 $%6.0f'
              % (A, B, K, cnt[key], eq, 100 * dd, m), flush=True)
    cand = [k for k in RES if k != 'итоговая']
    top = sorted(cand, key=lambda k: -sim_w.money_at_dd(RES[k], 0.12)[1])[:2]
    print('    ЗАЛИВКА 85%% против ИТОГОВОЙ: %s' % ', '.join(top), flush=True)
    for lbl in top:
        p = lbl.split()
        A, B, K = float(p[0].split('>=')[1]), float(p[1].split('<=')[1]), int(p[2][1:])
        g = 'g31' if 'стоп3' in lbl else 'g215'
        key = wk(A, B, K, g)
        A_, B_ = [], []
        for seed in range(1, 11):
            rng = np.random.default_rng(seed)
            sub = [x for x in sig_all if rng.random() < 0.85]
            r0, r1 = boost(book(sub, FULL)[0]), boost(book(sub, FULL | {key})[0])
            e0, d0 = fix2(r0); e1, d1 = fix2(r1)
            A_.append((e0, d0, sim_w.money_at_dd(r0, 0.12)[1], pyr(r0))); B_.append((e1, d1, sim_w.money_at_dd(r1, 0.12)[1], pyr(r1)))
        med = lambda X, i: float(np.median([x[i] for x in X]))
        yw = ' '.join('%d/10' % sum(1 for x, y in zip(A_, B_) if y[3][j] > x[3][j]) for j in range(5))
        print('      %-34s итоговая $%.0f %.1f%% DD12 $%.0f | + источник $%.0f %.1f%% (макс %.1f%%) DD12 $%.0f | лучше $ %d/10, DD12 %d/10 | годы %s'
              % (lbl, med(A_, 0), 100 * med(A_, 1), med(A_, 2), med(B_, 0), 100 * med(B_, 1), 100 * max(x[1] for x in B_), med(B_, 2),
                 sum(1 for x, y in zip(A_, B_) if y[0] > x[0]), sum(1 for x, y in zip(A_, B_) if y[2] > x[2]), yw), flush=True)
    print('    СКОЛЬЗЯЩИЙ ВЫБОР: по $ при DD12 на 4 годах, замер 5-го при 1.4% против ИТОГОВОЙ', flush=True)
    wins = 0
    for y in range(2022, 2027):
        lo, hi = YT[y], YT[y + 1]
        best, bk = -1, None
        for lbl, rows in RES.items():
            mm = sim_w.money_at_dd([x for x in rows if not (lo <= x[0] < hi)], 0.12)[1]
            if mm > best:
                best, bk = mm, lbl
        mp = sim_w.simulate([x for x in RES[bk] if lo <= x[0] < hi], 0.014)['eq']
        mb = sim_w.simulate([x for x in full if lo <= x[0] < hi], 0.014)['eq']
        wins += int(mp > mb)
        print('      %d выбрано «%s» | $%.0f против $%.0f %s' % (y, bk, mp, mb, 'лучше' if mp > mb else ('так же' if mp == mb else 'хуже')), flush=True)
    print('      лучше итоговой в %d из 5 лет' % wins, flush=True)
'''

src = head + TAIL
ast.parse(src)
io.open('wideshort.py', 'w', encoding='utf-8').write(src)
print('wideshort.py готов, синтаксис ок')
