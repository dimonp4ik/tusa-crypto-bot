"""Writes missed2.py: which price moves does the FINAL system still miss?

The missed-move analysis was run on the old bank-only system, before the pullback sources, the breadth
source and the size rules. The system now takes ~590 trades a year instead of ~430, and the breadth
work changed which moments it enters. Re-ask the question on the final system (boost n >= 10 + breadth
over 2 hours + item 10 weights, though weights do not change which moves are caught).

For every coin and day: the largest up-move (a low, then a higher high later that day) and the largest
down-move. A big move is in that coin's own top 20%. Caught = a trade of the right side was open at any
point between the move's start and its peak. Near = a trade of that side was entered within 6h of the
start.

Reported per book: caught share by year and by side, the mean R of the trades that caught them, and for
the missed ones - the state one hour before the move began (volatility regime, rsi48, rsi6, rsi14, BTC
24h, 24h return, position in range, regime, hour) against caught moves and against random ordinary
hours. Plus the specific near-miss question: how many missed up-moves started in an hour where the
strict pullback breadth was 1-3 coins (just under the threshold of 4) or where the coin itself passed
the relaxed thresholds but breadth was short.
"""
import ast
import io

s = io.open('rollbreadth.py', encoding='utf-8').read()
head = s[:s.index("CELLS = [")]

TAIL = r'''
import bisect
import datetime as DT
FULL = BASE_KEYS | {'r2_4'}
SHORT_KEYS = {'r%d' % k for k, r in enumerate(RULES) if r.get('side', 'LONG') == 'SHORT'}


def book_keyed(sig, keys):
    busy, out = {}, []
    for close, prio, key, s, e, atr, fav, adv, Mu, Md, on, cn, tt, slip, vr, hour in sig:
        if key not in keys or busy.get(s, 0) > close:
            continue
        sl, tp, hh = GEO.get(key, (3.0, 1.0, HOLDS.get(key, 48)))
        Hh = hh * 4
        sf = sl * atr / e
        if sf > 0.10:
            continue
        js = int(np.searchsorted(Md[:Hh], sl, side='left'))
        jt = int(np.searchsorted(Mu[:Hh], tp, side='left'))
        if js < Hh and js <= jt:
            jj, x = js, min(-sl, on[js])
        elif jt < Hh:
            jj, x = jt, tp
        else:
            jj, x = Hh - 1, cn[Hh - 1]
        end = int(tt[jj]) + 900
        out.append((close, end, (x * atr / e - slip - FEE) / sf, sf, s, key not in SHORT_KEYS))
        busy[s] = end
    out.sort()
    return out


rng = np.random.default_rng(5)
for book_name, coins in BOOKS3:
    H = {s: coin_hours(s) for s in coins}
    base_sig = make_sig_raw(coins)
    extra = roll_entries(coins, H, 'r2_4', 4, 2, 4, 0)
    sig_all = sorted(base_sig + extra, key=lambda x: (x[0], x[1]))
    rows = book_keyed(sig_all, FULL)
    by_coin = collections.defaultdict(list)
    for close, end, R, sf, s, lg in rows:
        by_coin[s].append((close, end, lg, R))
    breadth = collections.Counter()
    for s in coins:
        for close, (a, b, atr, ok) in H[s].items():
            if ok and a >= 56.3761 and b <= 33.7947:
                breadth[close] += 1
    roll_b = {}
    for t in set(c for s in coins for c in H[s]):
        roll_b[t] = breadth.get(t, 0) + breadth.get(t - 3600, 0)
    print('', flush=True)
    print('  ===== %s: сделок %d =====' % (book_name, len(rows)), flush=True)
    MOVES, BASE_H, FEAT = [], [], {}
    for s in coins:
        c = SP.CTX[s]
        t15, a15, T1, F = c['t15'], c['a15'], c['T1'], c['F']
        cl = c['B1'][:, 3]
        FEAT[s] = dict(T1=T1, volreg=F['volreg'], rsi14=F['rsi14'], rsi48=PB._rsi(cl, 48), rsi6=PB._rsi(cl, 6),
                       btc24=F['btc24'], ret24=F['ret24'], rng=F['rng'], iv=c['iv'], starts=c['starts'])
        day = t15 // 86400
        bounds = np.r_[0, np.flatnonzero(np.diff(day)) + 1, len(t15)]
        daily = []
        for a, b in zip(bounds[:-1], bounds[1:]):
            if b - a < 80:
                continue
            lo, hi = a15[a:b, 2], a15[a:b, 1]
            gain = hi / np.minimum.accumulate(lo) - 1
            k = int(np.argmax(gain))
            st = int(np.argmin(lo[:k + 1]))
            drop = 1 - lo / np.maximum.accumulate(hi)
            k2 = int(np.argmax(drop))
            st2 = int(np.argmin(-hi[:k2 + 1]))
            daily.append((int(t15[a + st]), int(t15[a + k]), float(gain[k]), int(t15[a + st2]), int(t15[a + k2]), float(drop[k2])))
        if not daily:
            continue
        up_thr, dn_thr = np.quantile([d[2] for d in daily], 0.8), np.quantile([d[5] for d in daily], 0.8)
        tr = sorted(by_coin[s])
        for st, pk, g, st2, pk2, dr in daily:
            for lg, a0, a1, size, thr in ((True, st, pk, g, up_thr), (False, st2, pk2, dr, dn_thr)):
                if size < thr or a1 <= a0:
                    continue
                over = [x for x in tr if x[2] == lg and x[0] < a1 and x[1] > a0]
                near = [x for x in tr if x[2] == lg and abs(x[0] - a0) <= 6 * 3600]
                MOVES.append(dict(coin=s, long=lg, start=a0, size=size, caught=bool(over), near=bool(near),
                                  R=float(np.mean([x[3] for x in over])) if over else float('nan')))
        idx = rng.choice(np.arange(800, len(T1)), size=min(300, max(1, len(T1) - 800)), replace=False)
        BASE_H += [(s, int(i)) for i in idx]

    def feats_at(s, ts):
        f = FEAT[s]
        i = int(np.searchsorted(f['T1'], ts - 3600, side='right')) - 1
        if i < 800 or i >= len(f['T1']):
            return None
        close = int(f['T1'][i]) + 3600
        j = bisect.bisect_right(f['starts'], close) - 1
        reg = f['iv'][j][2] if j >= 0 and f['iv'][j][0] <= close < f['iv'][j][1] else 'вне'
        return dict(volreg=f['volreg'][i], rsi48=f['rsi48'][i], rsi6=f['rsi6'][i], rsi14=f['rsi14'][i],
                    btc24=f['btc24'][i], ret24=f['ret24'][i], rng=f['rng'][i], reg=reg, close=close,
                    hour=DT.datetime.fromtimestamp(ts, DT.UTC).hour)

    for lg, name in ((True, 'ХОДЫ ВВЕРХ'), (False, 'ХОДЫ ВНИЗ')):
        mv = [m for m in MOVES if m['long'] == lg]
        if not mv:
            continue
        caught = [m for m in mv if m['caught']]
        print('    %s: крупных %d, средний размер %.1f%% | поймано %.1f%%, вход в 6ч от начала %.1f%%, ср R поймавших %+.3f'
              % (name, len(mv), 100 * np.mean([m['size'] for m in mv]), 100 * len(caught) / len(mv),
                 100 * sum(1 for m in mv if m['near']) / len(mv), np.nanmean([m['R'] for m in caught]) if caught else float('nan')), flush=True)
        yrs = collections.defaultdict(lambda: [0, 0])
        for m in mv:
            y = DT.datetime.fromtimestamp(m['start'], DT.UTC).year
            yrs[y][0] += 1; yrs[y][1] += int(m['caught'])
        print('      по годам: %s' % '  '.join('%d %.0f%%' % (y, 100 * v[1] / v[0]) for y, v in sorted(yrs.items())), flush=True)
        groups = {'пропущенные': [feats_at(m['coin'], m['start']) for m in mv if not m['caught']],
                  'пойманные': [feats_at(m['coin'], m['start']) for m in caught],
                  'обычные часы': [feats_at(s, int(FEAT[s]['T1'][i]) + 3601) for s, i in BASE_H]}
        groups = {k: [x for x in v if x] for k, v in groups.items()}
        print('      признак за час до хода       пропущенные          пойманные            обычные часы', flush=True)
        for fn in ('volreg', 'rsi48', 'rsi6', 'rsi14', 'btc24', 'ret24', 'rng'):
            cells = []
            for k in ('пропущенные', 'пойманные', 'обычные часы'):
                v = np.array([x[fn] for x in groups[k] if np.isfinite(x[fn])])
                cells.append('%7.3g [%6.3g..%6.3g]' % (np.median(v), np.quantile(v, .25), np.quantile(v, .75)) if len(v) else '     -')
            print('        %-8s                  %s' % (fn, ' '.join(cells)), flush=True)
        for k in ('пропущенные', 'пойманные'):
            cnt = collections.Counter(x['reg'] for x in groups[k])
            tot = sum(cnt.values())
            print('        режим %-12s %s' % (k, ', '.join('%s %.0f%%' % (r, 100 * n / tot) for r, n in cnt.most_common())), flush=True)
        if lg:
            miss = [m for m in mv if not m['caught']]
            b = [roll_b.get(feats_at(m['coin'], m['start'])['close'], 0) for m in miss if feats_at(m['coin'], m['start'])]
            b = np.array(b)
            print('        ширина в час начала пропущенного хода: 0 монет %.0f%%, 1-3 монеты %.0f%%, 4+ %.0f%%'
                  % (100 * np.mean(b == 0), 100 * np.mean((b >= 1) & (b <= 3)), 100 * np.mean(b >= 4)), flush=True)
            soft = [m for m in miss if (lambda f: f and f['rsi48'] >= 50 and f['rsi6'] <= 45)(feats_at(m['coin'], m['start']))]
            print('        из пропущенных прошли бы мягкие пороги (rsi48>=50, rsi6<=45): %.0f%% (%d из %d)'
                  % (100 * len(soft) / max(1, len(miss)), len(soft), len(miss)), flush=True)
'''

src = head + TAIL
ast.parse(src)
io.open('missed2.py', 'w', encoding='utf-8').write(src)
print('missed2.py готов, синтаксис ок')
