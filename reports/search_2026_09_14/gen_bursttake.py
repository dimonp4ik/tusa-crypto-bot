"""Writes bursttake.py: let the market-wide dips run further.

burst.py found trades opening together with 7+ others win 87-88% and are almost never stopped as a
group - a pullback across the whole market recovers. The system takes 1 ATR on every long, which
caps exactly those recoveries. A wider take on burst trades catches more of the move; it is an exit
change on (almost) the same set of trades, the kind of change that can actually be proven.

n for a trade = trades of the base book opening at the same timestamp (known at the moment of entry).
For LONG trades with n >= K: take 1.25 / 1.5 / 2.0 ATR, hold 48 or 72h, stop unchanged (3 ATR).
Control: the same exit applied to the same number of randomly chosen non-burst long trades
(median of 5 draws) - it separates "bursts recover further" from "a wider take is better anyway".
Both books, money at 1.4% and at 12% drawdown, per-year versus the system, blind choice on book 1.
"""
import ast
import io

s = io.open('burst.py', encoding='utf-8').read()
head = s[:s.index("for bi, (book_name, coins) in enumerate(BOOKS):")]

TAIL = r'''
SHORT_KEYS = {'r%d' % k for k, r in enumerate(RULES) if r.get('side', 'LONG') == 'SHORT'}


def book_t(sig, keys, wide, tp_w, hold_w):
    """wide: set of (close, coin) that get the wide exit."""
    busy, out = {}, []
    for close, prio, key, s, e, atr, fav, adv, Mu, Md, on, cn, tt, slip, vr, hour in sig:
        if key not in keys or busy.get(s, 0) > close:
            continue
        sl, tp, hh = GEO.get(key, (3.0, 1.0, HOLDS.get(key, 48)))
        if (close, s) in wide and key not in SHORT_KEYS:
            tp, hh = tp_w, hold_w
        H = min(hh * 4, len(Mu))
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


KS = (5, 7, 10)
EXITS = ((1.25, 48), (1.5, 48), (2.0, 48), (1.5, 72), (2.0, 72))

for bi, (book_name, coins) in enumerate(BOOKS):
    sig = make_sig_raw(coins)
    rows, _ = book(sig, BASE_KEYS)
    cnt = collections.Counter(x[0] for x in rows)
    long_rows = [(x[0], x[4]) for x in rows]
    key_at = {(x[0], x[3]): x[2] for x in sig}
    long_rows = [k for k in long_rows if key_at.get(k) not in SHORT_KEYS]
    beq, bdd = fix2(rows)
    bm = sim_w.money_at_dd(rows, 0.12)[1]
    bpy = pyr(rows)
    print('', flush=True)
    print('  ===== %s: система $%.0f просадка %.1f%% DD12 $%.0f, лонгов %d =====' % (book_name, beq, 100 * bdd, bm, len(long_rows)), flush=True)
    RES = {'система (сейчас)': rows}
    for K in KS:
        wide = {k for k in long_rows if cnt[k[0]] >= K}
        rest = [k for k in long_rows if cnt[k[0]] < K]
        for tp_w, hold_w in EXITS:
            rr = book_t(sig, BASE_KEYS, wide, tp_w, hold_w)
            eq, dd = fix2(rr)
            m = sim_w.money_at_dd(rr, 0.12)[1]
            py = pyr(rr)
            v = np.array([x[2] for x in rr if (x[0], x[4]) in wide])
            lbl = 'n>=%d тейк %.2f %dч' % (K, tp_w, hold_w)
            RES[lbl] = rr
            ce, cm = [], []
            for seed in range(5):
                rng = np.random.default_rng(seed)
                pick = {rest[i] for i in rng.choice(len(rest), size=min(len(wide), len(rest)), replace=False)}
                cr = book_t(sig, BASE_KEYS, pick, tp_w, hold_w)
                ce.append(fix2(cr)[0]); cm.append(sim_w.money_at_dd(cr, 0.12)[1])
            print('    %-24s пачечных %4d ВР %5.1f%% ср %+.3f | $%6.0f %4.1f%% | DD12 $%6.0f %s | лучше по годам %d/5 (%s) | КОНТРОЛЬ случайные $%.0f DD12 $%.0f'
                  % (lbl, len(v), 100 * np.mean(v > 0) if len(v) else 0, v.mean() if len(v) else 0, eq, 100 * dd, m,
                     '*' if (eq > beq and m > bm) else ' ', sum(1 for a, b in zip(bpy, py) if b > a),
                     ' '.join('%+.0f' % (b - a) for a, b in zip(bpy, py)), np.median(ce), np.median(cm)), flush=True)
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
            mb = sim_w.simulate([x for x in rows if lo <= x[0] < hi], 0.014)['eq']
            wins += int(mp > mb)
            print('      %d выбрано «%s» | $%.0f против $%.0f %s'
                  % (y, bk, mp, mb, 'лучше' if mp > mb else ('так же' if mp == mb else 'хуже')), flush=True)
        print('      лучше нынешней системы в %d из 5 лет' % wins, flush=True)
'''

src = head + TAIL
ast.parse(src)
io.open('bursttake.py', 'w', encoding='utf-8').write(src)
print('bursttake.py готов, синтаксис ок')
