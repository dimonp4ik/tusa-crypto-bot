"""Writes sidediag.py: are short bursts as good as long bursts, and should the boost count sides?

Proposal item 8 counts every trade opening in the same bar - longs and shorts together - and boosts all
of them at n >= 10. The burst evidence came from market-wide pullbacks, a long-side mechanism; the
short rules fire in the opposite BTC regime and never overlap with the pullback breadth. Short bursts
(many coins popping inside a downtrend at once) may behave differently.

1. Diagnosis on the accepted system per book and per side: trades grouped by how many trades of the SAME
   side opened at that timestamp (1, 2-3, 4-6, 7+, 10+): share, win rate, mean R, per-year mean R.
2. Variants against the accepted rule (n over all trades, boost all):
   - boost longs only (n over all trades)
   - n over longs only, boost longs only
   - n over same side, boost both sides
   Full fill both measures, and ten paired 85% fills.
"""
import ast
import io

s = io.open('combo.py', encoding='utf-8').read()
head = s[:s.index("VARS = (")]

TAIL = r'''
ACC = BASE_KEYS | {'w50'}
SHORT_KEYS = {'r%d' % k for k, r in enumerate(RULES) if r.get('side', 'LONG') == 'SHORT'}


def book_side(sig, keys):
    """book() with the side of every row kept alongside."""
    busy, out = {}, []
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
        out.append(((close, end, (x * atr / e - slip - FEE) / sf, sf, s, 1.0), key in SHORT_KEYS))
        busy[s] = end
    out.sort(key=lambda z: z[0])
    return [z[0] for z in out], [z[1] for z in out]


def weigh(rows, short, mode):
    all_n = collections.Counter(x[0] for x in rows)
    long_n = collections.Counter(x[0] for x, sh in zip(rows, short) if not sh)
    short_n = collections.Counter(x[0] for x, sh in zip(rows, short) if sh)
    out = []
    for x, sh in zip(rows, short):
        if mode == 'all':
            w = 1.25 if all_n[x[0]] >= 10 else 1.0
        elif mode == 'longs_only':
            w = 1.25 if (not sh and all_n[x[0]] >= 10) else 1.0
        elif mode == 'long_count':
            w = 1.25 if (not sh and long_n[x[0]] >= 10) else 1.0
        elif mode == 'same_side':
            w = 1.25 if ((short_n if sh else long_n)[x[0]] >= 10) else 1.0
        else:
            w = 1.0
        out.append(x[:5] + (w,))
    return out


def grp(n):
    return '1' if n == 1 else '2-3' if n <= 3 else '4-6' if n <= 6 else '7-9' if n <= 9 else '10+'


MODES = (('ПРИНЯТОЕ: n по всем, буст всем', 'all'), ('буст только лонгам (n по всем)', 'longs_only'),
         ('n по лонгам, буст лонгам', 'long_count'), ('n по своей стороне, буст обеим', 'same_side'))

for book_name, coins in BOOKS3:
    H = {s: coin_hours(s) for s in coins}
    base_sig = make_sig_raw(coins)
    extra = wide_entries(coins, H, 'w50', 4, 50.0, 45.0, 4, 0)
    sig_all = sorted(base_sig + extra, key=lambda x: (x[0], x[1]))
    rows, short = book_side(sig_all, ACC)
    print('', flush=True)
    print('  ===== %s: сделок %d (шортов %d) =====' % (book_name, len(rows), sum(short)), flush=True)
    for side_name, want in (('ЛОНГИ', False), ('ШОРТЫ', True)):
        same_n = collections.Counter(x[0] for x, sh in zip(rows, short) if sh == want)
        idx = [i for i, sh in enumerate(short) if sh == want]
        print('    %s: пачка своей стороны   доля    ВР     ср R    по годам' % side_name, flush=True)
        for g in ('1', '2-3', '4-6', '7-9', '10+'):
            ii = [i for i in idx if grp(same_n[rows[i][0]]) == g]
            if not ii:
                continue
            v = np.array([rows[i][2] for i in ii])
            yrs = []
            for y in range(2022, 2027):
                vy = [rows[i][2] for i in ii if YT[y] <= rows[i][0] < YT[y + 1]]
                yrs.append('%+.3f' % np.mean(vy) if len(vy) >= 8 else '  -   ')
            print('      %-5s                   %5.1f%% %5.1f%% %+.4f  %s' % (g, 100 * len(ii) / len(idx), 100 * np.mean(v > 0), v.mean(), ' '.join(yrs)), flush=True)
    res = {}
    for lbl, mode in MODES:
        wr = weigh(rows, short, mode)
        eq, dd = fix2(wr)
        m = sim_w.money_at_dd(wr, 0.12)[1]
        nb = sum(1 for x in wr if x[5] > 1)
        res[mode] = (eq, m)
        print('    100%% %-34s буст на %4d сделках | $%6.0f %4.1f%% DD12 $%6.0f' % (lbl, nb, eq, 100 * dd, m), flush=True)
    print('    ЗАЛИВКА 85% против ПРИНЯТОГО, 10 парных розыгрышей', flush=True)
    for lbl, mode in MODES[1:]:
        A, B = [], []
        for seed in range(1, 11):
            rng = np.random.default_rng(seed)
            sub = [x for x in sig_all if rng.random() < 0.85]
            r, sh = book_side(sub, ACC)
            r0, r1 = weigh(r, sh, 'all'), weigh(r, sh, mode)
            e0, d0 = fix2(r0); e1, d1 = fix2(r1)
            A.append((e0, d0, sim_w.money_at_dd(r0, 0.12)[1])); B.append((e1, d1, sim_w.money_at_dd(r1, 0.12)[1]))
        med = lambda X, i: float(np.median([x[i] for x in X]))
        print('      %-34s принятое $%.0f DD12 $%.0f | вариант $%.0f %.1f%% DD12 $%.0f | лучше $ %d/10, DD12 %d/10'
              % (lbl, med(A, 0), med(A, 2), med(B, 0), 100 * med(B, 1), med(B, 2),
                 sum(1 for x, y in zip(A, B) if y[0] > x[0]), sum(1 for x, y in zip(A, B) if y[2] > x[2])), flush=True)
'''

src = head + TAIL
ast.parse(src)
io.open('sidediag.py', 'w', encoding='utf-8').write(src)
print('sidediag.py готов, синтаксис ок')
