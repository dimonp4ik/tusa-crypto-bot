"""Writes srcweight.py: size by source quality, and the holding window of the shorts.

Two loose ends on the final system (boost x1.25 at n >= 10 + breadth over 2 hours):

1. The breadth source's own trades win 88-92% at +0.19..+0.24R against 80% and +0.10R for the book as a
   whole - the best-quality source in the system. The boost sizes trades by how many opened together,
   not by which rule opened them. Test an extra multiplier on the breadth trades (x1.25, x1.5) and, for
   comparison, on the pullback trades (x1.15). Control: the same multiplier given to a random set of
   trades of the same count (median of 5 draws) - it separates "this source deserves size" from "more
   size anywhere".
2. The shorts hold 48h like everything else; that window was never measured on the current system.
   Sweep 24 / 36 / 48 / 72h for both short rules at once, geometry unchanged.

Three books, full fill, money at 1.4% and at 12% drawdown, per-year against the final system; ten paired
85% fills for the best two variants; rolling choice by money at 12% drawdown.
"""
import ast
import io

s = io.open('rollbreadth.py', encoding='utf-8').read()
head = s[:s.index("CELLS = [")]

TAIL = r'''
FULL = BASE_KEYS | {'r2_4'}
SHORT_KEYS = ['r%d' % k for k, r in enumerate(RULES) if r.get('side', 'LONG') == 'SHORT']
BASE_GEO = dict(GEO)
PULL_KEYS = {'o1h', 'o2h'}


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
        out.append(((close, end, (x * atr / e - slip - FEE) / sf, sf, s, 1.0), key))
        busy[s] = end
    out.sort(key=lambda z: z[0])
    return [z[0] for z in out], [z[1] for z in out]


def weigh_src(rows, keys, want, mult):
    cnt = collections.Counter(x[0] for x in rows)
    out = []
    for x, k in zip(rows, keys):
        w = 1.25 if cnt[x[0]] >= 10 else 1.0
        if k in want:
            w *= mult
        out.append(x[:5] + (w,))
    return out


def weigh_rand(rows, n_target, mult, seed):
    rng = np.random.default_rng(seed)
    idx = set(rng.choice(len(rows), size=min(n_target, len(rows)), replace=False).tolist())
    cnt = collections.Counter(x[0] for x in rows)
    return [x[:5] + ((1.25 if cnt[x[0]] >= 10 else 1.0) * (mult if i in idx else 1.0),) for i, x in enumerate(rows)]


VARS = (('вес ширины x1.25', {'r2_4'}, 1.25), ('вес ширины x1.50', {'r2_4'}, 1.5),
        ('вес откатов x1.15', PULL_KEYS, 1.15), ('вес ширины и откатов x1.25', PULL_KEYS | {'r2_4'}, 1.25))
HOLDS_SHORT = (24, 36, 48, 72)

for book_name, coins in BOOKS3:
    H = {s: coin_hours(s) for s in coins}
    base_sig = make_sig_raw(coins)
    extra = roll_entries(coins, H, 'r2_4', 4, 2, 4, 0)
    sig_all = sorted(base_sig + extra, key=lambda x: (x[0], x[1]))
    GEO.clear(); GEO.update(BASE_GEO)
    rows, keys = book_keyed(sig_all, FULL)
    full = weigh_src(rows, keys, set(), 1.0)
    feq, fdd = fix2(full); fm = sim_w.money_at_dd(full, 0.12)[1]; fpy = pyr(full)
    print('', flush=True)
    print('  ===== %s: ИТОГОВАЯ $%.0f %.1f%% DD12 $%.0f, сделок %d =====' % (book_name, feq, 100 * fdd, fm, len(rows)), flush=True)
    RES = {'итоговая': full}
    print('    ВЕС ПО ИСТОЧНИКУ:', flush=True)
    for lbl, want, mult in VARS:
        wr = weigh_src(rows, keys, want, mult)
        n_t = sum(1 for k in keys if k in want)
        eq, dd = fix2(wr)
        m = sim_w.money_at_dd(wr, 0.12)[1]
        py = pyr(wr)
        RES[lbl] = wr
        ce, cm = [], []
        for seed in range(5):
            cr = weigh_rand(rows, n_t, mult, seed)
            ce.append(fix2(cr)[0]); cm.append(sim_w.money_at_dd(cr, 0.12)[1])
        print('      %-28s на %4d сделках | $%6.0f %4.1f%% DD12 $%6.0f %s | по годам %d/5 (%s) | КОНТРОЛЬ случайные $%.0f DD12 $%.0f'
              % (lbl, n_t, eq, 100 * dd, m, '+' if (eq > feq and m > fm) else ' ',
                 sum(1 for a, b in zip(fpy, py) if b > a), ' '.join('%+.0f' % (b - a) for a, b in zip(fpy, py)),
                 np.median(ce), np.median(cm)), flush=True)
    print('    УДЕРЖАНИЕ ШОРТОВ:', flush=True)
    for hh in HOLDS_SHORT:
        GEO.clear(); GEO.update(BASE_GEO)
        for k in SHORT_KEYS:
            sl, tp, _ = BASE_GEO.get(k, (3.0, 1.0, 48))
            GEO[k] = (sl, tp, hh)
        r2, k2 = book_keyed(sig_all, FULL)
        wr = weigh_src(r2, k2, set(), 1.0)
        eq, dd = fix2(wr)
        m = sim_w.money_at_dd(wr, 0.12)[1]
        py = pyr(wr)
        RES['удержание шортов %dч' % hh] = wr
        print('      %-28s сделок %4d | $%6.0f %4.1f%% DD12 $%6.0f %s | по годам %d/5 (%s)'
              % ('%dч' % hh, len(r2), eq, 100 * dd, m, '+' if (eq > feq and m > fm) else ' ',
                 sum(1 for a, b in zip(fpy, py) if b > a), ' '.join('%+.0f' % (b - a) for a, b in zip(fpy, py))), flush=True)
    GEO.clear(); GEO.update(BASE_GEO)
    cand = [k for k in RES if k != 'итоговая']
    top = sorted(cand, key=lambda k: -sim_w.money_at_dd(RES[k], 0.12)[1])[:2]
    print('    ЗАЛИВКА 85%% против ИТОГОВОЙ: %s' % ', '.join(top), flush=True)
    for lbl in top:
        A, B = [], []
        for seed in range(1, 11):
            rng = np.random.default_rng(seed)
            sub = [x for x in sig_all if rng.random() < 0.85]
            GEO.clear(); GEO.update(BASE_GEO)
            r0, k0 = book_keyed(sub, FULL)
            a0 = weigh_src(r0, k0, set(), 1.0)
            if lbl.startswith('удержание'):
                hh = int(lbl.split()[2][:-1])
                for k in SHORT_KEYS:
                    sl, tp, _ = BASE_GEO.get(k, (3.0, 1.0, 48))
                    GEO[k] = (sl, tp, hh)
                r1, k1 = book_keyed(sub, FULL)
                b1 = weigh_src(r1, k1, set(), 1.0)
            else:
                want, mult = next((w, m) for l, w, m in VARS if l == lbl)
                b1 = weigh_src(r0, k0, want, mult)
            e0, d0 = fix2(a0); e1, d1 = fix2(b1)
            A.append((e0, d0, sim_w.money_at_dd(a0, 0.12)[1], pyr(a0))); B.append((e1, d1, sim_w.money_at_dd(b1, 0.12)[1], pyr(b1)))
        GEO.clear(); GEO.update(BASE_GEO)
        med = lambda X, i: float(np.median([x[i] for x in X]))
        yw = ' '.join('%d/10' % sum(1 for x, y in zip(A, B) if y[3][j] > x[3][j]) for j in range(5))
        print('      %-28s итоговая $%.0f %.1f%% DD12 $%.0f | вариант $%.0f %.1f%% (макс %.1f%%) DD12 $%.0f | лучше $ %d/10, DD12 %d/10 | годы %s'
              % (lbl, med(A, 0), 100 * med(A, 1), med(A, 2), med(B, 0), 100 * med(B, 1), 100 * max(x[1] for x in B), med(B, 2),
                 sum(1 for x, y in zip(A, B) if y[0] > x[0]), sum(1 for x, y in zip(A, B) if y[2] > x[2]), yw), flush=True)
    print('    СКОЛЬЗЯЩИЙ ВЫБОР: по $ при DD12 на 4 годах, замер 5-го при 1.4% против ИТОГОВОЙ', flush=True)
    wins = 0
    for y in range(2022, 2027):
        lo, hi = YT[y], YT[y + 1]
        best, bk = -1, None
        for lbl, rr in RES.items():
            mm = sim_w.money_at_dd([x for x in rr if not (lo <= x[0] < hi)], 0.12)[1]
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
io.open('srcweight.py', 'w', encoding='utf-8').write(src)
print('srcweight.py готов, синтаксис ок')
