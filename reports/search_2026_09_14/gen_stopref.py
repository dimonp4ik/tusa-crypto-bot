"""Writes stopref.py: the live stop_ref knob, re-measured on the system with items 8-10.

stop_ref = 0.0394 shrinks a position when its stop is wider than that fraction of price
(pullback_live.py:254-255: margin *= stop_ref / sl_frac). It is the one live size rule never
re-measured since the boost (item 8) and the breadth weight (item 10) were added - and it pulls the
opposite way to them: the boost enlarges, stop_ref shrinks. If the current value is off, it is silently
taxing exactly the wide-dip trades, whose stops are the widest.

Values 0.030 / 0.035 / 0.0394 (live) / 0.045 / 0.060 / off, on the final system. Three books, full fill
at 1.4% and at 12% drawdown, then ten paired 85% draws for the best two against the live value, and the
rolling choice by money at 12% drawdown. A wider stop_ref is a size increase, so the equal-drawdown
measure decides, not the money at fixed risk.
"""
import ast
import io

s = io.open('srcweight.py', encoding='utf-8').read()
head = s[:s.index("VARS = (")]

TAIL = r'''
REFS = ((0.030, '0.030'), (0.035, '0.035'), (0.0394, '0.0394 (живой)'), (0.045, '0.045'),
        (0.060, '0.060'), (None, 'выключен'))


def with_ref(fn, ref):
    old = sim_w.REF
    try:
        sim_w.REF = ref if ref is not None else 10.0
        return fn()
    finally:
        sim_w.REF = old


for book_name, coins in BOOKS3:
    H = {s: coin_hours(s) for s in coins}
    base_sig = make_sig_raw(coins)
    extra = roll_entries(coins, H, 'r2_4', 4, 2, 4, 0)
    sig_all = sorted(base_sig + extra, key=lambda x: (x[0], x[1]))
    GEO.clear(); GEO.update(BASE_GEO)
    rows, keys = book_keyed(sig_all, FULL)
    fin = weigh_src(rows, keys, {'r2_4'}, 1.25)
    wide = [x for x, k in zip(rows, keys) if k == 'r2_4']
    sf_all = np.array([x[3] for x in rows]); sf_w = np.array([x[3] for x in wide]) if wide else np.zeros(1)
    print('', flush=True)
    print('  ===== %s: доля сделок с широким стопом (sf > 0.0394): вся книга %.0f%%, сделки ширины %.0f%% ====='
          % (book_name, 100 * np.mean(sf_all > 0.0394), 100 * np.mean(sf_w > 0.0394)), flush=True)
    RES = {}
    for ref, lbl in REFS:
        eq, dd = with_ref(lambda: fix2(fin), ref)
        m = with_ref(lambda: sim_w.money_at_dd(fin, 0.12)[1], ref)
        RES[lbl] = (ref, eq, m)
        print('    stop_ref %-14s $%6.0f %4.1f%% | DD12 $%6.0f' % (lbl, eq, 100 * dd, m), flush=True)
    base_ref = 0.0394
    top = sorted([l for l in RES if RES[l][0] != base_ref], key=lambda l: -RES[l][2])[:2]
    print('    ЗАЛИВКА 85%% против живого 0.0394: %s' % ', '.join(top), flush=True)
    subs = []
    for seed in range(1, 11):
        rng = np.random.default_rng(seed)
        sub = [x for x in sig_all if rng.random() < 0.85]
        r0, k0 = book_keyed(sub, FULL)
        subs.append(weigh_src(r0, k0, {'r2_4'}, 1.25))
    A = [with_ref(lambda: (fix2(r)[0], fix2(r)[1], sim_w.money_at_dd(r, 0.12)[1], pyr(r)), base_ref) for r in subs]
    for lbl in top:
        ref = RES[lbl][0]
        B = [with_ref(lambda: (fix2(r)[0], fix2(r)[1], sim_w.money_at_dd(r, 0.12)[1], pyr(r)), ref) for r in subs]
        med = lambda X, i: float(np.median([x[i] for x in X]))
        yw = ' '.join('%d/10' % sum(1 for x, y in zip(A, B) if y[3][j] > x[3][j]) for j in range(5))
        print('      %-14s живой $%.0f %.1f%% DD12 $%.0f | вариант $%.0f %.1f%% (макс %.1f%%) DD12 $%.0f | лучше $ %d/10, DD12 %d/10 | годы %s'
              % (lbl, med(A, 0), 100 * med(A, 1), med(A, 2), med(B, 0), 100 * med(B, 1), 100 * max(x[1] for x in B), med(B, 2),
                 sum(1 for x, y in zip(A, B) if y[0] > x[0]), sum(1 for x, y in zip(A, B) if y[2] > x[2]), yw), flush=True)
    print('    СКОЛЬЗЯЩИЙ ВЫБОР: значение по $ при DD12 на 4 годах, замер 5-го при 1.4% против живого', flush=True)
    wins = 0
    for y in range(2022, 2027):
        lo, hi = YT[y], YT[y + 1]
        tr = [x for x in fin if not (lo <= x[0] < hi)]
        te = [x for x in fin if lo <= x[0] < hi]
        best, bk = -1, None
        for lbl, (ref, _, _) in RES.items():
            mm = with_ref(lambda: sim_w.money_at_dd(tr, 0.12)[1], ref)
            if mm > best:
                best, bk = mm, lbl
        mp = with_ref(lambda: sim_w.simulate(te, 0.014)['eq'], RES[bk][0])
        mb = with_ref(lambda: sim_w.simulate(te, 0.014)['eq'], base_ref)
        wins += int(mp > mb)
        print('      %d выбрано «%s» | $%.0f против $%.0f %s' % (y, bk, mp, mb, 'лучше' if mp > mb else ('так же' if mp == mb else 'хуже')), flush=True)
    print('      лучше живого значения в %d из 5 лет' % wins, flush=True)
'''

src = head + TAIL
ast.parse(src)
io.open('stopref.py', 'w', encoding='utf-8').write(src)
print('stopref.py готов, синтаксис ок')
