"""Writes widestopref.py: should the breadth trades keep their own stop_ref?

stopref.py confirmed the live value 0.0394 for the book as a whole, and exposed a tension: the stop of a
breadth trade (item 9) is wider than the threshold in 76-85% of cases against 45% for the whole book. So
item 10 raises their size by 1.25 and stop_ref immediately cuts most of it back. The two rules pull
against each other on exactly the same trades.

Variants for the breadth source only (everything else keeps 0.0394): 0.0394 (as today), 0.045, 0.050,
0.060, and no stop_ref at all for that source. The simulator is a copy of sim_w.simulate that takes the
threshold per trade, with a self-test: with the same threshold everywhere it must reproduce sim_w to the
cent. Three books, full fill at 1.4% and at 12% drawdown, ten paired 85% draws for the best two, and
the rolling choice by money at 12% drawdown - a looser threshold is a size increase, so equal drawdown
decides.
"""
import ast
import io

s = io.open('srcweight.py', encoding='utf-8').read()
head = s[:s.index("VARS = (")]

TAIL = r'''
REFS = ((0.0394, 'как сейчас 0.0394'), (0.045, 'ширине 0.045'), (0.050, 'ширине 0.050'),
        (0.060, 'ширине 0.060'), (None, 'ширине без stop_ref'))


def sim_ref(trades, target, refs):
    """sim_w.simulate with a per-trade stop_ref (refs: list of thresholds, None = no shrink)."""
    eq = peak = sim_w.DEPOSIT
    open_pos = []
    day, day_start, day_paused, paused, paused_at = None, sim_w.DEPOSIT, False, False, None
    curve = []
    for (a, b, R, sf, s, w), ref in zip(trades, refs):
        while open_pos and open_pos[0][0] <= a:
            t_close, m, mpr, rr = open_pos.pop(0)
            eq += mpr * rr; peak = max(peak, eq); curve.append((t_close, eq))
        d = L.day_of(a)
        if d != day:
            day, day_start, day_paused = d, eq, False
        if not paused and eq <= peak * (1 - sim_w.MAX_DD):
            paused, paused_at = True, a
        if not day_paused and eq <= day_start * (1 - sim_w.MAX_DAILY):
            day_paused = True
        if paused or day_paused:
            continue
        margin = (target / (sim_w.LEV * sim_w.REF)) * eq
        if ref is not None and sf > ref:
            margin *= ref / sf
        margin *= w
        if sum(p[1] for p in open_pos) + margin > eq * sim_w.USABLE:
            continue
        open_pos.append((b, margin, margin * sim_w.LEV * sf, R))
        open_pos.sort(key=lambda p: p[0])
    for t_close, m, mpr, rr in open_pos:
        eq += mpr * rr; curve.append((t_close, eq))
    curve.sort()
    return dict(eq=eq, curve=curve, paused_at=paused_at)


def money_at_dd_ref(rows, refs, want=0.12):
    lo, hi = 0.0005, 0.030
    for _ in range(22):
        mid = (lo + hi) / 2
        r = sim_ref(rows, mid, refs)
        if bool(r['paused_at']) or abs(L.dd_of(r['curve'])) > want:
            hi = mid
        else:
            lo = mid
    return sim_ref(rows, lo, refs)['eq']


for bi, (book_name, coins) in enumerate(BOOKS3):
    H = {s: coin_hours(s) for s in coins}
    base_sig = make_sig_raw(coins)
    extra = roll_entries(coins, H, 'r2_4', 4, 2, 4, 0)
    sig_all = sorted(base_sig + extra, key=lambda x: (x[0], x[1]))
    GEO.clear(); GEO.update(BASE_GEO)
    rows, keys = book_keyed(sig_all, FULL)
    fin = weigh_src(rows, keys, {'r2_4'}, 1.25)
    if bi == 0:
        mine = sim_ref(fin, 0.014, [sim_w.REF] * len(fin))['eq']
        ref0 = sim_w.simulate(fin, 0.014)['eq']
        print('  САМОПРОВЕРКА: копия симулятора $%.2f против sim_w $%.2f %s'
              % (mine, ref0, 'совпало' if abs(mine - ref0) < 0.01 else 'РАСХОЖДЕНИЕ'), flush=True)
    base_refs = [sim_w.REF] * len(fin)
    beq = sim_ref(fin, 0.014, base_refs)
    bdd = abs(L.dd_of(beq['curve']))
    bm = money_at_dd_ref(fin, base_refs)
    print('', flush=True)
    print('  ===== %s: сейчас $%.0f %.1f%% DD12 $%.0f =====' % (book_name, beq['eq'], 100 * bdd, bm), flush=True)
    RES = {}
    for ref, lbl in REFS:
        refs = [(ref if k == 'r2_4' else sim_w.REF) for k in keys]
        r = sim_ref(fin, 0.014, refs)
        dd = abs(L.dd_of(r['curve']))
        m = money_at_dd_ref(fin, refs)
        RES[lbl] = (ref, r['eq'], m)
        print('    %-22s $%6.0f %4.1f%% DD12 $%6.0f %s' % (lbl, r['eq'], 100 * dd, m, '+' if (r['eq'] > beq['eq'] and m > bm) else ' '), flush=True)
    top = sorted([l for l in RES if RES[l][0] != 0.0394], key=lambda l: -RES[l][2])[:2]
    print('    ЗАЛИВКА 85%% против нынешнего: %s' % ', '.join(top), flush=True)
    subs = []
    for seed in range(1, 11):
        rng = np.random.default_rng(seed)
        sub = [x for x in sig_all if rng.random() < 0.85]
        r0, k0 = book_keyed(sub, FULL)
        subs.append((weigh_src(r0, k0, {'r2_4'}, 1.25), k0))
    A = []
    for rws, kk in subs:
        rr = [sim_w.REF] * len(rws)
        r = sim_ref(rws, 0.014, rr)
        A.append((r['eq'], abs(L.dd_of(r['curve'])), money_at_dd_ref(rws, rr)))
    for lbl in top:
        ref = RES[lbl][0]
        B = []
        for rws, kk in subs:
            rr = [(ref if k == 'r2_4' else sim_w.REF) for k in kk]
            r = sim_ref(rws, 0.014, rr)
            B.append((r['eq'], abs(L.dd_of(r['curve'])), money_at_dd_ref(rws, rr)))
        med = lambda X, i: float(np.median([x[i] for x in X]))
        print('      %-22s сейчас $%.0f %.1f%% DD12 $%.0f | вариант $%.0f %.1f%% (макс %.1f%%) DD12 $%.0f | лучше $ %d/10, DD12 %d/10'
              % (lbl, med(A, 0), 100 * med(A, 1), med(A, 2), med(B, 0), 100 * med(B, 1), 100 * max(x[1] for x in B), med(B, 2),
                 sum(1 for x, y in zip(A, B) if y[0] > x[0]), sum(1 for x, y in zip(A, B) if y[2] > x[2])), flush=True)
    print('    СКОЛЬЗЯЩИЙ ВЫБОР: значение по $ при DD12 на 4 годах, замер 5-го при 1.4% против нынешнего', flush=True)
    wins = 0
    for y in range(2022, 2027):
        lo, hi = YT[y], YT[y + 1]
        idx_tr = [i for i, x in enumerate(fin) if not (lo <= x[0] < hi)]
        idx_te = [i for i, x in enumerate(fin) if lo <= x[0] < hi]
        tr = [fin[i] for i in idx_tr]; te = [fin[i] for i in idx_te]
        best, bk = -1, None
        for lbl, (ref, _, _) in RES.items():
            rr = [(ref if keys[i] == 'r2_4' else sim_w.REF) for i in idx_tr]
            m = money_at_dd_ref(tr, rr)
            if m > best:
                best, bk = m, lbl
        refb = RES[bk][0]
        mp = sim_ref(te, 0.014, [(refb if keys[i] == 'r2_4' else sim_w.REF) for i in idx_te])['eq']
        mb = sim_ref(te, 0.014, [sim_w.REF] * len(idx_te))['eq']
        wins += int(mp > mb)
        print('      %d выбрано «%s» | $%.0f против $%.0f %s' % (y, bk, mp, mb, 'лучше' if mp > mb else ('так же' if mp == mb else 'хуже')), flush=True)
    print('      лучше нынешнего в %d из 5 лет' % wins, flush=True)
'''

src = head + TAIL
ast.parse(src)
io.open('widestopref.py', 'w', encoding='utf-8').write(src)
print('widestopref.py готов, синтаксис ок')
