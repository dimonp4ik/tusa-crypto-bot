"""Writes entryslip.py: how much a worse entry price costs the breadth source.

Every number so far enters at the next 15m open. The live bot does something else: it posts a watch and
enters when price touches the level inside the hour. In this project a worse live entry has already
eaten most of the profit once (stocks: 0.374% worse entry took +179R down to +37R), so the question is
not academic. The old sources were checked; the breadth source (item 9) was not - and its stops are the
widest in the book, which cuts both ways: a wide stop dilutes a fixed price slip.

A slip of d in price costs d/sf in R, where sf is the stop as a fraction of price. So the R of every
breadth trade is reduced by d/sf for d = 0.1%, 0.25%, 0.5%, and the system is re-simulated. Control: the
same penalty applied to the same number of randomly chosen non-breadth trades (median of 5 draws) - it
separates "this source is fragile to entry quality" from "any slip hurts".

Three books, money at 1.4% and at 12% drawdown, plus the same for the boost half alone as a reference.
"""
import ast
import io

s = io.open('widestopref.py', encoding='utf-8').read()
head = s[:s.index("for bi, (book_name, coins) in enumerate(BOOKS3):")]

TAIL = r'''
WIDE_REF = 0.060
SLIPS = (0.001, 0.0025, 0.005)


def penalise(rows, mask, d):
    """R -= d / sf для сделок, отмеченных маской."""
    out = []
    for x, m in zip(rows, mask):
        out.append(x[:2] + ((x[2] - d / x[3]) if m else x[2],) + x[3:])
    return out


for book_name, coins in BOOKS3:
    H = {s: coin_hours(s) for s in coins}
    base_sig = make_sig_raw(coins)
    extra = roll_entries(coins, H, 'r2_4', 4, 2, 4, 0)
    sig_all = sorted(base_sig + extra, key=lambda x: (x[0], x[1]))
    GEO.clear(); GEO.update(BASE_GEO)
    rows, keys = book_keyed(sig_all, FULL)
    fin = weigh_src(rows, keys, {'r2_4'}, 1.25)
    refs = [(WIDE_REF if k == 'r2_4' else sim_w.REF) for k in keys]
    is_wide = [k == 'r2_4' for k in keys]
    n_wide = sum(is_wide)
    sf_w = np.array([x[3] for x, m in zip(fin, is_wide) if m])
    sf_all = np.array([x[3] for x in fin])
    base = sim_ref(fin, 0.014, refs)
    bm = money_at_dd_ref(fin, refs)
    print('', flush=True)
    print('  ===== %s: сделок ширины %d из %d | медианный стоп: ширина %.1f%%, вся книга %.1f%% ====='
          % (book_name, n_wide, len(fin), 100 * np.median(sf_w), 100 * np.median(sf_all)), flush=True)
    print('    без ухудшения: $%.0f %.1f%% DD12 $%.0f' % (base['eq'], 100 * abs(L.dd_of(base['curve'])), bm), flush=True)
    for d in SLIPS:
        pen = penalise(fin, is_wide, d)
        r = sim_ref(pen, 0.014, refs)
        m = money_at_dd_ref(pen, refs)
        ce, cm = [], []
        idx_other = [i for i, m_ in enumerate(is_wide) if not m_]
        for seed in range(5):
            rng = np.random.default_rng(seed)
            pick = set(rng.choice(idx_other, size=min(n_wide, len(idx_other)), replace=False).tolist())
            cmask = [i in pick for i in range(len(fin))]
            cr = penalise(fin, cmask, d)
            ce.append(sim_ref(cr, 0.014, refs)['eq'])
            cm.append(money_at_dd_ref(cr, refs))
        print('    вход хуже на %.2f%%: $%6.0f (%+5.1f%%) %4.1f%% DD12 $%6.0f (%+5.1f%%) | КОНТРОЛЬ те же %d случайных сделок: $%.0f (%+.1f%%) DD12 $%.0f'
              % (100 * d, r['eq'], 100 * (r['eq'] / base['eq'] - 1), 100 * abs(L.dd_of(r['curve'])), m,
                 100 * (m / bm - 1), n_wide, np.median(ce), 100 * (np.median(ce) / base['eq'] - 1), np.median(cm)), flush=True)
    print('    средняя потеря R на сделку источника: %s'
          % ', '.join('%.2f%% -> %.3fR' % (100 * d, d / np.median(sf_w)) for d in SLIPS), flush=True)
'''

src = head + TAIL
ast.parse(src)
io.open('entryslip.py', 'w', encoding='utf-8').write(src)
print('entryslip.py готов, синтаксис ок')
