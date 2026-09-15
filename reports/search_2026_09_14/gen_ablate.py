"""Writes ablate.py: drop each source of the final system in turn.

ddattr.py named two suspects on book 1: the short rules decay year by year (mean R +0.273 -> +0.070,
stops 16.7% -> 28.5%) and the bank rule coin_run_evening is weak in every year (-0.03..+0.09R at
15-24% stops). An ablation answers it properly: remove one source, keep everything else, and read both
measures.

Sources: each bank rule separately, the pullback on 1h, the pullback on 2h, the breadth source, and all
short rules together. The final system is the system + boost (n >= 10, counted after the removal) +
breadth over 2 hours. Three books, full fill; ten paired 85% fills for the two most promising removals;
per-year versus the full system; rolling choice by money at 12% drawdown.
"""
import ast
import io

s = io.open('rollbreadth.py', encoding='utf-8').read()
head = s[:s.index("CELLS = [")]

TAIL = r'''
FULL = BASE_KEYS | {'r2_4'}
SHORT_KEYS = {'r%d' % k for k, r in enumerate(RULES) if r.get('side', 'LONG') == 'SHORT'}
NAMES = {}
for k in sorted(BASE_KEYS):
    if k.startswith('r') and k[1:].isdigit():
        NAMES[k] = RULES[int(k[1:])].get('name', k)
NAMES.update({'o1h': 'откат 1ч', 'o2h': 'откат 2ч', 'r2_4': 'ширина 2ч'})

DROPS = [('без %s' % NAMES[k], {k}) for k in sorted(NAMES) if k in BASE_KEYS or k == 'r2_4']
DROPS.append(('без всех шортов', set(SHORT_KEYS)))


def run(sig_all, keys):
    return boost(book(sig_all, keys)[0])


for book_name, coins in BOOKS3:
    H = {s: coin_hours(s) for s in coins}
    base_sig = make_sig_raw(coins)
    extra = roll_entries(coins, H, 'r2_4', 4, 2, 4, 0)
    sig_all = sorted(base_sig + extra, key=lambda x: (x[0], x[1]))
    full = run(sig_all, FULL)
    feq, fdd = fix2(full); fm = sim_w.money_at_dd(full, 0.12)[1]; fpy = pyr(full)
    print('', flush=True)
    print('  ===== %s: ИТОГОВАЯ $%.0f %.1f%% DD12 $%.0f, сделок %d =====' % (book_name, feq, 100 * fdd, fm, len(full)), flush=True)
    RES = {'итоговая': full}
    for lbl, drop in DROPS:
        keys = FULL - drop
        rows = run(sig_all, keys)
        if not rows:
            continue
        eq, dd = fix2(rows)
        m = sim_w.money_at_dd(rows, 0.12)[1]
        py = pyr(rows)
        RES[lbl] = rows
        print('    %-26s сделок %4d (%+4d) | $%6.0f %4.1f%% DD12 $%6.0f %s | по годам против итоговой %d/5 (%s)'
              % (lbl, len(rows), len(rows) - len(full), eq, 100 * dd, m, '+' if (eq > feq and m > fm) else ' ',
                 sum(1 for a, b in zip(fpy, py) if b > a), ' '.join('%+.0f' % (b - a) for a, b in zip(fpy, py))), flush=True)
    cand = [k for k in RES if k != 'итоговая']
    top = sorted(cand, key=lambda k: -sim_w.money_at_dd(RES[k], 0.12)[1])[:2]
    print('    ЗАЛИВКА 85%% против ИТОГОВОЙ: %s' % ', '.join(top), flush=True)
    for lbl in top:
        drop = dict(DROPS)[lbl]
        A, B = [], []
        for seed in range(1, 11):
            rng = np.random.default_rng(seed)
            sub = [x for x in sig_all if rng.random() < 0.85]
            r0, r1 = run(sub, FULL), run(sub, FULL - drop)
            e0, d0 = fix2(r0); e1, d1 = fix2(r1)
            A.append((e0, d0, sim_w.money_at_dd(r0, 0.12)[1], pyr(r0))); B.append((e1, d1, sim_w.money_at_dd(r1, 0.12)[1], pyr(r1)))
        med = lambda X, i: float(np.median([x[i] for x in X]))
        yw = ' '.join('%d/10' % sum(1 for x, y in zip(A, B) if y[3][j] > x[3][j]) for j in range(5))
        print('      %-26s итоговая $%.0f %.1f%% DD12 $%.0f | без него $%.0f %.1f%% (макс %.1f%%) DD12 $%.0f | лучше $ %d/10, DD12 %d/10 | годы %s'
              % (lbl, med(A, 0), 100 * med(A, 1), med(A, 2), med(B, 0), 100 * med(B, 1), 100 * max(x[1] for x in B), med(B, 2),
                 sum(1 for x, y in zip(A, B) if y[0] > x[0]), sum(1 for x, y in zip(A, B) if y[2] > x[2]), yw), flush=True)
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
io.open('ablate.py', 'w', encoding='utf-8').write(src)
print('ablate.py готов, синтаксис ок')
