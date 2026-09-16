"""Writes tailrisk.py: the tails behind the latch table, not the medians.

latchladder.py showed the +7.8%/month ceiling belongs to the 15% latch: raise it to 20% and risk 2%
earns ~11%/month, switch it off and risk 4% earns ~20%/month. Those were medians of ten fills. A risk
decision cannot be made on medians - the latch exists precisely for the tail.

Cells: latch 15% at risk 1.4% (today), latch 20% at 2.0%, latch 25% at 2.5%, latch 25% at 3.0%, no latch
at 4.0%. Thirty fills each, three books. Per cell: median, worst and best final money; median and worst
drawdown; the share of fills whose drawdown passes 20 / 25 / 30%; how often the latch fired; the longest
stretch under water in days; and the worst 3-month change of equity.
"""
import ast
import io

s = io.open('srcweight.py', encoding='utf-8').read()
head = s[:s.index("VARS = (")]

TAIL = r'''
import datetime as DT
CELLS = ((0.15, 0.014, 'сейчас: защёлка 15%, риск 1.4%'), (0.20, 0.020, 'защёлка 20%, риск 2.0%'),
         (0.25, 0.025, 'защёлка 25%, риск 2.5%'), (0.25, 0.030, 'защёлка 25%, риск 3.0%'),
         (None, 0.040, 'без защёлки, риск 4.0%'))
DRAWS = 30


def sim_latch(trades, target, latch):
    old = sim_w.MAX_DD
    try:
        sim_w.MAX_DD = latch if latch is not None else 10.0
        return sim_w.simulate(trades, target)
    finally:
        sim_w.MAX_DD = old


def underwater_days(curve):
    peak, start, worst = curve[0][1], curve[0][0], 0.0
    for t, v in curve:
        if v >= peak:
            worst = max(worst, (t - start) / 86400)
            peak, start = v, t
    worst = max(worst, (curve[-1][0] - start) / 86400)
    return worst


def worst_3m(curve):
    by_m = {}
    for t, v in curve:
        by_m[DT.datetime.fromtimestamp(t, DT.UTC).strftime('%Y-%m')] = v
    ks = sorted(by_m)
    vals = [by_m[k] for k in ks]
    out = 0.0
    for i in range(len(vals) - 3):
        out = min(out, vals[i + 3] / vals[i] - 1)
    return out


for book_name, coins in BOOKS3:
    H = {s: coin_hours(s) for s in coins}
    base_sig = make_sig_raw(coins)
    extra = roll_entries(coins, H, 'r2_4', 4, 2, 4, 0)
    sig_all = sorted(base_sig + extra, key=lambda x: (x[0], x[1]))
    GEO.clear(); GEO.update(BASE_GEO)
    subs = []
    for seed in range(1, DRAWS + 1):
        rng = np.random.default_rng(seed)
        sub = [x for x in sig_all if rng.random() < 0.85]
        r0, k0 = book_keyed(sub, FULL)
        subs.append(weigh_src(r0, k0, {'r2_4'}, 1.25))
    print('', flush=True)
    print('  ===== %s: %d розыгрышей заливки 85%% =====' % (book_name, DRAWS), flush=True)
    print('    клетка                            $ медиана / худший / лучший     просадка мед/худш   >20%% >25%% >30%%   защёлка   под водой   худшие 3 мес', flush=True)
    for latch, rk, lbl in CELLS:
        res = []
        for rows in subs:
            r = sim_latch(rows, rk, latch)
            dd = abs(L.dd_of(r['curve']))
            res.append((r['eq'], dd, bool(r['paused_at']), underwater_days(r['curve']), worst_3m(r['curve'])))
        eqs = np.array([x[0] for x in res]); dds = np.array([x[1] for x in res])
        print('    %-33s $%7.0f / $%6.0f / $%8.0f   %5.1f%% / %5.1f%%   %3.0f%% %3.0f%% %3.0f%%   %2d/%d      %4.0f дн     %+6.1f%%'
              % (lbl, np.median(eqs), eqs.min(), eqs.max(), 100 * np.median(dds), 100 * dds.max(),
                 100 * np.mean(dds > 0.20), 100 * np.mean(dds > 0.25), 100 * np.mean(dds > 0.30),
                 sum(1 for x in res if x[2]), DRAWS,
                 float(np.median([x[3] for x in res])), 100 * float(np.median([x[4] for x in res]))), flush=True)
'''

src = head + TAIL
ast.parse(src)
io.open('tailrisk.py', 'w', encoding='utf-8').write(src)
print('tailrisk.py готов, синтаксис ок')
