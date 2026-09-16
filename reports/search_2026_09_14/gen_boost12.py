"""Writes boost12.py: does a higher boost threshold remove the early-window drawdown?

The rolling windows showed the proposal wins money almost everywhere but on the full 15-coin book its
drawdown is higher than the live system in 11 of 14 windows, worst in 2022-2023 (8.8-9.6% -> 10.4%).
The breadth source has no trades in 2022, so the extra drawdown must come from item 8 - the x1.25 boost
at 10+ trades in one bar. A higher threshold makes the boost rarer and should hit the early windows
first.

Thresholds 10 (current), 12, 14 and 16 with the same x1.25, and item 9 + item 10 unchanged. Per book:
whole history at 1.4% and at 12% drawdown, then the 14 rolling 12-month windows under the 85% fill -
money and drawdown against the live system, so the count of windows won on each measure is comparable
with wf.py. A threshold is only better if it keeps the money and returns the early-window drawdown.
"""
import ast
import io

s = io.open('srcweight.py', encoding='utf-8').read()
head = s[:s.index("VARS = (")]

TAIL = r'''
import datetime as DT
NS = (10, 12, 14, 16)


def weigh_n(rows, keys, nmin):
    cnt = collections.Counter(x[0] for x in rows)
    out = []
    for x, k in zip(rows, keys):
        w = 1.25 if cnt[x[0]] >= nmin else 1.0
        if k == 'r2_4':
            w *= 1.25
        out.append(x[:5] + (w,))
    return out


def windows():
    d = DT.datetime(2022, 5, 1, tzinfo=DT.UTC)
    out = []
    while True:
        e = d
        for _ in range(12):
            e = (e.replace(day=28) + DT.timedelta(days=4)).replace(day=1)
        if e > DT.datetime(2026, 9, 1, tzinfo=DT.UTC):
            break
        out.append((int(d.timestamp()), int(e.timestamp()), d.strftime('%Y-%m')))
        for _ in range(3):
            d = (d.replace(day=28) + DT.timedelta(days=4)).replace(day=1)
    return out


WIN = windows()
for book_name, coins in BOOKS3:
    H = {s: coin_hours(s) for s in coins}
    base_sig = make_sig_raw(coins)
    extra = roll_entries(coins, H, 'r2_4', 4, 2, 4, 0)
    sig_all = sorted(base_sig + extra, key=lambda x: (x[0], x[1]))
    GEO.clear(); GEO.update(BASE_GEO)
    rows, keys = book_keyed(sig_all, FULL)
    live_rows, _ = book_keyed(sig_all, BASE_KEYS)
    lr = sim_w.simulate([x[:5] + (1.0,) for x in live_rows], 0.014)
    print('', flush=True)
    print('  ===== %s: живая $%.0f просадка %.1f%% =====' % (book_name, lr['eq'], 100 * abs(L.dd_of(lr['curve']))), flush=True)
    for nmin in NS:
        wr = weigh_n(rows, keys, nmin)
        eq, dd = fix2(wr)
        m = sim_w.money_at_dd(wr, 0.12)[1]
        nb = sum(1 for x in wr if x[5] > 1)
        print('    порог %2d: $%6.0f %4.1f%% DD12 $%6.0f | сделок с бустом %4d' % (nmin, eq, 100 * dd, m, nb), flush=True)
    print('    ОКНА (12 месяцев, шаг 3, заливка 85%, медиана 10 розыгрышей): деньги лучше живой / просадка не хуже живой', flush=True)
    stats = {n: [0, 0, []] for n in NS}
    for lo, hi, lbl in WIN:
        LV, FN = [], {n: [] for n in NS}
        for seed in range(1, 11):
            rng = np.random.default_rng(seed)
            sub = [x for x in sig_all if rng.random() < 0.85 and lo <= x[0] < hi]
            rl, _ = book_keyed(sub, BASE_KEYS)
            rf, kf = book_keyed(sub, FULL)
            if len(rl) < 20 or len(rf) < 20:
                continue
            r = sim_w.simulate([x[:5] + (1.0,) for x in rl], 0.014)
            LV.append((r['eq'], abs(L.dd_of(r['curve']))))
            for n in NS:
                rr = sim_w.simulate(weigh_n(rf, kf, n), 0.014)
                FN[n].append((rr['eq'], abs(L.dd_of(rr['curve']))))
        if not LV:
            continue
        med = lambda X, i: float(np.median([x[i] for x in X]))
        cells = []
        for n in NS:
            if not FN[n]:
                continue
            w1 = med(FN[n], 0) > med(LV, 0)
            w2 = med(FN[n], 1) <= med(LV, 1) + 1e-9
            stats[n][0] += int(w1); stats[n][1] += int(w2); stats[n][2].append(med(FN[n], 0))
            cells.append('%2d: $%5.0f %4.1f%%%s' % (n, med(FN[n], 0), 100 * med(FN[n], 1), '+' if (w1 and w2) else ' '))
        print('      %s живая $%5.0f %4.1f%% | %s' % (lbl, med(LV, 0), 100 * med(LV, 1), '  '.join(cells)), flush=True)
    print('    ИТОГ по окнам:', flush=True)
    for n in NS:
        print('      порог %2d: деньги лучше в %2d из %d окон, просадка не хуже в %2d из %d, средние деньги окна $%.0f'
              % (n, stats[n][0], len(WIN), stats[n][1], len(WIN), np.mean(stats[n][2]) if stats[n][2] else 0), flush=True)
'''

src = head + TAIL
ast.parse(src)
io.open('boost12.py', 'w', encoding='utf-8').write(src)
print('boost12.py готов, синтаксис ок')
