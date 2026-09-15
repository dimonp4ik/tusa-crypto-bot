"""Writes monthly.py: month by month, what the final system does.

The user thinks in percent per month. Averages hide the months that lose. For the final candidate -
the current system plus proposal items 8 (boost x1.25 at n >= 10) and 9 as amended (relaxed 50/45
entries while the strict 1h pullback fired on 4+ distinct coins over the current and previous hour) -
and for the system without them, under the 85% fill at 1.4% risk:

per book, the median of ten fills for every calendar month: return of the month on the equity at its
start, trades, win rate; then the share of losing months, the worst month, the median month, the
best month, the worst three consecutive months, and the average month per year.
"""
import ast
import io

s = io.open('rollbreadth.py', encoding='utf-8').read()
head = s[:s.index("CELLS = [")]

TAIL = r'''
import datetime as DT


def month_table(rows, target=0.014):
    r = sim_w.simulate(rows, target)
    curve = sorted(r['curve'])
    months = collections.OrderedDict()
    eq_at = sim_w.DEPOSIT
    ci = 0
    t0 = rows[0][0]
    d = DT.datetime.fromtimestamp(t0, DT.UTC).replace(day=1, hour=0, minute=0, second=0)
    end = DT.datetime.fromtimestamp(max(x[1] for x in rows), DT.UTC)
    while d <= end:
        nd = (d.replace(day=28) + DT.timedelta(days=4)).replace(day=1)
        lo, hi = int(d.timestamp()), int(nd.timestamp())
        while ci < len(curve) and curve[ci][0] < hi:
            eq_after = curve[ci][1]
            ci += 1
        else:
            pass
        last = [c[1] for c in curve if c[0] < hi]
        e_end = last[-1] if last else sim_w.DEPOSIT
        tr = [x for x in rows if lo <= x[0] < hi]
        months[d.strftime('%Y-%m')] = dict(start=eq_at, end=e_end, n=len(tr),
                                          wr=float(np.mean([x[2] > 0 for x in tr])) if tr else float('nan'))
        eq_at = e_end
        d = nd
    return months, r


for book_name, coins in BOOKS3:
    H = {s: coin_hours(s) for s in coins}
    base_sig = make_sig_raw(coins)
    extra = roll_entries(coins, H, 'r2_4', 4, 2, 4, 0)
    sig_all = sorted(base_sig + extra, key=lambda x: (x[0], x[1]))
    print('', flush=True)
    print('  ===== %s: заливка 85%%, риск 1.4%%, медиана 10 розыгрышей по каждому месяцу =====' % book_name, flush=True)
    for lbl, keys, do_boost in (('БЕЗ пунктов 8-9', BASE_KEYS, False), ('С пунктами 8-9 (буст + ширина за 2ч)', BASE_KEYS | {'r2_4'}, True)):
        per_month = collections.defaultdict(list)
        finals = []
        for seed in range(1, 11):
            rng = np.random.default_rng(seed)
            sub = [x for x in sig_all if rng.random() < 0.85]
            rows = book(sub, keys)[0]
            if do_boost:
                rows = boost(rows)
            mt, r = month_table(rows)
            finals.append(r['eq'])
            for k, v in mt.items():
                per_month[k].append((v['end'] / v['start'] - 1 if v['start'] > 0 else 0.0, v['n'], v['wr']))
        keys_sorted = sorted(per_month)
        med = {k: (float(np.median([x[0] for x in per_month[k]])), float(np.median([x[1] for x in per_month[k]])),
                   float(np.nanmedian([x[2] for x in per_month[k]])) if any(np.isfinite(x[2]) for x in per_month[k]) else float('nan'))
               for k in keys_sorted}
        rets = np.array([med[k][0] for k in keys_sorted])
        print('    --- %s: итог $%.0f (медиана) ---' % (lbl, np.median(finals)), flush=True)
        line = []
        for i, k in enumerate(keys_sorted):
            line.append('%s %+5.1f%%' % (k, 100 * med[k][0]))
            if len(line) == 6 or i == len(keys_sorted) - 1:
                print('      ' + '   '.join(line), flush=True)
                line = []
        roll3 = [np.prod(1 + rets[i:i + 3]) - 1 for i in range(len(rets) - 2)]
        print('      месяцев %d, в минусе %d (%.0f%%), медианный %+.2f%%, средний %+.2f%%, худший %+.2f%%, лучший %+.2f%%, худшие 3 месяца подряд %+.2f%%'
              % (len(rets), int(np.sum(rets < 0)), 100 * np.mean(rets < 0), 100 * np.median(rets), 100 * np.mean(rets),
                 100 * rets.min(), 100 * rets.max(), 100 * min(roll3)), flush=True)
        for y in range(2022, 2027):
            ys = [med[k][0] for k in keys_sorted if k.startswith(str(y))]
            ns = [med[k][1] for k in keys_sorted if k.startswith(str(y))]
            if ys:
                print('      %d: месяцев %2d, средний %+.2f%%, в минусе %d, сделок в месяц %.0f' % (y, len(ys), 100 * np.mean(ys), sum(1 for v in ys if v < 0), np.mean(ns)), flush=True)
'''

src = head + TAIL
ast.parse(src)
io.open('monthly.py', 'w', encoding='utf-8').write(src)
print('monthly.py готов, синтаксис ок')
