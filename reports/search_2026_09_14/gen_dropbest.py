"""Writes dropbest.py: does the gain survive without its best weeks?

detect showed that over 2-8 week windows the two books of trades are usually identical - the new items
fire on a handful of days in four years. So the whole gain is collected in a few episodes, and the
obvious risk is that it rests on one or two extreme ones (a single market-wide crash week). The
leave-one-out test was done across coins; across time it was not.

For each book: rank calendar weeks by how much the proposal earned over the live system in that week,
then re-measure the gap with the best week removed, the two best removed, and the five best removed -
removing the week from BOTH systems, so the comparison stays fair. If the gap survives without the five
best weeks, the edge is a mechanism; if it collapses, it is a story about one crash.
"""
import ast
import io

s = io.open('widestopref.py', encoding='utf-8').read()
head = s[:s.index("for bi, (book_name, coins) in enumerate(BOOKS3):")]

TAIL = r'''
import datetime as DT
WIDE_REF = 0.060


def boosted(rows, keys):
    cnt = collections.Counter(x[0] for x in rows)
    out = []
    for x, k in zip(rows, keys):
        w = 1.25 if cnt[x[0]] >= 10 else 1.0
        if k == 'r2_4':
            w *= 1.25
        out.append(x[:5] + (w,))
    return out


def wk(t):
    d = DT.datetime.fromtimestamp(t, DT.UTC)
    return (d - DT.timedelta(days=d.weekday())).strftime('%Y-%m-%d')


for book_name, coins in BOOKS3:
    H = {s: coin_hours(s) for s in coins}
    base_sig = make_sig_raw(coins)
    extra = roll_entries(coins, H, 'r2_4', 4, 2, 4, 0)
    sig_all = sorted(base_sig + extra, key=lambda x: (x[0], x[1]))
    GEO.clear(); GEO.update(BASE_GEO)
    r_live, k_live = book_keyed(sig_all, BASE_KEYS)
    r_full, k_full = book_keyed(sig_all, FULL)
    live = [x[:5] + (1.0,) for x in r_live]
    fin = boosted(r_full, k_full)
    refs_l = [sim_w.REF] * len(live)
    refs_f = [(WIDE_REF if k == 'r2_4' else sim_w.REF) for k in k_full]

    def run(drop_weeks):
        il = [i for i, x in enumerate(live) if wk(x[0]) not in drop_weeks]
        if_ = [i for i, x in enumerate(fin) if wk(x[0]) not in drop_weeks]
        a = sim_ref([live[i] for i in il], 0.014, [refs_l[i] for i in il])['eq']
        b = sim_ref([fin[i] for i in if_], 0.014, [refs_f[i] for i in if_])['eq']
        ma = money_at_dd_ref([live[i] for i in il], [refs_l[i] for i in il])
        mb = money_at_dd_ref([fin[i] for i in if_], [refs_f[i] for i in if_])
        return a, b, ma, mb

    # вклад недели = R-сумма сделок источника плюс прибавка буста в этой неделе
    contrib = collections.defaultdict(float)
    cnt = collections.Counter(x[0] for x in r_full)
    for x, k in zip(r_full, k_full):
        w = wk(x[0])
        if k == 'r2_4':
            contrib[w] += x[2]
        elif cnt[x[0]] >= 10:
            contrib[w] += 0.25 * x[2]
    top = [w for w, _ in sorted(contrib.items(), key=lambda kv: -kv[1])]
    a0, b0, ma0, mb0 = run(set())
    print('', flush=True)
    print('  ===== %s: живая $%.0f (DD12 $%.0f), предложение $%.0f (DD12 $%.0f), отрыв %+.0f%% / %+.0f%% ====='
          % (book_name, a0, ma0, b0, mb0, 100 * (b0 / a0 - 1), 100 * (mb0 / ma0 - 1)), flush=True)
    print('    лучшие недели по вкладу: %s' % ', '.join('%s (%.1fR)' % (w, contrib[w]) for w in top[:5]), flush=True)
    for n in (1, 2, 5):
        drop = set(top[:n])
        a, b, ma, mb = run(drop)
        print('    убрано %d лучш. недель: живая $%6.0f -> предложение $%6.0f | отрыв %+5.0f%% (деньги) / %+5.0f%% (равная просадка)'
              % (n, a, b, 100 * (b / a - 1), 100 * (mb / ma - 1)), flush=True)
'''

src = head + TAIL
ast.parse(src)
io.open('dropbest.py', 'w', encoding='utf-8').write(src)
print('dropbest.py готов, синтаксис ок')
