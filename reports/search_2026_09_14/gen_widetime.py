"""Writes widetime.py: how the breadth source's contribution is spread over time.

The day-one sheet showed the wide source firing once in the last 45 days, and one hour with breadth >= 4.
Over the whole history it is 52-87 trades in 4.3 years, so the owner deserves the honest shape of it:
if the value came from a few clusters in 2023-2024 and the source has been dormant since, no effect
should be expected in the coming weeks, and the proposal must be judged on that basis.

Per quarter and per book: hours where the strict pullback fired on 4+ distinct coins (the breadth
condition), trades the wide source opened, their win rate and mean R, and the money the system makes
with and without the source in that quarter alone (each quarter simulated from $120 at 1.4% risk, so
quarters are comparable). Also the longest gap in days between two wide trades.
"""
import ast
import io

s = io.open('widestopref.py', encoding='utf-8').read()
head = s[:s.index("for bi, (book_name, coins) in enumerate(BOOKS3):")]

TAIL = r'''
import datetime as DT
WIDE_REF = 0.060


def q_of(t):
    d = DT.datetime.fromtimestamp(t, DT.UTC)
    return '%d-Q%d' % (d.year, (d.month - 1) // 3 + 1)


for book_name, coins in BOOKS3:
    H = {s: coin_hours(s) for s in coins}
    base_sig = make_sig_raw(coins)
    extra = roll_entries(coins, H, 'r2_4', 4, 2, 4, 0)
    sig_all = sorted(base_sig + extra, key=lambda x: (x[0], x[1]))
    GEO.clear(); GEO.update(BASE_GEO)
    rows, keys = book_keyed(sig_all, FULL)
    fin = weigh_src(rows, keys, {'r2_4'}, 1.25)
    refs = [(WIDE_REF if k == 'r2_4' else sim_w.REF) for k in keys]
    base_rows, base_keys = book_keyed(sig_all, BASE_KEYS)
    strict = collections.defaultdict(set)
    for s in coins:
        for close, (a, b, atr, ok) in H[s].items():
            if ok and a >= 56.3761 and b <= 33.7947:
                strict[close].add(s)
    wide_hours = sorted({c for c in strict if len(strict.get(c, set()) | strict.get(c - 3600, set())) >= 4})
    wide_tr = [(x, w) for x, k, w in zip(rows, keys, [r[5] for r in fin]) if k == 'r2_4']
    print('', flush=True)
    print('  ===== %s: часов с шириной >=4 за всю историю %d, сделок источника %d ====='
          % (book_name, len(wide_hours), len(wide_tr)), flush=True)
    qs = sorted({q_of(x[0]) for x in rows})
    print('    квартал   часов ширины  сделок источника  ВР источника  ср R   | квартал с источником / без него ($120 при 1.4%)', flush=True)
    for q in qs:
        wh = sum(1 for c in wide_hours if q_of(c) == q)
        wt = [x for x, w in wide_tr if q_of(x[0]) == q]
        v = np.array([x[2] for x in wt]) if wt else np.zeros(0)
        idx = [i for i, x in enumerate(rows) if q_of(x[0]) == q]
        f_q = [fin[i] for i in idx]; r_q = [refs[i] for i in idx]
        b_q = [x[:5] + (1.0,) for x in base_rows if q_of(x[0]) == q]
        if len(f_q) < 10 or len(b_q) < 10:
            continue
        with_src = sim_ref(f_q, 0.014, r_q)['eq']
        without = sim_ref(b_q, 0.014, [sim_w.REF] * len(b_q))['eq']
        print('    %-9s %6d %13d %13s %+8s   | $%6.0f / $%6.0f  %+5.1f%%'
              % (q, wh, len(wt), ('%.0f%%' % (100 * np.mean(v > 0))) if len(v) else '-',
                 ('%.3f' % v.mean()) if len(v) else '-', with_src, without, 100 * (with_src / without - 1)), flush=True)
    ts = sorted(x[0] for x, w in wide_tr)
    gaps = [(ts[i + 1] - ts[i]) / 86400 for i in range(len(ts) - 1)]
    if gaps:
        print('    перерывы между сделками источника: медиана %.0f дней, самый долгий %.0f дней (с %s)'
              % (np.median(gaps), max(gaps),
                 DT.datetime.fromtimestamp(ts[int(np.argmax(gaps))], DT.UTC).strftime('%Y-%m-%d')), flush=True)
        print('    последняя сделка источника: %s' % DT.datetime.fromtimestamp(ts[-1], DT.UTC).strftime('%Y-%m-%d'), flush=True)
'''

src = head + TAIL
ast.parse(src)
io.open('widetime.py', 'w', encoding='utf-8').write(src)
print('widetime.py готов, синтаксис ок')
