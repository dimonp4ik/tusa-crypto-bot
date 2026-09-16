"""Writes final2.py: the closing table - the live system against the proposal with items 8 to 11.

Everything in the proposal is now measured; this run produces the single table the owner reads in the
morning, with item 11 (the looser stop_ref for breadth trades) included, so no number in the summary is
stale. Three books, 85% fill, ten paired draws, risk 1.4%.

Per book and per variant (live / items 8-10 / items 8-11): trades a year, win rate, percent a month,
$120 grown over the history, median and worst drawdown, money at 12% drawdown, and the monthly picture -
median month, average month, how many months of 52 end in the red, the worst month, and the worst three
months in a row. The per-trade stop_ref simulator carries a self-test against sim_w.
"""
import ast
import io

s = io.open('widestopref.py', encoding='utf-8').read()
head = s[:s.index("for bi, (book_name, coins) in enumerate(BOOKS3):")]

TAIL = r'''
import datetime as DT
WIDE_REF = 0.060


def monthly(curve):
    by_m = {}
    for t, v in sorted(curve):
        by_m[DT.datetime.fromtimestamp(t, DT.UTC).strftime('%Y-%m')] = v
    ks = sorted(by_m)
    vals = [sim_w.DEPOSIT] + [by_m[k] for k in ks]
    rets = [vals[i + 1] / vals[i] - 1 for i in range(len(vals) - 1)]
    r3 = [np.prod([1 + x for x in rets[i:i + 3]]) - 1 for i in range(max(0, len(rets) - 2))]
    return dict(med=float(np.median(rets)), avg=float(np.mean(rets)), neg=int(sum(1 for x in rets if x < 0)),
                n=len(rets), worst=float(min(rets)), worst3=float(min(r3)) if r3 else 0.0)


for bi, (book_name, coins) in enumerate(BOOKS3):
    H = {s: coin_hours(s) for s in coins}
    base_sig = make_sig_raw(coins)
    extra = roll_entries(coins, H, 'r2_4', 4, 2, 4, 0)
    sig_all = sorted(base_sig + extra, key=lambda x: (x[0], x[1]))
    GEO.clear(); GEO.update(BASE_GEO)
    print('', flush=True)
    print('  ===== %s: заливка 85%%, риск 1.4%%, медиана 10 парных розыгрышей =====' % book_name, flush=True)
    acc = {k: [] for k in ('живая', 'пункты 8-10', 'пункты 8-11')}
    for seed in range(1, 11):
        rng = np.random.default_rng(seed)
        sub = [x for x in sig_all if rng.random() < 0.85]
        rl, kl = book_keyed(sub, BASE_KEYS)
        rf, kf = book_keyed(sub, FULL)
        live = [x[:5] + (1.0,) for x in rl]
        fin = weigh_src(rf, kf, {'r2_4'}, 1.25)
        variants = (('живая', live, [sim_w.REF] * len(live)),
                    ('пункты 8-10', fin, [sim_w.REF] * len(fin)),
                    ('пункты 8-11', fin, [(WIDE_REF if k == 'r2_4' else sim_w.REF) for k in kf]))
        for lbl, rows, refs in variants:
            r = sim_ref(rows, 0.014, refs)
            v = np.array([x[2] for x in rows])
            months = (rows[-1][1] - rows[0][0]) / (365.25 * 86400 / 12)
            m = monthly(r['curve'])
            acc[lbl].append(dict(eq=r['eq'], dd=abs(L.dd_of(r['curve'])), tpy=len(v) / (months / 12),
                                 wr=float(np.mean(v > 0)), mo=(r['eq'] / 120) ** (1 / months) - 1,
                                 dd12=money_at_dd_ref(rows, refs), **m))
    for lbl in ('живая', 'пункты 8-10', 'пункты 8-11'):
        X = acc[lbl]
        med = lambda k: float(np.median([x[k] for x in X]))
        print('    %-12s %4.0f сд/год ВР %4.1f%% | %+.2f%%/мес | $120 -> $%5.0f | просадка %4.1f%% (макс %4.1f%%) | при равной просадке $%5.0f'
              % (lbl, med('tpy'), 100 * med('wr'), 100 * med('mo'), med('eq'), 100 * med('dd'),
                 100 * max(x['dd'] for x in X), med('dd12')), flush=True)
        print('                 помесячно: медиана %+.2f%%, средний %+.2f%%, в минусе %.0f из %.0f, худший %+.2f%%, худшие 3 подряд %+.2f%%'
              % (100 * med('med'), 100 * med('avg'), med('neg'), med('n'), 100 * med('worst'), 100 * med('worst3')), flush=True)
'''

src = head + TAIL
ast.parse(src)
io.open('final2.py', 'w', encoding='utf-8').write(src)
print('final2.py готов, синтаксис ок')
