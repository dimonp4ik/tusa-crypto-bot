"""Writes bookpick.py: which coin list to actually run.

Every test ran on three lists - without XLM and AAVE, all 15, and without AAVE but with XLM - and the
owner was never given a straight comparison of the lists themselves. The evidence is scattered across
runs: book 2 (all 15) earns most but is the fragile one (worst drawdown tails, -27% of headroom at a
0.25% worse entry, a negative three-month stretch), book 1 is the calmest, book 3 sits between.

This run puts them side by side on identical rules (items 8-11), 85% fill, ten paired draws: money,
month stats, drawdown and its worst case, money at equal drawdown, the loss at a 0.25% worse entry, and
how many coins each list needs. One table, one recommendation.
"""
import ast
import io

s = io.open('widestopref.py', encoding='utf-8').read()
head = s[:s.index("for bi, (book_name, coins) in enumerate(BOOKS3):")]

TAIL = r'''
import datetime as DT
WIDE_REF = 0.060
SLIP_TEST = 0.0025


def penalise(rows, mask, d):
    return [x[:2] + ((x[2] - d / x[3]) if m else x[2],) + x[3:] for x, m in zip(rows, mask)]


def month_stats(curve):
    by_m = {}
    for t, v in sorted(curve):
        by_m[DT.datetime.fromtimestamp(t, DT.UTC).strftime('%Y-%m')] = v
    ks = sorted(by_m)
    vals = [sim_w.DEPOSIT] + [by_m[k] for k in ks]
    rets = [vals[i + 1] / vals[i] - 1 for i in range(len(vals) - 1)]
    r3 = [np.prod([1 + x for x in rets[i:i + 3]]) - 1 for i in range(max(0, len(rets) - 2))]
    return float(np.median(rets)), int(sum(1 for x in rets if x < 0)), len(rets), float(min(r3)) if r3 else 0.0


print('  СРАВНЕНИЕ НАБОРОВ МОНЕТ: одинаковые правила (пункты 8-11), заливка 85%, медиана 10 розыгрышей', flush=True)
for book_name, coins in BOOKS3:
    H = {s: coin_hours(s) for s in coins}
    base_sig = make_sig_raw(coins)
    extra = roll_entries(coins, H, 'r2_4', 4, 2, 4, 0)
    sig_all = sorted(base_sig + extra, key=lambda x: (x[0], x[1]))
    GEO.clear(); GEO.update(BASE_GEO)
    E, D, M, MED, NEG, W3, PEN = [], [], [], [], [], [], []
    for seed in range(1, 11):
        rng = np.random.default_rng(seed)
        sub = [x for x in sig_all if rng.random() < 0.85]
        rf, kf = book_keyed(sub, FULL)
        fin = weigh_src(rf, kf, {'r2_4'}, 1.25)
        refs = [(WIDE_REF if k == 'r2_4' else sim_w.REF) for k in kf]
        r = sim_ref(fin, 0.014, refs)
        med, neg, nm, w3 = month_stats(r['curve'])
        E.append(r['eq']); D.append(abs(L.dd_of(r['curve']))); M.append(money_at_dd_ref(fin, refs))
        MED.append(med); NEG.append(neg); W3.append(w3)
        pen = penalise(fin, [k == 'r2_4' for k in kf], SLIP_TEST)
        PEN.append(sim_ref(pen, 0.014, refs)['eq'] / r['eq'] - 1)
    md = lambda X: float(np.median(X))
    print('', flush=True)
    print('  %s — монет %d' % (book_name, len(coins)), flush=True)
    print('    деньги $%.0f | в месяц медиана %+.2f%% | минусовых месяцев %.0f из %.0f | худшие 3 месяца %+.1f%%'
          % (md(E), 100 * md(MED), md(NEG), nm, 100 * md(W3)), flush=True)
    print('    просадка %.1f%% (худшая из розыгрышей %.1f%%) | при равной просадке $%.0f | вход хуже на 0.25%%: %+.1f%%'
          % (100 * md(D), 100 * max(D), md(M), 100 * md(PEN)), flush=True)
'''

src = head + TAIL
ast.parse(src)
io.open('bookpick.py', 'w', encoding='utf-8').write(src)
print('bookpick.py готов, синтаксис ок')
