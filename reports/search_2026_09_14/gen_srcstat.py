"""Writes srcstat.py: what each source actually contributes - trades, win rate, money, drawdown.

The owner keeps asking the same three questions - how much per month, what win rate, what drawdown - and
the answers so far are for the SYSTEM as a whole. Nobody has laid out the eight sources side by side:
five bank rules, the 1h and 2h pullbacks, and the wide pullback of item 9. Ablation showed none can be
removed, but that is not the same as knowing which one earns and which one merely does no harm.

Per source, on all three books, full fill and 85% (median of ten draws): trades, win rate, average R,
share of gross profit, share of gross loss, and how much of the WORST drawdown window each source sat in.
No new rule is proposed here - this is the table that answers the owner's own questions.
"""
import ast
import io

s = io.open('widestopref.py', encoding='utf-8').read()
head = s[:s.index("for bi, (book_name, coins) in enumerate(BOOKS3):")]

TAIL = r'''
WIDE_REF = 0.060
NAMES = {'r0': 'банк: вечерний памп BTC', 'r1': 'банк: разгон монеты вечером', 'r2': 'банк: ночной памп BTC',
         'r3': 'банк: утренний шорт', 'r4': 'банк: шорт на отскоке', 'o1h': 'откат 1ч',
         'o2h': 'откат 2ч', 'r2_4': 'широкий откат (пункт 9)'}
ORDER = ('o1h', 'o2h', 'r2_4', 'r0', 'r1', 'r2', 'r3', 'r4')


def build(sig):
    GEO.clear(); GEO.update(BASE_GEO)
    rows, keys = book_keyed(sig, FULL)
    fin = weigh_src(rows, keys, {'r2_4'}, 1.25)
    refs = [(WIDE_REF if k == 'r2_4' else sim_w.REF) for k in keys]
    return fin, refs, keys


def worst_window(curve):
    peak, pt, best = -1e18, None, (0.0, None, None)
    for t, v in curve:
        if v > peak:
            peak, pt = v, t
        d = v / peak - 1
        if d < best[0]:
            best = (d, pt, t)
    return best[1], best[2]


for book_name, coins in BOOKS3:
    H = {s: coin_hours(s) for s in coins}
    sig_all = sorted(make_sig_raw(coins) + roll_entries(coins, H, 'r2_4', 4, 2, 4, 0),
                     key=lambda x: (x[0], x[1]))
    fin, refs, keys = build(sig_all)
    r = sim_ref(fin, 0.014, refs)
    a, b = worst_window(r['curve'])
    gp = sum(x[2] * x[5] for x in fin if x[2] > 0)
    gl = -sum(x[2] * x[5] for x in fin if x[2] < 0)
    print('', flush=True)
    print('  %s — счёт $%.0f, худшая просадка %.1f%%' % (book_name, r['eq'], 100 * abs(L.dd_of(r['curve']))), flush=True)
    print('    источник                        сделок   ВР     ср.R    доля прибыли  доля убытка   в худшей просадке', flush=True)
    for k in ORDER:
        ix = [i for i, kk in enumerate(keys) if kk == k]
        if not ix:
            continue
        R = [fin[i][2] for i in ix]
        W = [fin[i][2] * fin[i][5] for i in ix]
        inw = sum(fin[i][2] * fin[i][5] for i in ix if a is not None and a <= fin[i][1] <= b)
        print('    %-30s %5d  %5.1f%%  %+.3f     %5.1f%%        %5.1f%%       %+7.2fR'
              % (NAMES[k], len(ix), 100 * sum(1 for x in R if x > 0) / len(R), float(np.mean(R)),
                 100 * sum(x for x in W if x > 0) / gp, 100 * -sum(x for x in W if x < 0) / gl, inw), flush=True)
    tot = [x[2] for x in fin]
    print('    %-30s %5d  %5.1f%%  %+.3f' % ('ВСЯ СИСТЕМА', len(tot),
          100 * sum(1 for x in tot if x > 0) / len(tot), float(np.mean(tot))), flush=True)

    ST = {k: [] for k in ORDER}
    WRS = {k: [] for k in ORDER}
    for seed in range(1, 11):
        rng = np.random.default_rng(seed)
        sub = [x for x in sig_all if rng.random() < 0.85]
        f2, r2, k2 = build(sub)
        for k in ORDER:
            ix = [i for i, kk in enumerate(k2) if kk == k]
            if ix:
                ST[k].append(len(ix))
                WRS[k].append(100 * sum(1 for i in ix if f2[i][2] > 0) / len(ix))
    print('    при заливке 85%% (медиана 10 розыгрышей): %s'
          % ', '.join('%s %d сд. ВР %.0f%%' % (NAMES[k].split(':')[-1].strip(), np.median(ST[k]), np.median(WRS[k]))
                      for k in ORDER if ST[k]), flush=True)
'''

src = head + TAIL
ast.parse(src)
io.open('srcstat.py', 'w', encoding='utf-8').write(src)
print('srcstat.py gotov, sintaksis ok')
