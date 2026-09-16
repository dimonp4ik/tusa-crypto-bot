"""Writes recent.py: what turning the proposal on 3, 6 and 12 months ago would have given.

The quarterly split showed the breadth source almost silent in the last two quarters while the boost
carried +3.1..3.8%. Before the owner decides, the plainest question deserves a plain answer: if the
patch had gone in three, six or twelve months ago, what would the account have done since - with the
live system, with the boost alone, and with everything.

Each window starts at $120 at 1.4% risk under the 85% fill, ten paired draws, on all three books. Also
printed: trades in the window, how many came from the breadth source, hours with breadth >= 4, and the
drawdown - so a quiet window is visible as quiet rather than as failure.
"""
import ast
import io

s = io.open('widestopref.py', encoding='utf-8').read()
head = s[:s.index("for bi, (book_name, coins) in enumerate(BOOKS3):")]

TAIL = r'''
import datetime as DT
WIDE_REF = 0.060
WINDOWS = ((3, 'последние 3 месяца'), (6, 'последние 6 месяцев'), (12, 'последние 12 месяцев'))


def boost_only(rows):
    cnt = collections.Counter(x[0] for x in rows)
    return [x[:5] + ((1.25 if cnt[x[0]] >= 10 else 1.0),) for x in rows]


for book_name, coins in BOOKS3:
    H = {s: coin_hours(s) for s in coins}
    base_sig = make_sig_raw(coins)
    extra = roll_entries(coins, H, 'r2_4', 4, 2, 4, 0)
    sig_all = sorted(base_sig + extra, key=lambda x: (x[0], x[1]))
    GEO.clear(); GEO.update(BASE_GEO)
    last = max(x[0] for x in sig_all)
    strict = collections.defaultdict(set)
    for s in coins:
        for close, (a, b, atr, ok) in H[s].items():
            if ok and a >= 56.3761 and b <= 33.7947:
                strict[close].add(s)
    print('', flush=True)
    print('  ===== %s: последняя свеча %s =====' % (book_name, DT.datetime.fromtimestamp(last, DT.UTC).strftime('%Y-%m-%d')), flush=True)
    for months, lbl in WINDOWS:
        lo = last - int(months * 30.44 * 86400)
        sub_all = [x for x in sig_all if x[0] >= lo]
        wide_hours = len({c for c in strict if c >= lo and len(strict.get(c, set()) | strict.get(c - 3600, set())) >= 4})
        A, B, C = [], [], []
        nw = []
        for seed in range(1, 11):
            rng = np.random.default_rng(seed)
            sub = [x for x in sub_all if rng.random() < 0.85]
            rl, kl = book_keyed(sub, BASE_KEYS)
            rf, kf = book_keyed(sub, FULL)
            if len(rl) < 20 or len(rf) < 20:
                continue
            live = [x[:5] + (1.0,) for x in rl]
            bo = boost_only(rl)
            fin = weigh_src(rf, kf, {'r2_4'}, 1.25)
            rr_l = [sim_w.REF] * len(rl)
            rr_f = [(WIDE_REF if k == 'r2_4' else sim_w.REF) for k in kf]
            for dst, rows_, refs_ in ((A, live, rr_l), (B, bo, rr_l), (C, fin, rr_f)):
                r = sim_ref(rows_, 0.014, refs_)
                dst.append((r['eq'], abs(L.dd_of(r['curve'])), len(rows_)))
            nw.append(sum(1 for k in kf if k == 'r2_4'))
        if not A:
            continue
        med = lambda X, i: float(np.median([x[i] for x in X]))
        print('    %-22s часов с шириной>=4: %2d | сделок в окне %4.0f, из них ширины %.0f'
              % (lbl, wide_hours, med(A, 2), np.median(nw)), flush=True)
        print('      живая     $%6.0f (просадка %4.1f%%)' % (med(A, 0), 100 * med(A, 1)), flush=True)
        print('      + буст    $%6.0f (%4.1f%%)  %+5.1f%% к живой' % (med(B, 0), 100 * med(B, 1), 100 * (med(B, 0) / med(A, 0) - 1)), flush=True)
        print('      всё       $%6.0f (%4.1f%%)  %+5.1f%% к живой' % (med(C, 0), 100 * med(C, 1), 100 * (med(C, 0) / med(A, 0) - 1)), flush=True)
'''

src = head + TAIL
ast.parse(src)
io.open('recent.py', 'w', encoding='utf-8').write(src)
print('recent.py готов, синтаксис ок')
