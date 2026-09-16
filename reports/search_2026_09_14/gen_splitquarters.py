"""Writes splitquarters.py: which half of the proposal works, and when.

The quarterly table showed 2025-Q4 gaining +1.2..2.2% with zero trades from the breadth source - that
was the boost (item 8) alone. The proposal has two independent halves: the size boost at 10+ trades in
an hour, and the breadth source with its own weight and stop_ref (items 9, 10, 11). The owner should see
which half earns, and in which quarters, rather than one blended number.

Per quarter and per book, each quarter simulated from $120 at 1.4% risk under the full fill: the live
system, the live system plus the boost only, the live system plus the breadth half only, and everything
together. Totals at the end: how many quarters each half wins, its median and worst quarter, and the
share of the total gain it accounts for.
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


def boost_only(rows, keys):
    cnt = collections.Counter(x[0] for x in rows)
    return [x[:5] + ((1.25 if cnt[x[0]] >= 10 else 1.0),) for x in rows]


for book_name, coins in BOOKS3:
    H = {s: coin_hours(s) for s in coins}
    base_sig = make_sig_raw(coins)
    extra = roll_entries(coins, H, 'r2_4', 4, 2, 4, 0)
    sig_all = sorted(base_sig + extra, key=lambda x: (x[0], x[1]))
    GEO.clear(); GEO.update(BASE_GEO)
    r_live, k_live = book_keyed(sig_all, BASE_KEYS)
    r_full, k_full = book_keyed(sig_all, FULL)
    live = [x[:5] + (1.0,) for x in r_live]
    only_boost = boost_only(r_live, k_live)
    only_wide = [x[:5] + ((1.25 if k == 'r2_4' else 1.0),) for x, k in zip(r_full, k_full)]
    both = weigh_src(r_full, k_full, {'r2_4'}, 1.25)
    refs_live = [sim_w.REF] * len(live)
    refs_wide = [(WIDE_REF if k == 'r2_4' else sim_w.REF) for k in k_full]
    print('', flush=True)
    print('  ===== %s: вклад по половинам предложения, каждый квартал со старта $120 =====' % book_name, flush=True)
    print('    квартал    живая   + буст (п.8)   + ширина (п.9-11)   всё вместе', flush=True)
    qs = sorted({q_of(x[0]) for x in r_live})
    w_b, w_w, tot_b, tot_w, tot_a = 0, 0, [], [], []
    for q in qs:
        il = [i for i, x in enumerate(live) if q_of(x[0]) == q]
        if_ = [i for i, x in enumerate(r_full) if q_of(x[0]) == q]
        if len(il) < 10 or len(if_) < 10:
            continue
        e_live = sim_ref([live[i] for i in il], 0.014, [refs_live[i] for i in il])['eq']
        e_b = sim_ref([only_boost[i] for i in il], 0.014, [refs_live[i] for i in il])['eq']
        e_w = sim_ref([only_wide[i] for i in if_], 0.014, [refs_wide[i] for i in if_])['eq']
        e_a = sim_ref([both[i] for i in if_], 0.014, [refs_wide[i] for i in if_])['eq']
        gb, gw, ga = 100 * (e_b / e_live - 1), 100 * (e_w / e_live - 1), 100 * (e_a / e_live - 1)
        w_b += int(gb > 0.05); w_w += int(gw > 0.05)
        tot_b.append(gb); tot_w.append(gw); tot_a.append(ga)
        print('    %-9s $%5.0f   $%5.0f %+5.1f%%   $%5.0f %+6.1f%%   $%5.0f %+6.1f%%'
              % (q, e_live, e_b, gb, e_w, gw, e_a, ga), flush=True)
    n = len(tot_a)
    print('    кварталов %d | буст лучше живой в %d, ширина в %d' % (n, w_b, w_w), flush=True)
    print('    медиана прибавки: буст %+.2f%%, ширина %+.2f%%, вместе %+.2f%%'
          % (np.median(tot_b), np.median(tot_w), np.median(tot_a)), flush=True)
    print('    худший квартал:   буст %+.2f%%, ширина %+.2f%%, вместе %+.2f%%'
          % (min(tot_b), min(tot_w), min(tot_a)), flush=True)
    print('    лучший квартал:   буст %+.2f%%, ширина %+.2f%%, вместе %+.2f%%'
          % (max(tot_b), max(tot_w), max(tot_a)), flush=True)
'''

src = head + TAIL
ast.parse(src)
io.open('splitquarters.py', 'w', encoding='utf-8').write(src)
print('splitquarters.py готов, синтаксис ок')
