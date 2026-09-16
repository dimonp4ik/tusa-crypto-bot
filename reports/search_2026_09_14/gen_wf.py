"""Writes wf.py: the final system against the live one, window by window.

Entries are exhausted as a direction - what is missed is the unpredictable class. What is not yet
established is how the proposal behaves across eras rather than over the whole history at once. Nothing
is tuned here: the final system (boost n >= 10 + breadth over 2 hours + breadth trades x1.25) and the
system as it stands today are run through rolling 12-month windows stepped by 3 months, each window
starting from $120 at 1.4% risk under the 85% fill (median of ten draws).

Per window and book: money, win rate, trades, drawdown, and the same for the live system; then how many
windows the final system wins on money and on drawdown, its worst window, and the worst 3-month stretch
inside each window. A proposal that wins the whole history but loses half the windows is era-specific.
"""
import ast
import io

s = io.open('srcweight.py', encoding='utf-8').read()
head = s[:s.index("VARS = (")]

TAIL = r'''
import datetime as DT

STEP = 3
LEN = 12


def windows():
    d = DT.datetime(2022, 5, 1, tzinfo=DT.UTC)
    out = []
    while True:
        e = d
        for _ in range(LEN):
            e = (e.replace(day=28) + DT.timedelta(days=4)).replace(day=1)
        if e > DT.datetime(2026, 9, 1, tzinfo=DT.UTC):
            break
        out.append((int(d.timestamp()), int(e.timestamp()), d.strftime('%Y-%m')))
        for _ in range(STEP):
            d = (d.replace(day=28) + DT.timedelta(days=4)).replace(day=1)
    return out


WIN = windows()
print('  окон по %d месяцев с шагом %d: %d' % (LEN, STEP, len(WIN)), flush=True)

for book_name, coins in BOOKS3:
    H = {s: coin_hours(s) for s in coins}
    base_sig = make_sig_raw(coins)
    extra = roll_entries(coins, H, 'r2_4', 4, 2, 4, 0)
    sig_all = sorted(base_sig + extra, key=lambda x: (x[0], x[1]))
    GEO.clear(); GEO.update(BASE_GEO)
    print('', flush=True)
    print('  ===== %s =====' % book_name, flush=True)
    print('    окно      живая: $ / ВР / сделок / просадка      итоговая: $ / ВР / сделок / просадка', flush=True)
    winm = wind = 0
    rows_live_all, rows_fin_all = [], []
    for lo, hi, lbl in WIN:
        LV, FN = [], []
        for seed in range(1, 11):
            rng = np.random.default_rng(seed)
            sub = [x for x in sig_all if rng.random() < 0.85 and lo <= x[0] < hi]
            r_live, k_live = book_keyed(sub, BASE_KEYS)
            live = [x[:5] + (1.0,) for x in r_live]
            r_fin, k_fin = book_keyed(sub, BASE_KEYS | {'r2_4'})
            fin = weigh_src(r_fin, k_fin, {'r2_4'}, 1.25)
            for dst, rr in ((LV, live), (FN, fin)):
                if len(rr) < 20:
                    continue
                r = sim_w.simulate(rr, 0.014)
                v = np.array([x[2] for x in rr])
                dst.append((r['eq'], float(np.mean(v > 0)), len(rr), abs(L.dd_of(r['curve']))))
        if not LV or not FN:
            continue
        med = lambda X, i: float(np.median([x[i] for x in X]))
        winm += int(med(FN, 0) > med(LV, 0))
        wind += int(med(FN, 3) <= med(LV, 3) + 1e-9)
        print('    %s   $%6.0f %4.1f%% %4.0f %5.1f%%        $%6.0f %4.1f%% %4.0f %5.1f%%   %s'
              % (lbl, med(LV, 0), 100 * med(LV, 1), med(LV, 2), 100 * med(LV, 3),
                 med(FN, 0), 100 * med(FN, 1), med(FN, 2), 100 * med(FN, 3),
                 '+' if (med(FN, 0) > med(LV, 0) and med(FN, 3) <= med(LV, 3) + 1e-9) else ''), flush=True)
    print('    итоговая лучше по деньгам в %d из %d окон, просадка не хуже в %d из %d' % (winm, len(WIN), wind, len(WIN)), flush=True)
'''

src = head + TAIL
ast.parse(src)
io.open('wf.py', 'w', encoding='utf-8').write(src)
print('wf.py готов, синтаксис ок')
