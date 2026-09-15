"""Writes capdiag.py: does the margin cap or the daily pause skip trades in the widest hours?

The accepted system concentrates trades in market-wide dips and sizes them up (x1.25 at n >= 10). The
live simulator skips an entry when open margin plus the new margin would exceed the usable share of
equity, and pauses new entries after a 3% daily loss. If those guards bite in exactly the best hours,
the order of entries inside an hour matters and the book is losing its best trades to the guard.

Per book, for the system, boost only, and the accepted system: trades taken, skipped by margin,
skipped by the daily pause, at 1.4% and at the risk that gives 12% drawdown; and how many of the
skipped trades opened in hours with 10+ entries or with strict breadth >= 4.
"""
import ast
import io

s = io.open('combo.py', encoding='utf-8').read()
head = s[:s.index("VARS = (")]

TAIL = r'''
ACC = BASE_KEYS | {'w50'}


def sim_trace(trades, target):
    """sim_w.simulate with the index of every skipped trade recorded."""
    margin_frac = target / (sim_w.LEV * sim_w.REF)
    eq = peak = sim_w.DEPOSIT
    open_pos = []
    day, day_start, day_paused = None, sim_w.DEPOSIT, False
    paused = False
    sk_m, sk_p = [], []
    for idx, (a, b, R, sf, s, w) in enumerate(trades):
        while open_pos and open_pos[0][0] <= a:
            t_close, m, money_per_R, rr = open_pos.pop(0)
            eq += money_per_R * rr
            peak = max(peak, eq)
        d = L.day_of(a)
        if d != day:
            day, day_start, day_paused = d, eq, False
        if not paused and eq <= peak * (1 - sim_w.MAX_DD):
            paused = True
        if not day_paused and eq <= day_start * (1 - sim_w.MAX_DAILY):
            day_paused = True
        if paused or day_paused:
            sk_p.append(idx)
            continue
        margin = margin_frac * eq
        if sf > sim_w.REF:
            margin *= sim_w.REF / sf
        margin *= w
        if sum(p[1] for p in open_pos) + margin > eq * sim_w.USABLE:
            sk_m.append(idx)
            continue
        open_pos.append((b, margin, margin * sim_w.LEV * sf, R))
        open_pos.sort(key=lambda p: p[0])
    for t_close, m, money_per_R, rr in open_pos:
        eq += money_per_R * rr
    return eq, sk_m, sk_p


for book_name, coins in BOOKS3:
    H = {s: coin_hours(s) for s in coins}
    base_sig = make_sig_raw(coins)
    extra = wide_entries(coins, H, 'w50', 4, 50.0, 45.0, 4, 0)
    sig_all = sorted(base_sig + extra, key=lambda x: (x[0], x[1]))
    strict_b = collections.Counter()
    for s in coins:
        for close, (a, r6, atr, ok) in H[s].items():
            if ok and a >= 56.3761 and r6 <= 33.7947:
                strict_b[close] += 1
    print('', flush=True)
    print('  ===== %s =====' % book_name, flush=True)
    for lbl, rows in (('система', book(sig_all, BASE_KEYS)[0]), ('только буст', boost(book(sig_all, BASE_KEYS)[0])),
                      ('принятое', boost(book(sig_all, ACC)[0]))):
        cnt = collections.Counter(x[0] for x in rows)
        k12 = sim_w.money_at_dd(rows, 0.12)[0]
        for tgt, tl in ((0.014, '1.4%'), (k12, 'DD12 %.2f%%' % (100 * k12))):
            eq, sk_m, sk_p = sim_trace(rows, tgt)
            wm = sum(1 for i in sk_m if cnt[rows[i][0]] >= 10)
            bm = sum(1 for i in sk_m if strict_b.get(rows[i][0], 0) >= 4)
            mr = [rows[i][2] for i in sk_m]
            print('    %-12s %-12s $%6.0f | сделок %4d, отказ по марже %3d (из них в часах n>=10: %3d, ширина>=4: %3d; их ср R %s), пауза %3d'
                  % (lbl, tl, eq, len(rows), len(sk_m), wm, bm, ('%+.3f' % np.mean(mr)) if mr else '  -  ', len(sk_p)), flush=True)
'''

src = head + TAIL
ast.parse(src)
io.open('capdiag.py', 'w', encoding='utf-8').write(src)
print('capdiag.py готов, синтаксис ок')
