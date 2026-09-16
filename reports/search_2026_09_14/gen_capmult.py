"""Writes capmult.py: capping the combined size multiplier to cut the entry-quality fragility.

entryslip showed the breadth trades are the fragile part: 0.25% of worse entry costs 5-9% of the money
and up to 27% of the drawdown headroom on the full book, while only 52-87 trades are touched. They are
fragile because they are the biggest trades in the book - item 8 (x1.25 in a wide hour) and item 10
(x1.25 for the source) multiply to 1.5625, and item 11 leaves them mostly un-shrunk by stop_ref.

So: cap the combined multiplier at 1.25 / 1.35 / 1.50 (today's effective cap is 1.5625) and measure both
sides of the trade-off - what the cap costs with a perfect entry, and what it saves when the live entry
is 0.25% worse. A cap that costs little and saves much is the safer default for launch day.

Three books, money at 1.4% and at 12% drawdown, full fill.
"""
import ast
import io

s = io.open('widestopref.py', encoding='utf-8').read()
head = s[:s.index("for bi, (book_name, coins) in enumerate(BOOKS3):")]

TAIL = r'''
WIDE_REF = 0.060
CAPS = (1.25, 1.35, 1.50, 1.5625)
SLIP_TEST = 0.0025


def weigh_cap(rows, keys, cap):
    cnt = collections.Counter(x[0] for x in rows)
    out = []
    for x, k in zip(rows, keys):
        w = 1.25 if cnt[x[0]] >= 10 else 1.0
        if k == 'r2_4':
            w *= 1.25
        out.append(x[:5] + (min(w, cap),))
    return out


def penalise(rows, mask, d):
    return [x[:2] + ((x[2] - d / x[3]) if m else x[2],) + x[3:] for x, m in zip(rows, mask)]


for book_name, coins in BOOKS3:
    H = {s: coin_hours(s) for s in coins}
    base_sig = make_sig_raw(coins)
    extra = roll_entries(coins, H, 'r2_4', 4, 2, 4, 0)
    sig_all = sorted(base_sig + extra, key=lambda x: (x[0], x[1]))
    GEO.clear(); GEO.update(BASE_GEO)
    rows, keys = book_keyed(sig_all, FULL)
    refs = [(WIDE_REF if k == 'r2_4' else sim_w.REF) for k in keys]
    is_wide = [k == 'r2_4' for k in keys]
    CNT = collections.Counter(x[0] for x in rows)      # сделок в одном баре — считаем один раз
    print('', flush=True)
    print('  ===== %s: потолок суммарного множителя =====' % book_name, flush=True)
    print('    потолок   идеальный вход: $ / просадка / DD12      вход хуже на 0.25%: $ / DD12     сделок под потолком', flush=True)
    base_eq = base_m = None
    for cap in CAPS:
        w = weigh_cap(rows, keys, cap)
        r = sim_ref(w, 0.014, refs)
        m = money_at_dd_ref(w, refs)
        pen = penalise(w, is_wide, SLIP_TEST)
        rp = sim_ref(pen, 0.014, refs)
        mp = money_at_dd_ref(pen, refs)
        n_cap = sum(1 for x, k in zip(rows, keys)
                    if (1.25 if CNT[x[0]] >= 10 else 1.0) * (1.25 if k == 'r2_4' else 1.0) > cap + 1e-9)
        if base_eq is None:
            base_eq, base_m = r['eq'], m
        print('    x%-6.4g  $%6.0f %4.1f%% $%6.0f            $%6.0f (%+5.1f%%) $%6.0f (%+5.1f%%)   %3d'
              % (cap, r['eq'], 100 * abs(L.dd_of(r['curve'])), m, rp['eq'], 100 * (rp['eq'] / r['eq'] - 1),
                 mp, 100 * (mp / m - 1), n_cap), flush=True)
    print('    (последняя строка x1.5625 — как в предложении сейчас; «сделок под потолком» = сколько сделок он реально урезает)', flush=True)
'''

src = head + TAIL
ast.parse(src)
io.open('capmult.py', 'w', encoding='utf-8').write(src)
print('capmult.py готов, синтаксис ок')
