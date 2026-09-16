"""Writes widegeo.py: does the breadth source deserve its own stop/take/hold?

The wide-pullback trades (item 9) inherit the bank's default geometry - stop 3 ATR, take 1 ATR, hold 48h -
because nobody ever asked whether that fits them. Item 11 already gave them their own stop_ref, so the
source is known to behave differently from the strict pullback. A wider entry band should mean a different
excursion profile, and geometry is the one knob never tuned per source.

Grid over stop 2.0-4.0, take 0.75-2.0, hold 24/48/72 applied ONLY to key r2_4, on all three books, both
measures, full fill. The control cell (3.0, 1.0, 48) must reproduce the accepted system exactly before any
other cell is read - otherwise the harness is lying.
"""
import ast
import io

s = io.open('widestopref.py', encoding='utf-8').read()
head = s[:s.index("for bi, (book_name, coins) in enumerate(BOOKS3):")]

TAIL = r'''
WIDE_REF = 0.060
SLS = (2.0, 2.5, 3.0, 3.5, 4.0)
TPS = (0.75, 1.0, 1.5, 2.0)
HHS = (24, 48, 72)

print('  ГЕОМЕТРИЯ ТОЛЬКО ДЛЯ ШИРОКИХ ОТКАТОВ (источник r2_4), полная заливка', flush=True)
for book_name, coins in BOOKS3:
    H = {s: coin_hours(s) for s in coins}
    base_sig = make_sig_raw(coins)
    extra = roll_entries(coins, H, 'r2_4', 4, 2, 4, 0)
    sig_all = sorted(base_sig + extra, key=lambda x: (x[0], x[1]))

    def cell(sl, tp, hh):
        GEO.clear(); GEO.update(BASE_GEO); GEO['r2_4'] = (sl, tp, hh)
        rows, keys = book_keyed(sig_all, FULL)
        fin = weigh_src(rows, keys, {'r2_4'}, 1.25)
        refs = [(WIDE_REF if k == 'r2_4' else sim_w.REF) for k in keys]
        r = sim_ref(fin, 0.014, refs)
        return (r['eq'], abs(L.dd_of(r['curve'])), money_at_dd_ref(fin, refs),
                sum(1 for k in keys if k == 'r2_4'), len(rows))

    e0, d0, m0, n0, t0 = cell(3.0, 1.0, 48)
    print('', flush=True)
    print('  %s' % book_name, flush=True)
    print('    КОНТРОЛЬ 3.0/1.0/48ч: $%.0f просадка %.1f%% DD12 $%.0f | широких сделок %d из %d'
          % (e0, 100 * d0, m0, n0, t0), flush=True)
    best = []
    for hh in HHS:
        for sl in SLS:
            for tp in TPS:
                e, d, m, n, t = cell(sl, tp, hh)
                best.append((min(e / e0, m / m0), sl, tp, hh, e, d, m, n))
                if (e > e0 and m > m0) or (sl, tp, hh) == (3.0, 1.0, 48):
                    print('    стоп %.1f тейк %.2f держ %2dч: $%6.0f (%+5.1f%%) просадка %.1f%% DD12 $%6.0f (%+5.1f%%) широких %d'
                          % (sl, tp, hh, e, 100 * (e / e0 - 1), 100 * d, m, 100 * (m / m0 - 1), n), flush=True)
    best.sort(reverse=True)
    print('    ЛУЧШИЕ ПО ХУДШЕЙ ИЗ ДВУХ МЕР:', flush=True)
    for w, sl, tp, hh, e, d, m, n in best[:5]:
        print('      %.1f/%.2f/%2dч  худшая мера %+5.1f%% | $%6.0f просадка %.1f%% DD12 $%6.0f'
              % (sl, tp, hh, 100 * (w - 1), e, 100 * d, m), flush=True)
'''

src = head + TAIL
ast.parse(src)
io.open('widegeo.py', 'w', encoding='utf-8').write(src)
print('widegeo.py gotov, sintaksis ok')
