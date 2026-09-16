"""Writes plainwide.py: the full ladder from the safest variant to the current proposal.

capmult showed the fragility is not the product of the multipliers but the fact that every breadth trade
is enlarged (item 10) and barely shrunk by stop_ref (item 11). So the honest next question is what the
source is worth WITHOUT its amplifiers, and what each amplifier costs when the live entry is worse than
the model.

Ladder, all on the same trades, three books, full fill at 1.4%:

    живая                     — as today
    + пункт 8                 — the boost only (no new trades at all)
    + пункты 8, 9             — the breadth source at normal size and normal stop_ref
    + пункты 8, 9, 10         — the source enlarged x1.25
    + пункты 8, 9, 10, 11     — plus its own stop_ref 0.060 (the proposal as written)

For each rung: money, drawdown, money at 12% drawdown, and the same after the breadth trades enter
0.25% worse - the number that decides how much amplification is safe to switch on first.
"""
import ast
import io

s = io.open('widestopref.py', encoding='utf-8').read()
head = s[:s.index("for bi, (book_name, coins) in enumerate(BOOKS3):")]

TAIL = r'''
WIDE_REF = 0.060
SLIP_TEST = 0.0025


def boosted(rows, wide_mult, keys):
    cnt = collections.Counter(x[0] for x in rows)
    out = []
    for x, k in zip(rows, keys):
        w = 1.25 if cnt[x[0]] >= 10 else 1.0
        if k == 'r2_4':
            w *= wide_mult
        out.append(x[:5] + (w,))
    return out


def penalise(rows, mask, d):
    return [x[:2] + ((x[2] - d / x[3]) if m else x[2],) + x[3:] for x, m in zip(rows, mask)]


for book_name, coins in BOOKS3:
    H = {s: coin_hours(s) for s in coins}
    base_sig = make_sig_raw(coins)
    extra = roll_entries(coins, H, 'r2_4', 4, 2, 4, 0)
    sig_all = sorted(base_sig + extra, key=lambda x: (x[0], x[1]))
    GEO.clear(); GEO.update(BASE_GEO)
    r_live, k_live = book_keyed(sig_all, BASE_KEYS)
    r_full, k_full = book_keyed(sig_all, FULL)
    live_rows = [x[:5] + (1.0,) for x in r_live]
    refs_live = [sim_w.REF] * len(r_live)
    wide_mask_full = [k == 'r2_4' for k in k_full]
    VAR = (
        ('живая', live_rows, refs_live, [False] * len(r_live)),
        ('+ пункт 8', boosted(r_live, 1.0, k_live), refs_live, [False] * len(r_live)),
        ('+ пункты 8,9', boosted(r_full, 1.0, k_full), [sim_w.REF] * len(r_full), wide_mask_full),
        ('+ пункты 8,9,10', boosted(r_full, 1.25, k_full), [sim_w.REF] * len(r_full), wide_mask_full),
        ('+ пункты 8,9,10,11', boosted(r_full, 1.25, k_full),
         [(WIDE_REF if k == 'r2_4' else sim_w.REF) for k in k_full], wide_mask_full),
    )
    print('', flush=True)
    print('  ===== %s: лестница вариантов =====' % book_name, flush=True)
    print('    вариант              идеальный вход: $ / просадка / DD12    вход ширины хуже на 0.25%: $ / DD12', flush=True)
    for lbl, rws, refs, mask in VAR:
        r = sim_ref(rws, 0.014, refs)
        m = money_at_dd_ref(rws, refs)
        if any(mask):
            pen = penalise(rws, mask, SLIP_TEST)
            rp = sim_ref(pen, 0.014, refs)
            mp = money_at_dd_ref(pen, refs)
            tail = '$%6.0f (%+5.1f%%) $%6.0f (%+5.1f%%)' % (rp['eq'], 100 * (rp['eq'] / r['eq'] - 1), mp, 100 * (mp / m - 1))
        else:
            tail = 'не касается (сделок источника нет)'
        print('    %-20s $%6.0f %4.1f%% $%6.0f          %s' % (lbl, r['eq'], 100 * abs(L.dd_of(r['curve'])), m, tail), flush=True)
'''

src = head + TAIL
ast.parse(src)
io.open('plainwide.py', 'w', encoding='utf-8').write(src)
print('plainwide.py готов, синтаксис ок')
