"""Writes loco.py: does the proposal survive without any single coin?

The breadth source (item 9) lives on 52-87 trades over 4.3 years. An edge that thin can rest on one or
two coins - and a coin can be delisted, lose liquidity or fail the spread gate at any time. The
leave-one-out test removes each coin completely: it neither trades nor counts towards breadth, so the
whole chain (strict pullback -> breadth >= 4 -> relaxed entries -> boost) is rebuilt without it.

Per book and per removed coin: money at 1.4% and money at 12% drawdown for the live system and for the
proposal (items 8-11), and the gap between them. What matters is the worst case: if removing one coin
turns the gap negative, the edge is that coin, not the mechanism.
"""
import ast
import io

s = io.open('widestopref.py', encoding='utf-8').read()
head = s[:s.index("for bi, (book_name, coins) in enumerate(BOOKS3):")]

TAIL = r'''
WIDE_REF = 0.060

for book_name, coins in BOOKS3:
    H = {s: coin_hours(s) for s in coins}
    full_sig = make_sig_raw(coins)
    GEO.clear(); GEO.update(BASE_GEO)
    print('', flush=True)
    print('  ===== %s =====' % book_name, flush=True)
    print('    убрана монета   живая: $ / DD12        предложение: $ / DD12     отрыв $ / DD12', flush=True)
    rows_out = []
    for drop in [None] + list(coins):
        cc = [c for c in coins if c != drop] if drop else list(coins)
        base = [x for x in full_sig if x[3] != drop] if drop else full_sig
        extra = roll_entries(cc, H, 'r2_4', 4, 2, 4, 0)
        sig_all = sorted(base + extra, key=lambda x: (x[0], x[1]))
        rl, kl = book_keyed(sig_all, BASE_KEYS)
        rf, kf = book_keyed(sig_all, FULL)
        live = [x[:5] + (1.0,) for x in rl]
        fin = weigh_src(rf, kf, {'r2_4'}, 1.25)
        refs_l = [sim_w.REF] * len(live)
        refs_f = [(WIDE_REF if k == 'r2_4' else sim_w.REF) for k in kf]
        le = sim_ref(live, 0.014, refs_l)['eq']; lm = money_at_dd_ref(live, refs_l)
        fe = sim_ref(fin, 0.014, refs_f)['eq']; fm = money_at_dd_ref(fin, refs_f)
        g1, g2 = 100 * (fe / le - 1), 100 * (fm / lm - 1)
        rows_out.append((drop, g1, g2))
        print('    %-14s $%6.0f / $%6.0f    $%6.0f / $%6.0f    %+5.0f%% / %+5.0f%%  %s'
              % (drop.replace('USDT', '') if drop else 'ничего', le, lm, fe, fm, g1, g2,
                 'ОТРЫВ ИСЧЕЗ' if (g1 <= 0 or g2 <= 0) else ''), flush=True)
    gaps1 = [g for d, g, _ in rows_out if d]
    gaps2 = [g for d, _, g in rows_out if d]
    print('    худший случай по деньгам %+.0f%% (монета %s), по равной просадке %+.0f%% (монета %s)'
          % (min(gaps1), rows_out[1 + int(np.argmin(gaps1))][0].replace('USDT', ''),
             min(gaps2), rows_out[1 + int(np.argmin(gaps2))][0].replace('USDT', '')), flush=True)
'''

src = head + TAIL
ast.parse(src)
io.open('loco.py', 'w', encoding='utf-8').write(src)
print('loco.py готов, синтаксис ок')
