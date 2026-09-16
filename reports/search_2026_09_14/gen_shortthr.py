"""Writes shortthr.py: are the short rules' thresholds still a plateau on the new system?

The shorts carry most of the money and most of the drawdown, and their numbers were mined years ago on
the old system: short_pop_in_downtrend takes rsi2 >= 71.02 and rsi14 <= 35.22, short_btc_up_morning
takes btc24 >= 0.01701 in hours 8-11. Geometry and holding window were re-measured and stand; the
thresholds themselves never were. If they sit on a cliff rather than a plateau, the shorts' decay
(mean R +0.21 -> +0.08, stops 19% -> 28%) may be the thresholds ageing.

Candidates are built once at the loosest thresholds (rsi2 >= 65 and rsi14 <= 41; btc24 >= 0.010 and
hours 7-12) and every cell of the grid filters that list, so the sweep is cheap. Everything else is the
final system (items 8-11). Per book: money at 1.4% and at 12% drawdown for each cell, with the live
cell marked; then the 85% fill for the two best cells of each rule and the rolling choice.
"""
import ast
import io

s = io.open('widestopref.py', encoding='utf-8').read()
head = s[:s.index("for bi, (book_name, coins) in enumerate(BOOKS3):")]

TAIL = r'''
import bisect
WIDE_REF = 0.060
SHORT_IDX = {k: r for k, r in enumerate(RULES) if r.get('side', 'LONG') == 'SHORT'}
POP = next(k for k, r in SHORT_IDX.items() if r.get('name') == 'short_pop_in_downtrend')
MOR = next(k for k, r in SHORT_IDX.items() if r.get('name') == 'short_btc_up_morning')
POP_A = (65.0, 68.0, 71.02, 74.0, 77.0)
POP_B = (30.0, 33.0, 35.22, 38.0, 41.0)
MOR_C = (0.010, 0.015, 0.01701, 0.020, 0.025)
MOR_H = ((7, 11), (8, 11), (8, 12), (9, 11))


def short_candidates(coins):
    """Every hour in the SHORT regime that could pass the loosest thresholds, with its feature values."""
    out = []
    for s in coins:
        c = SP.CTX[s]
        T1, F, iv, starts = c['T1'], c['F'], c['iv'], c['starts']
        t15, a15, pos = c['t15'], c['a15'], c['pos']
        slip = SLIP.get(s, 0.0003)
        for i in range(PB.MIN_HOURS, len(T1)):
            atr = F['atr'][i]
            if not np.isfinite(atr) or atr <= 0:
                continue
            close = int(T1[i]) + 3600
            j = bisect.bisect_right(starts, close) - 1
            if j < 0 or not (iv[j][0] <= close < iv[j][1]) or iv[j][2] != 'SHORT':
                continue
            r2, r14, b24 = float(F['rsi2'][i]), float(F['rsi14'][i]), float(F['btc24'][i])
            hour = int((int(T1[i]) // 3600) % 24)    # _match читает час НАЧАЛА бара, не закрытия
            pop_ok = r2 >= 65.0 and r14 <= 41.0
            mor_ok = b24 >= 0.010 and 7 <= hour <= 12
            if not (pop_ok or mor_ok):
                continue
            k = pos.get(close)
            if k is None or k + MAXH > len(t15):
                continue
            o, h, l, cc = (a15[k:k + MAXH, x] for x in range(4))
            e = o[0] * (1 - slip)
            fav, adv = (e - l) / atr, (h - e) / atr
            row = (close, 0, None, s, e, atr, fav, adv, np.maximum.accumulate(fav), np.maximum.accumulate(adv),
                   (e - o) / atr, (e - cc) / atr, t15[k:k + MAXH], slip, float('nan'), hour)
            out.append((row, r2, r14, b24, hour))
    return out


def short_entries(cands, a, b, cthr, hlo, hhi):
    out = []
    for row, r2, r14, b24, hour in cands:
        pop_ok = r2 >= a and r14 <= b
        mor_ok = b24 >= cthr and hlo <= hour <= hhi
        if not (pop_ok or mor_ok):
            continue
        # bank_signals берёт ПЕРВОЕ подходящее правило по порядку списка RULES
        k = min([i for i, ok in ((MOR, mor_ok), (POP, pop_ok)) if ok])
        out.append(row[:2] + ('r%d' % k,) + row[3:])
    return out


for book_name, coins in BOOKS3:
    H = {s: coin_hours(s) for s in coins}
    full_sig = make_sig_raw(coins)
    longs = [x for x in full_sig if x[2] not in ('r%d' % POP, 'r%d' % MOR)]
    extra = roll_entries(coins, H, 'r2_4', 4, 2, 4, 0)
    cands = short_candidates(coins)
    GEO.clear(); GEO.update(BASE_GEO)
    print('', flush=True)
    print('  ===== %s: кандидатов шорта %d =====' % (book_name, len(cands)), flush=True)

    def run(a, b, cthr, hlo, hhi):
        sig = sorted(longs + extra + short_entries(cands, a, b, cthr, hlo, hhi), key=lambda x: (x[0], x[1]))
        rf, kf = book_keyed(sig, FULL)
        fin = weigh_src(rf, kf, {'r2_4'}, 1.25)
        refs = [(WIDE_REF if k == 'r2_4' else sim_w.REF) for k in kf]
        r = sim_ref(fin, 0.014, refs)
        return r['eq'], abs(L.dd_of(r['curve'])), money_at_dd_ref(fin, refs), len(fin)

    beq, bdd, bm, bn = run(71.02, 35.22, 0.01701, 8, 11)
    print('    живые пороги: $%.0f %.1f%% DD12 $%.0f (%d сделок) — контроль сборки' % (beq, 100 * bdd, bm, bn), flush=True)
    print('    ПРАВИЛО short_pop (rsi2 >= A и rsi14 <= B), утреннее правило на живых порогах:', flush=True)
    print('      A\\B    ' + ''.join('%-22s' % ('<=%.2f' % b) for b in POP_B), flush=True)
    for a in POP_A:
        cells = []
        for b in POP_B:
            eq, dd, m, n = run(a, b, 0.01701, 8, 11)
            cells.append('%s$%6.0f %4.1f%% $%6.0f ' % ('*' if (eq > beq and m > bm) else (' ' if (a, b) != (71.02, 35.22) else '='), eq, 100 * dd, m))
        print('      >=%-5.2f %s' % (a, ''.join(cells)), flush=True)
    print('    ПРАВИЛО short_btc_up_morning (btc24 >= C, часы), short_pop на живых порогах:', flush=True)
    for hlo, hhi in MOR_H:
        cells = []
        for cthr in MOR_C:
            eq, dd, m, n = run(71.02, 35.22, cthr, hlo, hhi)
            cells.append('%s$%6.0f %4.1f%% $%6.0f ' % ('*' if (eq > beq and m > bm) else (' ' if (cthr, hlo, hhi) != (0.01701, 8, 11) else '='), eq, 100 * dd, m))
        print('      часы %d-%d  %s' % (hlo, hhi, ''.join(cells)), flush=True)
    print('      (столбцы C: %s)' % ', '.join('%.3f' % c for c in MOR_C), flush=True)
'''

src = head + TAIL
ast.parse(src)
io.open('shortthr.py', 'w', encoding='utf-8').write(src)
print('shortthr.py готов, синтаксис ок')
