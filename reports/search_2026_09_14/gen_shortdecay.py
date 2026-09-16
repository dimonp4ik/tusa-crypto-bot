"""Writes shortdecay.py: why do the shorts decay, and does a deeper downtrend gate fix it?

The drawdown attribution showed the short rules weakening year by year - mean R +0.21 -> +0.08, stops
19% -> 28% - while the threshold sweep just confirmed their entry conditions are already optimal and
their geometry and holding window were confirmed earlier. If the conditions are right and the edge still
fades, the cause is the state of the market when they fire, not the rule.

Part 1 - diagnosis of every short trade of the final system: per year, and split by how far BTC's daily
close sits below its SMA50 at entry (above the average, 0-2% below, 2-5%, 5-10%, more than 10%) and by
how long the SHORT regime has already lasted (under 24h, 1-3 days, 3-7 days, over a week).

Part 2 - the gate: take a short only when BTC is at least X below its SMA50 (X = 0 as today, 1%, 2%,
3%). Three books, money at 1.4% and at 12% drawdown, per-year against the final system. Control: the
same number of shorts removed at random (median of 5 draws) - it separates "this filter finds bad
shorts" from "fewer shorts is simply less risk".
"""
import ast
import io

s = io.open('widestopref.py', encoding='utf-8').read()
head = s[:s.index("for bi, (book_name, coins) in enumerate(BOOKS3):")]

TAIL = r'''
import bisect
import datetime as DT
WIDE_REF = 0.060
SHORT_KEYS = {'r%d' % k for k, r in enumerate(RULES) if r.get('side', 'LONG') == 'SHORT'}
GATES = (0.0, 0.01, 0.02, 0.03)


def btc_gap(coin, close):
    """Насколько дневное закрытие BTC ниже своей SMA50 на момент входа (доля), и возраст режима в часах."""
    c = SP.CTX[coin]
    bdt, bdc = c['bdt'], np.asarray(c['bdc'], dtype=float)
    sma = c['sma'][50] if 50 in c['sma'] else PB._sma(bdc, 50)
    d = int(np.searchsorted(bdt, close - 86400, side='right')) - 1
    gap = float('nan') if (d < 49 or not np.isfinite(sma[d]) or sma[d] <= 0) else float(1 - bdc[d] / sma[d])
    j = bisect.bisect_right(c['starts'], close) - 1
    age = (close - c['iv'][j][0]) / 3600 if j >= 0 and c['iv'][j][0] <= close < c['iv'][j][1] else float('nan')
    return gap, age


def gap_bucket(g):
    if not np.isfinite(g):
        return 'нет данных'
    if g <= 0:
        return 'BTC выше средней'
    return 'ниже 0-2%' if g <= 0.02 else 'ниже 2-5%' if g <= 0.05 else 'ниже 5-10%' if g <= 0.10 else 'ниже >10%'


def age_bucket(a):
    if not np.isfinite(a):
        return 'нет данных'
    return 'до 24ч' if a <= 24 else '1-3 дня' if a <= 72 else '3-7 дней' if a <= 168 else 'больше недели'


for book_name, coins in BOOKS3:
    H = {s: coin_hours(s) for s in coins}
    base_sig = make_sig_raw(coins)
    extra = roll_entries(coins, H, 'r2_4', 4, 2, 4, 0)
    sig_all = sorted(base_sig + extra, key=lambda x: (x[0], x[1]))
    GEO.clear(); GEO.update(BASE_GEO)
    rows, keys = book_keyed(sig_all, FULL)
    fin = weigh_src(rows, keys, {'r2_4'}, 1.25)
    refs = [(WIDE_REF if k == 'r2_4' else sim_w.REF) for k in keys]
    beq = sim_ref(fin, 0.014, refs); bdd = abs(L.dd_of(beq['curve'])); bm = money_at_dd_ref(fin, refs)
    sh = [(i, x) for i, (x, k) in enumerate(zip(rows, keys)) if k in SHORT_KEYS]
    info = {i: btc_gap(x[4], x[0]) for i, x in sh}
    print('', flush=True)
    print('  ===== %s: итоговая $%.0f %.1f%% DD12 $%.0f, шортов %d из %d ====='
          % (book_name, beq['eq'], 100 * bdd, bm, len(sh), len(rows)), flush=True)
    print('    ШОРТЫ ПО ГОДАМ: сделок / ВР / ср R / стопов', flush=True)
    for y in range(2022, 2027):
        v = [x[2] for i, x in sh if YT[y] <= x[0] < YT[y + 1]]
        if len(v) >= 10:
            print('      %d: %3d сд, ВР %4.1f%%, ср R %+.3f' % (y, len(v), 100 * np.mean(np.array(v) > 0), np.mean(v)), flush=True)
    print('    ПО ГЛУБИНЕ ПАДЕНИЯ BTC ПОД СРЕДНЕЙ (на момент входа):', flush=True)
    for b in ('BTC выше средней', 'ниже 0-2%', 'ниже 2-5%', 'ниже 5-10%', 'ниже >10%', 'нет данных'):
        v = [x[2] for i, x in sh if gap_bucket(info[i][0]) == b]
        if len(v) >= 10:
            print('      %-18s %4d сд, ВР %4.1f%%, ср R %+.3f' % (b, len(v), 100 * np.mean(np.array(v) > 0), np.mean(v)), flush=True)
    print('    ПО ВОЗРАСТУ НИСХОДЯЩЕГО РЕЖИМА:', flush=True)
    for b in ('до 24ч', '1-3 дня', '3-7 дней', 'больше недели', 'нет данных'):
        v = [x[2] for i, x in sh if age_bucket(info[i][1]) == b]
        if len(v) >= 10:
            print('      %-18s %4d сд, ВР %4.1f%%, ср R %+.3f' % (b, len(v), 100 * np.mean(np.array(v) > 0), np.mean(v)), flush=True)
    print('    ГЕЙТ «BTC ниже средней хотя бы на X» (0 = как сейчас):', flush=True)
    for g in GATES:
        drop = {i for i, x in sh if not (np.isfinite(info[i][0]) and info[i][0] >= g)}
        if g > 0 and not drop:
            continue
        keep = [j for j in range(len(rows)) if j not in drop]
        rr = [rows[j] for j in keep]; kk = [keys[j] for j in keep]
        f2 = weigh_src(rr, kk, {'r2_4'}, 1.25)
        r2 = [(WIDE_REF if k == 'r2_4' else sim_w.REF) for k in kk]
        r = sim_ref(f2, 0.014, r2)
        m = money_at_dd_ref(f2, r2)
        dropped_R = [rows[j][2] for j in drop]
        ce, cm = [], []
        if g > 0 and drop:
            idx_sh = [i for i, x in sh]
            for seed in range(5):
                rng = np.random.default_rng(seed)
                pick = set(rng.choice(idx_sh, size=len(drop), replace=False).tolist())
                kp = [j for j in range(len(rows)) if j not in pick]
                cr = [rows[j] for j in kp]; ck = [keys[j] for j in kp]
                cf = weigh_src(cr, ck, {'r2_4'}, 1.25)
                cr2 = [(WIDE_REF if k == 'r2_4' else sim_w.REF) for k in ck]
                ce.append(sim_ref(cf, 0.014, cr2)['eq']); cm.append(money_at_dd_ref(cf, cr2))
        print('      X=%.0f%%: убрано %3d шортов (их ср R %s) | $%6.0f %4.1f%% DD12 $%6.0f %s%s'
              % (100 * g, len(drop), ('%+.3f' % np.mean(dropped_R)) if dropped_R else '  -  ',
                 r['eq'], 100 * abs(L.dd_of(r['curve'])), m,
                 '+' if (r['eq'] > beq['eq'] and m > bm) else ' ',
                 ('| КОНТРОЛЬ случайные $%.0f DD12 $%.0f' % (np.median(ce), np.median(cm))) if ce else ''), flush=True)
'''

src = head + TAIL
ast.parse(src)
io.open('shortdecay.py', 'w', encoding='utf-8').write(src)
print('shortdecay.py готов, синтаксис ок')
