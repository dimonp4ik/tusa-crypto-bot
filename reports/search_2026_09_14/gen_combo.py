"""Writes combo.py: the two market-breadth findings together.

Two rules exploit the same moment - a pullback across many coins at once:
  item 8  - size x1.25 when 10+ trades open in the same bar (in the proposal);
  wide    - when the strict 1h pullback fires on K >= 4 coins in the same hour, also take coins with
            relaxed thresholds (53/38 or 50/45).
The wide source adds trades to exactly those hours, so it also raises n for the boost - they interact
and must be measured together, not added up.

Variants: system; + boost; + wide 53/38 K4; + wide 50/45 K4; + boost + wide 53/38 K4;
+ boost + wide 50/45 K4. n is counted on the trades that actually opened (after the fill).
Three books; full fill and ten paired 85% fills; money at 1.4% and at 12% drawdown; per-year wins;
rolling choice among all variants by money at 12% drawdown on four years, measured at 1.4% on the fifth.
"""
import ast
import io

s = io.open('breadth.py', encoding='utf-8').read()
head = s[:s.index("RELAX = ")]

TAIL = r'''
BOOKS3 = (('КНИГА 1 (без XLM, AAVE)', [c for c in SP.COINS if c not in ('BILLUSDT', 'XLMUSDT', 'AAVEUSDT')]),
          ('КНИГА 2 (все 15 монет)', [c for c in SP.COINS if c != 'BILLUSDT']),
          ('КНИГА 3 (без AAVE, с XLM)', [c for c in SP.COINS if c not in ('BILLUSDT', 'AAVEUSDT')]))


def boost(rows, nmin=10, mult=1.25):
    cnt = collections.Counter(x[0] for x in rows)
    return [x[:5] + ((mult if cnt[x[0]] >= nmin else 1.0),) for x in rows]


VARS = (('система (сейчас)', None, False),
        ('+ буст n>=10', None, False),
        ('+ ширина 53/38 K4', 'w53', False),
        ('+ ширина 50/45 K4', 'w50', False),
        ('+ буст + ширина 53/38 K4', 'w53', True),
        ('+ буст + ширина 50/45 K4', 'w50', True))
BOOSTED = {'+ буст n>=10'}


def build(sig_all, wide_key, do_boost, lbl):
    keys = BASE_KEYS | ({wide_key} if wide_key else set())
    rows, _ = book(sig_all, keys)
    if do_boost or lbl in BOOSTED:
        rows = boost(rows)
    return rows


for book_name, coins in BOOKS3:
    H = {s: coin_hours(s) for s in coins}
    base_sig = make_sig_raw(coins)
    extra = wide_entries(coins, H, 'w53', 3, 53.0, 38.0, 4, 0) + wide_entries(coins, H, 'w50', 4, 50.0, 45.0, 4, 0)
    sig_all = sorted(base_sig + extra, key=lambda x: (x[0], x[1]))
    print('', flush=True)
    print('  ===== %s =====' % book_name, flush=True)
    RES = {}
    for lbl, wk, bo in VARS:
        rows = build(sig_all, wk, bo, lbl)
        RES[lbl] = rows
        eq, dd = fix2(rows)
        m = sim_w.money_at_dd(rows, 0.12)[1]
        v = np.array([x[2] for x in rows])
        yrs = (rows[-1][0] - rows[0][0]) / (365.25 * 86400)
        print('    100%% %-28s %4.0f сд/год ВР %4.1f%% | $%6.0f %4.1f%% | DD12 $%6.0f' % (lbl, len(v) / yrs, 100 * np.mean(v > 0), eq, 100 * dd, m), flush=True)
    R = {lbl: [] for lbl, _, _ in VARS}
    for seed in range(1, 11):
        rng = np.random.default_rng(seed)
        sub = [x for x in sig_all if rng.random() < 0.85]
        for lbl, wk, bo in VARS:
            rows = build(sub, wk, bo, lbl)
            e, d = fix2(rows)
            v = np.array([x[2] for x in rows])
            months = (rows[-1][1] - rows[0][0]) / (365.25 * 86400 / 12)
            R[lbl].append(dict(eq=e, dd=d, m=sim_w.money_at_dd(rows, 0.12)[1], yrs=pyr(rows), wr=float(np.mean(v > 0)),
                               tpy=len(v) / (months / 12), mo=(e / 120) ** (1 / months) - 1))
    base = R['система (сейчас)']
    med = lambda X, k: float(np.median([x[k] for x in X]))
    print('    85%% заливки, медиана 10 розыгрышей:', flush=True)
    for lbl, _, _ in VARS:
        X = R[lbl]
        yw = ' '.join('%d/10' % sum(1 for a, x in zip(base, X) if x['yrs'][k] > a['yrs'][k]) for k in range(5))
        print('      %-28s %4.0f сд/год ВР %4.1f%% %+.2f%%/мес $%6.0f %4.1f%% (макс %4.1f%%) DD12 $%6.0f | лучше $ %2d/10 DD12 %2d/10 | по годам %s'
              % (lbl, med(X, 'tpy'), 100 * med(X, 'wr'), 100 * med(X, 'mo'), med(X, 'eq'), 100 * med(X, 'dd'), 100 * max(x['dd'] for x in X),
                 med(X, 'm'), sum(1 for a, x in zip(base, X) if x['eq'] > a['eq']), sum(1 for a, x in zip(base, X) if x['m'] > a['m']), yw), flush=True)
    print('    СКОЛЬЗЯЩИЙ ВЫБОР (полная заливка): вариант по $ при DD12 на 4 годах, замер 5-го при 1.4%', flush=True)
    wins = 0
    for y in range(2022, 2027):
        lo, hi = YT[y], YT[y + 1]
        best, bk = -1, None
        for lbl, rows in RES.items():
            mm = sim_w.money_at_dd([x for x in rows if not (lo <= x[0] < hi)], 0.12)[1]
            if mm > best:
                best, bk = mm, lbl
        mp = sim_w.simulate([x for x in RES[bk] if lo <= x[0] < hi], 0.014)['eq']
        mb = sim_w.simulate([x for x in RES['система (сейчас)'] if lo <= x[0] < hi], 0.014)['eq']
        wins += int(mp > mb)
        print('      %d выбрано «%s» | $%.0f против $%.0f %s' % (y, bk, mp, mb, 'лучше' if mp > mb else ('так же' if mp == mb else 'хуже')), flush=True)
    print('      лучше системы в %d из 5 лет' % wins, flush=True)
'''

src = head + TAIL
ast.parse(src)
io.open('combo.py', 'w', encoding='utf-8').write(src)
print('combo.py готов, синтаксис ок')
