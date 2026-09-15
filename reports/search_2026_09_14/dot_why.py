"""Why does one mediocre coin move the account by three quarters?

DOT averages +0.037R a trade on measured costs - unremarkable - yet removing it takes bank+rule from
$3,114 to $5,459. That is too much to come from the trades themselves, and the suspect is the
equal-drawdown comparison: removing a coin changes the path of the equity curve, the binary search
then permits a higher risk, and compounding multiplies the difference.

At a FIXED risk that mechanism is switched off. If the gap survives, DOT really is that costly. If
it collapses, the $5,459 is a property of the sizing search and not of the coin, and the honest
number is the fixed-risk one.
"""
import collections, sys
import numpy as np
sys.path.insert(0, 'C:/Users/Lenovo/Desktop/Торговля/crypto-bot')
import exec_model as EM
import live_rules_sim as L
import sim_w

ALL = EM.ALL
SETS = [
    ('без XLM и AAVE',        [s for s in ALL if s not in ('XLMUSDT', 'AAVEUSDT')]),
    ('и без DOT',             [s for s in ALL if s not in ('XLMUSDT', 'AAVEUSDT', 'DOTUSDT')]),
]
print('  === при ФИКСИРОВАННОМ риске (механизм подбора выключен) ===', flush=True)
print('  риск    без XLM/AAVE          и без DOT             разница', flush=True)
BOOKS = {lbl: EM.book(coins, EM.SIG_ALL, 'рынок')[0] for lbl, coins in SETS}
for risk in (0.008, 0.010, 0.0125, 0.014, 0.0155, 0.017):
    out = []
    for lbl, _ in SETS:
        r = sim_w.simulate(BOOKS[lbl], risk)
        out.append((r['eq'], abs(L.dd_of(r['curve'])), bool(r['paused_at'])))
    print('  %.2f%%  $%6.0f / %4.1f%% %s   $%6.0f / %4.1f%% %s   %+5.0f%%'
          % (100 * risk, out[0][0], 100 * out[0][1], 'З' if out[0][2] else ' ',
             out[1][0], 100 * out[1][1], 'З' if out[1][2] else ' ',
             100 * (out[1][0] / out[0][0] - 1)), flush=True)

print('', flush=True)
print('  === что вообще делают сделки DOT ===', flush=True)
tr = BOOKS['без XLM и AAVE']
dot = [x for x in tr if x[4] == 'DOTUSDT']
oth = [x for x in tr if x[4] != 'DOTUSDT']
for lbl, v in (('DOT', dot), ('остальные', oth)):
    a = np.array([x[2] for x in v])
    sf = np.array([x[3] for x in v])
    print('    %-10s n%4d ВР%5.1f%% ср%+.4fR сумма%+7.1fR | стоп медиана %.4f, 95%% %.4f'
          % (lbl, len(a), 100 * np.mean(a > 0), a.mean(), a.sum(),
             np.median(sf), np.quantile(sf, 0.95)), flush=True)

print('', flush=True)
print('  === и где именно DOT портит кривую ===', flush=True)
r1 = sim_w.simulate(BOOKS['без XLM и AAVE'], 0.014)
r2 = sim_w.simulate(BOOKS['и без DOT'], 0.014)
for lbl, r in (('с DOT', r1), ('без DOT', r2)):
    cur = np.array([e for t, e in r['curve']])
    ts = np.array([t for t, e in r['curve']])
    peak = np.maximum.accumulate(cur)
    dd = 1 - cur / peak
    j = int(np.argmin(cur / peak))
    import datetime
    print('    %-8s итог $%6.0f, худшая просадка %.1f%% около %s'
          % (lbl, r['eq'], 100 * dd[j],
             datetime.datetime.fromtimestamp(int(ts[j]), datetime.UTC).strftime('%y-%m-%d')),
          flush=True)
