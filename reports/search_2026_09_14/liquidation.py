"""Can the stop be further away than liquidation? The model would never notice.

Isolated margin at 10x: the position is liquidated when the loss eats the margin, which happens at
roughly a 10% adverse move in the instrument (less, once the maintenance margin and fees are taken
out). The bank's stop is three ATR away, expressed as a fraction of the entry price - the `sf` in
every trade row. If sf exceeds about 9%, the exchange closes the position before the stop is
reached, and the trade loses the whole margin instead of the 1R the model books.

The simulator has no liquidation, so it would show such a trade as a normal -1R stop. On a real
account it is a total loss of that position's margin.

Three things to find out: how wide the stops actually get, how much the stop_ref shrink protects,
and what the worst trades would have cost if liquidation were modelled.
"""
import collections, csv, sys
import numpy as np
sys.path.insert(0, 'C:/Users/Lenovo/Desktop/Торговля/crypto-bot')
import exec_model as EM
import live_rules_sim as L

LEV, REF = L.LEV, L.REF
TR = EM.book(EM.NOBAD, EM.SIG_ALL, 'рынок')[0]
sf = np.array([x[3] for x in TR])
R = np.array([x[2] for x in TR])
print('  плечо %g, опорный стоп %.4f, сделок %d' % (LEV, REF, len(TR)), flush=True)
print('', flush=True)
print('  === насколько широки стопы (доля цены входа) ===', flush=True)
for q in (0.5, 0.75, 0.9, 0.95, 0.99, 1.0):
    print('    %4.0f%% сделок стоп не шире %.4f  (это %.1f%% маржи при плече %g)'
          % (100 * q, np.quantile(sf, q), 100 * np.quantile(sf, q) * LEV, LEV), flush=True)

print('', flush=True)
print('  === где стоп дальше ликвидации ===', flush=True)
for lq in (0.10, 0.095, 0.09, 0.08):
    n = int((sf > lq).sum())
    print('    порог ликвидации %.1f%% хода: таких сделок %d (%.2f%%), из них убыточных %d'
          % (100 * lq, n, 100 * n / len(sf),
             int(((sf > lq) & (R < 0)).sum())), flush=True)

print('', flush=True)
print('  === что делает урезание по опорному стопу ===', flush=True)
eff = np.minimum(sf, REF)      # the shrink caps the RISK, not the stop distance
print('    сделок со стопом шире опорного: %d (%.0f%%) — их размер урезается в %.2f раза в среднем'
      % (int((sf > REF).sum()), 100 * (sf > REF).mean(),
         float(np.mean(sf[sf > REF] / REF)) if (sf > REF).any() else 1), flush=True)
print('    но урезается ДОЛЯ СЧЁТА под риском, а расстояние до ликвидации задаётся ПЛЕЧОМ', flush=True)
print('    и не меняется: позиция с плечом 10 сгорает на ходе ~10%% независимо от её размера',
      flush=True)

print('', flush=True)
print('  === худшие сделки: что модель записала и что было бы при ликвидации ===', flush=True)
idx = np.argsort(-sf)[:12]
print('    монета   стоп   ход до ликв.  записано R  было бы', flush=True)
for i in idx:
    a, b, r, s_, coin, w = TR[i]
    liq = 1.0 / LEV
    would = -1.0 if s_ <= liq else -(liq / s_)   # liquidation caps the loss at the margin
    mark = '  <- ликвидация раньше стопа' if s_ > liq else ''
    print('    %-7s %.4f   %.4f        %+.3f     %+.3f%s'
          % (coin.replace('USDT', ''), s_, liq, r, would if r < 0 else r, mark), flush=True)

worst = sf > 1.0 / LEV
if worst.any():
    print('', flush=True)
    print('    ИТОГО таких сделок %d, среди них убыточных %d; модель записала им суммарно %+.1fR'
          % (int(worst.sum()), int((worst & (R < 0)).sum()), float(R[worst].sum())), flush=True)
else:
    print('', flush=True)
    print('    сделок со стопом дальше ликвидации НЕТ — урезание и ширина стопов справляются',
          flush=True)
