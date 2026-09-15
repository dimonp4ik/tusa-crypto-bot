"""Every conclusion reached at "equal drawdown" re-checked at a fixed risk setting.

Removing DOT looked worth +75% at equal drawdown and costs 8-14% at a fixed risk: the gain came
entirely from the lower drawdown permitting a higher risk, and only materialises if the user
actually raises the risk. AAVE, the pullback rule and the stop cap were all judged the same way, so
they all need the same second look.

Equal drawdown is the right comparison for someone who re-levers to their risk appetite. A fixed
risk is the right one for someone who sets 1.4% and leaves it. Both are legitimate; what is not
legitimate is quoting one and meaning the other.
"""
import collections, sys
import numpy as np
sys.path.insert(0, 'C:/Users/Lenovo/Desktop/Торговля/crypto-bot')
import exec_model as EM
import live_rules_sim as L
import sim_w

ALL = EM.ALL
BASE = [s for s in ALL if s not in ('XLMUSDT',)]          # XLM is refused on both days, keep it out
CASES = [
    ('банк один, все монеты',      EM.book(BASE, EM.SIG_BANK, 'рынок')[0]),
    ('банк+откат, все монеты',     EM.book(BASE, EM.SIG_ALL, 'рынок')[0]),
    ('банк+откат, без AAVE',       EM.book([s for s in BASE if s != 'AAVEUSDT'], EM.SIG_ALL, 'рынок')[0]),
    ('банк+откат, без AAVE и DOT', EM.book([s for s in BASE if s not in ('AAVEUSDT', 'DOTUSDT')],
                                           EM.SIG_ALL, 'рынок')[0]),
]
CASES.append(('банк+откат, без AAVE, стоп<=0.10',
              [x for x in CASES[2][1] if x[3] <= 0.10]))
RISKS = (0.010, 0.0125, 0.014, 0.0155, 0.017)
print('  вариант                             %s' % '  '.join('%7.2f%%' % (100 * r) for r in RISKS),
      flush=True)
res = {}
for lbl, tr in CASES:
    cells = []
    for risk in RISKS:
        r = sim_w.simulate(tr, risk)
        cells.append((r['eq'], abs(L.dd_of(r['curve'])), bool(r['paused_at'])))
    res[lbl] = cells
    print('  %-34s %s' % (lbl, '  '.join('%7.0f%s' % (c[0], 'З' if c[2] else ' ') for c in cells)),
          flush=True)
print('  просадка:', flush=True)
for lbl, tr in CASES:
    print('  %-34s %s' % (lbl, '  '.join('%7.1f ' % (100 * c[1]) for c in res[lbl])), flush=True)

print('', flush=True)
print('  === то же, сказанное как вклад каждой правки при ФИКСИРОВАННОМ риске 1.40%% ===',
      flush=True)
i = RISKS.index(0.014)
b = res['банк один, все монеты'][i][0]
for lbl in ('банк+откат, все монеты', 'банк+откат, без AAVE', 'банк+откат, без AAVE и DOT',
            'банк+откат, без AAVE, стоп<=0.10'):
    print('    %-34s $%6.0f  (x%.2f к банку, просадка %.1f%%)'
          % (lbl, res[lbl][i][0], res[lbl][i][0] / b, 100 * res[lbl][i][1]), flush=True)

print('', flush=True)
print('  === и при равной просадке, для сравнения ===', flush=True)
for lbl, tr in CASES:
    k, m = sim_w.money_at_dd(tr, 0.12)
    print('    %-34s риск %.3f%%  $%6.0f' % (lbl, 100 * k, m), flush=True)
