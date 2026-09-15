"""Writes fill2h.py: the 2h pullback shifted by 1h, under the realistic 85% fill.

finer.py found one extra grid that held on both books and both measures: 2h bars that start on odd
hours, stacked onto the current system. The gain is +8.5%/+10% at a fixed 1.4% risk with the same
drawdown, but only 3 years in 5 were better. Before it is proposed it has to survive the 85% fill
used for the final numbers: ten random fills, the same subset of signals given to both variants, so
every seed is a paired comparison. Reported per book: median money, drawdown, latches, money at 12%
drawdown, seeds where the variant wins, and per-year paired wins over all seeds.
"""
import ast
import io

s = io.open('finer.py', encoding='utf-8').read()
head = s[:s.index("BASE_KEYS = {")]

TAIL = r'''
BASE_KEYS = {'r0', 'r1', 'r2', 'r3', 'r4', 'o1h', 'o2h'}
HOLDS.update({'o2h60': 72})
BOOKS = (('КНИГА 1 (без XLM, AAVE)', [c for c in SP.COINS if c not in ('BILLUSDT', 'XLMUSDT', 'AAVEUSDT')]),
         ('КНИГА 2 (все 15 монет)', [c for c in SP.COINS if c != 'BILLUSDT']))
VARS = (('система (сейчас)', BASE_KEYS), ('+ 2ч со сдвигом 1ч', BASE_KEYS | {'o2h60'}))


def summary(rows):
    v = np.array([x[2] for x in rows])
    months = (rows[-1][1] - rows[0][0]) / (365.25 * 86400 / 12)
    r = sim_w.simulate(rows, 0.014)
    return dict(tpy=len(v) / (months / 12), wr=float(np.mean(v > 0)), eq=r['eq'],
                mo=(r['eq'] / 120) ** (1 / months) - 1, dd=abs(L.dd_of(r['curve'])), latch=bool(r['paused_at']),
                dd12=sim_w.money_at_dd(rows, 0.12)[1], yrs=pyr(rows))


for book_name, coins in BOOKS:
    sig = make_sig_raw(coins) + long_entries(coins, 'o2h60', 3, 7200, 3600)
    sig.sort(key=lambda x: (x[0], x[1]))
    print('', flush=True)
    print('  ===== %s: заливка 85%%, 10 розыгрышей, одинаковые подмножества =====' % book_name, flush=True)
    R = {lbl: [] for lbl, _ in VARS}
    for seed in range(1, 11):
        rng = np.random.default_rng(seed)
        sub = [x for x in sig if rng.random() < 0.85]
        for lbl, keys in VARS:
            R[lbl].append(summary(book(sub, keys)[0]))
    for lbl, _ in VARS:
        res = R[lbl]
        med = lambda k: float(np.median([x[k] for x in res]))
        print('    %-22s %4.0f сд/год ВР %.1f%% %+.2f%%/мес $%.0f просадка %.1f%% (макс %.1f%%, защёлок %d/10) | DD12 $%.0f'
              % (lbl, med('tpy'), 100 * med('wr'), 100 * med('mo'), med('eq'), 100 * med('dd'),
                 100 * max(x['dd'] for x in res), sum(1 for x in res if x['latch']), med('dd12')), flush=True)
    a, b = R[VARS[0][0]], R[VARS[1][0]]
    print('    розыгрышей, где новое лучше: $ при 1.4%% %d/10, DD12 %d/10, просадка не хуже %d/10'
          % (sum(1 for x, y in zip(a, b) if y['eq'] > x['eq']), sum(1 for x, y in zip(a, b) if y['dd12'] > x['dd12']),
             sum(1 for x, y in zip(a, b) if y['dd'] <= x['dd'] + 1e-9)), flush=True)
    yw = ['%d %d/10' % (2022 + k, sum(1 for x, y in zip(a, b) if y['yrs'][k] > x['yrs'][k])) for k in range(5)]
    print('    по годам (новое лучше в розыгрышах): %s' % '  '.join(yw), flush=True)
'''

src = head + TAIL
ast.parse(src)
io.open('fill2h.py', 'w', encoding='utf-8').write(src)
print('fill2h.py готов, синтаксис ок')
