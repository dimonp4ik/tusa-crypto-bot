"""Refuse trades whose stop sits beyond liquidation. What does the caution cost?

At 10x isolated margin a 10% adverse move wipes the position, and 97 trades of 2,476 have a stop
further away than that. The exchange would close them before the bot's stop fires, charging a
liquidation fee the model does not know about, and the bot would lose control of the exit.

The shrink does not help: it scales the position, and the liquidation distance is set by the
leverage, not the size.

The fix is one comparison at signal time - the stop distance is known before entering - so it is
implementable. What it costs is the question: those trades are mildly profitable in the model.
Measured on four books, because that is the standard now.
"""
import collections, sys
import numpy as np
sys.path.insert(0, 'C:/Users/Lenovo/Desktop/Торговля/crypto-bot')
import exec_model as EM
import live_rules_sim as L
import sim_w

YT = EM.YT
BOOKS = [
    ('без AAVE и XLM, банк+откат', EM.book(EM.NOBAD, EM.SIG_ALL, 'рынок')[0]),
    ('все 15, банк+откат',          EM.book(EM.ALL,   EM.SIG_ALL, 'рынок')[0]),
    ('без AAVE и XLM, банк один',   EM.book(EM.NOBAD, EM.SIG_BANK, 'рынок')[0]),
    ('все 15, банк один',           EM.book(EM.ALL,   EM.SIG_BANK, 'рынок')[0]),
]
CAPS = [None, 0.15, 0.12, 0.10, 0.09, 0.08, 0.06]
print('  порог стопа: %s' % ' '.join('%7s' % ('нет' if c is None else '%.2f' % c) for c in CAPS),
      flush=True)
rows_all = []
for lbl, tr in BOOKS:
    cells, cnt = [], []
    for cap in CAPS:
        sub = tr if cap is None else [x for x in tr if x[3] <= cap]
        cells.append(sim_w.money_at_dd(sub, 0.12)[1])
        cnt.append(len(tr) - len(sub))
    best = max(range(len(CAPS)), key=lambda i: cells[i])
    print('  %-28s %s | лучший %s'
          % (lbl, ' '.join('%7.0f%s' % (c, '*' if i == best else ' ') for i, c in enumerate(cells)),
             'нет' if CAPS[best] is None else '%.2f' % CAPS[best]), flush=True)
    print('  %-28s %s | отсеяно сделок' % ('', ' '.join('%7d ' % c for c in cnt)), flush=True)
    rows_all.append(cells)

print('', flush=True)
print('  === место каждого порога по четырём книгам ===', flush=True)
rank = collections.defaultdict(list); wins = collections.Counter()
for cells in rows_all:
    order = sorted(range(len(CAPS)), key=lambda i: -cells[i])
    wins[CAPS[order[0]]] += 1
    for place, i in enumerate(order):
        rank[CAPS[i]].append(place + 1)
for c in CAPS:
    lbl = 'нет' if c is None else '%.2f' % c
    print('    %-5s лучший %d раз, среднее место %.1f, худшее %d'
          % (lbl, wins[c], float(np.mean(rank[c])), max(rank[c])), flush=True)

print('', flush=True)
print('  === и по годам для книги 1 ===', flush=True)
tr = BOOKS[0][1]
for cap in (None, 0.10, 0.09):
    per = []
    for y in range(2022, 2027):
        sub = [x for x in tr if YT[y] <= x[0] < YT[y+1] and (cap is None or x[3] <= cap)]
        per.append(sim_w.money_at_dd(sub, 0.12)[1] if len(sub) >= 50 else float('nan'))
    print('    %-5s %s' % ('нет' if cap is None else '%.2f' % cap,
                           ' '.join('%4.0f' % x if np.isfinite(x) else '   -' for x in per)),
          flush=True)
