"""Is "all four" robust, or is it standing on a knife edge?

Two of the four latch the account's drawdown guard when added to the bank alone, and stop latching
when a third is present. That is not a property of a rule - that is the path of the equity curve
moving a few percent and stepping back off a cliff. A result that depends on which side of the
cliff you land is not a result.

So: the same variants across a fine ladder of risk, and each rule knocked out of the full set in
turn. A rule that earns its place shows up as money lost when it is removed, at every risk setting,
not just at one.
"""
import datetime
import sys

import numpy as np

sys.path.insert(0, 'C:/Users/Lenovo/Desktop/Торговля/crypto-bot')
import gauntlet2 as G
import struct_params as SP
import stop_width as SW
import margin_cap as MC
import live_rules_sim as L

RULES = {
    'ОТКАТ': ([('rsi48', '>=', 56.3761), ('rsi6', '<=', 33.7947)], True),
    'ПАДЕНИЕ': ([('ret24', '<=', -1.5434), ('vol720', '>=', 1.4462)], True),
    'КАПИТУЛЯЦИЯ': ([('breadth', '<=', 0.0625), ('vol720', '>=', 1.4462)], True),
    'РАСХОЖДЕНИЕ': ([('btc168', '<=', -0.0558), ('pos48', '>=', 0.8794)], False),
}
S = {nm: G.simulate(c, lg)[0] for nm, (c, lg) in RULES.items()}
bank = SW.run(3.0, 1.0, coins=G.ALL, with_meta=True)
ALL4 = ['ОТКАТ', 'ПАДЕНИЕ', 'КАПИТУЛЯЦИЯ', 'РАСХОЖДЕНИЕ']


def merge(names):
    ev = [(a, b, R, sf, s, 'bank') for a, b, R, sf, s in bank]
    for nm in names:
        ev += [(a, b, R, sf, s, nm) for a, b, R, sf, s in S[nm]]
    pri = {'bank': 0}
    pri.update({nm: i + 1 for i, nm in enumerate(names)})
    ev.sort(key=lambda x: (x[0], pri[x[5]]))
    busy, out = {}, []
    for a, b, R, sf, s, tag in ev:
        if busy.get(s, 0) > a:
            continue
        busy[s] = b
        out.append((a, b, R, sf, s, tag))
    out.sort()
    return out


LAD = [0.010, 0.0115, 0.0125, 0.0135, 0.014, 0.0145, 0.015, 0.0155, 0.016, 0.017, 0.018]
VAR = [('банк', []), ('+откат', ['ОТКАТ']), ('+падение', ['ПАДЕНИЕ']),
       ('+капитуляция', ['КАПИТУЛЯЦИЯ']), ('+расхождение', ['РАСХОЖДЕНИЕ']),
       ('все четыре', ALL4),
       ('без падения', ['ОТКАТ', 'КАПИТУЛЯЦИЯ', 'РАСХОЖДЕНИЕ']),
       ('без капитуляц', ['ОТКАТ', 'ПАДЕНИЕ', 'РАСХОЖДЕНИЕ']),
       ('без расхожд', ['ОТКАТ', 'ПАДЕНИЕ', 'КАПИТУЛЯЦИЯ']),
       ('без отката', ['ПАДЕНИЕ', 'КАПИТУЛЯЦИЯ', 'РАСХОЖДЕНИЕ'])]

print('  === деньги по лестнице риска ($; З = защёлка сработала) ===', flush=True)
print('  %-14s %s' % ('вариант', ' '.join('%7.2f%%' % (100 * r) for r in LAD)), flush=True)
KEEP, MONEY = {}, {}
for nm, names in VAR:
    rows = merge(names)
    KEEP[nm] = rows
    MC.trades = [(a, b, R, sf, s) for a, b, R, sf, s, _ in rows]
    cells, mm = [], []
    for risk in LAD:
        r = L.simulate(target=risk)
        mm.append(r['eq'])
        cells.append('%7s' % ('%.0fЗ' % r['eq'] if r['paused_at'] else '%.0f' % r['eq']))
    MONEY[nm] = mm
    print('  %-14s %s' % (nm, ' '.join(cells)), flush=True)

print('', flush=True)
print('  === просадка по той же лестнице (%) ===', flush=True)
for nm, names in VAR:
    MC.trades = [(a, b, R, sf, s) for a, b, R, sf, s, _ in KEEP[nm]]
    cells = []
    for risk in LAD:
        r = L.simulate(target=risk)
        cells.append('%7.1f' % (100 * L.dd_of(r['curve'])))
    print('  %-14s %s' % (nm, ' '.join(cells)), flush=True)

print('', flush=True)
print('  === выбивание по одному: сколько денег теряется без правила ===', flush=True)
for nm in ('без отката', 'без падения', 'без капитуляц', 'без расхожд'):
    d = [MONEY['все четыре'][i] / max(MONEY[nm][i], 1) for i in range(len(LAD))]
    print('    %-14s все четыре / %s = %s  (медиана %.2fx)'
          % (nm, nm, ' '.join('%.2f' % x for x in d), float(np.median(d))), flush=True)

print('', flush=True)
print('  === что именно защёлкивает +падение: худшее окно ===', flush=True)
for nm in ('банк', '+падение', 'все четыре'):
    rows = KEEP[nm]
    MC.trades = [(a, b, R, sf, s) for a, b, R, sf, s, _ in rows]
    r = L.simulate(target=0.014)
    cur = np.array(r['curve'], dtype=float)
    peak = np.maximum.accumulate(cur)
    dd = cur / peak - 1
    j = int(np.argmin(dd))
    i = int(np.argmax(peak[:j + 1] == peak[j])) if j else 0
    ts = [x[0] for x in rows][:len(cur)]
    f = lambda k: datetime.datetime.fromtimestamp(int(ts[min(k, len(ts) - 1)]),
                                                  datetime.UTC).strftime('%y-%m-%d')
    seg = rows[i:j + 1]
    byt = {}
    for x in seg:
        byt[x[5]] = byt.get(x[5], 0) + 1
    print('    %-12s просадка %5.1f%%  %s..%s  сделок в яме %d  состав %s'
          % (nm, 100 * dd[j], f(i), f(j), len(seg),
             ' '.join('%s:%d' % kv for kv in sorted(byt.items(), key=lambda kv: -kv[1]))),
          flush=True)
