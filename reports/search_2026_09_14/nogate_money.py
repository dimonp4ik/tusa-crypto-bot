"""The gateless pullback rule as money, and the walk-forward that has killed everything else.

Dropping the regime gate gives the rule a third more trades at the same edge per trade, and every
attack it passed before it now passes harder - the mirror loses 0.1390R instead of hovering near
zero, because outside the gate there is no regime keeping the other side afloat.

What remains is the account. The bank keeps its own gate untouched and first claim on every slot;
only the pullback rule is freed. Money at equal drawdown, each year as its own account, and nine
forward half-years. The blind re-pick's preference for rsi6 <= 30 is priced too, because stricter
thresholds have bought R and lost money three times today.
"""
import collections
import datetime
import sys

import numpy as np

sys.path.insert(0, 'C:/Users/Lenovo/Desktop/Торговля/crypto-bot')
import struct_params as SP
import stop_width as SW
import sim_w
import nogate_attack as NA
import pullback_nogate as PN

YT = {y: int(datetime.datetime(y, 1, 1, tzinfo=datetime.UTC).timestamp()) for y in range(2022, 2028)}
bank = SW.run(3.0, 1.0, coins=PN.ALL, with_meta=True)


def account(cand):
    ev = [(a, b, R, sf, s, 0) for a, b, R, sf, s in bank] + \
         [(a, b, R, sf, s, 1) for a, b, R, sf, s in cand]
    ev.sort(key=lambda x: (x[0], x[5]))
    busy, out = {}, []
    for a, b, R, sf, s, t in ev:
        if busy.get(s, 0) > a:
            continue
        busy[s] = b
        out.append((a, b, R, sf, s, 1.0))
    out.sort()
    return out


VAR = [
    ('банк один', None, None),
    ('откат в гейте', 'гейт', [('rsi48', '>=', 56.3761), ('rsi6', '<=', 33.7947)]),
    ('откат без гейта', 'без режима', [('rsi48', '>=', 56.3761), ('rsi6', '<=', 33.7947)]),
    ('без гейта, rsi6<=30', 'без режима', [('rsi48', '>=', 56.3761), ('rsi6', '<=', 30.0)]),
    ('без гейта, rsi48>=59', 'без режима', [('rsi48', '>=', 59.0), ('rsi6', '<=', 33.7947)]),
    ('без гейта, rsi6<=37', 'без режима', [('rsi48', '>=', 56.3761), ('rsi6', '<=', 37.0)]),
]
ROWS = {}
print('  вариант                 своих  сделок  риск     деньги | по годам', flush=True)
for lbl, mode, conds in VAR:
    cand = NA.trades(mode, conds) if mode else []
    acc = account(cand)
    ROWS[lbl] = acc
    k, m = sim_w.money_at_dd(acc, 0.12)
    per = []
    for y in range(2022, 2027):
        sub = [r for r in acc if YT[y] <= r[0] < YT[y + 1]]
        per.append('%d $%3.0f' % (y, sim_w.money_at_dd(sub, 0.12)[1]) if len(sub) >= 50
                   else '%d   -' % y)
    print('  %-22s %5d %6d  %.3f%%  $%6.0f | %s'
          % (lbl, len(cand), len(acc), 100 * k, m, ' '.join(per)), flush=True)

print('', flush=True)
print('  === скользящий прогон против банка ===', flush=True)
b3 = [(a, b, R) for a, b, R, sf, s, w in ROWS['банк один']]
first, last = min(r[0] for r in b3), max(r[0] for r in b3)
for lbl in ('откат в гейте', 'откат без гейта', 'без гейта, rsi6<=30'):
    c3 = [(a, b, R) for a, b, R, sf, s, w in ROWS[lbl]]
    t, wr, wd, seen = first, 0, 0, 0
    lines = []
    while t < last:
        a, b = t, t + 182 * 86400
        s0, s1 = SP.stats(b3, lo=a, hi=b), SP.stats(c3, lo=a, hi=b)
        if s0 and s1:
            seen += 1
            wr += int(s1['r_mo'] > s0['r_mo'])
            wd += int(s1['dd'] > s0['dd'])
            lines.append('%s %+.2f/%+.2f' % (
                datetime.datetime.fromtimestamp(a, datetime.UTC).strftime('%y-%m'),
                s0['r_mo'], s1['r_mo']))
        t = b
    print('    %-22s R/мес %d/%d, просадка %d/%d' % (lbl, wr, seen, wd, seen), flush=True)
    print('      %s' % '  '.join(lines), flush=True)

print('', flush=True)
print('  === разброс по месяцам у варианта без гейта ===', flush=True)
cand = NA.trades('без режима', [('rsi48', '>=', 56.3761), ('rsi6', '<=', 33.7947)])
mons = collections.Counter(
    datetime.datetime.fromtimestamp(r[0], datetime.UTC).strftime('%y-%m') for r in cand)
print('    сделок %d в %d месяцах, максимум в одном %d (%.0f%%)'
      % (len(cand), len(mons), max(mons.values()), 100 * max(mons.values()) / len(cand)), flush=True)

print('', flush=True)
print('  === пересечение с банком ===', flush=True)
bk = {(a, s) for a, b, R, sf, s in bank}
ck = {(r[0], r[4]) for r in cand}
print('    сделок правила %d, из них банк берёт те же часы: %d' % (len(ck), len(ck & bk)), flush=True)
