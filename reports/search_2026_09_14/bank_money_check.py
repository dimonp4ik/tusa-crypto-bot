"""Two questions the blind test raised about the deployed bank, answered in money.

First: the blind re-pick wants every threshold stricter than the deployed one, in every fold. That
maximises mean R, and mean R is not what the account spends - the wider-target test today raised R
by 42% and halved the money. So the strict thresholds go through the real harness at equal drawdown.

Second: the night rule's window is (0,3), and given a free choice the blind re-pick took the evening
window every time. Either the night window carries its own edge or it is a weaker copy of the
evening one. Every four-hour window is ranked, per year, so the answer does not come from one year.

Nothing here is proposed. The bank's rules are deployed and the user has not asked for changes to
them.
"""
import collections
import datetime
import sys

import numpy as np

sys.path.insert(0, 'C:/Users/Lenovo/Desktop/Торговля/crypto-bot')
import struct_params as SP
import gauntlet2 as G
import sim_w
from src import pullback_bank as PB

YT = {y: int(datetime.datetime(y, 1, 1, tzinfo=datetime.UTC).timestamp()) for y in range(2022, 2028)}
BASE = SP.BASE
print('  правил в банке: %d' % len(BASE), flush=True)
for i, r in enumerate(BASE):
    print('    %d %s %s' % (i, r.get('name', '?'), r['conds']), flush=True)


def run(rules):
    """The bank's own trade generator with a modified rule list."""
    rows = []
    for s in G.ALL:
        sig = SP.bank_signals([dict(r, tp=1.0) for r in rules], s, 50)
        c = SP.CTX[s]
        slip = SP.SLIP_REAL.get(s, 0.0002)
        t15, A15, pos = c['t15'], c['a15'], c['pos']
        busy = 0
        for close in sorted(sig):
            if close < busy:
                continue
            ri, lim, atr = sig[close]
            lgs = rules[ri].get('side', 'LONG') == 'LONG'
            i = pos.get(close)
            if i is None or i + PB.HOLD_BARS > len(t15):
                continue
            o, h, l, cc = (A15[i:i + PB.HOLD_BARS, x] for x in range(4))
            if lgs:
                if o[0] <= lim:
                    e, touch = o[0] * (1 + slip), False
                elif l[0] <= lim:
                    e, touch = lim * (1 + slip), True
                else:
                    continue
                TP, SL = e + 1.0 * atr, e - 3.0 * atr
                hs, ht = l <= SL, h >= TP
            else:
                if o[0] >= lim:
                    e, touch = o[0] * (1 - slip), False
                elif h[0] >= lim:
                    e, touch = lim * (1 - slip), True
                else:
                    continue
                TP, SL = e - 1.0 * atr, e + 3.0 * atr
                hs, ht = h >= SL, l <= TP
            if touch:
                ht = ht.copy()
                ht[0] = False
            js = int(np.argmax(hs)) if hs.any() else 10 ** 9
            jt = int(np.argmax(ht)) if ht.any() else 10 ** 9
            if js <= jt and js < 10 ** 9:
                jj = js
                f_ = (min(SL, o[js]) * (1 - slip)) if lgs else (max(SL, o[js]) * (1 + slip))
            elif jt < 10 ** 9:
                jj, f_ = jt, (TP * (1 - slip) if lgs else TP * (1 + slip))
            else:
                jj = PB.HOLD_BARS - 1
                f_ = cc[-1] * (1 - slip) if lgs else cc[-1] * (1 + slip)
            ret = ((f_ / e - 1) if lgs else (1 - f_ / e)) - 0.0004
            rows.append((close, int(t15[i + jj]) + 900, ret / (3.0 * atr / e), 3.0 * atr / e, s, 1.0))
            busy = int(t15[i + jj]) + 900
    rows.sort()
    return rows


def show(lbl, rules):
    rows = run(rules)
    v = np.array([r[2] for r in rows])
    k, m = sim_w.money_at_dd(rows, 0.12)
    per = []
    for y in range(2022, 2027):
        sub = [r for r in rows if YT[y] <= r[0] < YT[y + 1]]
        if len(sub) >= 50:
            _, my = sim_w.money_at_dd(sub, 0.12)
            per.append('%d $%3.0f' % (y, my))
    print('  %-34s %4d сделок ср%+.4f риск %.3f%% $%6.0f | по годам %s'
          % (lbl, len(v), v.mean(), 100 * k, m, ' '.join(per)), flush=True)
    return m


print('', flush=True)
print('  === строгие пороги слепого переподбора, деньгами ===', flush=True)
m0 = show('рабочий банк', BASE)
STRICT = {'btc24': {0.04896: 0.060629, 0.01701: 0.024763}, 'ret24': {3.864: 6.1659}}
for i in range(len(BASE)):
    rules = [dict(r) for r in BASE]
    nm, op, t = rules[i]['conds'][0]
    if nm in STRICT and t in STRICT[nm]:
        rules[i] = dict(rules[i], conds=[(nm, op, STRICT[nm][t])] + list(rules[i]['conds'][1:]))
        show('строже только правило %d (%s)' % (i, BASE[i].get('name', '')), rules)
rules = [dict(r) for r in BASE]
for i in range(len(rules)):
    nm, op, t = rules[i]['conds'][0]
    if nm in STRICT and t in STRICT[nm]:
        rules[i] = dict(rules[i], conds=[(nm, op, STRICT[nm][t])] + list(rules[i]['conds'][1:]))
show('все строже сразу', rules)

print('', flush=True)
print('  === ночное окно: каждое четырёхчасовое окно для btc24>=0.04896 ===', flush=True)
res = {}
for s in range(24):
    w = (s, (s + 3) % 24)
    rules = [dict(BASE[2], conds=[('btc24', '>=', 0.04896), ('hour', 'in', w)])]
    rows = run(rules)
    if len(rows) < 80:
        continue
    v = np.array([r[2] for r in rows])
    yr = []
    for y in range(2022, 2027):
        sub = [r[2] for r in rows if YT[y] <= r[0] < YT[y + 1]]
        yr.append(np.mean(sub) if len(sub) >= 15 else np.nan)
    res[w] = (len(v), v.mean(), yr)
order = sorted(res, key=lambda w: -res[w][1])
print('    окно      сделок   ср R   | по годам', flush=True)
for w in order:
    n, mu, yr = res[w]
    mark = ''
    if w == (0, 3):
        mark = '  <<< РАБОЧЕЕ НОЧНОЕ'
    if w == (20, 23):
        mark = '  <<< рабочее вечернее'
    print('    %-8s  %5d  %+.4f | %s%s'
          % (str(w), n, mu, ' '.join('%+.3f' % x if np.isfinite(x) else '  -  ' for x in yr), mark),
          flush=True)
