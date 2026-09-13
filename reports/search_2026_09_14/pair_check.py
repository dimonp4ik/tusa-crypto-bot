"""The two extra rules, judged against what they are actually added to.

The incremental ladder stopped at two because that is where the money peaked, and the ladder read
all five years to decide. Every time this session has trusted a choice made that way, it has had to
be retracted. So the two are measured the only way that can settle it: against bank+pullback, which
is what they would really be added to, half-year by half-year, and year by year.

A rule that earns its place beats what it is added to in most windows. A rule that is noise wins
about half, and its yearly money moves around a lot more than its total suggests.
"""
import datetime
import pickle
import sys

import numpy as np

sys.path.insert(0, 'C:/Users/Lenovo/Desktop/Торговля/crypto-bot')
import gauntlet2 as G
import struct_params as SP
import stop_width as SW
import sim_w

PASS = dict((l, (s, c)) for l, s, c in pickle.load(open('gauntlet_pass.pkl', 'rb')))
A = 'SHORT pos24<=0.1649 + rsi48>=46.2810'
B = 'LONG btc6>=0.0126 + ema_slope<=0.5046'
bank = SW.run(3.0, 1.0, coins=G.ALL, with_meta=True)
S = {'ОТКАТ': G.simulate([('rsi48', '>=', 56.3761), ('rsi6', '<=', 33.7947)], True)[0]}
for lbl in (A, B):
    s, c = PASS[lbl]
    S[lbl] = G.simulate(c, s)[0]


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


V = {'банк': merge([]), 'банк+откат': merge(['ОТКАТ']),
     '+шорт pos24': merge(['ОТКАТ', A]), '+лонг btc6': merge(['ОТКАТ', B]),
     '+оба': merge(['ОТКАТ', A, B])}
R3 = {k: [(a, b, R) for a, b, R, sf, s, t in v] for k, v in V.items()}

print('  === полугодие за полугодием, всё против БАНКА+ОТКАТА ===', flush=True)
ref = R3['банк+откат']
first, last = min(r[0] for r in ref), max(r[0] for r in ref)
score = {k: [0, 0, 0] for k in V if k != 'банк+откат'}
t = first
while t < last:
    a, b = t, t + 182 * 86400
    s0 = SP.stats(ref, lo=a, hi=b)
    if s0:
        line = []
        for k in ('банк', '+шорт pos24', '+лонг btc6', '+оба'):
            s1 = SP.stats(R3[k], lo=a, hi=b)
            if s1:
                score[k][0] += int(s1['r_mo'] > s0['r_mo'])
                score[k][1] += int(s1['dd'] > s0['dd'])
                score[k][2] += 1
                line.append('%s%+6.2f/%+5.1f' % (k[:12], s1['r_mo'], s1['dd']))
        print('    %s  опора%+6.2f/%+5.1f | %s'
              % (datetime.datetime.fromtimestamp(a, datetime.UTC).strftime('%y-%m'),
                 s0['r_mo'], s0['dd'], '  '.join(line)), flush=True)
    t = b
print('', flush=True)
for k, (wr, wd, n) in score.items():
    print('    %-14s лучше банка+отката: R/мес %d/%d, просадка %d/%d' % (k, wr, n, wd, n),
          flush=True)

print('', flush=True)
print('  === деньги по годам (каждый год считается отдельным счётом) ===', flush=True)
YT = {y: int(datetime.datetime(y, 1, 1, tzinfo=datetime.UTC).timestamp()) for y in range(2022, 2028)}
print('  вариант        %s' % ' '.join('%9d' % y for y in range(2022, 2027)), flush=True)
for k in ('банк', 'банк+откат', '+шорт pos24', '+лонг btc6', '+оба'):
    cells = []
    for y in range(2022, 2027):
        sub = [(a, b, R, sf, s, 1.0) for a, b, R, sf, s, t in V[k]
               if YT[y] <= a < YT[y + 1]]
        if len(sub) < 50:
            cells.append('        -')
            continue
        kk, mm = sim_w.money_at_dd(sub, 0.12)
        cells.append('%9.0f' % mm)
    print('  %-14s %s' % (k, ' '.join(cells)), flush=True)

print('', flush=True)
print('  === и то же самое целиком ===', flush=True)
for k in ('банк', 'банк+откат', '+шорт pos24', '+лонг btc6', '+оба'):
    rows = [(a, b, R, sf, s, 1.0) for a, b, R, sf, s, t in V[k]]
    kk, mm = sim_w.money_at_dd(rows, 0.12)
    print('    %-14s %4d сделок риск %.3f%% $%6.0f' % (k, len(rows), 100 * kk, mm), flush=True)
