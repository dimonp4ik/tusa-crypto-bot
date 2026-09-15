"""Which price moves does the system miss, and what did they look like before they started?

The market moves 7% a day on average and the system takes about 1.5 trades a day, so most of the
movement goes uncaught. That is not a problem in itself - most of it is unpredictable - but the moves
it misses may share a recognisable shape that no current rule is looking for.

For every coin and day: the largest long move (a low followed later that day by a high) and the
largest short move. A "big move" is a day in the coin's own top 20% of such moves. A big move counts
as caught when a trade of the right side was open at some point between its start and its peak, and
as caught near the start when a trade was entered within 6 hours of the start.

Then the conditions one hour before each big move started - volatility, RSI at several lengths, BTC,
trend slope, regime, hour, weekday - are compared across caught moves, missed moves, and a random
sample of ordinary hours. A feature where missed moves differ from ordinary hours, and differ from
caught moves, is where a new rule might live.

Book: the current best system from stops.py (stop_meta.pkl).
"""
import bisect
import collections
import datetime
import pickle
import sys

import numpy as np

sys.path.insert(0, 'C:/Users/Lenovo/Desktop/Торговля/crypto-bot')
import struct_params as SP
from src import pullback_bank as PB

T = pickle.load(open('stop_meta.pkl', 'rb'))
SHORT_KEYS = {'r%d' % k for k, r in enumerate(SP.BASE) if r.get('side', 'LONG') == 'SHORT'}
COINS = sorted({t['coin'] for t in T})
by_coin = collections.defaultdict(list)
for t in T:
    by_coin[t['coin']].append((t['close'], t['end'], t['key'] not in SHORT_KEYS, t['R'], t['kind']))
print('  сделок в книге %d, монет %d' % (len(T), len(COINS)), flush=True)

rng = np.random.default_rng(3)
MOVES = []          # dict per big move
BASE_HOURS = []     # (coin, hour_index) random ordinary hours
FEAT = {}
for s in COINS:
    c = SP.CTX[s]
    t15, a15 = c['t15'], c['a15']
    T1, B1, F = c['T1'], c['B1'], c['F']
    cl = B1[:, 3]
    FEAT[s] = dict(T1=T1, volreg=F['volreg'], rsi14=F['rsi14'], rsi2=F['rsi2'], slope=F['slope'],
                   btc24=F['btc24'], ret24=F['ret24'], rng=F['rng'],
                   rsi48=PB._rsi(cl, 48), rsi6=PB._rsi(cl, 6), iv=c['iv'], starts=c['starts'])
    day = t15 // 86400
    edges = np.flatnonzero(np.diff(day)) + 1
    bounds = np.r_[0, edges, len(t15)]
    daily = []
    for a, b in zip(bounds[:-1], bounds[1:]):
        if b - a < 80:
            continue
        lo, hi = a15[a:b, 2], a15[a:b, 1]
        cmin = np.minimum.accumulate(lo)
        gain = hi / cmin - 1
        k = int(np.argmax(gain))
        st = int(np.argmin(lo[:k + 1]))
        cmax = np.maximum.accumulate(hi)
        drop = 1 - lo / cmax
        k2 = int(np.argmax(drop))
        st2 = int(np.argmin(-hi[:k2 + 1]))
        daily.append((int(t15[a + st]), int(t15[a + k]), float(gain[k]), int(t15[a + st2]), int(t15[a + k2]), float(drop[k2])))
    if not daily:
        continue
    up_thr = np.quantile([d[2] for d in daily], 0.8)
    dn_thr = np.quantile([d[5] for d in daily], 0.8)
    trades = sorted(by_coin[s])
    for st, pk, g, st2, pk2, dr in daily:
        for lg, a0, a1, size, thr in ((True, st, pk, g, up_thr), (False, st2, pk2, dr, dn_thr)):
            if size < thr or a1 <= a0:
                continue
            over = [x for x in trades if x[2] == lg and x[0] < a1 and x[1] > a0]
            near = [x for x in trades if x[2] == lg and abs(x[0] - a0) <= 6 * 3600]
            MOVES.append(dict(coin=s, long=lg, start=a0, peak=a1, size=size, caught=bool(over), near=bool(near),
                              R=float(np.mean([x[3] for x in over])) if over else float('nan')))
    idx = rng.choice(np.arange(800, len(T1)), size=min(400, len(T1) - 800), replace=False)
    BASE_HOURS += [(s, int(i)) for i in idx]


def feats_at(s, ts):
    f = FEAT[s]
    i = int(np.searchsorted(f['T1'], ts - 3600, side='right')) - 1     # last hour fully closed before ts
    if i < 800 or i >= len(f['T1']):
        return None
    close = int(f['T1'][i]) + 3600
    j = bisect.bisect_right(f['starts'], close) - 1
    reg = f['iv'][j][2] if j >= 0 and f['iv'][j][0] <= close < f['iv'][j][1] else 'вне'
    d = datetime.datetime.fromtimestamp(ts, datetime.UTC)
    return dict(volreg=f['volreg'][i], rsi48=f['rsi48'][i], rsi6=f['rsi6'][i], rsi14=f['rsi14'][i],
                btc24=f['btc24'][i], slope=f['slope'][i], ret24=f['ret24'][i], rng=f['rng'][i],
                hour=d.hour, wd=d.weekday(), reg=reg)


for lg, name in ((True, 'ХОДЫ ВВЕРХ'), (False, 'ХОДЫ ВНИЗ')):
    mv = [m for m in MOVES if m['long'] == lg]
    caught = [m for m in mv if m['caught']]
    near = [m for m in mv if m['near']]
    print('', flush=True)
    print('  ===== %s: крупных %d (верхние 20%% дней каждой монеты), средний размер %.1f%% =====' % (name, len(mv), 100 * np.mean([m['size'] for m in mv])), flush=True)
    print('    поймано (была открыта сделка нужной стороны): %.1f%%, вход в пределах 6ч от начала: %.1f%%'
          % (100 * len(caught) / len(mv), 100 * len(near) / len(mv)), flush=True)
    print('    средний R сделок, поймавших ход: %+.3f' % np.nanmean([m['R'] for m in caught]), flush=True)
    yrs = collections.defaultdict(lambda: [0, 0])
    for m in mv:
        y = datetime.datetime.fromtimestamp(m['start'], datetime.UTC).year
        yrs[y][0] += 1; yrs[y][1] += int(m['caught'])
    print('    поймано по годам: %s' % '  '.join('%d %.0f%%' % (y, 100 * v[1] / v[0]) for y, v in sorted(yrs.items())), flush=True)
    groups = {'пропущенные': [feats_at(m['coin'], m['start']) for m in mv if not m['caught']],
              'пойманные': [feats_at(m['coin'], m['start']) for m in mv if m['caught']],
              'обычные часы': [feats_at(s, int(FEAT[s]['T1'][i]) + 3600 + 1) for s, i in BASE_HOURS]}
    groups = {k: [x for x in v if x] for k, v in groups.items()}
    print('    признак за час до начала хода      пропущенные           пойманные             обычные часы', flush=True)
    for fn in ('volreg', 'rsi48', 'rsi6', 'rsi14', 'btc24', 'slope', 'ret24', 'rng'):
        cells = []
        for k in ('пропущенные', 'пойманные', 'обычные часы'):
            v = np.array([x[fn] for x in groups[k] if np.isfinite(x[fn])])
            cells.append('%6.3g [%6.3g..%6.3g]' % (np.median(v), np.quantile(v, .25), np.quantile(v, .75)) if len(v) else '     -')
        print('      %-10s                     %s' % (fn, '  '.join(cells)), flush=True)
    for fn, labels in (('reg', None), ('hour', None), ('wd', ['пн', 'вт', 'ср', 'чт', 'пт', 'сб', 'вс'])):
        cells = []
        for k in ('пропущенные', 'пойманные', 'обычные часы'):
            cnt = collections.Counter((x[fn] // 4 * 4 if fn == 'hour' else x[fn]) for x in groups[k])
            tot = sum(cnt.values())
            top = ', '.join('%s %.0f%%' % ((labels[kk] if labels else ('%02d-%02dч' % (kk, kk + 3) if fn == 'hour' else kk)), 100 * n / tot)
                            for kk, n in cnt.most_common(3))
            cells.append(top)
        print('      %-6s  пропущ: %s | пойман: %s | обычн: %s' % (fn, cells[0], cells[1], cells[2]), flush=True)
