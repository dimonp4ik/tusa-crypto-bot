"""The pullback rule without the bank's regime gate, attacked from scratch.

Dropping the gate changes WHICH trades the rule takes, so nothing it passed before carries over. The
full battery again: each condition alone, the inverse, the mirror, every coin, the threshold grid,
the blind-year re-pick. If this survives, the rule is not an addition to the bank - it is a second
strategy that happens to share an account.
"""
import bisect
import collections
import datetime
import sys

import numpy as np

sys.path.insert(0, 'C:/Users/Lenovo/Desktop/Торговля/crypto-bot')
import struct_params as SP
import gauntlet2 as G
import pullback_nogate as PN
from src import pullback_bank as PB

C = [('rsi48', '>=', 56.3761), ('rsi6', '<=', 33.7947)]
FLIP = {'>=': '<=', '<=': '>='}
YT = {y: int(datetime.datetime(y, 1, 1, tzinfo=datetime.UTC).timestamp()) for y in range(2022, 2028)}
BIG = 10 ** 9
M = 'без режима'


def sigs(coin, mode, conds):
    c = SP.CTX[coin]
    T1, f = G.feats(coin)
    bsm = c['sma'].setdefault(50, PB._sma(c['bdc'], 50))
    out = {}
    for i in range(PB.MIN_HOURS, len(T1)):
        close = int(T1[i]) + PB.HOUR
        j = bisect.bisect_right(c['starts'], close) - 1
        inside = j >= 0 and c['iv'][j][0] <= close < c['iv'][j][1] and c['iv'][j][2] == 'LONG'
        d = bisect.bisect_right(c['bdt'], close - 86400) - 1
        blocked = d >= 49 and c['bdc'][d] < bsm[d]
        if mode == 'гейт' and not (inside and not blocked):
            continue
        if mode == 'без режима' and blocked:
            continue
        if not np.isfinite(c['F']['atr'][i]):
            continue
        ok = True
        for nm, op, thr in conds:
            x = f[nm][i]
            if not np.isfinite(x) or (x < thr if op == '>=' else x > thr):
                ok = False
                break
        if ok:
            out[close] = (float(c['B1'][i, 3]), float(c['F']['atr'][i]))
    return out


def trades(mode, conds, lg=True, coins=None):
    rows = []
    for s in (coins or PN.ALL):
        sig = sigs(s, mode, conds)
        c = SP.CTX[s]
        slip = SP.SLIP_REAL.get(s, 0.0002)
        t15, A15, pos = c['t15'], c['a15'], c['pos']
        busy = 0
        for close in sorted(sig):
            if close < busy:
                continue
            lim, atr = sig[close]
            i = pos.get(close)
            if i is None or i + PB.HOLD_BARS > len(t15):
                continue
            o, h, l, cc = (A15[i:i + PB.HOLD_BARS, x] for x in range(4))
            if lg:
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
            js = int(np.argmax(hs)) if hs.any() else BIG
            jt = int(np.argmax(ht)) if ht.any() else BIG
            if js <= jt and js < BIG:
                jj = js
                f_ = (min(SL, o[js]) * (1 - slip)) if lg else (max(SL, o[js]) * (1 + slip))
            elif jt < BIG:
                jj, f_ = jt, (TP * (1 - slip) if lg else TP * (1 + slip))
            else:
                jj = PB.HOLD_BARS - 1
                f_ = cc[-1] * (1 - slip) if lg else cc[-1] * (1 + slip)
            ret = ((f_ / e - 1) if lg else (1 - f_ / e)) - 0.0004
            rows.append((close, int(t15[i + jj]) + 900, ret / (3.0 * atr / e), 3.0 * atr / e, s))
            busy = int(t15[i + jj]) + 900
    rows.sort()
    return rows


def st(rows, lab):
    v = np.array([r[2] for r in rows]) if rows else np.zeros(0)
    if not len(v):
        print('    %-36s сделок нет' % lab, flush=True)
        return
    print('    %-36s n%5d ВР%5.1f%% ср%+.4fR' % (lab, len(v), 100 * np.mean(v > 0), v.mean()),
          flush=True)


if __name__ == '__main__':
    base = trades(M, C)
    print('  === нулевая модель: весь поток БЕЗ режимного гейта ===', flush=True)
    st(trades(M, []), 'всё подряд LONG (это ноль)')
    print('', flush=True)
    print('  === атаки на правило без гейта ===', flush=True)
    st(base, 'ЦЕЛИКОМ')
    for c in C:
        st(trades(M, [c]), 'только %s%s%.4f' % c)
    st(trades(M, [(n, FLIP[o], t) for n, o, t in C]), 'ИНВЕРСИЯ (должна терять)')
    st(trades(M, C, lg=False), 'ЗЕРКАЛО SHORT (должно терять)')

    print('', flush=True)
    print('  по годам:', flush=True)
    for y in range(2022, 2027):
        sub = [r for r in base if YT[y] <= r[0] < YT[y + 1]]
        v = np.array([r[2] for r in sub])
        print('    %d n%3d ВР%5.1f%% ср%+.4f' % (y, len(v), 100 * np.mean(v > 0), v.mean()),
              flush=True)

    print('', flush=True)
    print('  по монетам:', flush=True)
    per = collections.defaultdict(list)
    for r in base:
        per[r[4]].append(r[2])
    for s, v in sorted(per.items(), key=lambda kv: -np.mean(kv[1])):
        print('    %-6s n%3d ВР%5.1f%% ср%+.4f' % (s.replace('USDT', ''), len(v),
                                                   100 * np.mean(np.array(v) > 0), np.mean(v)),
              flush=True)

    A = [50.0, 53.0, 56.3761, 59.0, 62.0, 65.0]
    B = [26.0, 30.0, 33.7947, 37.0, 41.0, 45.0]
    CACHE = {}
    for a in A:
        for b in B:
            CACHE[(a, b)] = trades(M, [('rsi48', '>=', a), ('rsi6', '<=', b)])

    print('', flush=True)
    print('  === сетка порогов (среднее R / сделок) ===', flush=True)
    print('    rsi48 \\ rsi6 %s' % ' '.join('%13.1f' % b for b in B), flush=True)
    for a in A:
        cells = []
        for b in B:
            v = np.array([x[2] for x in CACHE[(a, b)]]) if CACHE[(a, b)] else np.zeros(0)
            cells.append('%+.4f/%4d' % (v.mean(), len(v)) if len(v) else '       -     ')
        print('    %12.1f %s' % (a, ' '.join(cells)), flush=True)

    print('', flush=True)
    print('  === слепой переподбор порогов без гейта ===', flush=True)
    for hold in range(2022, 2027):
        lo, hi = YT[hold], YT[hold + 1]
        best, bk = None, None
        for k, rr in CACHE.items():
            tr = [x[2] for x in rr if not (lo <= x[0] < hi)]
            if len(tr) < 150:
                continue
            v = float(np.mean(tr))
            if best is None or v > best:
                best, bk = v, k
        ho = [x[2] for x in CACHE[bk] if lo <= x[0] < hi]
        cur = [x[2] for x in CACHE[(56.3761, 33.7947)] if lo <= x[0] < hi]
        f = lambda v: ('%+.4f' % np.mean(v)) if len(v) >= 12 else ' мало '
        print('    %d  выбрано rsi48>=%.1f rsi6<=%.1f | слепой год: выбор %s (n%d), рабочее %s (n%d)'
              % (hold, bk[0], bk[1], f(ho), len(ho), f(cur), len(cur)), flush=True)
