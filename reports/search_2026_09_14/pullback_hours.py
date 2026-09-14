"""Hours and coins for the pullback rule, with the confounding this test had the first time.

The earlier version of an hour test in this project stripped the hour condition and compared - which
silently changed which signals survive one-position-per-coin, and made a healthy rule look broken.
So it is done in two separate steps that answer two different questions:

  1. Attribution. The 346 trades are taken exactly as they are, and then sorted by the hour they
     opened. The trade set never changes, so nothing is confounded; this says where the money in the
     rule already comes from.
  2. The filter. The hour condition is put into the signal generation, which genuinely changes the
     trade set, and the account is measured in money at equal drawdown. This says what the filter
     would do.

Then the coins. AAVE loses money for the bank and for the pullback rule alike. The recorded rule is
that a coin may only be cut on weakness in a HOSTILE window, not on its average, so both are checked.
"""
import collections
import datetime
import sys

import numpy as np

sys.path.insert(0, 'C:/Users/Lenovo/Desktop/Торговля/crypto-bot')
import gauntlet2 as G
import stop_width as SW
import sim_w

C = [('rsi48', '>=', 56.3761), ('rsi6', '<=', 33.7947)]
rows, hours = G.simulate(C, True)
bank = SW.run(3.0, 1.0, coins=G.ALL, with_meta=True)
YT = {y: int(datetime.datetime(y, 1, 1, tzinfo=datetime.UTC).timestamp()) for y in range(2022, 2028)}


def hr(t):
    return datetime.datetime.fromtimestamp(int(t), datetime.UTC).hour


print('  === 1. КУДА уже попадают 346 сделок отката (набор сделок не меняется) ===', flush=True)
blocks = [('вечер-ночь 17-05', lambda h: h >= 17 or h <= 4),
          ('утро 05-11', lambda h: 5 <= h <= 11),
          ('день 12-16', lambda h: 12 <= h <= 16)]
for lbl, f in blocks:
    sub = [r for r in rows if f(hr(r[0]))]
    v = np.array([r[2] for r in sub])
    print('    %-18s n%4d (%2.0f%%) ВР%5.1f%% ср%+.4fR' % (lbl, len(v), 100 * len(v) / len(rows),
                                                           100 * np.mean(v > 0), v.mean()),
          flush=True)
print('    по часам:', flush=True)
for h in range(24):
    sub = [r for r in rows if hr(r[0]) == h]
    if len(sub) < 5:
        continue
    v = np.array([r[2] for r in sub])
    print('      %02d:00 n%3d ВР%5.1f%% ср%+.4fR' % (h, len(v), 100 * np.mean(v > 0), v.mean()),
          flush=True)

print('', flush=True)
print('  === 2. часовой ФИЛЬТР: набор сделок меняется, мерим деньгами ===', flush=True)


def merge(cand):
    ev = [(a, b, R, sf, s, 'bank') for a, b, R, sf, s in bank]
    ev += [(a, b, R, sf, s, 'cand') for a, b, R, sf, s in cand]
    ev.sort(key=lambda x: (x[0], 0 if x[5] == 'bank' else 1))
    busy, out = {}, []
    for a, b, R, sf, s, tag in ev:
        if busy.get(s, 0) > a:
            continue
        busy[s] = b
        out.append((a, b, R, sf, s, 1.0))
    out.sort()
    return out


def money(cand, lbl):
    m_ = merge(cand)
    k, mm = sim_w.money_at_dd(m_, 0.12)
    per = []
    for y in range(2022, 2027):
        sub = [r for r in m_ if YT[y] <= r[0] < YT[y + 1]]
        if len(sub) >= 50:
            per.append('%d $%3.0f' % (y, sim_w.money_at_dd(sub, 0.12)[1]))
    print('    %-28s сделок связки %4d, своих %4d, риск %.3f%%, $%6.0f | %s'
          % (lbl, len(m_), len(cand), 100 * k, mm, ' '.join(per)), flush=True)
    return mm


money([], 'банк один')
money(rows, 'банк+откат как есть')
for lbl, lo, hi in (('только 17-05', 17, 4), ('только 12-16 убрать', None, None)):
    pass
for lbl, keep in (('откат только 17-05', lambda h: h >= 17 or h <= 4),
                  ('откат без дня 12-16', lambda h: not (12 <= h <= 16)),
                  ('откат только 00-11', lambda h: h <= 11)):
    cand, _ = G.simulate(C, True, keep_hours=keep) if False else (None, None)
    # the hour gate must act at signal time, so it is applied inside signal generation
    sub = []
    for s in G.ALL:
        sig = {t: v for t, v in G.signals(s, C, True).items() if keep(hr(t))}
        sub.append((s, sig))
    cand = []
    import bisect
    from src import pullback_bank as PB
    for s, sig in sub:
        c = G.SP.CTX[s]
        slip = G.SP.SLIP_REAL.get(s, 0.0002)
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
            if o[0] <= lim:
                e, touch = o[0] * (1 + slip), False
            elif l[0] <= lim:
                e, touch = lim * (1 + slip), True
            else:
                continue
            TP, SL = e + 1.0 * atr, e - 3.0 * atr
            hs, ht = l <= SL, h >= TP
            if touch:
                ht = ht.copy()
                ht[0] = False
            js = int(np.argmax(hs)) if hs.any() else 10 ** 9
            jt = int(np.argmax(ht)) if ht.any() else 10 ** 9
            if js <= jt and js < 10 ** 9:
                jj, f_ = js, min(SL, o[js]) * (1 - slip)
            elif jt < 10 ** 9:
                jj, f_ = jt, TP * (1 - slip)
            else:
                jj, f_ = PB.HOLD_BARS - 1, cc[-1] * (1 - slip)
            ret = (f_ / e - 1) - 0.0004
            cand.append((close, int(t15[i + jj]) + 900, ret / (3.0 * atr / e), 3.0 * atr / e, s))
            busy = int(t15[i + jj]) + 900
    cand.sort()
    money(cand, lbl)

print('', flush=True)
print('  === 3. AAVE: убыточна и у банка, и у отката ===', flush=True)
for lbl, src in (('банк', [(a, b, R, sf, s) for a, b, R, sf, s in bank]),
                 ('откат', [(r[0], r[1], r[2], r[3], r[4]) for r in rows])):
    for y in range(2022, 2027):
        sub = [x[2] for x in src if x[4] == 'AAVEUSDT' and YT[y] <= x[0] < YT[y + 1]]
        if len(sub) >= 8:
            print('    %-6s AAVE %d: n%3d ср%+.4f ВР%3.0f%%'
                  % (lbl, y, len(sub), np.mean(sub), 100 * np.mean(np.array(sub) > 0)), flush=True)
coins_no = [s for s in G.ALL if s != 'AAVEUSDT']
bank2 = SW.run(3.0, 1.0, coins=coins_no, with_meta=True)
rows2, _ = G.simulate(C, True, coins=coins_no)
m1 = sim_w.money_at_dd([(a, b, R, sf, s, 1.0) for a, b, R, sf, s in bank], 0.12)
bank_saved = bank
bank = bank2
m2 = sim_w.money_at_dd([(a, b, R, sf, s, 1.0) for a, b, R, sf, s in bank2], 0.12)
print('    банк со всеми монетами $%.0f | без AAVE $%.0f' % (m1[1], m2[1]), flush=True)
money(rows2, 'банк+откат без AAVE')
bank = bank_saved
