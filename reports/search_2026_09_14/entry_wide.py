"""Entry searched wide: where and when to get in, on the same signals and the same exit.

Today every signal enters at market on the next 15m open. That is one point in a large space:

  рынок          market at next open (today)
  лимит k / W    resting limit at signal close - k*ATR, alive W 15m bars; untouched = no trade.
                 Better price, fewer fills - the question is which effect wins in money.
  подтверждение  wait for the first 15m bar that closes up (for a long) within 2h, enter at the
                 next open; no such bar = no trade.

Exit is fixed at stop 3 ATR / take 1 ATR / 48h measured FROM THE FILL, so only the entry differs.
Limit entries pay maker fee in (1bp) and no entry slippage; market entries pay measured slippage.
Equal risk per trade, one position per coin, bank priority. Top variants priced in money at a 12%
drawdown and re-chosen blind on four years, read on the fifth.
"""
import collections, csv, datetime, sys
import numpy as np
sys.path.insert(0, 'C:/Users/Lenovo/Desktop/Торговля/crypto-bot')
import struct_params as SP
import gauntlet2 as G
import nogate_attack as NA
import sim_w
import live_rules_sim as L

cost = collections.defaultdict(list)
for r in csv.DictReader(open('book_frozen.csv')):
    cost[r['coin'] + 'USDT'].append(float(r['cost109']))
SLIP = {k: float(np.median(v)) for k, v in cost.items()}
SP.SLIP_REAL.update(SLIP); G.FC.clear()
COINS = [s for s in SP.COINS if s not in ('BILLUSDT', 'XLMUSDT', 'AAVEUSDT')]
PULL = [('rsi48', '>=', 56.3761), ('rsi6', '<=', 33.7947)]
SL, TP, H = 3.0, 1.0, 192
YT = {y: int(datetime.datetime(y, 1, 1, tzinfo=datetime.UTC).timestamp()) for y in range(2022, 2028)}
BIG = 10 ** 9

SIG = []
for s in COINS:
    c = SP.CTX[s]
    bank = SP.bank_signals([dict(r, tp=1.0) for r in SP.BASE], s, 50)
    pull = NA.sigs(s, 'без режима', PULL)
    for t, v in bank.items():
        SIG.append((t, s, 0, SP.BASE[v[0]].get('side', 'LONG') == 'LONG', v[1], v[2]))
    for t, v in pull.items():
        SIG.append((t, s, 1, True, v[0], v[1]))
SIG.sort(key=lambda x: (x[0], x[2]))
print('  сигналов %d' % len(SIG), flush=True)


def exit_from(c, j, e, atr, lg, ex_slip):
    t15, a15 = c['t15'], c['a15']
    if j + H > len(t15):
        return None
    o, h, l, cc = (a15[j:j + H, x] for x in range(4))
    TP_, SL_ = (e + TP * atr, e - SL * atr) if lg else (e - TP * atr, e + SL * atr)
    hs, ht = (l <= SL_, h >= TP_) if lg else (h >= SL_, l <= TP_)
    js = int(np.argmax(hs)) if hs.any() else BIG
    jt = int(np.argmax(ht)) if ht.any() else BIG
    if js <= jt and js < BIG:
        jj, f = js, (min(SL_, o[js]) * (1 - ex_slip) if lg else max(SL_, o[js]) * (1 + ex_slip))
    elif jt < BIG:
        jj, f = jt, (TP_ * (1 - ex_slip) if lg else TP_ * (1 + ex_slip))
    else:
        jj, f = H - 1, (cc[-1] * (1 - ex_slip) if lg else cc[-1] * (1 + ex_slip))
    return jj, f


def book(variant):
    kind = variant[0]
    busy, out, missed = {}, [], 0
    for close, s, prio, lg, lim, atr in SIG:
        if busy.get(s, 0) > close or not np.isfinite(atr) or atr <= 0:
            continue
        c = SP.CTX[s]
        t15, a15, pos = c['t15'], c['a15'], c['pos']
        j0 = pos.get(close)
        if j0 is None or j0 + H + 16 > len(t15):
            continue
        slip = SLIP.get(s, 0.0003)
        if kind == 'рынок':
            j, e, fee = j0, (a15[j0, 0] * (1 + slip) if lg else a15[j0, 0] * (1 - slip)), 0.0004
        elif kind == 'лимит':
            k, W = variant[1], variant[2]
            px = lim - k * atr if lg else lim + k * atr
            j = None
            for w in range(W):
                o_, h_, l_ = a15[j0 + w, 0], a15[j0 + w, 1], a15[j0 + w, 2]
                if (lg and l_ <= px) or (not lg and h_ >= px):
                    j = j0 + w
                    e = min(o_, px) if lg else max(o_, px)
                    break
            if j is None:
                missed += 1
                continue
            fee = 0.0003
        else:  # подтверждение
            j = None
            for w in range(8):
                o_, c_ = a15[j0 + w, 0], a15[j0 + w, 3]
                if (lg and c_ > o_) or (not lg and c_ < o_):
                    j = j0 + w + 1
                    break
            if j is None:
                missed += 1
                continue
            e, fee = (a15[j, 0] * (1 + slip) if lg else a15[j, 0] * (1 - slip)), 0.0004
        sf = SL * atr / e
        if sf > 0.10:
            continue
        r = exit_from(c, j, e, atr, lg, slip)
        if r is None:
            continue
        jj, f = r
        ret = ((f / e - 1) if lg else (1 - f / e)) - fee
        end = int(t15[j + jj]) + 900
        out.append((close, end, ret / sf, sf, s))
        busy[s] = end
    return out, missed


def stats(tr):
    if len(tr) < 100:
        return None
    v = np.array([x[2] for x in tr])
    cum = np.cumsum(v[np.argsort([x[1] for x in tr])])
    dd = float(np.max(np.maximum.accumulate(cum) - cum))
    months = (tr[-1][1] - tr[0][0]) / (365.25 * 86400 / 12)
    return dict(n=len(v), wr=float(np.mean(v > 0)), avg=float(v.mean()), rm=v.sum() / months, dd=dd,
                ratio=(v.sum() / months) / max(dd, 1e-9), tpy=len(v) / (months / 12))


def money(tr):
    return sim_w.money_at_dd([(a, b, R, sf, s, L.REF / sf) for a, b, R, sf, s in tr], 0.12)


VARS = [('рынок',)]
for k in (0.0, 0.1, 0.25, 0.5, 1.0):
    for W in (4, 16):
        VARS.append(('лимит', k, W))
VARS.append(('подтверждение',))
res = {}
print('  вход                    | сделок/год  ВР     ср R    R/мес  просадка R  R/мес/DD  не залито | деньги DD 12%%', flush=True)
for v in VARS:
    tr, missed = book(v)
    st = stats(tr)
    if not st:
        continue
    res[v] = (st, tr)
    rk, m = money(tr)
    name = v[0] if len(v) == 1 else '%s %.2fATR %dч' % (v[0], v[1], v[2] // 4)
    print('  %-23s | %6.0f    %4.1f%%  %+.3f  %+5.2f   %6.1f     %.3f     %5d   | $%7.0f (%.2f%%)'
          % (name, st['tpy'], 100 * st['wr'], st['avg'], st['rm'], st['dd'], st['ratio'], missed, m, 100 * rk), flush=True)

print('  -- слепой выбор входа: 4 года выбор, 5-й замер --', flush=True)
wins = seen = 0
for y in range(2022, 2027):
    lo, hi = YT[y], YT[y + 1]
    best, bk = -1e9, None
    for v, (st, tr) in res.items():
        s2 = stats([x for x in tr if not (lo <= x[0] < hi)])
        if s2 and s2['ratio'] > best:
            best, bk = s2['ratio'], v
    pick = [x for x in res[bk][1] if lo <= x[0] < hi]
    base = [x for x in res[('рынок',)][1] if lo <= x[0] < hi]
    if len(pick) < 30 or len(base) < 30:
        continue
    mp, mb = money(pick)[1], money(base)[1]
    seen += 1; wins += int(mp > mb)
    print('    %d выбран %s | слепой год $%.0f против рынка $%.0f  %s' % (y, bk, mp, mb, 'лучше' if mp > mb else 'хуже'), flush=True)
print('    выбранный вслепую вход лучше рыночного в %d из %d лет' % (wins, seen), flush=True)
