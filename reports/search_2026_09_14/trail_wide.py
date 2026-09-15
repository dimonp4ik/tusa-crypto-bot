"""Exit searched wide: breakeven and trailing stops on the bank's and the pullback rule's signals.

Every exit so far was a fixed stop and a fixed take. A trade that runs +0.9 ATR and comes back to the
stop gives back everything, and a trade that would have run 4 ATR is cut at 1. Both are exit-shape
losses that no geometry grid can see.

  безубыток a   once the trade has gone +a ATR, the stop moves to entry
  трейл a d     once +a ATR, the stop follows the best price at distance d ATR (never loosens)
  тейк          fixed take, or none - the trail alone decides

The trail uses the best price up to the PREVIOUS 15m bar, so a bar's own high can never lift a stop
that the same bar's low then hits. Same bar stop-and-take: stop first. Gaps through the stop fill at
the open. Entry market at next open, measured costs, one position per coin, bank priority, stop 3 ATR,
hold 48h. Equal risk per trade for the money comparison; live sizing shown too. Blind choice on four
years, read on the fifth.
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
FEE = 0.0004
COINS = [s for s in SP.COINS if s not in ('BILLUSDT', 'XLMUSDT', 'AAVEUSDT')]
PULL = [('rsi48', '>=', 56.3761), ('rsi6', '<=', 33.7947)]
SL, H = 3.0, 192
YT = {y: int(datetime.datetime(y, 1, 1, tzinfo=datetime.UTC).timestamp()) for y in range(2022, 2028)}

SIG = []
for s in COINS:
    c = SP.CTX[s]
    t15, a15, pos = c['t15'], c['a15'], c['pos']
    slip = SLIP.get(s, 0.0003)
    bank = SP.bank_signals([dict(r, tp=1.0) for r in SP.BASE], s, 50)
    items = [(t, 0, SP.BASE[v[0]].get('side', 'LONG') == 'LONG', v[2]) for t, v in bank.items()]
    items += [(t, 1, True, v[1]) for t, v in NA.sigs(s, 'без режима', PULL).items()]
    for close, prio, lg, atr in items:
        j = pos.get(close)
        if j is None or j + H > len(t15) or not np.isfinite(atr) or atr <= 0:
            continue
        o, h, l, cc = (a15[j:j + H, x] for x in range(4))
        e = o[0] * (1 + slip) if lg else o[0] * (1 - slip)
        if SL * atr / e > 0.10:
            continue
        fav = (h - e) / atr if lg else (e - l) / atr
        adv = (e - l) / atr if lg else (h - e) / atr
        on = (o - e) / atr if lg else (e - o) / atr
        cn = (cc - e) / atr if lg else (e - cc) / atr
        mlag = np.concatenate([[-np.inf], np.maximum.accumulate(fav)[:-1]])
        SIG.append((close, s, prio, e, atr, fav, adv, on, cn, mlag, t15[j:j + H], slip))
SIG.sort(key=lambda x: (x[0], x[2]))
print('  сигналов %d' % len(SIG), flush=True)


def book(var):
    kind, a, d, tp = var
    busy, out = {}, []
    for close, s, prio, e, atr, fav, adv, on, cn, mlag, tt, slip in SIG:
        if busy.get(s, 0) > close:
            continue
        if kind == 'база':
            thr = np.full(H, SL)
        elif kind == 'безубыток':
            thr = np.where(mlag >= a, 0.0, SL)
        else:
            thr = np.where(mlag >= a, np.minimum(SL, d - mlag), SL)
        hs = adv >= thr
        ht = fav >= tp if tp is not None else np.zeros(H, dtype=bool)
        js = int(np.argmax(hs)) if hs.any() else H
        jt = int(np.argmax(ht)) if ht.any() else H
        if js < H and js <= jt:
            jj, x = js, min(-thr[js], on[js])
        elif jt < H:
            jj, x = jt, tp
        else:
            jj, x = H - 1, cn[H - 1]
        sf = SL * atr / e
        xr = x * atr / e - slip - FEE
        end = int(tt[jj]) + 900
        out.append((close, end, xr / sf, sf, s))
        busy[s] = end
    return out


def stats(tr):
    v = np.array([x[2] for x in tr])
    cum = np.cumsum(v[np.argsort([x[1] for x in tr])])
    dd = float(np.max(np.maximum.accumulate(cum) - cum))
    months = (tr[-1][1] - tr[0][0]) / (365.25 * 86400 / 12)
    return dict(n=len(v), wr=float(np.mean(v > 0)), avg=float(v.mean()), rm=v.sum() / months, dd=dd,
                ratio=(v.sum() / months) / max(dd, 1e-9), tpy=len(v) / (months / 12))


def m_eq(tr):
    return sim_w.money_at_dd([(a, b, R, sf, s, L.REF / sf) for a, b, R, sf, s in tr], 0.12)


def m_live(tr):
    r = sim_w.simulate([(a, b, R, sf, s, 1.0) for a, b, R, sf, s in tr], 0.014)
    return r['eq'], abs(L.dd_of(r['curve']))


VARS = [('база', 0, 0, 1.0)]
for a in (0.5, 1.0):
    for tp in (1.0, 2.0):
        VARS.append(('безубыток', a, 0, tp))
for a in (0.5, 1.0, 1.5):
    for d in (1.0, 1.5, 2.0):
        for tp in (None, 2.0, 3.0):
            VARS.append(('трейл', a, d, tp))
RES = {}
print('  выход                          | сделок/год  ВР     ср R   R/мес  просадка R | $ DD12%% одинак.риск | $ 1.4%% живой / просадка', flush=True)
for v in VARS:
    tr = book(v)
    st = stats(tr)
    k, m = m_eq(tr)
    mf, dd = m_live(tr)
    RES[v] = tr
    nm = 'база (стоп 3, тейк 1)' if v[0] == 'база' else (
        'безубыток после %.1f, тейк %.0f' % (v[1], v[3]) if v[0] == 'безубыток' else
        'трейл после %.1f дист %.1f %s' % (v[1], v[2], 'без тейка' if v[3] is None else 'тейк %.0f' % v[3]))
    print('  %-30s | %6.0f    %4.1f%%  %+.3f  %+5.2f   %6.1f    | $%6.0f (%.2f%%)     | $%6.0f / %4.1f%%'
          % (nm, st['tpy'], 100 * st['wr'], st['avg'], st['rm'], st['dd'], m, 100 * k, mf, 100 * dd), flush=True)

print('  -- слепой выбор выхода по деньгам (одинаковый риск): 4 года выбор, 5-й замер --', flush=True)
wins = seen = 0
for y in range(2022, 2027):
    lo, hi = YT[y], YT[y + 1]
    best, bk = -1, None
    for v, tr in RES.items():
        m = m_eq([x for x in tr if not (lo <= x[0] < hi)])[1]
        if m > best:
            best, bk = m, v
    mp = m_eq([x for x in RES[bk] if lo <= x[0] < hi])[1]
    mb = m_eq([x for x in RES[('база', 0, 0, 1.0)] if lo <= x[0] < hi])[1]
    seen += 1; wins += int(mp > mb)
    print('    %d выбран %s | слепой год $%.0f против базы $%.0f  %s'
          % (y, bk, mp, mb, 'лучше' if mp > mb else ('так же' if mp == mb else 'хуже')), flush=True)
print('    лучше базы в %d из %d лет' % (wins, seen), flush=True)
