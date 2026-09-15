"""Stop, take and hold searched wide, for the bank's signals and the pullback rule's.

The geometry was inherited from the bank (stop 3 ATR, take 1 ATR, 48h) and only the take was ever
varied. Here the signals stay fixed and every exit is re-simulated: stop 0.75-6 ATR, take 0.5-5 ATR,
hold 12-96h - 168 geometries per book.

Comparison is at EQUAL RISK per trade (each trade sized so its stop costs the same share of
equity). Under the live sizing a tighter stop simply carries more leverage and looks better for
that reason alone; equal risk removes that. Trades whose stop is beyond 10% are dropped - at 10x
leverage the exchange liquidates them before the stop. Entry is market at the next 15m open,
measured costs on both legs. One position per coin, bank signals have priority.

Selection metric, fast: R per month divided by the worst drawdown in R (both under equal risk).
The top geometries are then priced in money, and the winner is re-chosen blind on four years and
read on the fifth.
"""
import collections, csv, datetime, itertools, sys
import numpy as np
sys.path.insert(0, 'C:/Users/Lenovo/Desktop/Торговля/crypto-bot')
import struct_params as SP
import gauntlet2 as G
import nogate_attack as NA
import sim_w
import live_rules_sim as L
from src import pullback_bank as PB

cost = collections.defaultdict(list)
for r in csv.DictReader(open('book_frozen.csv')):
    cost[r['coin'] + 'USDT'].append(float(r['cost109']))
SLIP = {k: float(np.median(v)) for k, v in cost.items()}
SP.SLIP_REAL.update(SLIP); G.FC.clear()
FEE = 0.0004
COINS = [s for s in SP.COINS if s not in ('BILLUSDT', 'XLMUSDT', 'AAVEUSDT')]
PULL = [('rsi48', '>=', 56.3761), ('rsi6', '<=', 33.7947)]
MAXH = 96 * 4
SLS = (0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
TPS = (0.5, 1.0, 1.5, 2.0, 3.0, 5.0)
HOLDS = (12, 24, 48, 96)
YT = {y: int(datetime.datetime(y, 1, 1, tzinfo=datetime.UTC).timestamp()) for y in range(2022, 2028)}

# ---- signal paths, built once: normalised excursions so any geometry is a searchsorted away
SIG = []   # (close, coin, prio, long, e, atr, Mu, Md, o_norm, c_norm, t15_idx)
for s in COINS:
    c = SP.CTX[s]
    t15, a15, pos = c['t15'], c['a15'], c['pos']
    slip = SLIP.get(s, 0.0003)
    bank = SP.bank_signals([dict(r, tp=1.0) for r in SP.BASE], s, 50)
    pull = NA.sigs(s, 'без режима', PULL)
    items = [(t, 0, SP.BASE[v[0]].get('side', 'LONG') == 'LONG', v[2]) for t, v in bank.items()]
    items += [(t, 1, True, v[1]) for t, v in pull.items()]
    for close, prio, lg, atr in items:
        j = pos.get(close)
        if j is None or j + MAXH > len(t15) or not np.isfinite(atr) or atr <= 0:
            continue
        o, h, l, cc = (a15[j:j + MAXH, x] for x in range(4))
        e = o[0] * (1 + slip) if lg else o[0] * (1 - slip)
        fav = (h - e) / atr if lg else (e - l) / atr
        adv = (e - l) / atr if lg else (h - e) / atr
        SIG.append((close, s, prio, lg, e, atr, np.maximum.accumulate(fav), np.maximum.accumulate(adv),
                    (o - e) / atr if lg else (e - o) / atr, (cc - e) / atr if lg else (e - cc) / atr,
                    t15[j:j + MAXH], slip))
SIG.sort(key=lambda x: (x[0], x[2]))
print('  сигналов: %d (банк %d, откат %d)' % (len(SIG), sum(1 for x in SIG if x[2] == 0),
                                           sum(1 for x in SIG if x[2] == 1)), flush=True)


def book(sl, tp, hold_h, which):
    H = hold_h * 4
    busy, out, liq = {}, [], 0
    for close, s, prio, lg, e, atr, Mu, Md, on, cn, tt, slip in SIG:
        if which == 'bank' and prio != 0 or which == 'pull' and prio != 1:
            continue
        if busy.get(s, 0) > close:
            continue
        sf = sl * atr / e
        if sf > 0.10:
            liq += 1
            continue
        js = int(np.searchsorted(Md[:H], sl, side='left'))
        jt = int(np.searchsorted(Mu[:H], tp, side='left'))
        if js < H and js <= jt:
            x = min(-sl, on[js])                      # gap through the stop fills at the open
            jj = js
            xr = x * atr / e - slip - FEE
        elif jt < H:
            jj = jt
            xr = tp * atr / e - slip - FEE
        else:
            jj = H - 1
            xr = cn[H - 1] * atr / e - slip - FEE
        end = int(tt[jj]) + 900
        out.append((close, end, xr / sf, sf, s))
        busy[s] = end
    return out, liq


def stats(tr):
    if len(tr) < 100:
        return None
    v = np.array([x[2] for x in tr])
    order = np.argsort([x[1] for x in tr])
    cum = np.cumsum(v[order])
    dd = float(np.max(np.maximum.accumulate(cum) - cum))
    months = (tr[-1][1] - tr[0][0]) / (365.25 * 86400 / 12)
    rm = v.sum() / months
    return dict(n=len(v), wr=float(np.mean(v > 0)), avg=float(v.mean()), rm=rm, dd=dd,
                ratio=rm / max(dd, 1e-9), tpy=len(v) / (months / 12))


REF = L.REF
def money(tr, want=0.12):
    rows = [(a, b, R, sf, s, REF / sf) for a, b, R, sf, s in tr]
    return sim_w.money_at_dd(rows, want)


for which, lbl in (('both', 'БАНК + ОТКАТ'), ('pull', 'ТОЛЬКО ОТКАТ'), ('bank', 'ТОЛЬКО БАНК')):
    print('', flush=True)
    print('  ======== %s ========' % lbl, flush=True)
    res = {}
    for sl, tp, hh in itertools.product(SLS, TPS, HOLDS):
        tr, liq = book(sl, tp, hh, which)
        st = stats(tr)
        if st:
            res[(sl, tp, hh)] = (st, tr, liq)
    cur = res.get((3.0, 1.0, 48))
    top = sorted(res, key=lambda k: -res[k][0]['ratio'])[:8]
    if cur and (3.0, 1.0, 48) not in top:
        top.append((3.0, 1.0, 48))
    print('  стоп тейк удерж | сделок/год  ВР     ср R    R/мес  просадка R  R/мес/DD | деньги при DD 12%%  (риск)  ликвид.', flush=True)
    for k in top:
        st, tr, liq = res[k]
        rk, m = money(tr)
        print('  %4.2f %4.1f %3dч  | %6.0f    %4.1f%%  %+.3f  %+5.2f   %6.1f      %.3f   | $%8.0f  (%.2f%%)  %d%s'
              % (k[0], k[1], k[2], st['tpy'], 100 * st['wr'], st['avg'], st['rm'], st['dd'], st['ratio'],
                 m, 100 * rk, liq, '   <- сейчас' if k == (3.0, 1.0, 48) else ''), flush=True)
    # blind: choose geometry on four years by ratio, read the fifth in money
    print('  -- слепой выбор геометрии: 4 года выбор, 5-й замер --', flush=True)
    wins = 0; seen = 0
    for y in range(2022, 2027):
        lo, hi = YT[y], YT[y + 1]
        best, bk = -1e9, None
        for k, (st, tr, liq) in res.items():
            trn = [x for x in tr if not (lo <= x[0] < hi)]
            s2 = stats(trn)
            if s2 and s2['ratio'] > best:
                best, bk = s2['ratio'], k
        pick = [x for x in res[bk][1] if lo <= x[0] < hi]
        base = [x for x in res[(3.0, 1.0, 48)][1] if lo <= x[0] < hi] if (3.0, 1.0, 48) in res else []
        if len(pick) < 30 or len(base) < 30:
            continue
        mp, mb = money(pick)[1], money(base)[1]
        seen += 1; wins += int(mp > mb)
        print('    %d выбрано стоп %.2f тейк %.1f %dч | слепой год $%.0f против нынешней $%.0f  %s'
              % (y, bk[0], bk[1], bk[2], mp, mb, 'лучше' if mp > mb else 'хуже'), flush=True)
    print('    выбранная вслепую лучше нынешней в %d из %d лет' % (wins, seen), flush=True)
