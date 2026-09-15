"""Pair mean reversion searched wide: every pair of the thirteen coins, blind-year selection.

The project's earlier pair study tested three fixed pairs (BTC/ETH, SOL/AVAX, ADA/DOT) with one
window and found them negative or unstable. That is a sample of three from seventy-eight. Here every
pair is tried with a 72h or 168h window and an entry at 2, 2.5 or 3 standard deviations of the log
price ratio. Mean and deviation are frozen at entry. Exit on return to 0.5 sd, on divergence to 5 sd,
or after 48 hours, both legs at the next hourly open. Equal notional per leg, fixed quantities.
Costs: 4bp round-trip fee plus measured slippage twice on each leg.

Returns are per unit of gross pair notional. The question is only whether any edge survives choosing
pairs and parameters on four years and reading the fifth - money comes after, if it does.
"""
import collections, csv, datetime, itertools, sys
import numpy as np
sys.path.insert(0, 'C:/Users/Lenovo/Desktop/Торговля/crypto-bot')
import struct_params as SP

cost = collections.defaultdict(list)
for r in csv.DictReader(open('book_frozen.csv')):
    cost[r['coin'] + 'USDT'].append(float(r['cost109']))
SLIP = {k: float(np.median(v)) for k, v in cost.items()}
COINS = [s for s in SP.COINS if s not in ('BILLUSDT', 'XLMUSDT', 'AAVEUSDT')]
YEARS = [2022, 2023, 2024, 2025, 2026]
YT = {y: int(datetime.datetime(y, 1, 1, tzinfo=datetime.UTC).timestamp()) for y in YEARS + [2027]}
HOLD = 48


def rolling(x, w):
    cs = np.concatenate([[0.0], np.cumsum(x)])
    cs2 = np.concatenate([[0.0], np.cumsum(x * x)])
    m = np.full(len(x), np.nan)
    sd = np.full(len(x), np.nan)
    s1 = cs[w:] - cs[:-w]
    s2 = cs2[w:] - cs2[:-w]
    mu = s1 / w
    var = np.maximum(s2 / w - mu * mu, 0.0)
    m[w - 1:] = mu
    sd[w - 1:] = np.sqrt(var)
    return m, sd


def pair_trades(a, b, W, th):
    ca, cb = SP.CTX[a], SP.CTX[b]
    ta, tb = ca['T1'].astype(np.int64), cb['T1'].astype(np.int64)
    common, ia, ib = np.intersect1d(ta, tb, return_indices=True)
    if len(common) < W + HOLD + 10:
        return []
    Ba, Bb = ca['B1'][ia], cb['B1'][ib]
    lr = np.log(Ba[:, 3] / Bb[:, 3])
    mu, sd = rolling(lr, W)
    z = (lr - mu) / np.where(sd > 0, sd, np.nan)
    contiguous = np.diff(common) == 3600
    cost = 0.0004 + SLIP.get(a, 0.0003) + SLIP.get(b, 0.0003)
    out, i, n = [], W, len(common)
    cand = np.flatnonzero((np.abs(z) >= th) & (np.abs(z) < 5))
    ci = 0
    while ci < len(cand):
        i = cand[ci]
        if i + HOLD + 2 >= n or not contiguous[i:i + HOLD + 1].all():
            ci += 1
            continue
        side = -1.0 if z[i] > 0 else 1.0
        m0, s0 = mu[i], sd[i]
        zz = (lr[i + 1:i + 1 + HOLD] - m0) / s0
        conv = side * zz >= -0.5
        div = side * zz <= -5
        k_conv = int(np.argmax(conv)) if conv.any() else HOLD
        k_div = int(np.argmax(div)) if div.any() else HOLD
        k = min(k_conv, k_div, HOLD - 1)
        e = i + 1                       # enter at next hour's open
        x = i + 2 + k                   # exit at the open after the triggering close
        if x >= n:
            break
        a0, b0 = Ba[e, 0], Bb[e, 0]
        a1, b1 = Ba[x, 0], Bb[x, 0]
        net = 0.5 * side * (a1 / a0 - b1 / b0) - cost
        out.append((int(common[e]), int(common[x]), net))
        while ci < len(cand) and cand[ci] <= x:
            ci += 1
    return out


GRID = [(W, th) for W in (72, 168) for th in (2.0, 2.5, 3.0)]
RES = {}
for a, b in itertools.combinations(COINS, 2):
    for W, th in GRID:
        tr = pair_trades(a, b, W, th)
        if len(tr) >= 20:
            RES[(a, b, W, th)] = tr
print('  вариантов (пара x окно x порог) с 20+ сделками: %d' % len(RES), flush=True)

allv = np.array([t[2] for tr in RES.values() for t in tr])
print('  все варианты вместе: сделок %d, ВР %.1f%%, средний результат %+.4f%% номинала'
      % (len(allv), 100 * np.mean(allv > 0), 100 * allv.mean()), flush=True)

# --- blind: choose variants on four years, read the fifth
print('', flush=True)
print('  === слепой отбор: вариант берётся, если на 4 годах средний > 0, 20+ сделок и плюс в 3+ годах ===', flush=True)
tot_sel = []
for hold_y in YEARS:
    lo, hi = YT[hold_y], YT[hold_y + 1]
    chosen = []
    for key, tr in RES.items():
        trn = [t for t in tr if not (lo <= t[0] < hi)]
        if len(trn) < 20:
            continue
        v = np.array([t[2] for t in trn])
        yrs = []
        for y in YEARS:
            if y == hold_y:
                continue
            vy = [t[2] for t in trn if YT[y] <= t[0] < YT[y + 1]]
            if len(vy) >= 3:
                yrs.append(np.mean(vy))
        if v.mean() > 0 and sum(1 for m in yrs if m > 0) >= 3:
            chosen.append(key)
    blind = [t[2] for key in chosen for t in RES[key] if lo <= t[0] < hi]
    rest = [t[2] for key in RES if key not in chosen for t in RES[key] if lo <= t[0] < hi]
    tot_sel += blind
    if blind:
        print('    %d выбрано вариантов %3d | слепой год: сделок %4d ВР %4.1f%% ср %+.4f%% | невыбранные ср %+.4f%%'
              % (hold_y, len(chosen), len(blind), 100 * np.mean(np.array(blind) > 0), 100 * np.mean(blind),
                 100 * np.mean(rest) if rest else float('nan')), flush=True)
if tot_sel:
    v = np.array(tot_sel)
    print('    ИТОГО слепые годы: сделок %d, ВР %.1f%%, ср %+.4f%% номинала, сумма %+.2f номиналов'
          % (len(v), 100 * np.mean(v > 0), 100 * v.mean(), v.sum()), flush=True)

print('', flush=True)
print('  === лучшие варианты по всей истории (для справки, НЕ доказательство) ===', flush=True)
rank = sorted(RES, key=lambda k: -np.mean([t[2] for t in RES[k]]))[:12]
for k in rank:
    v = np.array([t[2] for t in RES[k]])
    yrs = []
    for y in YEARS:
        vy = [t[2] for t in RES[k] if YT[y] <= t[0] < YT[y + 1]]
        yrs.append('%+.2f' % (100 * np.mean(vy)) if len(vy) >= 3 else '  -  ')
    print('    %-5s/%-5s окно %3d порог %.1f | сделок %3d ВР %4.1f%% ср %+.3f%% | по годам %% %s'
          % (k[0].replace('USDT', ''), k[1].replace('USDT', ''), k[2], k[3], len(v), 100 * np.mean(v > 0),
             100 * v.mean(), ' '.join(yrs)), flush=True)
