"""Writes calmgate.py: can the wide-pullback hours be anticipated from market state?

bestweeks found the five decisive weeks share a picture: BTC sitting at or above its SMA50 (+1.3..+9.0%)
and moving little over the week (-3.0..+3.0%). If that picture really precedes wide hours, the state of
the market predicts when the system switches on - useful both for expectations and, possibly, as a gate.

Every hourly close of the live-like book is labelled by the state known BEFORE it: BTC's distance from
its SMA50, BTC's 7-day move, and the median 24h range across coins (a calm/violent measure). Then:

1. Diagnosis - the share of hours with breadth >= 4 inside each bucket of each feature, against the base
   rate, so it is visible whether calm really predicts width.
2. Gate - take the breadth source only when the state is in the favourable bucket, and measure both
   measures against the accepted system; control - drop the same number of wide trades at random.
"""
import ast
import io

s = io.open('widestopref.py', encoding='utf-8').read()
head = s[:s.index("for bi, (book_name, coins) in enumerate(BOOKS3):")]

TAIL = r'''
import datetime as DT
WIDE_REF = 0.060
BOOK = [c for c in SP.COINS if c not in ('BILLUSDT', 'AAVEUSDT')]

c = SP.CTX['BTCUSDT']
bdt, bdc = np.asarray(c['bdt'], dtype=np.int64), np.asarray(c['bdc'], dtype=float)
sma50 = PB._sma(bdc, 50)


def btc_feat(ts):
    d = int(np.searchsorted(bdt, ts - 86400, side='right')) - 1
    if d < 57 or not np.isfinite(sma50[d]) or sma50[d] <= 0:
        return None, None
    return float(bdc[d] / sma50[d] - 1), float(bdc[d] / bdc[d - 7] - 1)


H = {s: coin_hours(s) for s in BOOK}
strict = collections.defaultdict(set)
for s in BOOK:
    for close, (a, b, atr, ok) in H[s].items():
        if ok and a >= 56.3761 and b <= 33.7947:
            strict[close].add(s)
hours = sorted({t for s in BOOK for t in H[s]})
wide = {t for t in hours if len(strict.get(t, set()) | strict.get(t - 3600, set())) >= 4}
print('  часов всего %d, из них с шириной>=4: %d (%.2f%%)' % (len(hours), len(wide), 100 * len(wide) / len(hours)), flush=True)


def bucket_sma(v):
    if v is None:
        return None
    return 'ниже средней' if v < 0 else ('0..+3%' if v < 0.03 else ('+3..+8%' if v < 0.08 else 'выше +8%'))


def bucket_w(v):
    if v is None:
        return None
    return 'падал >3%' if v < -0.03 else ('-3..+3% (спокойно)' if v < 0.03 else ('+3..+8%' if v < 0.08 else 'рос >8%'))


for name, fn, order in (('BTC к своей SMA50', lambda t: bucket_sma(btc_feat(t)[0]),
                         ('ниже средней', '0..+3%', '+3..+8%', 'выше +8%')),
                        ('движение BTC за 7 дней', lambda t: bucket_w(btc_feat(t)[1]),
                         ('падал >3%', '-3..+3% (спокойно)', '+3..+8%', 'рос >8%'))):
    print('', flush=True)
    print('  ДОЛЯ ЧАСОВ С ШИРИНОЙ>=4 ПО СОСТОЯНИЮ: %s' % name, flush=True)
    for b in order:
        tot = [t for t in hours if fn(t) == b]
        if not tot:
            continue
        hit = sum(1 for t in tot if t in wide)
        print('    %-20s часов %6d, из них широких %3d (%.2f%%, база %.2f%%)'
              % (b, len(tot), hit, 100 * hit / len(tot), 100 * len(wide) / len(hours)), flush=True)

base_sig = make_sig_raw(BOOK)
extra = roll_entries(BOOK, H, 'r2_4', 4, 2, 4, 0)
sig_all = sorted(base_sig + extra, key=lambda x: (x[0], x[1]))
GEO.clear(); GEO.update(BASE_GEO)
rows, keys = book_keyed(sig_all, FULL)
fin = weigh_src(rows, keys, {'r2_4'}, 1.25)
refs = [(WIDE_REF if k == 'r2_4' else sim_w.REF) for k in keys]
b0 = sim_ref(fin, 0.014, refs); m0 = money_at_dd_ref(fin, refs)
print('', flush=True)
print('  ПРИНЯТОЕ: $%.0f %.1f%% DD12 $%.0f' % (b0['eq'], 100 * abs(L.dd_of(b0['curve'])), m0), flush=True)
print('  ГЕЙТ: брать сделки источника только в подходящем состоянии', flush=True)
GATES = (('BTC не ниже средней', lambda t: (btc_feat(t)[0] or -1) >= 0),
         ('BTC у средней 0..+8%', lambda t: 0 <= (btc_feat(t)[0] if btc_feat(t)[0] is not None else -1) < 0.08),
         ('BTC за 7 дней -3..+3%', lambda t: abs(btc_feat(t)[1] if btc_feat(t)[1] is not None else 1) < 0.03))
idx_wide = [i for i, k in enumerate(keys) if k == 'r2_4']
for lbl, ok in GATES:
    drop = {i for i in idx_wide if not ok(rows[i][0])}
    keep = [i for i in range(len(rows)) if i not in drop]
    rr = [rows[i] for i in keep]; kk = [keys[i] for i in keep]
    f2 = weigh_src(rr, kk, {'r2_4'}, 1.25)
    r2 = [(WIDE_REF if k == 'r2_4' else sim_w.REF) for k in kk]
    r = sim_ref(f2, 0.014, r2); m = money_at_dd_ref(f2, r2)
    ce, cm = [], []
    for seed in range(5):
        rng = np.random.default_rng(seed)
        pick = set(rng.choice(idx_wide, size=len(drop), replace=False).tolist()) if drop else set()
        kp = [i for i in range(len(rows)) if i not in pick]
        cr = [rows[i] for i in kp]; ck = [keys[i] for i in kp]
        cf = weigh_src(cr, ck, {'r2_4'}, 1.25)
        cr2 = [(WIDE_REF if k == 'r2_4' else sim_w.REF) for k in ck]
        ce.append(sim_ref(cf, 0.014, cr2)['eq']); cm.append(money_at_dd_ref(cf, cr2))
    print('    %-24s убрано %2d из %d сделок | $%6.0f (%+5.1f%%) DD12 $%6.0f (%+5.1f%%) | КОНТРОЛЬ случайные $%.0f DD12 $%.0f'
          % (lbl, len(drop), len(idx_wide), r['eq'], 100 * (r['eq'] / b0['eq'] - 1), m, 100 * (m / m0 - 1),
             np.median(ce), np.median(cm)), flush=True)
'''

src = head + TAIL
ast.parse(src)
io.open('calmgate.py', 'w', encoding='utf-8').write(src)
print('calmgate.py готов, синтаксис ок')
