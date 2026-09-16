"""Writes bestweeks.py: what actually happened in the five weeks that carry the gain.

dropbest found the same five weeks on all three books - 2023-01-09, 2024-11-04, 2025-05-05, 2025-07-07,
2026-08-17 - and they hold roughly two thirds of the whole advantage. If they share a market story, that
story can be described in advance, and the owner will know when the system is supposed to fire and when
silence is normal.

For each of those weeks (and, for contrast, five random ordinary weeks): what BTC did over the week and
in the 7 days before it, where BTC sat against its SMA50, how many hours had breadth >= 4, how many
trades the sources opened, their win rate and mean R, and which coins traded. Printed for the live-like
book (no AAVE, XLM kept).
"""
import ast
import io

s = io.open('widestopref.py', encoding='utf-8').read()
head = s[:s.index("for bi, (book_name, coins) in enumerate(BOOKS3):")]

TAIL = r'''
import datetime as DT
WIDE_REF = 0.060
BOOK = [c for c in SP.COINS if c not in ('BILLUSDT', 'AAVEUSDT')]
BEST = ('2023-01-09', '2024-11-04', '2025-05-05', '2025-07-07', '2026-08-17')


def wk(t):
    d = DT.datetime.fromtimestamp(t, DT.UTC)
    return (d - DT.timedelta(days=d.weekday())).strftime('%Y-%m-%d')


def week_bounds(label):
    d = DT.datetime.strptime(label, '%Y-%m-%d').replace(tzinfo=DT.UTC)
    return int(d.timestamp()), int((d + DT.timedelta(days=7)).timestamp())


H = {s: coin_hours(s) for s in BOOK}
base_sig = make_sig_raw(BOOK)
extra = roll_entries(BOOK, H, 'r2_4', 4, 2, 4, 0)
sig_all = sorted(base_sig + extra, key=lambda x: (x[0], x[1]))
GEO.clear(); GEO.update(BASE_GEO)
rows, keys = book_keyed(sig_all, FULL)
cnt_hour = collections.Counter(x[0] for x in rows)
strict = collections.defaultdict(set)
for s in BOOK:
    for close, (a, b, atr, ok) in H[s].items():
        if ok and a >= 56.3761 and b <= 33.7947:
            strict[close].add(s)

c = SP.CTX['BTCUSDT']
bdt, bdc = np.asarray(c['bdt'], dtype=np.int64), np.asarray(c['bdc'], dtype=float)
sma = PB._sma(bdc, 50)


def btc_state(ts):
    d = int(np.searchsorted(bdt, ts, side='right')) - 1
    if d < 50:
        return None
    return dict(px=bdc[d], w=100 * (bdc[d] / bdc[d - 7] - 1), prev=100 * (bdc[d - 7] / bdc[d - 14] - 1),
                sma=100 * (bdc[d] / sma[d] - 1))


rng = np.random.default_rng(11)
all_weeks = sorted({wk(x[0]) for x in rows})
others = [w for w in all_weeks if w not in BEST]
sample = list(rng.choice(others, size=5, replace=False))

print('  РАЗБОР НЕДЕЛЬ: живой набор (без AAVE, с XLM)', flush=True)
for title, weeks in (('ЛУЧШИЕ НЕДЕЛИ (несут ~2/3 выигрыша)', BEST), ('ОБЫЧНЫЕ НЕДЕЛИ ДЛЯ СРАВНЕНИЯ', sample)):
    print('', flush=True)
    print('  %s' % title, flush=True)
    for w in weeks:
        lo, hi = week_bounds(w)
        st = btc_state(lo)
        wide_h = sum(1 for t in set(strict) if lo <= t < hi and len(strict.get(t, set()) | strict.get(t - 3600, set())) >= 4)
        big_h = len({x[0] for x in rows if lo <= x[0] < hi and cnt_hour[x[0]] >= 10})
        tr = [(x, k) for x, k in zip(rows, keys) if lo <= x[0] < hi]
        wide_tr = [x for x, k in tr if k == 'r2_4']
        v = np.array([x[2] for x, k in tr]) if tr else np.zeros(0)
        vw = np.array([x[2] for x in wide_tr]) if wide_tr else np.zeros(0)
        coins_w = collections.Counter(x[4].replace('USDT', '') for x in wide_tr)
        print('    %s | BTC $%s, за неделю %+.1f%%, неделей раньше %+.1f%%, к средней %+.1f%%'
              % (w, ('%.0f' % st['px']) if st else '?', st['w'] if st else 0, st['prev'] if st else 0, st['sma'] if st else 0), flush=True)
        print('      часов ширины>=4: %d | часов с 10+ сделками: %d | сделок всего %d (ВР %.0f%%, ср R %+.3f)'
              % (wide_h, big_h, len(v), 100 * np.mean(v > 0) if len(v) else 0, v.mean() if len(v) else 0), flush=True)
        print('      сделок источника %d%s' % (len(vw), (' (ВР %.0f%%, ср R %+.3f, монеты: %s)'
              % (100 * np.mean(vw > 0), vw.mean(), ', '.join('%s %d' % x for x in coins_w.most_common(6)))) if len(vw) else ''), flush=True)
'''

src = head + TAIL
ast.parse(src)
io.open('bestweeks.py', 'w', encoding='utf-8').write(src)
print('bestweeks.py готов, синтаксис ок')
