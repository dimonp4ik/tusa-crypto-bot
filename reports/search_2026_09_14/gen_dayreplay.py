"""Writes dayreplay.py: the sheet to check the live bot against on day one.

Everything else can be verified after the fact; this cannot. When the owner says "да" and the patch
goes in, the first live hours have to be compared with what the model expects - signal by signal, not
by feel. So the last 45 days of history are replayed on the live-like book (no AAVE, XLM kept) exactly
as the proposal would run them, and printed hour by hour:

    which coins gave the strict pullback that hour, the breadth (this hour plus the previous one),
    whether the wide source opened anything and on which coins, how many trades opened in the hour,
    the size multiplier each one gets (x1.25 at 10+ trades, x1.25 for a wide trade, both = 1.5625),
    the stop fraction and whether stop_ref shrinks it (0.0394, or 0.060 for a wide trade).

Totals at the end: trades, wide trades, hours with breadth >= 4, hours with 10+ trades, and how many
trades each multiplier touched.
"""
import ast
import io

s = io.open('widestopref.py', encoding='utf-8').read()
head = s[:s.index("for bi, (book_name, coins) in enumerate(BOOKS3):")]

TAIL = r'''
import datetime as DT
WIDE_REF = 0.060
DAYS = 45
BOOK = [c for c in SP.COINS if c not in ('BILLUSDT', 'AAVEUSDT')]      # живой набор: без AAVE, с XLM


def ts(t):
    return DT.datetime.fromtimestamp(t, DT.UTC).strftime('%Y-%m-%d %H:%M')


H = {s: coin_hours(s) for s in BOOK}
base_sig = make_sig_raw(BOOK)
extra = roll_entries(BOOK, H, 'r2_4', 4, 2, 4, 0)
sig_all = sorted(base_sig + extra, key=lambda x: (x[0], x[1]))
GEO.clear(); GEO.update(BASE_GEO)
rows, keys = book_keyed(sig_all, FULL)
fin = weigh_src(rows, keys, {'r2_4'}, 1.25)

last = max(x[0] for x in rows)
lo = last - DAYS * 86400
strict = collections.defaultdict(list)
for s in BOOK:
    for close, (a, b, atr, ok) in H[s].items():
        if ok and a >= 56.3761 and b <= 33.7947:
            strict[close].append(s.replace('USDT', ''))
cnt_hour = collections.Counter(x[0] for x in rows)

print('  ЛИСТ СВЕРКИ НА ПЕРВЫЙ ДЕНЬ: последние %d дней, набор без AAVE с XLM (%d монет)' % (DAYS, len(BOOK)), flush=True)
print('  колонки: час | строгий откат (монеты) | ширина | сделок в часе | что открылось: монета/источник/множитель/стоп%', flush=True)
shown = 0
for close in sorted(set(x[0] for x in rows if x[0] >= lo)):
    st = sorted(strict.get(close, []))
    width = len(set(strict.get(close, [])) | set(strict.get(close - 3600, [])))
    n = cnt_hour[close]
    items = []
    for x, k, w in zip(rows, keys, [r[5] for r in fin]):
        if x[0] != close:
            continue
        src = 'ширина' if k == 'r2_4' else ('откат1ч' if k == 'o1h' else ('откат2ч' if k == 'o2h' else 'банк ' + RULES[int(k[1:])].get('name', k)))
        ref = WIDE_REF if k == 'r2_4' else sim_w.REF
        cut = ' урезан' if x[3] > ref else ''
        items.append('%s/%s/x%.4g/стоп %.1f%%%s' % (x[4].replace('USDT', ''), src, w, 100 * x[3], cut))
    print('  %s | %-28s | ширина %d | сделок %2d | %s'
          % (ts(close), ','.join(st) if st else '-', width, n, '; '.join(items)), flush=True)
    shown += 1

sel = [(x, k, r[5]) for x, k, r in zip(rows, keys, fin) if x[0] >= lo]
print('', flush=True)
print('  ИТОГО за %d дней: часов со сделками %d, сделок %d, из них ширины %d'
      % (DAYS, shown, len(sel), sum(1 for x, k, w in sel if k == 'r2_4')), flush=True)
print('  часов с шириной >= 4: %d | часов с 10+ сделками: %d'
      % (len({c for c in set(x[0] for x, k, w in sel) if len(set(strict.get(c, [])) | set(strict.get(c - 3600, []))) >= 4}),
         len({c for c in set(x[0] for x, k, w in sel) if cnt_hour[c] >= 10})), flush=True)
mult = collections.Counter('%.4g' % w for x, k, w in sel)
print('  множители размера: %s' % ', '.join('x%s — %d сделок' % (m, c) for m, c in sorted(mult.items())), flush=True)
cut_n = sum(1 for x, k, w in sel if x[3] > (WIDE_REF if k == 'r2_4' else sim_w.REF))
print('  урезано по stop_ref: %d из %d сделок (%.0f%%)' % (cut_n, len(sel), 100 * cut_n / max(1, len(sel))), flush=True)
'''

src = head + TAIL
ast.parse(src)
io.open('dayreplay.py', 'w', encoding='utf-8').write(src)
print('dayreplay.py готов, синтаксис ок')
