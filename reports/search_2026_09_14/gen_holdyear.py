"""Writes holdyear.py: is the 24h-hold gain one year wide, or spread?

widehold's plateau is jagged (36h loses to 48h), its blind-year selection wins 1 of 5 years on book 1 and
0 of 5 on book 2, and its per-year money ratios read +0.0/-0.1/+0.0/+0.0/+2.9%. That smells like the whole
"+3%" is a single year - and since the three books share the same coins, the same handful of trades would
be driving all three "independent" confirmations.

This run measures it directly: per book, per year, how many wide trades there were, how many of them
actually hit the 48h time exit (only those can change), and the money ratio of 24h against 48h. If the
gain sits in one year, the knob is not established and goes in the rejected pile.
"""
import ast
import io

s = io.open('widestopref.py', encoding='utf-8').read()
head = s[:s.index("for bi, (book_name, coins) in enumerate(BOOKS3):")]

TAIL = r'''
import datetime as DT
WIDE_REF = 0.060
YEARS = (2022, 2023, 2024, 2025, 2026)


def yr(ts):
    return DT.datetime.fromtimestamp(ts, DT.UTC).year


def build(sig, hh):
    GEO.clear(); GEO.update(BASE_GEO); GEO['r2_4'] = (3.0, 1.0, hh)
    rows, keys = book_keyed(sig, FULL)
    fin = weigh_src(rows, keys, {'r2_4'}, 1.25)
    refs = [(WIDE_REF if k == 'r2_4' else sim_w.REF) for k in keys]
    return fin, refs, keys


def money(fin, refs, y=None):
    if y is not None:
        ix = [i for i, x in enumerate(fin) if yr(x[0]) == y]
        fin = [fin[i] for i in ix]; refs = [refs[i] for i in ix]
    if not fin:
        return None, None
    r = sim_ref(fin, 0.014, refs)
    return r['eq'], money_at_dd_ref(fin, refs)


print('  РАЗБОР ПО ГОДАМ: откуда берётся выигрыш удержания 24ч', flush=True)
for book_name, coins in BOOKS3:
    H = {s: coin_hours(s) for s in coins}
    sig_all = sorted(make_sig_raw(coins) + roll_entries(coins, H, 'r2_4', 4, 2, 4, 0),
                     key=lambda x: (x[0], x[1]))
    f24, r24, k24 = build(sig_all, 24)
    f48, r48, k48 = build(sig_all, 48)
    print('', flush=True)
    print('  %s' % book_name, flush=True)
    # a wide trade can only change if its 48h result differs from its 24h result
    d24 = {(x[0], x[4]): x for x, k in zip(f24, k24) if k == 'r2_4'}
    d48 = {(x[0], x[4]): x for x, k in zip(f48, k48) if k == 'r2_4'}
    same = set(d24) & set(d48)
    changed = [t for t in same if abs(d24[t][2] - d48[t][2]) > 1e-9]
    print('    широких сделок %d, из них изменил исход срез до 24ч: %d (%.0f%%)'
          % (len(same), len(changed), 100 * len(changed) / max(1, len(same))), flush=True)
    ch_y = collections.Counter(yr(t[0]) for t in changed)
    gain = sum(d24[t][2] - d48[t][2] for t in changed)
    print('    суммарно по этим сделкам %+.2fR; изменённых по годам: %s'
          % (gain, ', '.join('%d: %d' % (y, ch_y.get(y, 0)) for y in YEARS)), flush=True)
    eA, mA = money(f24, r24); eB, mB = money(f48, r48)
    print('    вся история: $%.0f против $%.0f (%+.1f%%), при равной просадке %+.1f%%'
          % (eA, eB, 100 * (eA / eB - 1), 100 * (mA / mB - 1)), flush=True)
    for y in YEARS:
        eA, mA = money(f24, r24, y); eB, mB = money(f48, r48, y)
        if eA is None or eB is None:
            continue
        gy = sum(d24[t][2] - d48[t][2] for t in changed if yr(t[0]) == y)
        print('      %d: $%5.0f против $%5.0f (%+5.1f%%), при равной просадке %+5.1f%% | изменённых %d на %+.2fR'
              % (y, eA, eB, 100 * (eA / eB - 1), 100 * (mA / mB - 1), ch_y.get(y, 0), gy), flush=True)
'''

src = head + TAIL
ast.parse(src)
io.open('holdyear.py', 'w', encoding='utf-8').write(src)
print('holdyear.py gotov, sintaksis ok')
