"""Writes trimrun.py: cut the size of the weakest rule - "coin run in the evening" (r1).

srcstat put the eight sources side by side and exactly one weakness repeats on all three books: r1 earns
+0.038..+0.046R per trade (three to four times thinner than its neighbours in the bank), takes 25-27% of
all losses for 18-19% of the profit, and is negative inside every book's worst drawdown window. Ablation
says the rule cannot be REMOVED; nobody has asked whether it should be sized DOWN.

Our own discipline allows cutting below base only when the subset is weak in a hostile window - which is
what the worst-window column shows. So: multipliers 1.00/0.75/0.50/0.25 on r1 only, three books, both
measures, plus the two things that killed the last candidate -
  * per-year R of r1, to see whether the weakness is one era or all of them;
  * a CONTROL that trims the same number of randomly chosen trades from other sources;
  * a blind choice of the multiplier on four years, measured on the fifth.
"""
import ast
import io

s = io.open('widestopref.py', encoding='utf-8').read()
head = s[:s.index("for bi, (book_name, coins) in enumerate(BOOKS3):")]

TAIL = r'''
import datetime as DT
WIDE_REF = 0.060
MULTS = (1.00, 0.75, 0.50, 0.25)
YEARS = (2022, 2023, 2024, 2025, 2026)
TRIM = 'r1'


def yr(ts):
    return DT.datetime.fromtimestamp(ts, DT.UTC).year


def weigh_trim(rows, keys, mult, pick=None):
    cnt = collections.Counter(x[0] for x in rows)
    out = []
    for i, (x, k) in enumerate(zip(rows, keys)):
        w = 1.25 if cnt[x[0]] >= 10 else 1.0
        if k == 'r2_4':
            w *= 1.25
        if (pick is None and k == TRIM) or (pick is not None and i in pick):
            w *= mult
        out.append(x[:5] + (w,))
    return out


def meas(fin, refs, keep=None):
    if keep is not None:
        ix = [i for i, x in enumerate(fin) if keep(x[0])]
        fin = [fin[i] for i in ix]; refs = [refs[i] for i in ix]
    r = sim_ref(fin, 0.014, refs)
    return r['eq'], abs(L.dd_of(r['curve'])), money_at_dd_ref(fin, refs)


print('  УРЕЗАНИЕ РАЗМЕРА САМОГО СЛАБОГО ПРАВИЛА (разгон монеты вечером)', flush=True)
for book_name, coins in BOOKS3:
    H = {s: coin_hours(s) for s in coins}
    sig_all = sorted(make_sig_raw(coins) + roll_entries(coins, H, 'r2_4', 4, 2, 4, 0),
                     key=lambda x: (x[0], x[1]))
    GEO.clear(); GEO.update(BASE_GEO)
    rows, keys = book_keyed(sig_all, FULL)
    refs = [(WIDE_REF if k == 'r2_4' else sim_w.REF) for k in keys]
    idx = [i for i, k in enumerate(keys) if k == TRIM]
    print('', flush=True)
    print('  %s — сделок правила %d из %d' % (book_name, len(idx), len(rows)), flush=True)
    print('    ПО ГОДАМ (это правило): %s' % ', '.join(
        '%d: %d сд. ср%+.3f' % (y, sum(1 for i in idx if yr(rows[i][0]) == y),
                                float(np.mean([rows[i][2] for i in idx if yr(rows[i][0]) == y] or [0])))
        for y in YEARS), flush=True)
    base = weigh_trim(rows, keys, 1.0)
    e0, d0, m0 = meas(base, refs)
    print('    КОНТРОЛЬ без урезания: $%.0f просадка %.1f%% при равной просадке $%.0f' % (e0, 100 * d0, m0), flush=True)
    for mult in MULTS[1:]:
        fin = weigh_trim(rows, keys, mult)
        e, d, m = meas(fin, refs)
        ce, cm = [], []
        for seed in range(5):
            rng = np.random.default_rng(seed)
            other = [i for i in range(len(rows)) if keys[i] != TRIM]
            pick = set(rng.choice(other, size=min(len(idx), len(other)), replace=False).tolist())
            cf = weigh_trim(rows, keys, mult, pick=pick)
            x, _, z = meas(cf, refs)
            ce.append(x); cm.append(z)
        print('    x%.2f: $%6.0f (%+5.1f%%) просадка %.1f%% | при равной просадке $%6.0f (%+5.1f%%) | КОНТРОЛЬ случайные $%.0f (%+.1f%%) DD12 $%.0f (%+.1f%%)'
              % (mult, e, 100 * (e / e0 - 1), 100 * d, m, 100 * (m / m0 - 1),
                 np.median(ce), 100 * (np.median(ce) / e0 - 1), np.median(cm), 100 * (np.median(cm) / m0 - 1)), flush=True)

    print('    СЛЕПОЙ ВЫБОР множителя по четырём годам, замер на пятом', flush=True)
    wins = 0
    for y in YEARS:
        sc = []
        for mult in MULTS:
            fin = weigh_trim(rows, keys, mult)
            e, _, m = meas(fin, refs, keep=lambda t: yr(t) != y)
            sc.append((min(e / e0, m / m0), mult) if mult == 1.0 else (0, mult))
        sc = []
        eb, _, mb = meas(base, refs, keep=lambda t: yr(t) != y)
        for mult in MULTS:
            fin = weigh_trim(rows, keys, mult)
            e, _, m = meas(fin, refs, keep=lambda t: yr(t) != y)
            sc.append((min(e / eb, m / mb), mult))
        sc.sort(reverse=True)
        pick = sc[0][1]
        fin = weigh_trim(rows, keys, pick)
        e, _, m = meas(fin, refs, keep=lambda t: yr(t) == y)
        eb, _, mb = meas(base, refs, keep=lambda t: yr(t) == y)
        ok = (e > eb and m > mb)
        wins += ok
        print('      %d: выбрано x%.2f | на этом году %+5.1f%% денег, %+5.1f%% при равной просадке %s'
              % (y, pick, 100 * (e / eb - 1), 100 * (m / mb - 1), 'лучше' if ok else ''), flush=True)
    print('      лет, где выбранный вслепую множитель лучше базы по ОБЕИМ мерам: %d из %d' % (wins, len(YEARS)), flush=True)
'''

src = head + TAIL
ast.parse(src)
io.open('trimrun.py', 'w', encoding='utf-8').write(src)
print('trimrun.py gotov, sintaksis ok')
