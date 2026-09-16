"""Writes widehold.py: the one knob the geometry grid actually found - hold the wide pullbacks 24h.

widegeo swept stop x take x hold separately for source r2_4 on all three books. The stop/take winners
disagreed between books (overfitting), but ONE cell agreed everywhere: at unchanged 3.0/1.0 geometry,
cutting the hold from 48h to 24h gained +3.4/+3.0/+3.2% of money and +3.9/+3.3/+3.4% at equal drawdown,
with the trade COUNT unchanged (52/52, 87/87, 72/72) - a paired comparison where only the exit differs.

This run tests that single knob properly:
  1. Plateau - 6/12/18/24/30/36/48/72h, to see whether 24 sits on a ridge or a cliff.
  2. Blind - choose the hold on four years, measure it on the fifth, for each of the five.
  3. 85% fill - ten paired draws, the same subset given to both holds.
"""
import ast
import io

s = io.open('widestopref.py', encoding='utf-8').read()
head = s[:s.index("for bi, (book_name, coins) in enumerate(BOOKS3):")]

TAIL = r'''
import datetime as DT
WIDE_REF = 0.060
HOLDS_T = (6, 12, 18, 24, 30, 36, 48, 72)
YEARS = (2022, 2023, 2024, 2025, 2026)


def yr(ts):
    return DT.datetime.fromtimestamp(ts, DT.UTC).year


def build(sig, hh):
    GEO.clear(); GEO.update(BASE_GEO); GEO['r2_4'] = (3.0, 1.0, hh)
    rows, keys = book_keyed(sig, FULL)
    fin = weigh_src(rows, keys, {'r2_4'}, 1.25)
    refs = [(WIDE_REF if k == 'r2_4' else sim_w.REF) for k in keys]
    return fin, refs, keys


def measure(fin, refs, keep=None):
    if keep is not None:
        ix = [i for i, x in enumerate(fin) if keep(x[0])]
        fin = [fin[i] for i in ix]; refs = [refs[i] for i in ix]
    r = sim_ref(fin, 0.014, refs)
    return r['eq'], abs(L.dd_of(r['curve'])), money_at_dd_ref(fin, refs)


for book_name, coins in BOOKS3:
    H = {s: coin_hours(s) for s in coins}
    sig_all = sorted(make_sig_raw(coins) + roll_entries(coins, H, 'r2_4', 4, 2, 4, 0),
                     key=lambda x: (x[0], x[1]))
    print('', flush=True)
    print('  %s' % book_name, flush=True)
    cache = {hh: build(sig_all, hh) for hh in HOLDS_T}
    e0, d0, m0 = measure(*cache[48][:2])
    print('    ПЛАТО ПО УДЕРЖАНИЮ ШИРОКИХ ОТКАТОВ (геометрия 3.0/1.0 не меняется)', flush=True)
    for hh in HOLDS_T:
        fin, refs, keys = cache[hh]
        e, d, m = measure(fin, refs)
        mark = '  <-- как сейчас' if hh == 48 else ''
        print('      %2dч: $%6.0f (%+5.1f%%) просадка %.1f%% | при равной просадке $%6.0f (%+5.1f%%) | широких %d%s'
              % (hh, e, 100 * (e / e0 - 1), 100 * d, m, 100 * (m / m0 - 1),
                 sum(1 for k in keys if k == 'r2_4'), mark), flush=True)

    print('    СЛЕПОЙ ВЫБОР: удержание выбрано по четырём годам, замер на пятом', flush=True)
    wins = 0
    for y in YEARS:
        scores = []
        for hh in HOLDS_T:
            fin, refs, _ = cache[hh]
            e, _, m = measure(fin, refs, keep=lambda t: yr(t) != y)
            fb, rb, _ = cache[48]
            eb, _, mb = measure(fb, rb, keep=lambda t: yr(t) != y)
            scores.append((min(e / eb, m / mb), hh))
        scores.sort(reverse=True)
        pick = scores[0][1]
        fin, refs, _ = cache[pick]
        e, _, m = measure(fin, refs, keep=lambda t: yr(t) == y)
        fb, rb, _ = cache[48]
        eb, _, mb = measure(fb, rb, keep=lambda t: yr(t) == y)
        ok = (e > eb and m > mb)
        wins += ok
        print('      %d: выбрано %2dч | на этом году $%6.0f против $%6.0f (%+5.1f%%), при равной просадке %+5.1f%% %s'
              % (y, pick, e, eb, 100 * (e / eb - 1), 100 * (m / mb - 1), 'лучше' if ok else ''), flush=True)
    print('      лет, где выбранное вслепую удержание лучше нынешнего по ОБЕИМ мерам: %d из %d' % (wins, len(YEARS)), flush=True)

    print('    ЗАЛИВКА 85%, десять парных розыгрышей (24ч против 48ч на одном и том же наборе)', flush=True)
    RE, RM, RD = [], [], []
    for seed in range(1, 11):
        rng = np.random.default_rng(seed)
        sub = [x for x in sig_all if rng.random() < 0.85]
        f24, r24, _ = build(sub, 24); f48, r48, _ = build(sub, 48)
        e24, d24, m24 = measure(f24, r24); e48, d48, m48 = measure(f48, r48)
        RE.append(e24 / e48 - 1); RM.append(m24 / m48 - 1); RD.append((d24, d48))
    md = lambda X: float(np.median(X))
    print('      деньги %+.1f%% (худший розыгрыш %+.1f%%, лучший %+.1f%%) | при равной просадке %+.1f%% | просадка %.1f%% против %.1f%%'
          % (100 * md(RE), 100 * min(RE), 100 * max(RE), 100 * md(RM),
             100 * md([a for a, b in RD]), 100 * md([b for a, b in RD])), flush=True)
    print('      розыгрышей, где 24ч лучше по деньгам: %d из 10; по равной просадке: %d из 10'
          % (sum(1 for x in RE if x > 0), sum(1 for x in RM if x > 0)), flush=True)
'''

src = head + TAIL
ast.parse(src)
io.open('widehold.py', 'w', encoding='utf-8').write(src)
print('widehold.py gotov, sintaksis ok')
