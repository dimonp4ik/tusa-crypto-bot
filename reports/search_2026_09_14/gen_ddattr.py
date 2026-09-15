"""Writes ddattr.py: who makes the drawdown of the final system?

Every extension so far was judged by whether it deepened the drawdown. The final candidate (system +
boost n >= 10 + breadth over 2 hours) still has an 11-12% drawdown, and the drawdown sets the risk and
therefore the money. Before searching further, attribute it.

Per book at 1.4% risk, full fill:
  - the deepest drawdown episode: peak date, trough date, recovery date; trades that CLOSED inside the
    peak->trough window grouped by source (bank rule, pullback 1h, pullback 2h, breadth, short rules):
    count, stops, sum of money-weighted R, coins; and the hours of day and weekdays of those entries;
  - the five worst calendar weeks by equity change with the same source breakdown;
  - stop share and mean R per source per year.
"""
import ast
import io

s = io.open('rollbreadth.py', encoding='utf-8').read()
head = s[:s.index("CELLS = [")]

TAIL = r'''
import datetime as DT
SHORT_KEYS = {'r%d' % k for k, r in enumerate(RULES) if r.get('side', 'LONG') == 'SHORT'}
NAMES = {k: ('банк %s' % RULES[int(k[1:])].get('name', k)) for k in BASE_KEYS if k.startswith('r')}
NAMES.update({'o1h': 'откат 1ч', 'o2h': 'откат 2ч', 'r2_4': 'ширина 2ч'})


def book_keyed(sig, keys):
    busy, out = {}, []
    for close, prio, key, s, e, atr, fav, adv, Mu, Md, on, cn, tt, slip, vr, hour in sig:
        if key not in keys or busy.get(s, 0) > close:
            continue
        sl, tp, hh = GEO.get(key, (3.0, 1.0, HOLDS.get(key, 48)))
        H_ = hh * 4
        sf = sl * atr / e
        if sf > 0.10:
            continue
        js = int(np.searchsorted(Md[:H_], sl, side='left'))
        jt = int(np.searchsorted(Mu[:H_], tp, side='left'))
        if js < H_ and js <= jt:
            jj, x = js, min(-sl, on[js])
            kind = 'стоп'
        elif jt < H_:
            jj, x = jt, tp
            kind = 'тейк'
        else:
            jj, x = H_ - 1, cn[H_ - 1]
            kind = 'время'
        end = int(tt[jj]) + 900
        out.append(((close, end, (x * atr / e - slip - FEE) / sf, sf, s, 1.0), key, kind))
        busy[s] = end
    out.sort(key=lambda z: z[0])
    return out


def money_path(rows, target=0.014):
    """Replays sim_w.simulate and returns, per taken trade index, the money it made."""
    margin_frac = target / (sim_w.LEV * sim_w.REF)
    eq = peak = sim_w.DEPOSIT
    open_pos, curve, made = [], [], {}
    day, day_start, day_paused, paused = None, sim_w.DEPOSIT, False, False
    for idx, (a, b, R, sf, s, w) in enumerate(rows):
        while open_pos and open_pos[0][0] <= a:
            t_close, m, mpr, rr, j = open_pos.pop(0)
            eq += mpr * rr; peak = max(peak, eq); curve.append((t_close, eq)); made[j] = mpr * rr
        d = L.day_of(a)
        if d != day:
            day, day_start, day_paused = d, eq, False
        if not paused and eq <= peak * (1 - sim_w.MAX_DD):
            paused = True
        if not day_paused and eq <= day_start * (1 - sim_w.MAX_DAILY):
            day_paused = True
        if paused or day_paused:
            continue
        margin = margin_frac * eq
        if sf > sim_w.REF:
            margin *= sim_w.REF / sf
        margin *= w
        if sum(p[1] for p in open_pos) + margin > eq * sim_w.USABLE:
            continue
        open_pos.append((b, margin, margin * sim_w.LEV * sf, R, idx))
        open_pos.sort(key=lambda p: p[0])
    for t_close, m, mpr, rr, j in open_pos:
        eq += mpr * rr; curve.append((t_close, eq)); made[j] = mpr * rr
    curve.sort()
    return curve, made


def ts(t):
    return DT.datetime.fromtimestamp(t, DT.UTC).strftime('%Y-%m-%d')


def breakdown(items, label):
    by = collections.defaultdict(lambda: [0, 0, 0.0, collections.Counter()])
    for row, key, kind, money in items:
        src = 'шорт-правила' if key in SHORT_KEYS else NAMES.get(key, key)
        b = by[src]
        b[0] += 1; b[1] += int(kind == 'стоп'); b[2] += money; b[3][row[4].replace('USDT', '')] += 1
    tot = sum(v[2] for v in by.values())
    print('      %s: сделок %d, итог $%+.0f' % (label, len(items), tot), flush=True)
    for src, (n, st, mon, coins) in sorted(by.items(), key=lambda kv: kv[1][2]):
        print('        %-34s %3d сд, стопов %3d, $%+7.0f | монеты %s' % (src, n, st, mon, ', '.join('%s %d' % c for c in coins.most_common(5))), flush=True)


for book_name, coins in BOOKS3:
    H = {s: coin_hours(s) for s in coins}
    base_sig = make_sig_raw(coins)
    extra = roll_entries(coins, H, 'r2_4', 4, 2, 4, 0)
    sig_all = sorted(base_sig + extra, key=lambda x: (x[0], x[1]))
    kb = book_keyed(sig_all, BASE_KEYS | {'r2_4'})
    rows = boost([z[0] for z in kb])
    curve, made = money_path(rows)
    items = [(rows[i], kb[i][1], kb[i][2], made[i]) for i in made]
    print('', flush=True)
    print('  ===== %s: итог $%.0f =====' % (book_name, curve[-1][1]), flush=True)
    peak_t, peak_v, worst, ep = curve[0][0], curve[0][1], 0.0, None
    for t, v in curve:
        if v > peak_v:
            peak_t, peak_v = t, v
        dd = v / peak_v - 1
        if dd < worst:
            worst, ep = dd, (peak_t, t, peak_v, v)
    pt, tt_, pv, tv = ep
    rec = next((t for t, v in curve if t > tt_ and v >= pv), None)
    print('    ХУДШАЯ ПРОСАДКА %.1f%%: пик %s ($%.0f) -> дно %s ($%.0f) -> восстановление %s'
          % (100 * worst, ts(pt), pv, ts(tt_), tv, ts(rec) if rec else 'не восстановилась'), flush=True)
    win = [it for it in items if pt < it[0][1] <= tt_]
    breakdown(win, 'сделки, закрытые с пика до дна')
    hrs = collections.Counter(DT.datetime.fromtimestamp(it[0][0], DT.UTC).hour for it in win if it[2] == 'стоп')
    wds = collections.Counter(DT.datetime.fromtimestamp(it[0][0], DT.UTC).weekday() for it in win if it[2] == 'стоп')
    print('        стопы по часу входа: %s' % ', '.join('%02dч %d' % h for h in sorted(hrs.items())), flush=True)
    print('        стопы по дню недели: %s' % ', '.join('%s %d' % (['пн', 'вт', 'ср', 'чт', 'пт', 'сб', 'вс'][d], n) for d, n in sorted(wds.items())), flush=True)

    weeks = collections.defaultdict(float)
    for it in items:
        w = DT.datetime.fromtimestamp(it[0][1], DT.UTC)
        weeks[(w - DT.timedelta(days=w.weekday())).strftime('%Y-%m-%d')] += it[3]
    print('    ПЯТЬ ХУДШИХ НЕДЕЛЬ (по деньгам закрытых сделок):', flush=True)
    for wk, mon in sorted(weeks.items(), key=lambda kv: kv[1])[:5]:
        lo = int(DT.datetime.strptime(wk, '%Y-%m-%d').replace(tzinfo=DT.UTC).timestamp())
        breakdown([it for it in items if lo <= it[0][1] < lo + 7 * 86400], 'неделя с %s' % wk)

    print('    ИСТОЧНИКИ ПО ГОДАМ: стопов % / ср R / сделок', flush=True)
    srcs = sorted({('шорт-правила' if k in SHORT_KEYS else NAMES.get(k, k)) for _, k, _, _ in items})
    for src in srcs:
        cells = []
        for y in range(2022, 2027):
            v = [(it[0][2], it[2]) for it in items if ('шорт-правила' if it[1] in SHORT_KEYS else NAMES.get(it[1], it[1])) == src
                 and YT[y] <= it[0][0] < YT[y + 1]]
            cells.append('%4.1f%% %+.3f %3d' % (100 * np.mean([k == 'стоп' for _, k in v]), np.mean([r for r, _ in v]), len(v)) if v else '       -        ')
        print('      %-34s %s' % (src, ' | '.join(cells)), flush=True)
'''

src = head + TAIL
ast.parse(src)
io.open('ddattr.py', 'w', encoding='utf-8').write(src)
print('ddattr.py готов, синтаксис ок')
