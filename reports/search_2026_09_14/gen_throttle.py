"""Writes throttle.py: extra pullback sources, admitted only when the account can carry them.

finer.py showed the extra grids (45m, 1h shifted by 15/30/45m, 30m) are profitable on their own but
push the stacked drawdown to 12.7-15%: they add longs in the same sell-offs where the system's longs
are already open. A cap on all positions and a size cut in drawdown were closed before - both took
money from the recoveries of the core trades. Here the core system is never touched; only the NEW
sources are gated, by state known at the moment of entry:

  dd X%   - an extra trade is skipped while the account is more than X% below its peak
  open K  - an extra trade is skipped while K or more positions are already open

A copy of sim_w.simulate carries the gate; with no gate it must reproduce sim_w.simulate to the cent
(self-test printed). Both books, money at 1.4% and at 12% drawdown (bisection with the gate on),
per-year versus the current system, blind choice on book 1.
"""
import ast
import io

s = io.open('finer.py', encoding='utf-8').read()
head = s[:s.index("BASE_KEYS = {")]

TAIL = r'''
BASE_KEYS = {'r0', 'r1', 'r2', 'r3', 'r4', 'o1h', 'o2h'}
EXTRA = [('o30', 3, 1800, 0), ('o45', 4, 2700, 0), ('o1h15', 5, 3600, 900), ('o1h30', 6, 3600, 1800),
         ('o1h45', 7, 3600, 2700)]
MC, LL = sim_w.MC, sim_w.L


def book_k(sig, keys):
    busy, out = {}, []
    for close, prio, key, s, e, atr, fav, adv, Mu, Md, on, cn, tt, slip, vr, hour in sig:
        if key not in keys or busy.get(s, 0) > close:
            continue
        sl, tp, hh = GEO.get(key, (3.0, 1.0, HOLDS.get(key, 48)))
        H = hh * 4
        sf = sl * atr / e
        if sf > 0.10:
            continue
        js = int(np.searchsorted(Md[:H], sl, side='left'))
        jt = int(np.searchsorted(Mu[:H], tp, side='left'))
        if js < H and js <= jt:
            jj, x = js, min(-sl, on[js])
        elif jt < H:
            jj, x = jt, tp
        else:
            jj, x = H - 1, cn[H - 1]
        end = int(tt[jj]) + 900
        out.append((close, end, (x * atr / e - slip - FEE) / sf, sf, s, 1.0, key not in BASE_KEYS))
        busy[s] = end
    out.sort()
    return out


def sim_g(trades, target, gate=None):
    margin_frac = target / (sim_w.LEV * sim_w.REF)
    eq = peak = sim_w.DEPOSIT
    open_pos = []
    day, day_start, day_paused = None, sim_w.DEPOSIT, False
    paused, paused_at = False, None
    curve = []
    gated = 0
    for a, b, R, sf, s, w, extra in trades:
        while open_pos and open_pos[0][0] <= a:
            t_close, m, money_per_R, rr = open_pos.pop(0)
            eq += money_per_R * rr
            peak = max(peak, eq)
            curve.append((t_close, eq))
        d = LL.day_of(a)
        if d != day:
            day, day_start, day_paused = d, eq, False
        if not paused and eq <= peak * (1 - sim_w.MAX_DD):
            paused, paused_at = True, a
        if not day_paused and eq <= day_start * (1 - sim_w.MAX_DAILY):
            day_paused = True
        if paused or day_paused:
            continue
        if extra and gate is not None:
            kind, v = gate
            if (kind == 'dd' and eq <= peak * (1 - v)) or (kind == 'open' and len(open_pos) >= v):
                gated += 1
                continue
        margin = margin_frac * eq
        if sf > sim_w.REF:
            margin *= sim_w.REF / sf
        margin *= w
        used = sum(p[1] for p in open_pos)
        if used + margin > eq * sim_w.USABLE:
            continue
        open_pos.append((b, margin, margin * sim_w.LEV * sf, R))
        open_pos.sort(key=lambda p: p[0])
    for t_close, m, money_per_R, rr in open_pos:
        eq += money_per_R * rr
        curve.append((t_close, eq))
    return dict(eq=eq, curve=curve, paused_at=paused_at, gated=gated)


def dd12_g(trades, gate, want=0.12):
    lo, hi = 0.0005, 0.030
    for _ in range(20):
        mid = (lo + hi) / 2
        r = sim_g(trades, mid, gate)
        if bool(r['paused_at']) or abs(LL.dd_of(r['curve'])) > want:
            hi = mid
        else:
            lo = mid
    return sim_g(trades, lo, gate)['eq']


def yrs_g(rows, gate):
    return [sim_g([x for x in rows if YT[y] <= x[0] < YT[y + 1]], 0.014, gate)['eq'] for y in range(2022, 2027)]


SOURCES = (('45м', {'o45'}), ('1ч сдвиг 30м', {'o1h30'}), ('1ч сдвиги 15/30/45', {'o1h15', 'o1h30', 'o1h45'}),
           ('30м+45м+1ч сдвиги', {'o30', 'o45', 'o1h15', 'o1h30', 'o1h45'}))
GATES = (('без ограничителя', None), ('dd 3%', ('dd', 0.03)), ('dd 5%', ('dd', 0.05)), ('dd 7%', ('dd', 0.07)),
         ('open 3', ('open', 3)), ('open 5', ('open', 5)), ('open 8', ('open', 8)))
BOOKS = (('КНИГА 1 (без XLM, AAVE)', [c for c in SP.COINS if c not in ('BILLUSDT', 'XLMUSDT', 'AAVEUSDT')]),
         ('КНИГА 2 (все 15 монет)', [c for c in SP.COINS if c != 'BILLUSDT']))

for bi, (book_name, coins) in enumerate(BOOKS):
    sig = make_sig_raw(coins)
    for key, prio, sec, off in EXTRA:
        sig += long_entries(coins, key, prio, sec, off)
    sig.sort(key=lambda x: (x[0], x[1]))
    base_rows = book_k(sig, BASE_KEYS)
    if bi == 0:
        ref = sim_w.simulate([x[:6] for x in base_rows], 0.014)['eq']
        mine = sim_g(base_rows, 0.014)['eq']
        print('  САМОПРОВЕРКА: копия симулятора $%.2f против sim_w $%.2f %s'
              % (mine, ref, 'совпало' if abs(mine - ref) < 0.01 else 'РАСХОЖДЕНИЕ'), flush=True)
    b_eq = sim_g(base_rows, 0.014)
    base_y = yrs_g(base_rows, None)
    print('', flush=True)
    print('  ===== %s: система $%.0f просадка %.1f%% DD12 $%.0f =====' % (book_name, b_eq['eq'], 100 * abs(LL.dd_of(b_eq['curve'])), dd12_g(base_rows, None)), flush=True)
    RES = {'система (сейчас)': (base_rows, None)}
    for sname, skeys in SOURCES:
        rows = book_k(sig, BASE_KEYS | skeys)
        for gname, gate in GATES:
            r = sim_g(rows, 0.014, gate)
            py = yrs_g(rows, gate)
            lbl = '+ %s, %s' % (sname, gname)
            RES[lbl] = (rows, gate)
            print('    %-40s $%6.0f %4.1f%% | DD12 $%6.0f | отсечено доп %4d | лучше по годам %d/5 (%s)'
                  % (lbl, r['eq'], 100 * abs(LL.dd_of(r['curve'])), dd12_g(rows, gate), r['gated'],
                     sum(1 for a, b in zip(base_y, py) if b > a), ' '.join('%+.0f' % (b - a) for a, b in zip(base_y, py))), flush=True)
    if bi == 0:
        print('    -- слепой выбор (мера: $ при 1.4%): 4 года выбор, 5-й замер --', flush=True)
        wins = 0
        for y in range(2022, 2027):
            lo, hi = YT[y], YT[y + 1]
            best, bk = -1, None
            for lbl, (rows, gate) in RES.items():
                mm = sim_g([x for x in rows if not (lo <= x[0] < hi)], 0.014, gate)['eq']
                if mm > best:
                    best, bk = mm, lbl
            rows, gate = RES[bk]
            mp = sim_g([x for x in rows if lo <= x[0] < hi], 0.014, gate)['eq']
            mb = sim_g([x for x in base_rows if lo <= x[0] < hi], 0.014)['eq']
            wins += int(mp > mb)
            print('      %d выбрано «%s» | $%.0f против $%.0f %s'
                  % (y, bk, mp, mb, 'лучше' if mp > mb else ('так же' if mp == mb else 'хуже')), flush=True)
        print('      лучше нынешней системы в %d из 5 лет' % wins, flush=True)
'''

src = head + TAIL
ast.parse(src)
io.open('throttle.py', 'w', encoding='utf-8').write(src)
print('throttle.py готов, синтаксис ок')
