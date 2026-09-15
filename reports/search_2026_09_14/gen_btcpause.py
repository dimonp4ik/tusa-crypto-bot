"""Writes btcpause.py: no new longs while BTC is falling fast.

Every extra source died the same way: its trades are good on their own, but they arrive in the
sell-offs where the system's longs are already open, and the drawdown - not the edge - sets the
ceiling. The system's own drawdown is made by the same clusters. A pause that sees only the past:
when BTC's return over the last 4h / 12h / 24h (closed 15m bars before the signal) is below a
threshold, no new LONG is opened, from any source. Shorts are untouched.

Controls: the mirror (pause longs after BTC rose by the same amount) and a random skip of the same
number of longs (median of 5 seeds). Skipped trades' own mean R is printed - if it is positive the
pause is paying for drawdown with edge. Both books, money at 1.4% and at 12% drawdown, per-year
versus the current system, blind choice on book 1 among the real pauses and the current system.
"""
import ast
import io

s = io.open('finer.py', encoding='utf-8').read()
head = s[:s.index("BASE_KEYS = {")]

TAIL = r'''
BASE_KEYS = {'r0', 'r1', 'r2', 'r3', 'r4', 'o1h', 'o2h'}
SHORT_KEYS = {'r%d' % k for k, r in enumerate(RULES) if r.get('side', 'LONG') == 'SHORT'}
BT = np.asarray(SP.CTX['BTCUSDT']['t15'], dtype=np.int64)
BC = np.asarray(SP.CTX['BTCUSDT']['a15'], dtype=float)[:, 3]


def btc_ret(close, L):
    j = int(np.searchsorted(BT, close, side='left')) - 1      # last 15m bar that closed by `close`
    while j >= 0 and BT[j] + 900 > close:
        j -= 1
    if j - L < 0:
        return float('nan')
    return BC[j] / BC[j - L] - 1


def fix(rows):
    r = sim_w.simulate(rows, 0.014)
    return r['eq'], abs(L.dd_of(r['curve']))


PAUSES = [('4ч', 16, -0.015), ('4ч', 16, -0.025), ('4ч', 16, -0.04),
          ('12ч', 48, -0.03), ('12ч', 48, -0.05), ('12ч', 48, -0.07),
          ('24ч', 96, -0.04), ('24ч', 96, -0.06), ('24ч', 96, -0.09)]
BOOKS = (('КНИГА 1 (без XLM, AAVE)', [c for c in SP.COINS if c not in ('BILLUSDT', 'XLMUSDT', 'AAVEUSDT')]),
         ('КНИГА 2 (все 15 монет)', [c for c in SP.COINS if c != 'BILLUSDT']))

for bi, (book_name, coins) in enumerate(BOOKS):
    sig = [x for x in make_sig_raw(coins) if x[2] in BASE_KEYS]
    longs = [k for k, x in enumerate(sig) if x[2] not in SHORT_KEYS]
    RET = {Lb: np.array([btc_ret(sig[k][0], Lb) for k in longs]) for Lb in (16, 48, 96)}
    base_rows, _ = book(sig, BASE_KEYS)
    beq, bdd = fix(base_rows)
    bpy = pyr(base_rows)
    all_R = {}
    for r in base_rows:
        all_R[(r[0], r[4])] = r[2]
    print('', flush=True)
    print('  ===== %s: система $%.0f просадка %.1f%% DD12 $%.0f, лонг-сигналов %d =====' % (book_name, beq, 100 * bdd, sim_w.money_at_dd(base_rows, 0.12)[1], len(longs)), flush=True)
    RES = {'система (сейчас)': base_rows}

    def run(lbl, drop_idx, show=True):
        drop = set(drop_idx)
        sub = [x for k, x in enumerate(sig) if k not in drop]
        rows, _ = book(sub, BASE_KEYS)
        eq, dd = fix(rows)
        m = sim_w.money_at_dd(rows, 0.12)[1]
        py = pyr(rows)
        skippedR = [all_R[(sig[k][0], sig[k][3])] for k in drop if (sig[k][0], sig[k][3]) in all_R]
        if show:
            print('    %-34s пропущено сигналов %4d (из книги %4d, их ср R %s) | $%6.0f %4.1f%% | DD12 $%6.0f | лучше по годам %d/5 (%s)'
                  % (lbl, len(drop), len(skippedR), ('%+.3f' % np.mean(skippedR)) if skippedR else '  -  ', eq, 100 * dd, m,
                     sum(1 for a, b in zip(bpy, py) if b > a), ' '.join('%+.0f' % (b - a) for a, b in zip(bpy, py))), flush=True)
        return rows, eq, m

    for name, Lb, thr in PAUSES:
        r = RET[Lb]
        with np.errstate(invalid='ignore'):
            idx = [longs[i] for i in np.flatnonzero(r <= thr)]
            mir = [longs[i] for i in np.flatnonzero(r >= -thr)]
        rows, _, _ = run('пауза: BTC за %s <= %.1f%%' % (name, 100 * thr), idx)
        RES['пауза %s %.1f%%' % (name, 100 * thr)] = rows
        run('  зеркало: BTC за %s >= +%.1f%%' % (name, -100 * thr), mir)
        eqs, ms = [], []
        for seed in range(5):
            rng = np.random.default_rng(100 + seed)
            pick = list(rng.choice(longs, size=len(idx), replace=False)) if idx else []
            _, e, m = run('', pick, show=False)
            eqs.append(e); ms.append(m)
        print('      случайный пропуск %d лонгов: $%.0f, DD12 $%.0f (медиана 5 розыгрышей)' % (len(idx), np.median(eqs), np.median(ms)), flush=True)
    if bi == 0:
        print('    -- слепой выбор (мера: $ при 1.4%): 4 года выбор, 5-й замер --', flush=True)
        wins = 0
        for y in range(2022, 2027):
            lo, hi = YT[y], YT[y + 1]
            best, bk = -1, None
            for lbl, rows in RES.items():
                mm = sim_w.simulate([x for x in rows if not (lo <= x[0] < hi)], 0.014)['eq']
                if mm > best:
                    best, bk = mm, lbl
            mp = sim_w.simulate([x for x in RES[bk] if lo <= x[0] < hi], 0.014)['eq']
            mb = sim_w.simulate([x for x in base_rows if lo <= x[0] < hi], 0.014)['eq']
            wins += int(mp > mb)
            print('      %d выбрано «%s» | $%.0f против $%.0f %s'
                  % (y, bk, mp, mb, 'лучше' if mp > mb else ('так же' if mp == mb else 'хуже')), flush=True)
        print('      лучше нынешней системы в %d из 5 лет' % wins, flush=True)
'''

src = head + TAIL
ast.parse(src)
io.open('btcpause.py', 'w', encoding='utf-8').write(src)
print('btcpause.py готов, синтаксис ок')
