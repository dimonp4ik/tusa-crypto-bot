"""Writes confirm.py: enter the pullback only once the 15m tape turns up.

The stop anatomy found 29% of stopped trades reach the target later inside the window, and the
median stop comes 11h in: the dip keeps going after the signal before it recovers. The pullback
sources (1h and 2h) enter at the next 15m open. Here they wait for a turn: a 15m bar that closes
above the highest high of the 4 bars before it, inside a window of W bars after the signal; entry at
the next 15m open; no turn inside the window - no trade. Stop, take and hold are unchanged and
counted from the new entry. A limit entry deeper in the dip was closed before; this one buys higher,
after the turn.

Control: the same signals entered after a random delay drawn from the confirmed trades' delays -
this separates "waiting for a turn" from "simply entering later". Bank rules keep their entries.
Both books, money at 1.4% and at 12% drawdown, per-year versus the current system, blind choice on
book 1.
"""
import ast
import io

s = io.open('finer.py', encoding='utf-8').read()
head = s[:s.index("BASE_KEYS = {")]

TAIL = r'''
BASE_KEYS = {'r0', 'r1', 'r2', 'r3', 'r4', 'o1h', 'o2h'}


def entry_row(s, close, key, prio, atr, jj, c, slip):
    t15, a15 = c['t15'], c['a15']
    if jj + MAXH > len(t15):
        return None
    o, h, l, cc = (a15[jj:jj + MAXH, x] for x in range(4))
    e = o[0] * (1 + slip)
    fav = (h - e) / atr
    adv = (e - l) / atr
    return (int(t15[jj]), prio, key, s, e, atr, fav, adv, np.maximum.accumulate(fav), np.maximum.accumulate(adv),
            (o - e) / atr, (cc - e) / atr, t15[jj:jj + MAXH], slip, float('nan'), 0)


def pull_confirm(coins, W, delays=None, seed=0):
    rng = np.random.default_rng(seed)
    out, dl, nsig = [], [], 0
    for s in coins:
        c = SP.CTX[s]
        t15, a15, pos = c['t15'], c['a15'], c['pos']
        slip = SLIP.get(s, 0.0003)
        for key, prio, sec in (('o1h', 1, 3600), ('o2h', 2, 7200)):
            for close, atr in pull_signals(s, sec):
                j = pos.get(close)
                if j is None or j < 4 or atr <= 0:
                    continue
                nsig += 1
                if delays is None:
                    k = None
                    for kk in range(W):
                        b = j + kk
                        if b >= len(t15):
                            break
                        if a15[b, 3] > a15[b - 4:b, 1].max():
                            k = kk
                            break
                    if k is None:
                        continue
                    dl.append(k)
                else:
                    if rng.random() >= len(delays) / max(1, delays_n[0]):
                        continue
                    k = int(rng.choice(delays))
                r = entry_row(s, close, key, prio, atr, j + k + 1, c, slip)
                if r is not None:
                    out.append(r)
    return out, dl, nsig


def fix2(rows):
    r = sim_w.simulate(rows, 0.014)
    return r['eq'], abs(L.dd_of(r['curve']))


BOOKS = (('КНИГА 1 (без XLM, AAVE)', [c for c in SP.COINS if c not in ('BILLUSDT', 'XLMUSDT', 'AAVEUSDT')]),
         ('КНИГА 2 (все 15 монет)', [c for c in SP.COINS if c != 'BILLUSDT']))
delays_n = [1]

for bi, (book_name, coins) in enumerate(BOOKS):
    raw = make_sig_raw(coins)
    bank = [x for x in raw if x[2] not in ('o1h', 'o2h')]
    base_rows, _ = book(sorted(raw, key=lambda x: (x[0], x[1])), BASE_KEYS)
    beq, bdd = fix2(base_rows)
    bpy = pyr(base_rows)
    print('', flush=True)
    print('  ===== %s: система $%.0f просадка %.1f%% DD12 $%.0f =====' % (book_name, beq, 100 * bdd, sim_w.money_at_dd(base_rows, 0.12)[1]), flush=True)
    RES = {'система (сейчас)': base_rows}

    def show(lbl, pull):
        sig = sorted(bank + pull, key=lambda x: (x[0], x[1]))
        rows, cnt = book(sig, BASE_KEYS)
        eq, dd = fix2(rows)
        m = sim_w.money_at_dd(rows, 0.12)[1]
        py = pyr(rows)
        pr = [x[2] for x, kk in zip(rows, [None] * len(rows))]
        print('    %-36s откатов в книге %4d | $%6.0f %4.1f%% | DD12 $%6.0f | лучше по годам %d/5 (%s)'
              % (lbl, cnt['o1h'] + cnt['o2h'], eq, 100 * dd, m, sum(1 for a, b in zip(bpy, py) if b > a),
                 ' '.join('%+.0f' % (b - a) for a, b in zip(bpy, py))), flush=True)
        return rows

    for W in (4, 8, 16, 32):
        pull, dl, nsig = pull_confirm(coins, W)
        v = np.array([x[2] for x in book(sorted(pull, key=lambda x: (x[0], x[1])), {'o1h', 'o2h'})[0]])
        print('    окно %2d баров (%.0fч): подтвердилось %d из %d сигналов (%.0f%%), медиана задержки %.0f баров, откат сам по себе ВР %.1f%% ср %+.4f'
              % (W, W / 4, len(dl), nsig, 100 * len(dl) / max(1, nsig), np.median(dl) if dl else 0, 100 * np.mean(v > 0), v.mean()), flush=True)
        RES['подтверждение %dч' % (W // 4)] = show('  подтверждение за %.0fч' % (W / 4), pull)
        delays_n[0] = nsig
        eqs, ms = [], []
        for seed in range(3):
            pc, _, _ = pull_confirm(coins, W, delays=dl, seed=seed)
            sig = sorted(bank + pc, key=lambda x: (x[0], x[1]))
            rows, _ = book(sig, BASE_KEYS)
            eqs.append(fix2(rows)[0]); ms.append(sim_w.money_at_dd(rows, 0.12)[1])
        print('      КОНТРОЛЬ случайная задержка той же доли и длины: $%.0f, DD12 $%.0f (медиана 3)' % (np.median(eqs), np.median(ms)), flush=True)
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
io.open('confirm.py', 'w', encoding='utf-8').write(src)
print('confirm.py готов, синтаксис ок')
