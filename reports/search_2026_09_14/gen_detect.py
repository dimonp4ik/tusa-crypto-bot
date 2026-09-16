"""Writes detect.py: after how long can the change be judged at all?

The proposal adds +0.5..0.8 percentage points a month. A single week swings far more than that, so the
owner needs to know when the comparison becomes meaningful - otherwise two bad weeks look like proof of
failure and two good ones like proof of success.

The history is cut into non-overlapping windows of 2, 4, 8, 13 and 26 weeks. In each window both systems
start from $120 at 1.4% risk under the 85% fill (five draws). Reported per book and window length: the
share of windows where the proposal is ahead, the median difference, and the worst and best window - the
honest answer to "how long before I can tell".
"""
import ast
import io

s = io.open('widestopref.py', encoding='utf-8').read()
head = s[:s.index("for bi, (book_name, coins) in enumerate(BOOKS3):")]

TAIL = r'''
WIDE_REF = 0.060
WEEKS = (2, 4, 8, 13, 26)


def boosted(rows, keys, wide_mult=1.25):
    cnt = collections.Counter(x[0] for x in rows)
    out = []
    for x, k in zip(rows, keys):
        w = 1.25 if cnt[x[0]] >= 10 else 1.0
        if k == 'r2_4':
            w *= wide_mult
        out.append(x[:5] + (w,))
    return out


for book_name, coins in BOOKS3:
    H = {s: coin_hours(s) for s in coins}
    base_sig = make_sig_raw(coins)
    extra = roll_entries(coins, H, 'r2_4', 4, 2, 4, 0)
    sig_all = sorted(base_sig + extra, key=lambda x: (x[0], x[1]))
    GEO.clear(); GEO.update(BASE_GEO)
    t0 = min(x[0] for x in sig_all)
    t1 = max(x[0] for x in sig_all)
    print('', flush=True)
    print('  ===== %s: через сколько видно разницу =====' % book_name, flush=True)
    print('    окно      окон  новая впереди  медиана разницы  худшее окно  лучшее окно', flush=True)
    for w in WEEKS:
        span = w * 7 * 86400
        diffs = []
        for seed in range(1, 6):
            rng = np.random.default_rng(seed)
            sub = [x for x in sig_all if rng.random() < 0.85]
            rl, kl = book_keyed(sub, BASE_KEYS)
            rf, kf = book_keyed(sub, FULL)
            live = [x[:5] + (1.0,) for x in rl]
            fin = boosted(rf, kf)
            refs_f = [(WIDE_REF if k == 'r2_4' else sim_w.REF) for k in kf]
            lo = t0
            while lo + span <= t1:
                hi = lo + span
                a = [x for x in live if lo <= x[0] < hi]
                bidx = [i for i, x in enumerate(fin) if lo <= x[0] < hi]
                if len(a) >= 10 and len(bidx) >= 10:
                    ea = sim_ref(a, 0.014, [sim_w.REF] * len(a))['eq']
                    eb = sim_ref([fin[i] for i in bidx], 0.014, [refs_f[i] for i in bidx])['eq']
                    diffs.append(eb / ea - 1)
                lo = hi
        if not diffs:
            continue
        d = np.array(diffs)
        print('    %2d недель %5d %12.0f%% %+14.2f%% %+11.1f%% %+11.1f%%'
              % (w, len(d), 100 * np.mean(d > 0), 100 * np.median(d), 100 * d.min(), 100 * d.max()), flush=True)
'''

src = head + TAIL
ast.parse(src)
io.open('detect.py', 'w', encoding='utf-8').write(src)
print('detect.py готов, синтаксис ок')
