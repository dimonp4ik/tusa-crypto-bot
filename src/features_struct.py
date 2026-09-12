"""Market structure, computed for EVERY bar at once (no per-event loops).

Definitions follow the bot's own proven code (src/indicators.py) but are written as array passes so
the whole history of a coin costs one sweep instead of 40 000 function calls:

  swings            pivots confirmed by `k` bars on each side - known only k bars later
  HH / HL / LH / LL the last two confirmed highs and lows compared, plus how long the current
                    sequence has held (this is what "structure" actually means)
  BOS / CHoCH       a close beyond the last confirmed swing, and the flip of that direction
  premium/discount  where price sits inside the last swing range (0 = low, 1 = high)
  zones             support / resistance built from swing clusters: distance, touch count, age
  order blocks      the last opposite candle before an impulsive move that broke structure
  imbalance (FVG)   a gap between candle bodies left by an impulse, and whether it is still open
  sweep             a wick beyond a swing that closes back inside (stop run)
  candle geometry   body, wicks, range against ATR, engulfing, speed of the last n bars

Every value at bar i uses bars <= i only, and pivots additionally wait k bars for confirmation, so
nothing here can see the future. The audit in verify_struct.py proves it by truncation.
"""
import numpy as np

K = 3                     # bars each side to confirm a pivot
ZONE_TOL = 0.25           # a touch counts within this many ATR of the level


def pivots(h, l, k=K):
    """Confirmed pivot highs / lows. A pivot at i is only KNOWN at i + k."""
    n = len(h)
    ph = np.zeros(n, bool)
    pl = np.zeros(n, bool)
    for i in range(k, n - k):
        w_h = h[i - k:i + k + 1]
        w_l = l[i - k:i + k + 1]
        if h[i] >= w_h.max():
            ph[i] = True
        if l[i] <= w_l.min():
            pl[i] = True
    known_h = np.zeros(n, bool)
    known_l = np.zeros(n, bool)
    known_h[k:] = ph[:-k] if k else ph
    known_l[k:] = pl[:-k] if k else pl
    return ph, pl, known_h, known_l


def last_two(values, flags):
    """For every bar: the last and the previous value whose flag was already known."""
    n = len(values)
    a = np.full(n, np.nan)
    b = np.full(n, np.nan)
    ai = np.full(n, -1)
    bi = np.full(n, -1)
    cur_i = prev_i = -1
    for i in range(n):
        if flags[i]:
            prev_i, cur_i = cur_i, i - K            # the pivot itself sits K bars back
        a[i] = values[cur_i] if cur_i >= 0 else np.nan
        b[i] = values[prev_i] if prev_i >= 0 else np.nan
        ai[i], bi[i] = cur_i, prev_i
    return a, b, ai, bi


def rolling(x, n, fn):
    out = np.full(len(x), np.nan)
    for i in range(len(x)):
        lo = max(0, i - n + 1)
        out[i] = fn(x[lo:i + 1])
    return out


def structure(t, o, h, l, c, v, atr):
    """The whole structure pack for one coin, one bar per row."""
    n = len(c)
    ph, pl, kh, kl = pivots(h, l)
    hi, hi_prev, hi_i, hi_pi = last_two(h, kh)
    lo, lo_prev, lo_i, lo_pi = last_two(l, kl)

    higher_high = hi > hi_prev
    higher_low = lo > lo_prev
    lower_high = hi < hi_prev
    lower_low = lo < lo_prev
    # +1 uptrend structure (HH & HL), -1 downtrend (LH & LL), 0 mixed
    trend = np.where(higher_high & higher_low, 1.0,
                     np.where(lower_high & lower_low, -1.0, 0.0))
    # how many bars the current structure label has held
    hold = np.zeros(n)
    for i in range(1, n):
        hold[i] = hold[i - 1] + 1 if trend[i] == trend[i - 1] else 0

    rng = np.maximum(hi - lo, 1e-12)
    pos_in_range = (c - lo) / rng                    # 0 = at the swing low, 1 = at the swing high
    bos_up = (c > hi) & np.isfinite(hi)
    bos_dn = (c < lo) & np.isfinite(lo)
    # CHoCH: structure label flips against the previous one
    choch = np.zeros(n)
    for i in range(1, n):
        choch[i] = 1.0 if trend[i] != 0 and trend[i - 1] != 0 and trend[i] != trend[i - 1] else 0.0

    # distance to the nearest untested swing level above / below, in ATR, with touch counts
    d_up = np.full(n, 10.0)
    d_dn = np.full(n, 10.0)
    t_up = np.zeros(n)
    t_dn = np.zeros(n)
    age_up = np.full(n, np.nan)
    age_dn = np.full(n, np.nan)
    hh_idx = [i for i in range(n) if kh[i]]
    ll_idx = [i for i in range(n) if kl[i]]
    for i in range(n):
        a = atr[i]
        if not np.isfinite(a) or a <= 0:
            continue
        up = [h[j - K] for j in hh_idx if j <= i and h[j - K] > c[i]]
        dn = [l[j - K] for j in ll_idx if j <= i and l[j - K] < c[i]]
        if up:
            lvl = min(up)
            d_up[i] = (lvl - c[i]) / a
            t_up[i] = np.sum(np.abs(h[max(0, i - 300):i + 1] - lvl) < ZONE_TOL * a)
            j = int(np.argmin([abs(h[j - K] - lvl) for j in hh_idx if j <= i]))
            age_up[i] = i - hh_idx[j]
        if dn:
            lvl = max(dn)
            d_dn[i] = (c[i] - lvl) / a
            t_dn[i] = np.sum(np.abs(l[max(0, i - 300):i + 1] - lvl) < ZONE_TOL * a)
            j = int(np.argmin([abs(l[j - K] - lvl) for j in ll_idx if j <= i]))
            age_dn[i] = i - ll_idx[j]

    body = (c - o) / np.maximum(h - l, 1e-12)
    up_wick = (h - np.maximum(o, c)) / np.maximum(h - l, 1e-12)
    dn_wick = (np.minimum(o, c) - l) / np.maximum(h - l, 1e-12)
    range_atr = (h - l) / np.maximum(atr, 1e-12)
    body_atr = np.abs(c - o) / np.maximum(atr, 1e-12)
    # engulfing: this body covers the previous one and points the other way
    prev_o, prev_c = np.r_[o[0], o[:-1]], np.r_[c[0], c[:-1]]
    eng_up = ((c > o) & (prev_c < prev_o) & (c >= prev_o) & (o <= prev_c)).astype(float)
    eng_dn = ((c < o) & (prev_c > prev_o) & (c <= prev_o) & (o >= prev_c)).astype(float)
    # speed: how much of the last 6 bars' travel ended up as net movement
    net6 = np.r_[[np.nan] * 6, c[6:] - c[:-6]]
    path6 = rolling(np.abs(np.diff(c, prepend=c[0])), 6, np.sum)
    efficiency = net6 / np.maximum(path6, 1e-12)

    # imbalance (FVG): gap between bar i-2 and bar i bodies, and whether price has filled it
    fvg_up = np.zeros(n)
    fvg_dn = np.zeros(n)
    fvg_up_dist = np.full(n, 10.0)
    fvg_dn_dist = np.full(n, 10.0)
    for i in range(2, n):
        a = atr[i]
        if not np.isfinite(a) or a <= 0:
            continue
        if l[i] > h[i - 2]:                            # gap up
            fvg_up[i] = (l[i] - h[i - 2]) / a
        if h[i] < l[i - 2]:                            # gap down
            fvg_dn[i] = (l[i - 2] - h[i]) / a
    # nearest unfilled gap, looking back 100 bars
    for i in range(n):
        a = atr[i]
        if not np.isfinite(a) or a <= 0:
            continue
        best_u = best_d = None
        for j in range(max(2, i - 100), i + 1):
            if fvg_up[j] > 0 and l[j] > c[i]:
                best_u = min(best_u, l[j]) if best_u else l[j]
            if fvg_dn[j] > 0 and h[j] < c[i]:
                best_d = max(best_d, h[j]) if best_d else h[j]
        if best_u:
            fvg_up_dist[i] = (best_u - c[i]) / a
        if best_d:
            fvg_dn_dist[i] = (c[i] - best_d) / a

    # order block: last opposite-colour bar before an impulse that broke a swing
    ob_up_dist = np.full(n, 10.0)
    ob_dn_dist = np.full(n, 10.0)
    last_ob_up = last_ob_dn = None
    for i in range(1, n):
        a = atr[i]
        if bos_up[i] and body_atr[i] > 1.0:
            for j in range(i - 1, max(0, i - 10), -1):
                if c[j] < o[j]:
                    last_ob_up = (l[j] + h[j]) / 2
                    break
        if bos_dn[i] and body_atr[i] > 1.0:
            for j in range(i - 1, max(0, i - 10), -1):
                if c[j] > o[j]:
                    last_ob_dn = (l[j] + h[j]) / 2
                    break
        if last_ob_up and np.isfinite(a) and a > 0:
            ob_up_dist[i] = (c[i] - last_ob_up) / a
        if last_ob_dn and np.isfinite(a) and a > 0:
            ob_dn_dist[i] = (last_ob_dn - c[i]) / a

    # stop run: wick beyond a swing, close back inside
    sweep_up = ((h > hi) & (c < hi)).astype(float)
    sweep_dn = ((l < lo) & (c > lo)).astype(float)

    return dict(
        trend=trend, hold=hold, hh=higher_high.astype(float), hl=higher_low.astype(float),
        lh=lower_high.astype(float), ll=lower_low.astype(float),
        pos_in_range=pos_in_range, swing_rng_atr=rng / np.maximum(atr, 1e-12),
        bos_up=bos_up.astype(float), bos_dn=bos_dn.astype(float), choch=choch,
        d_up=d_up, d_dn=d_dn, t_up=t_up, t_dn=t_dn, age_up=age_up, age_dn=age_dn,
        body=body, up_wick=up_wick, dn_wick=dn_wick, range_atr=range_atr, body_atr=body_atr,
        eng_up=eng_up, eng_dn=eng_dn, efficiency=efficiency,
        fvg_up=fvg_up, fvg_dn=fvg_dn, fvg_up_dist=fvg_up_dist, fvg_dn_dist=fvg_dn_dist,
        ob_up_dist=ob_up_dist, ob_dn_dist=ob_dn_dist,
        sweep_up=sweep_up, sweep_dn=sweep_dn)


NAMES = ['trend', 'hold', 'hh', 'hl', 'lh', 'll', 'pos_in_range', 'swing_rng_atr',
         'bos_up', 'bos_dn', 'choch', 'd_up', 'd_dn', 't_up', 't_dn', 'age_up', 'age_dn',
         'body', 'up_wick', 'dn_wick', 'range_atr', 'body_atr', 'eng_up', 'eng_dn',
         'efficiency', 'fvg_up', 'fvg_dn', 'fvg_up_dist', 'fvg_dn_dist',
         'ob_up_dist', 'ob_dn_dist', 'sweep_up', 'sweep_dn']
