"""The second half of the feature factory: indicators across the whole scale, higher timeframes,
sessions, volume profile, relation to BTC, pressure and time-since-event.

Everything is an array pass over closed bars; higher-timeframe values are mapped onto the hourly
rows using only bars that had FULLY closed by that hour.
"""
import numpy as np

def ema(x, n):
    a = 2 / (n + 1)
    out = np.empty_like(x)
    out[0] = x[0]
    for i in range(1, len(x)):
        out[i] = a * x[i] + (1 - a) * out[i - 1]
    return out


def roll(x, n, fn):
    out = np.full(len(x), np.nan)
    for i in range(len(x)):
        out[i] = fn(x[max(0, i - n + 1):i + 1])
    return out


def rsi(c, n):
    """Wilder's RSI, same recursion the research library used."""
    d = np.diff(c, prepend=c[0])
    up = np.where(d > 0, d, 0.0)
    dn = np.where(d < 0, -d, 0.0)
    a = 1 / n
    ru = np.empty_like(c)
    rd = np.empty_like(c)
    ru[0] = rd[0] = 0.0
    for i in range(1, len(c)):
        ru[i] = ru[i - 1] + a * (up[i] - ru[i - 1])
        rd[i] = rd[i - 1] + a * (dn[i] - rd[i - 1])
    return 100 - 100 / (1 + ru / np.maximum(rd, 1e-12))


def macd(c):
    f, s = ema(c, 12), ema(c, 26)
    line = f - s
    sig = ema(line, 9)
    return line, sig, line - sig


def bollinger(c, n=20, k=2.0):
    m = roll(c, n, np.mean)
    sd = roll(c, n, np.std)
    up, dn = m + k * sd, m - k * sd
    width = (up - dn) / np.maximum(m, 1e-12)
    pos = (c - dn) / np.maximum(up - dn, 1e-12)
    return pos, width


def keltner(c, atr, n=20, k=1.5):
    m = ema(c, n)
    up, dn = m + k * atr, m - k * atr
    return (c - dn) / np.maximum(up - dn, 1e-12)


def obv_slope(c, v, n=24):
    sign = np.sign(np.diff(c, prepend=c[0]))
    obv = np.cumsum(sign * v)
    return np.r_[[np.nan] * n, (obv[n:] - obv[:-n])] / np.maximum(roll(v, n, np.sum), 1e-12)


def vwap_dist(h, l, c, v, atr, n=24):
    tp = (h + l + c) / 3
    num = roll(tp * v, n, np.sum)
    den = np.maximum(roll(v, n, np.sum), 1e-12)
    return (c - num / den) / np.maximum(atr, 1e-12)


def value_area(h, l, c, v, atr, n):
    """Distance to the price level that carried the most volume over the window (POC)."""
    out = np.full(len(c), np.nan)
    for i in range(n, len(c)):
        lo, hi = l[i - n + 1:i + 1].min(), h[i - n + 1:i + 1].max()
        if hi <= lo:
            continue
        edges = np.linspace(lo, hi, 21)
        mid = (h[i - n + 1:i + 1] + l[i - n + 1:i + 1] + c[i - n + 1:i + 1]) / 3
        idx = np.clip(np.searchsorted(edges, mid) - 1, 0, 19)
        prof = np.zeros(20)
        np.add.at(prof, idx, v[i - n + 1:i + 1])
        poc = (edges[prof.argmax()] + edges[prof.argmax() + 1]) / 2
        out[i] = (c[i] - poc) / max(atr[i], 1e-12)
    return out


def divergence(c, ind, n=14):
    """Price makes a new extreme over the window, the indicator does not (and the other way)."""
    out = np.zeros(len(c))
    for i in range(n, len(c)):
        pc, pi = c[i - n:i + 1], ind[i - n:i + 1]
        if c[i] >= pc.max() and ind[i] < pi.max():
            out[i] = -1.0                    # price up, momentum not: bearish divergence
        elif c[i] <= pc.min() and ind[i] > pi.min():
            out[i] = +1.0
    return out


def streak(c):
    """How many bars in a row have closed in the same direction (signed)."""
    d = np.sign(np.diff(c, prepend=c[0]))
    out = np.zeros(len(c))
    for i in range(1, len(c)):
        out[i] = out[i - 1] + d[i] if d[i] == d[i - 1] else d[i]
    return out


def since(flag):
    """Bars since the flag last fired."""
    out = np.full(len(flag), np.nan)
    last = -1
    for i in range(len(flag)):
        if flag[i]:
            last = i
        out[i] = (i - last) if last >= 0 else np.nan
    return out


def map_htf(t_low, t_high, values, bar_sec):
    """Value of the last FULLY closed higher-timeframe bar for every low-timeframe row."""
    j = np.searchsorted(t_high, t_low - bar_sec, 'right') - 1
    ok = j >= 0
    out = np.full(len(t_low), np.nan)
    out[ok] = values[np.clip(j[ok], 0, len(values) - 1)]
    return out


def session_feats(t, o, h, l, c, atr):
    """Asia 00-08, London 08-16, New York 16-24 UTC: range of the session so far and its break."""
    import datetime
    hours = np.array([datetime.datetime.fromtimestamp(int(x), datetime.UTC).hour for x in t])
    sess = np.where(hours < 8, 0, np.where(hours < 16, 1, 2))
    day = (t // 86400).astype(np.int64)
    rng_atr = np.full(len(c), np.nan)
    pos = np.full(len(c), np.nan)
    broke = np.zeros(len(c))
    cur = (-1, -1)
    hi = lo = None
    for i in range(len(c)):
        key = (day[i], sess[i])
        if key != cur:
            cur, hi, lo = key, h[i], l[i]
        else:
            if h[i] > hi:
                broke[i] = 1.0
            if l[i] < lo:
                broke[i] = -1.0
            hi, lo = max(hi, h[i]), min(lo, l[i])
        if atr[i] > 0:
            rng_atr[i] = (hi - lo) / atr[i]
            pos[i] = (c[i] - lo) / max(hi - lo, 1e-12)
    return sess.astype(float), rng_atr, pos, broke


def extras(sym, t1, A1, atr, btc_t, btc_A):
    """Everything in this module for one coin, aligned to its hourly bars."""
    o, h, l, c, v = A1.T
    f = {}
    for n in (2, 7, 14, 21, 50):
        f[f'rsi{n}'] = rsi(c, n)
    line, sig, hist = macd(c)
    f['macd'] = line / np.maximum(atr, 1e-12)
    f['macd_hist'] = hist / np.maximum(atr, 1e-12)
    f['macd_cross'] = np.sign(line - sig)
    f['bb_pos'], f['bb_width'] = bollinger(c)
    f['kelt_pos'] = keltner(c, atr)
    f['obv'] = obv_slope(c, v)
    f['vwap24'] = vwap_dist(h, l, c, v, atr, 24)
    f['vwap168'] = vwap_dist(h, l, c, v, atr, 168)
    f['poc24'] = value_area(h, l, c, v, atr, 24)
    f['poc168'] = value_area(h, l, c, v, atr, 168)
    f['div_rsi'] = divergence(c, rsi(c, 14))
    f['div_macd'] = divergence(c, hist)
    f['streak'] = streak(c)
    f['atr_chg'] = atr / np.maximum(np.r_[[np.nan] * 24, atr[:-24]], 1e-12)
    f['vol_chg'] = v / np.maximum(roll(v, 24, np.mean), 1e-12)
    for n in (6, 24, 72, 168):
        f[f'ret{n}'] = np.r_[[np.nan] * n, c[n:] / c[:-n] - 1]
        f[f'high{n}'] = (roll(h, n, np.max) - c) / np.maximum(atr, 1e-12)
        f[f'low{n}'] = (c - roll(l, n, np.min)) / np.maximum(atr, 1e-12)
    s, rng_atr, pos, broke = session_feats(t1, o, h, l, c, atr)
    f['session'], f['sess_range'], f['sess_pos'], f['sess_break'] = s, rng_atr, pos, broke
    # relation to BTC: rolling correlation, beta and relative strength over 7 days
    bc = map_htf(t1, btc_t, btc_A[:, 3], 3600)
    br = np.r_[np.nan, bc[1:] / bc[:-1] - 1]
    cr = np.r_[np.nan, c[1:] / c[:-1] - 1]
    corr = np.full(len(c), np.nan)
    beta = np.full(len(c), np.nan)
    for i in range(168, len(c)):
        x, y = br[i - 167:i + 1], cr[i - 167:i + 1]
        m = np.isfinite(x) & np.isfinite(y)
        if m.sum() > 50 and np.std(x[m]) > 0:
            corr[i] = np.corrcoef(x[m], y[m])[0, 1]
            beta[i] = np.cov(y[m], x[m])[0, 1] / np.var(x[m])
    f['btc_corr'] = corr
    f['btc_beta'] = beta
    f['rel_str'] = f['ret24'] - np.r_[[np.nan] * 24, bc[24:] / bc[:-24] - 1]
    # round numbers: distance to the nearest 1% level in ATR
    step = 10 ** np.floor(np.log10(np.maximum(c, 1e-9)) - 2)
    f['round_dist'] = np.abs(c - np.round(c / step) * step) / np.maximum(atr, 1e-12)
    return f


NAMES = (['rsi2', 'rsi7', 'rsi14', 'rsi21', 'rsi50', 'macd', 'macd_hist', 'macd_cross',
          'bb_pos', 'bb_width', 'kelt_pos', 'obv', 'vwap24', 'vwap168', 'poc24', 'poc168',
          'div_rsi', 'div_macd', 'streak', 'atr_chg', 'vol_chg']
         + [f'{p}{n}' for n in (6, 24, 72, 168) for p in ('ret', 'high', 'low')]
         + ['session', 'sess_range', 'sess_pos', 'sess_break',
            'btc_corr', 'btc_beta', 'rel_str', 'round_dist'])
