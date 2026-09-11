"""High win-rate pullback bank (research 11.09.2026, see reports/NIGHT_2026_09_11.md).

Regime-conditional: a coin is eligible only while the live 4h trend strategy
(src/trend4h.py, BTC gate on) holds a position on it. LONG rules act only while the trend
strategy is LONG, SHORT rules only while it is SHORT; flat -> nothing (no edge there).
Narrow filters ("rules") look for their own scenario on the CLOSED 1h bar. The first
matching rule (bank order) sets take-profit / stop in ATR14(1h) units. Shorts mirror longs.

Execution (the edge lives here - the same filters with a market entry are ~0):
  * LIMIT at the signal bar's close, alive for the first 15m bar of the next hour.
    - that bar opens through the limit -> immediate (taker) fill at the open, 4 bps slip,
      cost 4 bps round trip
    - it trades through the limit -> passive (maker) fill at the limit, cost 2 bps;
      no take-profit credit inside that bar (its extreme may precede the fill)
    - otherwise the order is missed (quick bounces are lost - adverse selection kept)
  * take-profit is a resting limit: filled only if price trades THROUGH it
  * stop is a market stop: gap-through filled at the bar open
  * time exit at the close of the 192nd 15m bar (48h)
One position per coin. Returns are fractions of entry price, costs included;
R = return / stop distance.

`advance()` is the live code path and `simulate()` replays history through it, so the
backtest and the paper runner cannot disagree. Every input is a CLOSED bar.
"""
from __future__ import annotations

import bisect

import numpy as np

from src import trend4h as T4

BAR = 900
HOUR = 3600
HOLD_BARS = 192
MIN_HOURS = 720            # warm-up: volatility regime uses a 720-hour median
TAKER_COST = 0.0004
TAKER_SLIP = 0.0004
PASSIVE_COST = 0.0002

# Mined inside the trend4h-LONG regime on 18 pinned coins < 2025, confirmed on 22 other
# coins < 2025 and in each of 2022/2023/2024 separately; 2025-2026 never used for choice.
# (feature, op, value); ("hour", "in", (a, b)) means a <= UTC hour of bar start <= b.
RULES = [
    dict(name="btc_pump_evening", tp=1.0, sl=4.0, conds=[("btc24", ">=", 0.04896), ("hour", "in", (20, 23))]),
    dict(name="high_vol_regime", tp=0.5, sl=3.0, conds=[("volreg", ">=", 1.887)]),
    dict(name="coin_run_evening", tp=0.5, sl=3.0, conds=[("ret24", ">=", 3.864), ("hour", "in", (20, 23))]),
    dict(name="btc_pump_night", tp=0.5, sl=3.0, conds=[("btc24", ">=", 0.04896), ("hour", "in", (0, 3))]),
    dict(name="btc_pump_red_candle", tp=0.5, sl=3.0, conds=[("btc24", ">=", 0.04896), ("body", "<=", -0.5)]),
    dict(name="btc_dip_coin_flat", tp=0.5, sl=3.0, conds=[("ret24", "<=", -0.2091), ("btc24", "<=", -0.02043)]),
    dict(name="rsi2_flush_afternoon", tp=0.5, sl=3.0, conds=[("rsi2", "<=", 5.475), ("hour", "in", (16, 19))]),
    dict(name="steep_trend_range_low", tp=0.5, sl=3.0, conds=[("slope", ">=", 1.542), ("rng", "<=", 0.06976)]),
    dict(name="coin_dip_night", tp=0.5, sl=3.0, conds=[("ret24", "<=", -2.1), ("hour", "in", (0, 3))]),
]

# Stricter mining (rule must hold in each HALF-year 2022-2024): three rules, all also in
# the bank above (the first with a smaller take/stop). Fewer trades, better per trade.
RULES_STRICT = [
    dict(name="btc_pump_evening", tp=0.5, sl=3.0, conds=[("btc24", ">=", 0.04896), ("hour", "in", (20, 23))]),
    dict(name="coin_run_evening", tp=0.5, sl=3.0, conds=[("ret24", ">=", 3.864), ("hour", "in", (20, 23))]),
    dict(name="btc_pump_night", tp=0.5, sl=3.0, conds=[("btc24", ">=", 0.04896), ("hour", "in", (0, 3))]),
]
# Mined inside the trend4h-SHORT regime with the same half-year stability test.
# Out of sample weaker than the longs (18 coins 2025-26 +11R, 22 other coins +20R).
SHORT_RULES = [
    dict(name="short_btc_up_morning", side="SHORT", tp=0.5, sl=3.0,
         conds=[("btc24", ">=", 0.01701), ("hour", "in", (8, 11))]),
    dict(name="short_pop_in_downtrend", side="SHORT", tp=0.5, sl=3.0,
         conds=[("rsi2", ">=", 71.02), ("rsi14", "<=", 35.22)]),
]
RULE_SETS = {"bank9": RULES, "strict3": RULES_STRICT, "short2": SHORT_RULES,
             "strict3+short2": RULES_STRICT + SHORT_RULES, "bank9+short2": RULES + SHORT_RULES}
# Same five rules with a wider take (0.75 ATR, stop 3 ATR). Chosen on 2022-24 with the owner's
# win-rate floor of 75%; more profit AND a smaller drawdown in both 2022-24 and 2025-26
# (18 coins, btc_sma=50 + stop sizing, margin 3%: +4.0%/+4.3% a month vs +3.2%/+3.1%,
# drawdown -9.1%/-10.3% vs -9.9%/-10.5%; win rate ~85%). reports/NIGHT_2026_09_11.md.
RULE_SETS["strict3+short2_wide"] = [dict(r, tp=0.75) for r in RULES_STRICT + SHORT_RULES]


# ---------------------------------------------------------------- bars and indicators
def build_bars(t15, a15, sec):
    """Aggregate CLOSED 15m bars (seconds, rows o,h,l,c[,v]) into complete, contiguous
    `sec` bars. Returns (keys int64, rows o,h,l,c,v)."""
    t15 = np.asarray(t15, dtype=np.int64)
    a15 = np.asarray(a15, dtype=float)
    if a15.shape[1] == 4:
        a15 = np.c_[a15, np.zeros(len(a15))]
    m = sec // BAR
    if len(t15) == 0:
        return np.zeros(0, np.int64), np.zeros((0, 5))
    key = t15 - t15 % sec
    idx = np.r_[0, np.nonzero(np.diff(key))[0] + 1, len(t15)]
    kt, rows = [], []
    for a, b in zip(idx[:-1], idx[1:]):
        if b - a != m or t15[b - 1] - t15[a] != (m - 1) * BAR:
            continue
        kt.append(key[a])
        rows.append((a15[a, 0], a15[a:b, 1].max(), a15[a:b, 2].min(), a15[b - 1, 3], a15[a:b, 4].sum()))
    return np.array(kt, dtype=np.int64), np.array(rows, dtype=float).reshape(-1, 5)


def _sma(x, n):
    o = np.full(len(x), np.nan)
    if len(x) >= n:
        c = np.cumsum(np.r_[0, x]); o[n - 1:] = (c[n:] - c[:-n]) / n
    return o


def _ema(x, n):
    e = np.empty_like(x)
    if len(x):
        a = 2 / (n + 1); e[0] = x[0]
        for i in range(1, len(x)):
            e[i] = a * x[i] + (1 - a) * e[i - 1]
    return e


def _rsi(c, n):
    d = np.diff(c, prepend=c[0]) if len(c) else c
    up = np.where(d > 0, d, 0.0); dn = np.where(d < 0, -d, 0.0)
    a = 1 / n; ru = np.zeros_like(c); rd = np.zeros_like(c)
    for i in range(1, len(c)):
        ru[i] = ru[i - 1] + a * (up[i] - ru[i - 1]); rd[i] = rd[i - 1] + a * (dn[i] - rd[i - 1])
    return 100 - 100 / (1 + ru / np.maximum(rd, 1e-12))


def _atr(B, n=14):
    h, l, c = B[:, 1], B[:, 2], B[:, 3]
    pc = np.r_[c[0], c[:-1]] if len(c) else c
    tr = np.maximum(h - l, np.maximum(abs(h - pc), abs(l - pc)))
    return _sma(tr, n)


def _lag(x, k):
    return np.r_[np.full(k, np.nan), x[:-k]] if len(x) > k else np.full(len(x), np.nan)


def _roll(x, w, fn):
    o = np.full(len(x), np.nan)
    if len(x) >= w:
        o[w - 1:] = fn(np.lib.stride_tricks.sliding_window_view(x, w), 1)
    return o


def features(T1, B1, btc_t15, btc_a15):
    """Causal features at the CLOSE of every 1h bar (all read bars <= i)."""
    o, h, l, c = B1[:, 0], B1[:, 1], B1[:, 2], B1[:, 3]
    at = _atr(B1); e200 = _ema(c, 200)
    hi20 = _lag(_roll(h, 20, np.max), 1); lo20 = _lag(_roll(l, 20, np.min), 1)
    atrp = at / c
    atrmed = _roll(atrp, MIN_HOURS, np.median)
    bT1, bB1 = build_bars(btc_t15, btc_a15, HOUR)
    bmap = dict(zip(bT1.tolist(), bB1[:, 3]))
    bc = np.array([bmap.get(int(t), np.nan) for t in T1])
    bc24 = np.array([bmap.get(int(t) - 86400, np.nan) for t in T1])
    return dict(
        atr=at,
        rsi2=_rsi(c, 2),
        rsi14=_rsi(c, 14),
        slope=(e200 - _lag(e200, 24)) / at,
        ret24=(c - _lag(c, 24)) / at,
        rng=(c - lo20) / (hi20 - lo20 + 1e-12),
        volreg=atrp / atrmed,
        btc24=bc / bc24 - 1,
        body=(c - o) / (h - l + 1e-12),
        hour=(T1 // HOUR) % 24,
    )


def btc_daily(btc_t15, btc_a15):
    Td, Bd = build_bars(btc_t15, btc_a15, 86400)
    return [int(t) for t in Td], Bd[:, 3]


OPEN_END = 2 ** 62


def regime_update(state, t15, a15, btc_times, btc_closes):
    """Advance the trend4h regime over every CLOSED 4h bar not yet seen (live path; the
    batch trend_regime() below is this with an empty state). Returns the interval list
    [[open_ts, exit_ts, side]]: at time x the live strategy holds `side` iff
    open_ts <= x < exit_ts (the exit is learnt when the exit 4h bar CLOSES; still-open ->
    OPEN_END). One position per coin, so intervals never overlap.
    state (mutated): {"t4": trend4h state, "iv": intervals}."""
    T4b, B4 = build_bars(t15, a15, T4.BAR_SEC)
    t4 = state.setdefault("t4", {})
    iv = state.setdefault("iv", [])
    for e in T4.advance(t4, T4b, B4[:, :4], btc_times, np.asarray(btc_closes, float), gate=True):
        if e[0] == "OPEN":
            iv.append([e[1], OPEN_END, e[2]])
        elif e[0] == "EXIT" and iv and iv[-1][1] == OPEN_END:
            iv[-1][1] = e[1]
    # A trend signal on the last closed 4h bar is filled at the open of the next bar, i.e.
    # at t4["last"] + BAR_SEC - the live strategy holds it from then on, although its OPEN
    # event only appears once that bar has closed. Returned, not stored.
    pend = t4.get("pending")
    if pend and t4.get("pos") is None and t4.get("last") is not None and \
            not (iv and iv[-1][1] == OPEN_END):
        return iv + [[int(t4["last"]) + T4.BAR_SEC, OPEN_END, pend[0]]]
    return list(iv)


def trend_regime(t15, a15, btc_times, btc_closes):
    """Batch form of regime_update() over the whole history."""
    return regime_update({}, t15, a15, btc_times, btc_closes)


def _match(rule, F, i):
    for f, op, v in rule["conds"]:
        x = F[f][i]
        if op == "in":
            if not (v[0] <= x <= v[1]):
                return False
        elif not (x >= v if op == ">=" else x <= v):   # NaN -> False
            return False
    return True


def hourly_signals(t15, a15, btc_t15, btc_a15, rules=RULES, regime=None, btc_sma=0):
    """{hour_close_ts: (rule_index, limit, atr)} for every closed 1h bar that would place
    an order if the coin were flat. Pure function of closed bars. `regime`: intervals from
    regime_update() (live, persisted); None = recompute from these bars (backtest).
    `btc_sma` > 0: LONG rules act only while BTC's last CLOSED daily close is at or above
    its SMA(btc_sma) - longs into a falling BTC were the whole Feb-Apr 2025 drawdown
    (reports/NIGHT_2026_09_11.md). With too little BTC history the filter stays open."""
    T1, B1 = build_bars(t15, a15, HOUR)
    if len(T1) <= MIN_HOURS:
        return {}
    F = features(T1, B1, btc_t15, btc_a15)
    if regime is None:
        bt, bc = btc_daily(btc_t15, btc_a15)
        regime = trend_regime(t15, a15, bt, bc)
    iv = regime
    starts = [x[0] for x in iv]
    btc_down = lambda close: False
    if btc_sma:
        bdt, bdc = btc_daily(btc_t15, btc_a15)
        bdc = np.asarray(bdc, dtype=float)
        bsm = _sma(bdc, btc_sma)

        def btc_down(close):
            d = bisect.bisect_right(bdt, close - 86400) - 1      # last closed daily bar
            return d >= btc_sma - 1 and bdc[d] < bsm[d]
    sig = {}
    for i in range(MIN_HOURS, len(T1)):
        close = int(T1[i]) + HOUR
        j = bisect.bisect_right(starts, close) - 1
        if j < 0 or not (iv[j][0] <= close < iv[j][1]) or np.isnan(F["atr"][i]):
            continue
        side = iv[j][2]
        if side == "LONG" and btc_down(close):
            continue
        for k, r in enumerate(rules):
            if r.get("side", "LONG") == side and _match(r, F, i):
                sig[close] = (k, float(B1[i, 3]), float(F["atr"][i]))
                break
    return sig


# ---------------------------------------------------------------- execution (live path)
def advance(state, t15, a15, signals, rules=RULES):
    """Process every CLOSED 15m bar not yet seen. `signals` from hourly_signals().

    state (mutated): {"last": ts of last processed 15m bar, "pos": dict|None,
                      "pending": dict|None}
    events: ("ORDER", ts, rule, limit), ("MISS", ts, rule),
            ("OPEN", ts, rule, entry, tp, sl, kind, side),
            ("EXIT", ts_bar_end, rule, reason, ret, R, open_ts)
    """
    t15 = np.asarray(t15, dtype=np.int64)
    ev = []
    last = state.get("last")
    start = 0 if last is None else int(np.searchsorted(t15, last, "right"))
    for k in range(start, len(t15)):
        tk = int(t15[k]); o, h, l, c = (float(x) for x in a15[k][:4])
        state["last"] = tk
        fresh = False
        pend = state.get("pending")
        if pend is not None:
            state["pending"] = None
            r = rules[pend["rule"]]; lim = pend["limit"]
            lg = r.get("side", "LONG") == "LONG"
            if tk == pend["at"] and (o <= lim if lg else o >= lim):
                e, cost, passive = o * (1 + TAKER_SLIP if lg else 1 - TAKER_SLIP), TAKER_COST, False
            elif tk == pend["at"] and (l < lim if lg else h > lim):
                e, cost, passive = lim, PASSIVE_COST, True
            else:
                e = None
                ev.append(("MISS", tk, r["name"]))
            if e is not None:
                a = pend["atr"]
                sgn = 1 if lg else -1
                state["pos"] = dict(rule=pend["rule"], side="LONG" if lg else "SHORT", entry=e,
                                    tp=e + sgn * r["tp"] * a, sl=e - sgn * r["sl"] * a,
                                    tp_ret=r["tp"] * a / e, risk=r["sl"] * a / e, cost=cost,
                                    open_ts=tk, deadline=tk + (HOLD_BARS - 1) * BAR)
                fresh, fresh_passive = True, passive
                ev.append(("OPEN", tk, r["name"], e, state["pos"]["tp"], state["pos"]["sl"],
                           "maker" if passive else "taker", state["pos"]["side"]))
        p = state.get("pos")
        if p is not None:
            reason = None
            lg = p.get("side", "LONG") == "LONG"
            if (l <= p["sl"]) if lg else (h >= p["sl"]):
                f = min(p["sl"], o) if lg else max(p["sl"], o)
                ret = f / p["entry"] - 1 if lg else 1 - f / p["entry"]; reason = "stop"
            elif ((h > p["tp"]) if lg else (l < p["tp"])) and not (fresh and fresh_passive):
                ret = p["tp_ret"]; reason = "take"
            elif tk >= p["deadline"]:
                ret = c / p["entry"] - 1 if lg else 1 - c / p["entry"]; reason = "time"
            if reason:
                ret -= p["cost"]
                ev.append(("EXIT", tk + BAR, rules[p["rule"]]["name"], reason, ret, ret / p["risk"], p["open_ts"]))
                state["pos"] = None
        close = tk + BAR
        if close % HOUR == 0 and state.get("pos") is None and state.get("pending") is None:
            s = signals.get(close)
            if s is not None:
                state["pending"] = dict(at=close, rule=s[0], limit=s[1], atr=s[2])
                ev.append(("ORDER", close, rules[s[0]]["name"], s[1]))
    return ev


def simulate(t15, a15, btc_t15, btc_a15, rules=RULES, btc_sma=0):
    """Backtest = the live path over the whole history. Returns closed trades
    (signal_close_ts, exit_ts, ret, risk, rule_name)."""
    sig = hourly_signals(t15, a15, btc_t15, btc_a15, rules, btc_sma=btc_sma)
    st = {}
    out, orders = [], {}
    for e in advance(st, t15, a15, sig, rules):
        if e[0] == "ORDER":
            last_order = e[1]
        elif e[0] == "OPEN":
            orders[e[1]] = last_order
        elif e[0] == "EXIT":
            ret, R = e[4], e[5]
            out.append((orders[e[6]], e[1], ret, ret / R if R else float("nan"), e[2]))
    return out
