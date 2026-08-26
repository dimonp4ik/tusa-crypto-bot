"""Compare alternative entry models against the SMC filter on identical data.

Three days of tuning have all been INSIDE one strategy. This asks the prior
question: is Smart-Money the best entry model available on this data at all?

Everything except the entry rule is held fixed and matches the shipped bot —
same candles, same stop geometry (swing + ATR buffer, clamped to
RISK_MIN/MAX_PCT), same TP1-arms-a-trail exit, same close-confirmed stop, same
fees and slippage. Only the question "when do we enter, and which way" changes.

Usage:  python tools_strategy_lab.py [--candles N] [--end-date YYYY-MM-DD]
"""
import argparse, os, statistics as st, sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from backtest import BACKTEST_SYMBOLS, fetch_history  # noqa: E402
from config import (ATR_PERIOD, RISK_MIN_PCT, RISK_MAX_PCT, SL_ATR_BUFFER,  # noqa: E402
                    TP1_R_MULT, TP2_R_MULT, TRAIL_ATR_MULT,
                    BACKTEST_FEE_RATE, BACKTEST_SLIPPAGE_RATE)

TF, TF_SEC = "15m", 900


def atr_at(h, l, c, i, n=ATR_PERIOD):
    if i < n + 1:
        return 0.0
    trs = []
    for k in range(i - n + 1, i + 1):
        trs.append(max(h[k] - l[k], abs(h[k] - c[k - 1]), abs(l[k] - c[k - 1])))
    return sum(trs) / len(trs)


def rsi_at(c, i, n=14):
    if i < n + 1:
        return 50.0
    g = s = 0.0
    for k in range(i - n + 1, i + 1):
        d = c[k] - c[k - 1]
        g += max(d, 0.0)
        s += max(-d, 0.0)
    if s == 0:
        return 100.0
    rs = (g / n) / (s / n)
    return 100 - 100 / (1 + rs)


def ema_series(v, n):
    k = 2 / (n + 1)
    out = [v[0]]
    for x in v[1:]:
        out.append(x * k + out[-1] * (1 - k))
    return out


def levels(price, direction, atr, h, l, i):
    """Stop from recent swing + ATR buffer, clamped — mirrors the live geometry."""
    look = 20
    if direction == "LONG":
        sw = min(l[max(0, i - look):i + 1])
        sl = sw - SL_ATR_BUFFER * atr
    else:
        sw = max(h[max(0, i - look):i + 1])
        sl = sw + SL_ATR_BUFFER * atr
    risk = abs(price - sl)
    risk = max(price * RISK_MIN_PCT, min(price * RISK_MAX_PCT, risk))
    sl = price - risk if direction == "LONG" else price + risk
    tp1 = price + risk * TP1_R_MULT if direction == "LONG" else price - risk * TP1_R_MULT
    tp2 = price + risk * TP2_R_MULT if direction == "LONG" else price - risk * TP2_R_MULT
    return sl, tp1, tp2, risk


def run_trade(direction, entry, sl, tp1, tp2, risk, atr, h, l, c, start, horizon=192):
    """TP1 arms a tight trail floored at entry; stop confirmed on close."""
    armed = False
    best = entry
    for j in range(start + 1, min(start + horizon, len(c))):
        if not armed:
            hit_sl = c[j] <= sl if direction == "LONG" else c[j] >= sl
            if hit_sl:
                return (c[j] - entry) / risk if direction == "LONG" else (entry - c[j]) / risk
            if (h[j] >= tp2) if direction == "LONG" else (l[j] <= tp2):
                return (tp2 - entry) / risk if direction == "LONG" else (entry - tp2) / risk
            if (h[j] >= tp1) if direction == "LONG" else (l[j] <= tp1):
                armed = True
                best = h[j] if direction == "LONG" else l[j]
        else:
            if direction == "LONG":
                stop = max(entry, best - TRAIL_ATR_MULT * atr)
                if l[j] <= stop:
                    return (stop - entry) / risk
                best = max(best, h[j])
            else:
                stop = min(entry, best + TRAIL_ATR_MULT * atr)
                if h[j] >= stop:
                    return (entry - stop) / risk
                best = min(best, l[j])
    return 0.0


# ---- entry models -----------------------------------------------------------
def sig_rsi_revert(h, l, c, i):
    r = rsi_at(c, i)
    if r < 30:
        return "LONG"
    if r > 70:
        return "SHORT"
    return None


def sig_donchian(h, l, c, i, n=20):
    if i < n + 1:
        return None
    if c[i] > max(h[i - n:i]):
        return "LONG"
    if c[i] < min(l[i - n:i]):
        return "SHORT"
    return None


def sig_squeeze(h, l, c, i, n=20):
    if i < n * 2 + 2:
        return None
    w = st.pstdev(c[i - n:i]) / (sum(c[i - n:i]) / n)
    wp = st.pstdev(c[i - 2 * n:i - n]) / (sum(c[i - 2 * n:i - n]) / n)
    if wp <= 0 or w > wp * 0.7:
        return None
    return "LONG" if c[i] > c[i - 1] else "SHORT"


def sig_ema_cross(h, l, c, i, fast=9, slow=21, _cache={}):
    key = id(c)
    if key not in _cache:
        _cache.clear()
        _cache[key] = (ema_series(c, fast), ema_series(c, slow))
    ef, es = _cache[key]
    if i < slow + 1:
        return None
    if ef[i - 1] <= es[i - 1] and ef[i] > es[i]:
        return "LONG"
    if ef[i - 1] >= es[i - 1] and ef[i] < es[i]:
        return "SHORT"
    return None


def sig_scalp(h, l, c, i, n=3):
    """Short-horizon continuation: n consecutive closes the same way, enter with it."""
    if i < n + 1:
        return None
    ups = all(c[k] > c[k - 1] for k in range(i - n + 1, i + 1))
    dns = all(c[k] < c[k - 1] for k in range(i - n + 1, i + 1))
    return "LONG" if ups else ("SHORT" if dns else None)


MODELS = {
    "RSI возврат к среднему": sig_rsi_revert,
    "пробой канала 20":       sig_donchian,
    "сжатие волатильности":   sig_squeeze,
    "пересечение EMA 9/21":   sig_ema_cross,
    "скальп: 3 свечи подряд": sig_scalp,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candles", type=int, default=6000)
    ap.add_argument("--end-date")
    ap.add_argument("--cooldown-bars", type=int, default=12)
    a = ap.parse_args()
    end_ms = None
    if a.end_date:
        end_ms = int(datetime.strptime(a.end_date, "%Y-%m-%d")
                     .replace(tzinfo=timezone.utc).timestamp() * 1000)

    data = {}
    for s in BACKTEST_SYMBOLS:
        try:
            d = fetch_history(s, TF, TF_SEC, a.candles, end_date_ms=end_ms)
            if d and len(d.get("close", [])) > 300:
                data[s] = d
        except Exception as e:
            print(f"  {s}: {type(e).__name__}")
    print(f"символов загружено: {len(data)}, свечей на символ ~{a.candles}\n")
    cost = (BACKTEST_FEE_RATE + BACKTEST_SLIPPAGE_RATE) * 2

    print(f"{'модель входа':<26}{'сделок':>8}{'винрейт':>9}{'R/сделку':>10}{'сумма R':>10}")
    for name, fn in MODELS.items():
        rs = []
        for sym, d in data.items():
            h, l, c = d["high"], d["low"], d["close"]
            last = -10 ** 9
            for i in range(60, len(c) - 200):
                if i - last < a.cooldown_bars:
                    continue
                dirn = fn(h, l, c, i)
                if not dirn:
                    continue
                atr = atr_at(h, l, c, i)
                if atr <= 0:
                    continue
                entry = c[i]
                sl, tp1, tp2, risk = levels(entry, dirn, atr, h, l, i)
                if risk <= 0:
                    continue
                r = run_trade(dirn, entry, sl, tp1, tp2, risk, atr, h, l, c, i)
                rs.append(r - cost * entry / risk)
                last = i
        if len(rs) < 30:
            print(f"{name:<26}{len(rs):>8}  — мало")
            continue
        w = sum(1 for x in rs if x > 0)
        print(f"{name:<26}{len(rs):>8}{100*w/len(rs):>8.1f}%{sum(rs)/len(rs):>+10.3f}{sum(rs):>+10.1f}")
    print("\nдля сравнения SMC на текущем окне: ~76.6% винрейта, +0.31R/сделку")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
