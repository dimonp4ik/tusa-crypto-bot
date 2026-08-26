"""Download OKX funding-rate history for the backtest symbols.

The live bot skips a LONG when funding is above +0.05% and a SHORT when it is
below -0.05%, on the theory that crowded positioning precedes a squeeze. That
gate has never been measured: funding is one of the nine live gates absent from
the backtest. OKX serves history at 8h granularity, 100 rows per call, so the
whole 18k-candle window is a few paginated calls per symbol.

Writes funding_history.json: {symbol: [[funding_time_sec, rate], ...]} sorted.
"""
import json, os, sys, time
import requests

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from backtest import BACKTEST_SYMBOLS  # noqa: E402

URL = "https://www.okx.com/api/v5/public/funding-rate-history"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "funding_history.json")
PAGES = 8          # 8 x 100 x 8h ~= 266 days, comfortably past an 18k-candle window
PAUSE = 0.35       # OKX public endpoints rate-limit readily


def fetch_symbol(inst: str) -> list:
    rows, before = [], None
    for _ in range(PAGES):
        params = {"instId": inst, "limit": "100"}
        if before:
            params["after"] = str(before)
        try:
            r = requests.get(URL, params=params, timeout=20)
            data = (r.json() or {}).get("data") or []
        except Exception as e:
            print(f"    {inst}: {type(e).__name__} {e}")
            break
        if not data:
            break
        for x in data:
            try:
                rows.append([int(x["fundingTime"]) // 1000, float(x["fundingRate"])])
            except (KeyError, TypeError, ValueError):
                pass
        before = int(data[-1]["fundingTime"])
        time.sleep(PAUSE)
    rows.sort()
    return rows


def main() -> int:
    out = {}
    if os.path.exists(OUT):
        try:
            out = json.load(open(OUT, encoding="utf-8"))
        except Exception:
            out = {}
    for sym in BACKTEST_SYMBOLS:
        if sym in out and len(out[sym]) > 400:
            print(f"  {sym}: уже есть {len(out[sym])}")
            continue
        inst = sym.replace("USDT", "") + "-USDT-SWAP"
        rows = fetch_symbol(inst)
        out[sym] = rows
        span = ""
        if rows:
            import datetime as dt
            a = dt.datetime.fromtimestamp(rows[0][0], dt.UTC).strftime("%Y-%m")
            b = dt.datetime.fromtimestamp(rows[-1][0], dt.UTC).strftime("%Y-%m")
            span = f"  {a} → {b}"
        print(f"  {sym}: {len(rows)} записей{span}")
        json.dump(out, open(OUT, "w", encoding="utf-8"))
    print(f"сохранено в {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
