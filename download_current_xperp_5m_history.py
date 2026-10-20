"""Download a frozen public X-Perp candle window without keys or orders."""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import time
from datetime import datetime, timezone
from pathlib import Path

import requests


BAR_SECONDS = {"5m": 300, "15m": 900}


def snapshot_anchor_ms(bar: str, now_seconds: int | None = None) -> int:
    """Return one synchronized open-time boundary for the whole snapshot."""
    if bar not in BAR_SECONDS:
        raise ValueError(f"unsupported bar: {bar}")
    current = int(time.time()) if now_seconds is None else int(now_seconds)
    seconds = BAR_SECONDS[bar]
    return (current // seconds) * seconds * 1000


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbols", required=True)
    parser.add_argument("--bars", type=int, default=18000)
    parser.add_argument("--bar", choices=("5m", "15m"), default="5m")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    response = requests.get("https://www.okx.com/api/v5/public/instruments",
        params={"instType": "FUTURES"}, timeout=25)
    response.raise_for_status()
    body = response.json()
    if body.get("code") != "0":
        raise RuntimeError(str(body))
    mapping = {item["instId"].split("-")[0]+"USDT": item["instId"]
               for item in body.get("data", [])
               if "_UM_XPERP-" in item.get("instId", "") and item.get("state") == "live"}
    args.out.mkdir(parents=True, exist_ok=True)
    anchor_ms = snapshot_anchor_ms(args.bar)
    report = {"status": "PUBLIC_CURRENT_VENUE_DATA", "bar": args.bar,
              "bars_requested": args.bars,
              "created_at": datetime.now(timezone.utc).isoformat(),
              "request_anchor_ms": anchor_ms, "symbols": {}}
    for symbol in (x.strip() for x in args.symbols.split(",") if x.strip()):
        instrument = mapping.get(symbol)
        if not instrument:
            report["symbols"][symbol] = {"status": "NO_INSTRUMENT"}
            continue
        after = anchor_ms
        received = {}
        for _ in range((args.bars+299)//300):
            response = requests.get("https://www.okx.com/api/v5/market/history-candles",
                params={"instId": instrument, "bar": args.bar,
                        "after": after, "limit": 300},
                timeout=25)
            response.raise_for_status()
            body = response.json()
            if body.get("code") != "0":
                raise RuntimeError(str(body))
            batch = body.get("data", [])
            if not batch:
                break
            for row in batch:
                if row[-1] == "1":
                    received[int(row[0])//1000] = row
            oldest = min(int(row[0]) for row in batch)
            if oldest >= after:
                raise RuntimeError(f"{symbol}: pagination stalled")
            after = oldest
            if len(received) >= args.bars:
                break
            time.sleep(.16)
        times = sorted(received)[-args.bars:]
        candles = {"time": times, **{
            key: [float(received[t][index]) for t in times]
            for index, key in enumerate(("open", "high", "low", "close", "volume"), 1)}}
        raw = pickle.dumps(candles)
        suffix = "5min" if args.bar == "5m" else "15min"
        path = args.out/f"{symbol}_{suffix}_current.pkl"
        path.write_bytes(raw)
        meta = {"status": "OK" if len(times) >= min(1500, args.bars) else "SHORT_HISTORY",
                "instrument": instrument, "bars": len(times),
                "start": times[0] if times else None, "end": times[-1] if times else None,
                "sha256": hashlib.sha256(raw).hexdigest(), "path": str(path)}
        report["symbols"][symbol] = meta
        print(json.dumps({"symbol": symbol, **meta}), flush=True)
    (args.out/"manifest.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
