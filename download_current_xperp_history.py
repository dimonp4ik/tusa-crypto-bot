"""Download a fresh public 15-minute X-Perp window without API keys/orders."""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import time
from datetime import datetime, timezone
from pathlib import Path

import requests


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbols", required=True)
    parser.add_argument("--bars", type=int, default=18000)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    response = requests.get(
        "https://www.okx.com/api/v5/public/instruments",
        params={"instType": "FUTURES"}, timeout=25)
    response.raise_for_status()
    body = response.json()
    if body.get("code") != "0":
        raise RuntimeError(str(body))
    instruments = body.get("data", [])
    mapping = {item["instId"].split("-")[0] + "USDT": item["instId"]
               for item in instruments
               if "_UM_XPERP-" in item.get("instId", "") and item.get("state") == "live"}
    args.out.mkdir(parents=True, exist_ok=True)
    # Freeze one pagination anchor for the whole cross-sectional snapshot.  If
    # a 15-minute candle closes while sequential symbol downloads are running,
    # recomputing this per symbol produces mismatched timelines.
    request_anchor_ms = int(time.time() * 1000) + 60_000
    report = {"status": "PUBLIC_CURRENT_VENUE_DATA", "bars_requested": args.bars,
              "created_at": datetime.now(timezone.utc).isoformat(),
              "request_anchor_ms": request_anchor_ms, "symbols": {}}
    for symbol in (value.strip() for value in args.symbols.split(",") if value.strip()):
        instrument = mapping.get(symbol)
        if not instrument:
            report["symbols"][symbol] = {"status": "NO_INSTRUMENT"}
            continue
        after = request_anchor_ms
        received = {}
        pages = (args.bars + 299) // 300
        for _ in range(pages):
            response = requests.get(
                "https://www.okx.com/api/v5/market/history-candles",
                params={"instId": instrument, "bar": "15m", "after": after, "limit": 300},
                timeout=25,
            )
            response.raise_for_status()
            body = response.json()
            if body.get("code") != "0":
                raise RuntimeError(str(body))
            batch = body.get("data", [])
            if not batch:
                break
            for row in batch:
                if row[-1] == "1":
                    received[int(row[0]) // 1000] = row
            oldest = min(int(row[0]) for row in batch)
            if oldest >= after:
                raise RuntimeError(f"{symbol}: pagination stalled")
            after = oldest
            if len(received) >= args.bars:
                break
            time.sleep(.16)
        times = sorted(received)[-args.bars:]
        candles = {"time": times, **{
            key: [float(received[timestamp][index]) for timestamp in times]
            for index, key in enumerate(("open", "high", "low", "close", "volume"), 1)
        }}
        raw = pickle.dumps(candles)
        path = args.out / f"{symbol}_15min_current.pkl"
        path.write_bytes(raw)
        meta = {
            "status": "OK" if len(times) >= 1500 else "SHORT_HISTORY",
            "instrument": instrument, "bars": len(times),
            "start": times[0] if times else None, "end": times[-1] if times else None,
            "sha256": hashlib.sha256(raw).hexdigest(), "path": str(path),
        }
        report["symbols"][symbol] = meta
        print(json.dumps({"symbol": symbol, **meta}), flush=True)
    manifest = args.out / "manifest.json"
    manifest.write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
