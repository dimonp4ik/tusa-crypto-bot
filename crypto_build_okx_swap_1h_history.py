"""Download reproducible public OKX USDT-swap one-hour history."""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import requests


SYMBOLS = ("BTCUSDT", "ETHUSDT", "XRPUSDT", "XLMUSDT", "LINKUSDT",
           "SOLUSDT", "AVAXUSDT", "AAVEUSDT")
ENDPOINT = "https://www.okx.com/api/v5/market/history-candles"


def fetch(symbol: str, start_ms: int, end_ms: int, out: Path) -> tuple[str, dict]:
    instrument = symbol.removesuffix("USDT") + "-USDT-SWAP"
    after = end_ms + 1
    rows = {}
    session = requests.Session()
    failures = 0
    while True:
        try:
            response = session.get(ENDPOINT, params={
                "instId": instrument, "bar": "1H", "after": after,
                "limit": 300,
            }, timeout=30)
            response.raise_for_status()
            body = response.json()
            if body.get("code") != "0":
                raise RuntimeError(str(body))
            batch = body.get("data", [])
            failures = 0
        except Exception:
            failures += 1
            if failures > 6:
                raise
            time.sleep(min(8, .5 * 2 ** failures))
            continue
        if not batch:
            break
        for row in batch:
            timestamp = int(row[0])
            if start_ms <= timestamp <= end_ms and row[-1] == "1":
                rows[timestamp] = row
        oldest = min(int(row[0]) for row in batch)
        if oldest <= start_ms:
            break
        if oldest >= after:
            raise RuntimeError(f"{symbol}: pagination stalled at {oldest}")
        after = oldest
        time.sleep(.12)
    ordered = sorted(rows)
    candles = {"time": [value // 1000 for value in ordered], **{
        name: [float(rows[value][index]) for value in ordered]
        for index, name in enumerate(("open", "high", "low", "close",
                                      "volume", "volume_ccy",
                                      "volume_quote"), 1)
    }}
    raw = pickle.dumps(candles, protocol=pickle.HIGHEST_PROTOCOL)
    path = out / f"{symbol}_1h_2020_2026.pkl"
    path.write_bytes(raw)
    return symbol, {
        "instrument": instrument, "path": str(path), "bars": len(ordered),
        "start": candles["time"][0] if ordered else None,
        "end": candles["time"][-1] if ordered else None,
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


def parse(value: str) -> int:
    stamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if stamp.tzinfo is None:
        stamp = stamp.replace(tzinfo=timezone.utc)
    return int(stamp.timestamp() * 1000)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default="2020-01-01T00:00:00Z")
    parser.add_argument("--end", default=None)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument(
        "--symbols", nargs="+", default=list(SYMBOLS),
        help="symbols such as BTCUSDT; defaults to the frozen eight-symbol set")
    parser.add_argument("--out", type=Path,
                        default=Path("reports/audit_2026_09_08/crypto_okx_swap_1h_history"))
    args = parser.parse_args()
    start_ms = parse(args.start)
    end_ms = parse(args.end) if args.end else int(time.time() * 1000)
    if start_ms >= end_ms or args.workers < 1:
        parser.error("invalid time range or worker count")
    symbols_requested = tuple(dict.fromkeys(
        str(symbol).strip().upper() for symbol in args.symbols))
    if not symbols_requested or any(
            not symbol.endswith("USDT") for symbol in symbols_requested):
        parser.error("symbols must be non-empty USDT symbols")
    args.out.mkdir(parents=True, exist_ok=True)
    symbols = {}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(fetch, symbol, start_ms, end_ms, args.out): symbol
                   for symbol in symbols_requested}
        for future in as_completed(futures):
            symbol, meta = future.result()
            symbols[symbol] = meta
            print(json.dumps({"symbol": symbol, **meta}), flush=True)
    manifest = {
        "status": "RESEARCH_INPUT", "venue": "OKX global USDT swap public API",
        "interval": "1H", "created_at": datetime.now(timezone.utc).isoformat(),
        "requested_start": start_ms // 1000, "requested_end": end_ms // 1000,
        "symbols": {symbol: symbols[symbol] for symbol in symbols_requested},
    }
    path = args.out / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"manifest": str(path), "symbols": len(symbols)}, indent=2))


if __name__ == "__main__":
    main()
