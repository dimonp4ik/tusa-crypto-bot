"""Download immutable hourly BTC/ETH Deribit volatility-index history."""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


ENDPOINT = "https://www.deribit.com/api/v2/public/get_volatility_index_data"


def timestamp(value: str) -> int:
    return int(datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp())


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fetch(currency: str, start: int, end: int, chunk_days: int,
          pause: float) -> tuple[dict, list[dict]]:
    values: dict[int, tuple[float, float, float, float]] = {}
    requests = []
    cursor = start
    chunk = chunk_days * 86400
    while cursor < end:
        boundary = min(end, cursor + chunk)
        query = urllib.parse.urlencode({
            "currency": currency,
            "start_timestamp": cursor * 1000,
            "end_timestamp": boundary * 1000,
            "resolution": 3600,
        })
        url = f"{ENDPOINT}?{query}"
        with urllib.request.urlopen(url, timeout=30) as response:
            payload = response.read()
        body = json.loads(payload)
        if body.get("error"):
            raise ValueError(f"Deribit error for {currency}: {body['error']}")
        result = body.get("result", {})
        rows = result.get("data", [])
        if result.get("continuation") is not None:
            raise ValueError(
                f"chunk unexpectedly truncated: {currency} {cursor} {boundary}")
        for row in rows:
            point = int(row[0]) // 1000
            if point < start or point >= end:
                continue
            ohlc = tuple(float(value) for value in row[1:5])
            old = values.get(point)
            if old is not None and old != ohlc:
                raise ValueError(f"overlap revision: {currency} {point}")
            values[point] = ohlc
        requests.append({
            "start": cursor, "end": boundary, "rows": len(rows),
            "payload_sha256": hashlib.sha256(payload).hexdigest(),
        })
        cursor = boundary
        if pause:
            time.sleep(pause)
    times = sorted(values)
    candles = {
        "time": times,
        "open": [values[value][0] for value in times],
        "high": [values[value][1] for value in times],
        "low": [values[value][2] for value in times],
        "close": [values[value][3] for value in times],
    }
    return candles, requests


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--currencies", default="BTC,ETH")
    parser.add_argument("--start", default="2021-04-01T00:00:00Z")
    parser.add_argument("--end", required=True)
    parser.add_argument("--chunk-days", type=int, default=35)
    parser.add_argument("--pause", type=float, default=.05)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    start, end = timestamp(args.start), timestamp(args.end)
    if start >= end or args.chunk_days < 1 or args.chunk_days > 40:
        parser.error("require start < end and chunk-days in 1..40")
    currencies = tuple(dict.fromkeys(
        value.strip().upper() for value in args.currencies.split(",")
        if value.strip()))
    args.out.mkdir(parents=True, exist_ok=True)
    report = {
        "status": "IMMUTABLE_PUBLIC_RESEARCH_INPUT",
        "venue": "Deribit", "source": ENDPOINT, "interval": "1H",
        "requested_start": start, "requested_end_exclusive": end,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "currencies": {},
    }
    for currency in currencies:
        candles, requests = fetch(
            currency, start, end, args.chunk_days, args.pause)
        times = candles["time"]
        if not times:
            raise ValueError(f"no DVOL rows for {currency}")
        gaps = [right - left for left, right in zip(times, times[1:])
                if right - left != 3600]
        path = args.out / f"{currency}_dvol_1h.pkl"
        path.write_bytes(pickle.dumps(candles, protocol=pickle.HIGHEST_PROTOCOL))
        report["currencies"][currency] = {
            "path": str(path), "rows": len(times),
            "start": times[0], "end": times[-1], "sha256": digest(path),
            "requests": len(requests),
            "request_payload_hashes": [row["payload_sha256"] for row in requests],
            "gaps_over_1h": len(gaps),
            "maximum_gap_seconds": max(gaps, default=0),
        }
        print(json.dumps({"currency": currency,
                          **report["currencies"][currency]}), flush=True)
    manifest = args.out / "manifest.json"
    manifest.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"manifest": str(manifest),
                      "sha256": digest(manifest)}), flush=True)


if __name__ == "__main__":
    main()
