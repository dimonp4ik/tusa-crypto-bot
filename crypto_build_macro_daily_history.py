"""Download hashed public daily cross-asset history for crypto research."""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


ENDPOINT = "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
ASSETS = {
    "qqq": "QQQ",
    "vix": "^VIX",
    "dxy": "DX-Y.NYB",
    "gold": "GC=F",
    "us10y": "^TNX",
}


def timestamp(value: str) -> int:
    return int(datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp())


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fetch(symbol: str, start: int, end: int) -> tuple[dict, bytes, dict]:
    query = urllib.parse.urlencode({
        "period1": start, "period2": end, "interval": "1d",
        "events": "history", "includeAdjustedClose": "true",
    })
    url = ENDPOINT.format(symbol=urllib.parse.quote(symbol, safe="")) + "?" + query
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request, timeout=30) as response:
        payload = response.read()
    body = json.loads(payload)
    if body.get("chart", {}).get("error"):
        raise ValueError(f"Yahoo chart error for {symbol}: {body['chart']['error']}")
    result = body["chart"]["result"][0]
    quote = result["indicators"]["quote"][0]
    adjusted = result["indicators"].get("adjclose", [{}])[0].get("adjclose", [])
    rows = []
    for index, stamp in enumerate(result.get("timestamp", [])):
        values = {name: quote.get(name, [None] * (index + 1))[index]
                  for name in ("open", "high", "low", "close", "volume")}
        if any(values[name] is None for name in ("open", "high", "low", "close")):
            continue
        rows.append({
            "time": int(stamp),
            **{name: float(value) if value is not None else None
               for name, value in values.items()},
            "adjusted_close": (float(adjusted[index])
                               if index < len(adjusted) and adjusted[index] is not None
                               else float(values["close"])),
        })
    meta = {
        "exchange_timezone": result.get("meta", {}).get("exchangeTimezoneName"),
        "instrument_type": result.get("meta", {}).get("instrumentType"),
        "currency": result.get("meta", {}).get("currency"),
    }
    return {key: [row[key] for row in rows] for key in rows[0]}, payload, meta


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default="2020-01-01T00:00:00Z")
    parser.add_argument("--end", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    start, end = timestamp(args.start), timestamp(args.end)
    if start >= end:
        parser.error("require start < end")
    args.out.mkdir(parents=True, exist_ok=True)
    manifest = {
        "status": "IMMUTABLE_PUBLIC_RESEARCH_INPUT",
        "provider": "Yahoo Finance chart API", "interval": "1d",
        "requested_start": start, "requested_end_exclusive": end,
        "created_at": datetime.now(timezone.utc).isoformat(), "assets": {},
    }
    for name, symbol in ASSETS.items():
        data, payload, meta = fetch(symbol, start, end)
        path = args.out / f"{name}_daily.pkl"
        path.write_bytes(pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL))
        times = data["time"]
        manifest["assets"][name] = {
            "symbol": symbol, "path": str(path), "rows": len(times),
            "start": times[0], "end": times[-1], "sha256": digest(path),
            "payload_sha256": hashlib.sha256(payload).hexdigest(), **meta,
        }
        print(json.dumps({"asset": name, **manifest["assets"][name]}), flush=True)
    path = args.out / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"manifest": str(path), "sha256": digest(path)}), flush=True)


if __name__ == "__main__":
    main()
