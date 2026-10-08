"""Append confirmed OKX one-hour candles into a new immutable snapshot."""
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


ENDPOINT = "https://www.okx.com/api/v5/market/history-candles"
FIELDS = ("open", "high", "low", "close", "volume",
          "volume_ccy", "volume_quote")


def digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def resolved(value: str, manifest: Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    cwd_path = Path.cwd() / path
    return cwd_path if cwd_path.exists() else manifest.parent / path


def fetch_latest(instrument: str, attempts: int = 6) -> list[list[str]]:
    for attempt in range(attempts):
        try:
            response = requests.get(ENDPOINT, params={
                "instId": instrument, "bar": "1H", "limit": 300,
            }, timeout=30)
            response.raise_for_status()
            body = response.json()
            if body.get("code") != "0":
                raise RuntimeError(str(body))
            return list(body.get("data", []))
        except Exception:
            if attempt + 1 == attempts:
                raise
            time.sleep(min(8, .5 * 2 ** attempt))
    raise RuntimeError("unreachable")


def append_symbol(symbol: str, item: dict, source_manifest: Path,
                  output: Path, end_ms: int) -> tuple[str, dict]:
    source_path = resolved(str(item["path"]), source_manifest)
    source_raw = source_path.read_bytes()
    if digest(source_raw) != item["sha256"]:
        raise ValueError(f"source hash mismatch: {symbol}")
    candle = pickle.loads(source_raw)
    old_times = list(map(int, candle["time"]))
    if not old_times or any(right - left != 3600
                            for left, right in zip(old_times, old_times[1:])):
        raise ValueError(f"source is not contiguous: {symbol}")
    rows = fetch_latest(str(item["instrument"]))
    confirmed = {int(row[0]) // 1000: row for row in rows
                 if row[-1] == "1" and int(row[0]) <= end_ms}
    overlap = 0
    for index, timestamp in enumerate(old_times):
        row = confirmed.get(timestamp)
        if row is None:
            continue
        overlap += 1
        incoming = tuple(float(row[position]) for position in range(1, 8))
        existing = tuple(float(candle[field][index]) for field in FIELDS)
        if incoming != existing:
            changed = [field for field, left, right in zip(
                FIELDS, existing, incoming) if left != right]
            raise ValueError(
                f"confirmed candle revision: {symbol} {timestamp} {changed}")
    additions = sorted(timestamp for timestamp in confirmed
                       if timestamp > old_times[-1])
    if additions and additions[0] != old_times[-1] + 3600:
        raise ValueError(f"append gap: {symbol} {old_times[-1]} -> {additions[0]}")
    if any(right - left != 3600 for left, right in zip(additions, additions[1:])):
        raise ValueError(f"new candles are not contiguous: {symbol}")
    merged = {field: list(candle[field]) for field in ("time", *FIELDS)}
    for timestamp in additions:
        row = confirmed[timestamp]
        merged["time"].append(timestamp)
        for position, field in enumerate(FIELDS, 1):
            merged[field].append(float(row[position]))
    output_path = output / source_path.name
    raw = pickle.dumps(merged, protocol=pickle.HIGHEST_PROTOCOL)
    output_path.write_bytes(raw)
    return symbol, {
        "instrument": item["instrument"], "path": str(output_path),
        "bars": len(merged["time"]), "start": merged["time"][0],
        "end": merged["time"][-1], "sha256": digest(raw),
        "source_sha256": item["sha256"],
        "overlap_bars_verified": overlap, "appended_bars": len(additions),
    }


def parse_end(value: str | None) -> int:
    if value is None:
        return int(time.time() * 1000)
    stamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if stamp.tzinfo is None:
        stamp = stamp.replace(tzinfo=timezone.utc)
    return int(stamp.timestamp() * 1000)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--end", default=None)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("workers must be positive")
    if (args.out / "manifest.json").exists():
        parser.error("output snapshot already exists")
    args.out.mkdir(parents=True, exist_ok=True)
    source_raw = args.source_manifest.read_bytes()
    source = json.loads(source_raw)
    end_ms = parse_end(args.end)
    completed = {}
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(
            append_symbol, symbol, item, args.source_manifest, args.out, end_ms): symbol
            for symbol, item in source["symbols"].items()}
        for future in as_completed(futures):
            symbol, item = future.result()
            completed[symbol] = item
            print(json.dumps({"symbol": symbol, **item}), flush=True)
    ordered = {symbol: completed[symbol] for symbol in source["symbols"]}
    report = {
        "status": "APPEND_ONLY_RESEARCH_INPUT",
        "venue": source.get("venue", "OKX global USDT swap public API"),
        "interval": "1H", "created_at": datetime.now(timezone.utc).isoformat(),
        "requested_end": end_ms // 1000,
        "source_manifest": str(args.source_manifest),
        "source_manifest_sha256": digest(source_raw), "symbols": ordered,
    }
    manifest = args.out / "manifest.json"
    manifest.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({
        "manifest": str(manifest), "symbols": len(ordered),
        "appended_bars": sum(item["appended_bars"] for item in ordered.values()),
        "overlap_bars_verified": sum(
            item["overlap_bars_verified"] for item in ordered.values()),
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
