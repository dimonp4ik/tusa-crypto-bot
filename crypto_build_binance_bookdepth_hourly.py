"""Build a hashed causal hourly archive from Binance Futures bookDepth files."""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import io
import json
import pickle
import time
import zipfile
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests


ROOT = Path("reports/audit_2026_09_08/crypto_binance_bookdepth_hourly_oct9")
CACHE = ROOT / "daily_cache"
COMMON_BANDS = (-5.0, -2.0, -1.0, 1.0, 2.0, 5.0)
BASE = ("https://data.binance.vision/data/futures/um/daily/bookDepth/"
        "{symbol}/{symbol}-bookDepth-{day}.zip")


def digest_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def days(start: date, end: date) -> list[date]:
    return [start + timedelta(days=value)
            for value in range((end-start).days+1)]


def download_one(symbol: str, day: date, retries: int = 4) -> dict:
    CACHE.mkdir(parents=True, exist_ok=True)
    label = day.isoformat()
    target = CACHE / f"{symbol}-{label}.zip"
    url = BASE.format(symbol=symbol, day=label)
    if target.exists() and target.stat().st_size > 300:
        raw = target.read_bytes()
        try:
            with zipfile.ZipFile(io.BytesIO(raw)) as archive:
                archive.testzip()
            return {"day": label, "path": str(target), "url": url,
                    "sha256": digest_bytes(raw), "bytes": len(raw),
                    "cached": True}
        except (zipfile.BadZipFile, OSError):
            target.unlink(missing_ok=True)
    error = None
    for attempt in range(retries):
        try:
            response = requests.get(url, timeout=45)
            if response.status_code == 404:
                return {"day": label, "url": url, "missing": True}
            response.raise_for_status()
            raw = response.content
            with zipfile.ZipFile(io.BytesIO(raw)) as archive:
                bad = archive.testzip()
                if bad:
                    raise ValueError(f"bad zip member {bad}")
            target.write_bytes(raw)
            return {"day": label, "path": str(target), "url": url,
                    "sha256": digest_bytes(raw), "bytes": len(raw),
                    "cached": False}
        except Exception as exc:  # network retries are recorded in the manifest
            error = f"{type(exc).__name__}: {exc}"
            time.sleep(1.5 * (attempt+1))
    return {"day": label, "url": url, "error": error}


def parse_day(item: dict) -> tuple[pd.DataFrame, dict]:
    if item.get("missing") or item.get("error"):
        return pd.DataFrame(), item
    raw = Path(item["path"]).read_bytes()
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        names = archive.namelist()
        if len(names) != 1:
            raise ValueError(f"unexpected members for {item['day']}: {names}")
        with archive.open(names[0]) as source:
            frame = pd.read_csv(source)
    required = {"timestamp", "percentage", "depth", "notional"}
    if set(frame.columns) != required:
        raise ValueError(f"schema mismatch for {item['day']}: {list(frame.columns)}")
    frame["percentage"] = frame.percentage.astype(float)
    found_bands = tuple(float(value) for value in sorted(
        frame.percentage.dropna().unique()))
    if not set(COMMON_BANDS).issubset(found_bands):
        raise ValueError(f"common bands missing for {item['day']}: {found_bands}")
    frame["timestamp"] = pd.to_datetime(frame.timestamp, utc=True, errors="coerce")
    frame["notional"] = pd.to_numeric(frame.notional, errors="coerce")
    frame = frame.dropna(subset=["timestamp", "notional"])
    table = frame.pivot_table(index="timestamp", columns="percentage",
                              values="notional", aggfunc="last")
    table = table.reindex(columns=COMMON_BANDS).dropna()
    features = pd.DataFrame(index=table.index)
    for band in (1.0, 2.0, 5.0):
        bid, ask = table[-band], table[band]
        total = (bid+ask).replace(0, np.nan)
        label = str(band).replace(".", "_")
        features[f"imbalance_{label}"] = (bid-ask)/total
        features[f"notional_{label}"] = total
    features = features.replace([np.inf, -np.inf], np.nan).dropna()
    hourly_mean = features.resample("1h", label="left", closed="left").mean()
    hourly_std = features[[name for name in features
                           if name.startswith("imbalance_")]].resample(
                               "1h", label="left", closed="left").std()
    hourly_std.columns = [f"{name}_std" for name in hourly_std]
    counts = features.resample("1h", label="left", closed="left").size()
    hourly = hourly_mean.join(hourly_std)
    hourly["samples"] = counts
    hourly = hourly[hourly.samples.ge(90)].dropna()
    # Pandas may preserve second or microsecond datetime resolution depending
    # on the parsed source.  Do not assume nanoseconds when converting to epoch.
    hourly["available_time"] = np.asarray(
        [int(value.timestamp()) + 3600 for value in hourly.index],
        dtype=np.int64,
    )
    diagnostic = {**item, "bands": list(found_bands),
                  "csv_rows": len(frame), "snapshots": len(table),
                  "qualified_hours": len(hourly),
                  "minimum_hour_samples": int(hourly.samples.min())
                  if not hourly.empty else 0,
                  "maximum_hour_samples": int(hourly.samples.max())
                  if not hourly.empty else 0}
    return hourly.reset_index(drop=True), diagnostic


def build(symbol: str, start: date, end: date, workers: int) -> tuple[dict, dict]:
    requested = days(start, end)
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        downloaded = list(pool.map(lambda value: download_one(symbol, value), requested))
    pieces, diagnostics = [], []
    for index, item in enumerate(downloaded, 1):
        frame, diagnostic = parse_day(item)
        diagnostics.append(diagnostic)
        if not frame.empty:
            pieces.append(frame)
        if index % 100 == 0:
            print(json.dumps({"symbol": symbol, "parsed": index,
                              "requested": len(downloaded)}), flush=True)
    if not pieces:
        raise ValueError(f"no valid bookDepth rows for {symbol}")
    frame = pd.concat(pieces, ignore_index=True).sort_values(
        "available_time").drop_duplicates("available_time")
    payload = {column: frame[column].to_numpy()
               for column in frame.columns}
    quality = {
        "requested_days": len(requested),
        "available_days": sum(not x.get("missing") and not x.get("error")
                              for x in diagnostics),
        "missing_days": [x["day"] for x in diagnostics if x.get("missing")],
        "error_days": [{"day": x["day"], "error": x.get("error")}
                       for x in diagnostics if x.get("error")],
        "days_with_less_than_20_hours": [
            x["day"] for x in diagnostics
            if not x.get("missing") and not x.get("error")
            and int(x.get("qualified_hours", 0)) < 20],
        "qualified_hours": len(frame),
        "first_available_time": int(frame.available_time.min()),
        "last_available_time": int(frame.available_time.max()),
        "daily": diagnostics,
    }
    return payload, quality


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--start", default="2023-01-01")
    parser.add_argument("--end", default="2025-12-31")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    start, end = date.fromisoformat(args.start), date.fromisoformat(args.end)
    ROOT.mkdir(parents=True, exist_ok=True)
    payload, quality = build(args.symbol, start, end, args.workers)
    data_path = ROOT / f"{args.symbol}_bookdepth_1h.pkl"
    data_path.write_bytes(pickle.dumps(payload, protocol=5))
    data_hash = hashlib.sha256(data_path.read_bytes()).hexdigest()
    manifest = {
        "status": "IMMUTABLE_PUBLIC_RESEARCH_INPUT",
        "provider": "Binance Futures public data",
        "source_template": BASE,
        "symbol": args.symbol, "start": args.start, "end": args.end,
        "common_bands_pct": list(COMMON_BANDS),
        "causal_availability": "hour H aggregate is available at the start of H+1",
        "hourly_minimum_snapshots": 90,
        "data_path": str(data_path), "data_sha256": data_hash,
        "quality": quality,
    }
    manifest_path = ROOT / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True),
                             encoding="utf-8")
    print(json.dumps({
        "data_path": str(data_path), "sha256": data_hash,
        "qualified_hours": quality["qualified_hours"],
        "available_days": quality["available_days"],
        "missing_days": len(quality["missing_days"]),
        "error_days": len(quality["error_days"]),
        "days_with_less_than_20_hours": len(
            quality["days_with_less_than_20_hours"]),
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
