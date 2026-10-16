"""Download the frozen 2026-10-01..15 Binance UM 5m holdout inputs."""
from __future__ import annotations

import csv
import datetime as dt
import hashlib
import io
import json
import pickle
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

import crypto_build_futures_flow_history as flow_schema
import crypto_build_futures_funding_history as time_source
import crypto_build_futures_metrics_history as metric_schema


ROOT = Path("reports/audit_2026_09_08/crypto_futures_oct16_holdout")
ZIP_ROOT = ROOT / "daily_zips"
SYMBOLS = time_source.SYMBOLS
START = dt.date(2026, 10, 1)
END = dt.date(2026, 10, 15)
URLS = {
    "klines": ("https://data.binance.vision/data/futures/um/daily/klines/"
               "{symbol}/5m/{symbol}-5m-{day}.zip"),
    "metrics": ("https://data.binance.vision/data/futures/um/daily/metrics/"
                "{symbol}/{symbol}-metrics-{day}.zip"),
}


def days():
    value = START
    while value <= END:
        yield value.isoformat()
        value += dt.timedelta(days=1)


def download(item: tuple[str, str, str]) -> dict:
    kind, symbol, day = item
    stem = f"{symbol}-5m-{day}.zip" if kind == "klines" else f"{symbol}-metrics-{day}.zip"
    target = ZIP_ROOT / kind / symbol / stem
    if target.exists() and target.stat().st_size > 100:
        raw = target.read_bytes()
        try:
            with zipfile.ZipFile(io.BytesIO(raw)) as archive:
                if archive.testzip() is None:
                    return {"kind": kind, "symbol": symbol, "day": day,
                            "status": "cached", "path": str(target),
                            "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
        except zipfile.BadZipFile:
            pass
        target.unlink(missing_ok=True)
    response = requests.get(URLS[kind].format(symbol=symbol, day=day), timeout=35)
    if response.status_code == 404:
        return {"kind": kind, "symbol": symbol, "day": day, "status": "missing"}
    response.raise_for_status()
    raw = response.content
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        bad = archive.testzip()
        if bad:
            raise ValueError(f"corrupt {bad}: {response.url}")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(raw)
    return {"kind": kind, "symbol": symbol, "day": day,
            "status": "downloaded", "path": str(target),
            "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def build_candles(items: list[dict]) -> dict:
    rows = {}
    for item in sorted(items, key=lambda value: value["day"]):
        if item["status"] == "missing":
            continue
        raw = Path(item["path"]).read_bytes()
        with zipfile.ZipFile(io.BytesIO(raw)) as archive:
            with archive.open(archive.namelist()[0]) as member:
                for row in csv.reader(io.TextIOWrapper(member, encoding="utf-8")):
                    if not row or not row[0].isdigit():
                        continue
                    stamp = time_source.normalize_ms(row[0]) // 1000
                    rows[stamp] = (
                        float(row[1]), float(row[2]), float(row[3]), float(row[4]),
                        float(row[5]), float(row[7]), float(row[8]), float(row[9]),
                        float(row[10]),
                    )
    ordered = sorted(rows)
    data = {"time": ordered}
    for index, field in enumerate(flow_schema.FIELDS):
        data[field] = [rows[stamp][index] for stamp in ordered]
    return data


def build_metrics(items: list[dict]) -> dict:
    rows = {}
    for item in sorted(items, key=lambda value: value["day"]):
        if item["status"] == "missing":
            continue
        raw = Path(item["path"]).read_bytes()
        with zipfile.ZipFile(io.BytesIO(raw)) as archive:
            with archive.open(archive.namelist()[0]) as member:
                for row in csv.DictReader(io.TextIOWrapper(member, encoding="utf-8")):
                    stamp = int(dt.datetime.fromisoformat(row["create_time"])
                                .replace(tzinfo=dt.timezone.utc).timestamp()) + 300
                    rows[stamp] = tuple(
                        float(row[field]) if row.get(field, "") else float("nan")
                        for field in metric_schema.FIELDS)
    ordered = sorted(rows)
    data = {"time": ordered}
    for index, field in enumerate(metric_schema.FIELDS):
        data[field] = [rows[stamp][index] for stamp in ordered]
    return data


def write_pickle(path: Path, data: dict) -> dict:
    payload = pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL)
    path.write_bytes(payload)
    times = data["time"]
    gaps = [right - left for left, right in zip(times, times[1:])]
    return {
        "path": str(path), "sha256": hashlib.sha256(payload).hexdigest(),
        "rows": len(times), "start": times[0] if times else None,
        "end": times[-1] if times else None,
        "gaps_over_5m": sum(gap > 300 for gap in gaps),
        "maximum_gap_seconds": max(gaps, default=0),
    }


def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    tasks = [(kind, symbol, day) for kind in URLS for symbol in SYMBOLS for day in days()]
    files = []
    with ThreadPoolExecutor(max_workers=16) as pool:
        futures = {pool.submit(download, item): item for item in tasks}
        for index, future in enumerate(as_completed(futures), 1):
            files.append(future.result())
            if index % 40 == 0:
                print(json.dumps({"checked": index, "total": len(tasks)}), flush=True)
    symbols = {}
    for symbol in SYMBOLS:
        selected = [item for item in files if item["symbol"] == symbol]
        candle = build_candles([item for item in selected if item["kind"] == "klines"])
        metrics = build_metrics([item for item in selected if item["kind"] == "metrics"])
        symbols[symbol] = {
            "candles": write_pickle(ROOT / f"{symbol}_flow_5m_holdout.pkl", candle),
            "metrics": write_pickle(ROOT / f"{symbol}_metrics_holdout.pkl", metrics),
        }
        print(json.dumps({"built": symbol, **symbols[symbol]}), flush=True)
    report = {
        "status": "RESEARCH_INPUT", "venue": "Binance UM perpetual public archive",
        "requested": [START.isoformat(), END.isoformat()],
        "availability_rule": "metrics period timestamp plus 300 seconds",
        "symbols": symbols,
        "files": sorted(files, key=lambda item: (item["kind"], item["symbol"], item["day"])),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    (ROOT / "manifest.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"manifest": str(ROOT / "manifest.json"), "symbols": symbols},
                     indent=2), flush=True)


if __name__ == "__main__":
    main()
