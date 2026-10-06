"""Download reproducible Binance UM 5m derivatives metrics, 2022-2026."""
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


ROOT = Path("reports/audit_2026_09_08/crypto_futures_metrics_history")
ZIP_ROOT = ROOT / "daily_zips"
SYMBOLS = ("BTCUSDT", "ETHUSDT", "XRPUSDT", "XLMUSDT", "LINKUSDT",
           "SOLUSDT", "AVAXUSDT", "AAVEUSDT")
START = dt.date(2022, 1, 1)
END = dt.date(2026, 9, 30)
URL = ("https://data.binance.vision/data/futures/um/daily/metrics/"
       "{symbol}/{symbol}-metrics-{day}.zip")
FIELDS = ("sum_open_interest", "sum_open_interest_value",
          "count_toptrader_long_short_ratio",
          "sum_toptrader_long_short_ratio", "count_long_short_ratio",
          "sum_taker_long_short_vol_ratio")


def days():
    value = START
    while value <= END:
        yield value.isoformat()
        value += dt.timedelta(days=1)


def download(item: tuple[str, str]) -> dict:
    symbol, day = item
    target = ZIP_ROOT / symbol / f"{symbol}-metrics-{day}.zip"
    if target.exists() and target.stat().st_size > 100:
        raw = target.read_bytes()
        try:
            with zipfile.ZipFile(io.BytesIO(raw)) as archive:
                if archive.testzip() is None:
                    return {"symbol": symbol, "day": day, "status": "cached",
                            "path": str(target), "bytes": len(raw),
                            "sha256": hashlib.sha256(raw).hexdigest()}
        except zipfile.BadZipFile:
            pass
        target.unlink(missing_ok=True)
    response = requests.get(URL.format(symbol=symbol, day=day), timeout=35)
    if response.status_code == 404:
        return {"symbol": symbol, "day": day, "status": "missing"}
    response.raise_for_status()
    raw = response.content
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        bad = archive.testzip()
        if bad:
            raise ValueError(f"corrupt {bad}: {response.url}")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(raw)
    return {"symbol": symbol, "day": day, "status": "downloaded",
            "path": str(target), "bytes": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest()}


def build_symbol(symbol: str, items: list[dict]) -> dict:
    rows: dict[int, tuple[float, ...]] = {}
    incomplete_rows = 0
    duplicate_timestamps = 0
    for item in sorted(items, key=lambda row: row["day"]):
        if item["status"] == "missing":
            continue
        raw = Path(item["path"]).read_bytes()
        with zipfile.ZipFile(io.BytesIO(raw)) as archive:
            with archive.open(archive.namelist()[0]) as member:
                reader = csv.DictReader(io.TextIOWrapper(member, encoding="utf-8"))
                for row in reader:
                    stamp = int(dt.datetime.fromisoformat(row["create_time"])
                                .replace(tzinfo=dt.timezone.utc).timestamp())
                    # Conservative availability: even if the label is the period
                    # start, the row is not usable until that five-minute period ends.
                    available = stamp + 300
                    values = tuple(float(row[field]) if row.get(field, "") else float("nan")
                                   for field in FIELDS)
                    incomplete_rows += int(any(value != value for value in values))
                    duplicate_timestamps += int(available in rows)
                    rows[available] = values
    ordered = sorted(rows)
    data = {"time": ordered}
    for position, field in enumerate(FIELDS):
        data[field] = [rows[stamp][position] for stamp in ordered]
    payload = pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL)
    path = ROOT / f"{symbol}_metrics_2022_2026.pkl"
    path.write_bytes(payload)
    gaps = [right-left for left, right in zip(ordered, ordered[1:])]
    return {"path": str(path), "rows": len(ordered),
            "start": ordered[0] if ordered else None,
            "end": ordered[-1] if ordered else None,
            "sha256": hashlib.sha256(payload).hexdigest(),
            "incomplete_rows": incomplete_rows,
            "duplicate_timestamps": duplicate_timestamps,
            "gaps_over_5m": sum(value > 300 for value in gaps),
            "maximum_gap_seconds": max(gaps, default=0)}


def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    tasks = [(symbol, day) for symbol in SYMBOLS for day in days()]
    results: list[dict] = []
    with ThreadPoolExecutor(max_workers=32) as pool:
        futures = {pool.submit(download, item): item for item in tasks}
        for index, future in enumerate(as_completed(futures), 1):
            results.append(future.result())
            if index % 500 == 0:
                print(json.dumps({"checked": index, "total": len(tasks)}), flush=True)
    symbols = {}
    for symbol in SYMBOLS:
        symbols[symbol] = build_symbol(
            symbol, [item for item in results if item["symbol"] == symbol])
        print(json.dumps({"built": symbol, **symbols[symbol]}), flush=True)
    report = {
        "status": "RESEARCH_INPUT",
        "venue": "Binance UM perpetual public data",
        "interval": "5m", "requested": [START.isoformat(), END.isoformat()],
        "availability_rule": "archive timestamp plus 300 seconds",
        "fields": FIELDS, "symbols": symbols,
        "files": sorted(results, key=lambda row: (row["symbol"], row["day"])),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    (ROOT / "manifest.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"manifest": str(ROOT / "manifest.json"),
                      "symbols": symbols}, indent=2), flush=True)


if __name__ == "__main__":
    main()
