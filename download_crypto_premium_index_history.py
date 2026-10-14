"""Download and hash official Binance UM premium-index 15m archives."""
from __future__ import annotations

import concurrent.futures
import csv
import hashlib
import io
import json
import pickle
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

import crypto_futures_funding_regime_oct6_audit as funding


OUT = Path("reports/audit_2026_09_08/crypto_premium_index_15m_oct15")
START = (2020, 1)
END = (2026, 9)


def months():
    year, month = START
    while (year, month) <= END:
        yield f"{year:04d}-{month:02d}"
        month += 1
        if month == 13:
            year, month = year+1, 1


def fetch(item: tuple[str, str]) -> dict:
    symbol, month = item
    name = f"{symbol}-15m-{month}.zip"
    url = ("https://data.binance.vision/data/futures/um/monthly/"
           f"premiumIndexKlines/{symbol}/15m/{name}")
    try:
        with urllib.request.urlopen(url, timeout=30) as response:
            raw = response.read()
    except urllib.error.HTTPError as error:
        if error.code == 404:
            return {"symbol": symbol, "month": month, "url": url,
                    "status": "missing"}
        raise
    digest = hashlib.sha256(raw).hexdigest()
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        names = archive.namelist()
        if len(names) != 1:
            raise ValueError(f"unexpected archive members: {url} {names}")
        text = archive.read(names[0]).decode("utf-8")
    rows = []
    for values in csv.reader(io.StringIO(text)):
        if not values or not values[0].isdigit():
            continue
        timestamp = int(values[0])
        if timestamp > 10**14:  # defensive normalization if archive uses microseconds
            timestamp //= 1000
        rows.append((timestamp//1000, *map(float, values[1:5])))
    if not rows:
        raise ValueError(f"empty archive: {url}")
    return {"symbol": symbol, "month": month, "url": url,
            "status": "ok", "sha256": digest, "rows": rows}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    work = [(symbol, month) for symbol in funding.SYMBOLS for month in months()]
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=12) as executor:
        for result in executor.map(fetch, work):
            results.append(result)
            if result["status"] == "ok":
                print(json.dumps({key: result[key]
                                  for key in ("symbol", "month", "status")}),
                      flush=True)

    manifest = {"status": "VALIDATED", "interval": "15m",
                "start_month": f"{START[0]:04d}-{START[1]:02d}",
                "end_month": f"{END[0]:04d}-{END[1]:02d}",
                "source": "official Binance public data archive",
                "symbols": {}, "archives": []}
    for symbol in funding.SYMBOLS:
        by_time = {}
        archives = [item for item in results if item["symbol"] == symbol]
        for item in archives:
            public = {key: value for key, value in item.items() if key != "rows"}
            manifest["archives"].append(public)
            for timestamp, open_, high, low, close in item.get("rows", []):
                value = (open_, high, low, close)
                if timestamp in by_time and by_time[timestamp] != value:
                    raise ValueError(f"conflicting duplicate {symbol} {timestamp}")
                by_time[timestamp] = value
        times = sorted(by_time)
        if not times:
            raise ValueError(f"no history for {symbol}")
        data = {"time": times}
        for position, key in enumerate(("open", "high", "low", "close")):
            data[key] = [by_time[timestamp][position] for timestamp in times]
        path = OUT / f"{symbol}_premium_15m_2020_2026.pkl"
        path.write_bytes(pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL))
        manifest["symbols"][symbol] = {
            "path": str(path.resolve()), "rows": len(times),
            "start": times[0], "end": times[-1],
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "missing_intervals": sum(
                right-left != 900 for left, right in zip(times, times[1:])),
        }
    manifest_path = OUT / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"manifest": str(manifest_path),
                      "symbols": manifest["symbols"],
                      "archives_ok": sum(item["status"] == "ok"
                                         for item in results),
                      "archives_missing": sum(item["status"] == "missing"
                                              for item in results)}, indent=2))


if __name__ == "__main__":
    main()
