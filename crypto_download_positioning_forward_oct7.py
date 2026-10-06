"""Freeze and download Oct 1-6 positioning data for a new-feature forward."""
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

import crypto_build_futures_metrics_history as history


ROOT = Path("reports/audit_2026_09_08/crypto_positioning_oct7_forward")
START = dt.date(2026, 10, 1)
END = dt.date(2026, 10, 6)
FREEZE = {
    "status": "FROZEN_EXPLORATORY_FAMILY_REQUIRES_NEW_FORWARD",
    "frozen_at_utc": None,
    "family": "top_divergence",
    "model": {"kind": "HistGradientBoostingClassifier", "leaves": 31,
              "leaf": 400, "l2": 6.0, "max_iter": 180,
              "learning_rate": .06, "random_state": 20261007},
    "threshold": 0.7050843607886652,
    "target_pct": .5, "stop_pct": 1.0, "max_bars": 24,
    "entry": "next 5m market open",
    "portfolio": "existing runtime_gate: at most two correlated entries per hour",
    "money": "3% equity margin, fixed 10x gross",
    "warning": "Selected after historical family inspection; forward evidence only.",
}


def dates():
    value = START
    while value <= END:
        yield value.isoformat()
        value += dt.timedelta(days=1)


def fetch(item: tuple[str, str]) -> dict:
    symbol, day = item
    url = history.URL.format(symbol=symbol, day=day)
    response = requests.get(url, timeout=35)
    if response.status_code == 404:
        return {"symbol": symbol, "day": day, "status": "missing", "url": url}
    response.raise_for_status()
    raw = response.content
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        bad = archive.testzip()
        if bad:
            raise ValueError(f"corrupt {bad}: {url}")
    path = ROOT / "zips" / symbol / f"{symbol}-metrics-{day}.zip"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    return {"symbol": symbol, "day": day, "status": "downloaded",
            "url": url, "path": str(path), "bytes": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest()}


def build(symbol: str, files: list[dict]) -> dict:
    rows = {}
    for item in sorted(files, key=lambda row: row["day"]):
        if item["status"] != "downloaded":
            continue
        with zipfile.ZipFile(item["path"]) as archive:
            with archive.open(archive.namelist()[0]) as member:
                for row in csv.DictReader(io.TextIOWrapper(member, encoding="utf-8")):
                    stamp = int(dt.datetime.fromisoformat(row["create_time"])
                                .replace(tzinfo=dt.timezone.utc).timestamp())+300
                    values = tuple(float(row[field]) if row.get(field, "") else float("nan")
                                   for field in history.FIELDS)
                    rows[stamp] = values
    ordered = sorted(rows)
    data = {"time": ordered}
    for index, field in enumerate(history.FIELDS):
        data[field] = [rows[stamp][index] for stamp in ordered]
    payload = pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL)
    path = ROOT / f"{symbol}_metrics_forward.pkl"
    path.write_bytes(payload)
    return {"path": str(path), "rows": len(ordered),
            "start": ordered[0] if ordered else None,
            "end": ordered[-1] if ordered else None,
            "sha256": hashlib.sha256(payload).hexdigest()}


def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    freeze_path = ROOT / "freeze.json"
    frozen = dict(FREEZE)
    frozen["frozen_at_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
    freeze_path.write_text(json.dumps(frozen, indent=2), encoding="utf-8")
    tasks = [(symbol, day) for symbol in history.SYMBOLS for day in dates()]
    files = []
    with ThreadPoolExecutor(max_workers=16) as pool:
        futures = [pool.submit(fetch, item) for item in tasks]
        for future in as_completed(futures):
            files.append(future.result())
    symbols = {symbol: build(symbol, [item for item in files
                                      if item["symbol"] == symbol])
               for symbol in history.SYMBOLS}
    report = {"status": "ISOLATED_NEW_FEATURE_FORWARD_INPUT",
              "requested": [START.isoformat(), END.isoformat()],
              "availability_rule": "source timestamp plus 300 seconds",
              "freeze_sha256": hashlib.sha256(freeze_path.read_bytes()).hexdigest(),
              "files": sorted(files, key=lambda row: (row["symbol"], row["day"])),
              "symbols": symbols,
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (ROOT / "manifest.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"status": report["status"],
                      "missing": sum(item["status"] == "missing" for item in files),
                      "symbols": symbols}, indent=2), flush=True)


if __name__ == "__main__":
    main()
