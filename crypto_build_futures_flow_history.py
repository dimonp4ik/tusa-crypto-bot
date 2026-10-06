"""Recover 5m trade-flow fields from cached Binance UM kline archives."""
from __future__ import annotations

import csv
import hashlib
import io
import json
import pickle
import zipfile
from pathlib import Path

import crypto_build_futures_funding_history as source


ROOT = Path("reports/audit_2026_09_08/crypto_futures_flow_history")
ZIP_ROOT = source.ZIP_DIR / "klines"
FIELDS = ("open", "high", "low", "close", "volume", "quote_volume",
          "trade_count", "taker_buy_volume", "taker_buy_quote_volume")


def build(symbol: str) -> tuple[dict, list[dict]]:
    rows = {}
    files = []
    for path in sorted((ZIP_ROOT / symbol).glob("*.zip")):
        raw = path.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        files.append({"path": str(path), "bytes": len(raw), "sha256": digest})
        with zipfile.ZipFile(io.BytesIO(raw)) as archive:
            bad = archive.testzip()
            if bad:
                raise ValueError(f"corrupt {bad}: {path}")
            with archive.open(archive.namelist()[0]) as member:
                for row in csv.reader(io.TextIOWrapper(member, encoding="utf-8")):
                    if not row or not row[0].isdigit():
                        continue
                    stamp = source.normalize_ms(row[0]) // 1000
                    rows[stamp] = (float(row[1]), float(row[2]), float(row[3]),
                                   float(row[4]), float(row[5]), float(row[7]),
                                   float(row[8]), float(row[9]), float(row[10]))
    ordered = sorted(rows)
    data = {"time": ordered}
    for index, field in enumerate(FIELDS):
        data[field] = [rows[stamp][index] for stamp in ordered]
    return data, files


def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    symbols, source_files = {}, {}
    for symbol in source.SYMBOLS:
        data, files = build(symbol)
        payload = pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL)
        path = ROOT / f"{symbol}_flow_5min_2020_2026.pkl"
        path.write_bytes(payload)
        gaps = [right-left for left, right in zip(data["time"], data["time"][1:])]
        symbols[symbol] = {"path": str(path), "rows": len(data["time"]),
                           "start": data["time"][0], "end": data["time"][-1],
                           "sha256": hashlib.sha256(payload).hexdigest(),
                           "gaps_over_5m": sum(gap > 300 for gap in gaps),
                           "maximum_gap_seconds": max(gaps, default=0)}
        source_files[symbol] = files
        print(json.dumps({"built": symbol, **symbols[symbol]}), flush=True)
    report = {"status": "RESEARCH_INPUT", "venue": "Binance UM perpetual",
              "interval": "5m", "fields": FIELDS, "symbols": symbols,
              "source_files": source_files,
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (ROOT / "manifest.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"manifest": str(ROOT / "manifest.json"),
                      "symbols": symbols}, indent=2), flush=True)


if __name__ == "__main__":
    main()
