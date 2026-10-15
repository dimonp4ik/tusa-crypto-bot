"""Download and causally aggregate official Binance BTC USD-M book-depth snapshots."""
from __future__ import annotations

import concurrent.futures
import hashlib
import io
import json
import time
import zipfile
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests


OUT = Path("reports/audit_2026_09_08/crypto_btc_bookdepth_5m_oct15")
START = date(2023, 1, 1)
END = date(2026, 9, 30)
BASE = ("https://data.binance.vision/data/futures/um/daily/bookDepth/"
        "BTCUSDT/BTCUSDT-bookDepth-{day}.zip")


def epoch_seconds(index: pd.DatetimeIndex) -> np.ndarray:
    """Convert any pandas datetime resolution to Unix seconds explicitly."""
    return (index.as_unit("ns").astype("int64")//1_000_000_000).to_numpy()


def days(start: date, end: date):
    current = start
    while current <= end:
        yield current
        current += timedelta(days=1)


def fetch(day: date) -> tuple[dict, pd.DataFrame | None]:
    label = day.isoformat()
    url = BASE.format(day=label)
    error = None
    for attempt in range(4):
        try:
            response = requests.get(url, timeout=45)
            if response.status_code == 404:
                return {"day": label, "url": url, "status": "missing"}, None
            response.raise_for_status()
            raw = response.content
            archive_hash = hashlib.sha256(raw).hexdigest()
            with zipfile.ZipFile(io.BytesIO(raw)) as archive:
                member = archive.namelist()[0]
                source = pd.read_csv(archive.open(member))
            required = {"timestamp", "percentage", "depth", "notional"}
            if not required.issubset(source.columns):
                raise ValueError(f"bad columns: {list(source.columns)}")
            source = source.drop_duplicates(["timestamp", "percentage"])
            source["timestamp"] = pd.to_datetime(source.timestamp, utc=True)
            for column in ("percentage", "depth", "notional"):
                source[column] = pd.to_numeric(source[column], errors="coerce")
            source = source.dropna(subset=list(required))
            counts = source.groupby("timestamp").percentage.nunique()
            # Binance added +/-0.2% bands in mid-January 2026.  The six bands
            # used below remain unchanged, so both the former 10-band and new
            # 12-band records are valid.
            good_times = counts[counts.ge(10)].index
            source = source[source.timestamp.isin(good_times)]
            depth = source.pivot(index="timestamp", columns="percentage", values="depth")
            notional = source.pivot(index="timestamp", columns="percentage", values="notional")
            needed = [-5, -2, -1, 1, 2, 5]
            if any(level not in notional for level in needed):
                raise ValueError("missing depth bands")
            snap = pd.DataFrame(index=notional.index)
            for level in (1, 2, 5):
                bid, ask = notional[-level], notional[level]
                snap[f"imbalance{level}"] = (bid-ask)/(bid+ask).replace(0, np.nan)
                snap[f"bid_notional{level}"] = bid
                snap[f"ask_notional{level}"] = ask
            implied_bid = notional[-1]/depth[-1].replace(0, np.nan)/.99
            implied_ask = notional[1]/depth[1].replace(0, np.nan)/1.01
            snap["implied_mid"] = (implied_bid+implied_ask)/2
            snap["bar_time"] = snap.index.floor("5min")
            grouped = snap.groupby("bar_time")
            result = grouped.last()
            for level in (1, 2, 5):
                result[f"imbalance{level}_mean"] = grouped[f"imbalance{level}"].mean()
                result[f"imbalance{level}_std"] = grouped[f"imbalance{level}"].std().fillna(0)
            result["snapshot_count"] = grouped.size()
            # pandas may parse this archive as datetime64[us]; normalize to ns
            # explicitly before conversion so epoch seconds are not 1000x short.
            result["available_time"] = (
                epoch_seconds(result.index)+300
            ).astype("int64")
            result = result.reset_index(drop=True)
            meta = {
                "day": label, "url": url, "status": "ok",
                "archive_sha256": archive_hash, "archive_bytes": len(raw),
                "raw_rows": int(len(source)), "snapshots": int(len(snap)),
                "bars": int(len(result)),
            }
            return meta, result.replace([np.inf, -np.inf], np.nan).dropna()
        except Exception as exc:  # retry transient archive/network failures
            error = f"{type(exc).__name__}: {exc}"
            time.sleep(1+attempt)
    return {"day": label, "url": url, "status": "error", "error": error}, None


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    archive_meta = []
    month_outputs = []
    schedule = list(days(START, END))
    by_month: dict[str, list[date]] = {}
    for day in schedule:
        by_month.setdefault(day.strftime("%Y-%m"), []).append(day)
    for month, month_days in by_month.items():
        path = OUT / f"BTCUSDT_bookdepth_5m_{month}.pkl"
        meta_path = OUT / f"BTCUSDT_bookdepth_5m_{month}.json"
        if path.exists() and meta_path.exists():
            meta = json.loads(meta_path.read_text())
            current_schema = meta.get("schema_version") == 3
            if (current_schema
                    and hashlib.sha256(path.read_bytes()).hexdigest() == meta.get("sha256")):
                archive_meta.extend(meta["archives"])
                month_outputs.append(meta)
                print(month, "cached", meta["rows"])
                continue
        frames, records = [], []
        with concurrent.futures.ThreadPoolExecutor(max_workers=20) as pool:
            futures = {pool.submit(fetch, day): day for day in month_days}
            for future in concurrent.futures.as_completed(futures):
                meta, frame = future.result()
                records.append(meta)
                if frame is not None:
                    frames.append(frame)
        records.sort(key=lambda item: item["day"])
        if not frames:
            raise RuntimeError(f"no usable book-depth files for {month}")
        combined = pd.concat(frames, ignore_index=True).sort_values("available_time")
        combined = combined.drop_duplicates("available_time", keep="last")
        combined.to_pickle(path)
        meta = {
            "schema_version": 3,
            "month": month, "path": str(path), "rows": int(len(combined)),
            "start": int(combined.available_time.min()),
            "end": int(combined.available_time.max()),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "archives": records,
        }
        meta_path.write_text(json.dumps(meta, indent=2))
        archive_meta.extend(records)
        month_outputs.append(meta)
        print(month, "downloaded", len(combined),
              "bad", sum(r["status"] != "ok" for r in records))
    manifest = {
        "status": "DOWNLOADED_AWAITING_PRICE_ALIGNMENT_VALIDATION",
        "source": "official Binance public USD-M daily bookDepth archive",
        "symbol": "BTCUSDT", "interval": "5m causal aggregates",
        "start": START.isoformat(), "end": END.isoformat(),
        "availability": "5m bucket close; source snapshots occur before close",
        "months": [{key: value for key, value in item.items() if key != "archives"}
                   for item in month_outputs],
        "archives": archive_meta,
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps({
        "months": len(month_outputs), "archives": len(archive_meta),
        "ok": sum(item["status"] == "ok" for item in archive_meta),
        "missing": sum(item["status"] == "missing" for item in archive_meta),
        "errors": sum(item["status"] == "error" for item in archive_meta),
    }, indent=2))


if __name__ == "__main__":
    main()
