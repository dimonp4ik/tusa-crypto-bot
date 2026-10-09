"""Capture a hashed causal daily BTC/ETH on-chain snapshot from Coin Metrics."""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests


ROOT = Path("reports/audit_2026_09_08/crypto_coinmetrics_onchain_daily_oct9")
ENDPOINT = "https://community-api.coinmetrics.io/v4/timeseries/asset-metrics"
ASSETS = ("btc", "eth")
METRICS = (
    "AssetEODCompletionTime", "AdrActCnt", "TxTfrCnt", "FeeTotNtv",
    "CapMVRVCur", "SplyCur",
)


def digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def fetch(start: str, end: str) -> tuple[bytes, str]:
    params = {
        "assets": ",".join(ASSETS), "metrics": ",".join(METRICS),
        "frequency": "1d", "start_time": start, "end_time": end,
        "page_size": 10000,
    }
    response = requests.get(
        ENDPOINT, params=params, timeout=90,
        headers={"User-Agent": "material-move-research/1.0"},
    )
    response.raise_for_status()
    return response.content, response.url


def normalize(raw: bytes, start: str, end: str) -> tuple[dict, dict]:
    decoded = json.loads(raw)
    if decoded.get("next_page_url"):
        raise ValueError("unexpected pagination; snapshot would be incomplete")
    frame = pd.DataFrame(decoded.get("data", []))
    required = {"asset", "time", *METRICS}
    if set(frame.columns) != required:
        raise ValueError(f"schema mismatch: {sorted(frame.columns)}")
    frame["source_time"] = pd.to_datetime(frame.pop("time"), utc=True)
    for name in METRICS:
        frame[name] = pd.to_numeric(frame[name], errors="coerce")
    if frame[list(METRICS)].isna().any().any():
        missing = frame[list(METRICS)].isna().sum()
        raise ValueError(f"missing metric values: {missing[missing.gt(0)].to_dict()}")
    frame["available_time"] = frame.AssetEODCompletionTime.astype(np.int64)
    frame["source_epoch"] = frame.source_time.map(
        lambda value: int(value.timestamp())).astype(np.int64)
    frame["availability_lag_hours"] = (
        frame.available_time-frame.source_epoch)/3600.0
    frame["causal_eligible"] = frame.availability_lag_hours.between(24, 72)
    frame = frame.sort_values(["asset", "source_time"]).reset_index(drop=True)
    if frame.duplicated(["asset", "source_time"]).any():
        raise ValueError("duplicate asset/day rows")
    quality = {}
    expected = pd.date_range(start, end, freq="D", tz="UTC")
    for asset, value in frame.groupby("asset", sort=True):
        missing = expected.difference(value.source_time)
        times = value.available_time.to_numpy(np.int64)
        quality[asset] = {
            "rows": len(value), "missing_days": [str(day.date()) for day in missing],
            "availability_monotonic": bool(np.all(np.diff(times) >= 0)),
            "eligible_lag_rows": int(value.causal_eligible.sum()),
            "lag_over_72h_rows": int(value.availability_lag_hours.gt(72).sum()),
            "lag_under_24h_rows": int(value.availability_lag_hours.lt(24).sum()),
            "lag_hours_min": round(float(value.availability_lag_hours.min()), 6),
            "lag_hours_median": round(float(value.availability_lag_hours.median()), 6),
            "lag_hours_max": round(float(value.availability_lag_hours.max()), 6),
            "eligible_by_source_year": {
                str(int(year)): int(count) for year, count in
                value[value.causal_eligible].groupby(value.source_time.dt.year).size().items()
            },
        }
    payload = {column: frame[column].to_numpy() for column in (
        "asset", "source_epoch", "available_time", "availability_lag_hours",
        "causal_eligible", "AdrActCnt", "TxTfrCnt", "FeeTotNtv",
        "CapMVRVCur", "SplyCur",
    )}
    return payload, quality


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2020-01-01")
    parser.add_argument("--end", default="2025-12-31")
    args = parser.parse_args()
    ROOT.mkdir(parents=True, exist_ok=True)
    raw, request_url = fetch(args.start, args.end)
    raw_path = ROOT / "coinmetrics_response.json"
    raw_path.write_bytes(raw)
    payload, quality = normalize(raw, args.start, args.end)
    data_path = ROOT / "btc_eth_onchain_daily.pkl"
    data_path.write_bytes(pickle.dumps(payload, protocol=5))
    manifest = {
        "status": "IMMUTABLE_PUBLIC_RESEARCH_INPUT",
        "provider": "Coin Metrics Community API",
        "endpoint": ENDPOINT, "request_url": request_url,
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        "assets": list(ASSETS), "metrics": list(METRICS),
        "start": args.start, "end": args.end,
        "causal_availability": (
            "A source day is usable only at its AssetEODCompletionTime; rows with "
            "availability lag outside 24-72 hours are excluded."),
        "reviewable_entity_metrics_excluded": [
            "FlowInExNtv", "FlowOutExNtv", "SplyExNtv"],
        "reviewable_exclusion_reason": (
            "Historical entity labels are backfilled from addresses currently known "
            "to belong to exchanges and observed status timestamps were in 2026."),
        "raw_path": str(raw_path), "raw_sha256": digest(raw),
        "data_path": str(data_path),
        "data_sha256": digest(data_path.read_bytes()),
        "quality": quality,
    }
    manifest_path = ROOT / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True),
                             encoding="utf-8")
    print(json.dumps({
        "raw_sha256": manifest["raw_sha256"],
        "data_sha256": manifest["data_sha256"], "quality": quality,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
