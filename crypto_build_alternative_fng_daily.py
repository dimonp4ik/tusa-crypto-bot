"""Capture a hashed, conservatively delayed Alternative.me FNG history."""
from __future__ import annotations

import hashlib
import json
import pickle
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests


ROOT = Path("reports/audit_2026_09_08/crypto_alternative_fng_daily_oct9")
ENDPOINT = "https://api.alternative.me/fng/"


def digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    response = requests.get(
        ENDPOINT, params={"limit": 0, "format": "json"}, timeout=60,
        headers={"User-Agent": "material-move-research/1.0"},
    )
    response.raise_for_status()
    raw = response.content
    decoded = json.loads(raw)
    if decoded.get("metadata", {}).get("error") is not None:
        raise ValueError(f"Alternative.me error: {decoded['metadata']['error']}")
    frame = pd.DataFrame(decoded.get("data", []))
    required = {"value", "value_classification", "timestamp"}
    if not required.issubset(frame.columns):
        raise ValueError(f"schema mismatch: {sorted(frame.columns)}")
    frame = frame[list(required)].copy()
    frame["value"] = pd.to_numeric(frame.value, errors="coerce")
    frame["source_epoch"] = pd.to_numeric(
        frame.timestamp, errors="coerce").astype("Int64")
    if frame[["value", "source_epoch"]].isna().any().any():
        raise ValueError("missing Fear and Greed value or timestamp")
    frame["source_epoch"] = frame.source_epoch.astype(np.int64)
    frame = frame.sort_values("source_epoch").drop_duplicates(
        "source_epoch", keep=False).reset_index(drop=True)
    if not frame.value.between(0, 100).all():
        raise ValueError("Fear and Greed value outside 0-100")
    if np.any(frame.source_epoch.to_numpy(np.int64) % 86400 != 0):
        raise ValueError("Fear and Greed timestamps are not UTC day boundaries")
    # Historical API rows do not expose publication timestamps. A full-day
    # delay is deliberately conservative and prevents same-day knowledge.
    frame["available_time"] = frame.source_epoch+86400
    times = frame.source_epoch.to_numpy(np.int64)
    gaps = np.diff(times)
    missing_days = []
    for left, gap in zip(times[:-1], gaps):
        for offset in range(86400, int(gap), 86400):
            missing_days.append(str(pd.to_datetime(
                left+offset, unit="s", utc=True).date()))
    raw_path = ROOT / "alternative_fng_response.json"
    raw_path.write_bytes(raw)
    data_path = ROOT / "alternative_fng_daily.pkl"
    payload = {column: frame[column].to_numpy() for column in (
        "source_epoch", "available_time", "value", "value_classification")}
    data_path.write_bytes(pickle.dumps(payload, protocol=5))
    manifest = {
        "status": "IMMUTABLE_PUBLIC_RESEARCH_INPUT",
        "provider": "Alternative.me Crypto Fear and Greed Index",
        "endpoint": ENDPOINT, "request_url": response.url,
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        "causal_availability": (
            "Historical rows expose UTC day labels but no publication time; "
            "each value is delayed by 24 hours and stale values expire after 48 hours."),
        "raw_path": str(raw_path), "raw_sha256": digest(raw),
        "data_path": str(data_path),
        "data_sha256": digest(data_path.read_bytes()),
        "quality": {
            "rows": len(frame), "first_source_time": int(times.min()),
            "last_source_time": int(times.max()), "missing_days": missing_days,
            "maximum_gap_days": int(gaps.max()//86400) if len(gaps) else 0,
            "duplicate_timestamps": 0,
            "utc_day_aligned": True, "minimum_value": int(frame.value.min()),
            "maximum_value": int(frame.value.max()),
        },
    }
    manifest_path = ROOT / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True),
                             encoding="utf-8")
    print(json.dumps({
        "raw_sha256": manifest["raw_sha256"],
        "data_sha256": manifest["data_sha256"],
        "quality": manifest["quality"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
