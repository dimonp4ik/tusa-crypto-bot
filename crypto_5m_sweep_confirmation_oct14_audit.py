"""Causal confirmation audit for 5m sweep-reclaim entries.

The base sweep geometry and fixed 1% target / 0.5% stop are inherited from
the earlier development route.  This audit changes only entry timing: it waits
for one fully closed confirmation bar and enters at the following 5m market
open.  Confirmation variants are semantic and fixed before annual comparison.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

import crypto_5m_liquidation_sweep_actual_oct5_audit as sweep
import crypto_5m_tree_filter_audit as base


ROOT = Path("reports/audit_2026_09_08")
OUT = ROOT / "crypto_5m_sweep_confirmation_oct14.json"
ENTRY_SPEC = (24, .10, .50, 1.25, .25)
TARGET_PCT = 1.0
STOP_PCT = .5
MAX_BARS = 24


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def confirmed_entries(entries: pd.DataFrame, candles: dict) -> pd.DataFrame:
    pieces = []
    for symbol, part in entries.groupby("symbol", sort=False):
        candle = candles[symbol]
        indices = part.entry_index.to_numpy(np.int64)
        valid = indices + 1 < len(candle["time"])
        valid &= (np.asarray(candle["time"])[np.minimum(
            indices + 1, len(candle["time"]) - 1)]
                  - np.asarray(candle["time"])[indices] == 300)
        found = part.loc[valid].copy()
        confirmation = indices[valid]
        signal = confirmation - 1
        direction = found.direction.to_numpy(np.int8)
        open_ = np.asarray(candle["open"], dtype=float)
        high = np.asarray(candle["high"], dtype=float)
        low = np.asarray(candle["low"], dtype=float)
        close = np.asarray(candle["close"], dtype=float)
        times = np.asarray(candle["time"], dtype=np.int64)
        midpoint = (high[signal] + low[signal]) / 2
        found["confirm_body_direction"] = (
            direction * (close[confirmation] - open_[confirmation])
            / open_[confirmation])
        found["confirm_beyond_sweep_midpoint"] = (
            direction * (close[confirmation] - midpoint) > 0)
        found["confirm_progress_from_sweep_close"] = (
            direction * (close[confirmation] - close[signal]) > 0)
        found["confirm_no_resweep"] = np.where(
            direction > 0,
            low[confirmation] >= low[signal],
            high[confirmation] <= high[signal])
        found["entry_index"] = confirmation + 1
        found["entry_time"] = times[confirmation + 1]
        stamp = pd.to_datetime(found.entry_time, unit="s", utc=True)
        found["year"] = stamp.dt.year.astype(int)
        found["month"] = (stamp.dt.year * 100 + stamp.dt.month).astype(int)
        pieces.append(found)
    return pd.concat(pieces, ignore_index=True)


def annual(frame: pd.DataFrame) -> dict:
    output = {}
    for year in range(2020, 2027):
        part = frame[frame.year.eq(year)]
        values = part.signed_return.to_numpy(float)
        output[str(year)] = {
            "n": len(part),
            "win_rate_pct": round(100 * float(np.mean(values > 0)), 4)
                if len(values) else 0.0,
            "mean_signed_price_return_pct": round(
                100 * float(values.mean()), 6) if len(values) else 0.0,
            "sum_signed_price_return_pct": round(
                100 * float(values.sum()), 6) if len(values) else 0.0,
        }
    return output


def main() -> None:
    candles, sources = base.load()
    raw = pd.concat([
        sweep.features(symbol, candle, ENTRY_SPEC[0])
        for symbol, candle in candles.items()
    ], ignore_index=True)
    entries = sweep.apply_spec(raw, ENTRY_SPEC)
    entries = confirmed_entries(entries, candles)
    variants = {
        "directional_body": entries.confirm_body_direction.gt(0),
        "beyond_sweep_midpoint": entries.confirm_beyond_sweep_midpoint,
        "progress_from_sweep_close":
            entries.confirm_progress_from_sweep_close,
        "no_resweep": entries.confirm_no_resweep,
        "midpoint_and_no_resweep": (
            entries.confirm_beyond_sweep_midpoint
            & entries.confirm_no_resweep),
        "midpoint_and_progress": (
            entries.confirm_beyond_sweep_midpoint
            & entries.confirm_progress_from_sweep_close),
        "all_confirmations": (
            entries.confirm_beyond_sweep_midpoint
            & entries.confirm_progress_from_sweep_close
            & entries.confirm_no_resweep),
    }
    results = {}
    for name, mask in variants.items():
        resolved = sweep.add_outcomes(
            entries.loc[mask].copy(), candles,
            TARGET_PCT, STOP_PCT, MAX_BARS)
        results[name] = {
            "entries": len(resolved),
            "annual": annual(resolved),
            "positive_mean_all_2020_2024": all(
                annual(resolved)[str(year)][
                    "mean_signed_price_return_pct"] > 0
                for year in range(2020, 2025)),
        }
    qualified = [name for name, value in results.items()
                 if value["positive_mean_all_2020_2024"]]
    report = {
        "status": "DEVELOPMENT_PASS" if qualified else "REJECTED_CONFIRMATION",
        "ready_for_live_orders": False,
        "protocol": {
            "base_entry": ENTRY_SPEC,
            "entry": "market proxy at open after one closed confirmation bar",
            "target_pct": TARGET_PCT,
            "stop_pct": STOP_PCT,
            "maximum_hold_bars": MAX_BARS,
            "collision_policy": "stop first",
            "annual_diagnostic": [2020, 2021, 2022, 2023, 2024, 2025, 2026],
            "money": "price-path diagnostic; no position-size optimization",
        },
        "raw_sweeps": len(raw),
        "base_entries_before_confirmation": len(entries),
        "results": results,
        "qualified_2020_2024": qualified,
        "decision": (
            "No runtime change. Confirmation is rejected unless its raw "
            "signed price expectancy stays positive in every 2020-2024 year."),
        "sources": {
            **sources,
            str(Path(base.__file__)): digest(Path(base.__file__)),
            str(Path(sweep.__file__)): digest(Path(sweep.__file__)),
            str(Path(__file__)): digest(Path(__file__)),
        },
        "limitations": [
            "Sweeps are reproducible OHLCV proxies, not historical order-book depth.",
            "2025-2026 were already diagnostic periods and are reported transparently.",
            "A negative raw price expectancy cannot be repaired by account sizing.",
        ],
    }
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({
        "status": report["status"],
        "base_entries": len(entries),
        "qualified": qualified,
        "results": results,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
