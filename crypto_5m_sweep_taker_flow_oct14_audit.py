"""Event-specific taker-flow audit for futures 5m sweep reversals."""
from __future__ import annotations

import hashlib
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from src import research_crypto_replay as sweep


ROOT = Path("reports/audit_2026_09_08")
MANIFEST = ROOT / "crypto_futures_flow_history" / "manifest.json"
OUT = ROOT / "crypto_5m_sweep_taker_flow_oct14.json"
ENTRY_SPEC = (24, .10, .50, 1.25, .25)
TARGET_PCT = 1.0
STOP_PCT = .5
MAX_BARS = 24


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load() -> tuple[dict, dict]:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    candles, sources = {}, {str(MANIFEST): digest(MANIFEST)}
    for symbol, item in sorted(manifest["symbols"].items()):
        path = Path(item["path"])
        raw = path.read_bytes()
        found = hashlib.sha256(raw).hexdigest()
        if found != item["sha256"]:
            raise ValueError(f"flow source hash mismatch: {symbol}")
        candle = pickle.loads(raw)
        required = ("time", "open", "high", "low", "close", "volume",
                    "quote_volume", "taker_buy_quote_volume")
        if any(len(candle[key]) != len(candle["time"]) for key in required):
            raise ValueError(f"flow length mismatch: {symbol}")
        candles[symbol] = candle
        sources[str(path)] = found
    return candles, sources


def enrich(entries: pd.DataFrame, candles: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    immediate, delayed = [], []
    for symbol, part in entries.groupby("symbol", sort=False):
        candle = candles[symbol]
        indices = part.entry_index.to_numpy(np.int64)
        quote = np.asarray(candle["quote_volume"], dtype=float)
        taker_buy = np.asarray(candle["taker_buy_quote_volume"], dtype=float)
        delta = np.divide(2 * taker_buy, quote, out=np.zeros_like(quote),
                          where=quote > 0) - 1
        direction = part.direction.to_numpy(np.int8)
        now = part.copy()
        now["signal_flow_absorption"] = -direction * delta[indices - 1]
        now["signal_flow_aligned"] = direction * delta[indices - 1]
        immediate.append(now)

        valid = indices + 1 < len(quote)
        valid &= (np.asarray(candle["time"])[np.minimum(
            indices + 1, len(quote) - 1)]
                  - np.asarray(candle["time"])[indices] == 300)
        later = part.loc[valid].copy()
        confirmation = indices[valid]
        later_direction = direction[valid]
        later["signal_flow_absorption"] = (
            -later_direction * delta[confirmation - 1])
        later["confirm_flow_aligned"] = (
            later_direction * delta[confirmation])
        later["flow_flip"] = later_direction * (
            delta[confirmation] - delta[confirmation - 1])
        later["entry_index"] = confirmation + 1
        later["entry_time"] = np.asarray(candle["time"])[confirmation + 1]
        stamp = pd.to_datetime(later.entry_time, unit="s", utc=True)
        later["year"] = stamp.dt.year.astype(int)
        later["month"] = (stamp.dt.year * 100 + stamp.dt.month).astype(int)
        delayed.append(later)
    return (pd.concat(immediate, ignore_index=True),
            pd.concat(delayed, ignore_index=True))


def annual(frame: pd.DataFrame) -> dict:
    result = {}
    for year in range(2020, 2027):
        values = frame.loc[frame.year.eq(year), "signed_return"].to_numpy(float)
        result[str(year)] = {
            "n": len(values),
            "win_rate_pct": round(100 * float(np.mean(values > 0)), 4)
                if len(values) else 0.0,
            "mean_signed_price_return_pct": round(
                100 * float(values.mean()), 6) if len(values) else 0.0,
            "sum_signed_price_return_pct": round(
                100 * float(values.sum()), 6) if len(values) else 0.0,
        }
    return result


def main() -> None:
    candles, sources = load()
    raw = pd.concat([
        sweep.features(symbol, candle, ENTRY_SPEC[0])
        for symbol, candle in candles.items()
    ], ignore_index=True)
    entries = sweep.apply_spec(raw, ENTRY_SPEC)
    immediate, delayed = enrich(entries, candles)
    variants = {}
    for threshold in (0.0, .1, .2, .3):
        variants[f"immediate_absorption_ge_{threshold}"] = (
            immediate, immediate.signal_flow_absorption.ge(threshold))
    for threshold in (0.0, .1, .2):
        variants[f"immediate_aligned_flow_ge_{threshold}"] = (
            immediate, immediate.signal_flow_aligned.ge(threshold))
    variants.update({
        "delayed_absorption0_confirmation0": (
            delayed, delayed.signal_flow_absorption.ge(0)
            & delayed.confirm_flow_aligned.ge(0)),
        "delayed_absorption01_confirmation0": (
            delayed, delayed.signal_flow_absorption.ge(.1)
            & delayed.confirm_flow_aligned.ge(0)),
        "delayed_absorption02_confirmation01": (
            delayed, delayed.signal_flow_absorption.ge(.2)
            & delayed.confirm_flow_aligned.ge(.1)),
        "delayed_flow_flip02": (delayed, delayed.flow_flip.ge(.2)),
        "delayed_absorption01_flow_flip02": (
            delayed, delayed.signal_flow_absorption.ge(.1)
            & delayed.flow_flip.ge(.2)),
    })
    results = {}
    for name, (frame, mask) in variants.items():
        resolved = sweep.add_outcomes(
            frame.loc[mask].copy(), candles,
            TARGET_PCT, STOP_PCT, MAX_BARS)
        yearly = annual(resolved)
        results[name] = {
            "entries": len(resolved), "annual": yearly,
            "positive_mean_all_2020_2024": all(
                yearly[str(year)]["mean_signed_price_return_pct"] > 0
                for year in range(2020, 2025)),
        }
    qualified = [name for name, value in results.items()
                 if value["positive_mean_all_2020_2024"]]
    report = {
        "status": "DEVELOPMENT_PASS" if qualified else "REJECTED_TAKER_FLOW",
        "ready_for_live_orders": False,
        "protocol": {
            "venue": "Binance USD-M futures",
            "flow": "5m taker-buy quote share transformed to signed imbalance",
            "base_entry": ENTRY_SPEC,
            "immediate_entry": "next 5m market-open proxy",
            "delayed_entry": "market-open proxy after one closed flow confirmation",
            "target_pct": TARGET_PCT, "stop_pct": STOP_PCT,
            "maximum_hold_bars": MAX_BARS, "collision_policy": "stop first",
            "annual_diagnostic": list(range(2020, 2027)),
            "money": "price-path diagnostic; no position-size optimization",
        },
        "raw_sweeps": len(raw), "base_entries": len(entries),
        "results": results, "qualified_2020_2024": qualified,
        "decision": (
            "No runtime change. Event-specific taker imbalance is rejected "
            "unless expectancy remains positive in every 2020-2024 year."),
        "sources": {
            **sources,
            str(Path(sweep.__file__)): digest(Path(sweep.__file__)),
            str(Path(__file__)): digest(Path(__file__)),
        },
        "limitations": [
            "Kline taker flow is aggregated and does not reconstruct queue depth.",
            "2025-2026 are previously observed diagnostic periods.",
            "A negative raw price expectancy cannot be repaired by sizing.",
        ],
    }
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({
        "status": report["status"], "base_entries": len(entries),
        "qualified": qualified,
        "annual_ranges": {name: {
            "minimum_wr": min(v["win_rate_pct"] for v in value["annual"].values()),
            "maximum_wr": max(v["win_rate_pct"] for v in value["annual"].values()),
            "negative_years": [year for year, v in value["annual"].items()
                               if v["mean_signed_price_return_pct"] <= 0],
        } for name, value in results.items()},
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
