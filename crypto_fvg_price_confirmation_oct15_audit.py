"""Causal closed-bar price confirmation audit for the historical FVG route."""
from __future__ import annotations

import hashlib
import json
import pickle
from pathlib import Path
from statistics import median

import numpy as np
import pandas as pd

import crypto_fvg_causal_stop_oct14_audit as fvg


ROOT = Path("reports/audit_2026_09_08")
OUT = ROOT / "crypto_fvg_price_confirmation_oct15.json"
ENDPOINT = {
    2023: 1690761600000,
    2024: 1722384000000,
    2025: 1753920000000,
    2026: 1787702400000,
}
CONFIGS = (
    {"name": "wick20_close60", "wick": .20, "close": .60},
    {"name": "wick35_close60", "wick": .35, "close": .60},
    {"name": "wick50_close60", "wick": .50, "close": .60},
    {"name": "wick20_close75", "wick": .20, "close": .75},
    {"name": "wick35_close75", "wick": .35, "close": .75},
    {"name": "wick50_close75", "wick": .50, "close": .75},
    {"name": "wick35_close60_body", "wick": .35, "close": .60,
     "body": 0.0},
    {"name": "wick35_close60_bos3", "wick": .35, "close": .60,
     "bos_age": 3},
    {"name": "wick35_close60_volume15", "wick": .35, "close": .60,
     "volume": 1.5},
    {"name": "sweep8_close60", "sweep": True, "close": .60},
    {"name": "sweep8_wick20", "sweep": True, "wick": .20},
    {"name": "sweep8_wick20_bos3", "sweep": True, "wick": .20,
     "bos_age": 3},
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def cache_path(symbol: str, year: int) -> Path:
    return Path("backtest_cache") / (
        f"{symbol}_15min_18000_end{ENDPOINT[year]}.pkl")


def reconstruct(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, str]]:
    output = frame.copy()
    for column in ("actual_wick", "actual_close_location", "actual_body",
                   "actual_volume_ratio", "actual_range_atr"):
        output[column] = np.nan
    output["actual_sweep8"] = False
    sources = {}
    missing = []
    for (year, symbol), indices in output.groupby(["year", "symbol"]).groups.items():
        path = cache_path(str(symbol), int(year))
        if not path.exists():
            raise FileNotFoundError(path)
        raw = path.read_bytes()
        sources[str(path)] = hashlib.sha256(raw).hexdigest()
        candle = pickle.loads(raw)
        lookup = {int(timestamp): i for i, timestamp in enumerate(candle["time"])}
        for index in indices:
            row = output.loc[index]
            entry_time = int(float(row.entry_time))
            entry_i = lookup.get(entry_time)
            if entry_i is None or entry_i < 21:
                missing.append((str(symbol), entry_time))
                continue
            signal = entry_i-1
            if int(candle["time"][signal]) != entry_time-900:
                missing.append((str(symbol), entry_time))
                continue
            open_ = float(candle["open"][signal])
            high = float(candle["high"][signal])
            low = float(candle["low"][signal])
            close = float(candle["close"][signal])
            span = max(high-low, 1e-12)
            direction = 1 if str(row.direction) == "LONG" else -1
            wick = ((min(open_, close)-low)/span if direction > 0
                    else (high-max(open_, close))/span)
            close_location = ((close-low)/span if direction > 0
                              else (high-close)/span)
            body = direction*(close-open_)/span
            prior_low = min(float(candle["low"][j])
                            for j in range(signal-8, signal))
            prior_high = max(float(candle["high"][j])
                             for j in range(signal-8, signal))
            sweep = ((low < prior_low and close > prior_low) if direction > 0
                     else (high > prior_high and close < prior_high))
            volumes = [float(candle["volume"][j])
                       for j in range(signal-20, signal)]
            volume_base = median(volumes)
            true_ranges = []
            for j in range(signal-14, signal):
                previous = float(candle["close"][j-1])
                true_ranges.append(max(
                    float(candle["high"][j])-float(candle["low"][j]),
                    abs(float(candle["high"][j])-previous),
                    abs(float(candle["low"][j])-previous)))
            scale = sum(true_ranges)/len(true_ranges)
            output.at[index, "actual_wick"] = wick
            output.at[index, "actual_close_location"] = close_location
            output.at[index, "actual_body"] = body
            output.at[index, "actual_sweep8"] = sweep
            output.at[index, "actual_volume_ratio"] = (
                float(candle["volume"][signal])/volume_base
                if volume_base > 0 else np.nan)
            output.at[index, "actual_range_atr"] = span/scale if scale > 0 else np.nan
    if missing:
        raise ValueError(f"closed-bar reconstruction missing {len(missing)} rows: {missing[:5]}")
    return output, sources


def mask(frame: pd.DataFrame, config: dict) -> pd.Series:
    value = pd.Series(True, index=frame.index)
    if "wick" in config:
        value &= frame.actual_wick.ge(config["wick"])
    if "close" in config:
        value &= frame.actual_close_location.ge(config["close"])
    if "body" in config:
        value &= frame.actual_body.gt(config["body"])
    if config.get("sweep"):
        value &= frame.actual_sweep8
    if "bos_age" in config:
        value &= frame.bos_candles_ago.le(config["bos_age"])
    if "volume" in config:
        value &= frame.actual_volume_ratio.ge(config["volume"])
    return value.fillna(False)


def acceptable(value: dict, minimum: int) -> bool:
    return bool(value["n"] >= minimum
                and value["trades_per_calendar_day"] >= .75
                and value["return_pct"] > 0
                and value["mean_signed_price_return_pct"] > 0
                and value["dd_pct"] <= 12)


def main() -> None:
    frame, coverage = fvg.load()
    frame, cache_sources = reconstruct(frame)
    development = []
    for config in CONFIGS:
        selected = frame[mask(frame, config)]
        value = fvg.exact_year(selected, 2023, coverage)
        development.append({"config": config, "development_2023": value})
    development.sort(key=lambda item: (
        item["development_2023"]["return_pct"] > 0,
        item["development_2023"]["return_pct"],
        item["development_2023"]["wr"],
        -item["development_2023"]["dd_pct"]), reverse=True)
    eligible = []
    for item in development:
        if not acceptable(item["development_2023"], 140):
            continue
        selected = frame[mask(frame, item["config"])]
        item["validation_2024"] = fvg.exact_year(selected, 2024, coverage)
        if acceptable(item["validation_2024"], 140):
            item["minimum_pretest_return_pct"] = min(
                item["development_2023"]["return_pct"],
                item["validation_2024"]["return_pct"])
            eligible.append(item)
    eligible.sort(key=lambda item: (
        item["minimum_pretest_return_pct"],
        min(item["development_2023"]["wr"], item["validation_2024"]["wr"]),
        -max(item["development_2023"]["dd_pct"],
             item["validation_2024"]["dd_pct"])), reverse=True)
    frozen = eligible[0] if eligible else None
    exact = checks = None
    if frozen:
        selected = frame[mask(frame, frozen["config"])]
        exact = {
            "2023": frozen["development_2023"],
            "2024": frozen["validation_2024"],
            "2025": fvg.exact_year(selected, 2025, coverage),
            "2026": fvg.exact_year(selected, 2026, coverage),
        }
        checks = {year: bool(
            value["n"] >= 100 and 75 <= value["wr"] <= 95
            and value["trades_per_calendar_day"] >= 3
            and value["negative_months"] == 0
            and value["dd_pct"] <= 5
            and value["economic_gate"]["passes"])
            for year, value in exact.items()}
    status = ("OBJECTIVE_PASS" if checks and all(checks.values())
              else "REJECTED_FORWARD" if frozen else "REJECTED_PRETEST")
    source_hash = digest(fvg.SOURCE)
    report = {
        "status": status, "ready_for_live_orders": False,
        "decision": "No runtime change unless every chronological objective passes.",
        "protocol": {
            "mechanism": "actual last fully closed 15m bar before FVG entry",
            "development": 2023, "validation": 2024,
            "holdout": 2025, "observed": 2026,
            "entry": "existing market-entry record; confirmation uses entry_time-900s",
            "targets": "existing literal 0.5-2% FVG targets",
            "position": "3% current-equity margin, fixed 10x gross",
            "configs_predeclared": len(CONFIGS),
        },
        "rows": len(frame), "coverage_days": coverage,
        "development": development,
        "qualified_before_2025": len(eligible),
        "frozen_before_2025": frozen,
        "exact_frozen_portfolio": exact,
        "objective_checks": checks,
        "sources": {
            str(fvg.SOURCE): source_hash,
            str(Path(fvg.__file__)): digest(Path(fvg.__file__)),
            str(Path(__file__)): digest(Path(__file__)),
            **cache_sources,
        },
        "limitations": [
            "The underlying FVG generator predates this filter audit.",
            "Binance spot candles are a structural proxy, not OKX X-Perp fills.",
            "Closed-bar confirmation can filter entries but cannot add frequency.",
        ],
    }
    OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"status": status, "qualified": len(eligible),
                      "frozen": frozen, "exact": exact,
                      "checks": checks}, indent=2))


if __name__ == "__main__":
    main()
