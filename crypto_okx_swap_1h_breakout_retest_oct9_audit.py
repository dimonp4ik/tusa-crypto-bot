"""Chronological breakout-retest-reclaim audit with 1.5-2% targets.

A setup requires three causal stages: a close through a prior 24/72-hour
level, a later revisit that does not invalidate that level, and a completed
reclaim candle.  Entry is the next hourly market open.  Simultaneous setups
are collapsed to one representative per direction and hour before outcomes.
"""
from __future__ import annotations

import hashlib
import json
import pickle
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

import crypto_okx_swap_1h_idiosyncratic_router_oct9_audit as prior
import crypto_okx_swap_1h_regime_oct7_audit as base


ROOT = Path("reports/audit_2026_09_08")
MANIFEST = ROOT / "crypto_okx_swap_1h_expanded_oct8" / "manifest.json"
OUT = ROOT / "crypto_okx_swap_1h_breakout_retest_oct9.json"
SELECTION_YEARS = (2020, 2021)
VALIDATION_YEAR = 2022
CONFIRMATION_YEAR = 2023
HIDDEN_YEARS = (2024, 2025, 2026)


@dataclass(frozen=True)
class Config:
    lookback: int
    maximum_wait: int
    retest_tolerance: float
    confirmation: str

    @property
    def target_pct(self) -> float:
        return 2.0 if self.lookback == 72 else 1.5

    @property
    def name(self) -> str:
        return (f"lookback={self.lookback}|wait={self.maximum_wait}|"
                f"tolerance={self.retest_tolerance}|{self.confirmation}")


CONFIGS = tuple(
    Config(lookback, wait, tolerance, confirmation)
    for lookback in (24, 72)
    for wait in (6, 12, 24)
    for tolerance in (.0025, .005)
    for confirmation in ("body", "rejection_wick")
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def resolve_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else Path.cwd() / path


def load() -> tuple[dict, dict]:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    candles, sources = {}, {str(MANIFEST): digest(MANIFEST)}
    for symbol, item in manifest["symbols"].items():
        path = resolve_path(item["path"])
        raw = path.read_bytes()
        found = hashlib.sha256(raw).hexdigest()
        if found != item["sha256"]:
            raise ValueError(f"source hash mismatch: {symbol}")
        candle = pickle.loads(raw)
        times = [int(value) for value in candle["time"]]
        if any(right - left != 3600 for left, right in zip(times, times[1:])):
            raise ValueError(f"non-contiguous source: {symbol}")
        candles[symbol] = candle
        sources[str(path)] = found
    return candles, sources


def prepare_symbol(candle: dict) -> dict:
    open_ = np.asarray(candle["open"], dtype=float)
    high = np.asarray(candle["high"], dtype=float)
    low = np.asarray(candle["low"], dtype=float)
    close = np.asarray(candle["close"], dtype=float)
    volume = np.asarray(candle["volume_quote"], dtype=float)
    times = np.asarray(candle["time"], dtype=np.int64)
    frame = pd.DataFrame({
        "open": open_, "high": high, "low": low, "close": close,
        "volume": volume,
    })
    candle_range = frame.high - frame.low
    range_median = candle_range.shift(1).rolling(
        168, min_periods=120).median().to_numpy(float)
    volume_median = frame.volume.shift(1).rolling(
        168, min_periods=120).median().to_numpy(float)
    ema48 = frame.close.ewm(span=48, adjust=False).mean().to_numpy(float)
    ema168 = frame.close.ewm(span=168, adjust=False).mean().to_numpy(float)
    return {
        "open": open_, "high": high, "low": low, "close": close,
        "volume": volume, "times": times,
        "candle_range": candle_range.to_numpy(float),
        "range_median": range_median, "volume_median": volume_median,
        "ema48": ema48, "ema168": ema168,
        "prior_high": {
            lookback: frame.high.shift(1).rolling(
                lookback, min_periods=lookback).max().to_numpy(float)
            for lookback in (24, 72)
        },
        "prior_low": {
            lookback: frame.low.shift(1).rolling(
                lookback, min_periods=lookback).min().to_numpy(float)
            for lookback in (24, 72)
        },
    }


def symbol_setups(symbol: str, prepared: dict, config: Config) -> pd.DataFrame:
    open_ = prepared["open"]
    high = prepared["high"]
    low = prepared["low"]
    close = prepared["close"]
    volume = prepared["volume"]
    times = prepared["times"]
    candle_range = prepared["candle_range"]
    range_median = prepared["range_median"]
    volume_median = prepared["volume_median"]
    ema48 = prepared["ema48"]
    ema168 = prepared["ema168"]
    prior_high = prepared["prior_high"][config.lookback]
    prior_low = prepared["prior_low"][config.lookback]

    results = []
    minimum_break = .0015
    maximum_invalidation = .006
    for breakout in range(max(168, config.lookback), len(times) - 2):
        if (not np.isfinite(range_median[breakout])
                or not np.isfinite(volume_median[breakout])
                or range_median[breakout] <= 0 or volume_median[breakout] <= 0):
            continue
        range_ratio = candle_range[breakout] / range_median[breakout]
        volume_ratio = volume[breakout] / volume_median[breakout]
        if range_ratio < 1.2 or volume_ratio < .8:
            continue

        directions = []
        if close[breakout] >= prior_high[breakout] * (1 + minimum_break):
            directions.append((1, prior_high[breakout]))
        if close[breakout] <= prior_low[breakout] * (1 - minimum_break):
            directions.append((-1, prior_low[breakout]))
        for direction, level in directions:
            if not np.isfinite(level) or level <= 0:
                continue
            breakout_strength = direction * (close[breakout] / level - 1)
            last = min(len(times) - 2, breakout + config.maximum_wait)
            for reclaim in range(breakout + 1, last + 1):
                if direction > 0:
                    invalid = close[reclaim] < level * (1 - maximum_invalidation)
                    touched = (low[reclaim] <= level * (1 + config.retest_tolerance)
                               and low[reclaim] >= level * (1 - maximum_invalidation))
                    reclaimed = close[reclaim] > level and close[reclaim] > open_[reclaim]
                    trend = close[reclaim] > ema48[reclaim] > ema168[reclaim]
                    wick = ((min(open_[reclaim], close[reclaim]) - low[reclaim])
                            / max(high[reclaim] - low[reclaim], 1e-15))
                else:
                    invalid = close[reclaim] > level * (1 + maximum_invalidation)
                    touched = (high[reclaim] >= level * (1 - config.retest_tolerance)
                               and high[reclaim] <= level * (1 + maximum_invalidation))
                    reclaimed = close[reclaim] < level and close[reclaim] < open_[reclaim]
                    trend = close[reclaim] < ema48[reclaim] < ema168[reclaim]
                    wick = ((high[reclaim] - max(open_[reclaim], close[reclaim]))
                            / max(high[reclaim] - low[reclaim], 1e-15))
                if invalid:
                    break
                confirmation = (config.confirmation == "body" or wick >= .30)
                if not (touched and reclaimed and trend and confirmation):
                    continue
                entry_index = reclaim + 1
                stamp = pd.Timestamp(int(times[entry_index]), unit="s", tz="UTC")
                reclaim_body = direction * (close[reclaim] / open_[reclaim] - 1)
                score = (100 * breakout_strength + range_ratio + .5 * volume_ratio
                         + 2 * wick + 100 * max(0.0, reclaim_body))
                results.append({
                    "symbol": symbol,
                    "entry_index": entry_index,
                    "entry_time": int(times[entry_index]),
                    "year": int(stamp.year),
                    "month": int(stamp.year * 100 + stamp.month),
                    "direction": int(direction),
                    "route": "breakout_retest_reclaim",
                    "config": config.name,
                    "lookback": config.lookback,
                    "breakout_time": int(times[breakout]),
                    "reclaim_time": int(times[reclaim]),
                    "hours_to_retest": reclaim - breakout,
                    "level": float(level),
                    "setup_score": float(score),
                    "selected_target_pct": config.target_pct,
                })
                break
    if not results:
        return pd.DataFrame()
    return pd.DataFrame(results).drop_duplicates(
        ["symbol", "entry_time", "direction"], keep="first")


def setup_frame(prepared: dict, config: Config) -> pd.DataFrame:
    parts = [symbol_setups(symbol, data, config)
             for symbol, data in prepared.items()]
    parts = [part for part in parts if not part.empty]
    if not parts:
        return pd.DataFrame()
    frame = pd.concat(parts, ignore_index=True)
    return (frame.sort_values(
        ["entry_time", "direction", "setup_score", "symbol"],
        ascending=[True, True, False, True])
        .drop_duplicates(["entry_time", "direction"], keep="first")
        .reset_index(drop=True))


def public(item: dict) -> dict:
    return {key: value for key, value in item.items() if key != "_frame"}


def main() -> None:
    candles, sources = load()
    prepared = {symbol: prepare_symbol(candle)
                for symbol, candle in candles.items()}
    evaluated = []
    for config in CONFIGS:
        candidates = setup_frame(prepared, config)
        if candidates.empty:
            continue
        found = prior.add_outcomes(candidates, candles)
        selection = prior.evaluate(found, SELECTION_YEARS)
        selection_pass = prior.screen(
            selection, config.target_pct, minimum_n=100)
        item = {
            "config": config.name,
            "parameters": asdict(config),
            "target_pct": config.target_pct,
            "rows": len(found),
            "selection_2020_2021": selection,
            "selection_screen_pass": selection_pass,
            "_frame": found,
        }
        if selection_pass:
            validation = prior.evaluate(found, (VALIDATION_YEAR,))
            item["validation_2022"] = validation
            item["validation_screen_pass"] = prior.screen(
                validation, config.target_pct, minimum_n=40)
            if item["validation_screen_pass"]:
                confirmation = prior.evaluate(found, (CONFIRMATION_YEAR,))
                item["confirmation_2023"] = confirmation
                item["confirmation_screen_pass"] = prior.screen(
                    confirmation, config.target_pct, minimum_n=40)
        evaluated.append(item)
        print(json.dumps({
            "config": config.name, "rows": len(found),
            "selection_screen_pass": selection_pass,
            "selection": selection,
        }), flush=True)

    developed = [item for item in evaluated if item["selection_screen_pass"]]
    validated = [item for item in developed
                 if item.get("validation_screen_pass", False)]
    confirmed = [item for item in validated
                 if item.get("confirmation_screen_pass", False)]
    confirmed.sort(key=lambda item: prior.rank([
        item["selection_2020_2021"], item["validation_2022"],
        item["confirmation_2023"]]), reverse=True)

    union_frame = None
    union = None
    if confirmed:
        union_frame = pd.concat([item["_frame"] for item in confirmed],
                                ignore_index=True, sort=False)
        union_frame = (union_frame.sort_values(
            ["entry_time", "direction", "selected_target_pct", "setup_score"],
            ascending=[True, True, False, False])
            .drop_duplicates(["entry_time", "direction"], keep="first"))
        union = {
            "selection_2020_2021": prior.evaluate(union_frame, SELECTION_YEARS),
            "validation_2022": prior.evaluate(union_frame, (VALIDATION_YEAR,)),
            "confirmation_2023": prior.evaluate(union_frame, (CONFIRMATION_YEAR,)),
        }
    early_objective = bool(union and all(
        prior.objective_pass(value) for value in union.values()))
    later = None
    if early_objective:
        later = {str(year): prior.evaluate(union_frame, (year,))
                 for year in HIDDEN_YEARS}
    later_objective = bool(later and all(
        prior.objective_pass(value) for value in later.values()))
    status = ("OBJECTIVE_PASS" if later_objective
              else "REJECTED_LATER" if later is not None
              else "REJECTED_EARLY_OBJECTIVE" if confirmed
              else "REJECTED_CONFIRMATION" if validated
              else "REJECTED_VALIDATION" if developed
              else "REJECTED_SELECTION")
    evaluated.sort(key=lambda item: prior.rank([
        item["selection_2020_2021"]]), reverse=True)
    report = {
        "status": status, "ready_for_live_orders": False,
        "protocol": {
            "venue": "OKX global USDT swap", "bar": "1H",
            "signal": "breakout, later retest without invalidation, completed reclaim",
            "entry": "next-hour market open",
            "selection": list(SELECTION_YEARS), "validation": VALIDATION_YEAR,
            "confirmation": CONFIRMATION_YEAR,
            "hidden_revealed_only_after_complete_early_objective": list(HIDDEN_YEARS),
            "targets_pct": [1.5, 2.0], "stop_pct": 1.0,
            "hold_hours": {"1.5": 36, "2.0": 48},
            "episode_rule": "one strongest asset per direction and entry hour",
            "money": "3% current equity margin, fixed 10x gross",
            "fees": "excluded by user instruction",
        },
        "tested_routes": len(evaluated),
        "selection_passed": len(developed),
        "validation_passed": len(validated),
        "confirmation_passed": len(confirmed),
        "top_routes": [public(item) for item in evaluated[:12]],
        "confirmed_routes": [public(item) for item in confirmed],
        "confirmed_union": union,
        "early_objective_pass": early_objective,
        "later": later, "later_objective_pass": later_objective,
        "decision": (
            "No runtime change. The multi-stage structure must pass every "
            "chronological stage and the full economic objective."),
        "sources": {**sources, str(MANIFEST): digest(MANIFEST),
                    str(Path(__file__)): digest(Path(__file__))},
    }
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({
        "status": status, "tested_routes": len(evaluated),
        "selection_passed": len(developed),
        "validation_passed": len(validated),
        "confirmation_passed": len(confirmed),
        "top_routes": [public(item) for item in evaluated[:5]],
        "confirmed_union": union, "later": later,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
