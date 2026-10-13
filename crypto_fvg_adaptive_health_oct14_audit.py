"""Causal trailing-health gate for FVG routes with shadow outcome updates."""
from __future__ import annotations

import collections
import hashlib
import heapq
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

import crypto_fvg_causal_stop_oct14_audit as fvg


ROOT = Path("reports/audit_2026_09_08")
OUT = ROOT / "crypto_fvg_adaptive_health_oct14.json"
BASE_VARIANTS = ("base", "confirmed", "extended_confirmed")
ROUTE_MODES = ("global", "trend4", "trend_pair", "trend_direction", "session_trend")
LOOKBACKS = (20, 40, 80)
MIN_OBSERVATIONS = (10, 20)
MEAN_THRESHOLDS = (0.0, .0005)
WIN_RATE_THRESHOLDS = (.50, .55, .60)
COLD_STARTS = ("pass", "shadow_warmup")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def base_masks(frame: pd.DataFrame) -> dict[str, pd.Series]:
    confirmed = frame.closed_confirmation_count.ge(1) & frame.bos_candles_ago.le(3)
    return {
        "base": pd.Series(True, index=frame.index),
        "confirmed": confirmed,
        "extended_confirmed": confirmed & frame.extension_atr.ge(1.0),
    }


def route_key(row: dict, mode: str) -> str:
    one = bool(row["aligned_1h"])
    four = bool(row["aligned_4h"])
    if mode == "global":
        return "global"
    if mode == "trend4":
        return f"four_{int(four)}"
    pair = f"one_{int(one)}_four_{int(four)}"
    if mode == "trend_pair":
        return pair
    if mode == "trend_direction":
        return f"{pair}_{row['direction']}"
    liquid = row["session"] in ("LONDON", "NEW_YORK", "OVERLAP")
    return f"four_{int(four)}_liquid_{int(liquid)}"


def adaptive_mask(frame: pd.DataFrame, year: int, base_mask: pd.Series,
                  mode: str, lookback: int, minimum: int,
                  mean_threshold: float, wr_threshold: float,
                  cold_start: str) -> pd.Series:
    year_frame = frame[frame.year.eq(year) & base_mask].sort_values(
        ["entry_time", "symbol"], kind="stable")
    histories: dict[str, collections.deque] = collections.defaultdict(
        lambda: collections.deque(maxlen=lookback))
    pending: list[tuple[float, int, str, float]] = []
    accepted = pd.Series(False, index=frame.index)
    serial = 0
    for index, row in year_frame.iterrows():
        now = float(row.entry_time)
        while pending and pending[0][0] < now:
            _, _, key, value = heapq.heappop(pending)
            histories[key].append(value)
        key = route_key(row.to_dict(), mode)
        history = histories[key]
        if len(history) < minimum:
            healthy = cold_start == "pass"
        else:
            values = np.asarray(history, dtype=float)
            healthy = bool(values.mean() >= mean_threshold
                           and np.mean(values > 0) >= wr_threshold)
        if healthy:
            accepted.loc[index] = True
        serial += 1
        heapq.heappush(pending, (
            float(row.exit_time), serial, key, float(row.signed_price_return)))
    return accepted


def configuration_mask(frame: pd.DataFrame, bases: dict, year: int,
                       config: dict) -> pd.Series:
    return adaptive_mask(
        frame, year, bases[config["base_variant"]], config["route_mode"],
        config["lookback"], config["minimum_observations"],
        config["mean_threshold"], config["win_rate_threshold"],
        config["cold_start"],
    )


def configs() -> list[dict]:
    return [{
        "base_variant": base_variant, "route_mode": route_mode,
        "lookback": lookback, "minimum_observations": minimum,
        "mean_threshold": mean_threshold,
        "win_rate_threshold": wr_threshold, "cold_start": cold_start,
    } for (base_variant, route_mode, lookback, minimum, mean_threshold,
           wr_threshold, cold_start) in itertools.product(
               BASE_VARIANTS, ROUTE_MODES, LOOKBACKS, MIN_OBSERVATIONS,
               MEAN_THRESHOLDS, WIN_RATE_THRESHOLDS, COLD_STARTS)]


def compact(value: dict) -> dict:
    keys = ("n", "wr", "return_pct", "dd_pct", "max_loss_streak",
            "negative_months", "trades_per_calendar_day",
            "mean_signed_price_return_pct")
    return {key: value[key] for key in keys}


def main() -> None:
    frame, coverage = fvg.load()
    bases = base_masks(frame)
    development = []
    for config in configs():
        mask = configuration_mask(frame, bases, 2023, config)
        value = fvg.exact_year(frame[mask], 2023, coverage)
        if value["n"] >= 100:
            development.append({"config": config, "development": value})
    development.sort(key=lambda item: fvg.score(item["development"]), reverse=True)
    shortlist = development[:100]
    for item in shortlist:
        mask = configuration_mask(frame, bases, 2024, item["config"])
        item["validation"] = fvg.exact_year(frame[mask], 2024, coverage)
        item["qualified_before_2025"] = bool(
            item["development"]["n"] >= 100
            and item["validation"]["n"] >= 100
            and item["development"]["return_pct"] > 0
            and item["validation"]["return_pct"] > 0
            and item["development"]["mean_signed_price_return_pct"] > 0
            and item["validation"]["mean_signed_price_return_pct"] > 0)
        item["minimum_pretest_return_pct"] = min(
            item["development"]["return_pct"],
            item["validation"]["return_pct"])
        item["minimum_pretest_wr"] = min(
            item["development"]["wr"], item["validation"]["wr"])
    qualified = [item for item in shortlist if item["qualified_before_2025"]]
    qualified.sort(key=lambda item: (
        item["minimum_pretest_return_pct"], item["minimum_pretest_wr"],
        -max(item["development"]["dd_pct"], item["validation"]["dd_pct"])),
        reverse=True)
    frozen = qualified[0] if qualified else None
    exact = checks = None
    if frozen:
        exact = {
            "2023": frozen["development"], "2024": frozen["validation"],
        }
        for year in (2025, 2026):
            mask = configuration_mask(frame, bases, year, frozen["config"])
            exact[str(year)] = fvg.exact_year(frame[mask], year, coverage)
        checks = {
            year: bool(value["n"] >= 100 and 75 <= value["wr"] <= 95
                       and value["trades_per_calendar_day"] >= 3
                       and value["negative_months"] == 0
                       and value["dd_pct"] <= 5
                       and value["economic_gate"]["passes"])
            for year, value in exact.items()
        }
    status = ("OBJECTIVE_PASS" if checks and all(checks.values())
              else "REJECTED_FORWARD" if frozen else "REJECTED_PRETEST")
    report = {
        "status": status, "ready_for_live_orders": False,
        "protocol": {
            "health_information": (
                "only shadow signals whose exit_time is strictly before the new entry"),
            "health_updates": (
                "all causally closed candidate outcomes, including signals not traded; "
                "this is implementable as a shadow route monitor"),
            "year_reset": "health histories restart at each historical slice",
            "development": 2023, "validation": 2024,
            "holdout": 2025, "observed": 2026,
            "position": "3% current-equity margin, fixed 10x gross",
            "orders": "market-entry records only",
            "grid": {
                "base_variants": list(BASE_VARIANTS),
                "route_modes": list(ROUTE_MODES),
                "lookbacks": list(LOOKBACKS),
                "minimum_observations": list(MIN_OBSERVATIONS),
                "mean_thresholds": list(MEAN_THRESHOLDS),
                "win_rate_thresholds": list(WIN_RATE_THRESHOLDS),
                "cold_starts": list(COLD_STARTS),
            },
        },
        "tested_development_configurations": len(development),
        "validation_shortlist": len(shortlist),
        "qualified_before_2025": len(qualified),
        "frozen_before_2025": frozen,
        "exact_frozen_portfolio": exact,
        "objective_checks": checks,
        "top_pretest": [{
            "config": item["config"],
            "development": compact(item["development"]),
            "validation": compact(item["validation"]),
            "minimum_pretest_return_pct": item["minimum_pretest_return_pct"],
            "minimum_pretest_wr": item["minimum_pretest_wr"],
        } for item in qualified[:20]],
        "decision": (
            "Do not add adaptive health gating to runtime unless its frozen "
            "configuration survives both later regimes and the full objective."),
        "sources": {
            str(fvg.SOURCE): digest(fvg.SOURCE),
            str(Path(fvg.__file__)): digest(Path(fvg.__file__)),
            str(Path(__file__)): digest(Path(__file__)),
        },
        "limitations": [
            "The monitor assumes every rejected route remains shadow-scored after its exit.",
            "Historical slices cover roughly six months per labeled year.",
            "A year reset is conservative and reduces initial trade frequency.",
        ],
    }
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({
        "status": status, "tested": len(development),
        "qualified": len(qualified), "frozen": frozen,
        "exact": exact, "objective_checks": checks,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
