"""Causal short-horizon loss brake for the material-move flow ladder."""
from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

import crypto_futures_trade_flow_health_gate_oct7_audit as health


OUT = Path("reports/audit_2026_09_08/crypto_futures_trade_flow_loss_brake_oct7.json")
REVIEW_HOURS = (1, 3, 6)
LOOKBACK_HOURS = (6, 12, 24, 48, 72, 168)
MINIMUMS = (4, 8, 16)
SUM_FLOORS = (-.02, -.015, -.01, -.005, 0.0)
GROUPS = ("global", "direction", "symbol")


def add_brake_features(frame: pd.DataFrame, review_hours: int,
                       lookback_hours: int, group: str) -> pd.DataFrame:
    source = frame.sort_values(["entry_time", "quality", "symbol"],
                               ascending=[True, False, True]).reset_index(drop=True)
    seconds = review_hours*3600
    source["review_start"] = (
        source.entry_time.astype("int64") // seconds) * seconds
    if group == "global":
        source["brake_group"] = "all"
    elif group == "direction":
        source["brake_group"] = source.direction.astype(str)
    elif group == "symbol":
        source["brake_group"] = source.symbol.astype(str)
    else:
        raise ValueError(group)
    histories = {}
    for bucket, prior in source.groupby("brake_group", sort=False, observed=True):
        ordered = prior.sort_values("exit_time")
        exits = ordered.exit_time.to_numpy(float)
        cumulative = np.r_[0.0, np.cumsum(
            ordered.signed_price_return.to_numpy(float))]
        histories[bucket] = (exits, cumulative)
    pieces = []
    for (start, bucket), current in source.groupby(
            ["review_start", "brake_group"], sort=True, observed=True):
        exits, cumulative = histories[bucket]
        left = int(np.searchsorted(
            exits, start-lookback_hours*3600, side="left"))
        right = int(np.searchsorted(exits, start, side="left"))
        found = current.copy()
        found["brake_n"] = right-left
        found["brake_sum"] = cumulative[right]-cumulative[left]
        pieces.append(found)
    return pd.concat(pieces, ignore_index=True).sort_values(
        ["entry_time", "quality", "symbol"], ascending=[True, False, True])


def apply_brake(frame: pd.DataFrame, spec: dict) -> pd.DataFrame:
    ready = frame.brake_n.ge(spec["minimum_completed"])
    safe = frame.brake_sum.ge(spec["sum_floor"])
    return frame.loc[~ready | safe].copy()


def selection_ok(value: dict) -> bool:
    return bool(value["n"] >= 900 and value["return_pct"] > 0
                and value["literal_target_hit_rate"] >= 67
                and value["dd_pct"] <= 5
                and value["trades_per_calendar_day"] >= 2.5)


def validation_ok(value: dict) -> bool:
    return bool(value["n"] >= 750 and value["return_pct"] > 0
                and value["literal_target_hit_rate"] >= 65
                and value["dd_pct"] <= 7
                and value["trades_per_calendar_day"] >= 2.0)


def main() -> None:
    base = pd.read_pickle(health.BASE_ROWS_OUT, compression="gzip")
    base_results = {str(year): health.evaluate(base, year)
                    for year in (2022, 2023, 2024)}
    cache = {}
    selection = []
    for group, review, lookback in itertools.product(
            GROUPS, REVIEW_HOURS, LOOKBACK_HOURS):
        if lookback < review:
            continue
        featured = add_brake_features(base, review, lookback, group)
        cache[(group, review, lookback)] = featured
        for minimum, floor in itertools.product(MINIMUMS, SUM_FLOORS):
            spec = {"group": group, "review_hours": review,
                    "lookback_hours": lookback,
                    "minimum_completed": minimum, "sum_floor": floor}
            value = health.evaluate(apply_brake(featured, spec), 2022)
            if selection_ok(value):
                selection.append({"spec": spec, "selection_2022": value})
    selection.sort(key=lambda item: (
        item["selection_2022"]["return_pct"]
        - 2*item["selection_2022"]["dd_pct"],
        item["selection_2022"]["literal_target_hit_rate"],
        item["selection_2022"]["trades_per_calendar_day"]), reverse=True)
    frozen = selection[0] if selection else None
    validation = checks = later = None
    if frozen:
        spec = frozen["spec"]
        chosen = apply_brake(cache[(spec["group"], spec["review_hours"],
                                    spec["lookback_hours"])], spec)
        validation = {str(year): health.evaluate(chosen, year)
                      for year in (2023, 2024)}
        checks = {year: validation_ok(value)
                  for year, value in validation.items()}
        if all(checks.values()):
            later = {str(year): health.evaluate(chosen, year)
                     for year in (2025, 2026)}
    status = ("RESEARCH_CANDIDATE_AWAITING_VENUE_FORWARD"
              if later and all(validation_ok(value) for value in later.values())
              else "REJECTED")
    report = {
        "status": status,
        "protocol": {
            "selection": "2022 only", "validation": [2023, 2024],
            "test_if_validated": [2025, 2026],
            "information": "only shadow exits before each review boundary",
            "exit": {"near_pct": .5, **health.EXIT,
                     "initial_stop_pct": 1.0},
            "money": "3% equity margin, fixed 10x gross",
        },
        "grid_size": len(GROUPS)*len(REVIEW_HOURS)*len(LOOKBACK_HOURS)
                     *len(MINIMUMS)*len(SUM_FLOORS),
        "base_pretest": base_results,
        "qualified_on_2022": len(selection),
        "frozen_before_validation": frozen,
        "validation": validation, "validation_checks": checks,
        "later_years": later, "selection_frontier": selection[:20],
        "limitations": [
            "The finite brake grid still creates 2022 selection risk.",
            "The brake requires recording hypothetical exits while paused.",
            "A historical pass still requires direct X-Perp shadow evidence.",
        ],
    }
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({key: report[key] for key in (
        "status", "qualified_on_2022", "frozen_before_validation",
        "validation", "validation_checks", "later_years")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
