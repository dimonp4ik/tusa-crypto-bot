"""Causal family health for the literal geometry 0.5%-to-1% ladder."""
from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

import crypto_5m_regime_router_literal_oct6_audit as base
import crypto_futures_trade_flow_health_gate_oct7_audit as health
import crypto_literal_geometry_ladder_oct7_audit as geometry_ladder


OUT = Path("reports/audit_2026_09_08/crypto_literal_geometry_family_health_oct7.json")
REVIEW_DAYS = (1, 3, 7)
LOOKBACK_DAYS = (14, 30, 60, 90)
MINIMUMS = (20, 40, 80)
HIT_MINS = (.60, .62, .64, .66)
MEAN_MINS = (-.0005, -.00025, 0.0)


def features(frame: pd.DataFrame, review_days: int,
             lookback_days: int) -> pd.DataFrame:
    source = frame.sort_values(["entry_time", "quality", "symbol"],
                               ascending=[True, False, True]).reset_index(drop=True)
    seconds = review_days*86400
    starts = (source.entry_time.to_numpy(np.int64)//seconds)*seconds
    unique_starts = np.unique(starts)
    result = source.copy()
    for family in base.FAMILIES:
        flag = f"family_{family}"
        prior = source[source[flag].gt(0)].sort_values("exit_time")
        exits = prior.exit_time.to_numpy(float)
        returns = prior.signed_price_return.to_numpy(float)
        hits = prior.target_hit.to_numpy(float)
        cumulative_returns = np.r_[0.0, np.cumsum(returns)]
        cumulative_hits = np.r_[0.0, np.cumsum(hits)]
        mapping = {}
        for start in unique_starts:
            left = int(np.searchsorted(
                exits, start-lookback_days*86400, side="left"))
            right = int(np.searchsorted(exits, start, side="left"))
            n = right-left
            mapping[int(start)] = (
                n,
                (cumulative_returns[right]-cumulative_returns[left])/n
                if n else np.nan,
                (cumulative_hits[right]-cumulative_hits[left])/n
                if n else np.nan,
            )
        values = [mapping[int(start)] for start in starts]
        result[f"health_n_{family}"] = [value[0] for value in values]
        result[f"health_mean_{family}"] = [value[1] for value in values]
        result[f"health_hit_{family}"] = [value[2] for value in values]
    return result


def apply(frame: pd.DataFrame, spec: dict) -> pd.DataFrame:
    allowed = np.zeros(len(frame), dtype=bool)
    for family in base.FAMILIES:
        member = frame[f"family_{family}"].gt(0).to_numpy()
        ready = frame[f"health_n_{family}"].ge(
            spec["minimum_completed"]).to_numpy()
        healthy = (frame[f"health_mean_{family}"].ge(spec["mean_min"])
                   & frame[f"health_hit_{family}"].ge(spec["hit_min"])).to_numpy()
        allowed |= member & (~ready | healthy)
    return frame.loc[allowed].copy()


def selection_ok(value: dict) -> bool:
    return bool(value["n"] >= 1200 and value["return_pct"] > 0
                and value["literal_target_hit_rate"] >= 68
                and value["dd_pct"] <= 6
                and value["trades_per_calendar_day"] >= 3)


def validation_ok(value: dict) -> bool:
    return bool(value["n"] >= 700 and value["return_pct"] > 0
                and value["literal_target_hit_rate"] >= 65
                and value["dd_pct"] <= 7
                and value["trades_per_calendar_day"] >= 2)


def main() -> None:
    rows = pd.read_pickle(geometry_ladder.ROWS_OUT, compression="gzip")
    cache = {}
    selection = []
    for review, lookback in itertools.product(REVIEW_DAYS, LOOKBACK_DAYS):
        featured = features(rows, review, lookback)
        cache[(review, lookback)] = featured
        for minimum, hit_min, mean_min in itertools.product(
                MINIMUMS, HIT_MINS, MEAN_MINS):
            spec = {"review_days": review, "lookback_days": lookback,
                    "minimum_completed": minimum,
                    "hit_min": hit_min, "mean_min": mean_min}
            value = health.evaluate(apply(featured, spec), 2022)
            if selection_ok(value):
                selection.append({"spec": spec, "selection_2022": value})
    validated = []
    for item in selection:
        spec = item["spec"]
        chosen = apply(cache[(spec["review_days"], spec["lookback_days"])], spec)
        value = health.evaluate(chosen, 2023)
        if validation_ok(value):
            candidate = {**item, "validation_2023": value}
            candidate["score"] = round(
                min(item["selection_2022"]["return_pct"], value["return_pct"])
                - 2*max(item["selection_2022"]["dd_pct"], value["dd_pct"])
                + min(item["selection_2022"]["trades_per_calendar_day"],
                      value["trades_per_calendar_day"]), 8)
            validated.append(candidate)
    validated.sort(key=lambda item: item["score"], reverse=True)
    frozen = validated[0] if validated else None
    test_2024 = later = checks = None
    if frozen:
        spec = frozen["spec"]
        chosen = apply(cache[(spec["review_days"], spec["lookback_days"])], spec)
        test_2024 = health.evaluate(chosen, 2024)
        if validation_ok(test_2024):
            later = {str(year): health.evaluate(chosen, year)
                     for year in (2025, 2026)}
            checks = {year: validation_ok(value)
                      for year, value in later.items()}
    report = {
        "status": ("RESEARCH_ONLY_NEEDS_NEW_FORWARD"
                   if checks and all(checks.values()) else "REJECTED"),
        "protocol": {
            "entry": "frozen literal geometry router",
            "exit": {"near_pct": .5, "far_pct": 1.0,
                     "lock_fraction": .75, "max_bars": 24,
                     "initial_stop_pct": 1.0},
            "family_selection": 2022, "family_validation_and_ranking": 2023,
            "untouched_test": 2024, "observed": [2025, 2026],
            "families": list(base.FAMILIES),
            "money": "3% equity margin, fixed 10x gross",
        },
        "qualified_2022": len(selection), "qualified_2023": len(validated),
        "frozen_before_2024": frozen, "test_2024": test_2024,
        "later": later, "later_checks": checks,
        "limitations": [
            "The family gate was proposed after observing the entry model's later history.",
            "All historical results are development evidence for this combination.",
            "A pass still requires genuinely new direct X-Perp shadow evidence.",
        ],
    }
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
