"""Causal per-family health routing for the material-move flow ladder."""
from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

import crypto_futures_trade_flow_health_gate_oct7_audit as health
import crypto_futures_trade_flow_regime_oct7_audit as flow


OUT = Path("reports/audit_2026_09_08/crypto_futures_trade_flow_family_health_oct7.json")
REVIEW_DAYS = (1, 3, 7)
LOOKBACK_DAYS = (14, 30, 60, 90)
MINIMUMS = (20, 40, 80)
HIT_MINS = (.60, .62, .64, .66)
MEAN_MINS = (-.0005, -.00025, 0.0)


def add_family_health(frame: pd.DataFrame, review_days: int,
                      lookback_days: int) -> pd.DataFrame:
    source = frame.sort_values(["entry_time", "quality", "symbol"],
                               ascending=[True, False, True]).reset_index(drop=True)
    seconds = review_days*86400
    starts = (source.entry_time.to_numpy(np.int64)//seconds)*seconds
    unique_starts = np.unique(starts)
    result = source.copy()
    for family in flow.FAMILIES:
        flag = f"flow_family_{family}"
        prior = source[source[flag].gt(0)].sort_values("exit_time")
        exits = prior.exit_time.to_numpy(float)
        returns = prior.signed_price_return.to_numpy(float)
        hits = prior.target_hit.to_numpy(float)
        return_cumulative = np.r_[0.0, np.cumsum(returns)]
        hit_cumulative = np.r_[0.0, np.cumsum(hits)]
        mapping = {}
        for start in unique_starts:
            left = int(np.searchsorted(
                exits, start-lookback_days*86400, side="left"))
            right = int(np.searchsorted(exits, start, side="left"))
            n = right-left
            mapping[int(start)] = (
                n,
                (return_cumulative[right]-return_cumulative[left])/n
                if n else np.nan,
                (hit_cumulative[right]-hit_cumulative[left])/n
                if n else np.nan,
            )
        values = [mapping[int(start)] for start in starts]
        result[f"health_n_{family}"] = [value[0] for value in values]
        result[f"health_mean_{family}"] = [value[1] for value in values]
        result[f"health_hit_{family}"] = [value[2] for value in values]
    return result


def apply_gate(frame: pd.DataFrame, spec: dict) -> pd.DataFrame:
    allowed = np.zeros(len(frame), dtype=bool)
    for family in flow.FAMILIES:
        member = frame[f"flow_family_{family}"].gt(0).to_numpy()
        ready = frame[f"health_n_{family}"].ge(
            spec["minimum_completed"]).to_numpy()
        healthy = (frame[f"health_mean_{family}"].ge(spec["mean_min"])
                   & frame[f"health_hit_{family}"].ge(spec["hit_min"])).to_numpy()
        allowed |= member & (~ready | healthy)
    return frame.loc[allowed].copy()


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
    for review, lookback in itertools.product(REVIEW_DAYS, LOOKBACK_DAYS):
        featured = add_family_health(base, review, lookback)
        cache[(review, lookback)] = featured
        for minimum, hit_min, mean_min in itertools.product(
                MINIMUMS, HIT_MINS, MEAN_MINS):
            spec = {"review_days": review, "lookback_days": lookback,
                    "minimum_completed": minimum,
                    "hit_min": hit_min, "mean_min": mean_min}
            value = health.evaluate(apply_gate(featured, spec), 2022)
            if selection_ok(value):
                selection.append({"spec": spec, "selection_2022": value})
    selection.sort(key=lambda item: (
        item["selection_2022"]["return_pct"]
        - 2*item["selection_2022"]["dd_pct"],
        item["selection_2022"]["literal_target_hit_rate"],
        item["selection_2022"]["trades_per_calendar_day"]), reverse=True)
    validated = []
    for item in selection:
        spec = item["spec"]
        chosen = apply_gate(cache[(spec["review_days"],
                                   spec["lookback_days"])], spec)
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
    test_2024 = test_pass = later = None
    if frozen:
        spec = frozen["spec"]
        chosen = apply_gate(cache[(spec["review_days"],
                                   spec["lookback_days"])], spec)
        test_2024 = health.evaluate(chosen, 2024)
        test_pass = validation_ok(test_2024)
        if test_pass:
            later = {str(year): health.evaluate(chosen, year)
                     for year in (2025, 2026)}
    status = ("RESEARCH_CANDIDATE_AWAITING_VENUE_FORWARD"
              if later and all(validation_ok(value) for value in later.values())
              else "REJECTED")
    report = {
        "status": status,
        "protocol": {
            "selection": 2022, "validation_and_ranking": 2023,
            "untouched_test": 2024,
            "test_if_validated": [2025, 2026],
            "families": list(flow.FAMILIES),
            "information": "family shadow exits before fixed review boundary",
            "exit": {"near_pct": .5, **health.EXIT,
                     "initial_stop_pct": 1.0},
            "money": "3% equity margin, fixed 10x gross",
        },
        "grid_size": len(REVIEW_DAYS)*len(LOOKBACK_DAYS)*len(MINIMUMS)
                     *len(HIT_MINS)*len(MEAN_MINS),
        "base_pretest": base_results,
        "qualified_on_2022": len(selection),
        "qualified_on_2023": len(validated),
        "frozen_before_test": frozen,
        "test_2024": test_2024, "test_pass": test_pass,
        "later_years": later, "selection_frontier": selection[:20],
        "limitations": [
            "The finite family-health grid still creates 2022 selection risk.",
            "Paused families require continued hypothetical outcome tracking.",
            "A historical pass still requires direct X-Perp shadow evidence.",
        ],
    }
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({key: report[key] for key in (
        "status", "qualified_on_2022", "qualified_on_2023",
        "frozen_before_test", "test_2024", "test_pass",
        "later_years")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
