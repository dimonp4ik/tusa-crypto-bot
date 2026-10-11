"""Chronological audit of level-aware target adaptation for the frozen router."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

import crypto_stop_cause_oct11_audit as causes


OUT = causes.ROOT / "crypto_level_aware_target_oct11.json"
TARGETS = (.5, 1.0, 1.5, 2.0)
BUFFERS = (.70, .80, .90, 1.0)
MODES = ("cap_or_skip", "skip_if_blocked")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def route(frame: pd.DataFrame, mode: str, buffer: float) -> pd.DataFrame:
    found = frame.copy()
    keep = pd.Series(True, index=found.index)
    chosen = found.selected_target_pct.astype(float).copy()
    for index, row in found.iterrows():
        room = float(row.opposing_distance_72_pct)
        original = float(row.selected_target_pct)
        if room < 0 or room >= original:
            continue
        if mode == "skip_if_blocked":
            keep.at[index] = False
            continue
        ceiling = room * buffer
        allowed = [target for target in TARGETS
                   if target <= original and target <= ceiling]
        if not allowed:
            keep.at[index] = False
        else:
            chosen.at[index] = max(allowed)
    found = found.loc[keep].copy()
    found["selected_target_pct"] = chosen.loc[found.index]
    for target in TARGETS:
        suffix = str(target).replace(".", "_")
        mask = found.selected_target_pct.eq(target)
        found.loc[mask, "signed_return"] = found.loc[mask, f"signed_{suffix}"]
        found.loc[mask, "exit_time"] = found.loc[mask, f"exit_{suffix}"]
        found.loc[mask, "target_hit"] = found.loc[mask, f"hit_{suffix}"]
        found.loc[mask, "exit_reason"] = found.loc[mask, f"reason_{suffix}"]
    return found


def dominates(candidate: dict, baseline: dict) -> bool:
    return bool(
        candidate["n"] >= 700
        and candidate["trades_per_calendar_day"] >= 2.0
        and candidate["return_pct"] >= baseline["return_pct"]
        and candidate["literal_target_hit_rate"] >= baseline["literal_target_hit_rate"]
        and candidate["dd_pct"] <= baseline["dd_pct"]
        and candidate["negative_months"] <= baseline["negative_months"]
        and candidate["minimum_month_pct"] >= baseline["minimum_month_pct"]
    )


def objective_pass(value: dict) -> bool:
    return bool(
        75 <= value["literal_target_hit_rate"] <= 95
        and value["trades_per_calendar_day"] >= 3
        and value["negative_months"] == 0
        and value["months_ge_30pct"] == value["months"]
    )


def main() -> None:
    candles, sources = causes.load_candles()
    original = pd.read_pickle(causes.ROWS)
    frame = causes.enrich(original, candles)
    baseline_2022 = causes.metrics(frame, (2022,))
    frontier = []
    eligible = []
    for mode in MODES:
        for buffer in BUFFERS:
            candidate = route(frame, mode, buffer)
            score = causes.metrics(candidate, (2022,))
            item = {
                "mode": mode,
                "level_buffer": buffer,
                "selection_2022": score,
                "dominates_baseline": dominates(score, baseline_2022),
            }
            frontier.append(item)
            if item["dominates_baseline"]:
                rank = (baseline_2022["negative_months"] - score["negative_months"],
                        score["minimum_month_pct"], score["return_pct"],
                        score["literal_target_hit_rate"], -score["dd_pct"])
                eligible.append((rank, mode, buffer, candidate, score))

    frozen = validation = test_2024 = confirmation = None
    status = "REJECTED_SELECTION"
    if eligible:
        _, mode, buffer, candidate, selection = max(eligible, key=lambda item: item[0])
        baseline_2023 = causes.metrics(frame, (2023,))
        validation = causes.metrics(candidate, (2023,))
        frozen = {
            "mode": mode,
            "level_buffer": buffer,
            "eligible_candidates": len(eligible),
            "selection_baseline_2022": baseline_2022,
            "selection_2022": selection,
        }
        status = "REJECTED_VALIDATION"
        if dominates(validation, baseline_2023):
            baseline_2024 = causes.metrics(frame, (2024,))
            test_2024 = causes.metrics(candidate, (2024,))
            status = "REJECTED_TEST_2024"
            if dominates(test_2024, baseline_2024):
                confirmation = {str(year): causes.metrics(candidate, (year,))
                                for year in (2025, 2026)}
                status = ("LEVEL_AWARE_RESEARCH_CANDIDATE"
                          if all(objective_pass(value)
                                 for value in confirmation.values())
                          else "REJECTED_OBJECTIVE")

    report = {
        "status": status,
        "ready_for_live_orders": False,
        "protocol": {
            "entry_rows": "unchanged frozen target-cascade selections",
            "level": "highest/lowest completed candle over prior 72 hours",
            "rule": "cap target below opposing level; skip when 0.5% does not fit",
            "selection": 2022,
            "validation": 2023,
            "conditional_test": 2024,
            "published_confirmation_only": [2025, 2026],
            "targets_pct": list(TARGETS),
            "buffers": list(BUFFERS),
            "market_entries_only": True,
            "money": "3% current equity margin, fixed 10x gross",
            "fees": "excluded by user instruction",
        },
        "selection_baseline_2022": baseline_2022,
        "selection_frontier": frontier,
        "frozen_before_validation": frozen,
        "validation_2023": validation,
        "test_2024": test_2024,
        "confirmation_2025_2026": confirmation,
        "sources": {
            **sources,
            str(causes.ROWS): digest(causes.ROWS),
            str(Path(causes.__file__).name): digest(Path(causes.__file__)),
            str(Path(__file__)): digest(Path(__file__)),
        },
        "limitations": [
            "The 72-hour high/low is a reproducible liquidity proxy, not resting order-book liquidity.",
            "Only literal 0.5/1/1.5/2% targets are allowed; the stop remains 1%.",
            "No result is eligible for runtime use without independent forward evidence.",
        ],
    }
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({
        "status": status,
        "selection_frontier": frontier,
        "frozen_before_validation": frozen,
        "validation_2023": validation,
        "test_2024": test_2024,
        "confirmation_2025_2026": confirmation,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
