"""Chronological audit of stop geometry for the frozen OKX one-hour router."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

import crypto_stop_cause_oct11_audit as causes
import crypto_okx_swap_1h_target_cascade_oct7_audit as cascade


OUT = causes.ROOT / "crypto_stop_geometry_oct11.json"
POLICIES = (
    {"name": "hard_1_25", "kind": "hard", "stop_pct": 1.25},
    {"name": "hard_1_50", "kind": "hard", "stop_pct": 1.50},
    {"name": "hard_2_00", "kind": "hard", "stop_pct": 2.00},
    {"name": "close_1_emergency_1_5", "kind": "close_confirm",
     "close_pct": 1.0, "stop_pct": 1.50},
    {"name": "close_1_emergency_2", "kind": "close_confirm",
     "close_pct": 1.0, "stop_pct": 2.00},
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def resolve_row(row: dict, candle: dict, policy: dict) -> tuple[float, int, bool, int]:
    open_ = np.asarray(candle["open"], dtype=float)
    high = np.asarray(candle["high"], dtype=float)
    low = np.asarray(candle["low"], dtype=float)
    close = np.asarray(candle["close"], dtype=float)
    times = np.asarray(candle["time"], dtype=np.int64)
    first = int(row["entry_index"])
    direction = int(row["direction"])
    target_pct = float(row["selected_target_pct"])
    hold = int(cascade.HOLD[target_pct])
    entry = float(open_[first])
    target = entry * (1 + direction * target_pct / 100)
    stop = entry * (1 - direction * float(policy["stop_pct"]) / 100)
    last = first
    end = min(len(times), first + hold)
    for index in range(first, end):
        if index > first and times[index] - times[index - 1] != 3600:
            break
        last = index
        hard_hit = low[index] <= stop if direction > 0 else high[index] >= stop
        target_hit = high[index] >= target if direction > 0 else low[index] <= target
        if hard_hit:
            fill = min(open_[index], stop) if direction > 0 else max(open_[index], stop)
            return direction * (fill / entry - 1), int(times[index] + 3600), False, 0
        if target_hit:
            return target_pct / 100, int(times[index] + 3600), True, 1
        if policy["kind"] == "close_confirm":
            close_loss = direction * (close[index] / entry - 1)
            if close_loss <= -float(policy["close_pct"]) / 100:
                next_index = index + 1
                if next_index < len(times) and times[next_index] - times[index] == 3600:
                    fill = float(open_[next_index])
                    return (direction * (fill / entry - 1), int(times[next_index]),
                            False, 0)
                return close_loss, int(times[index] + 3600), False, 0
    value = direction * (float(close[last]) / entry - 1)
    value = min(target_pct / 100, value)
    return value, int(times[last] + 3600), False, 2


def route(frame: pd.DataFrame, candles: dict, policy: dict) -> pd.DataFrame:
    found = frame.copy()
    prepared = {
        symbol: {key: np.asarray(candle[key], dtype=(np.int64 if key == "time" else float))
                 for key in ("open", "high", "low", "close", "time")}
        for symbol, candle in candles.items()
    }
    signed, exits, hits, reasons = [], [], [], []
    for row in found.to_dict("records"):
        value = resolve_row(row, prepared[str(row["symbol"])], policy)
        signed.append(value[0])
        exits.append(value[1])
        hits.append(value[2])
        reasons.append(value[3])
    found["signed_return"] = signed
    found["exit_time"] = exits
    found["target_hit"] = hits
    found["exit_reason"] = reasons
    return found


def selection_pass(candidate: dict, baseline: dict) -> bool:
    return bool(
        candidate["n"] >= 700
        and candidate["trades_per_calendar_day"] >= 2.0
        and candidate["return_pct"] >= baseline["return_pct"]
        and candidate["literal_target_hit_rate"] >= baseline["literal_target_hit_rate"]
        and candidate["dd_pct"] <= 7.0
        and candidate["negative_months"] <= baseline["negative_months"]
        and candidate["minimum_month_pct"] >= baseline["minimum_month_pct"]
    )


def validation_pass(candidate: dict, baseline: dict) -> bool:
    return bool(
        candidate["n"] >= 500
        and candidate["trades_per_calendar_day"] >= 2.0
        and candidate["return_pct"] >= baseline["return_pct"]
        and candidate["literal_target_hit_rate"] >= baseline["literal_target_hit_rate"]
        and candidate["dd_pct"] <= 7.0
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
    frame = pd.read_pickle(causes.ROWS)
    baseline_2022 = causes.metrics(frame, (2022,))
    candidates = []
    frontier = []
    routed = {}
    for policy in POLICIES:
        candidate = route(frame, candles, policy)
        routed[policy["name"]] = candidate
        score = causes.metrics(candidate, (2022,))
        passed = selection_pass(score, baseline_2022)
        item = {"policy": policy, "selection_2022": score,
                "selection_pass": passed}
        frontier.append(item)
        if passed:
            rank = (score["return_pct"] - 2 * score["dd_pct"]
                    + .2 * score["minimum_month_pct"]
                    + .1 * score["literal_target_hit_rate"])
            candidates.append((rank, policy, candidate, score))

    frozen = validation = test_2024 = confirmation = None
    status = "REJECTED_SELECTION"
    if candidates:
        _, policy, candidate, selection = max(candidates, key=lambda item: item[0])
        baseline_2023 = causes.metrics(frame, (2023,))
        validation = causes.metrics(candidate, (2023,))
        frozen = {
            "policy": policy,
            "eligible_candidates": len(candidates),
            "selection_baseline_2022": baseline_2022,
            "selection_2022": selection,
        }
        status = "REJECTED_VALIDATION"
        if validation_pass(validation, baseline_2023):
            baseline_2024 = causes.metrics(frame, (2024,))
            test_2024 = causes.metrics(candidate, (2024,))
            status = "REJECTED_TEST_2024"
            if validation_pass(test_2024, baseline_2024):
                confirmation = {str(year): causes.metrics(candidate, (year,))
                                for year in (2025, 2026)}
                status = ("STOP_GEOMETRY_RESEARCH_CANDIDATE"
                          if all(objective_pass(value)
                                 for value in confirmation.values())
                          else "REJECTED_OBJECTIVE")

    report = {
        "status": status,
        "ready_for_live_orders": False,
        "protocol": {
            "entry_and_target_router": "unchanged frozen target cascade",
            "selection": 2022,
            "validation": 2023,
            "conditional_test": 2024,
            "published_confirmation_only": [2025, 2026],
            "close_confirm_execution": "market exit at next hourly open",
            "same_bar_policy": "hard emergency stop before target",
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
            str(Path(cascade.__file__).name): digest(Path(cascade.__file__)),
            str(Path(__file__)): digest(Path(__file__)),
        },
        "limitations": [
            "Wider emergency stops increase loss size when they are reached.",
            "Hourly close confirmation can exit worse than the threshold after an opening gap.",
            "One-hour candles cannot determine tick order when stop and target share a bar.",
            "No historical result is eligible for runtime use without independent forward evidence.",
        ],
    }
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({
        "status": status,
        "selection_baseline_2022": baseline_2022,
        "selection_frontier": frontier,
        "frozen_before_validation": frozen,
        "validation_2023": validation,
        "test_2024": test_2024,
        "confirmation_2025_2026": confirmation,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
