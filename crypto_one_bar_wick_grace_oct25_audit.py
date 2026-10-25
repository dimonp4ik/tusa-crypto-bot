"""Chronological audit of one-bar wick grace for the frozen 1h router.

The candidate keeps the original 1% stop.  On structurally eligible entries,
the first intrabar stop touch may be ignored only when the hourly candle opens
inside the stop, does not also touch the target, and closes back inside the
stop.  From the next candle onward the original hard stop is restored.  If the
stop candle closes outside the stop, the position exits at the next open.

All eligibility fields are known before entry.  Selection uses 2022 and the
frozen candidate is validated on 2023 before any later year is evaluated.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

import crypto_stop_cause_oct11_audit as causes
import crypto_okx_swap_1h_target_cascade_oct7_audit as cascade


OUT = causes.ROOT / "crypto_one_bar_wick_grace_oct25.json"
ROWS_OUT = causes.ROOT / "crypto_one_bar_wick_grace_oct25.pkl"
STOP_PCT = 1.0
MASKS = (
    "safe_path",
    "aligned_safe_path",
    "atr_safe_path",
    "aligned_atr_safe_path",
    "aligned_atr_safe_or_supportive",
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def mask_for(frame: pd.DataFrame, name: str) -> pd.Series:
    safe = ~(frame.opposing_level_before_target.astype(bool)
             | frame.opposing_order_block.astype(bool)
             | frame.failed_directional_breakout.astype(bool))
    aligned = ~(frame.higher_timeframe_conflict.astype(bool)
                | frame.btc_direction_conflict.astype(bool)
                | frame.market_breadth_conflict.astype(bool))
    atr = frame.stop_inside_hourly_atr.astype(bool)
    supportive = frame.supportive_ob_flag.astype(bool)
    return {
        "safe_path": safe,
        "aligned_safe_path": aligned & safe,
        "atr_safe_path": atr & safe,
        "aligned_atr_safe_path": aligned & atr & safe,
        "aligned_atr_safe_or_supportive": aligned & safe & (atr | supportive),
    }[name]


def resolve_row(row: dict, candle: dict, eligible: bool) -> tuple:
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
    stop = entry * (1 - direction * STOP_PCT / 100)
    target = entry * (1 + direction * target_pct / 100)
    end = min(len(times), first + hold)
    grace_used = False
    recovered_touch = False
    worst_excursion = 0.0
    last = first

    for index in range(first, end):
        if index > first and times[index] - times[index - 1] != 3600:
            break
        last = index
        adverse = (direction * (low[index] / entry - 1) if direction > 0
                   else direction * (high[index] / entry - 1))
        worst_excursion = min(worst_excursion, float(adverse))
        stop_hit = low[index] <= stop if direction > 0 else high[index] >= stop
        target_hit = high[index] >= target if direction > 0 else low[index] <= target

        if stop_hit:
            open_beyond = open_[index] <= stop if direction > 0 else open_[index] >= stop
            close_inside = close[index] > stop if direction > 0 else close[index] < stop
            can_recover = (eligible and not grace_used and not open_beyond
                           and not target_hit and close_inside)
            if can_recover:
                grace_used = True
                recovered_touch = True
                continue
            if eligible and not grace_used and not open_beyond and not target_hit:
                # The candle confirmed the break.  It is observable at the
                # close, so a market exit is represented by the next open.
                next_index = index + 1
                if next_index < len(times) and times[next_index] - times[index] == 3600:
                    fill = float(open_[next_index])
                    return (direction * (fill / entry - 1), int(times[next_index]),
                            False, 0, grace_used, recovered_touch, worst_excursion)
                value = direction * (float(close[index]) / entry - 1)
                return (value, int(times[index] + 3600), False, 0,
                        grace_used, recovered_touch, worst_excursion)
            fill = min(open_[index], stop) if direction > 0 else max(open_[index], stop)
            return (direction * (fill / entry - 1), int(times[index] + 3600),
                    False, 0, grace_used, recovered_touch, worst_excursion)

        if target_hit:
            return (target_pct / 100, int(times[index] + 3600), True, 1,
                    grace_used, recovered_touch, worst_excursion)

    value = direction * (float(close[last]) / entry - 1)
    value = min(target_pct / 100, value)
    return (value, int(times[last] + 3600), False, 2,
            grace_used, recovered_touch, worst_excursion)


def route(frame: pd.DataFrame, candles: dict, mask_name: str) -> pd.DataFrame:
    found = frame.copy()
    eligibility = mask_for(found, mask_name)
    prepared = {
        symbol: {
            key: np.asarray(candle[key], dtype=(np.int64 if key == "time" else float))
            for key in ("open", "high", "low", "close", "time")
        }
        for symbol, candle in candles.items()
    }
    columns = [[] for _ in range(7)]
    for index, row in found.iterrows():
        value = resolve_row(row.to_dict(), prepared[str(row["symbol"])],
                            bool(eligibility.at[index]))
        for output, item in zip(columns, value):
            output.append(item)
    (found["signed_return"], found["exit_time"], found["target_hit"],
     found["exit_reason"], found["grace_used"], found["recovered_stop_touch"],
     found["worst_excursion"]) = columns
    found["wick_grace_eligible"] = eligibility
    return found


def tail_metrics(frame: pd.DataFrame, years: tuple[int, ...]) -> dict:
    accepted = causes.accepted_frame(frame, years)
    returns = accepted.signed_return.astype(float)
    excursions = accepted.worst_excursion.astype(float)
    return {
        "realized_min_signed_move_pct": round(100 * float(returns.min()), 6),
        "realized_p01_signed_move_pct": round(100 * float(returns.quantile(.01)), 6),
        "worst_intrabar_excursion_pct": round(100 * float(excursions.min()), 6),
        "grace_used": int(accepted.grace_used.astype(bool).sum()),
        "grace_target_hits": int((accepted.grace_used.astype(bool)
                                  & accepted.target_hit.astype(bool)).sum()),
        "grace_eventual_stops": int((accepted.grace_used.astype(bool)
                                     & accepted.exit_reason.eq(0)).sum()),
    }


def score(frame: pd.DataFrame, years: tuple[int, ...]) -> dict:
    return {**causes.metrics(frame, years), **tail_metrics(frame, years)}


def pass_stage(candidate: dict, baseline: dict) -> bool:
    return bool(
        candidate["n"] >= baseline["n"] * .97
        and candidate["trades_per_calendar_day"] >= 2.0
        and candidate["return_pct"] >= baseline["return_pct"]
        and candidate["literal_target_hit_rate"] >= baseline["literal_target_hit_rate"]
        and candidate["dd_pct"] <= baseline["dd_pct"]
        and candidate["negative_months"] <= baseline["negative_months"]
        and candidate["minimum_month_pct"] >= baseline["minimum_month_pct"]
        and candidate["max_loss_streak"] <= baseline["max_loss_streak"]
        and candidate["realized_min_signed_move_pct"] >= -2.0
    )


def objective_pass(value: dict) -> bool:
    return bool(
        value["trades_per_calendar_day"] >= 3
        and value["negative_months"] == 0
        and value["months_ge_30pct"] == value["months"]
        and value["dd_pct"] <= 10
    )


def baseline_score(frame: pd.DataFrame, years: tuple[int, ...]) -> dict:
    value = causes.metrics(frame, years)
    accepted = causes.accepted_frame(frame, years)
    returns = accepted.signed_return.astype(float)
    return {
        **value,
        "realized_min_signed_move_pct": round(100 * float(returns.min()), 6),
        "realized_p01_signed_move_pct": round(100 * float(returns.quantile(.01)), 6),
        "worst_intrabar_excursion_pct": None,
        "grace_used": 0,
        "grace_target_hits": 0,
        "grace_eventual_stops": 0,
    }


def main() -> None:
    candles, sources = causes.load_candles()
    original = pd.read_pickle(causes.ROWS)
    frame = causes.enrich(original, candles)
    baseline_2022 = baseline_score(frame, (2022,))
    frontier = []
    eligible = []
    routed = {}
    for mask_name in MASKS:
        candidate = route(frame, candles, mask_name)
        routed[mask_name] = candidate
        value = score(candidate, (2022,))
        passed = pass_stage(value, baseline_2022)
        frontier.append({"mask": mask_name, "selection_2022": value,
                         "selection_pass": passed})
        if passed:
            rank = (baseline_2022["negative_months"] - value["negative_months"],
                    value["minimum_month_pct"], value["return_pct"],
                    -value["dd_pct"], value["literal_target_hit_rate"])
            eligible.append((rank, mask_name, candidate, value))

    frozen = validation = test_2024 = confirmation = None
    status = "REJECTED_SELECTION"
    selected = None
    if eligible:
        _, mask_name, selected, selection = max(eligible, key=lambda item: item[0])
        baseline_2023 = baseline_score(frame, (2023,))
        validation = score(selected, (2023,))
        frozen = {
            "mask": mask_name,
            "eligible_candidates": len(eligible),
            "selection_baseline_2022": baseline_2022,
            "selection_2022": selection,
            "validation_baseline_2023": baseline_2023,
        }
        status = "REJECTED_VALIDATION"
        if pass_stage(validation, baseline_2023):
            baseline_2024 = baseline_score(frame, (2024,))
            test_2024 = score(selected, (2024,))
            status = "REJECTED_TEST_2024"
            if pass_stage(test_2024, baseline_2024):
                confirmation = {
                    str(year): {
                        "baseline": baseline_score(frame, (year,)),
                        "candidate": score(selected, (year,)),
                    } for year in (2025, 2026)
                }
                status = ("WICK_GRACE_RESEARCH_CANDIDATE"
                          if all(objective_pass(item["candidate"])
                                 for item in confirmation.values())
                          else "REJECTED_OBJECTIVE")
                selected.to_pickle(ROWS_OUT)

    report = {
        "status": status,
        "ready_for_live_orders": False,
        "protocol": {
            "entry_and_target_router": "unchanged frozen target cascade",
            "default_stop_pct": STOP_PCT,
            "grace": "ignore first intrabar stop touch only after recovered hourly close; restore hard stop next candle",
            "same_bar_target_and_stop": "conservative stop-first; no grace",
            "confirmed_break_execution": "market exit at next hourly open",
            "selection": 2022,
            "validation": 2023,
            "conditional_test": 2024,
            "published_confirmation_only": [2025, 2026],
            "market_entries_only": True,
            "money": "3% current equity margin, fixed 10x gross",
            "fees": "excluded by user instruction",
        },
        "mask_definitions": {
            "safe_path": "no opposing level, opposing block, or failed breakout",
            "aligned_safe_path": "BTC, breadth, 168h trend align and path is safe",
            "atr_safe_path": "14h ATR is at least 1% and path is safe",
            "aligned_atr_safe_path": "alignment, ATR and safe path",
            "aligned_atr_safe_or_supportive": "alignment and safe path, plus ATR or supportive order block",
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
            str(Path(__file__).name): digest(Path(__file__)),
        },
        "limitations": [
            "Hourly OHLC cannot reveal the tick path inside a candle.",
            "A close-confirmed stop can realize more than the original 1% price loss.",
            "The rule requires a virtual stop during an eligible first touch and therefore carries gap and intrabar tail risk.",
            "No historical result is eligible for runtime use without forward shadow evidence.",
        ],
    }
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({
        "status": status,
        "eligible_selection_candidates": len(eligible),
        "selection_frontier": frontier,
        "frozen_before_validation": frozen,
        "validation_2023": validation,
        "test_2024": test_2024,
        "confirmation_2025_2026": confirmation,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
