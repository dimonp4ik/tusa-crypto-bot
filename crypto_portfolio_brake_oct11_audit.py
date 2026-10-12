"""Causal audit of clustered-entry and closed-loss brakes for the frozen router."""
from __future__ import annotations

import collections
import hashlib
import heapq
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

import crypto_stop_cause_oct11_audit as causes
import crypto_okx_swap_1h_target_cascade_oct7_audit as cascade


OUT = causes.ROOT / "crypto_portfolio_brake_oct11.json"
CONFIGS = tuple(
    {"loss_pause": loss_pause, "hourly_cap": hourly_cap,
     "max_same_direction": max_same_direction,
     "cooldown_hours": cooldown_hours}
    for loss_pause, hourly_cap, max_same_direction, cooldown_hours
    in itertools.product((1, 2, 3), (1, 2), (1, 2, 3), (3, 6))
)
BASE = {"loss_pause": 3, "hourly_cap": 2,
        "max_same_direction": 3, "cooldown_hours": 3}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_rows(frame: pd.DataFrame, years: tuple[int, ...]) -> list[dict]:
    chosen = frame[frame.year.isin(years) & frame.quality.ge(.5)].copy()
    chosen = chosen.sort_values(
        ["entry_time", "quality", "symbol"], ascending=[True, False, True])
    chosen["source_priority"] = chosen.groupby("entry_time").cumcount()
    return [
        {**row,
         "direction": "LONG" if int(row["direction"]) > 0 else "SHORT",
         "source": "literal_regime_router", "module": "literal_regime_router",
         "module_priority": 0,
         "outcome": "SL" if float(row["signed_return"]) < 0 else "TP",
         "signed_price_return": float(row["signed_return"])}
        for row in chosen.to_dict("records")
    ]


def gate(rows: list[dict], config: dict) -> list[dict]:
    """Use only exposure and outcomes known before each candidate entry."""
    pending: list[tuple[float, int, dict]] = []
    opened: dict[str, dict] = {}
    last: dict[tuple[str, str], float] = {}
    accepted_times: collections.deque[float] = collections.deque()
    output: list[dict] = []
    day = None
    streak = 0
    paused = False
    ordered = sorted(rows, key=lambda row: (
        float(row["entry_time"]), int(row["source_priority"]),
        str(row["symbol"]), int(row.get("module_priority", 999))))
    for row in ordered:
        now = float(row["entry_time"])
        new_day = int(now // 86400)
        if new_day != day:
            day, streak, paused = new_day, 0, False
        while pending and pending[0][0] < now:
            _, _, done = heapq.heappop(pending)
            opened.pop(str(done["symbol"]), None)
            if int(float(done["exit_time"]) // 86400) == day:
                streak = streak + 1 if done["outcome"] == "SL" else 0
                if streak >= int(config["loss_pause"]):
                    paused = True
        while accepted_times and accepted_times[0] <= now - 3600:
            accepted_times.popleft()
        symbol = str(row["symbol"])
        direction = str(row["direction"])
        if paused or symbol in opened:
            continue
        if len(accepted_times) >= int(config["hourly_cap"]):
            continue
        if now - last.get((symbol, direction), -1e20) < 3600 * int(config["cooldown_hours"]):
            continue
        if sum(str(item["direction"]) == direction for item in opened.values()) \
                >= int(config["max_same_direction"]):
            continue
        opened[symbol] = row
        last[(symbol, direction)] = now
        accepted_times.append(now)
        output.append(row)
        heapq.heappush(pending, (float(row["exit_time"]), len(output), row))
    return output


def metrics(frame: pd.DataFrame, years: tuple[int, ...], config: dict) -> dict:
    rows = gate(source_rows(frame, years), config)
    value = cascade.portfolio.money(rows, cascade.portfolio.MARGIN_FRACTION)
    n = len(rows)
    months = list(value.get("monthly_return_pct", {}).values())
    value.update({
        "literal_target_hits": sum(bool(row["target_hit"]) for row in rows),
        "literal_target_hit_rate": round(
            100 * sum(bool(row["target_hit"]) for row in rows) / n, 4) if n else 0.0,
        "stops": sum(int(row["exit_reason"]) == 0 for row in rows),
        "timeouts": sum(int(row["exit_reason"]) == 2 for row in rows),
        "months": len(months),
        "months_ge_30pct": sum(number >= 30 for number in months),
        "minimum_month_pct": round(min(months), 4) if months else 0.0,
        "median_month_pct": round(float(np.median(months)), 4) if months else 0.0,
        "target_distribution": {
            str(target): sum(float(row["selected_target_pct"]) == target for row in rows)
            for target in cascade.TARGETS},
        "target_hits": {
            str(target): sum(float(row["selected_target_pct"]) == target
                             and bool(row["target_hit"]) for row in rows)
            for target in cascade.TARGETS},
        "average_selected_target_pct": round(
            sum(float(row["selected_target_pct"]) for row in rows) / n, 4)
            if n else 0.0,
    })
    return value


def dominates(candidate: dict, baseline: dict, minimum_n: int) -> bool:
    return bool(candidate["n"] >= minimum_n
                and candidate["trades_per_calendar_day"] >= 2.0
                and candidate["return_pct"] >= baseline["return_pct"]
                and candidate["literal_target_hit_rate"] >= baseline["literal_target_hit_rate"]
                and candidate["dd_pct"] <= baseline["dd_pct"]
                and candidate["negative_months"] <= baseline["negative_months"]
                and candidate["minimum_month_pct"] >= baseline["minimum_month_pct"]
                and candidate["max_loss_streak"] <= baseline["max_loss_streak"])


def cluster_attribution(frame: pd.DataFrame, year: int) -> dict:
    accepted = causes.accepted_frame(frame, (year,))
    groups = list(accepted.groupby("entry_time", sort=True))
    result = {}
    for label, subset in (
            ("single", accepted[accepted.groupby("entry_time")["entry_time"]
                                .transform("size").eq(1)]),
            ("paired", accepted[accepted.groupby("entry_time")["entry_time"]
                                .transform("size").eq(2)])):
        n = len(subset)
        result[label] = {
            "trades": int(n),
            "stop_rate_pct": round(100 * float(subset.exit_reason.eq(0).mean()), 4)
            if n else 0.0,
            "target_hit_rate_pct": round(100 * float(subset.target_hit.mean()), 4)
            if n else 0.0,
            "mean_signed_price_return_pct": round(
                100 * float(subset.signed_return.mean()), 6) if n else 0.0,
        }
    pairs = [part for _, part in groups if len(part) == 2]
    both_stop = [bool(part.exit_reason.eq(0).all()) for part in pairs]
    any_stop = [bool(part.exit_reason.eq(0).any()) for part in pairs]
    same_direction = [len(set(part.direction.astype(str))) == 1 for part in pairs]
    result["paired_hours"] = {
        "hours": len(pairs),
        "both_stop_hours": int(sum(both_stop)),
        "both_stop_rate_pct": round(100 * float(np.mean(both_stop)), 4)
        if pairs else 0.0,
        "any_stop_rate_pct": round(100 * float(np.mean(any_stop)), 4)
        if pairs else 0.0,
        "same_direction_rate_pct": round(100 * float(np.mean(same_direction)), 4)
        if pairs else 0.0,
    }
    cause_context = {}
    for name in causes.CAUSES:
        flagged = [bool(part[name].astype(bool).any()) for part in pairs]
        flagged_both = [stop for stop, flag in zip(both_stop, flagged) if flag]
        clear_both = [stop for stop, flag in zip(both_stop, flagged) if not flag]
        cause_context[name] = {
            "flagged_pair_hours": int(sum(flagged)),
            "both_stop_rate_when_flagged_pct": round(
                100 * float(np.mean(flagged_both)), 4) if flagged_both else None,
            "both_stop_rate_when_clear_pct": round(
                100 * float(np.mean(clear_both)), 4) if clear_both else None,
        }
    result["paired_hour_cause_context"] = cause_context
    return result


def main() -> None:
    candles, sources = causes.load_candles()
    frame = causes.enrich(pd.read_pickle(causes.ROWS), candles)
    baseline = {year: metrics(frame, (year,), BASE) for year in range(2022, 2027)}
    exact = {year: causes.metrics(frame, (year,)) for year in range(2022, 2027)}
    parity = all(baseline[year] == exact[year] for year in baseline)
    if not parity:
        raise ValueError("custom baseline does not reproduce the frozen runtime gate")

    frontier = []
    eligible = []
    for config in CONFIGS:
        score = metrics(frame, (2022,), config)
        passed = dominates(score, baseline[2022], 700)
        frontier.append({"config": config, "selection_2022": score,
                         "selection_pass": passed})
        if passed and config != BASE:
            rank = (baseline[2022]["negative_months"] - score["negative_months"],
                    baseline[2022]["max_loss_streak"] - score["max_loss_streak"],
                    score["minimum_month_pct"], score["return_pct"],
                    score["literal_target_hit_rate"], -score["dd_pct"])
            eligible.append((rank, config, score))

    frozen = validation = test = confirmation = None
    status = "REJECTED_SELECTION"
    if eligible:
        _, config, selection = max(eligible, key=lambda item: item[0])
        frozen = {"config": config, "selection_2022": selection,
                  "selection_baseline_2022": baseline[2022],
                  "eligible_candidates": len(eligible)}
        validation = metrics(frame, (2023,), config)
        status = "REJECTED_VALIDATION"
        if dominates(validation, baseline[2023], 500):
            test = metrics(frame, (2024,), config)
            status = "REJECTED_TEST_2024"
            if dominates(test, baseline[2024], 500):
                confirmation = {str(year): metrics(frame, (year,), config)
                                for year in (2025, 2026)}
                status = ("PORTFOLIO_BRAKE_RESEARCH_CANDIDATE"
                          if all(dominates(confirmation[str(year)], baseline[year], 400)
                                 for year in (2025, 2026))
                          else "REJECTED_CONFIRMATION")

    report = {
        "status": status, "ready_for_live_orders": False,
        "protocol": {
            "entry_target_stop": "unchanged frozen target cascade",
            "selection": 2022, "validation": 2023, "test": 2024,
            "confirmation_only": [2025, 2026],
            "causal": "only positions and closed outcomes known before entry are used",
            "entry_and_exit_orders": "market proxies only",
            "money": "3% current-equity margin, fixed 10x gross",
        },
        "baseline_config": BASE, "baseline": {str(k): v for k, v in baseline.items()},
        "baseline_parity": parity, "evaluated_configs": len(CONFIGS),
        "cluster_attribution": {
            str(year): cluster_attribution(frame, year) for year in range(2022, 2027)},
        "selection_frontier": frontier, "frozen_before_validation": frozen,
        "validation_2023": validation, "test_2024": test,
        "confirmation_2025_2026": confirmation,
        "sources": {**sources, str(causes.ROWS): digest(causes.ROWS),
                    str(Path(causes.__file__).name): digest(Path(causes.__file__)),
                    str(Path(cascade.__file__).name): digest(Path(cascade.__file__)),
                    str(Path(cascade.portfolio.__file__).name): digest(Path(cascade.portfolio.__file__)),
                    str(Path(__file__)): digest(Path(__file__))},
        "limitations": [
            "A stricter cluster brake reduces opportunity as well as correlated risk.",
            "The audit uses hourly OHLCV outcomes and cannot reconstruct tick-level fill order.",
            "No candidate may change runtime before genuinely new forward confirmation.",
        ],
    }
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"status": status, "eligible": len(eligible),
                      "baseline_parity": parity, "frozen": frozen,
                      "validation": validation, "test": test,
                      "confirmation": confirmation}, indent=2), flush=True)


if __name__ == "__main__":
    main()
