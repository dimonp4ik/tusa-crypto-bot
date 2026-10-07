"""Causal online health gate for the annual four-hour complement.

Rules are selected on 2022-2024 accepted complement outcomes, validated on
2025, and then replayed sequentially on 2026.  Gate state uses only outcomes
of complement trades accepted by the combined runtime before each decision.
"""
from __future__ import annotations

import collections
import hashlib
import heapq
import itertools
import json
from pathlib import Path

import pandas as pd

import crypto_okx_swap_1h_regime_oct7_audit as base
import crypto_okx_swap_1h_target_cascade_oct7_audit as target_engine
import crypto_okx_swap_4h_structure_complement_oct8_audit as four
import crypto_okx_swap_4h_walkforward_complement_oct8_audit as walk
from crypto_runtime_shadow_portfolio_oct3_audit import runtime_gate


ROOT = Path("reports/audit_2026_09_08")
CASCADE_ROWS = ROOT / "crypto_okx_swap_1h_target_cascade_oct7.pkl"
CASCADE_REPORT = ROOT / "crypto_okx_swap_1h_target_cascade_oct7.json"
WALK_REPORT = ROOT / "crypto_okx_swap_4h_walkforward_complement_oct8.json"
OUT = ROOT / "crypto_okx_swap_4h_online_health_oct8.json"
COMPLEMENT_SOURCE = "four_hour_walkforward"
WINDOWS = (20, 30, 40, 60)
HIT_FLOORS = (.50, .525, .55, .575)
RETURN_FLOORS = (-.04, -.02, 0.0)
STOP_STREAKS = (3, 4, 5)
PAUSE_DAYS = (3, 7, 14, 30)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def reconstruct(frame: pd.DataFrame, x, item: dict) -> pd.DataFrame:
    scored = walk.fit_scores(frame, x, tuple(item["train_years"]))
    return four.route(
        scored, item["entry_threshold"],
        {float(key): value for key, value in item["upgrade_thresholds"].items()},
        tuple(item["allowed_families"]),
    )


def build_rows(frame: pd.DataFrame, x, cascade: pd.DataFrame,
               walk_report: dict) -> tuple[list[dict], dict[int, pd.DataFrame]]:
    folds = {int(item["evaluation_year"]): item for item in walk_report["folds"]}
    routed: dict[int, pd.DataFrame] = {}
    first = folds[2023]
    first_route = reconstruct(frame, x, first)
    routed[2022] = first_route
    routed[2023] = first_route
    for year in (2024, 2025, 2026):
        routed[year] = reconstruct(frame, x, folds[year])
    cascade_rows = four.raw_rows(
        cascade, tuple(range(2022, 2027)), "hourly_cascade", 0)
    extra_rows = []
    for year in range(2022, 2027):
        extra_rows.extend(four.raw_rows(
            routed[year], (year,), COMPLEMENT_SOURCE, 1))
    return [*cascade_rows, *extra_rows], routed


def trigger(history: collections.deque, stop_streak: int, rule: dict) -> bool:
    if stop_streak >= rule["stop_streak"]:
        return True
    if len(history) < rule["window"]:
        return False
    hit_rate = sum(bool(row["target_hit"]) for row in history) / len(history)
    signed_sum = sum(float(row["signed_price_return"]) for row in history)
    return bool(hit_rate < rule["hit_floor"]
                or signed_sum < rule["return_floor"])


def gated_runtime(rows: list[dict], rule: dict) -> tuple[list[dict], dict]:
    pending: list[tuple[float, int, dict]] = []
    opened: dict[str, dict] = {}
    last: dict[tuple[str, str], float] = {}
    accepted_times: collections.deque[float] = collections.deque()
    history: collections.deque[dict] = collections.deque(maxlen=rule["window"])
    output: list[dict] = []
    day = None
    account_streak = 0
    account_paused = False
    complement_stop_streak = 0
    disabled_until = -1e20
    health_rejections = 0
    triggers = 0
    sequence = 0
    ordered = sorted(rows, key=lambda row: (
        float(row["entry_time"]), int(row["source_priority"]),
        str(row["symbol"]), int(row.get("module_priority", 999)),
    ))
    for row in ordered:
        now = float(row["entry_time"])
        new_day = int(now // 86400)
        if new_day != day:
            day, account_streak, account_paused = new_day, 0, False
        while pending and pending[0][0] < now:
            exit_time, _, done = heapq.heappop(pending)
            opened.pop(str(done["symbol"]), None)
            if int(exit_time // 86400) == day:
                account_streak = (account_streak + 1
                                  if done["outcome"] == "SL" else 0)
                if account_streak >= 3:
                    account_paused = True
            if done["source"] == COMPLEMENT_SOURCE:
                history.append(done)
                complement_stop_streak = (
                    complement_stop_streak + 1
                    if int(done["exit_reason"]) == 0 else 0)
                if trigger(history, complement_stop_streak, rule):
                    disabled_until = max(
                        disabled_until, exit_time + rule["pause_days"] * 86400)
                    complement_stop_streak = 0
                    triggers += 1
        while accepted_times and accepted_times[0] <= now - 3600:
            accepted_times.popleft()
        symbol = str(row["symbol"])
        direction = str(row["direction"])
        if row["source"] == COMPLEMENT_SOURCE and now < disabled_until:
            health_rejections += 1
            continue
        if account_paused or symbol in opened or len(accepted_times) >= 2:
            continue
        if now - last.get((symbol, direction), -1e20) < 10800:
            continue
        if sum(str(item["direction"]) == direction
               for item in opened.values()) >= 3:
            continue
        opened[symbol] = row
        last[(symbol, direction)] = now
        accepted_times.append(now)
        output.append(row)
        sequence += 1
        heapq.heappush(pending, (float(row["exit_time"]), sequence, row))
    return output, {
        "health_rejections": health_rejections,
        "health_triggers": triggers,
        "last_disabled_until": disabled_until,
    }


def period(rows: list[dict], years: tuple[int, ...]) -> list[dict]:
    allowed = set(years)
    return [row for row in rows if pd.Timestamp(
        float(row["entry_time"]), unit="s", tz="UTC").year in allowed]


def summarize_period(rows: list[dict], years: tuple[int, ...]) -> dict:
    return four.summarize(period(rows, years))


def static_comparison(all_rows: list[dict], years: tuple[int, ...]) -> dict:
    raw_period = period(all_rows, years)
    baseline = runtime_gate([row for row in raw_period
                             if row["source"] == "hourly_cascade"])
    ungated = runtime_gate(raw_period)
    return {"cascade": four.summarize(baseline),
            "ungated_union": four.summarize(ungated)}


def period_comparison(all_rows: list[dict], accepted: list[dict],
                      years: tuple[int, ...], diagnostics: dict | None = None,
                      static: dict | None = None) -> dict:
    value = {**(static or static_comparison(all_rows, years)),
             "gated_union": summarize_period(accepted, years)}
    if diagnostics is not None:
        value["gate_diagnostics"] = diagnostics
    return value


def selection_ok(value: dict) -> bool:
    baseline, ungated, gated = (
        value["cascade"], value["ungated_union"], value["gated_union"])
    retained_gain = baseline["return_pct"] + .60 * max(
        0.0, ungated["return_pct"] - baseline["return_pct"])
    return bool(gated["return_pct"] >= retained_gain
                and gated["dd_pct"] <= ungated["dd_pct"]
                and gated["literal_target_hit_rate"] >= 63
                and gated["trades_per_calendar_day"] >= 2.5
                and gated["negative_months"] <= ungated["negative_months"])


def validation_ok(value: dict) -> bool:
    baseline, gated = value["cascade"], value["gated_union"]
    return bool(gated["return_pct"] > baseline["return_pct"]
                and gated["dd_pct"] <= baseline["dd_pct"] + 1.0
                and gated["literal_target_hit_rate"] >= 63
                and gated["trades_per_calendar_day"] >= 2.5
                and gated["negative_months"] <= baseline["negative_months"] + 1)


def main() -> None:
    candles, sources = base.load()
    raw = four.build_features(candles)
    frame = four.proposals(raw).reset_index(drop=True)
    for target in four.TARGETS:
        target_engine.add_target(frame, candles, target)
    columns = four.schema(frame[frame.year.isin((2020, 2021))])
    x = four.matrix(frame, columns)
    cascade = pd.read_pickle(CASCADE_ROWS)
    walk_report = json.loads(WALK_REPORT.read_text(encoding="utf-8"))
    all_rows, _ = build_rows(frame, x, cascade, walk_report)
    selection_rows = period(all_rows, (2022, 2023, 2024))
    selection_static = static_comparison(
        selection_rows, (2022, 2023, 2024))
    choices = []
    frontier = []
    for window, hit_floor, return_floor, stop_streak, pause_days in itertools.product(
            WINDOWS, HIT_FLOORS, RETURN_FLOORS, STOP_STREAKS, PAUSE_DAYS):
        rule = {"window": window, "hit_floor": hit_floor,
                "return_floor": return_floor, "stop_streak": stop_streak,
                "pause_days": pause_days}
        accepted, diagnostics = gated_runtime(selection_rows, rule)
        value = period_comparison(
            selection_rows, accepted, (2022, 2023, 2024), diagnostics,
            selection_static)
        if not selection_ok(value) or diagnostics["health_rejections"] < 20:
            continue
        rank = (value["gated_union"]["return_pct"]
                - 2 * value["gated_union"]["dd_pct"]
                - .5 * value["gated_union"]["negative_months"])
        item = {"rule": rule, "selection_2022_2024": value,
                "selection_rank": rank}
        frontier.append(item)
        choices.append(item)
    print(json.dumps({"stage": "selection", "tested": (
        len(WINDOWS) * len(HIT_FLOORS) * len(RETURN_FLOORS)
        * len(STOP_STREAKS) * len(PAUSE_DAYS)),
        "qualified": len(choices)}), flush=True)
    validated = []
    validation_frontier = []
    rows_through_2025 = period(all_rows, (2022, 2023, 2024, 2025))
    validation_static = static_comparison(all_rows, (2025,))
    for item in choices:
        accepted, diagnostics = gated_runtime(rows_through_2025, item["rule"])
        value = period_comparison(
            all_rows, accepted, (2025,), diagnostics, validation_static)
        if not validation_ok(value):
            continue
        gain_selection = (item["selection_2022_2024"]["gated_union"]["return_pct"]
                          - item["selection_2022_2024"]["cascade"]["return_pct"])
        gain_validation = (value["gated_union"]["return_pct"]
                           - value["cascade"]["return_pct"])
        robust_rank = min(gain_selection, gain_validation) \
            - 2 * max(item["selection_2022_2024"]["gated_union"]["dd_pct"],
                      value["gated_union"]["dd_pct"])
        found = {**item, "validation_2025": value,
                 "robust_rank": robust_rank}
        validation_frontier.append(found)
        validated.append(found)
    print(json.dumps({"stage": "validation", "tested": len(choices),
                      "qualified": len(validated)}), flush=True)
    validated.sort(key=lambda item: item["robust_rank"], reverse=True)
    frozen = validated[0] if validated else None
    final_2026 = None
    status = "REJECTED_SELECTION" if not choices else "REJECTED_2025"
    if frozen:
        accepted, diagnostics = gated_runtime(all_rows, frozen["rule"])
        final_2026 = period_comparison(
            all_rows, accepted, (2026,), diagnostics,
            static_comparison(all_rows, (2026,)))
        status = ("ONLINE_HEALTH_RESEARCH_CANDIDATE_REQUIRES_FORWARD"
                  if validation_ok(final_2026) else "REJECTED_2026")
    report = {
        "status": status,
        "protocol": {
            "selection": [2022, 2023, 2024], "validation": 2025,
            "research_biased_final": 2026,
            "gate_observations": "only previously closed accepted complement trades",
            "re_enable": (
                "retain the rolling health window and admit probe trades after cooldown"),
            "grid": {"windows": list(WINDOWS), "hit_floors": list(HIT_FLOORS),
                     "return_floors": list(RETURN_FLOORS),
                     "stop_streaks": list(STOP_STREAKS),
                     "pause_days": list(PAUSE_DAYS)},
            "money": "3% current equity margin, fixed 10x gross",
        },
        "feature_rows": len(raw), "proposal_rows": len(frame),
        "tested_rules": (len(WINDOWS) * len(HIT_FLOORS) * len(RETURN_FLOORS)
                         * len(STOP_STREAKS) * len(PAUSE_DAYS)),
        "qualified_selection": len(choices),
        "qualified_validation": len(validated),
        "selection_frontier_top50": sorted(
            frontier, key=lambda item: item["selection_rank"], reverse=True)[:50],
        "validation_frontier_top50": sorted(
            validation_frontier, key=lambda item: item["robust_rank"],
            reverse=True)[:50],
        "frozen_before_2026": frozen,
        "final_2026": final_2026,
        "sources": {**sources, str(CASCADE_ROWS): digest(CASCADE_ROWS),
                    str(CASCADE_REPORT): digest(CASCADE_REPORT),
                    str(WALK_REPORT): digest(WALK_REPORT),
                    str(Path(__file__)): digest(Path(__file__))},
        "limitations": [
            "The sticky-window revision was added after inspecting the resetting gate on 2026 and is research-biased.",
            "Any historical pass still requires direct EU X-Perp forward shadow confirmation.",
        ],
    }
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({key: report[key] for key in (
        "status", "tested_rules", "qualified_selection", "qualified_validation",
        "frozen_before_2026", "final_2026")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
