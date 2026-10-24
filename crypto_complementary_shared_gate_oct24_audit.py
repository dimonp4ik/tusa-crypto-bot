"""Causal shared exposure gate for the complementary crypto portfolio.

The component signals are already frozen.  September selects a small bounded
set of shared-account exposure policies; October is the untouched test.
Every policy allows at most one entry per timestamp and one open position per
symbol, directly addressing clustered entries without changing trade exits.
"""
from __future__ import annotations

import collections
import hashlib
import heapq
import json
from pathlib import Path

import pandas as pd

import crypto_complementary_exact_portfolio_oct24_audit as portfolio


ROOT = portfolio.ROOT
SOURCE = ROOT / "crypto_complementary_exact_portfolio_oct24.json"
SOURCE_ROWS = ROOT / "crypto_complementary_exact_portfolio_oct24.pkl"
OUT = ROOT / "crypto_complementary_shared_gate_oct24.json"
ROWS_OUT = ROOT / "crypto_complementary_shared_gate_oct24.pkl"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def component(row: dict) -> str:
    return ("target" if str(row["source"]).startswith("walkforward")
            else "hourly")


def quality(row: dict) -> float:
    return float(row.get("model_probability", row.get(
        "model_quality", row.get("quality", 0.0))))


def shared_gate(rows: list[dict], *, priority: str, max_open: int,
                direction_cap: int, hourly_cap: int) -> list[dict]:
    pending = []
    opened = {}
    accepted_times = collections.deque()
    output = []
    sequence = 0
    order = {priority: 0, ("hourly" if priority == "target"
                           else "target"): 1}
    candidates = sorted(rows, key=lambda row: (
        float(row["entry_time"]), order[component(row)],
        -quality(row), str(row["symbol"])))
    last_timestamp = None
    for row in candidates:
        now = float(row["entry_time"])
        while pending and pending[0][0] < now:
            _, _, done = heapq.heappop(pending)
            opened.pop(str(done["symbol"]), None)
        while accepted_times and accepted_times[0] <= now-3600:
            accepted_times.popleft()
        symbol = str(row["symbol"])
        direction = str(row["direction"])
        if (now == last_timestamp or symbol in opened
                or len(opened) >= max_open
                or len(accepted_times) >= hourly_cap
                or sum(str(item["direction"]) == direction
                       for item in opened.values()) >= direction_cap):
            continue
        output.append(row)
        opened[symbol] = row
        accepted_times.append(now)
        last_timestamp = now
        sequence += 1
        heapq.heappush(pending, (float(row["exit_time"]), sequence, row))
    return output


def compact(value: dict) -> dict:
    return {name: value[name] for name in (
        "n", "return_pct", "dd_pct", "wr", "max_loss_streak",
        "trades_per_calendar_day", "negative_months")}


def main() -> None:
    frame = pd.read_pickle(SOURCE_ROWS)
    frame["month"] = pd.to_datetime(
        frame.entry_time, unit="s", utc=True).dt.strftime("%Y-%m")
    september = frame[frame.month.eq("2026-09")].to_dict("records")
    october = frame[frame.month.eq("2026-10")].to_dict("records")
    frontier = []
    for priority in ("target", "hourly"):
        for max_open in (4, 5, 6, 8):
            for direction_cap in (3, 4, 6):
                for hourly_cap in (3, 4, 6):
                    rows = shared_gate(
                        september, priority=priority, max_open=max_open,
                        direction_cap=direction_cap, hourly_cap=hourly_cap)
                    value = portfolio.metrics(rows, .03)
                    frontier.append({
                        "priority": priority, "max_open": max_open,
                        "direction_cap": direction_cap,
                        "hourly_cap": hourly_cap,
                        "selection": compact(value),
                        "score": round(
                            value["return_pct"]-.25*value["dd_pct"], 8),
                    })
    eligible = [item for item in frontier
                if item["selection"]["trades_per_calendar_day"] >= 3
                and item["selection"]["dd_pct"] <= 10]
    eligible.sort(key=lambda item: (
        item["score"], item["selection"]["return_pct"],
        -item["selection"]["dd_pct"]), reverse=True)
    selected = eligible[0]
    policy = {key: selected[key] for key in (
        "priority", "max_open", "direction_cap", "hourly_cap")}
    sep_rows = shared_gate(september, **policy)
    oct_rows = shared_gate(october, **policy)
    all_rows = [*sep_rows, *oct_rows]
    pd.DataFrame(all_rows).to_pickle(ROWS_OUT)
    results = {}
    for margin in (.03, .05):
        key = f"margin_{int(100*margin)}pct"
        results[key] = {
            "combined": portfolio.metrics(all_rows, margin),
            "months": {
                "2026-09": portfolio.metrics(sep_rows, margin),
                "2026-10": portfolio.metrics(oct_rows, margin),
            },
            "exposure": portfolio.exposure(all_rows, margin),
        }
    october_improved = bool(
        results["margin_3pct"]["months"]["2026-10"]["return_pct"]
        > json.loads(SOURCE.read_text(encoding="utf-8"))["results"]
        ["margin_3pct"]["months"]["2026-10"]["return_pct"])
    objective = bool(all(
        value["return_pct"] >= 30 and value["dd_pct"] <= 10
        for value in results["margin_3pct"]["months"].values()))
    report = {
        "status": ("SHARED_GATE_OBJECTIVE_PASSED" if objective else
                   "SHARED_GATE_OBJECTIVE_NOT_MET"),
        "ready_for_live_orders": False,
        "order_eligible": False,
        "protocol": {
            "selection": "September 2026 only",
            "test": "October 2026 through Oct 24",
            "tested_policies": len(frontier),
            "fixed_rules": ["one open position per symbol",
                            "one entry per timestamp"],
            "october_used_for_selection": False,
        },
        "selected_policy": selected,
        "results": results,
        "checks": {
            "october_improves_ungated_portfolio": october_improved,
            "october_30day_projection_pass_at_5pct_margin":
                results["margin_5pct"]["months"]["2026-10"]
                ["economic_gate"]["passes"],
            "objective_pass": objective,
        },
        "selection_frontier": eligible[:20],
        "sources": {
            str(SOURCE): digest(SOURCE),
            str(SOURCE_ROWS): digest(SOURCE_ROWS),
            str(Path(__file__).resolve()): digest(Path(__file__).resolve()),
        },
        "limitations": [
            "September is selection evidence, not final validation.",
            "October is one partial out-of-sample month.",
            "Gross results exclude fees at user request.",
        ],
    }
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True),
                   encoding="utf-8")
    print(json.dumps({
        "status": report["status"], "selected_policy": selected,
        "results": {key: {
            "combined": compact(value["combined"]),
            "months": {month: compact(result)
                       for month, result in value["months"].items()},
            "projections": {month: result["economic_gate"]
                            ["projected_monthly_return_pct"]
                            for month, result in value["months"].items()},
            "exposure": value["exposure"],
        } for key, value in results.items()},
        "checks": report["checks"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
