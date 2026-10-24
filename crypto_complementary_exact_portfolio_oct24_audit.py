"""Exact shared-account audit of the 15m target and stable 1h systems.

Both component policies were selected without October outcomes.  This audit
does not tune either component: it combines their accepted market entries,
sizes every trade from equity known at entry, and measures overlap and monthly
results at the client-selected 3% and 5% margin settings.
"""
from __future__ import annotations

import collections
import hashlib
import json
from pathlib import Path

import pandas as pd

from crypto_runtime_shadow_portfolio_oct3_audit import money
from src.research_profit_gate import economic_gate


ROOT = Path("reports/audit_2026_09_08")
TARGET_SEP = ROOT / "crypto_stable_route_group_oct24_selection.pkl"
TARGET_OCT = ROOT / "crypto_stable_route_group_oct24_audit.pkl"
HOURLY = ROOT / "crypto_okx_swap_1h_stable_material_exit_oct24_forward.pkl"
TARGET_REPORT = ROOT / "crypto_stable_route_group_oct24_audit.json"
HOURLY_REPORT = ROOT / "crypto_okx_swap_1h_stable_material_exit_oct24_forward.json"
OUT = ROOT / "crypto_complementary_exact_portfolio_oct24.json"
ROWS_OUT = ROOT / "crypto_complementary_exact_portfolio_oct24.pkl"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def month_rows(frame: pd.DataFrame, month: str) -> list[dict]:
    labels = pd.to_datetime(
        frame.entry_time, unit="s", utc=True).dt.strftime("%Y-%m")
    return frame[labels.eq(month)].to_dict("records")


def metrics(rows: list[dict], margin: float) -> dict:
    value = money(rows, margin)
    signed = [float(row["signed_price_return"]) for row in rows]
    value["economic_gate"] = economic_gate(
        signed, float(value["trades_per_calendar_day"]),
        margin_fraction=margin)
    return value


def exposure(rows: list[dict], margin: float) -> dict:
    events = []
    for index, row in enumerate(rows):
        events.append((float(row["entry_time"]), 0, index, row))
        events.append((float(row["exit_time"]), 1, index, row))
    active = {}
    maximum = same_symbol = 0
    maximum_direction = collections.Counter()
    for _, kind, index, row in sorted(events):
        if kind == 0:
            symbol = str(row["symbol"])
            same_symbol += int(any(
                str(item["symbol"]) == symbol for item in active.values()))
            active[index] = row
            maximum = max(maximum, len(active))
            direction_counts = collections.Counter(
                str(item["direction"]) for item in active.values())
            for direction, count in direction_counts.items():
                maximum_direction[direction] = max(
                    maximum_direction[direction], count)
        else:
            active.pop(index, None)
    clocks = collections.Counter(int(float(row["entry_time"])) for row in rows)
    hours = collections.Counter(int(float(row["entry_time"]) // 3600)
                             for row in rows)
    return {
        "max_concurrent_positions": maximum,
        "max_concurrent_margin_pct": round(100*margin*maximum, 4),
        "same_symbol_overlap_entries": same_symbol,
        "max_concurrent_by_direction": dict(maximum_direction),
        "max_entries_same_timestamp": max(clocks.values()),
        "max_entries_same_clock_hour": max(hours.values()),
    }


def compact(value: dict) -> dict:
    return {name: value[name] for name in (
        "n", "return_pct", "dd_pct", "wr", "max_loss_streak",
        "trades_per_calendar_day", "negative_months")}


def main() -> None:
    target_sep = pd.read_pickle(TARGET_SEP)
    target_oct = pd.read_pickle(TARGET_OCT)
    hourly = pd.read_pickle(HOURLY)
    hourly = hourly[hourly.outcome.ne("PENDING")].copy()

    month_sets = {
        "2026-09": [*target_sep.to_dict("records"),
                    *month_rows(hourly, "2026-09")],
        "2026-10": [*target_oct.to_dict("records"),
                    *month_rows(hourly, "2026-10")],
    }
    all_rows = [row for month in month_sets.values() for row in month]
    pd.DataFrame(all_rows).to_pickle(ROWS_OUT)

    results = {}
    for margin in (.03, .05):
        key = f"margin_{int(100*margin)}pct"
        results[key] = {
            "combined": metrics(all_rows, margin),
            "months": {month: metrics(rows, margin)
                       for month, rows in month_sets.items()},
            "exposure": exposure(all_rows, margin),
        }
    october_projection_pass_5 = bool(
        results["margin_5pct"]["months"]["2026-10"]
        ["economic_gate"]["passes"])
    objective = bool(all(
        value["return_pct"] >= 30 and value["dd_pct"] <= 10
        for value in results["margin_3pct"]["months"].values()))
    report = {
        "status": ("COMPLEMENTARY_PORTFOLIO_OBJECTIVE_PASSED"
                   if objective else
                   "COMPLEMENTARY_PORTFOLIO_IMPROVED_OBJECTIVE_NOT_MET"),
        "ready_for_live_orders": False,
        "order_eligible": False,
        "protocol": {
            "components": [
                "pre-October stable 15m exact-target group route",
                "frozen stable 1h fixed 2% target route",
            ],
            "selection_or_retuning": "none",
            "entry": "component market entry",
            "exit": "component market close",
            "money": "client-selected current-equity margin per trade, fixed 10x",
            "observed_months": ["2026-09", "2026-10 through Oct 24"],
        },
        "results": results,
        "checks": {
            "october_30day_projection_pass_at_5pct_margin":
                october_projection_pass_5,
            "every_observed_month_actual_return_ge_30pct_at_3pct_margin":
                objective,
            "every_observed_month_positive_at_3pct_margin": all(
                value["return_pct"] > 0 for value in
                results["margin_3pct"]["months"].values()),
            "combined_dd_le_10pct_at_5pct_margin":
                results["margin_5pct"]["combined"]["dd_pct"] <= 10,
        },
        "objective_pass": objective,
        "sources": {
            str(TARGET_SEP): digest(TARGET_SEP),
            str(TARGET_OCT): digest(TARGET_OCT),
            str(HOURLY): digest(HOURLY),
            str(TARGET_REPORT): digest(TARGET_REPORT),
            str(HOURLY_REPORT): digest(HOURLY_REPORT),
            str(Path(__file__).resolve()): digest(Path(__file__).resolve()),
        },
        "limitations": [
            "September selected the 15m group filter and is not a final holdout.",
            "October is one partial out-of-sample month.",
            "The 30-day economic gate is a projection, not a forecast.",
            "Gross results exclude fees at user request.",
        ],
    }
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True),
                   encoding="utf-8")
    print(json.dumps({
        "status": report["status"],
        "results": {margin: {
            "combined": compact(value["combined"]),
            "months": {month: compact(result)
                       for month, result in value["months"].items()},
            "projections": {month: result["economic_gate"]
                            ["projected_monthly_return_pct"]
                            for month, result in value["months"].items()},
            "exposure": value["exposure"],
        } for margin, value in results.items()},
        "checks": report["checks"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
