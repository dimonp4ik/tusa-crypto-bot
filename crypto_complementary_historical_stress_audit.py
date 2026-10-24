"""Later-year stress test for the complementary crypto portfolio.

The 1h entry/exit policy was selected on 2022-2023 and is replayed on later
calendar data.  It is combined without retuning with the independently
walk-forward 15m target-router rows for 2024-2026.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

import crypto_5m_regime_router_literal_oct6_audit as hourly_portfolio
import crypto_complementary_exact_portfolio_oct24_audit as recent
import crypto_okx_swap_1h_forward_append_oct16_audit as append
import crypto_okx_swap_1h_forward_append_oct8_audit as legacy
import crypto_okx_swap_1h_stable_direction_forward_oct22_audit as forward
import crypto_okx_swap_1h_stable_direction_oct22_audit as stable
import crypto_okx_swap_1h_stable_material_exit_oct22_audit as hourly


ROOT = recent.ROOT
TARGET_ROWS = ROOT / "crypto_profit_first_walkforward_target_router_oct18.pkl"
HOURLY_REPORT = ROOT / "crypto_okx_swap_1h_stable_material_exit_oct22.json"
STABLE_REPORT = ROOT / "crypto_okx_swap_1h_stable_direction_oct22.json"
OUT = ROOT / "crypto_complementary_historical_stress.json"
ROWS_OUT = ROOT / "crypto_complementary_historical_stress.pkl"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compact(value: dict) -> dict:
    return {name: value[name] for name in (
        "n", "return_pct", "dd_pct", "wr", "max_loss_streak",
        "trades_per_calendar_day", "negative_months",
        "monthly_return_pct")}


def main() -> None:
    material = json.loads(HOURLY_REPORT.read_text(encoding="utf-8"))
    stable_source = json.loads(STABLE_REPORT.read_text(encoding="utf-8"))
    selected = material["selected_before_later_years_and_direct"]
    policy = stable_source["selected_before_direct"]
    entries = stable.apply_policy(
        pd.read_pickle(hourly.FREQUENCY_ROWS), policy)
    candles, _, coverage = legacy.load_merged(
        (append.OLD_GLOBAL, forward.GLOBAL_MANIFEST),
        legacy.GLOBAL_FIELDS, 3600)
    resolved = hourly.add_hourly_outcomes(
        entries, candles, float(selected["target_pct"]),
        int(selected["hold_hours"]))
    target = pd.read_pickle(TARGET_ROWS).copy()
    target["entry_year"] = pd.to_datetime(
        target.entry_time, unit="s", utc=True).dt.year

    results = {}
    saved = []
    for year in (2024, 2025, 2026):
        hourly_rows = hourly_portfolio.portfolio_rows(
            resolved, (year,), 0)
        target_rows = target[target.entry_year.eq(year)].to_dict("records")
        combined = [*hourly_rows, *target_rows]
        saved.extend({**row, "stress_year": year} for row in combined)
        results[str(year)] = {
            "component_rows": {
                "hourly": len(hourly_rows), "target": len(target_rows)},
            "margin_3pct": recent.metrics(combined, .03),
            "margin_5pct": recent.metrics(combined, .05),
            "exposure_3pct": recent.exposure(combined, .03),
        }
    pd.DataFrame(saved).to_pickle(ROWS_OUT)
    all_months = [
        value for result in results.values()
        for value in result["margin_3pct"]["monthly_return_pct"].values()
    ]
    objective = bool(
        all_months and min(all_months) >= 30
        and all(result["margin_3pct"]["dd_pct"] <= 10
                for result in results.values()))
    report = {
        "status": ("HISTORICAL_COMPLEMENT_OBJECTIVE_PASSED" if objective
                   else "HISTORICAL_COMPLEMENT_OBJECTIVE_NOT_MET"),
        "ready_for_live_orders": False,
        "order_eligible": False,
        "protocol": {
            "hourly_selection": "2022-2023",
            "stress_years": [2024, 2025, 2026],
            "target_component": "independent chronological walk-forward rows",
            "portfolio_retuning": "none",
            "money": "3% or 5% current-equity margin per trade, fixed 10x",
        },
        "results": results,
        "checks": {
            "all_observed_months_positive_at_3pct_margin": bool(
                all_months and min(all_months) > 0),
            "all_observed_months_ge_30pct_at_3pct_margin": bool(
                all_months and min(all_months) >= 30),
            "all_year_drawdowns_le_10pct_at_3pct_margin": all(
                result["margin_3pct"]["dd_pct"] <= 10
                for result in results.values()),
        },
        "objective_pass": objective,
        "coverage": coverage,
        "sources": {
            str(TARGET_ROWS): digest(TARGET_ROWS),
            str(HOURLY_REPORT): digest(HOURLY_REPORT),
            str(STABLE_REPORT): digest(STABLE_REPORT),
            str(hourly.FREQUENCY_ROWS): digest(hourly.FREQUENCY_ROWS),
            str(forward.GLOBAL_MANIFEST): digest(forward.GLOBAL_MANIFEST),
            str(Path(__file__).resolve()): digest(Path(__file__).resolve()),
        },
        "limitations": [
            "The target-router research periods do not cover every calendar month.",
            "The hourly route's 2022-2023 period is selection evidence.",
            "Gross results exclude fees at user request.",
        ],
    }
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True),
                   encoding="utf-8")
    print(json.dumps({
        "status": report["status"],
        "results": {year: {
            "components": value["component_rows"],
            "margin_3pct": compact(value["margin_3pct"]),
            "margin_5pct": compact(value["margin_5pct"]),
        } for year, value in results.items()},
        "checks": report["checks"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
