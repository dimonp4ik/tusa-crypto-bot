"""No-retune complement test for the monthly-refit target router.

The exact monthly-refit rows are combined with the independently selected
stable hourly rows only on identical covered calendar months.  No portfolio
parameter is selected from the comparison period.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

import crypto_monthly_refit_target_router_audit as monthly


ROOT = monthly.ROOT
TARGET_ROWS = ROOT / "crypto_monthly_refit_target_router.pkl"
TARGET_REPORT = ROOT / "crypto_monthly_refit_target_router.json"
HISTORICAL_PORTFOLIO = ROOT / "crypto_complementary_historical_stress.pkl"
HISTORICAL_REPORT = ROOT / "crypto_complementary_historical_stress.json"
OUT = ROOT / "crypto_monthly_refit_complement.json"
ROWS_OUT = ROOT / "crypto_monthly_refit_complement.pkl"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def label(frame: pd.DataFrame) -> pd.Series:
    return pd.to_datetime(frame.entry_time, unit="s", utc=True).dt.strftime("%Y-%m")


def compact(value: dict) -> dict:
    return {name: value[name] for name in (
        "n", "return_pct", "dd_pct", "wr", "max_loss_streak",
        "trades_per_calendar_day", "negative_months",
        "monthly_return_pct", "literal_tp_rate_pct",
        "mean_signed_price_return_pct", "economic_gate")}


def main() -> None:
    target = pd.read_pickle(TARGET_ROWS).copy()
    target["covered_month"] = label(target)
    covered = set(target.covered_month.unique())
    prior = pd.read_pickle(HISTORICAL_PORTFOLIO).copy()
    hourly = prior[prior.source.eq("literal_regime_router")].copy()
    hourly["covered_month"] = label(hourly)
    hourly = hourly[hourly.covered_month.isin(covered)]
    combined = pd.concat([target, hourly], ignore_index=True, sort=False)
    combined.to_pickle(ROWS_OUT)

    results = {}
    baseline = {}
    for year in (2024, 2025, 2026):
        crows = combined[pd.to_datetime(
            combined.entry_time, unit="s", utc=True).dt.year.eq(year)].to_dict("records")
        trows = target[pd.to_datetime(
            target.entry_time, unit="s", utc=True).dt.year.eq(year)].to_dict("records")
        results[str(year)] = {
            f"margin_{int(100*margin)}pct": monthly.detailed_metrics(crows, margin)
            for margin in (.03, .05)}
        baseline[str(year)] = {
            f"margin_{int(100*margin)}pct": monthly.detailed_metrics(trows, margin)
            for margin in (.03, .05)}
    all_rows = combined.to_dict("records")
    total = {f"margin_{int(100*margin)}pct": monthly.detailed_metrics(
        all_rows, margin) for margin in (.03, .05)}
    months3 = total["margin_3pct"]["monthly_return_pct"]
    objective = bool(
        months3 and min(months3.values()) >= 30
        and total["margin_3pct"]["negative_months"] == 0
        and total["margin_3pct"]["dd_pct"] <= 10
        and total["margin_3pct"]["trades_per_calendar_day"] >= 3)
    improves_2025 = bool(
        results["2025"]["margin_3pct"]["return_pct"]
        > baseline["2025"]["margin_3pct"]["return_pct"]
        and results["2025"]["margin_3pct"]["dd_pct"]
        <= baseline["2025"]["margin_3pct"]["dd_pct"])
    report = {
        "status": ("MONTHLY_REFIT_COMPLEMENT_OBJECTIVE_PASSED" if objective
                   else "MONTHLY_REFIT_COMPLEMENT_REJECTED"),
        "ready_for_live_orders": False, "order_eligible": False,
        "protocol": {
            "selection_or_retuning": "none",
            "target": "monthly rolling exact target router",
            "complement": "independently selected stable hourly route",
            "comparison": "identical covered calendar months",
        },
        "results": results,
        "target_only": baseline,
        "combined": total,
        "checks": {"improves_2025_return_without_higher_dd": improves_2025,
                   "objective_pass": objective},
        "objective_pass": objective,
        "sources": {
            str(TARGET_ROWS): digest(TARGET_ROWS),
            str(TARGET_REPORT): digest(TARGET_REPORT),
            str(HISTORICAL_PORTFOLIO): digest(HISTORICAL_PORTFOLIO),
            str(HISTORICAL_REPORT): digest(HISTORICAL_REPORT),
            str(Path(__file__).resolve()): digest(Path(__file__).resolve()),
        },
        "limitations": [
            "Only months covered by the monthly target audit are compared.",
            "The hourly component was selected on earlier years.",
            "Gross results exclude fees at user request.",
        ],
    }
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({
        "status": report["status"],
        "results": {year: {key: compact(value) for key, value in result.items()}
                    for year, result in results.items()},
        "target_only": {year: {key: compact(value) for key, value in result.items()}
                        for year, result in baseline.items()},
        "checks": report["checks"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
