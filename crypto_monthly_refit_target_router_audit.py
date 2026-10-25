"""Monthly rolling refit of the exact 15-minute target router.

For every test month the three target regressors use only the preceding
24 months, excluding the immediately preceding calibration month.  That
calibration month selects the score cutoff and account rails.  The next month
is then evaluated once with exact 0.5R/1R/2R stop-first outcomes.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

import crypto_profit_first_walkforward_target_router_oct18_audit as base
from crypto_runtime_shadow_portfolio_oct3_audit import money
from src.research_profit_gate import economic_gate


ROOT = Path("reports/audit_2026_09_08")
BASELINE_ROWS = ROOT / "crypto_profit_first_walkforward_target_router_oct18.pkl"
OUT = ROOT / "crypto_monthly_refit_target_router.json"
ROWS_OUT = ROOT / "crypto_monthly_refit_target_router.pkl"
CHECKPOINT = ROOT / "crypto_monthly_refit_target_router_checkpoint.pkl"
ROLLING_MONTHS = 24
MIN_COMPLETE_ROWS = 1500
MAX_NEW_STEPS_PER_RUN = 4


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def month_label(frame: pd.DataFrame) -> pd.Series:
    return pd.to_datetime(frame.entry_time, unit="s", utc=True).dt.to_period("M")


def compact(value: dict) -> dict:
    return {name: value[name] for name in (
        "n", "return_pct", "dd_pct", "wr", "max_loss_streak",
        "trades_per_calendar_day", "negative_months",
        "monthly_return_pct")}


def detailed_metrics(rows: list[dict], margin: float) -> dict:
    value = money(rows, margin)
    n = len(rows)
    signed = [float(row["signed_price_return"]) for row in rows]
    timestamps = pd.Series([int(float(row["entry_time"])) for row in rows])
    clusters = timestamps.value_counts() if n else pd.Series(dtype=int)
    value.update({
        "literal_tp_rate_pct": round(100*sum(
            str(row.get("target_outcome")) == "TP2" for row in rows)/n, 4)
            if n else 0.0,
        "mean_signed_price_return_pct": round(
            100*sum(signed)/n, 6) if n else 0.0,
        "max_entries_same_timestamp": int(clusters.max()) if n else 0,
        "clustered_entry_share_pct": round(100*int(
            clusters[clusters > 1].sum())/n, 4) if n else 0.0,
        "economic_gate": economic_gate(
            signed, float(value["trades_per_calendar_day"]),
            margin_fraction=margin) if n else {"passes": False},
    })
    return value


def main() -> None:
    frame = base.historical_wide().copy()
    frame["month_period"] = month_label(frame)
    counts = frame.groupby("month_period").size()
    complete = set(counts[counts >= MIN_COMPLETE_ROWS].index)
    planned = []
    for month in sorted(complete):
        calibration = month-1
        if calibration not in complete or month.year < 2022:
            continue
        train_start = calibration-ROLLING_MONTHS
        train = frame[(frame.month_period >= train_start)
                      & (frame.month_period < calibration)].copy()
        if len(train) < 20000:
            continue
        planned.append((month, calibration, train_start))

    tests = pd.read_pickle(CHECKPOINT) if CHECKPOINT.exists() else []
    completed_months = {step["test_month"] for step in tests}
    new_steps = 0
    for month, calibration, train_start in planned:
        if str(month) in completed_months:
            continue
        train = frame[(frame.month_period >= train_start)
                      & (frame.month_period < calibration)].copy()
        scoring = frame[frame.month_period.isin(
            (calibration, month))].copy().reset_index(drop=True)
        predictions = base.fit_models_on_rows(train, scoring)
        routed = base.route(scoring, predictions)
        calibration_frame = routed[
            routed.month_period.eq(calibration)].copy()
        test_frame = routed[routed.month_period.eq(month)].copy()
        policy, frontier = base.choose_policy(calibration_frame)
        value, rows = base.evaluate(
            test_frame, float(policy["threshold"]),
            int(policy["loss_pause"]), int(policy["hourly_cap"]))
        tests.append({
            "test_month": str(month), "calibration_month": str(calibration),
            "train_start": str(train_start), "train_end_exclusive": str(calibration),
            "train_rows": len(train),
            "policy": {key: policy[key] for key in (
                "quantile", "threshold", "loss_pause", "hourly_cap",
                "score", "metrics")},
            "test_metrics": value, "rows": rows,
            "calibration_frontier": frontier,
        })
        pd.to_pickle(tests, CHECKPOINT)
        new_steps += 1
        print(json.dumps({"checkpoint": str(CHECKPOINT),
                          "completed": len(tests), "planned": len(planned),
                          "last_test_month": str(month)}), flush=True)
        if new_steps >= MAX_NEW_STEPS_PER_RUN:
            break

    if len(tests) < len(planned):
        print(json.dumps({
            "status": "MONTHLY_REFIT_CHECKPOINT_IN_PROGRESS",
            "completed": len(tests), "planned": len(planned),
            "remaining": len(planned)-len(tests),
        }, indent=2), flush=True)
        return

    all_rows = [row for step in tests for row in step.pop("rows")]
    pd.DataFrame(all_rows).to_pickle(ROWS_OUT)
    combined = {f"margin_{int(100*margin)}pct": detailed_metrics(all_rows, margin)
                for margin in (.03, .05)}
    years = {}
    for year in sorted({int(step["test_month"][:4]) for step in tests}):
        rows = [row for row in all_rows if pd.Timestamp(
            float(row["entry_time"]), unit="s", tz="UTC").year == year]
        years[str(year)] = {
            f"margin_{int(100*margin)}pct": detailed_metrics(rows, margin)
            for margin in (.03, .05)}

    baseline = pd.read_pickle(BASELINE_ROWS).copy()
    baseline["month_period"] = month_label(baseline)
    test_months = {pd.Period(step["test_month"], freq="M") for step in tests}
    baseline_rows = baseline[baseline.month_period.isin(
        test_months)].to_dict("records")
    baseline_metrics = {f"margin_{int(100*margin)}pct": detailed_metrics(
        baseline_rows, margin) for margin in (.03, .05)}

    months3 = combined["margin_3pct"]["monthly_return_pct"]
    objective = bool(
        months3 and min(months3.values()) >= 30
        and combined["margin_3pct"]["negative_months"] == 0
        and combined["margin_3pct"]["dd_pct"] <= 10
        and combined["margin_3pct"]["trades_per_calendar_day"] >= 3)
    report = {
        "status": ("MONTHLY_REFIT_OBJECTIVE_PASSED" if objective
                   else "MONTHLY_REFIT_OBJECTIVE_NOT_MET"),
        "ready_for_live_orders": False, "order_eligible": False,
        "protocol": {
            "fit_window_months": ROLLING_MONTHS,
            "fit_end": "before calibration month",
            "selection": "immediately preceding complete month",
            "test": "next complete month",
            "minimum_rows_for_complete_month": MIN_COMPLETE_ROWS,
            "targets_r": list(base.TARGETS),
            "entry": "next 15-minute market open",
            "exit": "exact stop-first target path",
            "retune_on_test": False,
        },
        "steps": tests,
        "combined": combined,
        "by_year": years,
        "coarse_walkforward_same_months": baseline_metrics,
        "objective_pass": objective,
        "sources": {
            **{str(base.base.cache_path(target, period)):
               digest(base.base.cache_path(target, period))
               for target in base.TARGETS for period in base.PERIODS},
            str(BASELINE_ROWS): digest(BASELINE_ROWS),
            str(Path(__file__).resolve()): digest(Path(__file__).resolve()),
        },
        "limitations": [
            "The historical snapshots have complete months only in their covered ranges.",
            "Monthly refitting cannot reproduce live latency or slippage.",
            "Gross results exclude fees at user request.",
        ],
    }
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({
        "status": report["status"], "test_months": len(tests),
        "first_test": tests[0]["test_month"], "last_test": tests[-1]["test_month"],
        "combined": {key: compact(value) for key, value in combined.items()},
        "by_year": {year: {key: compact(value) for key, value in result.items()}
                    for year, result in years.items()},
        "baseline": {key: compact(value) for key, value in baseline_metrics.items()},
        "objective_pass": objective,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
