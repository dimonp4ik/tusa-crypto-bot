"""Causal monthly adaptation of the selected raw crypto geometry to X-Perp.

The 1% target / 0.75% stop / 12-hour geometry and model family were frozen by
the historical audit.  For each direct test month, the model sees historical
data through 2025 plus only older X-Perp outcomes.  The immediately preceding
X-Perp month selects the cutoff and rails; the next month is untouched test.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

import crypto_raw_geometry_walkforward_oct20_audit as geometry_audit
import crypto_raw_hourly_direction_oct19_audit as parent
from crypto_runtime_shadow_portfolio_oct3_audit import money


OUT = parent.ROOT / "crypto_xperp_monthly_adaptive_oct20.json"
ROWS_OUT = parent.ROOT / "crypto_xperp_monthly_adaptive_oct20.pkl"
TEST_MONTHS = ("2026-06", "2026-07", "2026-08", "2026-09", "2026-10")
QUANTILES = (.30, .40, .50, .60, .70, .75, .80, .85, .875, .90, .925, .95)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def previous_month(value: str) -> str:
    return str(pd.Period(value, freq="M") - 1)


def policy_key(item: dict) -> tuple:
    metrics = item["metrics"]
    return (
        metrics["trades_per_calendar_day"] >= 3,
        metrics["dd_pct"] <= 10,
        metrics["negative_months"] == 0,
        metrics["return_pct"],
        -metrics["dd_pct"],
        -item["hourly_cap"],
    )


def choose_policy(calibration: pd.DataFrame) -> tuple[dict, list[dict]]:
    policies = []
    for quantile in QUANTILES:
        threshold = float(calibration.model_quality.quantile(quantile))
        for loss_pause in parent.LOSS_PAUSES:
            for hourly_cap in parent.HOURLY_CAPS:
                metrics, _ = parent.evaluate(
                    calibration, threshold, loss_pause, hourly_cap, .03)
                policies.append({
                    "quantile": quantile, "threshold": threshold,
                    "loss_pause": loss_pause, "hourly_cap": hourly_cap,
                    "metrics": metrics,
                })
    policies.sort(key=policy_key, reverse=True)
    return policies[0], policies[:10]


def main() -> None:
    frozen_report = json.loads(geometry_audit.OUT.read_text(encoding="utf-8"))
    frozen = frozen_report["selected_before_holdouts"]
    geometry = geometry_audit.geometry_tuple(frozen)
    target_name = str(frozen["model_target"])

    base_history = parent.build_history()
    historical = geometry_audit.relabel(base_history, geometry)
    historical_train = historical[~historical.period.eq("2026")].copy()

    direct_parts = []
    for symbol in parent.CORE_SYMBOLS:
        path = parent.rails.DIRECT_CACHE / f"{symbol}_15min_current.pkl"
        direct_parts.append(parent.build_symbol_period(
            "xperp", symbol, parent.load_candles(path)))
    direct = geometry_audit.relabel(
        pd.concat(direct_parts, ignore_index=True), geometry, direct=True)
    direct["month"] = pd.to_datetime(
        direct.entry_time, unit="s", utc=True).dt.strftime("%Y-%m")

    steps = []
    accepted_rows = []
    for test_month in TEST_MONTHS:
        calibration_month = previous_month(test_month)
        older_direct = direct[direct.month < calibration_month].copy()
        train = pd.concat([historical_train, older_direct], ignore_index=True)
        schema = parent.design(train)
        model = geometry_audit.fit_model(train, schema, target_name)
        scoring = direct[direct.month.isin(
            (calibration_month, test_month))].copy()
        scored = geometry_audit.score(scoring, model, schema, target_name)
        calibration = scored[scored.month.eq(calibration_month)].copy()
        test = scored[scored.month.eq(test_month)].copy()
        policy, frontier = choose_policy(calibration)
        metrics, rows = parent.evaluate(
            test, float(policy["threshold"]), int(policy["loss_pause"]),
            int(policy["hourly_cap"]), .03)
        accepted_rows.extend(rows)
        steps.append({
            "train_historical_periods": sorted(
                historical_train.period.unique().tolist()),
            "train_direct_before": calibration_month,
            "train_rows": len(train),
            "calibration_month": calibration_month,
            "test_month": test_month,
            "policy": policy,
            "calibration_frontier": frontier,
            "test_metrics_3pct": metrics,
        })

    # Preserve chronological symbol ownership and cooldown across month edges.
    combined_rows = parent.rails.ranked_runtime_gate(
        accepted_rows, loss_pause=999, hourly_cap=2)
    pd.DataFrame(combined_rows).to_pickle(ROWS_OUT)
    combined = {
        "margin_3pct": money(combined_rows, .03),
        "margin_5pct": money(combined_rows, .05),
    }
    complete_months = ("2026-06", "2026-07", "2026-08", "2026-09")
    five = combined["margin_5pct"]
    values = five.get("monthly_return_pct", {})
    objective = bool(
        set(complete_months).issubset(values)
        and min(float(values[month]) for month in complete_months) >= 30
        and five["dd_pct"] <= 10
        and five["trades_per_calendar_day"] >= 3
        and five["negative_months"] == 0)
    report = {
        "status": ("XPERP_MONTHLY_ADAPTIVE_SHADOW_CANDIDATE"
                   if objective else "REJECTED_OBJECTIVE"),
        "ready_for_live_orders": False,
        "protocol": {
            "frozen_geometry": {
                "target_pct": 100 * geometry[0],
                "stop_pct": 100 * geometry[1],
                "hold_bars": geometry[2],
                "model_target": target_name,
            },
            "historical_train": "2020-2025 only",
            "direct_train": "strictly before calibration month",
            "selection": "immediately preceding X-Perp month",
            "test_months": list(TEST_MONTHS),
            "entry": "market at next 15-minute open; hourly decisions",
            "orders": "market only",
            "money": "client 3% or 5% current-equity margin; fixed 10x",
            "priority": "calibration return first after frequency and DD rails",
        },
        "walkforward": steps,
        "combined": combined,
        "objective_complete_months": list(complete_months),
        "objective_pass": objective,
        "sources": {
            str(geometry_audit.OUT): digest(geometry_audit.OUT),
            **geometry_audit.SOURCE_HASHES,
            str(Path(__file__).resolve()): digest(Path(__file__).resolve()),
        },
        "limitations": [
            "April is partial and used only as older adaptation data.",
            "October is incomplete and remains diagnostic.",
            "Monthly threshold selection can be unstable across regime changes.",
            "Gross results exclude fees at user request.",
            "No backtest guarantees a future monthly return.",
        ],
    }
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({
        "status": report["status"],
        "steps": [{"test_month": step["test_month"],
                   "policy": {key: step["policy"][key] for key in (
                       "quantile", "threshold", "loss_pause", "hourly_cap")},
                   "metrics": step["test_metrics_3pct"]} for step in steps],
        "combined": combined,
        "objective_pass": objective,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
