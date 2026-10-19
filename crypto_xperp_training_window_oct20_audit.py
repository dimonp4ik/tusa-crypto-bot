"""Causal training-window selection for the native X-Perp model.

For every test month, multiple return models are trained on different trailing
windows ending before the calibration month.  The immediately preceding month
selects one model window, score cutoff and portfolio rails.  Only trades whose
outcomes closed before the training cutoff can enter a model.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

import crypto_raw_geometry_walkforward_oct20_audit as geometry_audit
import crypto_raw_hourly_direction_oct19_audit as parent
import crypto_xperp_monthly_adaptive_oct20_audit as monthly
from crypto_runtime_shadow_portfolio_oct3_audit import money


OUT = parent.ROOT / "crypto_xperp_training_window_oct20.json"
ROWS_OUT = parent.ROOT / "crypto_xperp_training_window_oct20.pkl"
WINDOW_DAYS = (21, 30, 45, 60, 90, None)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def causal_window(frame: pd.DataFrame, calibration_month: str,
                  window_days: int | None) -> pd.DataFrame:
    cutoff = pd.Period(calibration_month, freq="M").start_time.tz_localize(
        "UTC").timestamp()
    result = frame[frame.exit_time < cutoff].copy()
    if window_days is not None:
        lower = cutoff - window_days * 86400
        result = result[result.entry_time >= lower].copy()
    return result


def choice_key(item: dict) -> tuple:
    policy = item["policy"]
    return monthly.policy_key({
        "metrics": policy["metrics"],
        "hourly_cap": policy["hourly_cap"],
    })


def main() -> None:
    frozen_report = json.loads(
        geometry_audit.OUT.read_text(encoding="utf-8"))
    geometry = geometry_audit.geometry_tuple(
        frozen_report["selected_before_holdouts"])

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
    for test_month in monthly.TEST_MONTHS:
        calibration_month = monthly.previous_month(test_month)
        choices = []
        for window_days in WINDOW_DAYS:
            train = causal_window(direct, calibration_month, window_days)
            if len(train) < 1000:
                continue
            schema = parent.design(train)
            model = geometry_audit.fit_model(
                train, schema, "signed_return")
            scoring = direct[direct.month.isin(
                (calibration_month, test_month))].copy()
            scored = geometry_audit.score(
                scoring, model, schema, "signed_return")
            calibration = scored[
                scored.month.eq(calibration_month)].copy()
            test = scored[scored.month.eq(test_month)].copy()
            policy, frontier = monthly.choose_policy(calibration)
            choices.append({
                "window_days": window_days,
                "train_rows": len(train),
                "train_first_entry": int(train.entry_time.min()),
                "train_last_exit": int(train.exit_time.max()),
                "policy": policy,
                "calibration_frontier": frontier,
                "_test": test,
            })
        if not choices:
            raise RuntimeError(
                f"no training window for calibration {calibration_month}")
        choices.sort(key=choice_key, reverse=True)
        selected = choices[0]
        policy = selected["policy"]
        metrics, rows = parent.evaluate(
            selected["_test"], float(policy["threshold"]),
            int(policy["loss_pause"]), int(policy["hourly_cap"]), .03)
        accepted_rows.extend(rows)
        steps.append({
            "calibration_month": calibration_month,
            "test_month": test_month,
            "selected_window_days": selected["window_days"],
            "train_rows": selected["train_rows"],
            "train_first_entry": selected["train_first_entry"],
            "train_last_exit": selected["train_last_exit"],
            "policy": policy,
            "window_calibrations": [{
                "window_days": item["window_days"],
                "train_rows": item["train_rows"],
                "policy": item["policy"],
            } for item in choices],
            "test_metrics_3pct": metrics,
        })

    combined_rows = parent.rails.ranked_runtime_gate(
        accepted_rows, loss_pause=999, hourly_cap=2)
    pd.DataFrame(combined_rows).to_pickle(ROWS_OUT)
    combined = {
        "margin_3pct": money(combined_rows, .03),
        "margin_5pct": money(combined_rows, .05),
    }
    complete_months = ("2026-06", "2026-07", "2026-08", "2026-09")
    three = combined["margin_3pct"]
    values = three.get("monthly_return_pct", {})
    objective = bool(
        set(complete_months).issubset(values)
        and min(float(values[month]) for month in complete_months) >= 30
        and three["negative_months"] == 0
        and three["dd_pct"] <= 10
        and three["trades_per_calendar_day"] >= 3)
    report = {
        "status": ("XPERP_TRAINING_WINDOW_SHADOW_CANDIDATE"
                   if objective else "REJECTED_OBJECTIVE"),
        "ready_for_live_orders": False,
        "protocol": {
            "geometry": {"target_pct": 100 * geometry[0],
                         "stop_pct": 100 * geometry[1],
                         "hold_bars": geometry[2]},
            "training_windows_days": list(WINDOW_DAYS),
            "training_cutoff": "outcome exit_time strictly before calibration",
            "selection": "previous month chooses model window, cutoff and rails",
            "test_months": list(monthly.TEST_MONTHS),
            "entry": "market at next 15-minute open; hourly decisions",
            "orders": "market only",
            "money": "client 3% or 5% current-equity margin; fixed 10x",
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
            "The direct venue history begins in partial April 2026.",
            "Short windows are unavailable until at least 1,000 rows exist.",
            "October is incomplete and remains diagnostic.",
            "Gross returns exclude fees at user request.",
            "No backtest guarantees a future monthly return.",
        ],
    }
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True),
                   encoding="utf-8")
    print(json.dumps({
        "status": report["status"],
        "steps": [{
            "test_month": step["test_month"],
            "selected_window_days": step["selected_window_days"],
            "calibration_return_pct": step["policy"]["metrics"]["return_pct"],
            "test_metrics_3pct": step["test_metrics_3pct"],
        } for step in steps],
        "combined": combined,
        "objective_pass": objective,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
