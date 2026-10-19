"""Weekly causal refit of the native X-Perp return model.

Each seven-day test block uses a fourteen-day calibration window.  Candidate
models are trained on several trailing windows ending before calibration, with
outcomes required to close before that boundary.  Calibration chooses the
model window, score threshold and account rails; an unprofitable calibration
keeps the following week flat.
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


OUT = parent.ROOT / "crypto_xperp_weekly_rolling_model_oct20.json"
ROWS_OUT = parent.ROOT / "crypto_xperp_weekly_rolling_model_oct20.pkl"
TRAIN_WINDOWS_DAYS = (21, 30, 45, 60, 90, None)
CALIBRATION_DAYS = 14
TEST_DAYS = 7
FIRST_TEST = pd.Timestamp("2026-06-01", tz="UTC")
END_TEST = pd.Timestamp("2026-10-17", tz="UTC")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def causal_parts(frame: pd.DataFrame, test_start: pd.Timestamp,
                 test_end: pd.Timestamp,
                 train_window_days: int | None) -> tuple[pd.DataFrame,
                                                         pd.DataFrame,
                                                         pd.DataFrame]:
    start = test_start.timestamp()
    end = test_end.timestamp()
    calibration_start = (
        test_start - pd.Timedelta(days=CALIBRATION_DAYS)).timestamp()
    train = frame[frame.exit_time < calibration_start].copy()
    if train_window_days is not None:
        lower = calibration_start - train_window_days * 86400
        train = train[train.entry_time >= lower].copy()
    calibration = frame[
        frame.entry_time.ge(calibration_start)
        & frame.entry_time.lt(start)
        & frame.exit_time.lt(start)
    ].copy()
    test = frame[
        frame.entry_time.ge(start) & frame.entry_time.lt(end)].copy()
    return train, calibration, test


def choice_key(item: dict) -> tuple:
    policy = item["policy"]
    return monthly.policy_key({
        "metrics": policy["metrics"],
        "hourly_cap": policy["hourly_cap"],
    })


def should_trade(policy: dict) -> bool:
    value = policy["metrics"]
    return bool(value["return_pct"] > 0
                and value["negative_months"] == 0
                and value["dd_pct"] <= 10
                and value["trades_per_calendar_day"] >= 3)


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

    steps = []
    accepted_rows = []
    test_start = FIRST_TEST
    while test_start < END_TEST:
        test_end = min(test_start + pd.Timedelta(days=TEST_DAYS), END_TEST)
        choices = []
        for window_days in TRAIN_WINDOWS_DAYS:
            train, calibration, test = causal_parts(
                direct, test_start, test_end, window_days)
            if len(train) < 1000 or len(calibration) < 500:
                continue
            schema = parent.design(train)
            model = geometry_audit.fit_model(
                train, schema, "signed_return")
            scored = geometry_audit.score(
                pd.concat([calibration, test], ignore_index=True),
                model, schema, "signed_return")
            calibration_scored = scored[
                scored.entry_time < test_start.timestamp()].copy()
            test_scored = scored[
                scored.entry_time >= test_start.timestamp()].copy()
            policy, frontier = monthly.choose_policy(calibration_scored)
            choices.append({
                "window_days": window_days,
                "train_rows": len(train),
                "calibration_rows": len(calibration),
                "policy": policy,
                "calibration_frontier": frontier,
                "_test": test_scored,
            })
        choices.sort(key=choice_key, reverse=True)
        selected = choices[0] if choices else None
        traded = bool(selected and should_trade(selected["policy"]))
        test_metrics = None
        if traded:
            policy = selected["policy"]
            test_metrics, rows = parent.evaluate(
                selected["_test"], float(policy["threshold"]),
                int(policy["loss_pause"]), int(policy["hourly_cap"]), .03)
            accepted_rows.extend(rows)
        steps.append({
            "test_start": test_start.isoformat(),
            "test_end": test_end.isoformat(),
            "traded": traded,
            "selected_window_days": (
                selected["window_days"] if selected else None),
            "selected_calibration": (
                selected["policy"]["metrics"] if selected else None),
            "selected_policy": ({key: selected["policy"][key] for key in (
                "quantile", "threshold", "loss_pause", "hourly_cap")}
                if selected else None),
            "window_calibrations": [{
                "window_days": item["window_days"],
                "train_rows": item["train_rows"],
                "calibration_rows": item["calibration_rows"],
                "policy": item["policy"],
            } for item in choices],
            "test_metrics_3pct": test_metrics,
        })
        test_start = test_end

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
        "status": ("XPERP_WEEKLY_ROLLING_MODEL_SHADOW_CANDIDATE"
                   if objective else "REJECTED_OBJECTIVE"),
        "ready_for_live_orders": False,
        "protocol": {
            "geometry": {"target_pct": 100 * geometry[0],
                         "stop_pct": 100 * geometry[1],
                         "hold_bars": geometry[2]},
            "train_windows_days": list(TRAIN_WINDOWS_DAYS),
            "calibration_days": CALIBRATION_DAYS,
            "test_days": TEST_DAYS,
            "training_cutoff": "outcome exit_time strictly before calibration",
            "flat_rule": "skip week after unprofitable or invalid calibration",
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
            "Fourteen days remain a short calibration sample.",
            "October is incomplete and remains diagnostic.",
            "Gross returns exclude fees at user request.",
            "No backtest guarantees a future monthly return.",
        ],
    }
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True),
                   encoding="utf-8")
    print(json.dumps({
        "status": report["status"],
        "weeks": len(steps),
        "traded_weeks": sum(step["traded"] for step in steps),
        "selected_windows": [step["selected_window_days"] for step in steps],
        "combined": combined,
        "objective_pass": objective,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
