"""Monthly causal ensemble for the native X-Perp geometry.

The return regressor supplies direction-quality ranking.  A separately fit
profitable-exit classifier acts as a veto.  Both models see only direct venue
rows older than the calibration month; that month selects both thresholds and
the next month is evaluated without retuning.
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


OUT = parent.ROOT / "crypto_xperp_dual_model_oct20.json"
ROWS_OUT = parent.ROOT / "crypto_xperp_dual_model_oct20.pkl"
PROFIT_QUANTILES = (0.0, .30, .50, .60, .70, .80, .90)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dual_score(frame: pd.DataFrame, return_model, profit_model,
               schema: dict) -> pd.DataFrame:
    result = frame.copy()
    values = parent.matrix(result, schema)
    result["return_score"] = return_model.predict(values)
    result["profit_score"] = profit_model.predict_proba(values)[:, 1]
    result["model_quality"] = result["return_score"]
    return result


def gated(frame: pd.DataFrame, profit_threshold: float) -> pd.DataFrame:
    return frame[frame.profit_score >= profit_threshold].copy()


def choose_policy(calibration: pd.DataFrame) -> tuple[dict, list[dict]]:
    policies = []
    for return_quantile in monthly.QUANTILES:
        return_threshold = float(
            calibration.return_score.quantile(return_quantile))
        for profit_quantile in PROFIT_QUANTILES:
            profit_threshold = float(
                calibration.profit_score.quantile(profit_quantile))
            eligible = gated(calibration, profit_threshold)
            for loss_pause in parent.LOSS_PAUSES:
                for hourly_cap in parent.HOURLY_CAPS:
                    metrics, _ = parent.evaluate(
                        eligible, return_threshold, loss_pause,
                        hourly_cap, .03)
                    policies.append({
                        "return_quantile": return_quantile,
                        "return_threshold": return_threshold,
                        "profit_quantile": profit_quantile,
                        "profit_threshold": profit_threshold,
                        "loss_pause": loss_pause,
                        "hourly_cap": hourly_cap,
                        "metrics": metrics,
                    })
    policies.sort(key=lambda item: monthly.policy_key({
        "metrics": item["metrics"],
        "hourly_cap": item["hourly_cap"],
    }), reverse=True)
    return policies[0], policies[:10]


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
        train = monthly.causal_training_rows(direct, calibration_month)
        if len(train) < 1000:
            raise RuntimeError(
                f"insufficient direct training rows before {calibration_month}")
        schema = parent.design(train)
        return_model = geometry_audit.fit_model(
            train, schema, "signed_return")
        profit_model = geometry_audit.fit_model(
            train, schema, "profitable_exit")
        scoring = direct[direct.month.isin(
            (calibration_month, test_month))].copy()
        scored = dual_score(scoring, return_model, profit_model, schema)
        calibration = scored[scored.month.eq(calibration_month)].copy()
        test = scored[scored.month.eq(test_month)].copy()
        policy, frontier = choose_policy(calibration)
        test_eligible = gated(test, float(policy["profit_threshold"]))
        metrics, rows = parent.evaluate(
            test_eligible, float(policy["return_threshold"]),
            int(policy["loss_pause"]), int(policy["hourly_cap"]), .03)
        accepted_rows.extend(rows)
        steps.append({
            "train_direct_before": calibration_month,
            "train_months": sorted(train.month.unique().tolist()),
            "train_rows": len(train),
            "calibration_month": calibration_month,
            "test_month": test_month,
            "policy": policy,
            "calibration_frontier": frontier,
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
    monthly_values = three.get("monthly_return_pct", {})
    objective = bool(
        set(complete_months).issubset(monthly_values)
        and min(float(monthly_values[month])
                for month in complete_months) >= 30
        and three["negative_months"] == 0
        and three["dd_pct"] <= 10
        and three["trades_per_calendar_day"] >= 3)
    report = {
        "status": ("XPERP_DUAL_MODEL_SHADOW_CANDIDATE"
                   if objective else "REJECTED_OBJECTIVE"),
        "ready_for_live_orders": False,
        "protocol": {
            "geometry": {"target_pct": 100 * geometry[0],
                         "stop_pct": 100 * geometry[1],
                         "hold_bars": geometry[2]},
            "models": ["signed_return regressor",
                       "profitable_exit classifier veto"],
            "model_train": "all X-Perp rows strictly before calibration month",
            "selection": "immediately preceding X-Perp month",
            "test_months": list(monthly.TEST_MONTHS),
            "profit_quantiles": list(PROFIT_QUANTILES),
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
            "The earliest model has only partial April direct history.",
            "October is incomplete and remains diagnostic.",
            "The direct venue history does not cover a full market cycle.",
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
            "policy": {key: step["policy"][key] for key in (
                "return_quantile", "profit_quantile", "loss_pause",
                "hourly_cap")},
            "metrics": step["test_metrics_3pct"],
        } for step in steps],
        "combined": combined,
        "objective_pass": objective,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
