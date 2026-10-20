"""Portfolio audit for the pre-2024-stable bear/breakout/long route."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

import crypto_05_2r_precision_barbell_oct5_audit as legacy
import crypto_high_target_tree_oct5_audit as base
import crypto_material_exact_group_diagnostics_oct20 as groups
import crypto_material_exact_value_oct20_audit as value
import crypto_precision_be_runner_oct18_audit as exact
import crypto_profit_first_material_barbell_oct18_audit as prior
import crypto_profit_first_material_xperp_oct20_audit as refresh


ROOT = Path("reports/audit_2026_09_08")
OUT = ROOT / "crypto_material_exact_stable_route_oct20.json"
ROWS_OUT = ROOT / "crypto_material_exact_stable_route_xperp_oct20.pkl"
FRESH_OUT = ROOT / "crypto_material_exact_stable_route_xperp_oct20_fresh.pkl"
ROUTE = {"btc_regime": "bear", "family": "breakout", "direction": "LONG"}


def route(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame
    for field, expected in ROUTE.items():
        result = result[result[field].eq(expected)]
    return result.copy()


def gated(frame: pd.DataFrame, periods: tuple[str, ...], loss_pause: int,
          hourly_cap: int) -> list[dict]:
    chosen = route(base.stage(frame, periods))
    chosen["model_probability"] = 1.0
    chosen = value.with_fraction(chosen, value.FRACTIONS[0])
    return exact.ranked_runtime_gate(
        exact.rows_for(chosen, "material_exact_stable_route"),
        loss_pause=loss_pause, hourly_cap=hourly_cap)


def stage_value(rows: list[dict], fraction: float, margin: float) -> dict:
    return value.metrics_for(rows, fraction, margin)[0]


def floor_metric(items: list[dict]) -> float:
    months = [float(month) for item in items
              for month in item.get("monthly_return_pct", {}).values()]
    return min(months) if months else -999.0


def load_frames() -> tuple[pd.DataFrame, pd.DataFrame, float]:
    historical = exact.attach_runtime_runner(
        legacy.load_joined(), direct=False, activation_r=0.5, runner_r=2.0)
    old = prior.join_direct_outcomes(refresh.OLD_FAR, refresh.OLD_NEAR)
    direct = prior.join_direct_outcomes(refresh.NEW_FAR, refresh.NEW_NEAR)
    direct = direct[
        direct.symbol.isin(base.ROBUST_SYMBOLS)
        & direct.risk_pct.between(legacy.RISK_PCT_MIN, legacy.RISK_PCT_MAX,
                                  inclusive="both")].copy()
    direct["period"] = "xperp"
    original_cache = exact.DIRECT_CACHE
    exact.DIRECT_CACHE = ROOT / "xperp_15m_oct20_material"
    try:
        direct = exact.attach_runtime_runner(
            direct, direct=True, activation_r=0.5, runner_r=2.0)
    finally:
        exact.DIRECT_CACHE = original_cache
    return historical, direct, float(old.entry_time.max())


def main() -> None:
    historical, direct, cutoff = load_frames()
    candidates = []
    for loss_pause in value.LOSS_PAUSES:
        for hourly_cap in value.HOURLY_CAPS:
            staged_rows = {
                "fit_2020_2021": gated(historical, legacy.FIT, loss_pause, hourly_cap),
                "selection_2022h1": gated(historical, legacy.SELECT, loss_pause, hourly_cap),
                "confirmation_2022h2_2023": gated(
                    historical, legacy.CONFIRM, loss_pause, hourly_cap),
            }
            for fraction in value.FRACTIONS:
                metrics = {name: stage_value(rows, fraction, 0.03)
                           for name, rows in staged_rows.items()}
                candidates.append({
                    "near_fraction": fraction,
                    "loss_pause": loss_pause,
                    "hourly_cap": hourly_cap,
                    "development": metrics,
                    "development_floor_month_pct": round(
                        floor_metric(list(metrics.values())), 4),
                    "all_development_positive": all(
                        item["return_pct"] > 0 for item in metrics.values()),
                })
    candidates.sort(key=lambda item: (
        item["all_development_positive"], item["development_floor_month_pct"],
        -max(stage["dd_pct"] for stage in item["development"].values()),
        sum(stage["return_pct"] for stage in item["development"].values()),
    ), reverse=True)
    selected = candidates[0]
    fraction = float(selected["near_fraction"])
    loss_pause = int(selected["loss_pause"])
    hourly_cap = int(selected["hourly_cap"])

    holdouts = {}
    direct_values = {}
    fresh_values = {}
    direct_rows = gated(direct, ("xperp",), loss_pause, hourly_cap)
    fresh_frame = direct[direct.entry_time > cutoff].copy()
    fresh_rows = gated(fresh_frame, ("xperp",), loss_pause, hourly_cap)
    for margin in (0.03, 0.05):
        name = f"margin_{int(100 * margin)}pct"
        holdouts[name] = {
            "validation_2024": stage_value(gated(
                historical, legacy.VALIDATION, loss_pause, hourly_cap),
                fraction, margin),
            "later_2025_2026": stage_value(gated(
                historical, legacy.HIDDEN, loss_pause, hourly_cap),
                fraction, margin),
        }
        direct_values[name] = stage_value(direct_rows, fraction, margin)
        fresh_values[name] = stage_value(fresh_rows, fraction, margin)
    pd.DataFrame(value.reprice_rows(direct_rows, fraction)).to_pickle(ROWS_OUT)
    pd.DataFrame(value.reprice_rows(fresh_rows, fraction)).to_pickle(FRESH_OUT)

    direct3 = direct_values["margin_3pct"]
    objective = bool(
        selected["development_floor_month_pct"] >= 30
        and all(part["negative_months"] == 0
                for margin in holdouts.values() for part in margin.values())
        and direct3["negative_months"] == 0
        and direct3["trades_per_calendar_day"] >= 3)
    report = {
        "status": ("STABLE_ROUTE_SHADOW_CANDIDATE" if objective
                   else "STABLE_ROUTE_REJECTED_OBJECTIVE"),
        "ready_for_live_orders": False,
        "route_selected_before_2024": ROUTE,
        "protocol": {
            "route_selection": "positive raw exact mean R in fit, selection and confirmation",
            "parameter_selection": "worst development month, then drawdown and return",
            "exit": "selected fraction at +0.5R; remaining stop to entry; runner +2R",
            "money": "client 3% or 5% current-equity margin; fixed 10x",
        },
        "candidate_count": len(candidates),
        "selected_before_holdouts": selected,
        "holdouts": holdouts,
        "direct_xperp": direct_values,
        "fresh_after_oct17_cutoff": fresh_values,
        "objective_pass": objective,
        "sources": {
            str(groups.OUT): groups.digest(groups.OUT),
            str(Path(__file__).resolve()): groups.digest(Path(__file__).resolve()),
        },
    }
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({
        "status": report["status"], "selected": selected,
        "holdouts": holdouts, "direct": direct_values, "fresh": fresh_values,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
