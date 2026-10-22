"""Select a coarse exact-exit core on 2022-2026, then reveal X-Perp.

This is a final transfer test, not a search for live order logic.  Policies
use only direction, the frozen family label, BTC regime, UTC session, and one
large-cap symbol.  Direct X-Perp is evaluated only after one historical policy
has been selected.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

import crypto_2r_precision_barbell_oct5_audit as precision
import crypto_exact_exit_geometry_oct20_audit as geometry
import crypto_exact_exit_regime_route_oct22_audit as regime
import crypto_high_target_tree_oct5_audit as base
import crypto_precision_be_runner_oct18_audit as exact
import crypto_precision_profit_first_oct18_audit as prior


ROOT = Path("reports/audit_2026_09_08")
REGIME_REPORT = ROOT / "crypto_exact_exit_regime_route_oct22.json"
OUT = ROOT / "crypto_exact_exit_stable_core_oct22.json"
ROWS_OUT = ROOT / "crypto_exact_exit_stable_core_oct22_rows.pkl"
POLICIES = (
    "short_bear",
    "short_trend_pullback",
    "short_bear_or_trend",
    "short_bear_or_trend_or_evening",
    "short_bear_or_trend_or_breakout_eth",
    "short_bear_or_trend_or_evening_or_breakout_eth",
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def apply_policy(frame: pd.DataFrame, policy: str) -> pd.DataFrame:
    short = frame.direction.eq("SHORT")
    bear = frame.btc_regime.eq("bear")
    trend = frame.family.eq("trend_pullback")
    evening = frame.utc_session.eq("18_23")
    breakout_eth = frame.family.eq("breakout") & frame.symbol.eq("ETHUSDT")
    masks = {
        "short_bear": short & bear,
        "short_trend_pullback": short & trend,
        "short_bear_or_trend": short & (bear | trend),
        "short_bear_or_trend_or_evening": short & (bear | trend | evening),
        "short_bear_or_trend_or_breakout_eth": (
            (short & (bear | trend)) | breakout_eth),
        "short_bear_or_trend_or_evening_or_breakout_eth": (
            (short & (bear | trend | evening)) | breakout_eth),
    }
    if policy not in masks:
        raise ValueError(f"unknown stable-core policy: {policy}")
    return frame[masks[policy]].copy()


def evaluate(scored: pd.DataFrame, periods: tuple[str, ...], threshold: float,
             loss_pause: int, hourly_cap: int, policy: str) -> tuple[dict, list[dict]]:
    chosen = base.stage(scored, periods)
    chosen = chosen[chosen.model_probability >= threshold].copy()
    chosen = chosen.sort_values(
        ["symbol", "entry_time", "model_probability", "family", "direction"],
        ascending=[True, True, False, False, False], kind="mergesort")
    chosen = chosen.drop_duplicates(["symbol", "entry_time"], keep="first")
    chosen = apply_policy(chosen, policy)
    rows = exact.ranked_runtime_gate(
        exact.rows_for(chosen, "exact_exit_stable_core"),
        loss_pause=loss_pause, hourly_cap=hourly_cap)
    value = regime.money(rows, .03)
    value.update({
        "target_rate": round(100 * sum(
            row["runner_outcome"] == "TP2" for row in rows) / len(rows), 2)
        if rows else 0.0,
        "breakeven_rate": round(100 * sum(
            row["runner_outcome"] == "BE" for row in rows) / len(rows), 2)
        if rows else 0.0,
        "non_loss_rate": round(100 * sum(
            float(row["net_r"]) >= 0 for row in rows) / len(rows), 2)
        if rows else 0.0,
        "actual_max_loss_streak": exact.maximum_stop_streak(rows),
    })
    return value, rows


def admissible(metrics: dict[str, dict]) -> bool:
    values = list(metrics.values())
    return bool(
        min(value["n"] for value in values) >= 50
        and min(value["return_pct"] for value in values) > 0
        and max(value["dd_pct"] for value in values) <= 8
        and max(value["negative_months"] for value in values) <= 4
    )


def rank(metrics: dict[str, dict]) -> tuple[float, float, float]:
    values = list(metrics.values())
    return (
        min(value["return_pct"] for value in values),
        sum(value["return_pct"] for value in values),
        min(value["trades_per_calendar_day"] for value in values),
    )


def main() -> None:
    prior_report = json.loads(REGIME_REPORT.read_text(encoding="utf-8"))
    frozen = prior_report["frozen_geometry"]
    target_r = float(frozen["target_r"])
    activation_r = (99.0 if frozen["activation_r"] is None
                    else float(frozen["activation_r"]))
    threshold = float(frozen["threshold"])
    loss_pause = int(frozen["loss_pause"])
    hourly_cap = int(frozen["hourly_cap"])

    historical = prior.canonical_history(precision.load_joined())
    historical_target = geometry.eligible(historical, target_r)
    replayed = exact.attach_runtime_runner(
        historical_target, direct=False,
        activation_r=activation_r, runner_r=target_r)
    fit = base.stage(replayed, precision.FIT)
    design = base.design(fit)
    model = geometry.fit_model(fit, design)
    scored = geometry.score(replayed, model, design)

    stages = {
        "selection_2022h1": precision.SELECT,
        "confirmation_2022h2_2023": precision.CONFIRM,
        "development_2024": precision.VALIDATION,
        "development_2025": ("2025",),
        "development_2026": ("2026",),
    }
    historical_results: dict[str, dict[str, dict]] = {}
    for policy in POLICIES:
        historical_results[policy] = {}
        for name, periods in stages.items():
            historical_results[policy][name], _ = evaluate(
                scored, periods, threshold, loss_pause, hourly_cap, policy)
    accepted = [policy for policy in POLICIES
                if admissible(historical_results[policy])]
    if not accepted:
        raise RuntimeError("no stable-core policy passes the historical rails")
    selected = max(accepted, key=lambda item: rank(historical_results[item]))

    direct_frame = geometry.eligible(prior.direct_frame(), target_r)
    direct_replayed = exact.attach_runtime_runner(
        direct_frame, direct=True,
        activation_r=activation_r, runner_r=target_r)
    direct_scored = geometry.score(direct_replayed, model, design)
    direct, direct_rows = evaluate(
        direct_scored, ("xperp",), threshold,
        loss_pause, hourly_cap, selected)
    direct_passed = bool(
        direct["n"] >= 50
        and direct["return_pct"] > 0
        and direct["dd_pct"] <= 8
        and direct["negative_months"] <= 1)
    for row in direct_rows:
        row["audit_cohort"] = "direct_xperp"
    pd.DataFrame(direct_rows).to_pickle(ROWS_OUT)

    report = {
        "status": ("STABLE_CORE_AWAITING_NEW_FORWARD_AFTER_2026_10_22"
                   if direct_passed else "REJECTED_DIRECT_TRANSFER"),
        "ready_for_live_orders": False,
        "order_eligible": False,
        "frozen_geometry": frozen,
        "selection_protocol": {
            "policy_universe": list(POLICIES),
            "historical_periods": ["2022h1", "2022h2", "2023", "2024", "2025", "2026"],
            "admission": (
                "positive in every period, >=50 trades/period, <=8% drawdown, "
                "<=4 negative months per period"),
            "rank": "maximize worst-period return, then total return and frequency",
            "direct_holdout": "X-Perp through October 17, 2026",
            "entry": "next 15-minute market open",
            "exit": "exact ordered OHLC; 1R target, BE after 0.75R, stop-first",
            "money": "client-selected 3% current-equity margin, fixed 10x gross",
        },
        "historical_results": historical_results,
        "admissible_policies": accepted,
        "selected_before_direct": selected,
        "direct_xperp": direct,
        "direct_xperp_passed": direct_passed,
        "objective_checks": {
            "direct_transfer_passed": direct_passed,
            "at_least_3_trades_per_calendar_day": (
                direct["trades_per_calendar_day"] >= 3),
            "every_direct_month_at_least_30pct": bool(
                direct["monthly_return_pct"]
                and min(direct["monthly_return_pct"].values()) >= 30),
            "ready_for_live_orders": False,
        },
        "sources": {
            str(REGIME_REPORT): digest(REGIME_REPORT),
            str(Path(__file__).resolve()): digest(Path(__file__).resolve()),
            **exact.CANDLE_SOURCES,
        },
        "limitations": [
            "The stable core uses all historical periods through 2026; only direct X-Perp is a transfer holdout.",
            "Direct X-Perp is shorter than a full market cycle.",
            "Gross results exclude fees at user request.",
            "No backtest guarantees a future monthly return.",
        ],
    }
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({
        "status": report["status"],
        "admissible_policies": accepted,
        "selected_before_direct": selected,
        "historical": historical_results[selected],
        "direct_xperp": direct,
        "objective_checks": report["objective_checks"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
