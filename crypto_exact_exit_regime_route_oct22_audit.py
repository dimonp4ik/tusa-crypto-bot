"""Freeze a simple direction/regime route through 2024, then reveal 2025+.

The exact 1R exit candidate is inherited from the balanced holdout audit.
Only four coarse routing policies are compared on 2022-2024.  The selected
policy is rebuilt before runtime admission so excluded signals cannot consume
the hourly cap or suppress another eligible setup.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

import crypto_2r_precision_barbell_oct5_audit as precision
import crypto_exact_exit_geometry_oct20_audit as geometry
import crypto_high_target_tree_oct5_audit as base
import crypto_precision_be_runner_oct18_audit as exact
import crypto_precision_profit_first_oct18_audit as prior
from crypto_runtime_shadow_portfolio_oct3_audit import money


ROOT = Path("reports/audit_2026_09_08")
BALANCED_REPORT = ROOT / "crypto_exact_exit_balanced_holdout_oct22.json"
OUT = ROOT / "crypto_exact_exit_regime_route_oct22.json"
ROWS_OUT = ROOT / "crypto_exact_exit_regime_route_oct22_rows.pkl"
POLICIES = (
    "short_only",
    "short_plus_long_12_17",
    "short_plus_long_bull",
    "short_plus_long_range_bull",
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def apply_policy(frame: pd.DataFrame, policy: str) -> pd.DataFrame:
    short = frame.direction.eq("SHORT")
    long = frame.direction.eq("LONG")
    if policy == "short_only":
        mask = short
    elif policy == "short_plus_long_12_17":
        mask = short | (long & frame.utc_session.eq("12_17"))
    elif policy == "short_plus_long_bull":
        mask = short | (long & frame.btc_regime.eq("bull"))
    elif policy == "short_plus_long_range_bull":
        mask = short | (
            long & frame.family.eq("range_reversion")
            & frame.btc_regime.eq("bull"))
    else:
        raise ValueError(f"unknown route policy: {policy}")
    return frame[mask].copy()


def evaluate(scored: pd.DataFrame, periods: tuple[str, ...], threshold: float,
             loss_pause: int, hourly_cap: int, policy: str,
             margin: float = .03) -> tuple[dict, list[dict]]:
    chosen = base.stage(scored, periods)
    chosen = chosen[chosen.model_probability >= threshold].copy()
    chosen = chosen.sort_values(
        ["symbol", "entry_time", "model_probability", "family", "direction"],
        ascending=[True, True, False, False, False], kind="mergesort")
    chosen = chosen.drop_duplicates(["symbol", "entry_time"], keep="first")
    chosen = apply_policy(chosen, policy)
    rows = exact.ranked_runtime_gate(
        exact.rows_for(chosen, "exact_exit_regime_route"),
        loss_pause=loss_pause, hourly_cap=hourly_cap)
    value = money(rows, margin)
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


def development_pass(metrics: dict[str, dict]) -> bool:
    return bool(
        min(value["n"] for value in metrics.values()) >= 400
        and min(value["trades_per_calendar_day"] for value in metrics.values()) >= 2.9
        and min(value["return_pct"] for value in metrics.values()) > 0
        and max(value["dd_pct"] for value in metrics.values()) <= 8
        and max(value["negative_months"] for value in metrics.values()) <= 2
    )


def policy_rank(metrics: dict[str, dict]) -> tuple[float, float, float]:
    values = list(metrics.values())
    return (
        min(value["return_pct"] for value in values),
        sum(value["return_pct"] for value in values),
        -max(value["dd_pct"] for value in values),
    )


def future_pass(value: dict, maximum_negative_months: int) -> bool:
    return bool(
        value["n"] >= 100
        and value["trades_per_calendar_day"] >= 2.9
        and value["return_pct"] > 0
        and value["dd_pct"] <= 10
        and value["negative_months"] <= maximum_negative_months
    )


def main() -> None:
    balanced = json.loads(BALANCED_REPORT.read_text(encoding="utf-8"))
    frozen = balanced["selected_before_holdouts"]
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
    }
    policy_results: dict[str, dict[str, dict]] = {}
    for policy in POLICIES:
        policy_results[policy] = {}
        for name, periods in stages.items():
            policy_results[policy][name], _ = evaluate(
                scored, periods, threshold, loss_pause, hourly_cap, policy)

    admissible = [policy for policy in POLICIES
                  if development_pass(policy_results[policy])]
    if not admissible:
        raise RuntimeError("no coarse regime policy passes development rails")
    selected = max(admissible, key=lambda policy: policy_rank(policy_results[policy]))

    later, later_rows = evaluate(
        scored, precision.HIDDEN, threshold,
        loss_pause, hourly_cap, selected)
    later_passed = future_pass(later, maximum_negative_months=3)

    direct = None
    direct_rows: list[dict] = []
    direct_passed = False
    if later_passed:
        direct_frame = geometry.eligible(prior.direct_frame(), target_r)
        direct_replayed = exact.attach_runtime_runner(
            direct_frame, direct=True,
            activation_r=activation_r, runner_r=target_r)
        direct_scored = geometry.score(direct_replayed, model, design)
        direct, direct_rows = evaluate(
            direct_scored, ("xperp",), threshold,
            loss_pause, hourly_cap, selected)
        direct_passed = future_pass(direct, maximum_negative_months=1)

    for row in later_rows:
        row["audit_cohort"] = "later_2025_2026"
    for row in direct_rows:
        row["audit_cohort"] = "direct_xperp"
    pd.DataFrame(later_rows + direct_rows).to_pickle(ROWS_OUT)

    observed_future = [value for value in (later, direct) if value is not None]
    every_month_30 = bool(
        direct_passed and observed_future
        and all(month_return >= 30
                for value in observed_future
                for month_return in value["monthly_return_pct"].values()))
    report = {
        "status": ("AWAITING_GENUINELY_NEW_FORWARD_AFTER_2026_10_22"
                   if direct_passed else "REJECTED_FUTURE_EVIDENCE"),
        "ready_for_live_orders": False,
        "order_eligible": False,
        "frozen_geometry": {
            key: frozen[key] for key in (
                "target_r", "activation_r", "quantile", "threshold",
                "loss_pause", "hourly_cap")
        },
        "route_protocol": {
            "policy_universe": list(POLICIES),
            "development_periods": ["2022h1", "2022h2", "2023", "2024"],
            "admission": (
                "at least 400 trades and 2.9 trades/calendar-day in every "
                "stage, positive return, <=8% drawdown, <=2 negative months"),
            "rank": "maximize the worst stage return, then total return",
            "future_reveal": ["2025-2026", "direct X-Perp"],
            "entry": "next 15-minute market open",
            "exit": "exact ordered OHLC; 1R target, BE after 0.75R, stop-first",
            "money": "client-selected 3% current-equity margin, fixed 10x gross",
        },
        "policy_results_through_2024": policy_results,
        "admissible_policies": admissible,
        "selected_before_2025": selected,
        "later_2025_2026": later,
        "later_2025_2026_passed": later_passed,
        "direct_xperp": direct,
        "direct_xperp_passed": direct_passed,
        "objective_checks": {
            "future_evidence_passed": direct_passed,
            "every_observed_month_at_least_30pct": every_month_30,
            "ready_for_live_orders": False,
        },
        "sources": {
            str(BALANCED_REPORT): digest(BALANCED_REPORT),
            str(Path(__file__).resolve()): digest(Path(__file__).resolve()),
            **exact.CANDLE_SOURCES,
        },
        "limitations": [
            "The regime route is selected after observing 2024, so only 2025+ is future evidence for it.",
            "Direct X-Perp remains unread if 2025-2026 fails the frozen gate.",
            "Gross results exclude fees at user request.",
            "No backtest guarantees a future monthly return.",
        ],
    }
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({
        "status": report["status"],
        "admissible_policies": admissible,
        "selected_before_2025": selected,
        "development": policy_results[selected],
        "later_2025_2026": later,
        "later_2025_2026_passed": later_passed,
        "direct_xperp": direct,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
