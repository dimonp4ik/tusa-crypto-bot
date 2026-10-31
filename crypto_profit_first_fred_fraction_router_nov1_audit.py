"""Test macro-conditioned TP1 banking on the exact Profit-First route."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

import crypto_profit_first_exact_health_router_oct30_audit as health
import crypto_profit_first_exact_oct31_1700_forward as latest
import crypto_profit_first_fred_macro_oct31_audit as macro
import crypto_profit_first_runner_protection_oct30_audit as exact
from crypto_runtime_shadow_portfolio_oct3_audit import money


ROOT = Path("reports/audit_2026_09_08")
HISTORICAL = macro.HISTORICAL
DIRECT = macro.DIRECT
OUT = ROOT / "crypto_profit_first_fred_fraction_router_nov1.json"
ROWS_OUT = ROOT / "crypto_profit_first_fred_fraction_router_nov1.pkl"
BASE_FRACTION = .05
DEFENSIVE_FRACTIONS = (.10, .25, .50, .75, 1.0)
Condition = Callable[[pd.DataFrame], pd.Series]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def conditions() -> dict[str, Condition]:
    return {
        "risk_off_score_le_1": lambda x: x.macro_risk_on_score.le(1),
        "risk_on_score_ge_2": lambda x: x.macro_risk_on_score.ge(2),
        "vix_level_ge_20": lambda x: x.vix_level.ge(20),
        "vix_rising_20d": lambda x: x.vix_change_20.gt(0),
        "dollar_rising_20d": lambda x: x.dollar_return_20.gt(0),
        "yield_rising_20d": lambda x: x.yield_change_20.gt(0),
        "direction_macro_misaligned": lambda x: (
            x.direction.eq("LONG") & x.macro_risk_on_score.le(1)
        ) | (
            x.direction.eq("SHORT") & x.macro_risk_on_score.ge(2)
        ),
        "direction_macro_aligned": lambda x: (
            x.direction.eq("LONG") & x.macro_risk_on_score.ge(2)
        ) | (
            x.direction.eq("SHORT") & x.macro_risk_on_score.le(1)
        ),
    }


def policy_columns(frame: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.Series,
                                                        pd.Series, pd.Series]:
    historical_prefix = exact.policy_key(.5, 0.0)
    if f"{historical_prefix}_runner_r" in frame:
        return (
            frame[f"{historical_prefix}_runner_r"].astype(float),
            frame[f"{historical_prefix}_banked"].astype(bool),
            frame[f"{historical_prefix}_exit_time"].astype(float),
            frame["near_exit_time"].astype(float),
            frame[f"{historical_prefix}_label"].astype(str),
        )
    return (
        frame["sequential_runner_r"].astype(float),
        frame["sequential_banked"].astype(bool),
        frame["sequential_exit_time"].astype(float),
        frame["near_exit_time"].astype(float),
        frame["runner_outcome"].astype(str),
    )


def apply_policy(frame: pd.DataFrame, condition: Condition | None,
                 defensive_fraction: float) -> pd.DataFrame:
    result = frame.copy()
    runner, banked, runner_exit, near_exit, labels = policy_columns(result)
    mask = (condition(result).fillna(False) if condition is not None
            else pd.Series(False, index=result.index))
    fractions = pd.Series(
        np.where(mask, defensive_fraction, BASE_FRACTION), index=result.index,
        dtype=float)
    result["tp1_fraction"] = fractions
    result["net_r"] = np.where(
        banked,
        fractions * .5 + (1.0 - fractions) * runner,
        runner,
    )
    full_bank = banked & fractions.ge(1.0 - 1e-12)
    result["exit_time"] = np.where(full_bank, near_exit, runner_exit)
    result["runner_outcome"] = np.where(full_bank, "FULL_TP1", labels)
    result["signed_price_return"] = (
        result["risk_pct"].astype(float) * result["net_r"].astype(float))
    return result


def metric(frame: pd.DataFrame, period: str, module: str) -> dict:
    rows = health.accepted(frame[frame.period.eq(period)], module)
    return money(rows, .03)


def compact(value: dict) -> dict:
    return {key: value[key] for key in (
        "n", "wr", "return_pct", "dd_pct", "max_loss_streak",
        "negative_months", "trades_per_calendar_day", "monthly_return_pct")}


def period_ok(candidate: dict, baseline: dict) -> bool:
    return bool(
        candidate["return_pct"] >= baseline["return_pct"]
        and candidate["dd_pct"] <= baseline["dd_pct"]
        and candidate["negative_months"] <= baseline["negative_months"]
        and candidate["max_loss_streak"] <= baseline["max_loss_streak"] + 1
        and candidate["trades_per_calendar_day"]
            >= .75 * baseline["trades_per_calendar_day"])


def hidden_ok(candidate: dict, baseline: dict) -> bool:
    return bool(
        candidate["return_pct"] >= baseline["return_pct"]
        and candidate["dd_pct"] <= baseline["dd_pct"] + .5
        and candidate["negative_months"] <= baseline["negative_months"]
        and candidate["max_loss_streak"] <= baseline["max_loss_streak"] + 1)


def direct_metrics(frame: pd.DataFrame) -> dict:
    october = frame[frame.entry_time.ge(pd.Timestamp(
        "2026-10-01", tz="UTC").timestamp())]
    return {
        "all_direct": latest.metrics(frame, .03),
        "october": latest.metrics(october, .03),
    }


def main() -> None:
    historical_raw = macro.attach_macro(pd.read_pickle(HISTORICAL))
    direct_raw = macro.attach_macro(pd.read_pickle(DIRECT))
    baseline_frame = apply_policy(historical_raw, None, BASE_FRACTION)
    baseline = {period: metric(baseline_frame, period, "fred_fraction_baseline")
                for period in ("2022h1", "2022h2", "2023", "2024", "2025", "2026")}
    baseline_direct_frame = apply_policy(direct_raw, None, BASE_FRACTION)
    baseline_direct = direct_metrics(baseline_direct_frame)

    candidates = []
    eligible = []
    for condition_name, condition in conditions().items():
        for fraction in DEFENSIVE_FRACTIONS:
            frame = apply_policy(historical_raw, condition, fraction)
            periods = {period: metric(frame, period, "fred_fraction_candidate")
                       for period in ("2022h1", "2022h2", "2023")}
            passes = all(period_ok(periods[period], baseline[period])
                         for period in ("2022h1", "2022h2", "2023"))
            item = {
                "condition": condition_name,
                "defensive_tp1_fraction": fraction,
                "selection_2022h1": compact(periods["2022h1"]),
                "confirmation_2022h2": compact(periods["2022h2"]),
                "confirmation_2023": compact(periods["2023"]),
                "development_pass": passes,
            }
            candidates.append(item)
            if passes:
                eligible.append(item)
    eligible.sort(key=lambda item: (
        min(item["selection_2022h1"]["return_pct"]
            / max(baseline["2022h1"]["return_pct"], 1e-9),
            item["confirmation_2022h2"]["return_pct"]
            / max(baseline["2022h2"]["return_pct"], 1e-9),
            item["confirmation_2023"]["return_pct"]
            / max(baseline["2023"]["return_pct"], 1e-9)),
        -max(item["selection_2022h1"]["dd_pct"],
             item["confirmation_2022h2"]["dd_pct"],
             item["confirmation_2023"]["dd_pct"])), reverse=True)
    selected = eligible[0] if eligible else None
    hidden = candidate_direct = None
    robust = False
    selected_direct = None
    if selected:
        condition = conditions()[selected["condition"]]
        fraction = float(selected["defensive_tp1_fraction"])
        selected_historical = apply_policy(historical_raw, condition, fraction)
        hidden = {period: metric(selected_historical, period, "fred_fraction_hidden")
                  for period in ("2024", "2025", "2026")}
        selected_direct = apply_policy(direct_raw, condition, fraction)
        candidate_direct = direct_metrics(selected_direct)
        robust = bool(
            all(hidden_ok(hidden[period], baseline[period])
                for period in ("2024", "2025", "2026"))
            and hidden_ok(candidate_direct["all_direct"],
                          baseline_direct["all_direct"])
            and hidden_ok(candidate_direct["october"],
                          baseline_direct["october"]))
    if robust and selected_direct is not None:
        selected_direct.to_pickle(ROWS_OUT)

    report = {
        "status": ("FRED_FRACTION_SHADOW_CHALLENGER" if robust
                   else "REJECTED_FRED_FRACTION_ROUTER"),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "ready_for_live_orders": False,
        "orders_possible": False,
        "runtime_change_allowed": False,
        "protocol": {
            "base_exit": "5% at 0.5R; 95% at 2R or next-bar breakeven",
            "candidate": "bank a larger TP1 fraction only in one causal macro state",
            "selection": "2022H1",
            "confirmation": ["2022H2", "2023"],
            "sealed_holdouts": ["2024", "2025", "2026", "direct X-Perp"],
            "macro_availability": "observation date + 3 calendar days at 00:00 UTC",
            "conditions": list(conditions()),
            "defensive_fractions": list(DEFENSIVE_FRACTIONS),
            "money": "3% current-equity margin, fixed 10x gross",
            "orders": "market only",
        },
        "baseline": {period: compact(value) for period, value in baseline.items()},
        "baseline_direct": baseline_direct,
        "candidate_count": len(candidates),
        "development_pass_count": len(eligible),
        "selected_before_holdouts": selected,
        "hidden": ({period: compact(value) for period, value in hidden.items()}
                   if hidden else None),
        "direct_xperp": candidate_direct,
        "robust_improvement": robust,
        "target_30pct_each_month_proven": False,
        "candidates": candidates,
        "sources": {str(path): digest(path) for path in (
            HISTORICAL, DIRECT, macro.MANIFEST, Path(__file__).resolve())},
        "limitations": [
            "FRED current-vintage history may include later revisions.",
            "The direct replay is already portfolio-gated and freed slots are not refilled.",
            "Gross returns exclude fees at user request.",
        ],
    }
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({
        "status": report["status"], "development_pass_count": len(eligible),
        "selected": selected,
        "baseline_hidden": {period: compact(baseline[period])
                            for period in ("2024", "2025", "2026")},
        "hidden": report["hidden"],
        "baseline_direct": baseline_direct,
        "direct": candidate_direct,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
