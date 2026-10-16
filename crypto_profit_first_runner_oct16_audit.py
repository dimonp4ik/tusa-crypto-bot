"""Profit-first extension of frozen causal OKX one-hour entries.

Win rate is diagnostic only.  Selection ranks account growth and drawdown for
3% current-equity margin at fixed 10x leverage.  A small first target activates
a breakeven runner seeking a literal 2-8% move with market exits.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import math
from pathlib import Path

import pandas as pd

import crypto_5m_regime_router_literal_oct6_audit as portfolio
import crypto_okx_swap_1h_regime_oct7_audit as base
import crypto_okx_swap_1h_runner_oct7_audit as runner


SOURCE_REPORT = Path("reports/audit_2026_09_08/crypto_okx_swap_1h_regime_oct7.json")
SOURCE_ROWS = Path("reports/audit_2026_09_08/crypto_okx_swap_1h_regime_oct7.pkl")
OUT = Path("reports/audit_2026_09_08/crypto_profit_first_runner_oct16.json")
ROWS_OUT = Path("reports/audit_2026_09_08/crypto_profit_first_runner_oct16.pkl")
PARTIALS = (0.0, .10, .25)
RUNNER_TARGETS = (2.0, 3.0, 4.0, 6.0, 8.0)
HOLDS = (48, 96, 168)
DEVELOPMENT = (2022,)
VALIDATION = (2023,)
TEST = (2024,)
CONFIRMATION = (2025, 2026)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def monthly_equivalent(value: dict) -> float:
    months = int(value.get("months", 0) or len(value.get("monthly_return_pct", {})))
    factor = 1 + float(value.get("return_pct", 0)) / 100
    if months <= 0 or factor <= 0:
        return -100.0
    return round(100 * math.expm1(math.log(factor) / months), 6)


def enrich(value: dict) -> dict:
    output = dict(value)
    monthly = [float(item) for item in value.get("monthly_return_pct", {}).values()]
    output["months"] = len(monthly)
    output["negative_months"] = sum(item < 0 for item in monthly)
    output["equivalent_monthly_return_pct"] = monthly_equivalent(value)
    output["minimum_month_pct"] = round(min(monthly), 6) if monthly else 0.0
    output["months_ge_30pct"] = sum(item >= 30 for item in monthly)
    output["profit_first_objective_pass"] = bool(
        monthly
        and all(item >= 30 for item in monthly)
        and float(value["dd_pct"]) <= 10)
    return output


def evaluate(frame: pd.DataFrame, years: tuple[int, ...]) -> dict:
    return enrich(portfolio.evaluate(frame, years, 0))


def evaluate_margin(frame: pd.DataFrame, years: tuple[int, ...],
                    margin_fraction: float) -> dict:
    rows = portfolio.portfolio_rows(frame, years, 0)
    return enrich(portfolio.money(rows, margin_fraction))


def development_ok(combined: dict, annual: dict[str, dict]) -> bool:
    return bool(
        combined["n"] >= 700
        and combined["return_pct"] > 0
        and combined["dd_pct"] <= 15
        and combined["trades_per_calendar_day"] >= 2
        and all(item["return_pct"] > 0 for item in annual.values()))


def validation_ok(value: dict) -> bool:
    return bool(
        value["n"] >= 500
        and value["return_pct"] > 0
        and value["dd_pct"] <= 15
        and value["trades_per_calendar_day"] >= 2)


def score(development: dict, annual: dict[str, dict], validation: dict) -> float:
    returns = [item["equivalent_monthly_return_pct"] for item in annual.values()]
    returns.append(validation["equivalent_monthly_return_pct"])
    return round(
        min(returns) + .25 * sum(returns) / len(returns)
        - .10 * max(development["dd_pct"], validation["dd_pct"]), 8)


def public(item: dict | None) -> dict | None:
    if item is None:
        return None
    return {key: value for key, value in item.items()
            if key not in {"_key", "_rows"}}


def main() -> None:
    parent = json.loads(SOURCE_REPORT.read_text(encoding="utf-8"))
    if parent.get("status") != "RESEARCH_CANDIDATE_AWAITING_DIRECT_FORWARD":
        raise ValueError("parent OKX 1h entries are not frozen candidates")
    candles, sources = base.load()
    entries = pd.read_pickle(SOURCE_ROWS)
    frontier = []
    validated = []
    for partial, target, hold in itertools.product(
            PARTIALS, RUNNER_TARGETS, HOLDS):
        resolved = runner.add_outcomes(entries, candles, partial, target, hold)
        development = evaluate(resolved, DEVELOPMENT)
        annual = {str(year): evaluate(resolved, (year,))
                  for year in DEVELOPMENT}
        item = {
            "partial_fraction_at_0_5pct": partial,
            "runner_target_pct": target,
            "hold_bars": hold,
            "development": development,
            "development_years": annual,
            "development_pass": development_ok(development, annual),
            "_key": (partial, target, hold),
            "_rows": resolved,
        }
        frontier.append(item)
        if not item["development_pass"]:
            continue
        validation = evaluate(resolved, VALIDATION)
        item["validation"] = validation
        item["validation_pass"] = validation_ok(validation)
        if item["validation_pass"]:
            item["score"] = score(development, annual, validation)
            validated.append(item)

    frontier.sort(key=lambda item: (
        item["development_pass"],
        item["development"]["equivalent_monthly_return_pct"],
        -item["development"]["dd_pct"]), reverse=True)
    validated.sort(key=lambda item: item["score"], reverse=True)
    frozen = validated[0] if validated else None
    test = confirmation = checks = margin_sensitivity = None
    status = "REJECTED_BEFORE_TEST"
    if frozen:
        test = evaluate(frozen["_rows"], TEST)
        margin_sensitivity = {
            "five_percent_margin": {
                "selection_2022": evaluate_margin(
                    frozen["_rows"], DEVELOPMENT, .05),
                "validation_2023": evaluate_margin(
                    frozen["_rows"], VALIDATION, .05),
                "test_2024": evaluate_margin(frozen["_rows"], TEST, .05),
            },
        }
        checks = {"test_profit_objective": test["profit_first_objective_pass"]}
        status = "REJECTED_TEST"
        if checks["test_profit_objective"]:
            confirmation = {
                str(year): evaluate(frozen["_rows"], (year,))
                for year in CONFIRMATION}
            checks.update({
                str(year): confirmation[str(year)]["profit_first_objective_pass"]
                for year in CONFIRMATION})
            status = ("PROFIT_FIRST_RESEARCH_CANDIDATE"
                      if all(checks.values()) else "REJECTED_CONFIRMATION")
        frozen["_rows"].to_pickle(ROWS_OUT)

    report = {
        "status": status,
        "ready_for_live_orders": False,
        "protocol": {
            "priority": "account return first; win rate is diagnostic only",
            "parent_entry_model_fit": [2020, 2021],
            "development": list(DEVELOPMENT),
            "validation": list(VALIDATION),
            "conditional_test": list(TEST),
            "confirmation_only": list(CONFIRMATION),
            "entry": "frozen causal signal; next 1h market open",
            "first_target_pct": .5,
            "partial_fractions": list(PARTIALS),
            "runner_targets_pct": list(RUNNER_TARGETS),
            "initial_stop_pct": 1.0,
            "hold_bars": list(HOLDS),
            "runner_stop_after_first_target": "entry from next hourly bar",
            "orders": "market only",
            "money": "3% current-equity margin, fixed 10x gross",
            "objective": ">=30% account return in every month; DD <=10%",
        },
        "configurations": len(PARTIALS) * len(RUNNER_TARGETS) * len(HOLDS),
        "development_pass": sum(item["development_pass"] for item in frontier),
        "validation_pass": len(validated),
        "frozen_before_test": public(frozen),
        "test_2024": test,
        "confirmation_2025_2026": confirmation,
        "margin_sensitivity": margin_sensitivity,
        "objective_checks": checks,
        "development_frontier": [public(item) for item in frontier[:15]],
        "sources": {
            **sources,
            str(SOURCE_REPORT): digest(SOURCE_REPORT),
            str(SOURCE_ROWS): digest(SOURCE_ROWS),
            str(Path(runner.__file__)): digest(Path(runner.__file__)),
            str(Path(__file__)): digest(Path(__file__)),
        },
        "limitations": [
            "Gross returns intentionally follow the client's research scope.",
            "High historical return cannot replace direct venue forward evidence.",
            "The objective is evaluated on account equity, not raw price movement.",
        ],
    }
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({
        "status": status,
        "development_pass": report["development_pass"],
        "validation_pass": report["validation_pass"],
        "frozen_before_test": public(frozen),
        "test_2024": test,
        "objective_checks": checks,
        "confirmation": confirmation,
        "top_development": [public(item) for item in frontier[:5]],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
