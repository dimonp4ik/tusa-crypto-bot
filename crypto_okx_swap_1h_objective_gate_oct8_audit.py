"""Frozen economic-objective audit for the OKX 1H target cascade.

This audit does not fit a model or change a threshold.  It replays the saved
frozen candidate through the existing portfolio gate, then evaluates account
growth for 3% equity margin, fixed 10x leverage and a +30% monthly objective.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

import crypto_5m_regime_router_literal_oct6_audit as portfolio
import crypto_okx_swap_1h_target_cascade_oct7_audit as cascade
from src.research_profit_gate import economic_gate, minimum_binary_hit_rate_pct


ROOT = Path("reports/audit_2026_09_08")
FRAME = ROOT / "crypto_okx_swap_1h_target_cascade_oct7.pkl"
SOURCE_REPORT = ROOT / "crypto_okx_swap_1h_target_cascade_oct7.json"
OUT = ROOT / "crypto_okx_swap_1h_objective_gate_oct8.json"
YEARS = (2022, 2023, 2024, 2025, 2026)
MARGIN_FRACTION = .03
LEVERAGE = 10.0
MONTHLY_TARGET = .30
REQUIRED_TRADES_PER_DAY = 3.0
CALENDAR_DAYS_PER_MONTH = 30


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def target_summary(rows: list[dict]) -> dict:
    output = {}
    for target in cascade.TARGETS:
        selected = [row for row in rows
                    if float(row["selected_target_pct"]) == target]
        hits = sum(bool(row["target_hit"]) for row in selected)
        output[str(target)] = {
            "n": len(selected),
            "share_pct": round(100 * len(selected) / len(rows), 4) if rows else 0.0,
            "literal_hit_rate_pct": round(100 * hits / len(selected), 4)
            if selected else 0.0,
        }
    return output


def actual_month_summary(metrics: dict) -> dict:
    months = list(metrics.get("monthly_return_pct", {}).values())
    return {
        "months": len(months),
        "months_ge_30pct": sum(value >= 30 for value in months),
        "negative_months": sum(value < 0 for value in months),
        "minimum_month_pct": round(min(months), 4) if months else 0.0,
        "median_month_pct": round(float(np.median(months)), 4) if months else 0.0,
        "maximum_month_pct": round(max(months), 4) if months else 0.0,
    }


def evaluate(frame: pd.DataFrame, years: tuple[int, ...]) -> dict:
    rows = portfolio.portfolio_rows(frame, years, .5)
    metrics = cascade.evaluate(frame, years)
    signed = [float(row["signed_price_return"]) for row in rows]
    gate = economic_gate(
        signed, metrics["trades_per_calendar_day"],
        monthly_target=MONTHLY_TARGET,
        required_trades_per_day=REQUIRED_TRADES_PER_DAY,
        margin_fraction=MARGIN_FRACTION, leverage=LEVERAGE,
        market_days=CALENDAR_DAYS_PER_MONTH)
    actual = actual_month_summary(metrics)
    objective_pass = bool(
        gate["passes"]
        and 75 <= metrics["literal_target_hit_rate"] <= 95
        and metrics["dd_pct"] <= 5
        and actual["negative_months"] == 0
        and actual["months_ge_30pct"] == actual["months"])
    return {
        "years": list(years),
        "portfolio": metrics,
        "economic_gate": gate,
        "actual_months": actual,
        "target_summary": target_summary(rows),
        "objective_pass": objective_pass,
    }


def main() -> None:
    frame = pd.read_pickle(FRAME)
    yearly = {str(year): evaluate(frame, (year,)) for year in YEARS}
    combined = evaluate(frame, YEARS)
    idealized_hit_rate_floor = {
        str(target): minimum_binary_hit_rate_pct(
            target_pct=target, stop_pct=cascade.STOP_PCT,
            monthly_target=MONTHLY_TARGET,
            margin_fraction=MARGIN_FRACTION, leverage=LEVERAGE,
            trades_per_day=REQUIRED_TRADES_PER_DAY,
            market_days=CALENDAR_DAYS_PER_MONTH)
        for target in cascade.TARGETS
    }
    all_pass = combined["objective_pass"] and all(
        value["objective_pass"] for value in yearly.values())
    report = {
        "status": "OBJECTIVE_PASS" if all_pass else "REJECTED_OBJECTIVE_GATE",
        "ready_for_live_orders": False,
        "frozen_replay": True,
        "thresholds_refit": False,
        "protocol": {
            "venue": "OKX global USDT swap",
            "bar": "1H",
            "entry": "next hourly market open",
            "targets_pct": list(cascade.TARGETS),
            "stop_pct": cascade.STOP_PCT,
            "margin_fraction_of_current_equity": MARGIN_FRACTION,
            "leverage": LEVERAGE,
            "monthly_target_pct": 100 * MONTHLY_TARGET,
            "required_trades_per_calendar_day": REQUIRED_TRADES_PER_DAY,
            "fees_and_slippage": "excluded per client research scope",
        },
        "idealized_minimum_hit_rate_pct_by_fixed_target": idealized_hit_rate_floor,
        "yearly": yearly,
        "combined_2022_2026": combined,
        "decision": (
            "Keep shadow collection append-only, but do not treat this cascade "
            "as meeting the monthly profit objective or enable live orders."),
        "sources": {
            str(FRAME): digest(FRAME),
            str(SOURCE_REPORT): digest(SOURCE_REPORT),
            str(Path(cascade.__file__)): digest(Path(cascade.__file__)),
            str(Path(portfolio.__file__)): digest(Path(portfolio.__file__)),
            str(Path(__file__)): digest(Path(__file__)),
        },
    }
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
