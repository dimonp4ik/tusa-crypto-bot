"""Causal monthly switch for the complementary crypto components.

Each month can use only component returns from earlier months.  The switching
rule is selected on 2024, checked on 2025, and then revealed on 2026.
Trade size remains the client's fixed percentage for every accepted trade.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

import crypto_complementary_exact_portfolio_oct24_audit as portfolio


ROOT = portfolio.ROOT
SOURCE = ROOT / "crypto_complementary_historical_stress.pkl"
SOURCE_REPORT = ROOT / "crypto_complementary_historical_stress.json"
OUT = ROOT / "crypto_causal_component_switch.json"
ROWS_OUT = ROOT / "crypto_causal_component_switch.pkl"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def component(row: pd.Series | dict) -> str:
    source = str(row["source"])
    return "target" if source.startswith("walkforward") else "hourly"


def apply_rule(frame: pd.DataFrame, mode: str, window: int) -> pd.DataFrame:
    output = []
    months = sorted(frame.month.unique())
    for index, month in enumerate(months):
        previous = months[max(0, index-window):index]
        history = frame[frame.month.isin(previous)]
        scores = history.groupby("component").signed_price_return.sum().to_dict()
        if not previous or not scores:
            active = {"target", "hourly"}
        elif mode == "positive_each":
            active = {name for name in ("target", "hourly")
                      if scores.get(name, 0.0) > 0}
            if not active:
                active = {max(("target", "hourly"),
                              key=lambda name: scores.get(name, -1e20))}
        elif mode == "best_only":
            active = {max(("target", "hourly"),
                          key=lambda name: scores.get(name, -1e20))}
        elif mode == "both_if_portfolio_positive":
            active = ({"target", "hourly"} if sum(scores.values()) > 0
                      else {max(("target", "hourly"),
                                key=lambda name: scores.get(name, -1e20))})
        else:
            raise ValueError(mode)
        chosen = frame[frame.month.eq(month) & frame.component.isin(active)].copy()
        chosen["active_components"] = "+".join(sorted(active))
        output.append(chosen)
    return pd.concat(output, ignore_index=True)


def compact(value: dict) -> dict:
    return {name: value[name] for name in (
        "n", "return_pct", "dd_pct", "wr", "max_loss_streak",
        "trades_per_calendar_day", "negative_months",
        "monthly_return_pct")}


def score(value: dict) -> float:
    months = list(value["monthly_return_pct"].values())
    floor = min(months) if months else -100.0
    return round(floor + .1*value["return_pct"]-.25*value["dd_pct"], 8)


def main() -> None:
    frame = pd.read_pickle(SOURCE).copy()
    frame["month"] = pd.to_datetime(
        frame.entry_time, unit="s", utc=True).dt.strftime("%Y-%m")
    frame["entry_year"] = pd.to_datetime(
        frame.entry_time, unit="s", utc=True).dt.year
    frame["component"] = frame.apply(component, axis=1)
    candidates = []
    routed = {}
    for mode in ("positive_each", "best_only", "both_if_portfolio_positive"):
        for window in (1, 2, 3):
            selected = apply_rule(frame, mode, window)
            development_rows = selected[selected.entry_year.eq(2024)].to_dict("records")
            value = portfolio.metrics(development_rows, .03)
            item = {"mode": mode, "window_months": window,
                    "development": compact(value), "score": score(value)}
            candidates.append(item)
            routed[(mode, window)] = selected
    eligible = [item for item in candidates
                if item["development"]["trades_per_calendar_day"] >= 3
                and item["development"]["dd_pct"] <= 10]
    eligible.sort(key=lambda item: (
        item["score"], item["development"]["return_pct"]), reverse=True)
    chosen = eligible[0]
    selected = routed[(chosen["mode"], chosen["window_months"])]
    selected.to_pickle(ROWS_OUT)
    results = {}
    for year in (2024, 2025, 2026):
        rows = selected[selected.entry_year.eq(year)].to_dict("records")
        results[str(year)] = {
            "margin_3pct": portfolio.metrics(rows, .03),
            "margin_5pct": portfolio.metrics(rows, .05),
            "active_component_months": selected[
                selected.entry_year.eq(year)].groupby("month")
                .active_components.first().to_dict(),
        }
    validation_pass = bool(
        results["2025"]["margin_3pct"]["negative_months"] == 0
        and results["2025"]["margin_3pct"]["dd_pct"] <= 10)
    objective = bool(validation_pass and all(
        value >= 30 for year in ("2025", "2026")
        for value in results[year]["margin_3pct"]
        ["monthly_return_pct"].values()))
    report = {
        "status": ("CAUSAL_COMPONENT_SWITCH_OBJECTIVE_PASSED" if objective
                   else "CAUSAL_COMPONENT_SWITCH_REJECTED"),
        "ready_for_live_orders": False,
        "order_eligible": False,
        "protocol": {
            "development": 2024,
            "acceptance": 2025,
            "hidden": 2026,
            "decision_timing": "month start from earlier completed months only",
            "tested_rules": len(candidates),
            "position_size": "unchanged client-selected percentage",
        },
        "selected_on_2024": chosen,
        "results": results,
        "checks": {"validation_2025_pass": validation_pass,
                   "objective_pass": objective},
        "development_frontier": eligible,
        "sources": {
            str(SOURCE): digest(SOURCE),
            str(SOURCE_REPORT): digest(SOURCE_REPORT),
            str(Path(__file__).resolve()): digest(Path(__file__).resolve()),
        },
        "limitations": [
            "The source target-router periods do not cover every calendar month.",
            "Monthly switching reacts after losses rather than predicting them.",
            "Gross results exclude fees at user request.",
        ],
    }
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True),
                   encoding="utf-8")
    print(json.dumps({
        "status": report["status"], "selected_on_2024": chosen,
        "results": {year: {
            "margin_3pct": compact(value["margin_3pct"]),
            "margin_5pct": compact(value["margin_5pct"]),
            "active": value["active_component_months"],
        } for year, value in results.items()},
        "checks": report["checks"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
