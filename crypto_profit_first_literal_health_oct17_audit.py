"""Causal shadow-health gate for the frozen literal two-percent runner.

Every parent signal remains observable in shadow.  Before a new entry the gate
uses only outcomes whose exits have already occurred and may suspend a key
(direction, symbol, or symbol+direction) when recent signed expectancy and hit
rate are weak.  Parameters are chosen on 2022-2023, frozen for 2024 validation,
then revealed on 2025-2026.
"""
from __future__ import annotations

import collections
import hashlib
import heapq
import itertools
import json
from pathlib import Path

import pandas as pd

import crypto_5m_regime_router_literal_oct6_audit as portfolio
from crypto_profit_first_runner_oct16_audit import enrich


ROOT = Path("reports/audit_2026_09_08")
SOURCE = ROOT / "crypto_profit_first_literal_runner_oct16.pkl"
SOURCE_REPORT = ROOT / "crypto_profit_first_literal_runner_oct16.json"
OUT = ROOT / "crypto_profit_first_literal_health_oct17.json"
ROWS_OUT = ROOT / "crypto_profit_first_literal_health_oct17.pkl"
KEY_MODES = ("direction", "symbol", "symbol_direction")
WINDOWS = (20, 40, 80)
MIN_HISTORY = (10, 20)
MIN_MEANS = (-.001, 0.0, .001)
MIN_HIT_RATES = (.30, .33, .36)
SELECTION_YEARS = (2022, 2023)
VALIDATION_YEAR = 2024
REVEAL_YEARS = (2025, 2026)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def health_key(row: dict, mode: str) -> tuple[str, ...]:
    if mode == "direction":
        return (str(row["direction"]),)
    if mode == "symbol":
        return (str(row["symbol"]),)
    if mode == "symbol_direction":
        return (str(row["symbol"]), str(row["direction"]))
    raise ValueError(f"unknown key mode: {mode}")


def causal_gate(rows: list[dict], *, key_mode: str, window: int,
                min_history: int, min_mean: float,
                min_hit_rate: float) -> list[dict]:
    history = collections.defaultdict(lambda: collections.deque(maxlen=window))
    pending = []
    output = []
    sequence = 0
    ordered = sorted(rows, key=lambda row: (
        float(row["entry_time"]), int(row.get("source_priority", 0)),
        str(row["symbol"])))
    for row in ordered:
        now = float(row["entry_time"])
        while pending and pending[0][0] < now:
            _, _, key, outcome = heapq.heappop(pending)
            history[key].append(outcome)
        key = health_key(row, key_mode)
        recent = history[key]
        enabled = len(recent) < min_history
        if not enabled:
            mean = sum(recent) / len(recent)
            hit_rate = sum(value > 0 for value in recent) / len(recent)
            enabled = mean >= min_mean and hit_rate >= min_hit_rate
        outcome = float(row["signed_price_return"])
        heapq.heappush(pending, (float(row["exit_time"]), sequence,
                                key, outcome))
        sequence += 1
        if enabled:
            output.append(row)
    return output


def by_year(rows: list[dict], year: int) -> list[dict]:
    return [row for row in rows if int(row["year"]) == year]


def evaluate(rows: list[dict], year: int) -> dict:
    return enrich(portfolio.money(by_year(rows, year), .03))


def selection_ok(values: dict[str, dict]) -> bool:
    return all(value["return_pct"] > 0 and value["dd_pct"] <= 15
               and value["trades_per_calendar_day"] >= 3
               for value in values.values())


def score(values: dict[str, dict]) -> float:
    monthly = [value["equivalent_monthly_return_pct"]
               for value in values.values()]
    return round(min(monthly) + .25 * sum(monthly)
                 - .10 * max(value["dd_pct"] for value in values.values())
                 - .20 * sum(value["negative_months"]
                             for value in values.values()), 8)


def objective_ok(value: dict) -> bool:
    return bool(value["months"] > 0 and value["dd_pct"] <= 10
                and value["trades_per_calendar_day"] >= 3
                and value["months_ge_30pct"] == value["months"])


def main() -> None:
    parent = pd.read_pickle(SOURCE)
    rows = parent.to_dict("records")
    candidates = []
    for mode, window, minimum, mean, hit_rate in itertools.product(
            KEY_MODES, WINDOWS, MIN_HISTORY, MIN_MEANS, MIN_HIT_RATES):
        if minimum > window:
            continue
        parameters = {
            "key_mode": mode,
            "window": window,
            "min_history": minimum,
            "min_mean": mean,
            "min_hit_rate": hit_rate,
        }
        accepted = causal_gate(rows, **parameters)
        selection = {str(year): evaluate(accepted, year)
                     for year in SELECTION_YEARS}
        candidates.append({
            "parameters": parameters,
            "selection": selection,
            "selection_pass": selection_ok(selection),
            "score": score(selection),
            "accepted_count": len(accepted),
        })
    candidates.sort(key=lambda item: (
        item["selection_pass"], item["score"]), reverse=True)
    chosen = candidates[0] if candidates and candidates[0]["selection_pass"] else None
    validation = reveal = checks = None
    status = "REJECTED_SELECTION"
    if chosen:
        accepted = causal_gate(rows, **chosen["parameters"])
        validation = evaluate(accepted, VALIDATION_YEAR)
        reveal = {str(year): evaluate(accepted, year) for year in REVEAL_YEARS}
        yearly = {**chosen["selection"], str(VALIDATION_YEAR): validation,
                  **reveal}
        checks = {year: objective_ok(value) for year, value in yearly.items()}
        status = ("LITERAL_HEALTH_RESEARCH_CANDIDATE"
                  if all(checks.values()) else "REJECTED_OBJECTIVE")
        pd.DataFrame(accepted).to_pickle(ROWS_OUT)
    report = {
        "status": status,
        "ready_for_live_orders": False,
        "protocol": {
            "selection": list(SELECTION_YEARS),
            "validation": VALIDATION_YEAR,
            "revealed_after_freeze": list(REVEAL_YEARS),
            "disabled_signals": "shadow-observed and added only after exit",
            "parameter_grid": {
                "key_modes": list(KEY_MODES), "windows": list(WINDOWS),
                "min_history": list(MIN_HISTORY),
                "min_mean_signed_price_return": list(MIN_MEANS),
                "min_hit_rates": list(MIN_HIT_RATES),
            },
            "money": "3% current-equity margin, fixed 10x gross",
            "objective": ">=30% account return in every month; >=3 trades/day; DD <=10%",
        },
        "candidate_count": len(candidates),
        "top_selection": candidates[:20],
        "frozen_before_2024": chosen,
        "validation_2024": validation,
        "reveal_2025_2026": reveal,
        "objective_checks": checks,
        "sources": {
            str(SOURCE): digest(SOURCE),
            str(SOURCE_REPORT): digest(SOURCE_REPORT),
            str(Path(__file__)): digest(Path(__file__)),
        },
        "limitations": [
            "Filtering the parent accepted stream does not backfill a lower-ranked same-hour proposal.",
            "The gate needs shadow outcomes even while a key is disabled.",
            "Historical results still require direct shadow-forward confirmation.",
        ],
    }
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({
        "status": status, "candidate_count": len(candidates),
        "chosen": chosen, "validation_2024": validation,
        "reveal_2025_2026": reveal, "objective_checks": checks,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
