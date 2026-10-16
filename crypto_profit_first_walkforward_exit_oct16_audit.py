"""Strict walk-forward choice between fixed and ATR-scaled crypto exits.

For each year from 2023 onward, a bounded regressor learns only from completed
prior years whether the ATR runner adds return versus the fixed +0.5% exit.
Entries with positive above-median predicted improvement use the runner; every
other entry keeps the fixed exit.  No outcome from the routed year is used in
its own decision and win rate is not an optimization target.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

import crypto_5m_regime_router_literal_oct6_audit as portfolio
import crypto_okx_swap_1h_regime_oct7_audit as base
from crypto_profit_first_runner_oct16_audit import enrich


ROOT = Path("reports/audit_2026_09_08")
FIXED_ROWS = ROOT / "crypto_okx_swap_1h_regime_oct7.pkl"
ATR_ROWS = ROOT / "crypto_profit_first_atr_runner_oct16.pkl"
OUT = ROOT / "crypto_profit_first_walkforward_exit_oct16.json"
ROWS_OUT = ROOT / "crypto_profit_first_walkforward_exit_oct16.pkl"
YEARS = (2023, 2024, 2025, 2026)
OUTCOME_COLUMNS = (
    "signed_return", "exit_time", "target_hit", "exit_reason",
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def align(fixed: pd.DataFrame, atr: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    keys = ["symbol", "entry_time", "direction"]
    if fixed.duplicated(keys).any() or atr.duplicated(keys).any():
        raise ValueError("exit source contains duplicate entry keys")
    left = fixed.sort_values(keys).reset_index(drop=True)
    right = atr.sort_values(keys).reset_index(drop=True)
    if not left[keys].equals(right[keys]):
        raise ValueError("fixed and ATR exit rows are not aligned")
    return left, right


def model() -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        loss="squared_error", max_iter=180, learning_rate=.05,
        max_leaf_nodes=15, min_samples_leaf=100,
        l2_regularization=5.0, random_state=20261016)


def route_year(fixed: pd.DataFrame, atr: pd.DataFrame, year: int) -> tuple[pd.DataFrame, dict]:
    train_years = tuple(range(2022, year))[-3:]
    train_mask = fixed.year.isin(train_years)
    test_mask = fixed.year.eq(year)
    train = fixed.loc[train_mask]
    test = fixed.loc[test_mask]
    columns = base.schema(train)
    estimator = model()
    delta = (
        atr.loc[train_mask, "signed_return"].to_numpy(float)
        - train.signed_return.to_numpy(float))
    estimator.fit(base.matrix(train, columns), delta)
    prediction = estimator.predict(base.matrix(test, columns))
    median = float(np.median(prediction)) if len(prediction) else 0.0
    threshold = max(0.0, median)
    use_runner = prediction > threshold
    routed = test.copy()
    atr_test = atr.loc[test_mask]
    for column in OUTCOME_COLUMNS:
        routed.loc[use_runner, column] = atr_test.loc[use_runner, column].to_numpy()
    if "runner_hit" in atr_test:
        routed["runner_hit"] = False
        routed.loc[use_runner, "runner_hit"] = atr_test.loc[
            use_runner, "runner_hit"].to_numpy()
    routed["predicted_runner_delta"] = prediction
    routed["exit_profile"] = np.where(use_runner, "atr_runner", "fixed_0_5")
    diagnostics = {
        "year": year,
        "training_years": list(train_years),
        "training_rows": len(train),
        "test_rows": len(test),
        "predicted_delta_median_pct": round(100 * median, 6),
        "prediction_threshold_pct": round(100 * threshold, 6),
        "runner_rows": int(use_runner.sum()),
        "runner_share_pct": round(100 * float(use_runner.mean()), 4) if len(use_runner) else 0.0,
    }
    return routed, diagnostics


def evaluate(frame: pd.DataFrame, years: tuple[int, ...]) -> dict:
    return enrich(portfolio.evaluate(frame, years, 0))


def objective_ok(value: dict) -> bool:
    return bool(value["months"] > 0 and value["dd_pct"] <= 10
                and value["months_ge_30pct"] == value["months"])


def main() -> None:
    fixed, atr = align(pd.read_pickle(FIXED_ROWS), pd.read_pickle(ATR_ROWS))
    pieces = []
    routing = []
    yearly = {}
    baselines = {}
    for year in YEARS:
        rows, diagnostics = route_year(fixed, atr, year)
        pieces.append(rows)
        value = evaluate(rows, (year,))
        diagnostics["result"] = value
        routing.append(diagnostics)
        yearly[str(year)] = value
        baselines[str(year)] = {
            "fixed_0_5": evaluate(fixed, (year,)),
            "atr_runner": evaluate(atr, (year,)),
        }
    combined = pd.concat(pieces, ignore_index=True)
    total = evaluate(combined, YEARS)
    checks = {year: objective_ok(value) for year, value in yearly.items()}
    status = ("WALKFORWARD_EXIT_RESEARCH_CANDIDATE"
              if all(checks.values()) else "REJECTED_OBJECTIVE")
    combined.to_pickle(ROWS_OUT)
    report = {
        "status": status,
        "ready_for_live_orders": False,
        "protocol": {
            "priority": "account return first; win rate diagnostic only",
            "first_routed_year": 2023,
            "training_window": "up to three completed years, starting at 2022",
            "model": "bounded HistGradientBoostingRegressor on runner-minus-fixed return",
            "decision": "ATR runner only when predicted delta is positive and above current prediction median",
            "fixed_exit": "+0.5% target, -1% stop",
            "atr_exit": "25% at +0.5%; remainder at 3x entry ATR clamped 2-8%; -1% initial stop; 96h maximum",
            "entry": "frozen causal signal; next 1h market open",
            "orders": "market only",
            "money": "3% current-equity margin, fixed 10x gross",
            "objective": ">=30% account return in every month; DD <=10%",
        },
        "routing": routing,
        "yearly": yearly,
        "baselines": baselines,
        "combined_2023_2026": total,
        "objective_checks": checks,
        "sources": {
            str(FIXED_ROWS): digest(FIXED_ROWS),
            str(ATR_ROWS): digest(ATR_ROWS),
            str(Path(base.__file__)): digest(Path(base.__file__)),
            str(Path(__file__)): digest(Path(__file__)),
        },
        "limitations": [
            "Annual refits adapt slowly to within-year regime changes.",
            "The router cannot create edge when both available exits are weak.",
            "Historical OKX global results require EU X-Perp forward transfer.",
        ],
    }
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({
        "status": status,
        "routing": routing,
        "combined_2023_2026": total,
        "objective_checks": checks,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
