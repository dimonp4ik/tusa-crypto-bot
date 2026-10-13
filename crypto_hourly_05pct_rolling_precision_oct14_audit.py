"""Annual rolling audit of the global hourly literal 0.5% entry family."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

import crypto_5m_regime_router_literal_oct6_audit as portfolio
import crypto_okx_swap_1h_regime_oct7_audit as base
from src.research_profit_gate import economic_gate


ROOT = Path("reports/audit_2026_09_08")
OUT = ROOT / "crypto_hourly_05pct_rolling_precision_oct14.json"
QUANTILES = (.50, .60, .70, .80, .85, .90, .925, .95, .96, .97, .98, .99)
MODEL = {
    "max_iter": 180, "learning_rate": .06, "max_leaf_nodes": 31,
    "min_samples_leaf": 280, "l2_regularization": 5.0,
    "random_state": 20261014,
}
FOLDS = (
    ((2020, 2021), 2022, 2023),
    ((2021, 2022), 2023, 2024),
    ((2022, 2023), 2024, 2025),
    ((2023, 2024), 2025, 2026),
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def evaluate(frame: pd.DataFrame, year: int, threshold: float) -> dict:
    metrics = portfolio.evaluate(frame, (year,), threshold)
    rows = portfolio.portfolio_rows(frame, (year,), threshold)
    gate = economic_gate(
        [float(row["signed_price_return"]) for row in rows],
        float(metrics.get("trades_per_calendar_day", 0)), market_days=30)
    return {"portfolio": metrics, "economic_gate": gate}


def rank(item: dict) -> tuple:
    metrics = item["calibration"]["portfolio"]
    enough = metrics["trades_per_calendar_day"] >= 3
    return (int(enough),
            metrics["literal_target_hit_rate"] if enough else -1,
            item["calibration"]["economic_gate"]["projected_monthly_return_pct"],
            -metrics["negative_months"], -metrics["dd_pct"])


def objective(value: dict) -> dict:
    metrics = value["portfolio"]
    checks = {
        "win_rate_75_to_95": 75 <= metrics["literal_target_hit_rate"] <= 95,
        "minimum_three_per_calendar_day":
            metrics["trades_per_calendar_day"] >= 3,
        "no_negative_months": metrics["negative_months"] == 0,
        "every_observed_month_at_least_30pct": bool(
            metrics["months"] and
            metrics["months_ge_30pct"] == metrics["months"]),
        "drawdown_at_most_5pct": metrics["dd_pct"] <= 5,
        "economic_projection_30pct": value["economic_gate"]["passes"],
    }
    return {"checks": checks, "passes": all(checks.values())}


def main() -> None:
    candles, sources = base.load()
    frame = base.add_outcomes(base.proposals(base.build_features(candles)), candles)
    folds = []
    for train_years, calibration_year, test_year in FOLDS:
        train = frame.year.isin(train_years)
        columns = base.schema(frame.loc[train])
        estimator = HistGradientBoostingClassifier(**MODEL)
        estimator.fit(base.matrix(frame.loc[train], columns),
                      frame.loc[train, "target_hit"].astype(int).to_numpy())
        observed = frame[frame.year.isin((calibration_year, test_year))].copy()
        observed["quality"] = estimator.predict_proba(
            base.matrix(observed, columns))[:, 1]
        scores = observed.loc[observed.year.eq(calibration_year), "quality"]
        frontier = []
        for quantile in QUANTILES:
            threshold = float(scores.quantile(quantile))
            frontier.append({
                "quantile": quantile, "threshold": threshold,
                "calibration": evaluate(observed, calibration_year, threshold),
            })
        frontier.sort(key=rank, reverse=True)
        chosen = frontier[0]
        test = evaluate(observed, test_year, float(chosen["threshold"]))
        folds.append({
            "train_years": list(train_years),
            "calibration_year": calibration_year, "test_year": test_year,
            "chosen_quantile": chosen["quantile"],
            "chosen_threshold": chosen["threshold"],
            "calibration": chosen["calibration"], "test": test,
            "test_objective": objective(test),
            "calibration_frontier": frontier,
        })
    passed = all(fold["test_objective"]["passes"] for fold in folds)
    report = {
        "status": "OBJECTIVE_PASS" if passed else "REJECTED_ROLLING_VALIDATION",
        "ready_for_live_orders": False,
        "protocol": {
            "training": "two completed years",
            "threshold_selection": (
                "next completed year; prefer >=3 calendar trades/day, then "
                "maximum literal target hit rate"),
            "reveal": "one later year per fold", "features": "base schema",
            "model": MODEL, "tested_quantiles": list(QUANTILES),
            "target_pct": .5, "stop_pct": 1.0, "maximum_hold_hours": 12,
            "entry": "next hourly market-open proxy",
            "orders": "market proxy only; research cannot place orders",
            "money": "3% current-equity margin, fixed 10x gross",
            "portfolio_rails": "existing frozen portfolio rails",
        },
        "rows": len(frame), "folds": folds, "objective_pass": passed,
        "decision": (
            "No runtime change. This branch can only become a component if "
            "rolling tests prove both precision and the complete economics."),
        "sources": {
            **sources, str(Path(base.__file__)): digest(Path(base.__file__)),
            str(Path(portfolio.__file__)): digest(Path(portfolio.__file__)),
            str(Path(__file__)): digest(Path(__file__)),
        },
    }
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({
        "status": report["status"],
        "folds": [{
            "test_year": fold["test_year"],
            "chosen_quantile": fold["chosen_quantile"],
            "calibration": fold["calibration"]["portfolio"],
            "test": fold["test"]["portfolio"],
            "test_economics": fold["test"]["economic_gate"],
            "test_objective": fold["test_objective"],
        } for fold in folds],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
