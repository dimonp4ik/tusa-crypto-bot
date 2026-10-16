"""Strict yearly walk-forward crypto entry ranking for ATR-runner profit.

All seven semantic entry families remain available.  A bounded regressor is
refit for each year using only the preceding two completed years and predicts
the signed account-driving price return of the ATR-scaled runner.  The top 3%
of current-year predictions enter at the next hourly market open.  Win rate is
reported but is never a training or acceptance objective.
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
import crypto_profit_first_atr_runner_oct16_audit as atr
from crypto_profit_first_runner_oct16_audit import enrich


ROOT = Path("reports/audit_2026_09_08")
OUT = ROOT / "crypto_profit_first_rolling_entry_oct16.json"
ROWS_OUT = ROOT / "crypto_profit_first_rolling_entry_oct16.pkl"
TEST_YEARS = (2022, 2023, 2024, 2025, 2026)
TRAINING_YEARS = {
    2022: (2020, 2021),
    2023: (2021, 2022),
    2024: (2022, 2023),
    2025: (2023, 2024),
    2026: (2024, 2025),
}
SELECTION_QUANTILE = .97


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def model() -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        loss="squared_error", max_iter=160, learning_rate=.05,
        max_leaf_nodes=31, min_samples_leaf=280,
        l2_regularization=5.0, random_state=20261016)


def training_cutoff(predictions: np.ndarray) -> float:
    """Return the frozen selection cutoff from already completed samples."""
    return float(np.quantile(np.asarray(predictions, dtype=float),
                             SELECTION_QUANTILE))


def score_year(frame: pd.DataFrame, year: int) -> tuple[pd.DataFrame, dict]:
    train_years = TRAINING_YEARS[year]
    train = frame[frame.year.isin(train_years)]
    test = frame[frame.year.eq(year)].copy()
    columns = base.schema(train)
    estimator = model()
    train_matrix = base.matrix(train, columns)
    estimator.fit(train_matrix, train.signed_return.to_numpy(float))
    test["quality"] = estimator.predict(base.matrix(test, columns))
    # Freeze the cutoff before the tested year begins.  A quantile of the
    # tested year's predictions would quietly use its future score
    # distribution and could not be reproduced by the live bot.
    training_predictions = estimator.predict(train_matrix)
    threshold = training_cutoff(training_predictions)
    accepted = portfolio.portfolio_rows(test, (year,), threshold)
    result = enrich(portfolio.evaluate(test, (year,), threshold))
    diagnostics = {
        "year": year,
        "training_years": list(train_years),
        "training_rows": len(train),
        "candidate_rows": len(test),
        "prediction_threshold_price_return_pct": round(100 * threshold, 6),
        "pre_portfolio_rows": int(test.quality.ge(threshold).sum()),
        "accepted_rows": len(accepted),
        "result": result,
    }
    return pd.DataFrame(accepted), diagnostics


def objective_ok(value: dict) -> bool:
    return bool(value["months"] > 0 and value["dd_pct"] <= 10
                and value["trades_per_calendar_day"] >= 3
                and value["months_ge_30pct"] == value["months"])


def main() -> None:
    candles, sources = base.load()
    features = base.build_features(candles)
    proposals = base.proposals(features)
    resolved = atr.add_outcomes(
        proposals, candles, partial=.25, atr_multiplier=3.0, hold=96)
    pieces = []
    diagnostics = []
    yearly = {}
    for year in TEST_YEARS:
        accepted, item = score_year(resolved, year)
        pieces.append(accepted)
        diagnostics.append(item)
        yearly[str(year)] = item["result"]
    emitted = pd.concat(pieces, ignore_index=True)
    combined = enrich(portfolio.money(emitted.to_dict("records"), .03))
    checks = {year: objective_ok(value) for year, value in yearly.items()}
    status = ("ROLLING_ENTRY_RESEARCH_CANDIDATE"
              if all(checks.values()) else "REJECTED_OBJECTIVE")
    emitted.to_pickle(ROWS_OUT)
    report = {
        "status": status,
        "ready_for_live_orders": False,
        "protocol": {
            "priority": "account return first; win rate diagnostic only",
            "families": list(base.FAMILIES),
            "training": "preceding two completed calendar years",
            "test_years": list(TEST_YEARS),
            "model": "bounded HistGradientBoostingRegressor on signed ATR-runner return",
            "selection_quantile": SELECTION_QUANTILE,
            "threshold_source": "in-sample predictions from completed training years",
            "exit": "25% at +0.5%; remainder at 3x entry ATR clamped 2-8%; -1% initial stop; 96h maximum",
            "entry": "next 1h market open",
            "orders": "market only",
            "money": "3% current-equity margin, fixed 10x gross",
            "objective": ">=30% account return in every month; >=3 trades/day; DD <=10%",
        },
        "feature_rows": len(features),
        "proposal_rows": len(proposals),
        "walkforward": diagnostics,
        "yearly": yearly,
        "combined_2022_2026": combined,
        "objective_checks": checks,
        "sources": {
            **sources,
            str(Path(base.__file__)): digest(Path(base.__file__)),
            str(Path(atr.__file__)): digest(Path(atr.__file__)),
            str(Path(__file__)): digest(Path(__file__)),
        },
        "limitations": [
            "Annual refits cannot react to a regime change until the next year.",
            "The top-3% cutoff comes only from completed training-year predictions.",
            "Historical OKX global results require EU X-Perp forward transfer.",
        ],
    }
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({
        "status": status,
        "feature_rows": len(features),
        "proposal_rows": len(proposals),
        "walkforward": diagnostics,
        "combined_2022_2026": combined,
        "objective_checks": checks,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
