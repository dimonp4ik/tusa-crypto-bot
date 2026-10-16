"""Yearly walk-forward classifier for the frozen two-percent crypto runner.

The literal 2% target, 1% stop, 24-hour hold and top-10% selection rate come
from the earlier 2022 development audit.  For every later calendar year the
classifier is refitted on only the two preceding completed years.  Its numeric
cutoff is the 90th percentile of predictions on those completed training rows,
so the tested year's score distribution is never used.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

import crypto_5m_regime_router_literal_oct6_audit as portfolio
import crypto_okx_swap_1h_regime_oct7_audit as base
import crypto_profit_first_literal_runner_oct16_audit as literal
from crypto_profit_first_runner_oct16_audit import enrich


ROOT = Path("reports/audit_2026_09_08")
OUT = ROOT / "crypto_profit_first_rolling_literal_runner_oct17.json"
ROWS_OUT = ROOT / "crypto_profit_first_rolling_literal_runner_oct17.pkl"
TARGET_PCT = 2.0
HOLD_BARS = 24
SELECTION_QUANTILE = .90
TEST_YEARS = (2022, 2023, 2024, 2025, 2026)
TRAINING_YEARS = {
    2022: (2020, 2021),
    2023: (2021, 2022),
    2024: (2022, 2023),
    2025: (2023, 2024),
    2026: (2024, 2025),
}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def training_cutoff(predictions: np.ndarray) -> float:
    return float(np.quantile(np.asarray(predictions, dtype=float),
                             SELECTION_QUANTILE))


def objective_ok(value: dict) -> bool:
    return bool(value["months"] > 0 and value["dd_pct"] <= 10
                and value["trades_per_calendar_day"] >= 3
                and value["months_ge_30pct"] == value["months"])


def score_year(frame: pd.DataFrame, year: int) -> tuple[pd.DataFrame, dict]:
    training_years = TRAINING_YEARS[year]
    train = frame[frame.year.isin(training_years)]
    test = frame[frame.year.eq(year)].copy()
    columns = base.schema(train)
    train_matrix = base.matrix(train, columns)
    estimator = literal.model()
    estimator.fit(train_matrix, train.target_hit.to_numpy(np.int8))
    train_predictions = estimator.predict_proba(train_matrix)[:, 1]
    threshold = training_cutoff(train_predictions)
    test["quality"] = estimator.predict_proba(base.matrix(test, columns))[:, 1]
    accepted = portfolio.portfolio_rows(test, (year,), threshold)
    result = enrich(portfolio.evaluate(test, (year,), threshold))
    return pd.DataFrame(accepted), {
        "year": year,
        "training_years": list(training_years),
        "training_rows": len(train),
        "candidate_rows": len(test),
        "threshold": threshold,
        "pre_portfolio_rows": int(test.quality.ge(threshold).sum()),
        "accepted_rows": len(accepted),
        "result": result,
    }


def main() -> None:
    candles, sources = base.load()
    features = base.build_features(candles)
    proposals = base.proposals(features)
    frame = literal.add_outcomes(
        proposals, candles, TARGET_PCT, HOLD_BARS)
    pieces = []
    diagnostics = []
    yearly = {}
    for year in TEST_YEARS:
        accepted, item = score_year(frame, year)
        pieces.append(accepted)
        diagnostics.append(item)
        yearly[str(year)] = item["result"]
    emitted = pd.concat(pieces, ignore_index=True)
    combined = enrich(portfolio.money(emitted.to_dict("records"), .03))
    checks = {year: objective_ok(value) for year, value in yearly.items()}
    status = ("ROLLING_LITERAL_RUNNER_RESEARCH_CANDIDATE"
              if all(checks.values()) else "REJECTED_OBJECTIVE")
    emitted.to_pickle(ROWS_OUT)
    report = {
        "status": status,
        "ready_for_live_orders": False,
        "protocol": {
            "priority": "account return first; win rate diagnostic only",
            "training": "two preceding completed calendar years",
            "threshold": "90th percentile of completed training-year predictions",
            "target_pct": TARGET_PCT,
            "stop_pct": 1.0,
            "hold_hours": HOLD_BARS,
            "entry": "next one-hour market open",
            "orders": "market only",
            "money": "3% current-equity margin, fixed 10x gross",
            "objective": ">=30% account return in every month; >=3 trades/day; DD <=10%",
        },
        "feature_rows": len(features),
        "proposal_rows": len(proposals),
        "walkforward": diagnostics,
        "combined_2022_2026": combined,
        "objective_checks": checks,
        "sources": {
            **sources,
            str(Path(base.__file__)): digest(Path(base.__file__)),
            str(Path(literal.__file__)): digest(Path(literal.__file__)),
            str(Path(__file__)): digest(Path(__file__)),
        },
        "limitations": [
            "The 2022 row remains development because it fixed the target and selection rate.",
            "Only 2023 onward is strict walk-forward evidence for this exact system.",
            "Historical OKX global results require direct EU X-Perp shadow transfer.",
        ],
    }
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({
        "status": status,
        "walkforward": diagnostics,
        "combined_2022_2026": combined,
        "objective_checks": checks,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
