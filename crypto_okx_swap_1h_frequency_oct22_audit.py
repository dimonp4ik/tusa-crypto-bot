"""Sequential audit of a frequency-first OKX one-hour candidate.

The model/configuration search is inherited unchanged from the October 7
audit.  A candidate is selected using 2022-2023 only with a hard three-trades
per calendar-day floor, then 2024 and 2025-2026 are revealed in sequence.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from sklearn.ensemble import HistGradientBoostingClassifier

import crypto_5m_regime_router_literal_oct6_audit as portfolio
import crypto_okx_swap_1h_regime_oct7_audit as base


ROOT = Path("reports/audit_2026_09_08")
SOURCE_REPORT = ROOT / "crypto_okx_swap_1h_regime_oct7.json"
OUT = ROOT / "crypto_okx_swap_1h_frequency_oct22.json"
ROWS_OUT = ROOT / "crypto_okx_swap_1h_frequency_oct22.pkl"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def candidate_ok(candidate: dict) -> bool:
    selection = candidate["selection_2022"]
    validation = candidate["validation_2023"]
    return bool(
        base.selection_ok(selection)
        and base.validation_ok(validation)
        and selection["trades_per_calendar_day"] >= 3
        and validation["trades_per_calendar_day"] >= 3
    )


def rank(candidate: dict) -> tuple[float, float, float]:
    selection = candidate["selection_2022"]
    validation = candidate["validation_2023"]
    return (
        min(selection["return_pct"], validation["return_pct"]),
        selection["return_pct"] + validation["return_pct"],
        min(selection["trades_per_calendar_day"],
            validation["trades_per_calendar_day"]),
    )


def future_ok(value: dict) -> bool:
    return bool(
        base.validation_ok(value)
        and value["trades_per_calendar_day"] >= 3
    )


def main() -> None:
    source = json.loads(SOURCE_REPORT.read_text(encoding="utf-8"))
    candidates = [item for item in source["validation_diagnostics"]
                  if candidate_ok(item)]
    if not candidates:
        raise RuntimeError("no frequency-first pre-2024 candidate")
    frozen = max(candidates, key=rank)

    candles, sources = base.load()
    raw = base.build_features(candles)
    frame = base.add_outcomes(base.proposals(raw), candles)
    fit = frame.year.isin((2020, 2021)).to_numpy()
    columns = base.schema(frame.loc[fit])
    x = base.matrix(frame, columns)
    labels = frame.target_hit.astype(int).to_numpy()
    config = frozen["model"]
    model = HistGradientBoostingClassifier(
        max_iter=180, learning_rate=.06,
        max_leaf_nodes=int(config["leaves"]),
        min_samples_leaf=int(config["leaf"]),
        l2_regularization=float(config["l2"]),
        random_state=20261007)
    model.fit(x[fit], labels[fit])
    frame["quality"] = model.predict_proba(x)[:, 1]
    threshold = float(frozen["threshold"])

    rebuilt = {
        "selection_2022": portfolio.evaluate(frame, (2022,), threshold),
        "validation_2023": portfolio.evaluate(frame, (2023,), threshold),
    }
    for stage in rebuilt:
        expected = frozen[stage]
        for field in ("n", "return_pct", "dd_pct",
                      "literal_target_hit_rate", "trades_per_calendar_day"):
            if rebuilt[stage][field] != expected[field]:
                raise RuntimeError(
                    f"{stage} mismatch {field}: "
                    f"{rebuilt[stage][field]} != {expected[field]}")

    test_2024 = portfolio.evaluate(frame, (2024,), threshold)
    test_2024_passed = future_ok(test_2024)
    later = None
    later_checks = None
    if test_2024_passed:
        later = {str(year): portfolio.evaluate(frame, (year,), threshold)
                 for year in (2025, 2026)}
        later_checks = {year: future_ok(value)
                        for year, value in later.items()}

    selected_rows = frame.loc[frame.quality.ge(threshold)].copy()
    selected_rows.to_pickle(ROWS_OUT)
    passed = bool(later_checks and all(later_checks.values()))
    report = {
        "status": ("FREQUENCY_ROUTE_AWAITING_DIRECT_XPERP"
                   if passed else "REJECTED_SEQUENTIAL_HOLDOUT"),
        "ready_for_live_orders": False,
        "order_eligible": False,
        "protocol": {
            "fit": [2020, 2021],
            "selection_and_ranking": [2022, 2023],
            "sequential_holdouts": [2024, 2025, 2026],
            "candidate_source": "October 7 diagnostics generated before 2024 reveal",
            "frequency_floor": "3 trades per calendar day in every stage",
            "rank": "maximize worst development return, then total return",
            "entry": "next 1-hour market open",
            "target_pct": .5,
            "stop_pct": 1.0,
            "hold_bars": 12,
            "money": "client-selected 3% current-equity margin, fixed 10x gross",
        },
        "eligible_before_2024": len(candidates),
        "frozen_before_2024": {
            "model": config,
            "quantile": frozen["quantile"],
            "threshold": threshold,
            **rebuilt,
        },
        "test_2024": test_2024,
        "test_2024_passed": test_2024_passed,
        "later": later,
        "later_checks": later_checks,
        "objective_checks": {
            "all_historical_holdouts_passed": passed,
            "ready_for_live_orders": False,
        },
        "sources": {
            **sources,
            str(SOURCE_REPORT): digest(SOURCE_REPORT),
            str(Path(__file__).resolve()): digest(Path(__file__).resolve()),
        },
        "limitations": [
            "The model uses hourly OHLCV and has no historical order-book depth.",
            "Direct X-Perp is intentionally not evaluated unless every historical holdout passes.",
            "Gross results exclude fees at user request.",
            "No backtest guarantees a future monthly return.",
        ],
    }
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({
        "status": report["status"],
        "eligible_before_2024": len(candidates),
        "frozen_before_2024": report["frozen_before_2024"],
        "test_2024": test_2024,
        "test_2024_passed": test_2024_passed,
        "later": later,
        "later_checks": later_checks,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
