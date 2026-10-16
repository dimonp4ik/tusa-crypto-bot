"""Per-signal expected-value exit router for causal crypto entries.

Four classifiers fitted on 2020-2021 estimate whether each literal target from
0.5% to 2% is reached before a 1% stop within 24 hours.  At entry the router
chooses the target with the largest predicted binary payoff expectancy.  The
minimum score is selected on 2022 and frozen before 2023-2026 are evaluated.
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
OUT = ROOT / "crypto_profit_first_expected_value_router_oct17.json"
ROWS_OUT = ROOT / "crypto_profit_first_expected_value_router_oct17.pkl"
TARGETS_PCT = (.5, 1.0, 1.5, 2.0)
HOLD_BARS = 24
STOP_PCT = 1.0
QUANTILES = (.70, .75, .80, .85, .875, .90, .925, .95, .96, .97)
FIT_YEARS = (2020, 2021)
SELECTION_YEAR = 2022
VALIDATION_YEAR = 2023
TEST_YEAR = 2024
CONFIRMATION_YEARS = (2025, 2026)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def expected_payoff(probability: np.ndarray, target_pct: float) -> np.ndarray:
    probability = np.asarray(probability, dtype=float)
    return probability * (target_pct / 100.0) - (1.0 - probability) * (STOP_PCT / 100.0)


def route_profiles(frames: list[pd.DataFrame], probabilities: np.ndarray,
                   targets: tuple[float, ...] = TARGETS_PCT) -> pd.DataFrame:
    if probabilities.shape != (len(frames[0]), len(targets)):
        raise ValueError("probability matrix does not match rows and targets")
    keys = ["symbol", "entry_time", "direction"]
    reference = frames[0][keys].reset_index(drop=True)
    for frame in frames[1:]:
        if not reference.equals(frame[keys].reset_index(drop=True)):
            raise ValueError("literal target frames are not row-aligned")
    edges = np.column_stack([
        expected_payoff(probabilities[:, index], target)
        for index, target in enumerate(targets)
    ])
    choice = np.argmax(edges, axis=1)
    output = frames[0].copy().reset_index(drop=True)
    outcome_columns = ["signed_return", "exit_time", "target_hit",
                       "exit_reason", "literal_target_pct", "hold_bars"]
    for index, frame in enumerate(frames):
        mask = choice == index
        if mask.any():
            source = frame.reset_index(drop=True)
            # Preserve bool/int/float dtypes independently.  Assigning this
            # mixed block as one object matrix fails on strict pandas builds.
            for column in outcome_columns:
                output.loc[mask, column] = source.loc[mask, column].to_numpy()
    rows = np.arange(len(output))
    output["quality"] = edges[rows, choice]
    output["predicted_target_probability"] = probabilities[rows, choice]
    output["routed_target_pct"] = np.asarray(targets, dtype=float)[choice]
    return output


def evaluate(frame: pd.DataFrame, years: tuple[int, ...], threshold: float) -> dict:
    value = enrich(portfolio.evaluate(frame, years, threshold))
    rows = portfolio.portfolio_rows(frame, years, threshold)
    counts = pd.Series([float(row["routed_target_pct"]) for row in rows]).value_counts()
    value["routed_target_counts"] = {
        str(target): int(counts.get(target, 0)) for target in TARGETS_PCT}
    return value


def selection_ok(value: dict) -> bool:
    return bool(value["return_pct"] > 0 and value["dd_pct"] <= 15
                and value["trades_per_calendar_day"] >= 3)


def profile_score(value: dict) -> float:
    return round(value["equivalent_monthly_return_pct"]
                 - .10 * value["dd_pct"]
                 - .20 * value["negative_months"], 8)


def objective_ok(value: dict) -> bool:
    return bool(value["months"] > 0 and value["dd_pct"] <= 10
                and value["trades_per_calendar_day"] >= 3
                and value["months_ge_30pct"] == value["months"])


def main() -> None:
    candles, sources = base.load()
    features = base.build_features(candles)
    proposals = base.proposals(features)
    frames = []
    predictions = []
    model_metadata = []
    for target in TARGETS_PCT:
        frame = literal.add_outcomes(
            proposals, candles, target, HOLD_BARS).sort_values(
                ["symbol", "entry_time", "direction"]).reset_index(drop=True)
        train = frame[frame.year.isin(FIT_YEARS)]
        columns = base.schema(train)
        estimator = literal.model()
        estimator.fit(base.matrix(train, columns),
                      train.target_hit.to_numpy(np.int8))
        probability = estimator.predict_proba(base.matrix(frame, columns))[:, 1]
        frames.append(frame)
        predictions.append(probability)
        model_metadata.append({
            "target_pct": target,
            "fit_rows": len(train),
            "fit_target_prevalence_pct": round(100 * float(train.target_hit.mean()), 4),
        })
    routed = route_profiles(frames, np.column_stack(predictions))
    selection_scores = routed.loc[routed.year.eq(SELECTION_YEAR), "quality"]
    candidates = []
    for quantile in QUANTILES:
        threshold = float(selection_scores.quantile(quantile))
        value = evaluate(routed, (SELECTION_YEAR,), threshold)
        candidates.append({
            "selection_quantile": quantile,
            "threshold_expected_price_return": threshold,
            "selection_2022": value,
            "selection_pass": selection_ok(value),
            "score": profile_score(value),
        })
    candidates.sort(key=lambda item: (
        item["selection_pass"], item["score"]), reverse=True)
    frozen = candidates[0] if candidates and candidates[0]["selection_pass"] else None
    validation = test = confirmation = checks = None
    status = "REJECTED_SELECTION"
    if frozen:
        threshold = frozen["threshold_expected_price_return"]
        validation = evaluate(routed, (VALIDATION_YEAR,), threshold)
        test = evaluate(routed, (TEST_YEAR,), threshold)
        confirmation = {str(year): evaluate(routed, (year,), threshold)
                        for year in CONFIRMATION_YEARS}
        yearly = {str(SELECTION_YEAR): frozen["selection_2022"],
                  str(VALIDATION_YEAR): validation, str(TEST_YEAR): test,
                  **confirmation}
        checks = {year: objective_ok(value) for year, value in yearly.items()}
        status = ("EXPECTED_VALUE_ROUTER_RESEARCH_CANDIDATE"
                  if all(checks.values()) else "REJECTED_OBJECTIVE")
        accepted = portfolio.portfolio_rows(
            routed, (SELECTION_YEAR, VALIDATION_YEAR, TEST_YEAR,
                     *CONFIRMATION_YEARS), threshold)
        pd.DataFrame(accepted).to_pickle(ROWS_OUT)
    report = {
        "status": status, "ready_for_live_orders": False,
        "protocol": {
            "priority": "account return first; win rate diagnostic only",
            "fit": list(FIT_YEARS), "router_and_cutoff_selection": SELECTION_YEAR,
            "frozen_validation": VALIDATION_YEAR, "untouched_test": TEST_YEAR,
            "confirmation_only": list(CONFIRMATION_YEARS),
            "targets_pct": list(TARGETS_PCT), "stop_pct": STOP_PCT,
            "hold_hours": HOLD_BARS,
            "routing": "maximum predicted target-vs-stop binary expectancy at entry",
            "entry": "next one-hour market open", "orders": "market only",
            "money": "3% current-equity margin, fixed 10x gross",
            "objective": ">=30% account return in every month; >=3 trades/day; DD <=10%",
        },
        "feature_rows": len(features), "proposal_rows": len(proposals),
        "models": model_metadata, "candidate_count": len(candidates),
        "selection_frontier": candidates,
        "frozen_before_2023": frozen,
        "validation_2023": validation, "test_2024": test,
        "confirmation_2025_2026": confirmation,
        "objective_checks": checks,
        "sources": {**sources, str(Path(base.__file__)): digest(Path(base.__file__)),
                    str(Path(literal.__file__)): digest(Path(literal.__file__)),
                    str(Path(__file__)): digest(Path(__file__))},
        "limitations": [
            "Binary target-vs-stop expectancy approximates timeout returns as full losses.",
            "The target router and numeric cutoff are selected on 2022 only.",
            "Historical OKX global results require direct EU X-Perp shadow transfer.",
        ],
    }
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({
        "status": status, "models": model_metadata,
        "frozen": frozen, "validation_2023": validation,
        "test_2024": test, "confirmation_2025_2026": confirmation,
        "objective_checks": checks,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
