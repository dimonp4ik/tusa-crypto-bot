"""Bounded family-subset selection for the pretest-stable trade-flow model."""
from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

import crypto_5m_regime_router_literal_oct6_audit as price
import crypto_futures_trade_flow_regime_oct7_audit as flow


OUT = Path("reports/audit_2026_09_08/crypto_futures_trade_flow_family_selection_oct7.json")
ROWS_OUT = Path("reports/audit_2026_09_08/crypto_futures_trade_flow_family_selection_oct7.pkl")
MODEL = {"leaves": 31, "leaf": 400, "l2": 5.0}
QUANTILE = .95


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def evaluate_subset(frame, families: tuple[str, ...], year: int, threshold: float):
    mask = np.zeros(len(frame), dtype=bool)
    for family in families:
        mask |= frame[f"flow_family_{family}"].eq(1).to_numpy()
    return price.evaluate(frame.loc[mask].copy(), (year,), threshold)


def pretest_ok(values: dict) -> bool:
    return all(item["n"] >= 300 and item["return_pct"] > 0
               and item["literal_target_hit_rate"] >= 67
               and item["dd_pct"] <= 7
               and item["trades_per_calendar_day"] >= 2.5
               for item in values.values())


def objective_ok(value: dict) -> bool:
    return bool(value["n"] >= 500 and value["return_pct"] > 0
                and 75 <= value["literal_target_hit_rate"] <= 95
                and value["dd_pct"] <= 5 and value["negative_months"] == 0
                and value["trades_per_calendar_day"] >= 3
                and value["months_ge_30pct"] == value["months"])


def main() -> None:
    candles, sources = flow.load()
    features = flow.attach_flow(price.build_features(candles), candles)
    frame = price.add_outcomes(flow.proposals(features), candles)
    fit = frame.year.isin((2020, 2021)).to_numpy()
    selection = frame.year.eq(2022).to_numpy()
    schema = price.columns(frame.loc[fit])
    _, matrix = flow.matrices(frame, schema)
    y = frame.target_hit.astype(int).to_numpy()
    model = HistGradientBoostingClassifier(
        max_iter=180, learning_rate=.06, max_leaf_nodes=MODEL["leaves"],
        min_samples_leaf=MODEL["leaf"], l2_regularization=MODEL["l2"],
        random_state=20261007)
    model.fit(matrix[fit], y[fit])
    scored = frame.copy()
    scored["quality"] = model.predict_proba(matrix)[:, 1]
    threshold = float(scored.loc[selection, "quality"].quantile(QUANTILE))

    candidates, frontier = [], []
    for count in range(1, len(flow.FAMILIES)+1):
        for families in itertools.combinations(flow.FAMILIES, count):
            values = {str(year): evaluate_subset(scored, families, year, threshold)
                      for year in (2022, 2023, 2024)}
            item = {"families": families, "pretest": values}
            item["score"] = round(sum(
                value["return_pct"]-2*value["dd_pct"]
                +.15*value["literal_target_hit_rate"]
                for value in values.values()), 8)
            frontier.append(item)
            if pretest_ok(values):
                candidates.append(item)
    candidates.sort(key=lambda item: item["score"], reverse=True)
    frontier.sort(key=lambda item: (
        min(value["literal_target_hit_rate"] for value in item["pretest"].values()),
        min(value["trades_per_calendar_day"] for value in item["pretest"].values()),
        item["score"]), reverse=True)
    frozen = candidates[0] if candidates else None
    results = checks = None
    if frozen:
        results = {str(year): evaluate_subset(
            scored, tuple(frozen["families"]), year, threshold)
            for year in (2022, 2023, 2024, 2025, 2026)}
        checks = {year: objective_ok(value) for year, value in results.items()}
        mask = np.zeros(len(scored), dtype=bool)
        for family in frozen["families"]:
            mask |= scored[f"flow_family_{family}"].eq(1).to_numpy()
        scored.loc[mask & scored.quality.ge(threshold)].to_pickle(
            ROWS_OUT, compression="gzip")
    report = {
        "status": ("RESEARCH_CANDIDATE_AWAITING_VENUE_FORWARD"
                   if checks and all(checks.values()) else "REJECTED"),
        "protocol": {"fit": "2020-2021", "threshold_selection": "2022",
                     "family_selection": "2022-2024",
                     "family_subsets": 2**len(flow.FAMILIES)-1,
                     "frozen_before": ["2025", "2026"],
                     "model": MODEL, "quantile": QUANTILE,
                     "threshold": threshold, "target_pct": .5,
                     "stop_pct": 1.0, "max_bars": 24,
                     "money": "3% equity margin, fixed 10x gross"},
        "qualified_before_test": len(candidates),
        "frozen_before_test": frozen,
        "top_before_test": candidates[:15],
        "pretest_frontier": frontier[:20],
        "results": results, "checks": checks,
        "sources": {**sources, str(Path(flow.__file__)): digest(Path(flow.__file__)),
                    str(Path(__file__)): digest(Path(__file__))},
        "limitations": [
            "Even a bounded 63-subset search creates selection risk.",
            "The family flags overlap and do not identify independent trades.",
            "Any historical pass still requires untouched direct X-Perp evidence.",
        ],
    }
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"status": report["status"],
                      "qualified_before_test": len(candidates),
                      "frozen": frozen, "results": results,
                      "checks": checks}, indent=2), flush=True)


if __name__ == "__main__":
    main()
