"""Literal 0.5%-to-2% ladder on the frozen geometry-router entries."""
from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

import crypto_5m_literal_geometry_router_oct6_audit as geometry
import crypto_5m_regime_router_literal_oct6_audit as base
import crypto_5m_tree_filter_audit as candle_source
import crypto_futures_trade_flow_ladder_oct7_audit as ladder


OUT = Path("reports/audit_2026_09_08/crypto_literal_geometry_ladder_oct7.json")
ROWS_OUT = Path("reports/audit_2026_09_08/crypto_literal_geometry_ladder_oct7.pkl")
THRESHOLD = .6916756310092697
FAR_TARGETS = (1.0, 1.5, 2.0)
LOCK_FRACTIONS = (.25, .50, .75)
MAX_BARS_VALUES = (24, 48, 144)


def selection_ok(value: dict) -> bool:
    return bool(value["n"] >= 1200 and value["return_pct"] > 0
                and value["literal_target_hit_rate"] >= 68
                and value["dd_pct"] <= 7
                and value["trades_per_calendar_day"] >= 3)


def validation_ok(value: dict) -> bool:
    return bool(value["n"] >= 700 and value["return_pct"] > 0
                and value["literal_target_hit_rate"] >= 65
                and value["dd_pct"] <= 7
                and value["trades_per_calendar_day"] >= 2)


def main() -> None:
    candles, sources = candle_source.load()
    raw = base.build_features(candles)
    proposals = base.proposals(raw)
    labelled = geometry.add_outcomes(proposals, candles, 1.0, 24)
    schema = base.columns(labelled[labelled.year.isin((2020, 2021))])
    matrix = base.matrix(labelled, schema)
    fit = labelled.year.isin((2020, 2021)).to_numpy()
    model = HistGradientBoostingClassifier(
        max_iter=180, learning_rate=.06, max_leaf_nodes=15,
        min_samples_leaf=300, l2_regularization=4.0,
        random_state=20261006)
    model.fit(matrix[fit], labelled.loc[fit, "target_hit"].astype(int))
    quality = model.predict_proba(matrix)[:, 1]

    cache = {}
    selection = []
    frontier = []
    for far, lock, bars in itertools.product(
            FAR_TARGETS, LOCK_FRACTIONS, MAX_BARS_VALUES):
        resolved = ladder.add_ladder(proposals, candles, far, lock, bars)
        resolved["quality"] = quality
        cache[(far, lock, bars)] = resolved
        value = base.evaluate(resolved, (2022,), THRESHOLD)
        item = {"exit": {"near_pct": .5, "far_pct": far,
                          "lock_fraction": lock, "max_bars": bars},
                "selection_2022": value}
        frontier.append(item)
        if selection_ok(value):
            selection.append(item)
    validated = []
    for item in selection:
        spec = item["exit"]
        resolved = cache[(spec["far_pct"], spec["lock_fraction"],
                          spec["max_bars"])]
        value = base.evaluate(resolved, (2023,), THRESHOLD)
        if validation_ok(value):
            candidate = {**item, "validation_2023": value}
            candidate["score"] = round(
                min(item["selection_2022"]["return_pct"], value["return_pct"])
                - 2*max(item["selection_2022"]["dd_pct"], value["dd_pct"]), 8)
            validated.append(candidate)
    validated.sort(key=lambda item: item["score"], reverse=True)
    frontier.sort(key=lambda item: (
        item["selection_2022"]["return_pct"]
        - 2*item["selection_2022"]["dd_pct"]), reverse=True)
    frozen = validated[0] if validated else None
    test_2024 = later = checks = None
    if frozen:
        spec = frozen["exit"]
        resolved = cache[(spec["far_pct"], spec["lock_fraction"],
                          spec["max_bars"])]
        test_2024 = base.evaluate(resolved, (2024,), THRESHOLD)
        if validation_ok(test_2024):
            later = {str(year): base.evaluate(resolved, (year,), THRESHOLD)
                     for year in (2025, 2026)}
            checks = {year: validation_ok(value)
                      for year, value in later.items()}
        rows = []
        for year in range(2022, 2027):
            rows.extend(base.portfolio_rows(resolved, (year,), THRESHOLD))
        pd.DataFrame(rows).to_pickle(ROWS_OUT, compression="gzip")
    report = {
        "status": ("RESEARCH_ONLY_NEEDS_NEW_FORWARD"
                   if checks and all(checks.values()) else "REJECTED"),
        "protocol": {
            "entry_model": "literal +0.5% geometry model frozen before 2025",
            "threshold": THRESHOLD,
            "exit_selection": 2022, "exit_validation_and_ranking": 2023,
            "untouched_exit_test": 2024, "observed": [2025, 2026],
            "near_pct": .5, "far_pct": FAR_TARGETS,
            "lock_fraction": LOCK_FRACTIONS, "max_bars": MAX_BARS_VALUES,
            "initial_stop_pct": 1.0,
            "money": "3% equity margin, fixed 10x gross",
            "warning": "ladder proposed after later history was known",
        },
        "proposal_rows": len(proposals),
        "qualified_2022": len(selection), "qualified_2023": len(validated),
        "frozen_before_2024": frozen, "test_2024": test_2024,
        "later": later, "later_checks": checks,
        "frontier": frontier,
        "sources": sources,
        "limitations": [
            "The ladder was added after the entry model's later history was known.",
            "All historical ladder results are development evidence.",
            "A pass requires genuinely new direct X-Perp shadow evidence.",
        ],
    }
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({key: report[key] for key in (
        "status", "proposal_rows", "qualified_2022", "qualified_2023",
        "frozen_before_2024", "test_2024", "later", "later_checks")},
        indent=2), flush=True)


if __name__ == "__main__":
    main()
