"""Reuse the frozen 2R entry model with literal 0.5%-to-2% exits."""
from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

import crypto_2r_precision_barbell_oct5_audit as old
import crypto_5m_regime_router_literal_oct6_audit as price
import crypto_futures_trade_flow_health_gate_oct7_audit as health
import crypto_futures_trade_flow_ladder_oct7_audit as ladder
import crypto_futures_trade_flow_regime_oct7_audit as flow
import crypto_high_target_tree_oct5_audit as base


OUT = Path("reports/audit_2026_09_08/crypto_2r_model_literal_ladder_oct7.json")
MODEL = {"max_leaf_nodes": 15, "min_samples_leaf": 300,
         "learning_rate": .05}
THRESHOLD = .3492227142
FAR_TARGETS = (1.0, 1.5, 2.0)
LOCK_FRACTIONS = (.25, .50, .75)
MAX_BARS_VALUES = (24, 48, 144, 288)


def attach_indices(frame: pd.DataFrame, candles: dict) -> pd.DataFrame:
    pieces = []
    for symbol, part in frame.groupby("symbol", sort=False):
        if symbol not in candles:
            continue
        times = np.asarray(candles[symbol]["time"], dtype=np.int64)
        entries = part.entry_time.to_numpy(np.int64)
        indices = np.searchsorted(times, entries)
        valid = (indices < len(times)) & (times[np.minimum(indices, len(times)-1)] == entries)
        found = part.loc[valid].copy()
        found["entry_index"] = indices[valid]
        pieces.append(found)
    return pd.concat(pieces, ignore_index=True)


def resolve(frame: pd.DataFrame, candles: dict, far_pct: float,
            lock_fraction: float, max_bars: int) -> pd.DataFrame:
    source = frame.copy()
    source["direction"] = source.direction.map({"LONG": 1, "SHORT": -1}).astype(np.int8)
    result = ladder.add_ladder(source, candles, far_pct, lock_fraction, max_bars)
    result["month"] = (pd.to_datetime(result.entry_time, unit="s", utc=True).dt.year*100
                       + pd.to_datetime(result.entry_time, unit="s", utc=True).dt.month)
    return result


def selection_ok(value: dict) -> bool:
    return bool(value["n"] >= 600 and value["return_pct"] > 0
                and value["literal_target_hit_rate"] >= 70
                and value["dd_pct"] <= 6
                and value["trades_per_calendar_day"] >= 2.5)


def validation_ok(value: dict) -> bool:
    return bool(value["n"] >= 500 and value["return_pct"] > 0
                and value["literal_target_hit_rate"] >= 68
                and value["dd_pct"] <= 7
                and value["trades_per_calendar_day"] >= 2.0)


def main() -> None:
    candles, sources = flow.load()
    frame = old.load_joined()
    fit = base.stage(frame, old.FIT)
    schema = base.design(fit)
    model = HistGradientBoostingClassifier(
        learning_rate=MODEL["learning_rate"], max_iter=150,
        max_leaf_nodes=MODEL["max_leaf_nodes"],
        min_samples_leaf=MODEL["min_samples_leaf"],
        l2_regularization=1.0, early_stopping=False, random_state=20260908)
    model.fit(base.matrix(fit, schema), fit.outcome.eq("TP2").astype(int))
    scored = old.scored(frame, model, schema)
    scored = scored[scored.model_probability.ge(THRESHOLD)].copy()
    scored = attach_indices(scored, candles)
    scored["quality"] = scored.model_probability

    resolved_cache = {}
    selection = []
    frontier = []
    for far, lock, bars in itertools.product(
            FAR_TARGETS, LOCK_FRACTIONS, MAX_BARS_VALUES):
        resolved = resolve(scored, candles, far, lock, bars)
        resolved_cache[(far, lock, bars)] = resolved
        value = price.evaluate(resolved, (2022,), THRESHOLD)
        row = {"exit": {"near_pct": .5, "far_pct": far,
                         "lock_fraction": lock, "max_bars": bars},
               "selection_2022": value}
        frontier.append(row)
        if selection_ok(value):
            selection.append(row)
    frontier.sort(key=lambda item: (
        item["selection_2022"]["return_pct"]
        - 2*item["selection_2022"]["dd_pct"]), reverse=True)
    validated = []
    for item in selection:
        spec = item["exit"]
        resolved = resolved_cache[(spec["far_pct"], spec["lock_fraction"],
                                   spec["max_bars"])]
        value = price.evaluate(resolved, (2023,), THRESHOLD)
        if validation_ok(value):
            candidate = {**item, "validation_2023": value}
            candidate["score"] = round(
                min(item["selection_2022"]["return_pct"], value["return_pct"])
                - 2*max(item["selection_2022"]["dd_pct"], value["dd_pct"]), 8)
            validated.append(candidate)
    validated.sort(key=lambda item: item["score"], reverse=True)
    frozen = validated[0] if validated else None
    test_2024 = later = checks = None
    if frozen:
        spec = frozen["exit"]
        resolved = resolved_cache[(spec["far_pct"], spec["lock_fraction"],
                                   spec["max_bars"])]
        test_2024 = price.evaluate(resolved, (2024,), THRESHOLD)
        if validation_ok(test_2024):
            later = {str(year): price.evaluate(resolved, (year,), THRESHOLD)
                     for year in (2025, 2026)}
            checks = {year: validation_ok(value)
                      for year, value in later.items()}
    diagnostic_transfer = None
    if frontier:
        spec = frontier[0]["exit"]
        resolved = resolved_cache[(spec["far_pct"], spec["lock_fraction"],
                                   spec["max_bars"])]
        diagnostic_transfer = {
            str(year): price.evaluate(resolved, (year,), THRESHOLD)
            for year in (2023, 2024, 2025, 2026)}
    report = {
        "status": ("RESEARCH_ONLY_NEEDS_NEW_FORWARD"
                   if checks and all(checks.values()) else "REJECTED"),
        "protocol": {
            "entry_model": "2R classifier frozen before 2024",
            "model": MODEL, "threshold": THRESHOLD,
            "new_exit_selection": 2022, "new_exit_validation": 2023,
            "new_exit_test": 2024, "observed": [2025, 2026],
            "near_pct": .5, "far_pct": FAR_TARGETS,
            "lock_fraction": LOCK_FRACTIONS, "max_bars": MAX_BARS_VALUES,
            "initial_stop_pct": 1.0,
            "money": "3% equity margin, fixed 10x gross",
            "warning": "literal exit grid was proposed after later history was known",
        },
        "scored_rows_with_candles": len(scored),
        "qualified_2022": len(selection), "qualified_2023": len(validated),
        "frozen_before_2024": frozen, "test_2024": test_2024,
        "later": later, "later_checks": checks,
        "best_rejected_transfer_diagnostic": diagnostic_transfer,
        "selection_frontier": frontier,
        "sources": sources,
        "limitations": [
            "The new literal exit policy was devised after observing later periods.",
            "All historical results are development evidence for this combination.",
            "A pass requires genuinely new direct X-Perp shadow evidence.",
        ],
    }
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({key: report[key] for key in (
        "status", "scored_rows_with_candles", "qualified_2022",
        "qualified_2023", "frozen_before_2024", "test_2024",
        "later", "later_checks", "best_rejected_transfer_diagnostic")},
        indent=2), flush=True)
    print(json.dumps({"top_2022": frontier[:5]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
