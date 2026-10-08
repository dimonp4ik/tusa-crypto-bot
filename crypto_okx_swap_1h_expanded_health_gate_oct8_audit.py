"""Causal online health gate for the expanded 2% OKX hourly route.

The underlying cross-symbol model and entry threshold are frozen from the
2022 development route.  The gate sees only outcomes whose exit time is known
at the next entry and keeps observing rejected signals in shadow.  Gate
parameters are selected on 2022 and then applied unchanged to 2023.
"""
from __future__ import annotations

import hashlib
import heapq
import itertools
import json
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

import crypto_5m_regime_router_literal_oct6_audit as portfolio
import crypto_okx_swap_1h_expanded_2pct_oct8_audit as expanded
import crypto_okx_swap_1h_regime_oct7_audit as base
import crypto_okx_swap_1h_target_cascade_oct7_audit as target_engine
from src.research_profit_gate import economic_gate


ROOT = Path("reports/audit_2026_09_08")
MANIFEST = ROOT / "crypto_okx_swap_1h_expanded_oct8" / "manifest.json"
SOURCE_REPORT = ROOT / "crypto_okx_swap_1h_expanded_2pct_oct8.json"
OUT = ROOT / "crypto_okx_swap_1h_expanded_health_gate_oct8.json"
MODEL_CONFIG = {"leaves": 15, "leaf": 900, "l2": 6.0}
BASE_THRESHOLD = 0.41346649684842546
WINDOWS = (40, 80, 160)
MIN_HIT_RATES = (.30, .32, .34, .36, .38)
MIN_MEANS = (-.0005, 0.0, .0005, .001)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_scored() -> tuple[pd.DataFrame, dict]:
    base.MANIFEST = MANIFEST
    candles, sources = base.load()
    raw = base.build_features(candles)
    frame = base.proposals(raw).reset_index(drop=True)
    target_engine.add_target(frame, candles, 2.0)
    frame["signed_return"] = frame["signed_2_0"]
    frame["exit_time"] = frame["exit_2_0"]
    frame["target_hit"] = frame["hit_2_0"]
    frame["exit_reason"] = frame["reason_2_0"]
    frame["selected_target_pct"] = 2.0
    fit = frame.year.isin((2020, 2021)).to_numpy()
    columns = [*base.FEATURES,
               *[f"family_{name}" for name in base.FAMILIES]]
    x = base.matrix(frame, columns)
    labels = frame.target_hit.astype(int).to_numpy()
    model = HistGradientBoostingClassifier(
        max_iter=180, learning_rate=.06,
        max_leaf_nodes=MODEL_CONFIG["leaves"],
        min_samples_leaf=MODEL_CONFIG["leaf"],
        l2_regularization=MODEL_CONFIG["l2"], random_state=20261008)
    model.fit(x[fit], labels[fit])
    frame["quality"] = model.predict_proba(x)[:, 1]
    return frame, sources


def health_filter(candidates: pd.DataFrame, window: int,
                  min_hit_rate: float, min_mean: float,
                  output_year: int) -> pd.DataFrame:
    """Return entries accepted from a trailing, fully resolved outcome gate."""
    ordered = candidates.sort_values(
        ["entry_time", "quality", "symbol"],
        ascending=[True, False, True]).to_dict("records")
    pending: list[tuple[float, int, dict]] = []
    history: deque[tuple[float, float]] = deque(maxlen=window)
    accepted = []
    sequence = 0
    for row in ordered:
        now = float(row["entry_time"])
        while pending and pending[0][0] <= now:
            _, _, done = heapq.heappop(pending)
            history.append((float(bool(done["target_hit"])),
                            float(done["signed_return"])))
        sequence += 1
        heapq.heappush(pending, (float(row["exit_time"]), sequence, row))
        if int(row["year"]) != output_year or len(history) < window:
            continue
        hits = sum(item[0] for item in history) / len(history)
        mean = sum(item[1] for item in history) / len(history)
        if hits >= min_hit_rate and mean >= min_mean:
            accepted.append(row)
    return pd.DataFrame(accepted, columns=candidates.columns)


def evaluate(filtered: pd.DataFrame, year: int) -> dict:
    if filtered.empty:
        metrics = portfolio.evaluate(filtered, (year,), .5)
        gate = economic_gate([], 0.0)
        return {"portfolio": metrics, "economic_gate": gate}
    found = filtered.copy()
    found["quality"] = 1.0
    metrics = portfolio.evaluate(found, (year,), .5)
    rows = portfolio.portfolio_rows(found, (year,), .5)
    gate = economic_gate(
        [float(row["signed_price_return"]) for row in rows],
        metrics["trades_per_calendar_day"], market_days=30)
    return {"portfolio": metrics, "economic_gate": gate}


def passes(value: dict) -> bool:
    metrics, gate = value["portfolio"], value["economic_gate"]
    return bool(metrics["n"] >= 900
                and metrics["literal_target_hit_rate"] >= 65
                and metrics["dd_pct"] <= 8
                and metrics["negative_months"] == 0
                and gate["passes"])


def rank(value: dict) -> tuple[float, float, float, float]:
    metrics, gate = value["portfolio"], value["economic_gate"]
    return (gate["projected_monthly_return_pct"],
            metrics["literal_target_hit_rate"],
            -metrics["negative_months"], -metrics["dd_pct"])


def main() -> None:
    frame, sources = build_scored()
    candidates = frame[
        frame.quality.ge(BASE_THRESHOLD) & frame.year.between(2021, 2023)
    ].copy()
    selection = []
    for window, min_hit_rate, min_mean in itertools.product(
            WINDOWS, MIN_HIT_RATES, MIN_MEANS):
        filtered = health_filter(
            candidates, window, min_hit_rate, min_mean, 2022)
        value = evaluate(filtered, 2022)
        selection.append({
            "window": window, "minimum_trailing_hit_rate": min_hit_rate,
            "minimum_trailing_mean_signed_return": min_mean,
            "selection_2022": value,
        })
    selection.sort(key=lambda item: rank(item["selection_2022"]), reverse=True)
    validation = []
    for item in selection[:20]:
        filtered = health_filter(
            candidates, item["window"], item["minimum_trailing_hit_rate"],
            item["minimum_trailing_mean_signed_return"], 2023)
        value = evaluate(filtered, 2023)
        validation.append({**item, "validation_2023": value,
                           "development_pass": (
                               passes(item["selection_2022"])
                               and passes(value))})
    validation.sort(key=lambda item: min(
        rank(item["selection_2022"])[0],
        rank(item["validation_2023"])[0]), reverse=True)
    eligible = [item for item in validation if item["development_pass"]]
    report = {
        "status": "DEVELOPMENT_PASS" if eligible else "REJECTED_DEVELOPMENT",
        "ready_for_live_orders": False,
        "protocol": {
            "source_model": MODEL_CONFIG,
            "source_threshold_frozen_on_2022": BASE_THRESHOLD,
            "target_pct": 2.0, "stop_pct": 1.0, "hold_hours": 48,
            "health_observation": "all source signals including rejected shadow signals",
            "causality": "only outcomes with exit_time <= current entry_time",
            "gate_selection": 2022, "gate_validation": 2023,
            "hidden_not_revealed": [2024, 2025, 2026],
            "money": "3% current equity margin, fixed 10x gross",
        },
        "source_candidates_2021_2023": len(candidates),
        "tested_gate_combinations": len(selection),
        "selection_top20": selection[:20],
        "validation_top20": validation,
        "qualified_development": eligible,
        "decision": "No runtime change; hidden years remain unopened on failure.",
        "sources": {**sources, str(MANIFEST): digest(MANIFEST),
                    str(SOURCE_REPORT): digest(SOURCE_REPORT),
                    str(Path(__file__)): digest(Path(__file__))},
    }
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({
        "status": report["status"],
        "source_candidates": len(candidates),
        "tested": len(selection),
        "best_validation": validation[:5],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
