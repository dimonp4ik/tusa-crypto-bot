"""Causal trailing-quality calibration for the fixed trade-flow model."""
from __future__ import annotations

import collections
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

import crypto_5m_regime_router_literal_oct6_audit as price
import crypto_futures_trade_flow_regime_oct7_audit as flow


OUT = Path("reports/audit_2026_09_08/crypto_futures_trade_flow_rolling_calibration_oct7.json")
ROWS_OUT = Path("reports/audit_2026_09_08/crypto_futures_trade_flow_rolling_calibration_oct7.pkl")
MODEL = {"leaves": 31, "leaf": 400, "l2": 5.0}
LOOKBACKS = (14, 30, 60)
QUANTILES = (.90, .925, .95, .96, .97, .98)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def calibrate(frame, lookback: int, quantile: float):
    result = frame.copy()
    days = (result.entry_time.to_numpy(np.int64)//86400).astype(np.int64)
    unique_days = np.unique(days)
    by_day = {int(day): result.quality.to_numpy(float)[days == day]
              for day in unique_days}
    thresholds = {}
    window = collections.deque()
    for day in unique_days:
        while window and window[0][0] < day-lookback:
            window.popleft()
        values = np.concatenate([item[1] for item in window]) if window else np.empty(0)
        thresholds[int(day)] = (float(np.quantile(values, quantile))
                                if len(values) >= 500 else float("inf"))
        window.append((int(day), by_day[int(day)]))
    threshold_values = np.asarray([thresholds[int(day)] for day in days])
    accepted = result.quality.to_numpy(float) >= threshold_values
    result["rolling_threshold"] = threshold_values
    result["raw_quality"] = result.quality
    result["quality"] = np.where(accepted, 1+result.raw_quality, 0.0)
    return result


def evaluate(frame, year: int):
    return price.evaluate(frame, (year,), 1.0)


def selection_ok(value: dict) -> bool:
    return bool(value["n"] >= 700 and value["return_pct"] > 0
                and value["literal_target_hit_rate"] >= 68
                and value["dd_pct"] <= 8
                and value["trades_per_calendar_day"] >= 3)


def validation_ok(value: dict) -> bool:
    return bool(value["n"] >= 500 and value["return_pct"] > 0
                and value["literal_target_hit_rate"] >= 66
                and value["dd_pct"] <= 8
                and value["trades_per_calendar_day"] >= 2.5)


def objective_ok(value: dict) -> bool:
    return bool(value["n"] >= 500 and value["return_pct"] > 0
                and 75 <= value["literal_target_hit_rate"] <= 95
                and value["dd_pct"] <= 5 and value["negative_months"] == 0
                and value["trades_per_calendar_day"] >= 3
                and value["months_ge_30pct"] == value["months"])


def public(item: dict | None):
    if item is None:
        return None
    return {key: value for key, value in item.items() if key != "_frame"}


def main() -> None:
    candles, sources = flow.load()
    features = flow.attach_flow(price.build_features(candles), candles)
    frame = price.add_outcomes(flow.proposals(features), candles)
    fit = frame.year.isin((2020, 2021)).to_numpy()
    schema = price.columns(frame.loc[fit])
    _, matrix = flow.matrices(frame, schema)
    model = HistGradientBoostingClassifier(
        max_iter=180, learning_rate=.06, max_leaf_nodes=MODEL["leaves"],
        min_samples_leaf=MODEL["leaf"], l2_regularization=MODEL["l2"],
        random_state=20261007)
    model.fit(matrix[fit], frame.target_hit.astype(int).to_numpy()[fit])
    scored = frame.copy()
    scored["quality"] = model.predict_proba(matrix)[:, 1]
    candidates, frontier, validation_diagnostics = [], [], []
    for lookback in LOOKBACKS:
        for quantile in QUANTILES:
            calibrated = calibrate(scored, lookback, quantile)
            selection = evaluate(calibrated, 2022)
            item = {"lookback_days": lookback, "quantile": quantile,
                    "selection_2022": selection}
            frontier.append(item)
            if not selection_ok(selection):
                continue
            item["validation_2023"] = evaluate(calibrated, 2023)
            item["validation_2024"] = evaluate(calibrated, 2024)
            if not (validation_ok(item["validation_2023"])
                    and validation_ok(item["validation_2024"])):
                validation_diagnostics.append(public(item))
                continue
            item["score"] = round(sum(
                item[name]["return_pct"]-2*item[name]["dd_pct"]
                for name in ("selection_2022", "validation_2023",
                             "validation_2024")), 8)
            item["_frame"] = calibrated
            candidates.append(item)
    candidates.sort(key=lambda item: item["score"], reverse=True)
    frontier.sort(key=lambda item: (
        item["selection_2022"]["literal_target_hit_rate"],
        item["selection_2022"]["trades_per_calendar_day"]), reverse=True)
    frozen = candidates[0] if candidates else None
    results = checks = None
    if frozen:
        results = {str(year): evaluate(frozen["_frame"], year)
                   for year in (2022, 2023, 2024, 2025, 2026)}
        checks = {year: objective_ok(value) for year, value in results.items()}
        frozen["_frame"][frozen["_frame"].quality.ge(1)].to_pickle(
            ROWS_OUT, compression="gzip")
    report = {
        "status": ("RESEARCH_CANDIDATE_AWAITING_VENUE_FORWARD"
                   if checks and all(checks.values()) else "REJECTED"),
        "protocol": {"fit": "2020-2021", "selection": "2022",
                     "validation": ["2023", "2024"], "test": "2025",
                     "observed": "2026", "model": MODEL,
                     "lookback_days": LOOKBACKS, "quantiles": QUANTILES,
                     "calibration": "each UTC day uses only prior completed days",
                     "target_pct": .5, "stop_pct": 1.0, "max_bars": 24,
                     "money": "3% equity margin, fixed 10x gross"},
        "qualified_before_test": len(candidates),
        "frozen_before_test": public(frozen),
        "top_before_test": [public(item) for item in candidates[:12]],
        "selection_frontier": frontier,
        "validation_diagnostics": validation_diagnostics,
        "results": results, "checks": checks,
        "sources": {**sources, str(Path(flow.__file__)): digest(Path(flow.__file__)),
                    str(Path(__file__)): digest(Path(__file__))},
        "limitations": [
            "Rolling calibration adapts score distribution, not market edge.",
            "The bounded lookback/quantile grid still creates selection risk.",
            "Any historical pass requires untouched direct X-Perp evidence.",
        ],
    }
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"status": report["status"],
                      "qualified_before_test": len(candidates),
                      "frozen": public(frozen), "results": results,
                      "checks": checks}, indent=2), flush=True)


if __name__ == "__main__":
    main()
