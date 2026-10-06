"""Annual causal walk-forward models for literal-target trade-flow entries."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

import crypto_5m_regime_router_literal_oct6_audit as price
import crypto_futures_trade_flow_regime_oct7_audit as flow


OUT = Path("reports/audit_2026_09_08/crypto_futures_trade_flow_walkforward_oct7.json")
ROWS_OUT = Path("reports/audit_2026_09_08/crypto_futures_trade_flow_walkforward_oct7.pkl")
YEARS = (2022, 2023, 2024, 2025, 2026)
CONFIGS = flow.CONFIGS
QUANTILES = (.90, .925, .95, .96, .97)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def masks(frame, target_year: int):
    stamp = pd.to_datetime(frame.entry_time, unit="s", utc=True)
    split = pd.Timestamp(f"{target_year-1}-07-01", tz="UTC")
    calibration_end = pd.Timestamp(f"{target_year}-01-01", tz="UTC")
    train = (stamp < split).to_numpy()
    calibration = ((stamp >= split) & (stamp < calibration_end)).to_numpy()
    target = frame.year.eq(target_year).to_numpy()
    return train, calibration, target


def pretest_ok(value: dict) -> bool:
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


def evaluate(item: dict, year: int, quantile: float):
    threshold = float(np.quantile(item[year]["calibration_scores"], quantile))
    scored = item[year]["target_frame"].copy()
    scored["quality"] = item[year]["target_scores"]
    return price.evaluate(scored, (year,), threshold), threshold, scored


def public(item: dict | None):
    if item is None:
        return None
    return {key: value for key, value in item.items()
            if key not in ("_models", "_scored")}


def main() -> None:
    candles, sources = flow.load()
    features = flow.attach_flow(price.build_features(candles), candles)
    frame = price.add_outcomes(flow.proposals(features), candles)
    schema = price.columns(frame[frame.year.isin((2020, 2021))])
    _, matrix = flow.matrices(frame, schema)
    labels = frame.target_hit.astype(int).to_numpy()
    model_sets = []
    for config in CONFIGS:
        yearly = {}
        for year in YEARS:
            train, calibration, target = masks(frame, year)
            model = HistGradientBoostingClassifier(
                max_iter=180, learning_rate=.06,
                max_leaf_nodes=config["leaves"], min_samples_leaf=config["leaf"],
                l2_regularization=config["l2"], random_state=20261007+year)
            model.fit(matrix[train], labels[train])
            yearly[year] = {
                "calibration_scores": model.predict_proba(matrix[calibration])[:, 1],
                "target_scores": model.predict_proba(matrix[target])[:, 1],
                "target_frame": frame.loc[target].copy(),
                "train_rows": int(train.sum()),
                "calibration_rows": int(calibration.sum()),
                "target_rows": int(target.sum()),
            }
        model_sets.append((config, yearly))

    candidates, frontier = [], []
    for config, yearly in model_sets:
        for quantile in QUANTILES:
            pretest, thresholds = {}, {}
            scored = {}
            for year in (2022, 2023, 2024):
                value, threshold, target_frame = evaluate(yearly, year, quantile)
                pretest[str(year)] = value
                thresholds[str(year)] = threshold
                scored[year] = target_frame
            item = {"model": config, "quantile": quantile,
                    "thresholds": thresholds, "pretest": pretest}
            frontier.append(item)
            if not all(pretest_ok(value) for value in pretest.values()):
                continue
            item["score"] = round(sum(
                value["return_pct"]-2*value["dd_pct"]
                for value in pretest.values()), 8)
            item["_models"] = yearly
            item["_scored"] = scored
            candidates.append(item)
    candidates.sort(key=lambda item: item["score"], reverse=True)
    frontier.sort(key=lambda item: (
        min(value["literal_target_hit_rate"] for value in item["pretest"].values()),
        min(value["trades_per_calendar_day"] for value in item["pretest"].values()),
        item.get("score", -1e9)), reverse=True)
    frozen = candidates[0] if candidates else None
    results = checks = None
    if frozen:
        results, selected_frames = {}, []
        for year in YEARS:
            value, threshold, scored = evaluate(
                frozen["_models"], year, frozen["quantile"])
            results[str(year)] = value
            frozen["thresholds"][str(year)] = threshold
            selected_frames.append(scored[scored.quality.ge(threshold)])
        checks = {year: objective_ok(value) for year, value in results.items()}
        pd.concat(selected_frames, ignore_index=True).to_pickle(
            ROWS_OUT, compression="gzip")
    report = {
        "status": ("RESEARCH_CANDIDATE_AWAITING_VENUE_FORWARD"
                   if checks and all(checks.values()) else "REJECTED"),
        "protocol": {"algorithm_selection": "2022-2024",
                     "test": "2025", "observed": "2026",
                     "annual_fit": "all rows before July 1 of previous year",
                     "annual_calibration": "July-December of previous year, labels unused",
                     "annual_test": "following calendar year",
                     "target_pct": .5, "stop_pct": 1.0, "max_bars": 24,
                     "money": "3% equity margin, fixed 10x gross"},
        "rows": len(frame), "qualified_before_test": len(candidates),
        "frozen_before_test": public(frozen),
        "top_before_test": [public(item) for item in candidates[:10]],
        "pretest_frontier": [public(item) for item in frontier],
        "results": results, "checks": checks,
        "training_rows": {str(config): {str(year): {
            key: yearly[year][key] for key in ("train_rows", "calibration_rows",
                                               "target_rows")}
            for year in YEARS} for config, yearly in model_sets},
        "sources": {**sources, str(Path(flow.__file__)): digest(Path(flow.__file__)),
                    str(Path(__file__)): digest(Path(__file__))},
        "limitations": [
            "Annual retraining adapts slowly to regime changes within a year.",
            "The bounded model/quantile grid creates selection risk.",
            "Any historical pass still requires untouched direct X-Perp evidence.",
        ],
    }
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"status": report["status"],
                      "qualified_before_test": len(candidates),
                      "frozen": public(frozen), "results": results,
                      "checks": checks}, indent=2), flush=True)


if __name__ == "__main__":
    main()
