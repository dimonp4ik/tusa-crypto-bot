"""Causal walk-forward recovery audit for the 0.5R/2R crypto barbell."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from sklearn.ensemble import HistGradientBoostingClassifier

import crypto_05_2r_near_precision_barbell_oct5_audit as source
import crypto_high_target_tree_oct5_audit as base


OUT = source.ROOT / "crypto_05_2r_barbell_walkforward_oct13.json"
CONFIG = {"max_leaf_nodes": 15, "min_samples_leaf": 300,
          "learning_rate": .10}
TARGETS = ("2022h2", "2023", "2024", "2025", "2026")
QUANTILES = (.75, .80, .85, .875, .90, .925, .95, .97)
FRACTIONS = (.30, .50, .70)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def calibration_pass(value: dict) -> bool:
    all_ = value["all"]
    return bool(
        all_["n"] >= 200
        and all_["wr"] >= 65
        and all_["return_pct"] > 0
        and all_["dd_pct"] <= 7
        and all_["negative_months"] <= 1
        and min(all_["monthly_return_pct"].values()) >= -1
        and all_["trades_per_calendar_day"] >= 2.0
        and all_["max_loss_streak"] <= 8
    )


def forward_pass(value: dict) -> bool:
    all_ = value["all"]
    return bool(
        all_["n"] >= 150
        and all_["wr"] >= 65
        and all_["return_pct"] > 0
        and all_["dd_pct"] <= 7
        and all_["negative_months"] <= 1
        and min(all_["monthly_return_pct"].values()) >= -1
        and all_["trades_per_calendar_day"] >= 2.0
        and all_["max_loss_streak"] <= 8
    )


def train_and_score(frame, train_periods):
    fit = base.stage(frame, tuple(train_periods))
    schema = base.design(fit)
    model = HistGradientBoostingClassifier(
        learning_rate=CONFIG["learning_rate"], max_iter=150,
        max_leaf_nodes=CONFIG["max_leaf_nodes"],
        min_samples_leaf=CONFIG["min_samples_leaf"],
        l2_regularization=1.0, early_stopping=False,
        random_state=20261013)
    model.fit(base.matrix(fit, schema),
              fit.near_outcome.eq("TP2").astype(int).to_numpy())
    return source.scored(frame, model, schema)


def select_policy(scored, calibration_period):
    probabilities = base.stage(
        scored, (calibration_period,)).model_probability
    frontier = []
    eligible = []
    for quantile in QUANTILES:
        threshold = float(probabilities.quantile(quantile))
        for fraction in FRACTIONS:
            score = source.metrics(
                scored, (calibration_period,), threshold, fraction)
            passed = calibration_pass(score)
            item = {"quantile": quantile, "threshold": round(threshold, 10),
                    "near_fraction": fraction, "metrics": score,
                    "calibration_pass": passed}
            frontier.append(item)
            if passed:
                rank = (min(score["all"]["monthly_return_pct"].values()),
                        score["all"]["return_pct"], score["all"]["wr"],
                        -score["all"]["dd_pct"])
                eligible.append((rank, item))
    frontier.sort(key=lambda item: (
        item["calibration_pass"],
        min(item["metrics"]["all"]["monthly_return_pct"].values()),
        item["metrics"]["all"]["return_pct"]), reverse=True)
    return (max(eligible, key=lambda value: value[0])[1]
            if eligible else None), frontier[:10]


def main() -> None:
    frame = source.load_joined()
    periods = list(base.PERIODS)
    steps = []
    for target in TARGETS:
        target_index = periods.index(target)
        calibration = periods[target_index - 1]
        train_periods = periods[:target_index - 1]
        if len(train_periods) < 4:
            raise ValueError(f"insufficient walk-forward history for {target}")
        scored = train_and_score(frame, train_periods)
        policy, frontier = select_policy(scored, calibration)
        target_metrics = None
        passed = False
        if policy:
            target_metrics = source.metrics(
                scored, (target,), float(policy["threshold"]),
                float(policy["near_fraction"]))
            passed = forward_pass(target_metrics)
        steps.append({
            "target_period": target, "train_periods": train_periods,
            "calibration_period": calibration, "selected_policy": policy,
            "calibration_frontier_top10": frontier,
            "target_metrics": target_metrics, "target_pass": passed,
        })

    first_failure = next((step["target_period"] for step in steps
                          if not step["target_pass"]), None)
    status = ("BARBELL_WALKFORWARD_RESEARCH_CANDIDATE"
              if first_failure is None else f"REJECTED_{first_failure.upper()}")
    report = {
        "status": status, "ready_for_live_orders": False,
        "protocol": {
            "role": "adaptive supplement to the hourly router",
            "frozen_model_config": CONFIG,
            "walk_forward": "fit through period before calibration; calibrate on immediately prior period; never use target labels",
            "target_periods": list(TARGETS),
            "threshold_quantiles": list(QUANTILES),
            "near_fractions": list(FRACTIONS),
            "entry": "next 15-minute market open",
            "exit": "0.5R partial, breakeven remainder, 2R runner",
            "price_moves": "near 0.5%-1.0%; runner 2%-4%",
            "money": "client-selected current-equity margin; evaluated at 3% and fixed 10x gross",
        },
        "first_failure": first_failure, "steps": steps,
        "sources": {
            str(base.cache_path(target, period)):
                digest(base.cache_path(target, period))
            for target in (.5, 2.0) for period in base.PERIODS
        } | {str(Path(source.__file__)): digest(Path(source.__file__)),
             str(Path(__file__)): digest(Path(__file__))},
        "limitations": [
            "Each named annual period contains the available six-month research window.",
            "Binance spot history is structural evidence for OKX X-Perp transfer.",
            "A passing adaptive backtest would still need fresh shadow validation.",
            "Fees are excluded by user instruction.",
        ],
    }
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"status": status, "first_failure": first_failure,
                      "steps": [{"target": step["target_period"],
                                 "calibration": step["calibration_period"],
                                 "policy": step["selected_policy"],
                                 "target_metrics": step["target_metrics"],
                                 "pass": step["target_pass"]}
                                for step in steps]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
