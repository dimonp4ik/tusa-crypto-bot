"""Re-evaluate the 0.5R/2R barbell as a profitable supplement.

The original audit required 75% win rate from every standalone module.  This
audit follows the user's later instruction that a lower-win-rate filter may be
kept when the combined system remains visibly profitable.  Selection still
ends in 2022H1; all later periods are opened sequentially.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

import crypto_05_2r_near_precision_barbell_oct5_audit as source
import crypto_high_target_tree_oct5_audit as base


OUT = source.ROOT / "crypto_05_2r_barbell_recovery_oct13.json"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def selection_pass(value: dict) -> bool:
    all_ = value["all"]
    return bool(
        all_["n"] >= 300
        and all_["wr"] >= 68.0
        and all_["return_pct"] > 0
        and all_["dd_pct"] <= 5.5
        and all_["negative_months"] == 0
        and min(all_["monthly_return_pct"].values()) >= 0
        and all_["trades_per_calendar_day"] >= 2.5
        and all_["max_loss_streak"] <= 7
    )


def forward_pass(value: dict, *, minimum_daily: float,
                 maximum_negative_months: int) -> bool:
    all_ = value["all"]
    periods = list(value["periods"].values())
    return bool(
        all_["n"] >= 300
        and all_["wr"] >= 65.0
        and all_["return_pct"] > 0
        and all_["dd_pct"] <= 7.0
        and all_["negative_months"] <= maximum_negative_months
        and min(all_["monthly_return_pct"].values()) >= -1.0
        and all_["trades_per_calendar_day"] >= minimum_daily
        and all_["max_loss_streak"] <= 8
        and all(item["return_pct"] > 0 for item in periods)
    )


def objective(value: dict) -> dict:
    all_ = value["all"]
    months = all_["monthly_return_pct"]
    return {
        "win_rate_75_to_95": 75 <= all_["wr"] <= 95,
        "at_least_3_calendar_trades_per_day":
            all_["trades_per_calendar_day"] >= 3,
        "no_negative_months": all_["negative_months"] == 0,
        "every_month_at_least_30pct":
            bool(months) and min(months.values()) >= 30,
    }


def main() -> None:
    frame = source.load_joined()
    fit = base.stage(frame, source.FIT)
    schema = base.design(fit)
    x_fit = base.matrix(fit, schema)
    y_fit = fit.near_outcome.eq("TP2").astype(int).to_numpy()
    frontier = []
    private = []
    for max_leaf_nodes, min_samples_leaf, learning_rate in source.CONFIGS:
        model = HistGradientBoostingClassifier(
            learning_rate=learning_rate, max_iter=150,
            max_leaf_nodes=max_leaf_nodes,
            min_samples_leaf=min_samples_leaf,
            l2_regularization=1.0, early_stopping=False,
            random_state=20260908)
        model.fit(x_fit, y_fit)
        scored = source.scored(frame, model, schema)
        probabilities = base.stage(scored, source.SELECT).model_probability
        thresholds = sorted(set(float(probabilities.quantile(q))
                                for q in source.QUANTILES), reverse=True)
        for fraction in source.FRACTIONS:
            for threshold in thresholds:
                selection = source.metrics(
                    scored, source.SELECT, threshold, fraction)
                passed = selection_pass(selection)
                public = {
                    "max_leaf_nodes": max_leaf_nodes,
                    "min_samples_leaf": min_samples_leaf,
                    "learning_rate": learning_rate,
                    "near_fraction": fraction,
                    "threshold": round(threshold, 10),
                    "selection_2022h1": selection,
                    "selection_pass": passed,
                }
                frontier.append(public)
                if passed:
                    rank = (selection["all"]["negative_months"] == 0,
                            min(selection["all"]["monthly_return_pct"].values()),
                            selection["all"]["return_pct"],
                            selection["all"]["wr"],
                            -selection["all"]["dd_pct"])
                    private.append((rank, public, scored))

    frontier.sort(key=lambda item: (
        item["selection_pass"],
        min(item["selection_2022h1"]["all"]["monthly_return_pct"].values()),
        item["selection_2022h1"]["all"]["return_pct"],
        item["selection_2022h1"]["all"]["wr"]), reverse=True)
    frozen = validation = test = confirmation = None
    status = "REJECTED_SELECTION"
    if private:
        _, frozen_public, scored = max(private, key=lambda item: item[0])
        frozen = frozen_public
        threshold = float(frozen["threshold"])
        fraction = float(frozen["near_fraction"])
        validation = source.metrics(
            scored, source.CONFIRM, threshold, fraction)
        status = "REJECTED_VALIDATION"
        if forward_pass(validation, minimum_daily=2.5,
                        maximum_negative_months=2):
            test = source.metrics(
                scored, source.VALIDATION, threshold, fraction)
            status = "REJECTED_TEST"
            if forward_pass(test, minimum_daily=2.5,
                            maximum_negative_months=1):
                confirmation = source.metrics(
                    scored, source.HIDDEN, threshold, fraction)
                status = ("BARBELL_SUPPLEMENT_RESEARCH_CANDIDATE"
                          if forward_pass(confirmation, minimum_daily=2.5,
                                          maximum_negative_months=2)
                          else "REJECTED_CONFIRMATION")

    report = {
        "status": status, "ready_for_live_orders": False,
        "protocol": {
            "role": "supplement to the higher-win-rate hourly router",
            "fit": list(source.FIT), "selection": list(source.SELECT),
            "validation": list(source.CONFIRM), "test": list(source.VALIDATION),
            "confirmation_only": list(source.HIDDEN),
            "entry": "next 15-minute market open",
            "exit": "take a fixed fraction at 0.5R, move remainder to breakeven, target 2R",
            "price_moves": "near 0.5%-1.0%; runner 2%-4%",
            "money": "client-selected current-equity margin; evaluated at 3% and fixed 10x gross",
            "selection_reason": "profit, no negative month, drawdown, frequency and >=68% supplement win rate",
            "candidate_count": len(frontier),
        },
        "frozen_before_validation": frozen,
        "validation_2022h2_2023": validation,
        "test_2024": test, "confirmation_2025_2026": confirmation,
        "objective_checks": {
            "selection": objective(frozen["selection_2022h1"]) if frozen else None,
            "validation": objective(validation) if validation else None,
            "test": objective(test) if test else None,
            "confirmation": objective(confirmation) if confirmation else None,
        },
        "selection_frontier_top20": frontier[:20],
        "sources": {
            str(base.cache_path(target, period)):
                digest(base.cache_path(target, period))
            for target in (.5, 2.0) for period in base.PERIODS
        } | {str(Path(source.__file__)): digest(Path(source.__file__)),
             str(Path(__file__)): digest(Path(__file__))},
        "limitations": [
            "The inherited raw-router caches contain roughly six months per named period, not complete calendar years.",
            "Binance spot history is structural evidence for OKX X-Perp transfer.",
            "The relaxed win-rate gate is only valid for a supplemental route; portfolio win rate must still be tested.",
            "Fees are excluded by user instruction.",
        ],
    }
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"status": status, "eligible": len(private),
                      "frozen": frozen, "validation": validation,
                      "test": test, "confirmation": confirmation,
                      "objective_checks": report["objective_checks"]},
                     indent=2), flush=True)


if __name__ == "__main__":
    main()
