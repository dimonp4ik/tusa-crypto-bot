"""Fit crypto quality models on the exact breakeven-runner outcome.

The previous profit-first classifier learned a fixed-stop 2R label even though
the shadow runtime moves the whole stop to entry after +0.25R.  This audit
replays that contract first, then fits only on the resulting -1R/BE/2R value.
Policy selection ends in 2023; 2024+, direct X-Perp and October stay holdouts.
"""
from __future__ import annotations

import hashlib
import json
import collections
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import (HistGradientBoostingClassifier,
                              HistGradientBoostingRegressor)

import crypto_2r_precision_barbell_oct5_audit as precision
import crypto_high_target_tree_oct5_audit as base
import crypto_precision_be_runner_oct18_audit as exact
import crypto_precision_profit_first_oct18_audit as prior
from crypto_runtime_shadow_portfolio_oct3_audit import money


OUT = exact.ROOT / "crypto_exact_be_expected_value_oct19.json"
ROWS_OUT = exact.ROOT / "crypto_exact_be_expected_value_xperp_oct19.pkl"
QUANTILES = (.50, .60, .70, .75, .80, .85, .875, .90, .925, .95)
LOSS_PAUSES = (3, 4, 5)
HOURLY_CAPS = (1, 2)
CONFIGS = tuple(
    (leaves, leaf, l2)
    for leaves in (7, 15)
    for leaf in (100, 300)
    for l2 in (1.0, 5.0)
)
TARGETS = ("exact_r", "tp2", "non_loss")
MIN_TARGET_MOVE = .005
MAX_TARGET_MOVE = .02


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fit_model(train: pd.DataFrame, design: dict, target_name: str,
              config: tuple[int, int, float]):
    leaves, leaf, l2 = config
    common = dict(
        learning_rate=.05, max_iter=150, max_leaf_nodes=leaves,
        min_samples_leaf=leaf, l2_regularization=l2,
        early_stopping=False, random_state=20260908)
    if target_name == "exact_r":
        model = HistGradientBoostingRegressor(**common)
        target = train.net_r.to_numpy(float)
    else:
        model = HistGradientBoostingClassifier(**common)
        target = ((train.runner_outcome == "TP2").astype(int).to_numpy()
                  if target_name == "tp2"
                  else (train.net_r >= 0).astype(int).to_numpy())
    model.fit(base.matrix(train, design), target)
    return model


def score(frame: pd.DataFrame, model, design: dict,
          target_name: str) -> pd.DataFrame:
    result = frame.copy()
    values = base.matrix(result, design)
    result["model_probability"] = (
        model.predict(values) if target_name == "exact_r"
        else model.predict_proba(values)[:, 1])
    return result


def eligible(frame: pd.DataFrame) -> pd.DataFrame:
    move = 2.0 * frame.risk_pct.astype(float)
    return frame[move.between(MIN_TARGET_MOVE, MAX_TARGET_MOVE)].copy()


def evaluate(frame: pd.DataFrame, periods: tuple[str, ...], threshold: float,
             loss_pause: int, hourly_cap: int, margin: float) -> tuple[dict, list[dict]]:
    chosen = base.stage(frame, periods)
    chosen = eligible(chosen[chosen.model_probability >= threshold])
    chosen = chosen.sort_values(
        ["symbol", "entry_time", "model_probability", "family", "direction"],
        ascending=[True, True, False, False, False], kind="mergesort")
    chosen = chosen.drop_duplicates(["symbol", "entry_time"], keep="first")
    rows = exact.ranked_runtime_gate(
        exact.rows_for(chosen, "exact_be_expected_value"),
        loss_pause=loss_pause, hourly_cap=hourly_cap)
    value = money(rows, margin)
    actual_stop_streak = exact.maximum_stop_streak(rows)
    value.update({
        "tp2_rate": round(100 * sum(
            row["runner_outcome"] == "TP2" for row in rows) / len(rows), 2)
        if rows else 0.0,
        "breakeven_rate": round(100 * sum(
            row["runner_outcome"] == "BE" for row in rows) / len(rows), 2)
        if rows else 0.0,
        "non_loss_rate": round(100 * sum(
            float(row["net_r"]) >= 0 for row in rows) / len(rows), 2)
        if rows else 0.0,
        "actual_max_stop_streak": actual_stop_streak,
        "max_entries_same_timestamp": max(collections.Counter(
            int(float(row["entry_time"])) for row in rows).values())
        if rows else 0,
    })
    return value, rows


def month_floor(value: dict, months: tuple[str, ...]) -> float:
    observed = value.get("monthly_return_pct", {})
    return (min(float(observed[month]) for month in months)
            if set(months).issubset(observed) else -999.0)


def admissible(value: dict) -> bool:
    return bool(value["n"] >= 100 and value["dd_pct"] <= 10
                and value["trades_per_calendar_day"] >= 3
                and value["negative_months"] == 0)


def main() -> None:
    raw = precision.load_joined()
    historical = prior.canonical_history(raw)
    replayed = exact.attach_runtime_runner(historical, direct=False)
    fit = base.stage(replayed, precision.FIT)
    design = base.design(fit)
    candidates = []
    for target_name in TARGETS:
        for config in CONFIGS:
            model = fit_model(fit, design, target_name, config)
            scored = score(replayed, model, design, target_name)
            selection_scores = base.stage(
                eligible(scored), precision.SELECT).model_probability
            for quantile in QUANTILES:
                threshold = float(selection_scores.quantile(quantile))
                for loss_pause in LOSS_PAUSES:
                    for hourly_cap in HOURLY_CAPS:
                        selection, _ = evaluate(
                            scored, precision.SELECT, threshold,
                            loss_pause, hourly_cap, .03)
                        confirmation, _ = evaluate(
                            scored, precision.CONFIRM, threshold,
                            loss_pause, hourly_cap, .03)
                        floor = min(
                            month_floor(selection, exact.SELECTION_MONTHS),
                            month_floor(confirmation, exact.CONFIRMATION_MONTHS))
                        candidates.append({
                            "target": target_name,
                            "max_leaf_nodes": config[0],
                            "min_samples_leaf": config[1],
                            "l2_regularization": config[2],
                            "quantile": quantile,
                            "threshold": threshold,
                            "loss_pause": loss_pause,
                            "hourly_cap": hourly_cap,
                            "selection": selection,
                            "confirmation": confirmation,
                            "development_floor_pct": round(floor, 4),
                            "admissible": (admissible(selection)
                                           and admissible(confirmation)),
                            "_model": model,
                            "_scored": scored,
                        })
    candidates.sort(key=lambda item: (
        item["admissible"], item["development_floor_pct"],
        -max(item["selection"]["dd_pct"], item["confirmation"]["dd_pct"]),
        item["selection"]["return_pct"] + item["confirmation"]["return_pct"],
    ), reverse=True)
    selected = candidates[0]
    model = selected["_model"]
    scored = selected["_scored"]
    threshold = float(selected["threshold"])
    loss_pause = int(selected["loss_pause"])
    hourly_cap = int(selected["hourly_cap"])

    holdouts = {}
    holdout_rows = {}
    for margin in (.03, .05):
        name = f"margin_{round(100 * margin)}pct"
        validation, _ = evaluate(
            scored, precision.VALIDATION, threshold,
            loss_pause, hourly_cap, margin)
        later, _ = evaluate(
            scored, precision.HIDDEN, threshold,
            loss_pause, hourly_cap, margin)
        holdouts[name] = {"validation_2024": validation,
                          "later_2025_2026": later}

    direct = prior.direct_frame().copy()
    direct["period"] = "xperp"
    direct_replayed = exact.attach_runtime_runner(direct, direct=True)
    direct_scored = score(
        direct_replayed, model, design, str(selected["target"]))
    direct_values = {}
    direct_rows = []
    for margin in (.03, .05):
        value, rows = evaluate(
            direct_scored, ("xperp",), threshold,
            loss_pause, hourly_cap, margin)
        direct_values[f"margin_{round(100 * margin)}pct"] = value
        if margin == .03:
            direct_rows = rows
    pd.DataFrame(direct_rows).to_pickle(ROWS_OUT)

    public = {key: value for key, value in selected.items()
              if not key.startswith("_")}
    complete_direct = direct_values["margin_5pct"]["monthly_return_pct"]
    status = "REJECTED_OBJECTIVE"
    if (selected["admissible"] and selected["development_floor_pct"] >= 30
            and holdouts["margin_5pct"]["validation_2024"]["negative_months"] == 0
            and holdouts["margin_5pct"]["later_2025_2026"]["negative_months"] == 0
            and complete_direct and min(complete_direct.values()) >= 30):
        status = "EXACT_BE_EXPECTED_VALUE_SHADOW_CANDIDATE"
    report = {
        "status": status,
        "ready_for_live_orders": False,
        "protocol": {
            "fit": list(precision.FIT),
            "selection": list(precision.SELECT),
            "confirmation": list(precision.CONFIRM),
            "holdouts": [*precision.VALIDATION, *precision.HIDDEN, "direct X-Perp"],
            "entry": "next 15-minute market open",
            "exit": "2R target; full stop to entry after +0.25R",
            "target_price_move_range_pct": [100 * MIN_TARGET_MOVE,
                                            100 * MAX_TARGET_MOVE],
            "money": "client 3% or 5% current-equity margin; fixed 10x",
            "priority": "worst complete-month return; win rate diagnostic only",
        },
        "candidate_count": len(candidates),
        "admissible_count": sum(item["admissible"] for item in candidates),
        "selected_before_holdouts": public,
        "top20_before_holdouts": [
            {key: value for key, value in item.items() if not key.startswith("_")}
            for item in candidates[:20]],
        "holdouts": holdouts,
        "direct_xperp": direct_values,
        "sources": {
            **exact.CANDLE_SOURCES,
            str(prior.FRESH_NEAR): digest(prior.FRESH_NEAR),
            str(prior.FRESH_FAR): digest(prior.FRESH_FAR),
            str(Path(__file__).resolve()): digest(Path(__file__).resolve()),
        },
        "limitations": [
            "Gross results exclude fees at user request.",
            "Historical named periods are available six-month windows.",
            "Direct X-Perp history is shorter than a full market cycle.",
            "No backtest guarantees a future monthly return.",
        ],
    }
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({
        "status": status,
        "candidate_count": len(candidates),
        "admissible_count": report["admissible_count"],
        "selected": public,
        "holdouts": holdouts,
        "direct_xperp": direct_values,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
