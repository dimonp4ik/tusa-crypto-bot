"""Causal exact-path search for profitable crypto exit geometry.

Every target and breakeven activation is replayed in OHLC order.  Models fit
on 2020-2021, thresholds and account rails use 2022-2023, and 2024+, direct
X-Perp history remain holdouts.  Win rate is diagnostic only.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

import crypto_2r_precision_barbell_oct5_audit as precision
import crypto_high_target_tree_oct5_audit as base
import crypto_precision_be_runner_oct18_audit as exact
import crypto_precision_profit_first_oct18_audit as prior
from crypto_runtime_shadow_portfolio_oct3_audit import money


OUT = exact.ROOT / "crypto_exact_exit_geometry_oct20.json"
ROWS_OUT = exact.ROOT / "crypto_exact_exit_geometry_xperp_oct20.pkl"
TARGETS_R = (1.0, 1.5, 2.0)
ACTIVATIONS_R = (.25, .5, .75, 99.0)
QUANTILES = (.50, .60, .70, .80, .85, .90, .925, .95)
LOSS_PAUSES = (3, 5)
HOURLY_CAPS = (1, 2)
MIN_TARGET_MOVE = .005
MAX_TARGET_MOVE = .02


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def eligible(frame: pd.DataFrame, target_r: float) -> pd.DataFrame:
    move = target_r * frame.risk_pct.astype(float)
    return frame[move.between(MIN_TARGET_MOVE, MAX_TARGET_MOVE)].copy()


def fit_model(train: pd.DataFrame, design: dict) -> HistGradientBoostingClassifier:
    model = HistGradientBoostingClassifier(
        learning_rate=.05, max_iter=150, max_leaf_nodes=15,
        min_samples_leaf=300, l2_regularization=1.0,
        early_stopping=False, random_state=20260908)
    model.fit(base.matrix(train, design),
              train.runner_outcome.eq("TP2").astype(int))
    return model


def score(frame: pd.DataFrame, model, design: dict) -> pd.DataFrame:
    result = frame.copy()
    result["model_probability"] = model.predict_proba(
        base.matrix(result, design))[:, 1]
    return result


def evaluate(scored: pd.DataFrame, periods: tuple[str, ...], threshold: float,
             loss_pause: int, hourly_cap: int, margin: float) -> tuple[dict, list[dict]]:
    chosen = base.stage(scored, periods)
    chosen = chosen[chosen.model_probability >= threshold].copy()
    chosen = chosen.sort_values(
        ["symbol", "entry_time", "model_probability", "family", "direction"],
        ascending=[True, True, False, False, False], kind="mergesort")
    chosen = chosen.drop_duplicates(["symbol", "entry_time"], keep="first")
    rows = exact.ranked_runtime_gate(
        exact.rows_for(chosen, "exact_exit_geometry"),
        loss_pause=loss_pause, hourly_cap=hourly_cap)
    value = money(rows, margin)
    value.update({
        "target_rate": round(100 * sum(
            row["runner_outcome"] == "TP2" for row in rows) / len(rows), 2)
        if rows else 0.0,
        "breakeven_rate": round(100 * sum(
            row["runner_outcome"] == "BE" for row in rows) / len(rows), 2)
        if rows else 0.0,
        "non_loss_rate": round(100 * sum(
            float(row["net_r"]) >= 0 for row in rows) / len(rows), 2)
        if rows else 0.0,
        "actual_max_loss_streak": exact.maximum_stop_streak(rows),
    })
    return value, rows


def month_floor(value: dict, months: tuple[str, ...]) -> float:
    observed = value.get("monthly_return_pct", {})
    return (min(float(observed[month]) for month in months)
            if set(months).issubset(observed) else -999.0)


def development_admissible(value: dict) -> bool:
    return bool(value["n"] >= 100 and value["dd_pct"] <= 10
                and value["trades_per_calendar_day"] >= 3)


def public(candidate: dict) -> dict:
    return {key: value for key, value in candidate.items()
            if not key.startswith("_")}


def main() -> None:
    historical = prior.canonical_history(precision.load_joined())
    direct = prior.direct_frame().assign(period="xperp")
    candidates = []
    geometry_reports = []
    for target_r in TARGETS_R:
        historical_target = eligible(historical, target_r)
        direct_target = eligible(direct, target_r)
        for activation_r in ACTIVATIONS_R:
            replayed = exact.attach_runtime_runner(
                historical_target, direct=False,
                activation_r=activation_r, runner_r=target_r)
            fit = base.stage(replayed, precision.FIT)
            design = base.design(fit)
            model = fit_model(fit, design)
            scored = score(replayed, model, design)
            selection_scores = base.stage(
                scored, precision.SELECT).model_probability
            count_before = len(candidates)
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
                            "target_r": target_r,
                            "activation_r": None if activation_r == 99 else activation_r,
                            "quantile": quantile,
                            "threshold": threshold,
                            "loss_pause": loss_pause,
                            "hourly_cap": hourly_cap,
                            "selection": selection,
                            "confirmation": confirmation,
                            "development_floor_pct": round(floor, 4),
                            "admissible": (development_admissible(selection)
                                           and development_admissible(confirmation)),
                            "_model": model, "_design": design,
                            "_direct": direct_target,
                            "_activation_value": activation_r,
                        })
            geometry_reports.append({
                "target_r": target_r,
                "activation_r": None if activation_r == 99 else activation_r,
                "historical_rows": len(historical_target),
                "direct_rows": len(direct_target),
                "candidate_count": len(candidates) - count_before,
            })
            print(json.dumps(geometry_reports[-1]), flush=True)

    candidates.sort(key=lambda item: (
        item["admissible"], item["development_floor_pct"],
        item["selection"]["return_pct"] + item["confirmation"]["return_pct"],
        -max(item["selection"]["dd_pct"], item["confirmation"]["dd_pct"]),
    ), reverse=True)
    selected = candidates[0]
    activation = float(selected["_activation_value"])
    target = float(selected["target_r"])
    threshold = float(selected["threshold"])
    loss_pause = int(selected["loss_pause"])
    hourly_cap = int(selected["hourly_cap"])

    # Recreate only the chosen geometry for holdouts and direct evidence.
    historical_selected = eligible(historical, target)
    replayed = exact.attach_runtime_runner(
        historical_selected, direct=False,
        activation_r=activation, runner_r=target)
    scored = score(replayed, selected["_model"], selected["_design"])
    holdouts = {}
    for margin in (.03, .05):
        validation, _ = evaluate(scored, precision.VALIDATION, threshold,
                                 loss_pause, hourly_cap, margin)
        later, _ = evaluate(scored, precision.HIDDEN, threshold,
                            loss_pause, hourly_cap, margin)
        holdouts[f"margin_{round(100 * margin)}pct"] = {
            "validation_2024": validation,
            "later_2025_2026": later,
        }

    direct_replayed = exact.attach_runtime_runner(
        selected["_direct"], direct=True,
        activation_r=activation, runner_r=target)
    direct_scored = score(
        direct_replayed, selected["_model"], selected["_design"])
    direct_values = {}
    direct_rows = []
    for margin in (.03, .05):
        value, rows = evaluate(direct_scored, ("xperp",), threshold,
                               loss_pause, hourly_cap, margin)
        direct_values[f"margin_{round(100 * margin)}pct"] = value
        if margin == .03:
            direct_rows = rows
    pd.DataFrame(direct_rows).to_pickle(ROWS_OUT)

    direct_months = direct_values["margin_5pct"]["monthly_return_pct"]
    complete_direct = {month: direct_months[month] for month in
                       ("2026-05", "2026-06", "2026-07", "2026-08", "2026-09")
                       if month in direct_months}
    objective = {
        "development_admissible": bool(selected["admissible"]),
        "development_no_negative_complete_months": (
            selected["selection"]["negative_months"] == 0
            and selected["confirmation"]["negative_months"] == 0),
        "holdouts_no_negative_months_5pct": (
            holdouts["margin_5pct"]["validation_2024"]["negative_months"] == 0
            and holdouts["margin_5pct"]["later_2025_2026"]["negative_months"] == 0),
        "direct_complete_months_at_least_30pct_5pct": bool(
            len(complete_direct) == 5
            and min(complete_direct.values()) >= 30),
        "direct_drawdown_at_most_10pct_5pct": (
            direct_values["margin_5pct"]["dd_pct"] <= 10),
    }
    passed = all(objective.values())
    report = {
        "status": ("EXACT_EXIT_GEOMETRY_SHADOW_CANDIDATE"
                   if passed else "REJECTED_OBJECTIVE"),
        "ready_for_live_orders": False,
        "protocol": {
            "fit": list(precision.FIT),
            "selection": list(precision.SELECT),
            "confirmation": list(precision.CONFIRM),
            "holdouts": [*precision.VALIDATION, *precision.HIDDEN, "direct X-Perp"],
            "target_r_candidates": list(TARGETS_R),
            "breakeven_activation_r_candidates": [
                None if value == 99 else value for value in ACTIVATIONS_R],
            "literal_target_move_pct": [100 * MIN_TARGET_MOVE,
                                         100 * MAX_TARGET_MOVE],
            "entry": "next 15-minute market open",
            "exit": "exact ordered OHLC replay; stop-first ambiguous bars",
            "money": "client 3% or 5% current-equity margin; fixed 10x gross",
            "priority": "worst development month, return, drawdown; win rate diagnostic only",
        },
        "geometry_reports": geometry_reports,
        "candidate_count": len(candidates),
        "admissible_count": sum(item["admissible"] for item in candidates),
        "selected_before_holdouts": public(selected),
        "top20_before_holdouts": [public(item) for item in candidates[:20]],
        "holdouts": holdouts,
        "direct_xperp": direct_values,
        "complete_direct_months_5pct": complete_direct,
        "objective_checks": objective,
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
        "status": report["status"],
        "selected": public(selected),
        "holdouts": holdouts,
        "direct_xperp": direct_values,
        "objective_checks": objective,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
