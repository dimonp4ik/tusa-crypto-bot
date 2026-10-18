"""Chronological profit-first audit for a fixed 1R market-exit strategy.

The classifier is fitted on 2020-2021 only.  Threshold and portfolio rails are
selected on 2022-2023, then frozen before 2024+ and direct X-Perp evaluation.
Every outcome comes from one causal OHLC replay; no independent target labels
are combined.
"""
from __future__ import annotations

import collections
import json
import pickle
from pathlib import Path

import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

import crypto_high_target_tree_oct5_audit as base
import crypto_precision_be_runner_oct18_audit as be
import crypto_precision_profit_first_oct18_audit as prior
from crypto_runtime_shadow_portfolio_oct3_audit import money


OUT = be.ROOT / "crypto_profit_first_exact_1r_oct18.json"
TARGET_R = 1.0
QUANTILES = (.05, .10, .20, .30, .40, .50, .60, .70, .80, .85, .90, .925, .95)
LOSS_PAUSES = (3, 4, 5, 6, 7, 8)
HOURLY_CAPS = (1, 2, 3, 4)
FIT = ("2020h1", "2020h2", "2021h1", "2021h2")
SELECT = ("2022h1",)
CONFIRM = ("2022h2", "2023")
VALIDATE = ("2024",)
LATER = ("2025", "2026")
SELECT_MONTHS = ("2022-02", "2022-03", "2022-04", "2022-05", "2022-06")
CONFIRM_MONTHS = (
    "2022-08", "2022-09", "2022-10", "2022-11", "2022-12",
    "2023-02", "2023-03", "2023-04", "2023-05", "2023-06")


def direct_fixed_frame(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy().reset_index(drop=True)
    values = {}
    for symbol, group in result.groupby("symbol", sort=False):
        path = be.DIRECT_CACHE / f"{symbol}_15min_current.pkl"
        candles = pickle.loads(path.read_bytes())
        lookup = {int(ts): i for i, ts in enumerate(candles["time"])}
        for row in group.itertuples():
            i = lookup[int(row.entry_time)]
            sign = 1.0 if row.direction == "LONG" else -1.0
            entry = float(row.entry)
            sl = float(row.sl)
            risk = abs(entry - sl)
            target = entry + sign * risk * TARGET_R
            outcome = "EXPIRED"
            net_r = 0.0
            exit_time = float(candles["time"][i + 47]) + 900
            for bar in range(i, i + 48):
                opened = float(candles["open"][bar])
                high = float(candles["high"][bar])
                low = float(candles["low"][bar])
                stop_hit = low <= sl if sign > 0 else high >= sl
                if stop_hit:
                    fill = min(opened, sl) if sign > 0 else max(opened, sl)
                    outcome = "SL"
                    net_r = sign * (fill - entry) / risk
                    exit_time = float(candles["time"][bar]) + 900
                    break
                target_hit = high >= target if sign > 0 else low <= target
                if target_hit:
                    outcome = "TP2"
                    net_r = TARGET_R
                    exit_time = float(candles["time"][bar]) + 900
                    break
            else:
                close = float(candles["close"][i + 47])
                net_r = sign * (close - entry) / risk
            values[int(row.Index)] = (outcome, net_r, exit_time)
    ordered = [values[int(i)] for i in result.index]
    result["fixed_outcome"] = [value[0] for value in ordered]
    result["net_r"] = [value[1] for value in ordered]
    result["exit_time"] = [value[2] for value in ordered]
    return result


def rows_for(frame: pd.DataFrame) -> list[dict]:
    rows = base.standardized(frame, "exact_fixed_1r")
    for row in rows:
        raw = str(row.get("fixed_outcome") or row.get("outcome"))
        row["fixed_outcome"] = raw
        row["outcome"] = "SL" if raw == "SL" else "TP"
    return rows


def max_stop_streak(rows: list[dict]) -> int:
    current = maximum = 0
    for row in sorted(rows, key=lambda value: (
            float(value["exit_time"]), float(value["entry_time"]))):
        current = current + 1 if row["fixed_outcome"] == "SL" else 0
        maximum = max(maximum, current)
    return maximum


def evaluate(scored: pd.DataFrame, periods: tuple[str, ...], threshold: float,
             loss_pause: int, hourly_cap: int, margin: float = .03) -> dict:
    chosen = base.stage(scored, periods)
    chosen = chosen[chosen.model_probability >= threshold].copy()
    chosen = chosen.sort_values(
        ["symbol", "entry_time", "model_probability", "family", "direction"],
        ascending=[True, True, False, False, False], kind="mergesort")
    chosen = chosen.drop_duplicates(["symbol", "entry_time"], keep="first")
    accepted = be.ranked_runtime_gate(
        rows_for(chosen), loss_pause=loss_pause, hourly_cap=hourly_cap)
    result = money(accepted, margin)
    result.update({
        "target_hit_rate": round(100 * sum(
            row["fixed_outcome"] == "TP2" for row in accepted) / len(accepted), 2),
        "non_loss_rate": round(100 * sum(
            float(row["net_r"]) >= 0 for row in accepted) / len(accepted), 2),
        "actual_max_stop_streak": max_stop_streak(accepted),
        "max_entries_same_timestamp": max(collections.Counter(
            int(float(row["entry_time"])) for row in accepted).values()),
    })
    return result


def complete(result: dict, months: tuple[str, ...]) -> dict:
    values = result["monthly_return_pct"]
    return ({month: values[month] for month in months}
            if set(months).issubset(values) else {})


def main() -> None:
    historical = prior.canonical_history(base.load_target(TARGET_R))
    historical["fixed_outcome"] = historical["outcome"].astype(str)
    fit = base.stage(historical, FIT)
    schema = base.design(fit)
    model = HistGradientBoostingClassifier(
        learning_rate=.05, max_iter=150, max_leaf_nodes=15,
        min_samples_leaf=300, l2_regularization=1.0,
        early_stopping=False, random_state=20260908)
    model.fit(base.matrix(fit, schema), fit.outcome.eq("TP2").astype(int))
    historical["model_probability"] = model.predict_proba(
        base.matrix(historical, schema))[:, 1]
    select_scores = base.stage(historical, SELECT).model_probability
    sweep = []
    for quantile in QUANTILES:
        threshold = float(select_scores.quantile(quantile))
        for loss_pause in LOSS_PAUSES:
            for hourly_cap in HOURLY_CAPS:
                selection = evaluate(
                    historical, SELECT, threshold, loss_pause, hourly_cap)
                confirmation = evaluate(
                    historical, CONFIRM, threshold, loss_pause, hourly_cap)
                months = [*complete(selection, SELECT_MONTHS).values(),
                          *complete(confirmation, CONFIRM_MONTHS).values()]
                floor = min(months) if months else -999.0
                sweep.append({
                    "quantile": quantile, "threshold": threshold,
                    "loss_pause": loss_pause, "hourly_cap": hourly_cap,
                    "complete_month_floor_pct": floor,
                    "selection": selection, "confirmation": confirmation,
                    "score": round(floor - .1 * max(
                        selection["dd_pct"], confirmation["dd_pct"]), 8),
                })
    sweep.sort(key=lambda item: (item["score"], -item["hourly_cap"]),
               reverse=True)
    frozen = sweep[0]
    threshold = float(frozen["threshold"])
    loss_pause = int(frozen["loss_pause"])
    hourly_cap = int(frozen["hourly_cap"])
    validation = evaluate(
        historical, VALIDATE, threshold, loss_pause, hourly_cap)
    later = evaluate(historical, LATER, threshold, loss_pause, hourly_cap)
    direct = direct_fixed_frame(prior.direct_frame())
    direct["model_probability"] = model.predict_proba(
        base.matrix(direct, schema))[:, 1]
    fresh = evaluate(direct.assign(period="xperp"), ("xperp",), threshold,
                     loss_pause, hourly_cap)
    report = {
        "status": "RESEARCH_ONLY",
        "protocol": {
            "fit": list(FIT), "selection": [*SELECT, *CONFIRM],
            "target_r": TARGET_R, "entry": "next 15m market open",
            "exit": "fixed 1R target, original 1R stop, 48-bar expiry",
            "ranking": "highest model probability, one per timestamp",
        },
        "frozen_before_2024": frozen,
        "validation_2024": validation,
        "later_2025_2026": later,
        "fresh_direct_xperp": fresh,
        "top_development_sweep": sweep[:20],
    }
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({
        "frozen": {key: frozen[key] for key in (
            "quantile", "threshold", "loss_pause", "hourly_cap",
            "complete_month_floor_pct", "score")},
        "selection": frozen["selection"],
        "confirmation": frozen["confirmation"],
        "validation": validation, "later": later, "fresh": fresh,
    }, indent=2))


if __name__ == "__main__":
    main()
