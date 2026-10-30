"""Re-fit Profit-First on the literal sequential 2R-before-BE outcome.

The earlier model predicted whether 2R arrived before the original stop.  That
label is incompatible with the live exit, which closes the runner at entry on
the first post-TP1 retrace.  This audit builds the label from raw candles in
the exact live order, fits only 2020-2021, selects cutoff and first-bank share
on 2022-2023, and opens 2024+ only after development selection is frozen.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

import crypto_05_2r_precision_barbell_oct5_audit as legacy
import crypto_high_target_tree_oct5_audit as base
import crypto_profit_first_material_barbell_oct18_audit as joiner
import crypto_profit_first_material_target_oct18_audit as frozen
import crypto_profit_first_material_target_oct30_1000_forward as forward
import crypto_profit_first_runner_protection_oct30_audit as exact


ROOT = Path("reports/audit_2026_09_08")
OUT = ROOT / "crypto_profit_first_sequential_runner_oct30.json"
ROWS_OUT = ROOT / "crypto_profit_first_sequential_runner_oct30.pkl"
QUANTILES = (.50, .60, .70, .75, .80, .825, .85, .875, .90, .925, .95,
             .96, .97, .98)
FRACTIONS = (.05, .10, .25, .50, .75)
MODEL = {"max_leaf_nodes": 7, "min_samples_leaf": 300,
         "learning_rate": .05}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def attach_current(frame: pd.DataFrame,
                   paths: dict[tuple[str, str], Path]) -> pd.DataFrame:
    records = []
    for (period, symbol), part in frame.groupby(["period", "symbol"], sort=False):
        path = paths.get((str(period), str(symbol)))
        if path is None:
            raise FileNotFoundError(f"missing candle cache: {period} {symbol}")
        candle = exact.load_pickle(path)
        lookup = {int(value): index for index, value in enumerate(candle["time"])}
        for row in part.to_dict("records"):
            index = lookup.get(int(row["entry_time"]))
            if index is None or index + exact.HORIZON - 1 >= len(candle["time"]):
                raise RuntimeError(
                    f"missing 48-bar runner window: {symbol} {row['entry_time']}")
            label, runner_r, banked, exit_time = exact.replay_runner(
                candle, index, direction=str(row["direction"]),
                entry=float(row["entry"]), stop=float(row["sl"]),
                activation_r=.5, lock_r=0.0)
            records.append({
                **row, "sequential_outcome": label,
                "sequential_runner_r": runner_r,
                "sequential_banked": banked,
                "sequential_exit_time": exit_time,
            })
    return pd.DataFrame.from_records(records)


def apply_fraction(frame: pd.DataFrame, fraction: float) -> pd.DataFrame:
    result = frame.copy()
    runner = result.sequential_runner_r.astype(float)
    banked = result.sequential_banked.astype(bool)
    result["net_r"] = np.where(
        banked, fraction * .5 + (1 - fraction) * runner, runner)
    result["exit_time"] = result.sequential_exit_time.astype(float)
    result["runner_outcome"] = result.sequential_outcome.astype(str)
    return result


def rows_for(frame: pd.DataFrame, periods: tuple[str, ...], threshold: float,
             fraction: float) -> list[dict]:
    part = base.stage(frame, periods)
    part = part[part.model_probability.ge(threshold)].copy()
    part = apply_fraction(part, fraction)
    key = str(fraction).replace(".", "p")
    return exact.runtime_gate(base.standardized(
        part, f"sequential_runner_f{key}"))


def metrics(rows: list[dict], margin: float = .03) -> dict:
    value = exact.money(rows, margin)
    labels = pd.Series([row["runner_outcome"] for row in rows], dtype=str)
    value["outcome_counts"] = labels.value_counts().sort_index().to_dict()
    value["literal_runner_hit_rate_pct"] = round(
        100 * float(labels.eq("TP_RUNNER").mean()), 4) if len(labels) else 0.0
    value["mean_r"] = round(float(np.mean(
        [row["net_r"] for row in rows])), 6) if rows else 0.0
    return value


def evaluate(frame: pd.DataFrame, periods: tuple[str, ...], threshold: float,
             fraction: float) -> tuple[dict, list[dict]]:
    rows = rows_for(frame, periods, threshold, fraction)
    return metrics(rows), rows


def compact(value: dict) -> dict:
    return {key: value[key] for key in (
        "n", "wr", "return_pct", "dd_pct", "max_loss_streak",
        "negative_months", "trades_per_calendar_day", "monthly_return_pct",
        "outcome_counts", "literal_runner_hit_rate_pct", "mean_r")}


def monthly_equivalent(value: dict) -> float:
    months = max(1, len(value.get("monthly_return_pct", {})))
    factor = 1 + float(value["return_pct"]) / 100
    return 100 * math.expm1(math.log(factor) / months) if factor > 0 else -100.0


def development_ok(selection: dict, confirmation: dict) -> bool:
    return bool(
        selection["n"] >= 300 and confirmation["n"] >= 300
        and selection["return_pct"] > 0 and confirmation["return_pct"] > 0
        and selection["dd_pct"] <= 10 and confirmation["dd_pct"] <= 10
        and selection["trades_per_calendar_day"] >= 2
        and confirmation["trades_per_calendar_day"] >= 1.5
        and selection["max_loss_streak"] <= 8
        and confirmation["max_loss_streak"] <= 8)


def candidate_score(selection: dict, confirmation: dict) -> float:
    monthly = min(monthly_equivalent(selection), monthly_equivalent(confirmation))
    dd = max(selection["dd_pct"], confirmation["dd_pct"])
    negative = selection["negative_months"] + confirmation["negative_months"]
    return round(monthly - .10 * dd - .12 * negative, 8)


def direct_slices(rows: list[dict], cutoff: int) -> dict:
    frame = pd.DataFrame(rows)
    october = frame[frame.entry_time.ge(pd.Timestamp(
        "2026-10-01", tz="UTC").timestamp())].to_dict("records")
    unseen = frame[frame.entry_time.gt(cutoff)].to_dict("records")
    return {"october": compact(metrics(october)),
            "unseen_after_oct16": compact(metrics(unseen))}


def objective_gate(value: dict) -> dict:
    monthly = monthly_equivalent(value)
    return {
        "monthly_equivalent_return_pct": round(monthly, 4),
        "required_monthly_return_pct": 30.0,
        "required_trades_per_calendar_day": 3.0,
        "return_pass": monthly >= 30,
        "frequency_pass": value["trades_per_calendar_day"] >= 3,
        "no_negative_month_pass": value["negative_months"] == 0,
        "passes": bool(monthly >= 30
                       and value["trades_per_calendar_day"] >= 3
                       and value["negative_months"] == 0),
    }


def main() -> None:
    source = json.loads(forward.SOURCE_REPORT.read_text(encoding="utf-8"))
    all_periods = (*frozen.FIT, *frozen.SELECT, *frozen.CONFIRM,
                   *frozen.VALIDATION, *frozen.LATER)
    historical_paths = exact.candle_paths(all_periods)
    joined = frozen.load_joined(2.0)
    literal = attach_current(joined, historical_paths)
    fit = base.stage(literal, frozen.FIT)
    schema = base.design(fit)
    model = HistGradientBoostingClassifier(
        **MODEL, max_iter=150, l2_regularization=1.0,
        early_stopping=False, random_state=20260908)
    model.fit(base.matrix(fit, schema),
              fit.sequential_outcome.eq("TP_RUNNER").astype(int))
    scored = legacy.scored(literal, model, schema)
    selection_scores = base.stage(scored, frozen.SELECT).model_probability
    thresholds = {quantile: float(selection_scores.quantile(quantile))
                  for quantile in QUANTILES}

    frontier = []
    eligible = []
    for quantile, threshold in thresholds.items():
        for fraction in FRACTIONS:
            selection, _ = evaluate(
                scored, frozen.SELECT, threshold, fraction)
            confirmation, _ = evaluate(
                scored, frozen.CONFIRM, threshold, fraction)
            passed = development_ok(selection, confirmation)
            item = {
                "selection_quantile": quantile, "threshold": threshold,
                "first_fraction": fraction,
                "selection_2022h1": compact(selection),
                "confirmation_2022h2_2023": compact(confirmation),
                "development_pass": passed,
                "score": candidate_score(selection, confirmation),
            }
            frontier.append(item)
            if passed:
                eligible.append(item)
    frontier.sort(key=lambda item: (item["development_pass"], item["score"]),
                  reverse=True)
    eligible.sort(key=lambda item: item["score"], reverse=True)
    chosen = eligible[0] if eligible else None

    validation = later = direct_metrics = direct_rows = None
    direct_paths = exact.direct_paths()
    raw_direct = joiner.join_direct_outcomes(forward.FAR, forward.NEAR)
    raw_direct = raw_direct[
        raw_direct.symbol.isin(base.ROBUST_SYMBOLS)
        & raw_direct.risk_pct.between(legacy.RISK_PCT_MIN, legacy.RISK_PCT_MAX,
                                      inclusive="both")].copy()
    direct_literal = attach_current(raw_direct.assign(period="direct"), direct_paths)
    direct_scored = legacy.scored(direct_literal, model, schema)
    if chosen:
        threshold = float(chosen["threshold"])
        fraction = float(chosen["first_fraction"])
        validation, _ = evaluate(
            scored, frozen.VALIDATION, threshold, fraction)
        later, _ = evaluate(scored, frozen.LATER, threshold, fraction)
        _, direct_rows = evaluate(
            direct_scored, ("direct",), threshold, fraction)
        pd.DataFrame(direct_rows).to_pickle(ROWS_OUT)
        cutoff = int(pd.Timestamp(
            source["fresh_direct_coverage"]["last_entry"]).timestamp())
        direct_metrics = direct_slices(direct_rows, cutoff)

    robust = bool(
        chosen and validation["return_pct"] > 0 and later["return_pct"] > 0
        and validation["dd_pct"] <= 10 and later["dd_pct"] <= 10
        and validation["negative_months"] <= 2
        and later["negative_months"] <= 4
        and direct_metrics["unseen_after_oct16"]["n"] >= 20
        and direct_metrics["unseen_after_oct16"]["return_pct"] > 0)
    direct_gate = (objective_gate(direct_metrics["unseen_after_oct16"])
                   if direct_metrics else None)
    report = {
        "status": ("SEQUENTIAL_RUNNER_SHADOW_CHALLENGER" if robust
                   else "REJECTED_SEQUENTIAL_RUNNER"),
        "ready_for_live_orders": False,
        "integrity_finding": {
            "previous_label": "2R before original structural stop",
            "live_label": "2R before post-TP1 breakeven",
            "conclusion": "previous forward profit is not valid for the live exit order",
        },
        "protocol": {
            "fit": list(frozen.FIT),
            "selection": [*frozen.SELECT, *frozen.CONFIRM],
            "sealed_until_selection": [*frozen.VALIDATION, *frozen.LATER, "direct"],
            "target_label": "literal 2R before post-0.5R breakeven",
            "first_fractions": list(FRACTIONS),
            "threshold_quantiles": list(QUANTILES),
            "stop_target_collision": "stop first",
            "breakeven_activation": "bar after 0.5R first touch",
            "margin_fraction": .03, "leverage": 10,
        },
        "candidate_count": len(frontier),
        "development_pass_count": len(eligible),
        "selected_before_holdouts": chosen,
        "validation_2024": compact(validation) if validation else None,
        "later_2025_2026": compact(later) if later else None,
        "direct_xperp": direct_metrics,
        "direct_economic_gate": direct_gate,
        "robust_improvement": robust,
        "development_frontier": frontier,
        "sources": {str(path): digest(path) for path in (
            *historical_paths.values(), *direct_paths.values(),
            forward.FAR, forward.NEAR, forward.SOURCE_REPORT,
            Path(__file__).resolve())},
        "limitations": [
            "OHLC cannot reveal intrabar path; ambiguous bars are resolved against the strategy.",
            "Gross returns exclude fees at user request.",
            "A candle replay cannot reproduce market-order latency or slippage.",
        ],
    }
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({
        "status": report["status"],
        "development_pass_count": len(eligible),
        "selected": chosen,
        "validation": report["validation_2024"],
        "later": report["later_2025_2026"],
        "direct": direct_metrics,
        "direct_gate": direct_gate,
        "robust": robust,
        "frontier_top5": frontier[:5],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
