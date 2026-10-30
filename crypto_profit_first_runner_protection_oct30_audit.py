"""Causal audit of the Profit-First runner stop after the 0.5R bank.

The entry model, probability threshold, 5% first bank and 2R final target are
frozen.  Only the runner protection changes.  Candidate policies are selected
on 2022-2023, then inspected once on 2024, 2025-2026 and the latest direct
X-Perp snapshot.  Raw 15-minute candles use pessimistic stop-first ordering;
protection becomes active on the bar after its activation level is observed.
"""
from __future__ import annotations

import hashlib
import json
import math
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

import crypto_05_2r_precision_barbell_oct5_audit as legacy
import crypto_exact_router_replay as exact
import crypto_high_target_tree_oct5_audit as base
import crypto_profit_first_material_barbell_oct18_audit as joiner
import crypto_profit_first_material_target_oct18_audit as frozen
import crypto_profit_first_material_target_oct30_1000_forward as forward
from crypto_runtime_shadow_portfolio_oct3_audit import money, runtime_gate


ROOT = Path("reports/audit_2026_09_08")
OUT = ROOT / "crypto_profit_first_runner_protection_oct30.json"
ROWS_OUT = ROOT / "crypto_profit_first_runner_protection_oct30.pkl"
BACKTEST_CACHE = Path("backtest_cache")
DIRECT_CACHE = ROOT / "xperp_15m_oct30_1000_material"
INTERVAL = 900
HORIZON = 48
FIRST_TARGET_R = .5
FIRST_FRACTION = .05
RUNNER_FRACTION = 1 - FIRST_FRACTION
FINAL_TARGET_R = 2.0
ACTIVATIONS = (.5, .75, 1.0, 1.25, 99.0)
LOCKS = (0.0, .10, .25, .50)
FRACTIONS = (.05, .10, .25, .50, .75)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_pickle(path: Path) -> dict:
    return pickle.loads(path.read_bytes())


def candle_paths(periods: tuple[str, ...]) -> dict[tuple[str, str], Path]:
    result = {}
    for period in periods:
        suffix = exact.PERIOD_CACHE[period]
        for symbol in base.ROBUST_SYMBOLS:
            path = BACKTEST_CACHE / f"{symbol}_15min_{suffix}.pkl"
            if path.exists():
                result[(period, symbol)] = path
    return result


def direct_paths() -> dict[tuple[str, str], Path]:
    return {
        ("direct", path.name.split("_15min_")[0]): path
        for path in DIRECT_CACHE.glob("*_15min_current.pkl")
    }


def _stop_fill(opened: float, level: float, sign: int) -> float:
    return min(opened, level) if sign > 0 else max(opened, level)


def replay_runner(candle: dict, index: int, *, direction: str, entry: float,
                  stop: float, activation_r: float,
                  lock_r: float) -> tuple[str, float, bool, int]:
    """Replay one runner policy with runtime-equivalent activation timing."""
    if lock_r > activation_r:
        raise ValueError("a locked stop cannot exceed its activation level")
    sign = 1 if direction == "LONG" else -1
    risk = abs(entry - stop)
    if risk <= 0:
        raise ValueError("non-positive structural risk")
    first_target = entry + sign * risk * FIRST_TARGET_R
    final_target = entry + sign * risk * FINAL_TARGET_R
    activation = entry + sign * risk * activation_r
    protected_stop = entry + sign * risk * lock_r
    reached_first = False
    protected = False
    last = min(index + HORIZON - 1, len(candle["time"]) - 1)
    previous_time = None
    for bar in range(index, last + 1):
        timestamp = int(candle["time"][bar])
        if previous_time is not None and timestamp - previous_time != INTERVAL:
            last = bar - 1
            break
        previous_time = timestamp
        opened = float(candle["open"][bar])
        high = float(candle["high"][bar])
        low = float(candle["low"][bar])
        stop_level = protected_stop if protected else stop
        stop_hit = low <= stop_level if sign > 0 else high >= stop_level
        if stop_hit:
            fill = _stop_fill(opened, stop_level, sign)
            runner_r = sign * (fill - entry) / risk
            if not reached_first:
                return "SL", runner_r, False, timestamp + INTERVAL
            label = "LOCK" if protected else "TP1_STOP"
            return label, runner_r, True, timestamp + INTERVAL

        target_hit = high >= final_target if sign > 0 else low <= final_target
        if target_hit:
            return "TP_RUNNER", FINAL_TARGET_R, True, timestamp + INTERVAL

        first_hit = high >= first_target if sign > 0 else low <= first_target
        reached_first = reached_first or first_hit
        activation_hit = high >= activation if sign > 0 else low <= activation
        # As in main.py, the old stop remains active for the entire activation
        # candle.  The new stop participates from the following candle.
        if reached_first and activation_hit:
            protected = True

    exit_time = int(candle["time"][last]) + INTERVAL
    if reached_first:
        return "TP_EXPIRED", 0.0, True, exit_time
    close = float(candle["close"][last])
    marked_r = sign * (close - entry) / risk
    return "EXPIRED", marked_r, False, exit_time


def attach_outcomes(frame: pd.DataFrame,
                    paths: dict[tuple[str, str], Path]) -> pd.DataFrame:
    """Attach every bounded policy outcome in one raw-candle pass."""
    records = []
    for (period, symbol), part in frame.groupby(["period", "symbol"], sort=False):
        path = paths.get((str(period), str(symbol)))
        if path is None:
            raise FileNotFoundError(f"missing candle cache: {period} {symbol}")
        candle = load_pickle(path)
        lookup = {int(value): index for index, value in enumerate(candle["time"])}
        for row in part.to_dict("records"):
            index = lookup.get(int(row["entry_time"]))
            if index is None or index + HORIZON - 1 >= len(candle["time"]):
                raise RuntimeError(
                    f"missing 48-bar runner window: {symbol} {row['entry_time']}")
            output = dict(row)
            for activation_r in ACTIVATIONS:
                for lock_r in LOCKS:
                    if activation_r == 99.0 and lock_r != 0.0:
                        continue
                    if lock_r > activation_r:
                        continue
                    label, runner_r, banked, exit_time = replay_runner(
                        candle, index, direction=str(row["direction"]),
                        entry=float(row["entry"]), stop=float(row["sl"]),
                        activation_r=activation_r, lock_r=lock_r)
                    prefix = policy_key(activation_r, lock_r)
                    output[f"{prefix}_label"] = label
                    output[f"{prefix}_runner_r"] = runner_r
                    output[f"{prefix}_banked"] = banked
                    output[f"{prefix}_exit_time"] = exit_time
            records.append(output)
    return pd.DataFrame.from_records(records)


def policy_key(activation_r: float, lock_r: float) -> str:
    return f"a{activation_r:g}_l{lock_r:g}".replace(".", "p")


def rows_for(frame: pd.DataFrame, periods: tuple[str, ...], activation_r: float,
             lock_r: float, fraction: float) -> list[dict]:
    part = base.stage(frame, periods).copy()
    prefix = policy_key(activation_r, lock_r)
    runner_r = part[f"{prefix}_runner_r"].astype(float)
    banked = part[f"{prefix}_banked"].astype(bool)
    part["net_r"] = np.where(
        banked,
        fraction * FIRST_TARGET_R + (1 - fraction) * runner_r,
        runner_r)
    part["exit_time"] = part[f"{prefix}_exit_time"].astype(float)
    part["runner_outcome"] = part[f"{prefix}_label"].astype(str)
    fraction_key = str(fraction).replace(".", "p")
    rows = base.standardized(part, f"runner_protection_{prefix}_f{fraction_key}")
    return runtime_gate(rows)


def evaluate(frame: pd.DataFrame, periods: tuple[str, ...], activation_r: float,
             lock_r: float, fraction: float,
             margin: float = .03) -> tuple[dict, list[dict]]:
    rows = rows_for(frame, periods, activation_r, lock_r, fraction)
    result = money(rows, margin)
    labels = pd.Series([row["runner_outcome"] for row in rows], dtype=str)
    result["outcome_counts"] = labels.value_counts().sort_index().to_dict()
    result["mean_r"] = round(float(np.mean([row["net_r"] for row in rows])), 6)
    return result, rows


def compact(value: dict) -> dict:
    keys = (
        "n", "wr", "return_pct", "dd_pct", "max_loss_streak",
        "negative_months", "trades_per_calendar_day", "monthly_return_pct",
        "outcome_counts", "mean_r", "projected_30d_return_pct")
    return {key: value[key] for key in keys if key in value}


def month_floor(value: dict) -> float:
    return min((float(item) for item in value["monthly_return_pct"].values()),
               default=-999.0)


def development_pass(candidate: dict, baseline: dict) -> bool:
    return bool(
        candidate["return_pct"] >= baseline["return_pct"]
        and candidate["dd_pct"] <= 1.05 * baseline["dd_pct"]
        and candidate["negative_months"] <= baseline["negative_months"]
        and candidate["max_loss_streak"] <= baseline["max_loss_streak"]
        and candidate["trades_per_calendar_day"] >= .98 * baseline["trades_per_calendar_day"])


def score(selection: dict, confirmation: dict,
          base_selection: dict, base_confirmation: dict) -> float:
    return round(
        min(selection["return_pct"] / max(base_selection["return_pct"], 1e-9),
            confirmation["return_pct"] / max(base_confirmation["return_pct"], 1e-9))
        + .1 * min(base_selection["dd_pct"] / max(selection["dd_pct"], 1e-9),
                   base_confirmation["dd_pct"] / max(confirmation["dd_pct"], 1e-9))
        + .002 * min(month_floor(selection) - month_floor(base_selection),
                     month_floor(confirmation) - month_floor(base_confirmation)), 8)


def dominates(candidate: dict, baseline: dict) -> bool:
    return bool(candidate["return_pct"] >= baseline["return_pct"]
                and candidate["dd_pct"] <= baseline["dd_pct"]
                and candidate["negative_months"] <= baseline["negative_months"]
                and candidate["max_loss_streak"] <= baseline["max_loss_streak"])


def prepare_historical(model, schema: dict, threshold: float) -> tuple[pd.DataFrame, dict]:
    joined = frozen.load_joined(FINAL_TARGET_R)
    scored = legacy.scored(joined, model, schema)
    periods = (*frozen.SELECT, *frozen.CONFIRM, *frozen.VALIDATION, *frozen.LATER)
    chosen = base.stage(scored, periods)
    chosen = chosen[chosen.model_probability.ge(threshold)].copy()
    paths = candle_paths(periods)
    return attach_outcomes(chosen, paths), paths


def prepare_direct(model, schema: dict, threshold: float) -> tuple[pd.DataFrame, dict]:
    joined = joiner.join_direct_outcomes(forward.FAR, forward.NEAR)
    eligible = joined[
        joined.symbol.isin(base.ROBUST_SYMBOLS)
        & joined.risk_pct.between(legacy.RISK_PCT_MIN, legacy.RISK_PCT_MAX,
                                  inclusive="both")].copy()
    scored = legacy.scored(eligible, model, schema)
    chosen = scored[scored.model_probability.ge(threshold)].copy()
    chosen["period"] = "direct"
    paths = direct_paths()
    return attach_outcomes(chosen, paths), paths


def direct_slices(frame: pd.DataFrame, activation_r: float, lock_r: float,
                  fraction: float, original_cutoff: int) -> tuple[dict, list[dict]]:
    rows = rows_for(frame, ("direct",), activation_r, lock_r, fraction)
    full = pd.DataFrame(rows)
    unseen = full[full.entry_time.gt(original_cutoff)].to_dict("records")
    october = full[full.entry_time.ge(pd.Timestamp(
        "2026-10-01", tz="UTC").timestamp())].to_dict("records")
    return {
        "october": compact_metrics(money(october, .03), october),
        "unseen_after_oct16": compact_metrics(money(unseen, .03), unseen),
    }, rows


def compact_metrics(value: dict, rows: list[dict]) -> dict:
    labels = pd.Series([row["runner_outcome"] for row in rows], dtype=str)
    value["outcome_counts"] = labels.value_counts().sort_index().to_dict()
    value["mean_r"] = round(float(np.mean([row["net_r"] for row in rows])), 6) if rows else 0.0
    if rows:
        first = pd.Timestamp(min(float(row["entry_time"]) for row in rows),
                             unit="s", tz="UTC")
        last = pd.Timestamp(max(float(row["entry_time"]) for row in rows),
                            unit="s", tz="UTC")
        days = max(1, (last.date() - first.date()).days + 1)
        factor = 1 + float(value["return_pct"]) / 100
        value["projected_30d_return_pct"] = round(
            100 * (factor ** (30 / days) - 1), 4) if factor > 0 else -100.0
    else:
        value["projected_30d_return_pct"] = 0.0
    return compact(value)


def main() -> None:
    source = json.loads(forward.SOURCE_REPORT.read_text(encoding="utf-8"))
    policy = source["frozen_before_2024"]
    threshold = float(policy["threshold"])
    if float(policy["near_fraction"]) != FIRST_FRACTION:
        raise ValueError("source policy is no longer the frozen 5% bank")
    joined = frozen.load_joined(FINAL_TARGET_R)
    fit = base.stage(joined, frozen.FIT)
    schema = base.design(fit)
    model = HistGradientBoostingClassifier(
        **frozen.MODEL, max_iter=150, l2_regularization=1.0,
        early_stopping=False, random_state=20260908)
    model.fit(base.matrix(fit, schema), fit.outcome.eq("TP2").astype(int))

    historical, historical_sources = prepare_historical(model, schema, threshold)
    baseline_selection, _ = evaluate(
        historical, frozen.SELECT, .5, 0.0, FIRST_FRACTION)
    baseline_confirmation, _ = evaluate(
        historical, frozen.CONFIRM, .5, 0.0, FIRST_FRACTION)
    frontier = []
    eligible = []
    for activation_r in ACTIVATIONS:
        for lock_r in LOCKS:
            if (activation_r == 99.0 and lock_r != 0.0) or lock_r > activation_r:
                continue
            for fraction in FRACTIONS:
                if (activation_r, lock_r, fraction) == (.5, 0.0, FIRST_FRACTION):
                    continue
                selection, _ = evaluate(
                    historical, frozen.SELECT, activation_r, lock_r, fraction)
                confirmation, _ = evaluate(
                    historical, frozen.CONFIRM, activation_r, lock_r, fraction)
                passed = (development_pass(selection, baseline_selection)
                          and development_pass(confirmation, baseline_confirmation))
                item = {
                    "activation_r": activation_r, "lock_r": lock_r,
                    "first_fraction": fraction,
                    "selection_2022h1": compact(selection),
                    "confirmation_2022h2_2023": compact(confirmation),
                    "development_pass": passed,
                    "score": score(selection, confirmation,
                                   baseline_selection, baseline_confirmation),
                }
                frontier.append(item)
                if passed:
                    eligible.append(item)
    frontier.sort(key=lambda item: (item["development_pass"], item["score"]),
                  reverse=True)
    eligible.sort(key=lambda item: item["score"], reverse=True)
    chosen = eligible[0] if eligible else None

    baseline_validation, _ = evaluate(
        historical, frozen.VALIDATION, .5, 0.0, FIRST_FRACTION)
    baseline_later, _ = evaluate(
        historical, frozen.LATER, .5, 0.0, FIRST_FRACTION)
    direct, direct_sources = prepare_direct(model, schema, threshold)
    previous_far = direct.outcome.eq("TP2")
    literal_runner = direct.a0p5_l0_label.eq("TP_RUNNER")
    label_integrity = {
        "selected_candidates": len(direct),
        "previous_2r_positive_labels": int(previous_far.sum()),
        "literal_2r_before_be_labels": int(literal_runner.sum()),
        "previous_positive_but_live_be_first": int(
            (previous_far & ~literal_runner).sum()),
        "conclusion": (
            "the independent 2R label overstates runners because it does not "
            "stop the path when the live breakeven exit occurs"),
    }
    cutoff = int(pd.Timestamp(source["fresh_direct_coverage"]["last_entry"]).timestamp())
    baseline_direct, _ = direct_slices(
        direct, .5, 0.0, FIRST_FRACTION, cutoff)
    validation = later = selected_direct = selected_rows = None
    if chosen:
        activation_r = float(chosen["activation_r"])
        lock_r = float(chosen["lock_r"])
        fraction = float(chosen["first_fraction"])
        validation, _ = evaluate(
            historical, frozen.VALIDATION, activation_r, lock_r, fraction)
        later, _ = evaluate(
            historical, frozen.LATER, activation_r, lock_r, fraction)
        selected_direct, selected_rows = direct_slices(
            direct, activation_r, lock_r, fraction, cutoff)

    robust = bool(
        chosen and dominates(validation, baseline_validation)
        and dominates(later, baseline_later)
        and selected_direct["unseen_after_oct16"]["return_pct"]
            >= baseline_direct["unseen_after_oct16"]["return_pct"]
        and selected_direct["unseen_after_oct16"]["dd_pct"]
            <= baseline_direct["unseen_after_oct16"]["dd_pct"])
    if robust and selected_rows is not None:
        pd.DataFrame(selected_rows).to_pickle(ROWS_OUT)

    report = {
        "status": ("RUNNER_PROTECTION_SHADOW_CANDIDATE" if robust else
                   "BACKTEST_INTEGRITY_FAILURE_NO_ROBUST_REPLACEMENT"),
        "ready_for_live_orders": False,
        "protocol": {
            "entry_model_and_threshold": "frozen before 2024",
            "selected_on": [*frozen.SELECT, *frozen.CONFIRM],
            "untouched_checks": [*frozen.VALIDATION, *frozen.LATER, "direct"],
            "first_bank": "5% at 0.5R", "runner_target": "95% at 2R",
            "activations_r": list(ACTIVATIONS), "locks_r": list(LOCKS),
            "first_fractions": list(FRACTIONS),
            "ordering": "resting stop first; protection starts next bar",
            "horizon_bars": HORIZON, "margin_fraction": .03, "leverage": 10,
        },
        "candidate_count": len(frontier),
        "development_pass_count": len(eligible),
        "label_integrity": label_integrity,
        "baseline": {
            "policy": {"activation_r": .5, "lock_r": 0.0,
                       "first_fraction": FIRST_FRACTION},
            "selection_2022h1": compact(baseline_selection),
            "confirmation_2022h2_2023": compact(baseline_confirmation),
            "validation_2024": compact(baseline_validation),
            "later_2025_2026": compact(baseline_later),
            "direct": baseline_direct,
        },
        "selected_before_holdouts": chosen,
        "validation_2024": compact(validation) if validation else None,
        "later_2025_2026": compact(later) if later else None,
        "direct_xperp": selected_direct,
        "robust_improvement": robust,
        "development_frontier": frontier,
        "sources": {str(path): digest(path) for path in (
            *historical_sources.values(), *direct_sources.values(),
            forward.FAR, forward.NEAR, forward.SOURCE_REPORT,
            Path(__file__).resolve())},
        "limitations": [
            "OHLC cannot reveal the intrabar path, so ambiguous candles are resolved against the strategy.",
            "Gross returns exclude fees at user request.",
            "A candle replay cannot reproduce market-order latency or slippage.",
        ],
    }
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({
        "status": report["status"],
        "development_pass_count": len(eligible),
        "selected": chosen,
        "baseline_validation": compact(baseline_validation),
        "validation": report["validation_2024"],
        "baseline_later": compact(baseline_later),
        "later": report["later_2025_2026"],
        "baseline_direct": baseline_direct,
        "direct": selected_direct,
        "robust_improvement": robust,
        "frontier_top5": frontier[:5],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
