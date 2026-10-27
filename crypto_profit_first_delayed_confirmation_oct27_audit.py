"""Causal delayed-entry audit for the frozen Profit-First crypto route.

The frozen model still selects the setup at the original decision time.  A
candidate can then wait for one complete 15-minute candle and enter with a
market order at the following open if a small, interpretable confirmation rule
passes.  The structural stop is retained and both 0.5R and 2R outcomes are
replayed from the delayed entry with pessimistic stop-first ordering.

Rules and scopes are ranked on 2022 through 2023 only.  The winner is frozen
before 2024, 2025-2026 and the Oct-16 forward interval are inspected.
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
import crypto_profit_first_material_target_oct27_0418_forward as forward
from crypto_runtime_shadow_portfolio_oct3_audit import money, runtime_gate
from src.backtest_integrity import simulate_exit


ROOT = Path("reports/audit_2026_09_08")
OUT = ROOT / "crypto_profit_first_delayed_confirmation_oct27.json"
ROWS_OUT = ROOT / "crypto_profit_first_delayed_confirmation_oct27.pkl"
BACKTEST_CACHE = Path("backtest_cache")
DIRECT_CACHE = ROOT / "xperp_15m_oct27_0418_material"
RULES = (
    "body_aligned", "progress_hold", "body_progress_hold", "strong_close",
    "break_signal_extreme", "progress_shallow_05r", "sweep_reclaim",
)
SCOPES = ("all", "clustered", "range_reversion", "trend_pullback", "breakout")
INTERVAL = 900
HORIZON = 48


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_pickle(path: Path) -> dict:
    return pickle.loads(path.read_bytes())


def candle_paths(periods: tuple[str, ...]) -> dict[tuple[str, str], Path]:
    paths: dict[tuple[str, str], Path] = {}
    for period in periods:
        suffix = exact.PERIOD_CACHE[period]
        for symbol in base.ROBUST_SYMBOLS:
            path = BACKTEST_CACHE / f"{symbol}_15min_{suffix}.pkl"
            if path.exists():
                paths[(period, symbol)] = path
    return paths


def direct_candle_paths() -> dict[tuple[str, str], Path]:
    return {
        ("direct", path.name.split("_15min_")[0]): path
        for path in DIRECT_CACHE.glob("*_15min_current.pkl")
    }


def scope_mask(frame: pd.DataFrame, scope: str) -> pd.Series:
    if scope == "all":
        return pd.Series(True, index=frame.index)
    if scope == "clustered":
        return frame.cluster_symbols.ge(2)
    if scope not in {"range_reversion", "trend_pullback", "breakout"}:
        raise ValueError(f"unknown confirmation scope: {scope}")
    return frame.family.eq(scope)


def add_cluster_size(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    keys = ["period", "entry_time", "direction"]
    result["cluster_symbols"] = result.groupby(keys, sort=False).symbol.transform(
        "nunique")
    return result


def _confirmation_values(row: pd.Series, candle: dict, index: int) -> dict:
    """Features available after the original entry candle has closed."""
    direction = 1 if str(row.direction) == "LONG" else -1
    open_ = float(candle["open"][index])
    high = float(candle["high"][index])
    low = float(candle["low"][index])
    close = float(candle["close"][index])
    prior_high = float(candle["high"][index - 1])
    prior_low = float(candle["low"][index - 1])
    entry = float(row.entry)
    stop = float(row.sl)
    risk = abs(entry - stop)
    span = max(high - low, 1e-15)
    body_r = direction * (close - open_) / risk
    progress_r = direction * (close - entry) / risk
    close_location = ((close - low) / span if direction > 0
                      else (high - close) / span)
    held_stop = low > stop if direction > 0 else high < stop
    broke = close > prior_high if direction > 0 else close < prior_low
    swept = ((low < entry and close > entry) if direction > 0
             else (high > entry and close < entry))
    return {
        "confirmation_body_r": body_r,
        "confirmation_progress_r": progress_r,
        "confirmation_close_location": close_location,
        "confirmation_held_stop": bool(held_stop),
        "confirmation_broke_signal_extreme": bool(broke),
        "confirmation_sweep_reclaim": bool(swept),
    }


def rule_mask(frame: pd.DataFrame, rule: str) -> pd.Series:
    valid = frame.confirmation_valid.astype(bool)
    held = valid & frame.confirmation_held_stop.astype(bool)
    body = frame.confirmation_body_r.gt(0)
    progress = frame.confirmation_progress_r.gt(0)
    if rule == "body_aligned":
        return held & body
    if rule == "progress_hold":
        return held & progress
    if rule == "body_progress_hold":
        return held & body & progress
    if rule == "strong_close":
        return held & body & progress & frame.confirmation_close_location.ge(.70)
    if rule == "break_signal_extreme":
        return held & frame.confirmation_broke_signal_extreme.astype(bool)
    if rule == "progress_shallow_05r":
        return held & progress & frame.confirmation_progress_r.le(.50)
    if rule == "sweep_reclaim":
        return held & body & frame.confirmation_sweep_reclaim.astype(bool)
    raise ValueError(f"unknown confirmation rule: {rule}")


def _resolve(candle: dict, first: int, direction: str, entry: float,
             stop: float, target_r: float):
    sign = 1 if direction == "LONG" else -1
    risk = abs(entry - stop)
    target = entry + sign * risk * target_r
    end = min(len(candle["time"]), first + HORIZON)
    indices = []
    for index in range(first, end):
        if index > first and int(candle["time"][index]) \
                - int(candle["time"][index - 1]) != INTERVAL:
            break
        indices.append(index)
    if not indices:
        return None
    return simulate_exit(
        candle, indices, direction=direction, entry=entry, sl=stop,
        tp1=target, tp2=target, atr=risk, tp1_fraction=0, trail=False,
        trail_mult=0, stop_on_close=False, backstop_r=1,
        choose_trail=lambda *_: 0,
    )


def enrich_delayed(frame: pd.DataFrame,
                   paths: dict[tuple[str, str], Path]) -> pd.DataFrame:
    """Attach confirmation features and literal delayed outcomes."""
    records = []
    invalid = {
        "confirmation_valid": False,
        "confirmation_body_r": np.nan,
        "confirmation_progress_r": np.nan,
        "confirmation_close_location": np.nan,
        "confirmation_held_stop": False,
        "confirmation_broke_signal_extreme": False,
        "confirmation_sweep_reclaim": False,
        "delayed_entry_time": np.nan,
        "delayed_entry": np.nan,
        "delayed_risk_pct": np.nan,
        "delayed_near_outcome": "INVALID",
        "delayed_near_exit_time": np.nan,
        "delayed_far_outcome": "INVALID",
        "delayed_far_exit_time": np.nan,
    }
    for (period, symbol), part in frame.groupby(["period", "symbol"], sort=False):
        path = paths.get((str(period), str(symbol)))
        if path is None:
            records.extend({**row, **invalid}
                           for row in part.to_dict("records"))
            continue
        candle = load_pickle(path)
        times = np.asarray(candle["time"], dtype=np.int64)
        lookup = {int(value): index for index, value in enumerate(times)}
        for _, row in part.iterrows():
            original_time = int(row.entry_time)
            index = lookup.get(original_time)
            if index is None or index < 1 or index + 1 >= len(times):
                records.append({**row.to_dict(), **invalid})
                continue
            if (int(times[index + 1]) - int(times[index]) != INTERVAL
                    or int(times[index]) - int(times[index - 1]) != INTERVAL):
                records.append({**row.to_dict(), **invalid})
                continue
            values = _confirmation_values(row, candle, index)
            delayed_entry = float(candle["open"][index + 1])
            direction = str(row.direction)
            sign = 1 if direction == "LONG" else -1
            delayed_risk = sign * (delayed_entry - float(row.sl))
            if delayed_entry <= 0 or delayed_risk <= 0:
                records.append({**row.to_dict(), **invalid, **values})
                continue
            delayed_risk_pct = delayed_risk / delayed_entry
            near = _resolve(candle, index + 1, direction, delayed_entry,
                            float(row.sl), .5)
            far = _resolve(candle, index + 1, direction, delayed_entry,
                           float(row.sl), 2.0)
            if near is None or far is None:
                records.append({**row.to_dict(), **invalid, **values})
                continue
            records.append({
                **row.to_dict(), **values,
                "confirmation_valid": True,
                "delayed_entry_time": int(times[index + 1]),
                "delayed_entry": delayed_entry,
                "delayed_risk_pct": delayed_risk_pct,
                "delayed_near_outcome": near.outcome,
                "delayed_near_exit_time": int(times[near.bar]) + INTERVAL,
                "delayed_far_outcome": far.outcome,
                "delayed_far_exit_time": int(times[far.bar]) + INTERVAL,
            })
    return pd.DataFrame.from_records(records)


def apply_variant(enriched: pd.DataFrame, scope: str, rule: str) -> pd.DataFrame:
    affected = scope_mask(enriched, scope)
    passed = rule_mask(enriched, rule)
    keep = ~affected | passed
    result = enriched.loc[keep].copy()
    delayed = affected.reindex(result.index).fillna(False).astype(bool)
    valid_risk = result.delayed_risk_pct.between(
        legacy.RISK_PCT_MIN, legacy.RISK_PCT_MAX, inclusive="both")
    result = result.loc[~delayed | valid_risk].copy()
    delayed = delayed.reindex(result.index).fillna(False).astype(bool)
    result.loc[delayed, "entry_time"] = result.loc[
        delayed, "delayed_entry_time"].astype(np.int64)
    result.loc[delayed, "entry"] = result.loc[delayed, "delayed_entry"]
    result.loc[delayed, "risk_pct"] = result.loc[delayed, "delayed_risk_pct"]
    result.loc[delayed, "near_outcome"] = result.loc[
        delayed, "delayed_near_outcome"]
    result.loc[delayed, "near_exit_time"] = result.loc[
        delayed, "delayed_near_exit_time"].astype(np.int64)
    result.loc[delayed, "outcome"] = result.loc[delayed, "delayed_far_outcome"]
    result.loc[delayed, "exit_time"] = result.loc[
        delayed, "delayed_far_exit_time"].astype(np.int64)
    result["confirmation_delayed"] = delayed
    return result


def accepted(frame: pd.DataFrame, periods: tuple[str, ...], fraction: float,
             scope: str | None = None, rule: str | None = None) -> list[dict]:
    part = base.stage(frame, periods).copy()
    if scope is not None and rule is not None:
        part = apply_variant(part, scope, rule)
    else:
        part["confirmation_delayed"] = False
    part = frozen.barbell_frame(part, fraction, 2.0)
    return runtime_gate(base.standardized(part, "profit_first_delayed_confirmation"))


def evaluate(frame: pd.DataFrame, periods: tuple[str, ...], fraction: float,
             scope: str | None = None, rule: str | None = None,
             margin: float = .03) -> dict:
    return money(accepted(frame, periods, fraction, scope, rule), margin)


def month_floor(value: dict) -> float:
    return min((float(item) for item in value.get("monthly_return_pct", {}).values()),
               default=-999.0)


def compact(value: dict) -> dict:
    return {key: value[key] for key in (
        "n", "wr", "return_pct", "dd_pct", "max_loss_streak",
        "negative_months", "trades_per_calendar_day", "monthly_return_pct")}


def development_ok(selection: dict, confirmation: dict,
                   baseline_selection: dict, baseline_confirmation: dict) -> bool:
    return bool(
        selection["return_pct"] >= .85 * baseline_selection["return_pct"]
        and confirmation["return_pct"] >= .85 * baseline_confirmation["return_pct"]
        and selection["dd_pct"] <= 1.10 * baseline_selection["dd_pct"]
        and confirmation["dd_pct"] <= 1.10 * baseline_confirmation["dd_pct"]
        and selection["negative_months"] <= baseline_selection["negative_months"]
        and confirmation["negative_months"] <= baseline_confirmation["negative_months"]
        and selection["trades_per_calendar_day"] >= 3
        and confirmation["trades_per_calendar_day"] >= 2
        and selection["max_loss_streak"] <= baseline_selection["max_loss_streak"]
        and confirmation["max_loss_streak"] <= baseline_confirmation["max_loss_streak"])


def development_score(selection: dict, confirmation: dict,
                      baseline_selection: dict,
                      baseline_confirmation: dict) -> float:
    retained = min(selection["return_pct"] / baseline_selection["return_pct"],
                   confirmation["return_pct"] / baseline_confirmation["return_pct"])
    dd_gain = min(
        baseline_selection["dd_pct"] / max(selection["dd_pct"], 1e-9),
        baseline_confirmation["dd_pct"] / max(confirmation["dd_pct"], 1e-9))
    wr_gain = min(selection["wr"] - baseline_selection["wr"],
                  confirmation["wr"] - baseline_confirmation["wr"])
    floor_gain = min(month_floor(selection) - month_floor(baseline_selection),
                     month_floor(confirmation) - month_floor(baseline_confirmation))
    return round(retained + .08 * dd_gain + .002 * wr_gain + .002 * floor_gain, 8)


def dominates(candidate: dict, baseline: dict) -> bool:
    return bool(candidate["return_pct"] >= baseline["return_pct"]
                and candidate["dd_pct"] <= baseline["dd_pct"]
                and candidate["negative_months"] <= baseline["negative_months"]
                and candidate["max_loss_streak"] <= baseline["max_loss_streak"])


def identity(rows: list[dict] | pd.DataFrame) -> set[tuple]:
    frame = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
    return set(zip(frame.symbol.astype(str),
                   pd.to_numeric(frame.entry_time).astype(np.int64),
                   frame.direction.astype(str), frame.family.astype(str)))


def main() -> None:
    source = json.loads(forward.SOURCE_REPORT.read_text(encoding="utf-8"))
    policy = source["frozen_before_2024"]
    threshold = float(policy["threshold"])
    fraction = float(policy["near_fraction"])

    historical = frozen.load_joined(2.0)
    fit = base.stage(historical, frozen.FIT)
    schema = base.design(fit)
    model = HistGradientBoostingClassifier(
        **frozen.MODEL, max_iter=150, l2_regularization=1.0,
        early_stopping=False, random_state=20260908)
    model.fit(base.matrix(fit, schema), fit.outcome.eq("TP2").astype(int))
    scored = legacy.scored(historical, model, schema)
    research_periods = (
        *frozen.SELECT, *frozen.CONFIRM, *frozen.VALIDATION, *frozen.LATER)
    selected = add_cluster_size(base.stage(
        scored[scored.model_probability.ge(threshold)].copy(),
        research_periods))
    historical_paths = candle_paths(
        research_periods)
    enriched = enrich_delayed(selected, historical_paths)
    original_ids = identity(selected)
    enriched_ids = identity(enriched)
    if original_ids != enriched_ids:
        raise ValueError({
            "missing_confirmation_rows": len(original_ids - enriched_ids),
            "unexpected_confirmation_rows": len(enriched_ids - original_ids),
        })

    baseline_selection = evaluate(enriched, frozen.SELECT, fraction)
    baseline_confirmation = evaluate(enriched, frozen.CONFIRM, fraction)
    frontier = []
    eligible = []
    for scope in SCOPES:
        for rule in RULES:
            selection = evaluate(enriched, frozen.SELECT, fraction, scope, rule)
            confirmation = evaluate(
                enriched, frozen.CONFIRM, fraction, scope, rule)
            passed = development_ok(
                selection, confirmation, baseline_selection,
                baseline_confirmation)
            item = {
                "scope": scope, "rule": rule,
                "selection_2022h1": compact(selection),
                "confirmation_2022h2_2023": compact(confirmation),
                "development_pass": passed,
                "score": development_score(
                    selection, confirmation, baseline_selection,
                    baseline_confirmation),
            }
            frontier.append(item)
            if passed:
                eligible.append(item)
    frontier.sort(key=lambda item: (
        item["development_pass"], item["score"]), reverse=True)
    eligible.sort(key=lambda item: item["score"], reverse=True)
    chosen = eligible[0] if eligible else None

    baseline_validation = evaluate(enriched, frozen.VALIDATION, fraction)
    baseline_later = evaluate(enriched, frozen.LATER, fraction)
    validation = later = None
    if chosen:
        validation = evaluate(enriched, frozen.VALIDATION, fraction,
                              chosen["scope"], chosen["rule"])
        later = evaluate(enriched, frozen.LATER, fraction,
                         chosen["scope"], chosen["rule"])

    direct = joiner.join_direct_outcomes(forward.FAR, forward.NEAR)
    direct = direct[
        direct.symbol.isin(base.ROBUST_SYMBOLS)
        & direct.risk_pct.between(legacy.RISK_PCT_MIN, legacy.RISK_PCT_MAX,
                                  inclusive="both")].copy()
    direct = legacy.scored(direct, model, schema)
    direct = direct[direct.model_probability.ge(threshold)].copy()
    direct["period"] = "direct"
    direct = add_cluster_size(direct)
    direct_enriched = enrich_delayed(direct, direct_candle_paths())
    if identity(direct) != identity(direct_enriched):
        raise ValueError("direct confirmation enrichment lost frozen candidates")
    baseline_direct_rows = accepted(
        direct_enriched, ("direct",), fraction)
    frozen_rows = pd.read_pickle(forward.ROWS_OUT)
    baseline_parity = identity(baseline_direct_rows) == identity(frozen_rows)
    if not baseline_parity:
        raise ValueError("baseline delayed-entry audit does not reproduce frozen route")

    direct_metrics = selected_direct_metrics = None
    direct_rows: list[dict] = []
    original_cutoff = int(pd.Timestamp(
        source["fresh_direct_coverage"]["last_entry"]).timestamp())
    if chosen:
        direct_rows = accepted(direct_enriched, ("direct",), fraction,
                               chosen["scope"], chosen["rule"])
        pd.DataFrame(direct_rows).to_pickle(ROWS_OUT)

        def slices(rows: list[dict]) -> dict:
            current = pd.DataFrame(rows)
            october = current[current.entry_time.ge(pd.Timestamp(
                "2026-10-01", tz="UTC").timestamp())]
            unseen = current[current.entry_time.gt(original_cutoff)]
            return {
                "october": money(october.to_dict("records"), .03),
                "unseen_after_oct16": money(unseen.to_dict("records"), .03),
            }

        direct_metrics = slices(baseline_direct_rows)
        selected_direct_metrics = slices(direct_rows)

    robust = bool(
        chosen and dominates(validation, baseline_validation)
        and dominates(later, baseline_later)
        and selected_direct_metrics["unseen_after_oct16"]["return_pct"]
            >= direct_metrics["unseen_after_oct16"]["return_pct"]
        and selected_direct_metrics["unseen_after_oct16"]
            ["trades_per_calendar_day"] >= 3)
    report = {
        "status": ("DELAYED_CONFIRMATION_SHADOW_CANDIDATE"
                   if robust else "REJECTED_NO_ROBUST_IMPROVEMENT"),
        "ready_for_live_orders": False,
        "protocol": {
            "model_and_threshold": "frozen before 2024",
            "policy_selection": [*frozen.SELECT, *frozen.CONFIRM],
            "validation": list(frozen.VALIDATION),
            "later": list(frozen.LATER),
            "rules": list(RULES), "scopes": list(SCOPES),
            "entry": "market at next 15-minute open after one closed confirmation candle",
            "exit": "retained structural stop; 5% at 0.5R and 95% runner at 2R",
            "collision": "stop first", "timeout_bars": HORIZON,
            "money": "client-selected current-equity margin; audited at 3%, fixed 10x",
        },
        "candidate_count": len(frontier),
        "development_pass_count": len(eligible),
        "historical_enrichment_integrity": {
            "selected_rows": len(selected), "enriched_rows": len(enriched),
            "passes": original_ids == enriched_ids,
        },
        "baseline": {
            "selection_2022h1": compact(baseline_selection),
            "confirmation_2022h2_2023": compact(baseline_confirmation),
            "validation_2024": compact(baseline_validation),
            "later_2025_2026": compact(baseline_later),
            "direct_replay_parity": baseline_parity,
            "direct": direct_metrics,
        },
        "selected_before_validation": chosen,
        "validation_2024": compact(validation) if validation else None,
        "later_2025_2026": compact(later) if later else None,
        "direct_xperp": selected_direct_metrics,
        "robust_improvement": robust,
        "development_frontier": frontier,
        "sources": {
            str(path): digest(path)
            for path in (
                *historical_paths.values(), *direct_candle_paths().values(),
                forward.FAR, forward.NEAR, forward.SOURCE_REPORT,
                forward.ROWS_OUT, Path(__file__).resolve())
        },
        "limitations": [
            "A candle-open replay cannot reproduce market-order latency or slippage.",
            "Delayed execution changes the stop distance and therefore re-applies the frozen 1%-2% risk-distance gate.",
            "Gross returns exclude fees at user request.",
        ],
    }
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True),
                   encoding="utf-8")
    print(json.dumps({
        "status": report["status"],
        "development_pass_count": len(eligible),
        "selected": chosen,
        "baseline_validation": compact(baseline_validation),
        "validation": report["validation_2024"],
        "baseline_later": compact(baseline_later),
        "later": report["later_2025_2026"],
        "direct": selected_direct_metrics,
        "robust_improvement": robust,
        "frontier_top5": frontier[:5],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
