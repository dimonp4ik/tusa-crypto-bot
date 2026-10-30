"""Causal rolling-health router on exact Profit-First outcomes.

Each decision uses only candidate outcomes whose exits precede the new entry.
The frozen entry model and live 5%@0.5R / 95%@2R exit remain unchanged.  The
router can pause a family/session/regime bucket whose recent exact expectancy
is weak.  Policies are selected on 2022-2023 and only then opened on 2024+ and
the current direct X-Perp replay.
"""
from __future__ import annotations

import collections
from datetime import datetime, timezone
import hashlib
import heapq
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

import crypto_high_target_tree_oct5_audit as base
import crypto_profit_first_material_target_oct18_audit as frozen
import crypto_profit_first_runner_protection_oct30_audit as exact
import crypto_profit_first_exact_oct30_2030_forward as latest
from crypto_runtime_shadow_portfolio_oct3_audit import money, runtime_gate


ROOT = Path("reports/audit_2026_09_08")
OUT = ROOT / "crypto_profit_first_exact_health_router_oct30.json"
ROWS_OUT = ROOT / "crypto_profit_first_exact_health_router_oct30.pkl"
SCOPES = {
    "family": ("family",),
    "session": ("utc_session",),
    "btc_regime": ("btc_regime",),
    "family_regime": ("family", "btc_regime"),
    "family_session": ("family", "utc_session"),
    "regime_session": ("btc_regime", "utc_session"),
}
LOOKBACKS = (20, 40, 80)
THRESHOLDS = (-.05, 0.0, .05)
FRACTION = .05


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fit_model():
    historical = frozen.load_joined(2.0)
    fit = base.stage(historical, frozen.FIT)
    schema = base.design(fit)
    model = HistGradientBoostingClassifier(
        **frozen.MODEL, max_iter=150, l2_regularization=1.0,
        early_stopping=False, random_state=20260908)
    model.fit(base.matrix(fit, schema), fit.outcome.eq("TP2").astype(int))
    return model, schema


def current_outcomes(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    prefix = exact.policy_key(.5, 0.0)
    runner = result[f"{prefix}_runner_r"].astype(float)
    banked = result[f"{prefix}_banked"].astype(bool)
    result["net_r"] = np.where(
        banked, FRACTION * .5 + (1 - FRACTION) * runner, runner)
    result["exit_time"] = result[f"{prefix}_exit_time"].astype(float)
    result["runner_outcome"] = result[f"{prefix}_label"].astype(str)
    return result


def group_key(row: dict, fields: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(str(row[field]) for field in fields)


def apply_health(frame: pd.DataFrame, *, scope: str, lookback: int,
                 threshold: float) -> pd.DataFrame:
    """Filter causally; rejected rows still mature as observable paper labels."""
    fields = SCOPES[scope]
    histories: dict[tuple[str, ...], collections.deque[float]] = {}
    pending: list[tuple[float, int, tuple[str, ...], float]] = []
    keep = []
    ordered = frame.sort_values(
        ["entry_time", "symbol", "family", "direction"], kind="mergesort")
    for sequence, (index, row) in enumerate(ordered.iterrows()):
        now = float(row.entry_time)
        while pending and pending[0][0] < now:
            _, _, key, net_r = heapq.heappop(pending)
            history = histories.setdefault(
                key, collections.deque(maxlen=lookback))
            history.append(net_r)
        record = row.to_dict()
        key = group_key(record, fields)
        history = histories.setdefault(key, collections.deque(maxlen=lookback))
        minimum = max(10, lookback // 2)
        healthy = len(history) < minimum or float(np.mean(history)) >= threshold
        if healthy:
            keep.append(index)
        heapq.heappush(pending, (
            float(row.exit_time), sequence, key, float(row.net_r)))
    return frame.loc[keep].copy()


def accepted(frame: pd.DataFrame, module: str) -> list[dict]:
    return runtime_gate(base.standardized(frame, module))


def metrics(rows: list[dict], periods: tuple[str, ...]) -> dict:
    selected = [row for row in rows if str(row["period"]) in periods]
    value = money(selected, .03)
    labels = pd.Series([row["runner_outcome"] for row in selected], dtype=str)
    value["outcome_counts"] = labels.value_counts().sort_index().to_dict()
    value["mean_r"] = round(float(np.mean(
        [row["net_r"] for row in selected])), 6) if selected else 0.0
    return value


def compact(value: dict) -> dict:
    return {key: value[key] for key in (
        "n", "wr", "return_pct", "dd_pct", "max_loss_streak",
        "negative_months", "trades_per_calendar_day", "monthly_return_pct",
        "outcome_counts", "mean_r")}


def month_floor(value: dict) -> float:
    return min((float(item) for item in value["monthly_return_pct"].values()),
               default=-999.0)


def development_pass(candidate: dict, baseline: dict) -> bool:
    return bool(
        candidate["return_pct"] >= baseline["return_pct"]
        and candidate["dd_pct"] <= baseline["dd_pct"]
        and candidate["negative_months"] <= baseline["negative_months"]
        and candidate["max_loss_streak"] <= baseline["max_loss_streak"]
        and candidate["trades_per_calendar_day"]
            >= .75 * baseline["trades_per_calendar_day"])


def score(selection: dict, confirmation: dict,
          base_selection: dict, base_confirmation: dict) -> float:
    retained = min(
        selection["return_pct"] / max(base_selection["return_pct"], 1e-9),
        confirmation["return_pct"] / max(base_confirmation["return_pct"], 1e-9))
    dd_gain = min(
        base_selection["dd_pct"] / max(selection["dd_pct"], 1e-9),
        base_confirmation["dd_pct"] / max(confirmation["dd_pct"], 1e-9))
    floor_gain = min(month_floor(selection) - month_floor(base_selection),
                     month_floor(confirmation) - month_floor(base_confirmation))
    return round(retained + .08 * dd_gain + .002 * floor_gain, 8)


def dominates(candidate: dict, baseline: dict) -> bool:
    return bool(candidate["return_pct"] >= baseline["return_pct"]
                and candidate["dd_pct"] <= baseline["dd_pct"]
                and candidate["negative_months"] <= baseline["negative_months"]
                and candidate["max_loss_streak"] <= baseline["max_loss_streak"])


def direct_metrics(rows: list[dict], original_cutoff: int) -> dict:
    frame = pd.DataFrame(rows)
    october = frame[frame.entry_time.ge(pd.Timestamp(
        "2026-10-01", tz="UTC").timestamp())].to_dict("records")
    unseen = frame[frame.entry_time.gt(original_cutoff)].to_dict("records")
    return {
        "october": latest.metrics(pd.DataFrame(october), .03),
        "unseen_after_oct16": latest.metrics(pd.DataFrame(unseen), .03),
    }


def prepare_latest_direct(model, schema: dict,
                          threshold: float) -> tuple[pd.DataFrame, dict]:
    joined = exact.joiner.join_direct_outcomes(latest.NEW_FAR, latest.NEW_NEAR)
    eligible = joined[
        joined.symbol.isin(base.ROBUST_SYMBOLS)
        & joined.risk_pct.between(
            exact.legacy.RISK_PCT_MIN, exact.legacy.RISK_PCT_MAX,
            inclusive="both")].copy()
    scored = exact.legacy.scored(eligible, model, schema)
    selected = scored[scored.model_probability.ge(threshold)].copy()
    selected["period"] = "direct"
    candle_paths = latest.paths(latest.NEW_MANIFEST.parent)
    return exact.attach_outcomes(selected, candle_paths), candle_paths


def main() -> None:
    model, schema = fit_model()
    threshold = float(json.loads(
        latest.SOURCE_REPORT.read_text(encoding="utf-8"))
        ["frozen_before_2024"]["threshold"])
    historical_raw, historical_sources = exact.prepare_historical(
        model, schema, threshold)
    historical = current_outcomes(historical_raw)
    baseline_rows = accepted(historical, "exact_health_baseline")
    base_selection = metrics(baseline_rows, frozen.SELECT)
    base_confirmation = metrics(baseline_rows, frozen.CONFIRM)

    frontier = []
    eligible = []
    for scope in SCOPES:
        for lookback in LOOKBACKS:
            for threshold_r in THRESHOLDS:
                filtered = apply_health(
                    historical, scope=scope, lookback=lookback,
                    threshold=threshold_r)
                rows = accepted(filtered, "exact_health_candidate")
                selection = metrics(rows, frozen.SELECT)
                confirmation = metrics(rows, frozen.CONFIRM)
                passed = (development_pass(selection, base_selection)
                          and development_pass(confirmation, base_confirmation))
                item = {
                    "scope": scope, "lookback": lookback,
                    "minimum_history": max(10, lookback // 2),
                    "mean_r_threshold": threshold_r,
                    "selection_2022h1": compact(selection),
                    "confirmation_2022h2_2023": compact(confirmation),
                    "development_pass": passed,
                    "score": score(selection, confirmation,
                                   base_selection, base_confirmation),
                }
                frontier.append(item)
                if passed:
                    eligible.append(item)
    frontier.sort(key=lambda item: (item["development_pass"], item["score"]),
                  reverse=True)
    eligible.sort(key=lambda item: item["score"], reverse=True)
    chosen = eligible[0] if eligible else None

    base_validation = metrics(baseline_rows, frozen.VALIDATION)
    base_later = metrics(baseline_rows, frozen.LATER)
    validation = later = None
    selected_rows = None
    if chosen:
        filtered = apply_health(
            historical, scope=chosen["scope"], lookback=chosen["lookback"],
            threshold=chosen["mean_r_threshold"])
        selected_rows = accepted(filtered, "exact_health_candidate")
        validation = metrics(selected_rows, frozen.VALIDATION)
        later = metrics(selected_rows, frozen.LATER)

    direct_raw, direct_sources = prepare_latest_direct(model, schema, threshold)
    direct = current_outcomes(direct_raw)
    direct_baseline_rows = accepted(direct, "exact_health_baseline")
    source = json.loads(latest.SOURCE_REPORT.read_text(encoding="utf-8"))
    original_cutoff = int(pd.Timestamp(
        source["fresh_direct_coverage"]["last_entry"]).timestamp())
    baseline_direct = direct_metrics(direct_baseline_rows, original_cutoff)
    candidate_direct = None
    direct_rows = None
    if chosen:
        filtered_direct = apply_health(
            direct, scope=chosen["scope"], lookback=chosen["lookback"],
            threshold=chosen["mean_r_threshold"])
        direct_rows = accepted(filtered_direct, "exact_health_candidate")
        candidate_direct = direct_metrics(direct_rows, original_cutoff)

    robust = bool(
        chosen and dominates(validation, base_validation)
        and dominates(later, base_later)
        and candidate_direct["unseen_after_oct16"]["return_pct"]
            >= baseline_direct["unseen_after_oct16"]["return_pct"]
        and candidate_direct["unseen_after_oct16"]["dd_pct"]
            <= baseline_direct["unseen_after_oct16"]["dd_pct"]
        and candidate_direct["unseen_after_oct16"]["trades_per_calendar_day"] >= 3)
    if robust and direct_rows is not None:
        pd.DataFrame(direct_rows).to_pickle(ROWS_OUT)
    report = {
        "status": ("EXACT_HEALTH_SHADOW_CHALLENGER_AWAIT_FORWARD" if robust
                   else "REJECTED_EXACT_HEALTH_ROUTER"),
        "ready_for_live_orders": False, "orders_possible": False,
        "runtime_change_allowed": False,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "protocol": {
            "entry_and_exit": "frozen; exact 5%@0.5R then 95%@2R/BE",
            "causality": "only outcomes with exit_time before entry_time",
            "scopes": SCOPES, "lookbacks": list(LOOKBACKS),
            "mean_r_thresholds": list(THRESHOLDS),
            "selection": [*frozen.SELECT, *frozen.CONFIRM],
            "holdouts": [*frozen.VALIDATION, *frozen.LATER, "direct"],
            "orders": "market only", "position_fraction": "client setting",
        },
        "candidate_count": len(frontier),
        "development_pass_count": len(eligible),
        "baseline": {
            "selection_2022h1": compact(base_selection),
            "confirmation_2022h2_2023": compact(base_confirmation),
            "validation_2024": compact(base_validation),
            "later_2025_2026": compact(base_later),
            "direct": baseline_direct,
        },
        "selected_before_holdouts": chosen,
        "validation_2024": compact(validation) if validation else None,
        "later_2025_2026": compact(later) if later else None,
        "direct_xperp": candidate_direct,
        "robust_improvement": robust,
        "development_frontier": frontier,
        "sources": {str(path): digest(path) for path in (
            *historical_sources.values(), *direct_sources.values(),
            latest.SOURCE_REPORT, latest.NEW_FAR, latest.NEW_NEAR,
            latest.NEW_MANIFEST, Path(__file__).resolve())},
        "limitations": [
            "A health router can reduce frequency during prolonged weak periods.",
            "Rejected candidates must remain paper-tracked so future health labels stay observable.",
            "Gross returns exclude fees at user request.",
        ],
    }
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({
        "status": report["status"], "development_pass_count": len(eligible),
        "selected": chosen, "baseline_validation": compact(base_validation),
        "validation": report["validation_2024"],
        "baseline_later": compact(base_later),
        "later": report["later_2025_2026"],
        "baseline_direct": baseline_direct, "direct": candidate_direct,
        "robust": robust, "frontier_top5": frontier[:5],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
