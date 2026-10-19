"""Search stable scenario filters on the selected exact crypto exit path.

Buckets must have positive expectancy independently in 2022H1 and
2022H2-2023.  Filter fields and thresholds are frozen before all 2024+ and
direct X-Perp evaluation.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

import crypto_2r_precision_barbell_oct5_audit as precision
import crypto_exact_exit_geometry_oct20_audit as geometry
import crypto_high_target_tree_oct5_audit as base
import crypto_precision_be_runner_oct18_audit as exact
import crypto_precision_profit_first_oct18_audit as prior


OUT = exact.ROOT / "crypto_exact_scenario_filter_oct20.json"
GEOMETRY_REPORT = exact.ROOT / "crypto_exact_exit_geometry_oct20.json"
FIELD_SETS = (
    ("btc_regime",), ("family",), ("symbol",), ("utc_session",),
    ("btc_regime", "direction"), ("btc_regime", "family"),
    ("btc_regime", "utc_session"), ("family", "direction"),
    ("family", "utc_session"), ("symbol", "direction"),
    ("btc_regime", "family", "direction"),
)
MIN_COUNTS = (10, 20, 40)
MIN_MEAN_R = (0.0, .025, .05, .075, .10)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stable_buckets(selection: pd.DataFrame, confirmation: pd.DataFrame,
                   fields: tuple[str, ...], min_count: int,
                   min_mean_r: float) -> set[tuple]:
    def summary(frame: pd.DataFrame) -> pd.DataFrame:
        return frame.groupby(list(fields), dropna=False).net_r.agg(
            count="size", mean="mean")

    left = summary(selection)
    right = summary(confirmation)
    joined = left.join(right, how="inner", lsuffix="_selection",
                       rsuffix="_confirmation")
    good = joined[
        (joined.count_selection >= min_count)
        & (joined.count_confirmation >= min_count)
        & (joined.mean_selection >= min_mean_r)
        & (joined.mean_confirmation >= min_mean_r)]
    if len(fields) == 1:
        return {(value,) for value in good.index.tolist()}
    return {tuple(value) for value in good.index.tolist()}


def apply_buckets(frame: pd.DataFrame, fields: tuple[str, ...],
                  allowed: set[tuple]) -> pd.DataFrame:
    keys = pd.Series(list(map(tuple, frame[list(fields)].to_numpy())),
                     index=frame.index)
    return frame[keys.isin(allowed)].copy()


def main() -> None:
    frozen_report = json.loads(GEOMETRY_REPORT.read_text())
    frozen = frozen_report["selected_before_holdouts"]
    target = float(frozen["target_r"])
    activation = (99.0 if frozen["activation_r"] is None
                  else float(frozen["activation_r"]))
    threshold = float(frozen["threshold"])
    loss_pause = int(frozen["loss_pause"])
    hourly_cap = int(frozen["hourly_cap"])

    historical = geometry.eligible(
        prior.canonical_history(precision.load_joined()), target)
    replayed = exact.attach_runtime_runner(
        historical, direct=False, activation_r=activation, runner_r=target)
    fit = base.stage(replayed, precision.FIT)
    design = base.design(fit)
    model = geometry.fit_model(fit, design)
    scored = geometry.score(replayed, model, design)
    development_selection = base.stage(scored, precision.SELECT)
    development_selection = development_selection[
        development_selection.model_probability >= threshold]
    development_confirmation = base.stage(scored, precision.CONFIRM)
    development_confirmation = development_confirmation[
        development_confirmation.model_probability >= threshold]

    candidates = []
    for fields in FIELD_SETS:
        for min_count in MIN_COUNTS:
            for min_mean_r in MIN_MEAN_R:
                allowed = stable_buckets(
                    development_selection, development_confirmation,
                    fields, min_count, min_mean_r)
                if not allowed:
                    continue
                filtered = apply_buckets(scored, fields, allowed)
                selection, _ = geometry.evaluate(
                    filtered, precision.SELECT, threshold,
                    loss_pause, hourly_cap, .03)
                confirmation, _ = geometry.evaluate(
                    filtered, precision.CONFIRM, threshold,
                    loss_pause, hourly_cap, .03)
                floor = min(
                    geometry.month_floor(selection, exact.SELECTION_MONTHS),
                    geometry.month_floor(confirmation, exact.CONFIRMATION_MONTHS))
                candidates.append({
                    "fields": list(fields), "min_count_each_period": min_count,
                    "min_mean_r_each_period": min_mean_r,
                    "allowed_buckets": [list(value) for value in sorted(
                        allowed, key=lambda value: tuple(map(str, value)))],
                    "selection": selection, "confirmation": confirmation,
                    "development_floor_pct": round(floor, 4),
                    "development_negative_months": (
                        selection["negative_months"]
                        + confirmation["negative_months"]),
                    "risk_admissible": (
                        selection["n"] >= 50 and confirmation["n"] >= 50
                        and selection["dd_pct"] <= 10
                        and confirmation["dd_pct"] <= 10),
                    "frequency_goal_met": (
                        selection["trades_per_calendar_day"] >= 3
                        and confirmation["trades_per_calendar_day"] >= 3),
                })
    candidates.sort(key=lambda item: (
        item["risk_admissible"], -item["development_negative_months"],
        item["development_floor_pct"],
        item["selection"]["return_pct"] + item["confirmation"]["return_pct"],
        -max(item["selection"]["dd_pct"], item["confirmation"]["dd_pct"]),
    ), reverse=True)
    selected = candidates[0] if candidates else None
    holdouts = direct_values = objective = None
    if selected:
        fields = tuple(selected["fields"])
        allowed = {tuple(value) for value in selected["allowed_buckets"]}
        filtered = apply_buckets(scored, fields, allowed)
        holdouts = {}
        for margin in (.03, .05):
            validation, _ = geometry.evaluate(
                filtered, precision.VALIDATION, threshold,
                loss_pause, hourly_cap, margin)
            later, _ = geometry.evaluate(
                filtered, precision.HIDDEN, threshold,
                loss_pause, hourly_cap, margin)
            holdouts[f"margin_{round(100 * margin)}pct"] = {
                "validation_2024": validation,
                "later_2025_2026": later,
            }

        direct = geometry.eligible(
            prior.direct_frame().assign(period="xperp"), target)
        direct_replayed = exact.attach_runtime_runner(
            direct, direct=True, activation_r=activation, runner_r=target)
        direct_scored = geometry.score(direct_replayed, model, design)
        direct_filtered = apply_buckets(direct_scored, fields, allowed)
        direct_values = {}
        for margin in (.03, .05):
            value, _ = geometry.evaluate(
                direct_filtered, ("xperp",), threshold,
                loss_pause, hourly_cap, margin)
            direct_values[f"margin_{round(100 * margin)}pct"] = value
        complete = direct_values["margin_5pct"]["monthly_return_pct"]
        complete = {month: complete[month] for month in
                    ("2026-05", "2026-06", "2026-07", "2026-08", "2026-09")
                    if month in complete}
        objective = {
            "development_positive_months": (
                selected["selection"]["negative_months"] == 0
                and selected["confirmation"]["negative_months"] == 0),
            "holdouts_positive_months_5pct": (
                holdouts["margin_5pct"]["validation_2024"]["negative_months"] == 0
                and holdouts["margin_5pct"]["later_2025_2026"]["negative_months"] == 0),
            "direct_complete_months_at_least_30pct_5pct": bool(
                len(complete) == 5 and min(complete.values()) >= 30),
            "direct_drawdown_at_most_10pct_5pct": (
                direct_values["margin_5pct"]["dd_pct"] <= 10),
        }

    passed = bool(objective and all(objective.values()))
    report = {
        "status": ("EXACT_SCENARIO_FILTER_SHADOW_CANDIDATE"
                   if passed else "REJECTED_OBJECTIVE"),
        "ready_for_live_orders": False,
        "protocol": {
            "base_geometry_report": str(GEOMETRY_REPORT),
            "bucket_rule": "positive mean R independently in selection and confirmation",
            "filter_fields": [list(value) for value in FIELD_SETS],
            "fit": list(precision.FIT),
            "development": [*precision.SELECT, *precision.CONFIRM],
            "holdouts": [*precision.VALIDATION, *precision.HIDDEN, "direct X-Perp"],
            "priority": "worst development month, return, drawdown; no win-rate gate",
        },
        "base_geometry": frozen,
        "candidate_count": len(candidates),
        "risk_admissible_count": sum(
            item["risk_admissible"] for item in candidates),
        "selected_before_holdouts": selected,
        "top20_before_holdouts": candidates[:20],
        "holdouts": holdouts,
        "direct_xperp": direct_values,
        "objective_checks": objective,
        "sources": {
            **exact.CANDLE_SOURCES,
            str(GEOMETRY_REPORT): digest(GEOMETRY_REPORT),
            str(Path(__file__).resolve()): digest(Path(__file__).resolve()),
        },
        "limitations": [
            "Gross results exclude fees at user request.",
            "Bucket filters are development-selected and require holdout confirmation.",
            "Direct X-Perp history is shorter than a full market cycle.",
        ],
    }
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({
        "status": report["status"], "selected": selected,
        "holdouts": holdouts, "direct_xperp": direct_values,
        "objective_checks": objective,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
