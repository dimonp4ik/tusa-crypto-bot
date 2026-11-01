"""Audit static exact-route bucket vetoes with sealed historical holdouts.

A bucket may be vetoed only when its exact-R expectancy is weak independently
in 2022H1 and in 2022H2-2023.  The chosen rule is then opened on 2024,
2025-2026 and finally on the direct X-Perp route.  Entry geometry, market
execution, client position size and the frozen Profit-First exit stay fixed.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import pandas as pd

import crypto_high_target_tree_oct5_audit as base
import crypto_profit_first_exact_health_router_oct30_audit as health
import crypto_profit_first_material_target_oct18_audit as frozen
from crypto_runtime_shadow_portfolio_oct3_audit import money, runtime_gate


ROOT = Path("reports/audit_2026_09_08")
ROWS = ROOT / "_tmp_exact_historical_rows.pkl"
DIRECT = ROOT / "crypto_profit_first_exact_oct31_1700.pkl"
OUT = ROOT / "crypto_profit_first_static_veto_nov1.json"
SCOPES = (
    ("family",), ("direction",), ("btc_regime",), ("utc_session",),
    ("family", "direction"), ("direction", "btc_regime"),
    ("family", "btc_regime"), ("direction", "utc_session"),
    ("btc_regime", "utc_session"),
    ("family", "direction", "btc_regime"),
)
MIN_COUNTS = (10, 20, 40)
MAX_MEAN_R = (-.05, 0.0)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def accepted(frame: pd.DataFrame, module: str) -> pd.DataFrame:
    rows = runtime_gate(base.standardized(frame, module))
    return pd.DataFrame(rows)


def metrics(frame: pd.DataFrame, periods: tuple[str, ...],
            margin: float = .03) -> dict:
    selected = frame[frame.period.astype(str).isin(periods)]
    return money(selected.to_dict("records"), margin)


def direct_metrics(frame: pd.DataFrame, margin: float) -> dict:
    return money(frame.to_dict("records"), margin)


def compact(value: dict) -> dict:
    return {key: value[key] for key in (
        "n", "wr", "return_pct", "dd_pct", "max_loss_streak",
        "negative_months", "trades_per_calendar_day", "monthly_return_pct")}


def bucket_keys(frame: pd.DataFrame,
                scope: tuple[str, ...]) -> pd.Series:
    return pd.Series(
        list(map(tuple, frame[list(scope)].astype(str).to_numpy())),
        index=frame.index)


def weak_buckets(selection: pd.DataFrame, confirmation: pd.DataFrame,
                 scope: tuple[str, ...], minimum: int,
                 maximum_mean_r: float) -> set[tuple[str, ...]]:
    def summary(frame: pd.DataFrame) -> pd.DataFrame:
        return frame.groupby(list(scope), dropna=False).net_r.agg(
            count="size", mean="mean")

    joined = summary(selection).join(
        summary(confirmation), how="inner",
        lsuffix="_selection", rsuffix="_confirmation")
    weak = joined[
        (joined.count_selection >= minimum)
        & (joined.count_confirmation >= minimum)
        & (joined.mean_selection <= maximum_mean_r)
        & (joined.mean_confirmation <= maximum_mean_r)]
    if len(scope) == 1:
        return {(str(value),) for value in weak.index.tolist()}
    return {tuple(map(str, value)) for value in weak.index.tolist()}


def apply_veto(frame: pd.DataFrame, scope: tuple[str, ...],
               vetoed: set[tuple[str, ...]]) -> pd.DataFrame:
    return frame[~bucket_keys(frame, scope).isin(vetoed)].copy()


def development_pass(candidate: dict, baseline: dict) -> bool:
    return bool(
        candidate["return_pct"] >= baseline["return_pct"]
        and candidate["dd_pct"] <= 1.10 * baseline["dd_pct"]
        and candidate["negative_months"] <= baseline["negative_months"]
        and candidate["trades_per_calendar_day"]
            >= .80 * baseline["trades_per_calendar_day"])


def holdout_pass(candidate: dict, baseline: dict) -> bool:
    return bool(
        candidate["return_pct"] >= baseline["return_pct"]
        and candidate["dd_pct"] <= baseline["dd_pct"]
        and candidate["negative_months"] <= baseline["negative_months"])


def main() -> None:
    raw = pd.read_pickle(ROWS)
    exact = health.current_outcomes(raw)
    baseline = accepted(exact, "static_veto_baseline")
    periods = {
        "selection_2022h1": frozen.SELECT,
        "confirmation_2022h2_2023": frozen.CONFIRM,
        "validation_2024": frozen.VALIDATION,
        "later_2025_2026": frozen.LATER,
    }
    baseline_values = {
        name: metrics(baseline, period) for name, period in periods.items()}
    selection = exact[exact.period.astype(str).isin(frozen.SELECT)]
    confirmation = exact[exact.period.astype(str).isin(frozen.CONFIRM)]

    frontier = []
    candidates = {}
    for scope in SCOPES:
        for minimum in MIN_COUNTS:
            for maximum_mean_r in MAX_MEAN_R:
                vetoed = weak_buckets(
                    selection, confirmation, scope, minimum, maximum_mean_r)
                if not vetoed:
                    continue
                filtered = apply_veto(exact, scope, vetoed)
                candidate = accepted(filtered, "static_veto_candidate")
                selection_value = metrics(candidate, frozen.SELECT)
                confirmation_value = metrics(candidate, frozen.CONFIRM)
                passed = (
                    development_pass(
                        selection_value,
                        baseline_values["selection_2022h1"])
                    and development_pass(
                        confirmation_value,
                        baseline_values["confirmation_2022h2_2023"]))
                minimum_gain = min(
                    selection_value["return_pct"]
                        - baseline_values["selection_2022h1"]["return_pct"],
                    confirmation_value["return_pct"]
                        - baseline_values["confirmation_2022h2_2023"]["return_pct"])
                config = {
                    "scope": list(scope), "minimum_per_development_split": minimum,
                    "maximum_mean_r_each_split": maximum_mean_r,
                    "vetoed_buckets": [list(value) for value in sorted(vetoed)],
                }
                item = {
                    "config": config, "development_pass": passed,
                    "minimum_development_return_gain_pct": round(minimum_gain, 6),
                    "selection_2022h1": compact(selection_value),
                    "confirmation_2022h2_2023": compact(confirmation_value),
                }
                frontier.append(item)
                candidates[json.dumps(config, sort_keys=True)] = candidate

    eligible = [item for item in frontier if item["development_pass"]]
    eligible.sort(key=lambda item: (
        -item["minimum_development_return_gain_pct"],
        json.dumps(item["config"], sort_keys=True)))
    chosen = eligible[0] if eligible else None
    historical_holdouts = direct = None
    historical_pass = direct_pass = False
    if chosen:
        config = chosen["config"]
        candidate = candidates[json.dumps(config, sort_keys=True)]
        historical_holdouts = {
            name: metrics(candidate, periods[name])
            for name in ("validation_2024", "later_2025_2026")}
        historical_pass = all(holdout_pass(
            historical_holdouts[name], baseline_values[name])
            for name in historical_holdouts)

        direct_base = pd.read_pickle(DIRECT)
        direct_candidate = apply_veto(
            direct_base, tuple(config["scope"]),
            {tuple(value) for value in config["vetoed_buckets"]})
        direct = {
            "baseline_3pct": direct_metrics(direct_base, .03),
            "candidate_3pct": direct_metrics(direct_candidate, .03),
            "baseline_5pct": direct_metrics(direct_base, .05),
            "candidate_5pct": direct_metrics(direct_candidate, .05),
            "post_filter_only": True,
        }
        direct_pass = bool(
            direct["candidate_3pct"]["return_pct"]
                >= direct["baseline_3pct"]["return_pct"]
            and direct["candidate_3pct"]["dd_pct"]
                <= direct["baseline_3pct"]["dd_pct"])

    robust = bool(chosen and historical_pass and direct_pass)
    report = {
        "status": ("STATIC_VETO_SHADOW_CHALLENGER_AWAIT_FORWARD" if robust
                   else "REJECTED_STATIC_VETO_HOLDOUT" if chosen
                   else "REJECTED_STATIC_VETO_DEVELOPMENT"),
        "ready_for_live_orders": False,
        "orders_possible": False,
        "runtime_change_allowed": False,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "protocol": {
            "entry_exit": "frozen exact Profit-First route",
            "selection": [*frozen.SELECT, *frozen.CONFIRM],
            "sealed_holdouts": [*frozen.VALIDATION, *frozen.LATER, "direct X-Perp"],
            "scopes": [list(value) for value in SCOPES],
            "minimum_counts": list(MIN_COUNTS),
            "maximum_mean_r_each_split": list(MAX_MEAN_R),
            "position_size": "client setting",
            "orders": "market only",
        },
        "candidate_count": len(frontier),
        "development_pass_count": len(eligible),
        "baseline": {name: compact(value)
                     for name, value in baseline_values.items()},
        "selected_before_holdouts": chosen,
        "historical_holdouts": ({name: compact(value)
                                 for name, value in historical_holdouts.items()}
                                if historical_holdouts else None),
        "historical_holdout_pass": historical_pass,
        "direct_xperp": ({key: compact(value) if isinstance(value, dict) else value
                          for key, value in direct.items()} if direct else None),
        "direct_pass": direct_pass,
        "robust_improvement": robust,
        "frontier": sorted(frontier, key=lambda item: (
            item["development_pass"],
            item["minimum_development_return_gain_pct"]), reverse=True),
        "sources": {str(path): digest(path) for path in (
            ROWS, DIRECT, Path(__file__).resolve())},
        "limitations": [
            "Direct evaluation is a conservative post-filter and does not refill freed caps.",
            "A passing retrospective rule would still require genuinely new forward outcomes.",
            "Gross returns exclude fees at user request.",
        ],
    }
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({
        "status": report["status"], "selected": chosen,
        "baseline_holdouts": {
            name: compact(baseline_values[name])
            for name in ("validation_2024", "later_2025_2026")},
        "historical_holdouts": report["historical_holdouts"],
        "direct": report["direct_xperp"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
