"""Expanding causal walk-forward for the frozen material crypto policy.

The trade geometry, tree configuration, score quantile and transparent expert
fields are inherited unchanged from the preceding audits.  For target year Y,
the model fits through Y-2, calibrates its score threshold on Y-1 without using
Y-1 labels, and trades Y.  Starting in 2024, expert groups must have positive
realized expectancy in each of the two preceding causal out-of-sample folds.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

import crypto_multiday_material_trend_oct21_audit as features_base
import crypto_raw_material_direction_model_oct21_audit as model_base
import crypto_raw_material_regime_experts_oct21_audit as expert_base


OUT = features_base.ROOT / "crypto_raw_material_walkforward_oct21.json"
ROWS_OUT = features_base.ROOT / "crypto_raw_material_walkforward_oct21.pkl"
TARGET_YEARS = (2022, 2023, 2024, 2025, 2026)
ROUTED_YEARS = (2024, 2025, 2026)
EXPERT_FIELDS = (
    "direction_name", "score_regime", "breadth_alignment",
    "dispersion_regime",
)
MINIMUM_COUNT = 30
MINIMUM_MEAN = 0.0


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fold_years(target_year: int) -> tuple[tuple[int, ...], int]:
    if target_year < 2022:
        raise ValueError("walk-forward starts in 2022")
    return tuple(range(2020, target_year - 1)), target_year - 1


def attach_context(labeled: pd.DataFrame,
                   directional: pd.DataFrame) -> pd.DataFrame:
    identity = ["symbol", "entry_index", "entry_time", "year", "month",
                "direction"]
    context = directional[[
        *identity, "market_ret24", "breadth24", "dispersion24",
        "volume_ratio",
    ]]
    return labeled.merge(
        context, on=identity, how="left", validate="one_to_one")


def stable_experts(frame: pd.DataFrame, years: tuple[int, int]) -> pd.DataFrame:
    stats = frame[frame.year.isin(years)].groupby(
        ["year", *EXPERT_FIELDS], dropna=False).signed_return.agg(
            count="size", mean="mean",
            win_rate=lambda values: float((values > 0).mean())).reset_index()
    good = stats[
        stats["count"].ge(MINIMUM_COUNT)
        & stats["mean"].ge(MINIMUM_MEAN)]
    stable = good.groupby(list(EXPERT_FIELDS), dropna=False).filter(
        lambda part: set(part.year.astype(int)) == set(years))
    if stable.empty:
        return pd.DataFrame()
    return stable.groupby(list(EXPERT_FIELDS), dropna=False).agg(
        worst_year_mean=("mean", "min"),
        average_mean=("mean", "mean"),
        worst_year_win_rate=("win_rate", "min"),
        observations=("count", "sum")).reset_index()


def route(frame: pd.DataFrame, experts: pd.DataFrame,
          year: int) -> pd.DataFrame:
    if experts.empty:
        return frame.iloc[:0].copy()
    selected = frame.merge(
        experts[[*EXPERT_FIELDS, "worst_year_mean", "average_mean"]],
        on=list(EXPERT_FIELDS), how="inner", validate="many_to_one")
    selected["model_probability"] = (
        selected.model_probability
        + 100 * selected.worst_year_mean.clip(lower=0))
    selected["spec_name"] = f"walkforward_regime_expert_{year}"
    return selected


def compact(stage: dict) -> dict:
    value = stage["margin_3pct"]
    return {
        key: value[key] for key in (
            "n", "wr", "return_pct", "dd_pct", "mtm_dd_pct",
            "negative_months", "monthly_return_pct",
            "trades_per_calendar_day", "literal_target_rate_pct",
            "mean_signed_price_return_pct", "projected_30_day_pct",
            "max_loss_streak", "max_concurrent_positions",
            "max_gross_notional_equity_pct")
    }


def robust(stage: dict) -> bool:
    value = stage["margin_3pct"]
    return bool(
        value["n"] >= 700 and value["return_pct"] > 0
        and value["trades_per_calendar_day"] >= 2.5
        and max(value["dd_pct"], value["mtm_dd_pct"]) <= 10)


def main() -> None:
    policy = expert_base.frozen()
    candles, sources = features_base.load()
    features = features_base.build_features(candles)
    directional = model_base.directional_frame(features)
    design = model_base.schema(directional)
    x = model_base.matrix(directional, design)
    geometry = (
        float(policy["geometry"]["target_pct"]),
        float(policy["geometry"]["stop_pct"]),
        int(policy["geometry"]["hold_bars_5m"]),
    )
    labeled = attach_context(
        model_base.label_identity(directional, candles, geometry), directional)
    config = policy["config"]
    config_tuple = (
        int(config["max_leaf_nodes"]), int(config["min_samples_leaf"]),
        float(config["l2_regularization"]),
    )

    causal_frames: dict[int, pd.DataFrame] = {}
    folds: dict[str, dict] = {}
    routed_frames = []
    accepted_rows = []
    for target_year in TARGET_YEARS:
        fit_years, calibration_year = fold_years(target_year)
        fit_mask = directional.year.isin(fit_years).to_numpy()
        fitted = model_base.fit_model(
            x, labeled, fit_mask, str(policy["model_target"]), config_tuple)
        scores = model_base.predict(
            fitted, x, str(policy["model_target"]))
        calibration_mask = directional.year.eq(calibration_year).to_numpy()
        threshold = float(np.quantile(
            scores[calibration_mask], float(policy["selection_quantile"])))
        scored = model_base.scored_frame(
            labeled, scores, f"walkforward_base_{target_year}")
        current = scored[
            scored.year.eq(target_year)
            & scored.model_probability.ge(threshold)].copy()
        current.sort_values(
            ["symbol", "entry_time", "model_probability", "direction"],
            ascending=[True, True, False, False], inplace=True,
            kind="mergesort")
        current.drop_duplicates(["symbol", "entry_time"], inplace=True)

        # Quantile cuts are recomputed from the preceding causal folds only.
        if causal_frames:
            prior = pd.concat(list(causal_frames.values()), ignore_index=True)
            cuts = {
                "dispersion24": [float(value) for value in
                    prior.dispersion24.quantile([1/3, 2/3])],
                "volume_ratio": [float(value) for value in
                    prior.volume_ratio.quantile([1/3, 2/3])],
                "model_probability": [float(value) for value in
                    prior.model_probability.quantile([1/3, 2/3])],
            }
        else:
            cuts = {
                "dispersion24": [0.0, 1.0], "volume_ratio": [0.0, 1.0],
                "model_probability": [0.0, 1.0],
            }
        enriched = expert_base.enrich(current, cuts)
        causal_frames[target_year] = enriched
        base_stage, _ = model_base.exact_stage(
            enriched, (target_year,), candles)

        experts = pd.DataFrame()
        traded = enriched
        expert_years = None
        if target_year in ROUTED_YEARS:
            expert_years = (target_year - 2, target_year - 1)
            history = pd.concat(
                [causal_frames[year] for year in expert_years],
                ignore_index=True)
            experts = stable_experts(history, expert_years)
            traded = route(enriched, experts, target_year)
            routed_frames.append(traded)
        routed_stage, rows = model_base.exact_stage(
            traded, (target_year,), candles)
        if target_year in ROUTED_YEARS:
            accepted_rows.extend(rows)
        folds[str(target_year)] = {
            "fit_years": list(fit_years),
            "calibration_year": calibration_year,
            "threshold": threshold,
            "expert_years": list(expert_years) if expert_years else None,
            "expert_count": int(len(experts)),
            "base": compact(base_stage),
            "routed": compact(routed_stage),
            "routed_robust_gate": robust(routed_stage),
        }
        print(json.dumps({
            "target_year": target_year,
            "fit_years": fit_years,
            "threshold": threshold,
            "expert_count": len(experts),
            "base": compact(base_stage),
            "routed": compact(routed_stage),
        }), flush=True)

    combined_frame = pd.concat(routed_frames, ignore_index=True)
    combined, combined_rows = model_base.exact_stage(
        combined_frame, ROUTED_YEARS, candles)
    pd.DataFrame(combined_rows).to_pickle(ROWS_OUT)
    objective = all(
        model_base.objective_ok({
            "margin_3pct": folds[str(year)]["routed"],
            "annual_3pct": {},
        }) for year in ROUTED_YEARS)
    all_robust = all(
        folds[str(year)]["routed_robust_gate"] for year in ROUTED_YEARS)
    report = {
        "status": ("WALK_FORWARD_CANDIDATE" if all_robust
                   else "REJECTED_WALK_FORWARD"),
        "ready_for_live_orders": False,
        "orders_possible": False,
        "protocol": {
            "target_years": list(TARGET_YEARS),
            "routed_evaluation_years": list(ROUTED_YEARS),
            "fit_rule": "all completed years through target minus two",
            "threshold_rule": "frozen 0.925 score quantile in prior year",
            "expert_rule": (
                "same frozen direction/score/breadth/dispersion group must "
                "have at least 30 trades and nonnegative expectancy in each "
                "of the two preceding causal folds"),
            "entry": "next five-minute market open",
            "exit": "2% target, 1% stop, maximum 24 hours; stop first",
            "money": "client-selected 3% current-equity margin; fixed 10x",
            "drawdown": "five-minute marked concurrent equity",
        },
        "frozen_policy": policy,
        "folds": folds,
        "combined_routed_2024_2026": combined,
        "all_routed_years_pass_robust_gate": all_robust,
        "all_routed_years_meet_monthly_objective": objective,
        "sources": {
            **sources,
            str(model_base.OUT): digest(model_base.OUT),
            str(expert_base.OUT): digest(expert_base.OUT),
            str(Path(model_base.__file__).resolve()):
                digest(Path(model_base.__file__).resolve()),
            str(Path(expert_base.__file__).resolve()):
                digest(Path(expert_base.__file__).resolve()),
            str(Path(__file__).resolve()): digest(Path(__file__).resolve()),
        },
        "limitations": [
            "The walk-forward is causal, but this policy was designed after "
            "earlier research had already exposed every historical year.",
            "A new shadow-forward period is required before live activation.",
            "Binance spot history remains a proxy for X-Perp execution.",
            "The 2026 archive ends in September.",
            "Gross results exclude fees at user request.",
        ],
    }
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({
        "status": report["status"],
        "combined": compact(combined),
        "all_robust": all_robust,
        "all_monthly_objective": objective,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
