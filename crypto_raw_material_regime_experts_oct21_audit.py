"""Causal regime experts for the frozen raw material crypto model.

The base 2%/1% model and score threshold are frozen from the preceding audit.
Experts are learned only from 2022 and 2023 groups that have positive raw
expectancy in each year separately.  One policy is selected before 2024; the
2025-2026 labels remain sealed unless that policy passes the 2024 gate.
"""
from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

import crypto_multiday_material_trend_oct21_audit as features_base
import crypto_raw_material_direction_model_oct21_audit as model_base


OUT = features_base.ROOT / "crypto_raw_material_regime_experts_oct21.json"
ROWS_OUT = features_base.ROOT / "crypto_raw_material_regime_experts_oct21.pkl"
DEVELOPMENT = (2022, 2023)
TEST = (2024,)
HIDDEN = (2025, 2026)
FIELD_SETS = (
    ("direction_name", "market_trend_alignment", "volatility_regime"),
    ("direction_name", "breadth_alignment", "dispersion_regime"),
    ("direction_name", "symbol", "market_trend_alignment"),
    ("direction_name", "utc_session", "market_trend_alignment"),
    ("direction_name", "score_regime", "market_trend_alignment"),
    ("direction_name", "score_regime", "breadth_alignment",
     "dispersion_regime"),
    ("direction_name", "symbol", "score_regime"),
)
MIN_COUNTS = (30, 75)
MIN_MEANS = (0.0, .00025, .0005)
TOP_EXACT = 10


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def frozen() -> dict:
    report = json.loads(model_base.OUT.read_text(encoding="utf-8"))
    selected = report["selected_before_test"]
    expected = {
        "model_id": "tp2_sl1_h288_literal_target_l31_n800",
        "geometry": {"target_pct": 2.0, "stop_pct": 1.0,
                     "hold_bars_5m": 288},
        "model_target": "literal_target",
        "config": {"max_leaf_nodes": 31, "min_samples_leaf": 800,
                   "l2_regularization": 10.0},
        "selection_quantile": .925,
        "threshold": 0.3705791593304682,
    }
    found = {key: selected[key] for key in expected}
    if found != expected:
        raise ValueError(f"frozen base model drifted: {found}")
    return expected


def rebuild_scored(candles: dict, features: pd.DataFrame,
                   years: tuple[int, ...], policy: dict,
                   fitted=None, design=None) -> tuple[pd.DataFrame, object, dict]:
    directional = model_base.directional_frame(
        features[features.year.isin(years)].copy())
    if design is None:
        # A fit design needs 2020-2021.  Callers pass the full design after the
        # first rebuild; hidden/test frames must never refit preprocessing.
        design = model_base.schema(directional)
    x = model_base.matrix(directional, design)
    geometry = (
        float(policy["geometry"]["target_pct"]),
        float(policy["geometry"]["stop_pct"]),
        int(policy["geometry"]["hold_bars_5m"]),
    )
    labeled = model_base.label_identity(directional, candles, geometry)
    # label_identity resolves outcomes symbol by symbol, so its output order is
    # not guaranteed to match the feature frame.  Join causal signal-time
    # fields back by the immutable trade identity instead of assigning them by
    # position.
    identity = ["symbol", "entry_index", "entry_time", "year", "month",
                "direction"]
    context = directional[[
        *identity, "market_ret24", "breadth24", "dispersion24",
        "volume_ratio",
    ]]
    labeled = labeled.merge(
        context, on=identity, how="left", validate="one_to_one")
    if fitted is None:
        config = policy["config"]
        fitted = model_base.fit_model(
            x, labeled, directional.year.isin(model_base.FIT).to_numpy(),
            str(policy["model_target"]),
            (int(config["max_leaf_nodes"]), int(config["min_samples_leaf"]),
             float(config["l2_regularization"])))
    scores = model_base.predict(fitted, x, str(policy["model_target"]))
    scored = model_base.scored_frame(labeled, scores, str(policy["model_id"]))
    scored = scored[scored.model_probability.ge(
        float(policy["threshold"]))].copy()
    scored.sort_values(
        ["symbol", "entry_time", "model_probability", "direction"],
        ascending=[True, True, False, False], inplace=True, kind="mergesort")
    scored.drop_duplicates(["symbol", "entry_time"], inplace=True)
    return scored.reset_index(drop=True), fitted, design


def thresholds(frame: pd.DataFrame) -> dict:
    development = frame[frame.year.isin(DEVELOPMENT)]
    return {
        "dispersion24": [float(value) for value in
                         development.dispersion24.quantile([1/3, 2/3])],
        "volume_ratio": [float(value) for value in
                         development.volume_ratio.quantile([1/3, 2/3])],
        "model_probability": [float(value) for value in
                              development.model_probability.quantile([1/3, 2/3])],
    }


def enrich(frame: pd.DataFrame, cuts: dict) -> pd.DataFrame:
    result = frame.copy()
    direction = result.direction.to_numpy(float)
    result["direction_name"] = np.where(direction > 0, "LONG", "SHORT")
    result["market_trend_alignment"] = pd.cut(
        direction * result.market_ret24,
        [-np.inf, -.005, .005, np.inf],
        labels=["adverse", "flat", "supportive"]).astype(str)
    aligned_breadth = np.where(
        direction > 0, result.breadth24, 1 - result.breadth24)
    result["breadth_alignment"] = pd.cut(
        aligned_breadth, [-np.inf, .4, .6, np.inf],
        labels=["adverse", "mixed", "supportive"]).astype(str)
    result["dispersion_regime"] = pd.cut(
        result.dispersion24,
        [-np.inf, *cuts["dispersion24"], np.inf],
        labels=["low", "normal", "high"]).astype(str)
    result["volatility_regime"] = pd.cut(
        result.volume_ratio,
        [-np.inf, *cuts["volume_ratio"], np.inf],
        labels=["low", "normal", "high"]).astype(str)
    result["score_regime"] = pd.cut(
        result.model_probability,
        [-np.inf, *cuts["model_probability"], np.inf],
        labels=["low", "normal", "high"]).astype(str)
    hour = pd.to_datetime(
        result.entry_time, unit="s", utc=True).dt.hour
    result["utc_session"] = pd.cut(
        hour, [-1, 6, 14, 21, 24], right=False,
        labels=["asia", "europe", "us", "late"]).astype(str)
    return result


def stable_experts(frame: pd.DataFrame, fields: tuple[str, ...],
                   minimum_count: int, minimum_mean: float) -> pd.DataFrame:
    stats = frame[frame.year.isin(DEVELOPMENT)].groupby(
        ["year", *fields], dropna=False).signed_return.agg(
            count="size", mean="mean",
            win_rate=lambda values: float((values > 0).mean())).reset_index()
    good = stats[
        stats["count"].ge(minimum_count)
        & stats["mean"].ge(minimum_mean)]
    if good.empty:
        return pd.DataFrame()
    stable = good.groupby(list(fields), dropna=False).filter(
        lambda part: set(part.year.astype(int)) == set(DEVELOPMENT))
    if stable.empty:
        return pd.DataFrame()
    return stable.groupby(list(fields), dropna=False).agg(
        worst_year_mean=("mean", "min"),
        average_mean=("mean", "mean"),
        worst_year_win_rate=("win_rate", "min"),
        observations=("count", "sum")).reset_index()


def route(frame: pd.DataFrame, fields: tuple[str, ...],
          experts: pd.DataFrame, name: str) -> pd.DataFrame:
    if experts.empty:
        return frame.iloc[:0].copy()
    selected = frame.merge(
        experts[[*fields, "worst_year_mean", "average_mean"]],
        on=list(fields), how="inner", validate="many_to_one")
    selected["model_probability"] = (
        selected.model_probability
        + 100 * selected.worst_year_mean.clip(lower=0))
    selected["spec_name"] = name
    return selected


def month_floor(stage: dict) -> float:
    months = stage["margin_3pct"].get("monthly_return_pct", {})
    return min(months.values(), default=-999.0)


def rank(stage: dict) -> tuple:
    value = stage["margin_3pct"]
    return (
        value["trades_per_calendar_day"] >= 3
        and value["return_pct"] > 0 and value["dd_pct"] <= 12,
        -int(value["negative_months"]), month_floor(stage),
        float(value["projected_30_day_pct"]),
        float(value["return_pct"]), -float(value["dd_pct"]),
    )


def exact_rank(stage: dict) -> tuple:
    value = stage["margin_3pct"]
    return (
        value["trades_per_calendar_day"] >= 3
        and value["return_pct"] > 0
        and max(value["dd_pct"], value["mtm_dd_pct"]) <= 12,
        -int(value["negative_months"]), month_floor(stage),
        float(value["projected_30_day_pct"]),
        float(value["return_pct"]),
        -max(float(value["dd_pct"]), float(value["mtm_dd_pct"])),
    )


def test_ok(stage: dict) -> bool:
    value = stage["margin_3pct"]
    return bool(
        value["n"] >= 900 and value["return_pct"] > 0
        and value["trades_per_calendar_day"] >= 2.5
        and value["negative_months"] == 0
        and max(value["dd_pct"], value["mtm_dd_pct"]) <= 10)


def public(item: dict) -> dict:
    return {key: value for key, value in item.items()
            if key not in ("_experts", "_frame", "_accepted")}


def main() -> None:
    policy = frozen()
    candles, sources = features_base.load()
    features = features_base.build_features(candles)
    # Include fit years to reconstruct the frozen model, but no post-2024 rows.
    prehidden, fitted, design = rebuild_scored(
        candles, features, (*model_base.FIT, *DEVELOPMENT, *TEST), policy)
    cuts = thresholds(prehidden)
    frame = enrich(prehidden, cuts)

    candidates = []
    for fields, minimum_count, minimum_mean in itertools.product(
            FIELD_SETS, MIN_COUNTS, MIN_MEANS):
        experts = stable_experts(
            frame, fields, minimum_count, minimum_mean)
        name = ("regime:" + "|".join(fields)
                + f":n{minimum_count}:m{minimum_mean}")
        selected = route(frame, fields, experts, name)
        stage, _ = features_base.exact_stage(selected, DEVELOPMENT)
        candidates.append({
            "fields": list(fields), "minimum_count_each_year": minimum_count,
            "minimum_mean_each_year_pct": 100 * minimum_mean,
            "expert_count": len(experts), "development_2022_2023": stage,
            "_experts": experts, "_frame": selected,
        })
    candidates.sort(
        key=lambda item: rank(item["development_2022_2023"]), reverse=True)

    replayed = []
    for item in candidates[:TOP_EXACT]:
        stage, rows = model_base.exact_stage(
            item["_frame"], DEVELOPMENT, candles)
        replayed.append({**item, "development_2022_2023": stage,
                         "_accepted": rows})
    replayed.sort(
        key=lambda item: exact_rank(item["development_2022_2023"]),
        reverse=True)
    selected = replayed[0] if replayed else None

    test = hidden = None
    accepted_rows = selected["_accepted"] if selected else []
    status = "REJECTED_DEVELOPMENT"
    if selected:
        status = "REJECTED_TEST"
        test_frame = selected["_frame"][
            selected["_frame"].year.isin(TEST)]
        test, test_rows = model_base.exact_stage(test_frame, TEST, candles)
        accepted_rows = test_rows
        if test_ok(test):
            hidden_scored, _, _ = rebuild_scored(
                candles, features, HIDDEN, policy,
                fitted=fitted, design=design)
            hidden_enriched = enrich(hidden_scored, cuts)
            hidden_selected = route(
                hidden_enriched, tuple(selected["fields"]),
                selected["_experts"], "frozen_regime_expert")
            hidden, accepted_rows = model_base.exact_stage(
                hidden_selected, HIDDEN, candles)
            status = ("REGIME_EXPERT_CANDIDATE" if test_ok(hidden)
                      else "REJECTED_HIDDEN")

    pd.DataFrame(accepted_rows).to_pickle(ROWS_OUT)
    objective = bool(
        selected and test and hidden
        and model_base.objective_ok(selected["development_2022_2023"])
        and model_base.objective_ok(test)
        and model_base.objective_ok(hidden))
    report = {
        "status": status, "ready_for_live_orders": False,
        "orders_possible": False,
        "protocol": {
            "frozen_base": policy,
            "expert_learning": list(DEVELOPMENT),
            "untouched_test": list(TEST), "hidden_reveal": list(HIDDEN),
            "expert_rule": (
                "positive raw expectancy and minimum count separately in "
                "both 2022 and 2023"),
            "regime_thresholds": "quantiles frozen on 2022-2023",
            "entry": "next five-minute market open",
            "orders": "market only", "collision": "stop first",
            "money": "client-selected 3% margin; fixed 10x",
            "drawdown": "five-minute marked equity for concurrent positions",
        },
        "regime_thresholds": cuts,
        "grid": {
            "field_sets": len(FIELD_SETS),
            "candidate_policies": len(candidates),
            "exact_replayed": len(replayed),
        },
        "selected_before_test": public(selected) if selected else None,
        "test_2024": test, "hidden_2025_2026": hidden,
        "all_stages_meet_monthly_objective": objective,
        "top_development": [public(item) for item in replayed[:10]],
        "sources": {
            **sources,
            str(model_base.OUT): digest(model_base.OUT),
            str(Path(model_base.__file__).resolve()):
                digest(Path(model_base.__file__).resolve()),
            str(Path(__file__).resolve()): digest(Path(__file__).resolve()),
        },
        "limitations": [
            "Regime experts are a second selection layer and require a strict holdout.",
            "Binance spot history remains a proxy for X-Perp execution.",
            "The 2026 archive ends in September.",
            "Gross results exclude fees at user request.",
        ],
    }
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({
        "status": status, "grid": report["grid"],
        "selected": report["selected_before_test"],
        "test": test, "hidden": hidden,
        "all_stages_meet_monthly_objective": objective,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
