"""Causal loss-cause experts for the frozen 1.5% material crypto model."""
from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

import crypto_5m_material_regime_model_oct22_audit as base
import crypto_5m_regime_router_literal_oct6_audit as price
import crypto_raw_material_direction_model_oct21_audit as metrics


OUT = price.ROOT / "crypto_5m_material_loss_cause_oct22.json"
ROWS_OUT = price.ROOT / "crypto_5m_material_loss_cause_oct22.pkl"
DEVELOPMENT = (2022, 2023)
TEST = (2024,)
HIDDEN = (2025, 2026)
FIELDS = (
    ("direction_name", "primary_family", "btc_alignment"),
    ("direction_name", "utc_session", "btc_alignment"),
    ("primary_family", "btc_alignment", "breakout_context"),
    ("direction_name", "relative_alignment", "volatility_regime"),
    ("primary_family", "score_regime", "btc_alignment"),
    ("direction_name", "wick_regime", "breakout_context"),
)
MIN_COUNTS = (30, 75)
MIN_MEANS = (0.0, .00025)
TOP_EXACT = 8


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def frozen() -> dict:
    report = json.loads(base.OUT.read_text(encoding="utf-8"))
    selected = report["selected_before_test"]
    expected = {
        "model_id": "tp1.5_sl1_h288_literal_target_l31_n600",
        "geometry": {"target_pct": 1.5, "stop_pct": 1.0,
                     "hold_bars_5m": 288},
        "model_target": "literal_target",
        "config": {"max_leaf_nodes": 31, "min_samples_leaf": 600,
                   "l2_regularization": 10.0},
        "selection_quantile": .8,
        "threshold": 0.43479755868469755,
    }
    found = {key: selected[key] for key in expected}
    if found != expected:
        raise ValueError(f"frozen material model drifted: {found}")
    return expected


def rebuild(candles: dict, policy: dict) -> pd.DataFrame:
    features = price.build_features(candles)
    proposals = price.proposals(features).reset_index(drop=True)
    geometry = (policy["geometry"]["target_pct"],
                policy["geometry"]["stop_pct"],
                policy["geometry"]["hold_bars_5m"])
    frame = base.add_outcomes(proposals, candles, geometry)
    schema = price.columns(frame[frame.year.isin(base.FIT)])
    x = price.matrix(frame, schema)
    config = policy["config"]
    fitted = base.fit_model(
        x, frame, frame.year.isin(base.FIT).to_numpy(),
        policy["model_target"],
        (config["max_leaf_nodes"], config["min_samples_leaf"],
         config["l2_regularization"]))
    scores = base.predict(fitted, x, policy["model_target"])
    return base.scored(
        frame, scores, policy["model_id"], policy["threshold"])


def cuts(frame: pd.DataFrame) -> dict:
    part = frame[frame.year.isin(DEVELOPMENT)]
    return {
        name: [float(x) for x in part[name].quantile([1/3, 2/3])]
        for name in ("range_ratio", "eff72", "model_probability")
    }


def enrich(frame: pd.DataFrame, thresholds: dict) -> pd.DataFrame:
    result = frame.copy()
    direction = result.direction.to_numpy(float)
    result["direction_name"] = np.where(direction > 0, "LONG", "SHORT")
    flags = [f"family_{name}" for name in price.FAMILIES]
    result["primary_family"] = result[flags].apply(
        lambda row: "+".join(name for name, flag in zip(price.FAMILIES, row)
                             if flag) or "none", axis=1)
    hour = pd.to_datetime(result.entry_time, unit="s", utc=True).dt.hour
    result["utc_session"] = pd.cut(
        hour, [-1, 6, 14, 21, 24], right=False,
        labels=["asia", "europe", "us", "late"]).astype(str)
    btc = direction * result.btc_ret72
    result["btc_alignment"] = pd.cut(
        btc, [-np.inf, -.004, .004, np.inf],
        labels=["adverse", "flat", "supportive"]).astype(str)
    relative = direction * (result.ret72 - result.btc_ret72)
    result["relative_alignment"] = pd.cut(
        relative, [-np.inf, -.004, .004, np.inf],
        labels=["lagging", "neutral", "leading"]).astype(str)
    breakout = np.where(direction > 0, result.breakout_up, result.breakout_down)
    result["breakout_context"] = pd.cut(
        breakout, [-np.inf, -.002, 0, np.inf],
        labels=["inside", "testing", "broken"]).astype(str)
    result["volatility_regime"] = pd.cut(
        result.range_ratio, [-np.inf, *thresholds["range_ratio"], np.inf],
        labels=["low", "normal", "high"]).astype(str)
    result["score_regime"] = pd.cut(
        result.model_probability,
        [-np.inf, *thresholds["model_probability"], np.inf],
        labels=["low", "normal", "high"]).astype(str)
    directed_wick = np.where(
        direction > 0, result.lower_wick, result.upper_wick)
    result["wick_regime"] = pd.cut(
        directed_wick, [-np.inf, .2, .5, np.inf],
        labels=["weak", "normal", "strong"]).astype(str)
    return result


def experts(frame: pd.DataFrame, fields: tuple[str, ...], minimum_count: int,
            minimum_mean: float) -> pd.DataFrame:
    stats = frame[frame.year.isin(DEVELOPMENT)].groupby(
        ["year", *fields], dropna=False).signed_return.agg(
            count="size", mean="mean",
            win_rate=lambda x: float((x > 0).mean())).reset_index()
    good = stats[stats["count"].ge(minimum_count)
                 & stats["mean"].ge(minimum_mean)]
    stable = good.groupby(list(fields), dropna=False).filter(
        lambda part: set(part.year.astype(int)) == set(DEVELOPMENT))
    if stable.empty:
        return pd.DataFrame()
    return stable.groupby(list(fields), dropna=False).agg(
        worst_year_mean=("mean", "min"), average_mean=("mean", "mean"),
        worst_year_win_rate=("win_rate", "min"),
        observations=("count", "sum")).reset_index()


def route(frame: pd.DataFrame, fields: tuple[str, ...], allowed: pd.DataFrame,
          name: str) -> pd.DataFrame:
    if allowed.empty:
        return frame.iloc[:0].copy()
    selected = frame.merge(
        allowed[[*fields, "worst_year_mean", "average_mean"]],
        on=list(fields), how="inner", validate="many_to_one")
    selected["model_probability"] += 100 * selected.worst_year_mean
    selected["spec_name"] = name
    return selected


def rank(stage: dict) -> tuple:
    value = stage["margin_3pct"]
    floor = min(value.get("monthly_return_pct", {}).values(), default=-999.)
    return (value["trades_per_calendar_day"] >= 2.5,
            value["projected_30_day_pct"], -value["negative_months"],
            floor, value["return_pct"], -value["dd_pct"])


def test_ok(stage: dict) -> bool:
    value = stage["margin_3pct"]
    return bool(value["n"] >= 700 and value["return_pct"] > 0
                and value["trades_per_calendar_day"] >= 2.5
                and value["negative_months"] == 0
                and max(value["dd_pct"], value["mtm_dd_pct"]) <= 10)


def public(item: dict) -> dict:
    return {key: value for key, value in item.items()
            if not key.startswith("_")}


def main() -> None:
    policy = frozen()
    candles, sources = price.base.load()
    scored = rebuild(candles, policy)
    thresholds = cuts(scored)
    frame = enrich(scored, thresholds)
    candidates = []
    for fields, minimum_count, minimum_mean in itertools.product(
            FIELDS, MIN_COUNTS, MIN_MEANS):
        allowed = experts(frame, fields, minimum_count, minimum_mean)
        name = ("loss_cause:" + "|".join(fields)
                + f":n{minimum_count}:m{minimum_mean}")
        selected = route(frame, fields, allowed, name)
        stage, _ = base.stage(selected, DEVELOPMENT, candles, marked=False)
        candidates.append({
            "fields": list(fields), "minimum_count_each_year": minimum_count,
            "minimum_mean_each_year_pct": 100 * minimum_mean,
            "expert_count": len(allowed), "development": stage,
            "_experts": allowed, "_frame": selected,
        })
    candidates.sort(key=lambda item: rank(item["development"]), reverse=True)
    replayed = []
    for item in candidates[:TOP_EXACT]:
        stage, rows = base.stage(item["_frame"], DEVELOPMENT, candles,
                                 marked=True)
        replayed.append({**item, "development": stage, "_rows": rows})
    replayed.sort(key=lambda item: rank(item["development"]), reverse=True)
    selected = replayed[0] if replayed else None
    test = hidden = None
    rows = selected["_rows"] if selected else []
    status = "REJECTED_DEVELOPMENT"
    if selected:
        test, rows = base.stage(selected["_frame"], TEST, candles, marked=True)
        status = "REJECTED_TEST"
        if test_ok(test):
            hidden, rows = base.stage(
                selected["_frame"], HIDDEN, candles, marked=True)
            status = ("LOSS_CAUSE_SHADOW_CANDIDATE" if test_ok(hidden)
                      else "REJECTED_HIDDEN")
    pd.DataFrame(rows).to_pickle(ROWS_OUT)
    objective = bool(selected and test and hidden
                     and metrics.objective_ok(selected["development"])
                     and metrics.objective_ok(test)
                     and metrics.objective_ok(hidden))
    report = {
        "status": status, "ready_for_live_orders": False,
        "orders_possible": False,
        "protocol": {"frozen_base": policy,
                     "expert_learning": list(DEVELOPMENT),
                     "untouched_test": list(TEST),
                     "hidden_reveal": list(HIDDEN),
                     "entry": "next five-minute market open",
                     "orders": "market only",
                     "money": "client-selected 3% current-equity margin; fixed 10x"},
        "grid": {"base_scored_rows": len(scored),
                 "candidate_policies": len(candidates),
                 "exact_replayed": len(replayed)},
        "thresholds": thresholds,
        "selected_before_test": public(selected) if selected else None,
        "test_2024": test, "hidden_2025_2026": hidden,
        "all_stages_meet_monthly_objective": objective,
        "top_development": [public(item) for item in replayed],
        "sources": {**sources, str(base.OUT): digest(base.OUT),
                    str(Path(base.__file__).resolve()): digest(Path(base.__file__).resolve()),
                    str(Path(__file__).resolve()): digest(Path(__file__).resolve())},
        "limitations": ["Post-2024 history remains sealed unless the 2024 gate passes.",
                        "Binance spot history is a proxy for X-Perp execution.",
                        "Gross results exclude fees at user request."],
    }
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"status": status, "grid": report["grid"],
                      "selected": report["selected_before_test"],
                      "test": test, "hidden": hidden,
                      "objective": objective}, indent=2), flush=True)


if __name__ == "__main__":
    main()
