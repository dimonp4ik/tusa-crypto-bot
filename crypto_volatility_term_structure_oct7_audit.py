"""Causal literal-target audit of multi-horizon volatility term structure."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

import crypto_5m_regime_router_literal_oct6_audit as price
import crypto_futures_trade_flow_regime_oct7_audit as source


OUT = Path("reports/audit_2026_09_08/crypto_volatility_term_structure_oct7.json")
FAMILIES = ("compression_release", "controlled_expansion",
            "volatility_exhaustion", "relative_vol_leader", "calm_trend_pullback")
VOL_FEATURES = (
    "rv6", "rv12", "rv72", "rv288", "rv2016",
    "fast_day", "hour_day", "day_week", "vol_acceleration",
    "term_slope", "term_curvature", "vol_cross_z",
    "ret12_dispersion", "vol_breadth_expanding", "release_ratio3",
)
CONFIGS = (
    {"leaves": 15, "leaf": 250, "l2": 3.0},
    {"leaves": 31, "leaf": 400, "l2": 5.0},
)
QUANTILES = (.75, .80, .85, .90, .925, .95, .96, .97, .98, .985, .99)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def symbol_term(candle: dict) -> pd.DataFrame:
    close = pd.Series(np.asarray(candle["close"], dtype=float))
    returns = np.log(close).diff()
    frame = pd.DataFrame({"signal_time": np.asarray(candle["time"], dtype=np.int64)})
    for window in (6, 12, 72, 288, 2016):
        frame[f"rv{window}"] = returns.rolling(
            window, min_periods=max(4, int(.8*window))).std(ddof=0)
    epsilon = 1e-10
    frame["fast_day"] = frame.rv6/(frame.rv288+epsilon)
    frame["hour_day"] = frame.rv12/(frame.rv288+epsilon)
    frame["day_week"] = frame.rv288/(frame.rv2016+epsilon)
    frame["vol_acceleration"] = frame.rv6/(frame.rv72+epsilon)
    frame["term_slope"] = np.log((frame.rv12+epsilon)/(frame.rv288+epsilon))
    frame["term_curvature"] = (
        np.log((frame.rv6+epsilon)/(frame.rv72+epsilon))
        - np.log((frame.rv72+epsilon)/(frame.rv2016+epsilon)))
    frame["release_ratio3"] = frame.fast_day/(frame.fast_day.shift(3)+epsilon)
    return frame.replace([np.inf, -np.inf], np.nan).dropna()


def attach(frame: pd.DataFrame, candles: dict) -> pd.DataFrame:
    pieces = []
    for symbol, part in frame.groupby("symbol", sort=False):
        left = part.copy()
        left["signal_time"] = left.entry_time.astype(np.int64)-300
        pieces.append(left.merge(symbol_term(candles[symbol]),
                                 on="signal_time", how="inner",
                                 validate="one_to_one"))
    result = pd.concat(pieces, ignore_index=True)
    grouped = result.groupby("entry_time", sort=False)
    mean = grouped.fast_day.transform("mean")
    std = grouped.fast_day.transform("std").replace(0, np.nan)
    result["vol_cross_z"] = (result.fast_day-mean)/std
    result["ret12_dispersion"] = grouped.ret12.transform("std")
    result["vol_breadth_expanding"] = grouped.fast_day.transform(
        lambda part: (part > 1.0).mean())
    return result.replace([np.inf, -np.inf], np.nan).dropna(
        subset=list(VOL_FEATURES))


def proposals(frame: pd.DataFrame) -> pd.DataFrame:
    pieces = []

    def add(name: str, direction: pd.Series, mask: pd.Series) -> None:
        selected = mask & direction.ne(0)
        part = frame.loc[selected].copy()
        part["direction"] = direction.loc[selected].astype(np.int8)
        part[f"vol_family_{name}"] = 1
        pieces.append(part)

    short_direction = np.sign(frame.ret3)
    add("compression_release", short_direction,
        frame.hour_day.between(.65, 1.35)
        & frame.release_ratio3.ge(1.35)
        & frame.day_week.le(.95)
        & (frame.ret3*short_direction).ge(.0008)
        & frame.range_ratio.ge(1.0))

    trend_direction = np.sign(frame.ret72)
    add("controlled_expansion", trend_direction,
        frame.hour_day.between(.85, 1.8)
        & frame.day_week.between(.65, 1.35)
        & frame.eff72.ge(.22)
        & (frame.ret12*trend_direction).ge(.001)
        & frame.vol_breadth_expanding.between(.25, .875))

    move = np.sign(frame.ret12)
    reversal = -move
    rejection = frame.lower_wick.where(reversal > 0, frame.upper_wick)
    add("volatility_exhaustion", reversal,
        frame.fast_day.ge(2.0) & frame.day_week.ge(1.0)
        & (frame.ret12.abs()/(frame.rv12+1e-10)).ge(2.0)
        & rejection.ge(.25) & (frame.body1*reversal).gt(0))

    leader = np.sign(frame.ret12)
    add("relative_vol_leader", leader,
        frame.vol_cross_z.ge(1.0)
        & (frame.ret12*leader).ge(.0015)
        & frame.ret12_dispersion.ge(.001)
        & frame.hour_day.between(.8, 2.2)
        & (frame.breadth12.where(leader > 0, 1-frame.breadth12)).ge(.5))

    long_direction = np.sign(frame.ret288)
    add("calm_trend_pullback", long_direction,
        frame.day_week.le(.85) & frame.hour_day.between(.55, 1.25)
        & (frame.ret288*long_direction).ge(.006)
        & (frame.ret6*long_direction).le(-.0005)
        & (frame.body1*long_direction).gt(0)
        & frame.eff72.ge(.12))

    combined = pd.concat(pieces, ignore_index=True)
    for family in FAMILIES:
        column = f"vol_family_{family}"
        if column not in combined:
            combined[column] = 0
    keys = ["symbol", "entry_time", "direction"]
    flags = combined.groupby(keys, sort=False)[
        [f"vol_family_{family}" for family in FAMILIES]].max().reset_index()
    base_rows = combined.sort_values(keys).drop_duplicates(keys).drop(
        columns=[f"vol_family_{family}" for family in FAMILIES])
    result = base_rows.merge(flags, on=keys, how="left", validate="one_to_one")
    for family in price.FAMILIES:
        result[f"family_{family}"] = 0
    return result


def matrix(frame: pd.DataFrame, schema: list[str]) -> np.ndarray:
    base_matrix = price.matrix(frame, schema)
    direction = frame.direction.to_numpy(float)
    directional = {"term_slope", "term_curvature", "vol_cross_z"}
    columns = []
    for name in VOL_FEATURES:
        values = frame[name].to_numpy(float)
        columns.append((values*direction if name in directional else values)[:, None])
        if name in directional:
            columns.append(np.abs(values)[:, None])
    for family in FAMILIES:
        columns.append(frame[f"vol_family_{family}"].to_numpy(float)[:, None])
    return np.hstack((base_matrix, np.hstack(columns).astype(np.float32)))


def selection_ok(value: dict) -> bool:
    return bool(value["n"] >= 700 and value["return_pct"] > 0
                and value["literal_target_hit_rate"] >= 68
                and value["dd_pct"] <= 7
                and value["trades_per_calendar_day"] >= 2.5)


def validation_ok(value: dict) -> bool:
    return bool(value["n"] >= 500 and value["return_pct"] > 0
                and value["literal_target_hit_rate"] >= 65
                and value["dd_pct"] <= 7
                and value["trades_per_calendar_day"] >= 2.0)


def public(item: dict | None) -> dict | None:
    if item is None:
        return None
    return {key: value for key, value in item.items()
            if key not in ("_frame", "_model")}


def main() -> None:
    candles, sources = source.load()
    raw = price.build_features(candles)
    enriched = attach(raw, candles)
    frame = price.add_outcomes(proposals(enriched), candles)
    fit = frame.year.isin((2020, 2021)).to_numpy()
    schema = price.columns(frame.loc[fit])
    x = matrix(frame, schema)
    labels = frame.target_hit.astype(int).to_numpy()
    selection = []
    for config in CONFIGS:
        model = HistGradientBoostingClassifier(
            max_iter=180, learning_rate=.06, max_leaf_nodes=config["leaves"],
            min_samples_leaf=config["leaf"], l2_regularization=config["l2"],
            random_state=20261007)
        model.fit(x[fit], labels[fit])
        scored = frame.copy()
        scored["quality"] = model.predict_proba(x)[:, 1]
        scores = scored.loc[scored.year.eq(2022), "quality"]
        for quantile in QUANTILES:
            threshold = float(scores.quantile(quantile))
            value = price.evaluate(scored, (2022,), threshold)
            if selection_ok(value):
                selection.append({"model": config, "quantile": quantile,
                                  "threshold": threshold,
                                  "selection_2022": value,
                                  "_frame": scored, "_model": model})
    validated = []
    validation_diagnostics = []
    for item in selection:
        value = price.evaluate(item["_frame"], (2023,), item["threshold"])
        validation_diagnostics.append(public({**item, "validation_2023": value}))
        if validation_ok(value):
            candidate = {**item, "validation_2023": value}
            candidate["score"] = round(
                min(item["selection_2022"]["return_pct"], value["return_pct"])
                - 2*max(item["selection_2022"]["dd_pct"], value["dd_pct"])
                + min(item["selection_2022"]["trades_per_calendar_day"],
                      value["trades_per_calendar_day"]), 8)
            validated.append(candidate)
    validated.sort(key=lambda item: item["score"], reverse=True)
    frozen = validated[0] if validated else None
    test_2024 = later = checks = None
    if frozen:
        test_2024 = price.evaluate(frozen["_frame"], (2024,), frozen["threshold"])
        if validation_ok(test_2024):
            later = {str(year): price.evaluate(
                frozen["_frame"], (year,), frozen["threshold"])
                for year in (2025, 2026)}
            checks = {year: validation_ok(value)
                      for year, value in later.items()}
    report = {
        "status": ("RESEARCH_CANDIDATE_AWAITING_DIRECT_FORWARD"
                   if checks and all(checks.values()) else "REJECTED"),
        "protocol": {
            "fit": [2020, 2021], "selection": 2022,
            "validation_and_ranking": 2023, "untouched_test": 2024,
            "hidden_if_test_passes": [2025, 2026],
            "entry": "next 5m market open", "target_pct": .5,
            "stop_pct": 1.0, "max_bars": 24,
            "families": list(FAMILIES), "model_grid": list(CONFIGS),
            "quantiles": list(QUANTILES),
            "money": "3% equity margin, fixed 10x gross",
        },
        "enriched_rows": len(enriched), "proposal_rows": len(frame),
        "fit_prevalence_pct": round(100*labels[fit].mean(), 4),
        "qualified_2022": len(selection), "qualified_2023": len(validated),
        "frozen_before_2024": public(frozen), "test_2024": test_2024,
        "later": later, "later_checks": checks,
        "selection_frontier": [public(item) for item in selection],
        "validation_diagnostics": validation_diagnostics,
        "sources": {**sources, str(Path(__file__)): digest(Path(__file__))},
        "limitations": [
            "The semantic term-structure families create finite selection risk.",
            "Historical Binance perpetual evidence requires direct X-Perp transfer.",
            "Any historical pass remains shadow-only pending new forward outcomes.",
        ],
    }
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({key: report[key] for key in (
        "status", "enriched_rows", "proposal_rows", "fit_prevalence_pct",
        "qualified_2022", "qualified_2023", "frozen_before_2024",
        "test_2024", "later", "later_checks")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
