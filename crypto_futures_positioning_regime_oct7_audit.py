"""Literal-target regimes from OI, trader positioning, and taker imbalance.

Full derivatives metrics start cleanly in 2023.  The model fits 2023, selects
on 2024-H1, validates on 2024-H2, and only then reveals 2025 and 2026.
Archived metric timestamps are conservatively lagged five minutes upstream.
"""
from __future__ import annotations

import hashlib
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

import crypto_5m_regime_router_literal_oct6_audit as price
import crypto_futures_funding_regime_oct6_audit as futures_source


METRICS_ROOT = Path("reports/audit_2026_09_08/crypto_futures_metrics_history")
OUT = Path("reports/audit_2026_09_08/crypto_futures_positioning_regime_oct7.json")
ROWS_OUT = Path("reports/audit_2026_09_08/crypto_futures_positioning_regime_oct7.pkl")
FAMILIES = ("oi_breakout", "liquidation_reversal", "crowd_squeeze",
            "top_divergence", "taker_absorption", "oi_price_divergence")
METRIC_FEATURES = (
    "oi_ret1", "oi_ret3", "oi_ret12", "oi_ret36", "oi_ret72",
    "oi_value_ret12", "taker_log", "taker_mean3", "taker_mean12",
    "taker_z288", "global_log", "global_delta12", "global_z288",
    "top_count_log", "top_position_log", "top_count_delta12",
    "top_position_delta12", "top_global_divergence",
    "oi_cross_z12", "taker_cross_z", "global_cross_z",
)
CONFIGS = (
    {"leaves": 15, "leaf": 250, "l2": 4.0},
    {"leaves": 31, "leaf": 400, "l2": 6.0},
)
QUANTILES = (.60, .70, .75, .80, .85, .90, .925, .95, .96, .97, .98)
PERIODS = {
    "fit_2023": (pd.Timestamp("2023-01-01", tz="UTC"),
                 pd.Timestamp("2024-01-01", tz="UTC")),
    "selection_2024_h1": (pd.Timestamp("2024-01-01", tz="UTC"),
                          pd.Timestamp("2024-07-01", tz="UTC")),
    "validation_2024_h2": (pd.Timestamp("2024-07-01", tz="UTC"),
                           pd.Timestamp("2025-01-01", tz="UTC")),
    "test_2025": (pd.Timestamp("2025-01-01", tz="UTC"),
                  pd.Timestamp("2026-01-01", tz="UTC")),
    "observed_2026": (pd.Timestamp("2026-01-01", tz="UTC"),
                      pd.Timestamp("2027-01-01", tz="UTC")),
}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_metrics() -> tuple[dict, dict]:
    manifest_path = METRICS_ROOT / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    result, sources = {}, {str(manifest_path): digest(manifest_path)}
    for symbol in futures_source.SYMBOLS:
        path = Path(manifest["symbols"][symbol]["path"])
        raw = path.read_bytes()
        found = hashlib.sha256(raw).hexdigest()
        if found != manifest["symbols"][symbol]["sha256"]:
            raise ValueError(f"metrics hash mismatch: {symbol}")
        result[symbol] = pickle.loads(raw)
        sources[str(path)] = found
    return result, sources


def metric_frame(data: dict) -> pd.DataFrame:
    frame = pd.DataFrame({key: np.asarray(value, dtype=float)
                          for key, value in data.items()})
    frame["metric_time"] = frame.pop("time").astype(np.int64)
    oi = frame.sum_open_interest.replace(0, np.nan)
    oi_value = frame.sum_open_interest_value.replace(0, np.nan)
    for window in (1, 3, 12, 36, 72):
        frame[f"oi_ret{window}"] = oi/oi.shift(window)-1
    frame["oi_value_ret12"] = oi_value/oi_value.shift(12)-1
    frame["taker_log"] = np.log(
        frame.sum_taker_long_short_vol_ratio.replace(0, np.nan))
    frame["taker_mean3"] = frame.taker_log.rolling(3, min_periods=3).mean()
    frame["taker_mean12"] = frame.taker_log.rolling(12, min_periods=9).mean()
    taker_mean = frame.taker_log.rolling(288, min_periods=192).mean()
    taker_std = frame.taker_log.rolling(288, min_periods=192).std().replace(0, np.nan)
    frame["taker_z288"] = (frame.taker_log-taker_mean)/taker_std
    frame["global_log"] = np.log(frame.count_long_short_ratio.replace(0, np.nan))
    frame["global_delta12"] = frame.global_log-frame.global_log.shift(12)
    global_mean = frame.global_log.rolling(288, min_periods=192).mean()
    global_std = frame.global_log.rolling(288, min_periods=192).std().replace(0, np.nan)
    frame["global_z288"] = (frame.global_log-global_mean)/global_std
    frame["top_count_log"] = np.log(
        frame.count_toptrader_long_short_ratio.replace(0, np.nan))
    frame["top_position_log"] = np.log(
        frame.sum_toptrader_long_short_ratio.replace(0, np.nan))
    frame["top_count_delta12"] = frame.top_count_log-frame.top_count_log.shift(12)
    frame["top_position_delta12"] = frame.top_position_log-frame.top_position_log.shift(12)
    frame["top_global_divergence"] = frame.top_count_log-frame.global_log
    keep = ["metric_time", *[name for name in METRIC_FEATURES
                              if not name.endswith("cross_z12")
                              and not name.endswith("cross_z")]]
    return frame[keep].replace([np.inf, -np.inf], np.nan).dropna()


def attach_metrics(frame: pd.DataFrame, metrics: dict) -> pd.DataFrame:
    pieces = []
    for symbol, part in frame.groupby("symbol", sort=False):
        right = metric_frame(metrics[symbol])
        left = part.sort_values("entry_time").copy()
        found = left.merge(right, left_on="entry_time", right_on="metric_time",
                           how="inner", validate="one_to_one")
        pieces.append(found)
    result = pd.concat(pieces, ignore_index=True)
    grouped = result.groupby("entry_time", sort=False)
    for source, target in (("oi_ret12", "oi_cross_z12"),
                           ("taker_mean3", "taker_cross_z"),
                           ("global_log", "global_cross_z")):
        mean = grouped[source].transform("mean")
        std = grouped[source].transform("std").replace(0, np.nan)
        result[target] = (result[source]-mean)/std
    return result.replace([np.inf, -np.inf], np.nan).dropna(
        subset=list(METRIC_FEATURES))


def proposals(frame: pd.DataFrame) -> pd.DataFrame:
    pieces = []

    def add(name: str, direction: pd.Series, mask: pd.Series) -> None:
        selected = mask & direction.ne(0)
        part = frame.loc[selected].copy()
        part["direction"] = direction.loc[selected].astype(np.int8)
        part[f"positioning_family_{name}"] = 1
        pieces.append(part)

    move12 = np.sign(frame.ret12)
    add("oi_breakout", move12,
        frame.ret12.abs().ge(.002) & frame.oi_ret12.ge(.002)
        & (frame.taker_mean3*move12).ge(.03) & frame.range_ratio.ge(1.0))

    reverse12 = -move12
    rejection = frame.lower_wick.where(reverse12 > 0, frame.upper_wick)
    add("liquidation_reversal", reverse12,
        frame.ret12.abs().ge(.0035) & frame.oi_ret12.le(-.0025)
        & (frame.taker_mean3*move12).ge(.05) & rejection.ge(.18))

    squeeze = -np.sign(frame.global_log)
    add("crowd_squeeze", squeeze,
        frame.global_log.abs().ge(.08) & (frame.ret3*squeeze).ge(.0005)
        & (frame.taker_mean3*squeeze).ge(.02) & frame.oi_ret12.ge(-.001))

    top_direction = np.sign(frame.top_global_divergence)
    add("top_divergence", top_direction,
        frame.top_global_divergence.abs().ge(.06)
        & (frame.taker_mean3*top_direction).ge(0)
        & (frame.ret3*top_direction).ge(0))

    taker_direction = np.sign(frame.taker_z288)
    absorption = -taker_direction
    add("taker_absorption", absorption,
        frame.taker_z288.abs().ge(1.0) & frame.range_ratio.ge(1.0)
        & (frame.ret3*taker_direction).le(.0008)
        & (frame.body1*absorption).gt(0))

    divergence = -np.sign(frame.ret36)
    add("oi_price_divergence", divergence,
        frame.ret36.abs().ge(.004) & frame.oi_ret36.le(-.003)
        & (frame.taker_mean3*np.sign(frame.ret36)).le(.05)
        & (frame.body1*divergence).gt(0))

    combined = pd.concat(pieces, ignore_index=True)
    for family in FAMILIES:
        column = f"positioning_family_{family}"
        if column not in combined:
            combined[column] = 0
    keys = ["symbol", "entry_time", "direction"]
    flags = combined.groupby(keys, sort=False)[
        [f"positioning_family_{family}" for family in FAMILIES]].max().reset_index()
    base_rows = combined.sort_values(keys).drop_duplicates(keys).drop(
        columns=[f"positioning_family_{family}" for family in FAMILIES])
    result = base_rows.merge(flags, on=keys, how="left", validate="one_to_one")
    for family in price.FAMILIES:
        result[f"family_{family}"] = 0
    return result


def matrices(frame: pd.DataFrame, schema: list[str]) -> tuple[np.ndarray, np.ndarray]:
    base_matrix = price.matrix(frame, schema)
    direction = frame.direction.to_numpy(float)
    columns = []
    directional = {"taker_log", "taker_mean3", "taker_mean12", "taker_z288",
                   "global_log", "global_delta12", "global_z288",
                   "top_count_log", "top_position_log", "top_count_delta12",
                   "top_position_delta12", "top_global_divergence",
                   "oi_cross_z12", "taker_cross_z", "global_cross_z"}
    for name in METRIC_FEATURES:
        values = frame[name].to_numpy(float)
        columns.append((values*direction if name in directional else values)[:, None])
        if name in directional:
            columns.append(np.abs(values)[:, None])
    for family in FAMILIES:
        columns.append(frame[f"positioning_family_{family}"].to_numpy(float)[:, None])
    full = np.hstack((base_matrix, np.hstack(columns).astype(np.float32)))
    return base_matrix, full


def period_mask(frame: pd.DataFrame, name: str) -> np.ndarray:
    start, end = PERIODS[name]
    stamp = pd.to_datetime(frame.entry_time, unit="s", utc=True)
    return ((stamp >= start) & (stamp < end)).to_numpy()


def evaluate_period(frame: pd.DataFrame, name: str, threshold: float) -> dict:
    subset = frame.loc[period_mask(frame, name)].copy()
    years = tuple(sorted(subset.year.unique().astype(int)))
    return price.evaluate(subset, years, threshold) if years else price.evaluate(
        subset, (1900,), threshold)


def selection_ok(value: dict) -> bool:
    return bool(value["n"] >= 400 and value["return_pct"] > 0
                and value["literal_target_hit_rate"] >= 68
                and value["dd_pct"] <= 8
                and value["trades_per_calendar_day"] >= 2.5)


def validation_ok(value: dict) -> bool:
    return bool(value["n"] >= 350 and value["return_pct"] > 0
                and value["literal_target_hit_rate"] >= 67
                and value["dd_pct"] <= 8
                and value["trades_per_calendar_day"] >= 2.3)


def objective_ok(value: dict) -> bool:
    return bool(value["n"] >= 500 and value["return_pct"] > 0
                and 75 <= value["literal_target_hit_rate"] <= 95
                and value["dd_pct"] <= 5 and value["negative_months"] == 0
                and value["trades_per_calendar_day"] >= 3
                and value["months_ge_30pct"] == value["months"])


def public(item: dict | None) -> dict | None:
    if item is None:
        return None
    return {key: value for key, value in item.items()
            if key not in ("_frame", "_model")}


def main() -> None:
    candles, _, price_sources = futures_source.load()
    metrics, metric_sources = load_metrics()
    raw = price.build_features(candles)
    enriched = attach_metrics(raw, metrics)
    frame = price.add_outcomes(proposals(enriched), candles)
    fit_mask = period_mask(frame, "fit_2023")
    selection_mask = period_mask(frame, "selection_2024_h1")
    schema = price.columns(frame.loc[fit_mask])
    x_price, x_full = matrices(frame, schema)
    y = frame.target_hit.astype(int).to_numpy()
    candidates, frontier = [], []
    for model_kind, matrix in (("price_only_control", x_price),
                               ("positioning_plus_price", x_full)):
        for config in CONFIGS:
            model = HistGradientBoostingClassifier(
                max_iter=180, learning_rate=.06,
                max_leaf_nodes=config["leaves"],
                min_samples_leaf=config["leaf"],
                l2_regularization=config["l2"], random_state=20261007)
            model.fit(matrix[fit_mask], y[fit_mask])
            scored = frame.copy()
            scored["quality"] = model.predict_proba(matrix)[:, 1]
            for quantile in QUANTILES:
                threshold = float(scored.loc[selection_mask, "quality"].quantile(quantile))
                selection = evaluate_period(scored, "selection_2024_h1", threshold)
                item = {"model_kind": model_kind, "model": config,
                        "quantile": quantile, "threshold": threshold,
                        "selection_2024_h1": selection}
                frontier.append(item)
                if not selection_ok(selection):
                    continue
                validation = evaluate_period(scored, "validation_2024_h2", threshold)
                item["validation_2024_h2"] = validation
                if not validation_ok(validation):
                    continue
                item["score"] = round(
                    selection["return_pct"]+2*validation["return_pct"]
                    -selection["dd_pct"]-2*validation["dd_pct"], 8)
                item["_frame"], item["_model"] = scored, model
                candidates.append(item)
    candidates.sort(key=lambda item: item["score"], reverse=True)
    positioning = [item for item in candidates
                   if item["model_kind"] == "positioning_plus_price"]
    frozen = positioning[0] if positioning else None
    control = next((item for item in candidates
                    if item["model_kind"] == "price_only_control"), None)
    results = checks = None
    if frozen:
        threshold = frozen["threshold"]
        results = {name: evaluate_period(frozen["_frame"], name, threshold)
                   for name in PERIODS if name != "fit_2023"}
        checks = {name: objective_ok(value) for name, value in results.items()}
        frozen["_frame"][frozen["_frame"].quality.ge(threshold)].to_pickle(
            ROWS_OUT, compression="gzip")
    status = ("RESEARCH_CANDIDATE_AWAITING_VENUE_FORWARD"
              if checks and all(checks.values()) else "REJECTED")
    report = {
        "status": status,
        "protocol": {"fit": "2023", "selection": "2024-H1",
                     "validation": "2024-H2", "test": "2025",
                     "observed": "2026 through September",
                     "entry": "next 5m perpetual market open",
                     "metrics_availability": "archive timestamp plus 5 minutes",
                     "target_pct": .5, "stop_pct": 1.0, "max_bars": 24,
                     "money": "3% equity margin, fixed 10x gross",
                     "win_definition": "literal target touch before stop/timeout"},
        "rows": {"raw_price": len(raw), "enriched": len(enriched),
                 "proposals": len(frame)},
        "fit_literal_prevalence_pct": round(100*float(y[fit_mask].mean()), 4),
        "qualified_before_test": len(candidates),
        "frozen_positioning_before_test": public(frozen),
        "best_price_control_before_test": public(control),
        "top_before_test": [public(item) for item in candidates[:20]],
        "selection_frontier": [public(item) for item in sorted(
            frontier, key=lambda row: (
                row["selection_2024_h1"]["literal_target_hit_rate"],
                row["selection_2024_h1"]["trades_per_calendar_day"]),
            reverse=True)[:30]],
        "results": results, "checks": checks,
        "sources": {**price_sources, **metric_sources,
                    str(Path(__file__)): digest(Path(__file__))},
        "limitations": [
            "Official metrics contain historical gaps recorded in the manifest.",
            "The bounded family/model/threshold grid still creates selection risk.",
            "Any survivor requires untouched direct X-Perp forward evidence.",
        ],
    }
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"status": status, "rows": report["rows"],
                      "fit_literal_prevalence_pct": report["fit_literal_prevalence_pct"],
                      "qualified_before_test": len(candidates),
                      "frozen": public(frozen), "control": public(control),
                      "results": results, "checks": checks}, indent=2), flush=True)


if __name__ == "__main__":
    main()
