"""Causal spot/perpetual price-impact response audit with literal targets.

This branch models how much completed-candle price movement each unit of
aggressive flow produced.  It separates high-impact informed flow from large
flow absorbed with little price response.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

import crypto_5m_regime_router_literal_oct6_audit as price
import crypto_futures_trade_flow_regime_oct7_audit as perp
import crypto_spot_perp_flow_divergence_oct7_audit as dual


OUT = Path("reports/audit_2026_09_08/crypto_price_impact_response_oct7.json")
ROWS_OUT = Path("reports/audit_2026_09_08/crypto_price_impact_response_oct7.pkl")
FAMILIES = ("informed_impact", "passive_absorption", "spot_informed_lead",
            "residual_breakout", "impact_exhaustion", "cross_impact_leader")
BASE_IMPACT_FEATURES = (
    "imbalance", "flow12", "flow72", "quote_volume_ratio",
    "response1", "response12", "impact_per_flow", "impact_z288",
    "beta288", "beta2016", "beta_ratio", "residual_z288",
    "absorption", "absorption_z288", "impact_persistence12",
    "flow_persistence12", "price_efficiency12",
)
FEATURES = tuple(
    f"{venue}_{name}" for venue in ("perp", "spot")
    for name in BASE_IMPACT_FEATURES
) + (
    "impact_gap", "absorption_gap", "residual_gap",
    "perp_residual_cross_z", "spot_residual_cross_z",
    "impact_cross_dispersion", "absorption_cross_breadth",
)
DIRECTIONAL_SUFFIXES = {
    "imbalance", "flow12", "flow72", "response1", "response12",
    "impact_per_flow", "impact_z288", "beta288", "beta2016",
    "beta_ratio", "residual_z288",
}
CONFIGS = (
    {"leaves": 15, "leaf": 250, "l2": 3.0},
    {"leaves": 31, "leaf": 400, "l2": 5.0},
)
QUANTILES = (.70, .75, .80, .85, .90, .925, .95, .96, .97, .98, .985, .99)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def shifted_z(value: pd.Series, window: int, minimum: int) -> pd.Series:
    mean = value.rolling(window, min_periods=minimum).mean().shift(1)
    std = value.rolling(window, min_periods=minimum).std(ddof=0).shift(1)
    return (value - mean) / std.replace(0, np.nan)


def impact_frame(candle: dict, prefix: str) -> pd.DataFrame:
    frame = pd.DataFrame({key: np.asarray(value, dtype=float)
                          for key, value in candle.items()})
    signal_time = frame.pop("time").astype(np.int64)
    quote = frame.quote_volume.replace(0, np.nan)
    signed_quote = 2 * frame.taker_buy_quote_volume - frame.quote_volume
    imbalance = signed_quote / quote
    close = frame.close
    ret1 = np.log(close).diff()
    ret12 = np.log(close / close.shift(12))
    volatility = ret1.rolling(288, min_periods=192).std(ddof=0).shift(1)
    quote_ratio = quote / quote.rolling(288, min_periods=192).median().shift(1)
    flow12 = signed_quote.rolling(12, min_periods=9).sum() / quote.rolling(
        12, min_periods=9).sum()
    flow72 = signed_quote.rolling(72, min_periods=54).sum() / quote.rolling(
        72, min_periods=54).sum()
    covariance288 = ret1.rolling(288, min_periods=192).cov(imbalance).shift(1)
    variance288 = imbalance.rolling(288, min_periods=192).var(ddof=0).shift(1)
    covariance2016 = ret1.rolling(2016, min_periods=1200).cov(imbalance).shift(1)
    variance2016 = imbalance.rolling(2016, min_periods=1200).var(ddof=0).shift(1)
    beta288 = covariance288 / variance288.replace(0, np.nan)
    beta2016 = covariance2016 / variance2016.replace(0, np.nan)
    response1 = ret1 / volatility.replace(0, np.nan)
    response12 = ret12 / (np.sqrt(12) * volatility).replace(0, np.nan)
    expected = beta288 * imbalance
    residual = ret1 - expected
    residual_z = shifted_z(residual, 288, 192)
    aligned_response = response1 * np.sign(imbalance)
    impact = aligned_response / (imbalance.abs() + .03)
    absorption = (imbalance.abs() * quote_ratio
                  / (response1.abs() + .25))
    absolute_path = ret1.abs().rolling(12, min_periods=9).sum()
    result = pd.DataFrame({
        "signal_time": signal_time,
        f"{prefix}_imbalance": imbalance,
        f"{prefix}_flow12": flow12,
        f"{prefix}_flow72": flow72,
        f"{prefix}_quote_volume_ratio": quote_ratio,
        f"{prefix}_response1": response1,
        f"{prefix}_response12": response12,
        f"{prefix}_impact_per_flow": impact,
        f"{prefix}_impact_z288": shifted_z(impact, 288, 192),
        f"{prefix}_beta288": beta288,
        f"{prefix}_beta2016": beta2016,
        f"{prefix}_beta_ratio": beta288 / beta2016.abs().clip(lower=1e-8),
        f"{prefix}_residual_z288": residual_z,
        f"{prefix}_absorption": absorption,
        f"{prefix}_absorption_z288": shifted_z(absorption, 288, 192),
        f"{prefix}_impact_persistence12": (ret1 * imbalance > 0).rolling(
            12, min_periods=9).mean(),
        f"{prefix}_flow_persistence12": np.sign(imbalance).rolling(
            12, min_periods=9).mean().abs(),
        f"{prefix}_price_efficiency12": ret12.abs() / absolute_path,
    })
    return result.replace([np.inf, -np.inf], np.nan).dropna()


def attach(frame: pd.DataFrame, futures: dict, spot: dict) -> pd.DataFrame:
    parts = []
    for symbol, part in frame.groupby("symbol", sort=False):
        left = part.copy()
        left["signal_time"] = left.entry_time.astype(np.int64) - 300
        joined = left.merge(impact_frame(futures[symbol], "perp"),
                            on="signal_time", how="inner", validate="one_to_one")
        joined = joined.merge(impact_frame(spot[symbol], "spot"),
                              on="signal_time", how="inner", validate="one_to_one")
        joined["impact_gap"] = (joined.spot_impact_per_flow
                                - joined.perp_impact_per_flow)
        joined["absorption_gap"] = (joined.spot_absorption
                                    - joined.perp_absorption)
        joined["residual_gap"] = (joined.spot_residual_z288
                                  - joined.perp_residual_z288)
        parts.append(joined)
    result = pd.concat(parts, ignore_index=True)
    grouped = result.groupby("entry_time", sort=False)
    for source, target in (("perp_residual_z288", "perp_residual_cross_z"),
                           ("spot_residual_z288", "spot_residual_cross_z")):
        mean = grouped[source].transform("mean")
        std = grouped[source].transform("std").replace(0, np.nan)
        result[target] = (result[source] - mean) / std
    result["impact_cross_dispersion"] = grouped.perp_impact_per_flow.transform("std")
    result["absorption_cross_breadth"] = grouped.perp_absorption_z288.transform(
        lambda values: (values > 1).mean())
    return result.replace([np.inf, -np.inf], np.nan).dropna(
        subset=list(FEATURES))


def proposals(frame: pd.DataFrame) -> pd.DataFrame:
    pieces = []

    def add(name: str, direction: pd.Series, mask: pd.Series) -> None:
        selected = mask & direction.ne(0)
        part = frame.loc[selected].copy()
        part["direction"] = direction.loc[selected].astype(np.int8)
        part[f"impact_family_{name}"] = 1
        pieces.append(part)

    informed = np.sign(frame.perp_flow12)
    add("informed_impact", informed,
        frame.perp_flow12.abs().ge(.04)
        & (frame.perp_response12 * informed).ge(.8)
        & frame.perp_impact_persistence12.ge(.58)
        & frame.perp_quote_volume_ratio.ge(.9)
        & frame.perp_beta288.gt(0))

    absorbed = -np.sign(frame.perp_imbalance)
    rejection = frame.lower_wick.where(absorbed > 0, frame.upper_wick)
    add("passive_absorption", absorbed,
        frame.perp_imbalance.abs().ge(.15)
        & frame.perp_absorption_z288.ge(1.0)
        & frame.perp_quote_volume_ratio.ge(1.0)
        & rejection.ge(.18)
        & (frame.body1 * absorbed).gt(0))

    spot_lead = np.sign(frame.spot_flow12)
    add("spot_informed_lead", spot_lead,
        frame.spot_flow12.abs().ge(.04)
        & (frame.spot_response12 * spot_lead).ge(.7)
        & (frame.impact_gap * spot_lead).ge(0)
        & (frame.perp_response12 * spot_lead).le(1.2)
        & frame.spot_quote_volume_ratio.ge(.9))

    residual = np.sign(frame.perp_residual_z288)
    add("residual_breakout", residual,
        frame.perp_residual_z288.abs().ge(1.5)
        & (frame.body1 * residual).gt(0)
        & frame.perp_quote_volume_ratio.ge(1.1)
        & frame.perp_impact_persistence12.ge(.5))

    move = np.sign(frame.perp_response12)
    exhausted = -move
    exhaustion_wick = frame.lower_wick.where(exhausted > 0, frame.upper_wick)
    add("impact_exhaustion", exhausted,
        frame.perp_response12.abs().ge(2.0)
        & frame.perp_impact_z288.abs().ge(1.0)
        & exhaustion_wick.ge(.22)
        & (frame.body1 * exhausted).gt(0))

    leader = np.sign(frame.perp_residual_cross_z)
    add("cross_impact_leader", leader,
        frame.perp_residual_cross_z.abs().ge(1.0)
        & (frame.perp_residual_z288 * leader).ge(1.0)
        & (frame.ret3 * leader).ge(.0005)
        & frame.impact_cross_dispersion.ge(.5))

    combined = pd.concat(pieces, ignore_index=True)
    for family in FAMILIES:
        column = f"impact_family_{family}"
        if column not in combined:
            combined[column] = 0
    keys = ["symbol", "entry_time", "direction"]
    family_columns = [f"impact_family_{family}" for family in FAMILIES]
    flags = combined.groupby(keys, sort=False)[family_columns].max().reset_index()
    base = combined.sort_values(keys).drop_duplicates(keys).drop(
        columns=family_columns)
    result = base.merge(flags, on=keys, how="left", validate="one_to_one")
    for family in price.FAMILIES:
        result[f"family_{family}"] = 0
    return result


def matrix(frame: pd.DataFrame, schema: list[str]) -> np.ndarray:
    base = price.matrix(frame, schema)
    direction = frame.direction.to_numpy(float)
    columns = []
    for name in FEATURES:
        values = frame[name].to_numpy(float)
        suffix = name.split("_", 1)[1] if name.startswith(("perp_", "spot_")) else name
        directional = suffix in DIRECTIONAL_SUFFIXES or name in (
            "impact_gap", "residual_gap", "perp_residual_cross_z",
            "spot_residual_cross_z")
        columns.append((values * direction if directional else values)[:, None])
        if directional:
            columns.append(np.abs(values)[:, None])
    for family in FAMILIES:
        columns.append(frame[f"impact_family_{family}"].to_numpy(float)[:, None])
    return np.hstack((base, np.hstack(columns).astype(np.float32)))


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
    futures, future_sources = perp.load()
    spot, spot_sources = dual.load_spot()
    raw = price.build_features(futures)
    enriched = attach(raw, futures, spot)
    frame = price.add_outcomes(proposals(enriched), futures)
    fit = frame.year.isin((2020, 2021)).to_numpy()
    schema = price.columns(frame.loc[fit])
    x = matrix(frame, schema)
    labels = frame.target_hit.astype(int).to_numpy()
    selection = []
    frontier = []
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
            item = {"model": config, "quantile": quantile,
                    "threshold": threshold, "selection_2022": value}
            frontier.append(item)
            if selection_ok(value):
                selection.append({**item, "_frame": scored, "_model": model})
    validated = []
    validation_diagnostics = []
    for item in selection:
        value = price.evaluate(item["_frame"], (2023,), item["threshold"])
        validation_diagnostics.append(public({**item, "validation_2023": value}))
        if validation_ok(value):
            candidate = {**item, "validation_2023": value}
            candidate["score"] = round(
                min(item["selection_2022"]["return_pct"], value["return_pct"])
                - 2 * max(item["selection_2022"]["dd_pct"], value["dd_pct"])
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
        frozen["_frame"].loc[
            frozen["_frame"].quality.ge(frozen["threshold"])].to_pickle(ROWS_OUT)
    report = {
        "status": ("RESEARCH_CANDIDATE_AWAITING_DIRECT_FORWARD"
                   if checks and all(checks.values()) else "REJECTED"),
        "protocol": {
            "fit": [2020, 2021], "selection": 2022,
            "validation_and_ranking": 2023, "untouched_test": 2024,
            "hidden_if_test_passes": [2025, 2026],
            "families": list(FAMILIES), "model_grid": list(CONFIGS),
            "quantiles": list(QUANTILES),
            "entry": "next 5m market open", "target_pct": .5,
            "stop_pct": 1.0, "max_bars": 24,
            "money": "3% equity margin, fixed 10x gross",
        },
        "enriched_rows": len(enriched), "proposal_rows": len(frame),
        "fit_literal_target_prevalence_pct": round(100 * float(labels[fit].mean()), 4),
        "qualified_2022": len(selection), "qualified_2023": len(validated),
        "selection_frontier": frontier,
        "validation_diagnostics": validation_diagnostics,
        "frozen_before_2024": public(frozen), "test_2024": test_2024,
        "later": later, "later_checks": checks,
        "sources": {**future_sources, **spot_sources,
                    str(Path(__file__)): digest(Path(__file__))},
        "limitations": [
            "Impact is estimated from completed five-minute aggregates, not order-book events.",
            "The finite semantic proposal and model grid still create selection risk.",
            "Any historical pass remains shadow-only pending direct X-Perp forward outcomes.",
        ],
    }
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({key: report[key] for key in (
        "status", "enriched_rows", "proposal_rows",
        "fit_literal_target_prevalence_pct", "qualified_2022",
        "qualified_2023", "frozen_before_2024", "test_2024", "later",
        "later_checks")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
