"""Literal-target audit of causal spot/perpetual trade-flow divergence."""
from __future__ import annotations

import hashlib
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

import crypto_5m_regime_router_literal_oct6_audit as price
import crypto_futures_trade_flow_regime_oct7_audit as perp


SPOT_ROOT = Path("reports/audit_2026_09_08/crypto_spot_flow_history")
OUT = Path("reports/audit_2026_09_08/crypto_spot_perp_flow_divergence_oct7.json")
ROWS_OUT = Path("reports/audit_2026_09_08/crypto_spot_perp_flow_divergence_oct7.pkl")
FAMILIES = ("spot_led", "confirmed_flow", "perp_crowding_fade",
            "basis_reversion", "spot_cvd_reversal", "spot_absorption")
FEATURES = (
    "spot_imbalance", "spot_imbalance_mean3", "spot_imbalance_mean12",
    "spot_imbalance_mean36", "spot_imbalance_mean72", "spot_imbalance_z288",
    "spot_quote_imbalance", "spot_cvd12", "spot_cvd72",
    "spot_trade_count_ratio", "spot_quote_volume_ratio", "spot_avg_trade_ratio",
    "perp_imbalance", "perp_imbalance_mean3", "perp_imbalance_mean12",
    "perp_imbalance_mean36", "perp_imbalance_mean72", "perp_imbalance_z288",
    "perp_quote_imbalance", "perp_cvd12", "perp_cvd72",
    "perp_trade_count_ratio", "perp_quote_volume_ratio", "perp_avg_trade_ratio",
    "imbalance_gap3", "imbalance_gap12", "cvd_gap12", "cvd_gap72",
    "basis", "basis_z288", "basis_change3", "normalized_volume_gap",
)
DIRECTIONAL = {
    "spot_imbalance", "spot_imbalance_mean3", "spot_imbalance_mean12",
    "spot_imbalance_mean36", "spot_imbalance_mean72", "spot_imbalance_z288",
    "spot_quote_imbalance", "spot_cvd12", "spot_cvd72",
    "perp_imbalance", "perp_imbalance_mean3", "perp_imbalance_mean12",
    "perp_imbalance_mean36", "perp_imbalance_mean72", "perp_imbalance_z288",
    "perp_quote_imbalance", "perp_cvd12", "perp_cvd72",
    "imbalance_gap3", "imbalance_gap12", "cvd_gap12", "cvd_gap72",
    "basis", "basis_z288", "basis_change3",
}
CONFIGS = (
    {"leaves": 15, "leaf": 250, "l2": 3.0},
    {"leaves": 31, "leaf": 400, "l2": 5.0},
)
QUANTILES = (.75, .80, .85, .90, .925, .95, .96, .97, .98, .985, .99)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_spot() -> tuple[dict, dict]:
    manifest_path = SPOT_ROOT / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    candles, sources = {}, {str(manifest_path): digest(manifest_path)}
    for symbol, meta in manifest["symbols"].items():
        path = Path(meta["path"])
        raw = path.read_bytes()
        found = hashlib.sha256(raw).hexdigest()
        if found != meta["sha256"]:
            raise ValueError(f"spot-flow hash mismatch: {symbol}")
        candles[symbol] = pickle.loads(raw)
        sources[str(path)] = found
    return candles, sources


def flow_frame(candle: dict, prefix: str) -> pd.DataFrame:
    found = perp.symbol_flow(candle)
    raw = pd.DataFrame({"signal_time": np.asarray(candle["time"], dtype=np.int64),
                        f"{prefix}_close": np.asarray(candle["close"], dtype=float)})
    found = found.merge(raw, on="signal_time", how="inner", validate="one_to_one")
    return found.rename(columns={name: f"{prefix}_{name}"
                                 for name in found.columns
                                 if name != "signal_time"
                                 and not name.startswith(f"{prefix}_")})


def attach(frame: pd.DataFrame, futures: dict, spot: dict) -> pd.DataFrame:
    pieces = []
    keep_flow = [name for name in perp.FLOW_FEATURES
                 if name not in ("flow_cross_z", "cvd_cross_z")]
    for symbol, part in frame.groupby("symbol", sort=False):
        left = part.copy()
        left["signal_time"] = left.entry_time.astype(np.int64)-300
        future_flow = flow_frame(futures[symbol], "perp")
        spot_flow = flow_frame(spot[symbol], "spot")
        joined = left.merge(future_flow, on="signal_time", how="inner",
                            validate="one_to_one")
        joined = joined.merge(spot_flow, on="signal_time", how="inner",
                              validate="one_to_one")
        for window in (3, 12):
            joined[f"imbalance_gap{window}"] = (
                joined[f"spot_imbalance_mean{window}"]
                - joined[f"perp_imbalance_mean{window}"])
        for window in (12, 72):
            joined[f"cvd_gap{window}"] = (
                joined[f"spot_cvd{window}"]-joined[f"perp_cvd{window}"])
        joined["basis"] = joined.perp_close/joined.spot_close-1
        # These rolling calculations use only aligned, completed observations.
        basis_mean = joined.basis.rolling(288, min_periods=192).mean()
        basis_std = joined.basis.rolling(288, min_periods=192).std().replace(0, np.nan)
        joined["basis_z288"] = (joined.basis-basis_mean)/basis_std
        joined["basis_change3"] = joined.basis-joined.basis.shift(3)
        joined["normalized_volume_gap"] = (
            np.log(joined.spot_quote_volume_ratio.clip(lower=1e-6))
            - np.log(joined.perp_quote_volume_ratio.clip(lower=1e-6)))
        pieces.append(joined)
    result = pd.concat(pieces, ignore_index=True)
    required = [
        *FEATURES,
        *[f"spot_{name}" for name in keep_flow],
        *[f"perp_{name}" for name in keep_flow],
    ]
    return result.replace([np.inf, -np.inf], np.nan).dropna(
        subset=list(dict.fromkeys(required)))


def proposals(frame: pd.DataFrame) -> pd.DataFrame:
    pieces = []

    def add(name: str, direction: pd.Series, mask: pd.Series) -> None:
        selected = mask & direction.ne(0)
        part = frame.loc[selected].copy()
        part["direction"] = direction.loc[selected].astype(np.int8)
        part[f"dual_family_{name}"] = 1
        pieces.append(part)

    spot_direction = np.sign(frame.spot_imbalance_mean3)
    add("spot_led", spot_direction,
        frame.spot_imbalance_mean3.abs().ge(.08)
        & (frame.imbalance_gap3*spot_direction).ge(.04)
        & (frame.ret3*spot_direction).ge(.0005)
        & frame.spot_quote_volume_ratio.ge(.9))

    confirmed = np.sign(frame.spot_imbalance_mean3+frame.perp_imbalance_mean3)
    add("confirmed_flow", confirmed,
        (frame.spot_imbalance_mean3*confirmed).ge(.04)
        & (frame.perp_imbalance_mean3*confirmed).ge(.04)
        & (frame.ret3*confirmed).ge(.0008)
        & frame.spot_quote_volume_ratio.ge(.8)
        & frame.perp_quote_volume_ratio.ge(.8))

    perp_direction = np.sign(frame.perp_imbalance_mean3)
    fade = -perp_direction
    rejection = frame.lower_wick.where(fade > 0, frame.upper_wick)
    add("perp_crowding_fade", fade,
        frame.perp_imbalance_mean3.abs().ge(.12)
        & (frame.spot_imbalance_mean3*perp_direction).le(.02)
        & (frame.ret3*perp_direction).ge(.0012)
        & rejection.ge(.18))

    basis_direction = -np.sign(frame.basis_z288)
    add("basis_reversion", basis_direction,
        frame.basis_z288.abs().ge(1.5)
        & (frame.basis_change3*np.sign(frame.basis_z288)).gt(0)
        & (frame.spot_imbalance_mean3*basis_direction).ge(-.02))

    cvd_direction = np.sign(frame.spot_cvd12)
    add("spot_cvd_reversal", cvd_direction,
        frame.spot_cvd12.abs().ge(.05)
        & (frame.ret12*cvd_direction).le(-.0015)
        & (frame.cvd_gap12*cvd_direction).ge(.03)
        & frame.spot_quote_volume_ratio.ge(.8))

    absorption = -np.sign(frame.perp_imbalance)
    add("spot_absorption", absorption,
        frame.perp_imbalance.abs().ge(.20)
        & (frame.spot_imbalance*absorption).ge(.02)
        & frame.perp_quote_volume_ratio.ge(1.0)
        & (frame.body1*absorption).gt(0))

    combined = pd.concat(pieces, ignore_index=True)
    for family in FAMILIES:
        column = f"dual_family_{family}"
        if column not in combined:
            combined[column] = 0
    keys = ["symbol", "entry_time", "direction"]
    flags = combined.groupby(keys, sort=False)[
        [f"dual_family_{family}" for family in FAMILIES]].max().reset_index()
    base_rows = combined.sort_values(keys).drop_duplicates(keys).drop(
        columns=[f"dual_family_{family}" for family in FAMILIES])
    result = base_rows.merge(flags, on=keys, how="left", validate="one_to_one")
    for family in price.FAMILIES:
        result[f"family_{family}"] = 0
    return result


def matrix(frame: pd.DataFrame, schema: list[str]) -> np.ndarray:
    base_matrix = price.matrix(frame, schema)
    direction = frame.direction.to_numpy(float)
    columns = []
    for name in FEATURES:
        values = frame[name].to_numpy(float)
        columns.append((values*direction if name in DIRECTIONAL else values)[:, None])
        if name in DIRECTIONAL:
            columns.append(np.abs(values)[:, None])
    for family in FAMILIES:
        columns.append(frame[f"dual_family_{family}"].to_numpy(float)[:, None])
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
    futures, future_sources = perp.load()
    spot, spot_sources = load_spot()
    raw = price.build_features(futures)
    enriched = attach(raw, futures, spot)
    candidate_rows = proposals(enriched)
    frame = price.add_outcomes(candidate_rows, futures)
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
        selection_scores = scored.loc[scored.year.eq(2022), "quality"]
        for quantile in QUANTILES:
            threshold = float(selection_scores.quantile(quantile))
            value = price.evaluate(scored, (2022,), threshold)
            if selection_ok(value):
                selection.append({"model": config, "quantile": quantile,
                                  "threshold": threshold,
                                  "selection_2022": value,
                                  "_frame": scored, "_model": model})
    validated = []
    for item in selection:
        value = price.evaluate(item["_frame"], (2023,), item["threshold"])
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
        test_2024 = price.evaluate(
            frozen["_frame"], (2024,), frozen["threshold"])
        selected_rows = []
        for year in range(2022, 2027):
            selected_rows.extend(price.portfolio_rows(
                frozen["_frame"], (year,), frozen["threshold"]))
        pd.DataFrame(selected_rows).to_pickle(ROWS_OUT, compression="gzip")
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
        "sources": {**future_sources, **spot_sources,
                    str(Path(__file__)): digest(Path(__file__))},
        "limitations": [
            "The semantic dual-venue families create finite selection risk.",
            "Binance spot/perpetual transfer still requires direct X-Perp evidence.",
            "A historical pass remains shadow-only pending new forward outcomes.",
        ],
    }
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({key: report[key] for key in (
        "status", "enriched_rows", "proposal_rows", "fit_prevalence_pct",
        "qualified_2022", "qualified_2023", "frozen_before_2024",
        "test_2024", "later", "later_checks")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
