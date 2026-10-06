"""Literal-target crypto regimes from 5m taker and trade-flow information."""
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


ROOT = Path("reports/audit_2026_09_08/crypto_futures_flow_history")
OUT = Path("reports/audit_2026_09_08/crypto_futures_trade_flow_regime_oct7.json")
ROWS_OUT = Path("reports/audit_2026_09_08/crypto_futures_trade_flow_regime_oct7.pkl")
FAMILIES = ("flow_continuation", "flow_absorption", "large_trade_breakout",
            "cvd_divergence", "flow_exhaustion", "cross_flow_leader")
FLOW_FEATURES = (
    "imbalance", "imbalance_mean3", "imbalance_mean12", "imbalance_mean36",
    "imbalance_mean72", "imbalance_z288", "quote_imbalance", "cvd12", "cvd72",
    "trade_count_ratio", "quote_volume_ratio", "avg_trade_ratio",
    "flow_price_alignment", "imbalance_persistence12", "flow_cross_z",
    "cvd_cross_z",
)
CONFIGS = (
    {"leaves": 15, "leaf": 250, "l2": 3.0},
    {"leaves": 31, "leaf": 400, "l2": 5.0},
)
QUANTILES = (.70, .75, .80, .85, .90, .925, .95, .96, .97, .98, .985, .99)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load() -> tuple[dict, dict]:
    manifest_path = ROOT / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    candles, sources = {}, {str(manifest_path): digest(manifest_path)}
    for symbol in futures_source.SYMBOLS:
        path = Path(manifest["symbols"][symbol]["path"])
        raw = path.read_bytes()
        found = hashlib.sha256(raw).hexdigest()
        if found != manifest["symbols"][symbol]["sha256"]:
            raise ValueError(f"flow hash mismatch: {symbol}")
        candles[symbol] = pickle.loads(raw)
        sources[str(path)] = found
    return candles, sources


def symbol_flow(candle: dict) -> pd.DataFrame:
    frame = pd.DataFrame({key: np.asarray(value, dtype=float)
                          for key, value in candle.items()})
    frame["signal_time"] = frame.pop("time").astype(np.int64)
    volume = frame.volume.replace(0, np.nan)
    quote_volume = frame.quote_volume.replace(0, np.nan)
    trades = frame.trade_count.replace(0, np.nan)
    frame["imbalance"] = 2*frame.taker_buy_volume/volume-1
    frame["quote_imbalance"] = 2*frame.taker_buy_quote_volume/quote_volume-1
    for window in (3, 12, 36, 72):
        frame[f"imbalance_mean{window}"] = frame.imbalance.rolling(
            window, min_periods=max(2, int(.75*window))).mean()
    mean288 = frame.imbalance.rolling(288, min_periods=192).mean()
    std288 = frame.imbalance.rolling(288, min_periods=192).std().replace(0, np.nan)
    frame["imbalance_z288"] = (frame.imbalance-mean288)/std288
    signed_quote = 2*frame.taker_buy_quote_volume-frame.quote_volume
    frame["cvd12"] = signed_quote.rolling(12, min_periods=9).sum() / quote_volume.rolling(
        12, min_periods=9).sum()
    frame["cvd72"] = signed_quote.rolling(72, min_periods=54).sum() / quote_volume.rolling(
        72, min_periods=54).sum()
    frame["trade_count_ratio"] = trades/trades.shift(1).rolling(
        288, min_periods=192).median()
    frame["quote_volume_ratio"] = quote_volume/quote_volume.shift(1).rolling(
        288, min_periods=192).median()
    avg_trade = quote_volume/trades
    frame["avg_trade_ratio"] = avg_trade/avg_trade.shift(1).rolling(
        288, min_periods=192).median()
    ret1 = frame.close/frame.close.shift(1)-1
    frame["flow_price_alignment"] = ret1*frame.imbalance
    frame["imbalance_persistence12"] = frame.imbalance.rolling(
        12, min_periods=9).apply(lambda value: abs(np.sign(value).mean()), raw=True)
    keep = ["signal_time", *[name for name in FLOW_FEATURES
                              if name not in ("flow_cross_z", "cvd_cross_z")]]
    return frame[keep].replace([np.inf, -np.inf], np.nan).dropna()


def attach_flow(frame: pd.DataFrame, candles: dict) -> pd.DataFrame:
    pieces = []
    for symbol, part in frame.groupby("symbol", sort=False):
        left = part.copy()
        left["signal_time"] = left.entry_time.astype(np.int64)-300
        right = symbol_flow(candles[symbol])
        pieces.append(left.merge(right, on="signal_time", how="inner",
                                 validate="one_to_one"))
    result = pd.concat(pieces, ignore_index=True)
    grouped = result.groupby("entry_time", sort=False)
    for source, target in (("imbalance_mean3", "flow_cross_z"),
                           ("cvd12", "cvd_cross_z")):
        mean = grouped[source].transform("mean")
        std = grouped[source].transform("std").replace(0, np.nan)
        result[target] = (result[source]-mean)/std
    return result.replace([np.inf, -np.inf], np.nan).dropna(
        subset=list(FLOW_FEATURES))


def proposals(frame: pd.DataFrame) -> pd.DataFrame:
    pieces = []

    def add(name: str, direction: pd.Series, mask: pd.Series) -> None:
        selected = mask & direction.ne(0)
        part = frame.loc[selected].copy()
        part["direction"] = direction.loc[selected].astype(np.int8)
        part[f"flow_family_{name}"] = 1
        pieces.append(part)

    aggressive = np.sign(frame.imbalance_mean3)
    add("flow_continuation", aggressive,
        frame.imbalance_mean3.abs().ge(.07)
        & (frame.ret3*aggressive).ge(.0008)
        & frame.quote_volume_ratio.ge(.9) & frame.trade_count_ratio.ge(.8))

    reversal = -np.sign(frame.imbalance)
    add("flow_absorption", reversal,
        frame.imbalance.abs().ge(.20) & frame.quote_volume_ratio.ge(1.15)
        & frame.range_ratio.ge(1.0) & (frame.body1*reversal).gt(0))

    breakout = np.sign(frame.ret3)
    add("large_trade_breakout", breakout,
        frame.ret3.abs().ge(.0015) & frame.avg_trade_ratio.ge(1.20)
        & frame.quote_volume_ratio.ge(1.10)
        & (frame.imbalance_mean3*breakout).ge(.04))

    cvd_direction = np.sign(frame.cvd12)
    add("cvd_divergence", cvd_direction,
        frame.cvd12.abs().ge(.04) & (frame.ret12*cvd_direction).le(-.001)
        & frame.quote_volume_ratio.ge(.8))

    move = np.sign(frame.ret12)
    exhaustion = -move
    rejection = frame.lower_wick.where(exhaustion > 0, frame.upper_wick)
    add("flow_exhaustion", exhaustion,
        frame.ret12.abs().ge(.004) & (frame.imbalance_z288*move).ge(1.0)
        & rejection.ge(.18) & (frame.body1*exhaustion).gt(0))

    leader = np.sign(frame.flow_cross_z)
    add("cross_flow_leader", leader,
        frame.flow_cross_z.abs().ge(1.0) & (frame.ret3*leader).ge(.0005)
        & frame.quote_volume_ratio.ge(.9))

    combined = pd.concat(pieces, ignore_index=True)
    for family in FAMILIES:
        column = f"flow_family_{family}"
        if column not in combined:
            combined[column] = 0
    keys = ["symbol", "entry_time", "direction"]
    flags = combined.groupby(keys, sort=False)[
        [f"flow_family_{family}" for family in FAMILIES]].max().reset_index()
    base_rows = combined.sort_values(keys).drop_duplicates(keys).drop(
        columns=[f"flow_family_{family}" for family in FAMILIES])
    result = base_rows.merge(flags, on=keys, how="left", validate="one_to_one")
    for family in price.FAMILIES:
        result[f"family_{family}"] = 0
    return result


def matrices(frame: pd.DataFrame, schema: list[str]) -> tuple[np.ndarray, np.ndarray]:
    base_matrix = price.matrix(frame, schema)
    direction = frame.direction.to_numpy(float)
    columns = []
    directional = {"imbalance", "imbalance_mean3", "imbalance_mean12",
                   "imbalance_mean36", "imbalance_mean72", "imbalance_z288",
                   "quote_imbalance", "cvd12", "cvd72", "flow_cross_z",
                   "cvd_cross_z"}
    for name in FLOW_FEATURES:
        values = frame[name].to_numpy(float)
        columns.append((values*direction if name in directional else values)[:, None])
        if name in directional:
            columns.append(np.abs(values)[:, None])
    for family in FAMILIES:
        columns.append(frame[f"flow_family_{family}"].to_numpy(float)[:, None])
    return base_matrix, np.hstack(
        (base_matrix, np.hstack(columns).astype(np.float32)))


def acceptable(value: dict, *, validation: bool) -> bool:
    return bool(value["n"] >= (250 if validation else 700)
                and value["return_pct"] > 0
                and value["literal_target_hit_rate"] >= (66 if validation else 69)
                and value["dd_pct"] <= 8
                and value["trades_per_calendar_day"] >= (2.0 if validation else 3.0))


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
    candles, sources = load()
    raw = price.build_features(candles)
    enriched = attach_flow(raw, candles)
    frame = price.add_outcomes(proposals(enriched), candles)
    fit = frame.year.isin((2020, 2021)).to_numpy()
    selection = frame.year.eq(2022).to_numpy()
    schema = price.columns(frame.loc[fit])
    x_price, x_full = matrices(frame, schema)
    y = frame.target_hit.astype(int).to_numpy()
    candidates, frontier, validation_diagnostics = [], [], []
    for model_kind, matrix in (("price_only_control", x_price),
                               ("trade_flow_plus_price", x_full)):
        for config in CONFIGS:
            model = HistGradientBoostingClassifier(
                max_iter=180, learning_rate=.06, max_leaf_nodes=config["leaves"],
                min_samples_leaf=config["leaf"], l2_regularization=config["l2"],
                random_state=20261007)
            model.fit(matrix[fit], y[fit])
            scored = frame.copy()
            scored["quality"] = model.predict_proba(matrix)[:, 1]
            for quantile in QUANTILES:
                threshold = float(scored.loc[selection, "quality"].quantile(quantile))
                s22 = price.evaluate(scored, (2022,), threshold)
                item = {"model_kind": model_kind, "model": config,
                        "quantile": quantile, "threshold": threshold,
                        "selection_2022": s22}
                frontier.append(item)
                if not acceptable(s22, validation=False):
                    continue
                item["validation_2023"] = price.evaluate(scored, (2023,), threshold)
                item["validation_2024"] = price.evaluate(scored, (2024,), threshold)
                if not (acceptable(item["validation_2023"], validation=True)
                        and acceptable(item["validation_2024"], validation=True)):
                    validation_diagnostics.append(public(item))
                    continue
                item["score"] = round(sum(
                    item[name]["return_pct"]-item[name]["dd_pct"]
                    for name in ("selection_2022", "validation_2023",
                                 "validation_2024")), 8)
                item["_frame"], item["_model"] = scored, model
                candidates.append(item)
    candidates.sort(key=lambda item: item["score"], reverse=True)
    flow_candidates = [item for item in candidates
                       if item["model_kind"] == "trade_flow_plus_price"]
    frozen = flow_candidates[0] if flow_candidates else None
    control = next((item for item in candidates
                    if item["model_kind"] == "price_only_control"), None)
    results = checks = None
    if frozen:
        threshold = frozen["threshold"]
        results = {str(year): price.evaluate(frozen["_frame"], (year,), threshold)
                   for year in (2022, 2023, 2024, 2025, 2026)}
        checks = {year: objective_ok(value) for year, value in results.items()}
        frozen["_frame"][frozen["_frame"].quality.ge(threshold)].to_pickle(
            ROWS_OUT, compression="gzip")
    status = ("RESEARCH_CANDIDATE_AWAITING_VENUE_FORWARD"
              if checks and all(checks.values()) else "REJECTED")
    report = {
        "status": status,
        "protocol": {"fit": "2020-2021", "selection": "2022",
                     "validation": ["2023", "2024"], "test": "2025",
                     "observed": "2026 through September",
                     "entry": "next 5m futures market open",
                     "target_pct": .5, "stop_pct": 1.0, "max_bars": 24,
                     "money": "3% equity margin, fixed 10x gross",
                     "win_definition": "literal target touch before stop/timeout"},
        "rows": {"raw": len(raw), "enriched": len(enriched),
                 "proposals": len(frame)},
        "fit_literal_prevalence_pct": round(100*float(y[fit].mean()), 4),
        "qualified_before_test": len(candidates),
        "frozen_flow_before_test": public(frozen),
        "best_price_control_before_test": public(control),
        "top_before_test": [public(item) for item in candidates[:20]],
        "selection_frontier": [public(item) for item in sorted(
            frontier, key=lambda row: (
                row["selection_2022"]["literal_target_hit_rate"],
                row["selection_2022"]["trades_per_calendar_day"]),
            reverse=True)[:30]],
        "validation_diagnostics": validation_diagnostics,
        "results": results, "checks": checks,
        "sources": {**sources, str(Path(__file__)): digest(Path(__file__))},
        "limitations": [
            "Kline taker flow is aggregated and cannot reconstruct the order book.",
            "The bounded family/model/threshold grid creates selection risk.",
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
