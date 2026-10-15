"""Causal BTC futures depth audit for the premium-dislocation signal families."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

import crypto_5m_regime_router_literal_oct6_audit as price
import crypto_futures_funding_regime_oct6_audit as funding
import crypto_premium_dislocation_oct15_audit as premium
import crypto_premium_exit_geometry_oct15_audit as geometry


ROOT = Path("reports/audit_2026_09_08")
DEPTH_ROOT = ROOT / "crypto_btc_bookdepth_5m_oct15"
MANIFEST = DEPTH_ROOT / "manifest.json"
OUT = ROOT / "crypto_btc_bookdepth_router_oct15.json"
TARGET_PCT, STOP_PCT, MAX_BARS = .75, .75, 24
QUANTILES = (.50, .60, .70, .75, .80, .85, .90)
CONFIGS = (
    {"max_leaf_nodes": 15, "min_samples_leaf": 250, "l2": 3.0},
    {"max_leaf_nodes": 31, "min_samples_leaf": 400, "l2": 5.0},
    {"max_leaf_nodes": 31, "min_samples_leaf": 250, "l2": 8.0},
)
DEPTH_FEATURES = (
    "depth_imbalance1", "depth_imbalance2", "depth_imbalance5",
    "depth_imbalance1_mean", "depth_imbalance2_mean", "depth_imbalance5_mean",
    "depth_imbalance1_std", "depth_imbalance2_std", "depth_imbalance5_std",
    "depth_imbalance_delta12", "depth_near_share",
    "depth_total1_z", "depth_total5_z", "depth_total1_change12",
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def consecutive_same(values: pd.Series, times: pd.Series) -> pd.Series:
    same = values.eq(values.shift()) & times.sub(times.shift()).eq(300)
    groups = (~same).cumsum()
    return same.groupby(groups).cumcount()


def load_depth(candles: dict) -> tuple[pd.DataFrame, dict[str, str], dict]:
    manifest_raw = MANIFEST.read_bytes()
    manifest = json.loads(manifest_raw)
    pieces, sources = [], {str(MANIFEST): hashlib.sha256(manifest_raw).hexdigest()}
    for meta in manifest["months"]:
        path = Path(meta["path"])
        found = digest(path)
        if found != meta["sha256"]:
            raise ValueError(f"depth source hash mismatch: {path}")
        pieces.append(pd.read_pickle(path))
        sources[str(path)] = found
    frame = pd.concat(pieces, ignore_index=True).sort_values("available_time")
    frame = frame.drop_duplicates("available_time", keep="last").reset_index(drop=True)
    btc = candles["BTCUSDT"]
    market = pd.DataFrame({
        "available_time": np.asarray(btc["time"], dtype=np.int64)+300,
        "btc_close_at_depth": np.asarray(btc["close"], dtype=float),
    })
    frame = frame.merge(market, on="available_time", how="inner", validate="one_to_one")
    frame["alignment_error"] = frame.implied_mid/frame.btc_close_at_depth-1
    frame["ask_stale_bars"] = consecutive_same(frame.ask_notional1, frame.available_time)
    frame["bid_stale_bars"] = consecutive_same(frame.bid_notional1, frame.available_time)
    frame["quality_ok"] = (
        frame.alignment_error.abs().le(.02)
        & frame.ask_stale_bars.lt(12) & frame.bid_stale_bars.lt(12)
        & frame.snapshot_count.ge(5))
    total1 = frame.bid_notional1+frame.ask_notional1
    total5 = frame.bid_notional5+frame.ask_notional5
    log1, log5 = np.log(total1.replace(0, np.nan)), np.log(total5.replace(0, np.nan))
    mean1 = log1.rolling(2016, min_periods=576).mean()
    std1 = log1.rolling(2016, min_periods=576).std().replace(0, np.nan)
    mean5 = log5.rolling(2016, min_periods=576).mean()
    std5 = log5.rolling(2016, min_periods=576).std().replace(0, np.nan)
    frame["depth_total1_z"] = (log1-mean1)/std1
    frame["depth_total5_z"] = (log5-mean5)/std5
    frame["depth_total1_change12"] = total1/total1.shift(12)-1
    frame["depth_imbalance_delta12"] = frame.imbalance1-frame.imbalance1.shift(12)
    frame["depth_near_share"] = total1/total5.replace(0, np.nan)
    rename = {name: f"depth_{name}" for name in (
        "imbalance1", "imbalance2", "imbalance5", "imbalance1_mean",
        "imbalance2_mean", "imbalance5_mean", "imbalance1_std",
        "imbalance2_std", "imbalance5_std")}
    frame = frame.rename(columns=rename)
    diagnostics = {
        "rows": int(len(frame)),
        "quality_ok": int(frame.quality_ok.sum()),
        "alignment_rejected": int(frame.alignment_error.abs().gt(.02).sum()),
        "stale_rejected": int((frame.ask_stale_bars.ge(12)
                                | frame.bid_stale_bars.ge(12)).sum()),
        "sparse_rejected": int(frame.snapshot_count.lt(5).sum()),
        "alignment_abs_quantiles": {
            str(q): round(float(frame.alignment_error.abs().quantile(q)), 8)
            for q in (.5, .9, .95, .99, .999, 1.0)},
    }
    return frame, sources, diagnostics


def attach(frame: pd.DataFrame, depth: pd.DataFrame) -> pd.DataFrame:
    columns = ["available_time", "quality_ok", *DEPTH_FEATURES]
    joined = pd.merge_asof(
        frame.sort_values("entry_time"), depth[columns].sort_values("available_time"),
        left_on="entry_time", right_on="available_time", direction="backward",
        tolerance=300)
    return joined[joined.quality_ok.fillna(False)].replace(
        [np.inf, -np.inf], np.nan).dropna(subset=list(DEPTH_FEATURES))


def depth_matrix(frame: pd.DataFrame, base: np.ndarray) -> np.ndarray:
    direction = frame.direction.to_numpy(float)
    directional = {
        "depth_imbalance1", "depth_imbalance2", "depth_imbalance5",
        "depth_imbalance1_mean", "depth_imbalance2_mean", "depth_imbalance5_mean",
        "depth_imbalance_delta12",
    }
    columns = []
    for name in DEPTH_FEATURES:
        values = frame[name].to_numpy(float)
        if name in directional:
            values = values*direction
        columns.append(values[:, None])
    return np.hstack((base, np.hstack(columns).astype(np.float32)))


def gate(value: dict) -> bool:
    return bool(value["n"] >= 450 and value["return_pct"] > 0
                and value["dd_pct"] <= 9
                and value["trades_per_calendar_day"] >= 2.5)


def public(item: dict) -> dict:
    return {key: value for key, value in item.items()
            if key not in ("_frame", "_model")}


def economic_metrics(value: dict) -> dict:
    months = int(value.get("months", 0))
    total_factor = 1+float(value["return_pct"])/100
    projected = ((total_factor**(1/months)-1)*100
                 if months and total_factor > 0 else -100.0)
    n = int(value.get("n", 0))
    equivalent_price = (((total_factor**(1/n)-1)/(.03*10))*100
                        if n and total_factor > 0 else -100.0)
    return {
        "projected_monthly_return_pct": round(projected, 4),
        "equivalent_constant_signed_price_return_per_trade_pct": round(
            equivalent_price, 6),
        "required_crypto_price_return_per_trade_pct_at_3_trades_day": .973137,
        "all_actual_months_ge_30pct": bool(
            value.get("months_ge_30pct", 0) == months and months > 0),
        "frequency_at_least_3_per_day": bool(
            value.get("trades_per_calendar_day", 0) >= 3),
    }


def main() -> None:
    candles, _, candle_sources = funding.load()
    premium_data, premium_sources = premium.load_premium()
    depth, depth_sources, depth_diagnostics = load_depth(candles)
    raw = price.build_features(candles)
    proposals = premium.proposals(premium.attach(raw, premium_data))
    proposals = attach(proposals, depth)
    frame = geometry.add_outcomes(
        proposals, candles, TARGET_PCT, STOP_PCT, MAX_BARS)
    fit = frame.year.eq(2023).to_numpy()
    selection_mask = frame.year.eq(2024).to_numpy()
    schema = price.columns(frame.loc[fit])
    _, base = premium.matrices(frame, schema)
    matrices = {
        "premium_price_control": base,
        "premium_price_btc_depth": depth_matrix(frame, base),
    }
    target = frame.target_hit.astype(int).to_numpy()
    candidates, frontier = [], []
    for kind, matrix in matrices.items():
        for config in CONFIGS:
            model = HistGradientBoostingClassifier(
                max_iter=180, learning_rate=.06,
                max_leaf_nodes=config["max_leaf_nodes"],
                min_samples_leaf=config["min_samples_leaf"],
                l2_regularization=config["l2"], random_state=20261015)
            model.fit(matrix[fit], target[fit])
            scored = frame.copy()
            scored["quality"] = model.predict_proba(matrix)[:, 1]
            for quantile in QUANTILES:
                threshold = float(scored.loc[selection_mask, "quality"].quantile(quantile))
                result = price.evaluate(scored, (2024,), threshold)
                item = {
                    "model_kind": kind, "model": config, "quantile": quantile,
                    "threshold": threshold, "selection_2024": result,
                }
                frontier.append(item)
                if not gate(result):
                    continue
                item["score"] = round(
                    result["return_pct"]-result["dd_pct"]
                    +.1*result["literal_target_hit_rate"], 8)
                item["_frame"], item["_model"] = scored, model
                candidates.append(item)
    frozen = {}
    for kind in matrices:
        found = sorted((item for item in candidates if item["model_kind"] == kind),
                       key=lambda item: item["score"], reverse=True)
        if found:
            frozen[kind] = found[0]
    results = {
        kind: {str(year): price.evaluate(item["_frame"], (year,), item["threshold"])
               for year in (2023, 2024, 2025, 2026)}
        for kind, item in frozen.items()}
    economic = {kind: {year: economic_metrics(value)
                       for year, value in yearly.items()}
                for kind, yearly in results.items()}
    control = results.get("premium_price_control")
    depth_result = results.get("premium_price_btc_depth")
    confirmation_better = bool(control and depth_result and all(
        depth_result[str(year)]["return_pct"] > control[str(year)]["return_pct"]
        and depth_result[str(year)]["dd_pct"] <= control[str(year)]["dd_pct"]
        for year in (2025, 2026)))
    checks = ({str(year): price.objective_ok(value)
               for year, value in depth_result.items()} if depth_result else None)
    status = ("OBJECTIVE_PASS" if checks and all(checks.values())
              else "RESEARCH_CANDIDATE" if confirmation_better
              else "REJECTED_CONFIRMATION")
    report = {
        "status": status, "ready_for_live_orders": False,
        "decision": "No runtime change unless chronological objectives pass.",
        "protocol": {
            "fit": 2023, "selection": 2024, "confirmation_only": [2025, 2026],
            "entry": "next 5m market open", "target_pct": TARGET_PCT,
            "stop_pct": STOP_PCT, "max_bars": MAX_BARS,
            "ambiguity": "stop first", "features": DEPTH_FEATURES,
            "depth_availability": "last complete 5m depth bucket before entry",
            "quality": "price alignment <=2%, <12 causally stale bars, >=5 snapshots",
            "position": "3% equity margin, fixed 10x",
        },
        "depth_diagnostics": depth_diagnostics,
        "rows_after_causal_join": len(frame),
        "fit_target_prevalence_pct": round(100*target[fit].mean(), 4),
        "qualified_before_confirmation": len(candidates),
        "frozen_before_confirmation": {kind: public(item) for kind, item in frozen.items()},
        "results": results, "depth_beats_control_2025_2026": confirmation_better,
        "economic_gate": economic,
        "objective_checks": checks,
        "selection_frontier": [public(item) for item in sorted(
            frontier, key=lambda item: item["selection_2024"]["return_pct"],
            reverse=True)[:24]],
        "sources": {
            **candle_sources, **premium_sources, **depth_sources,
            str(Path(__file__)): digest(Path(__file__)),
        },
        "limitations": [
            "BTC aggregate depth is market context, not per-altcoin order-book depth.",
            "The public depth archive begins in 2023, so no 2020-2022 fit is possible.",
            "Known archive staleness and price misalignment are causally excluded.",
            "Binance futures remain a proxy for OKX X-Perp execution.",
        ],
    }
    OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(json.dumps({
        "status": status, "depth_diagnostics": depth_diagnostics,
        "rows": len(frame), "qualified": len(candidates),
        "depth_beats_control": confirmation_better, "results": results,
    }, indent=2))


if __name__ == "__main__":
    main()
