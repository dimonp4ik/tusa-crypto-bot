"""Causal keep-or-reverse router for frozen expert-agreement entries.

The router learns only whether the proposed direction should be kept.  Its new
information is multi-horizon serial dependence and variance-ratio structure;
entry selection and exit geometry remain frozen.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

import crypto_5m_regime_router_literal_oct6_audit as price
import crypto_futures_trade_flow_regime_oct7_audit as flow


ROOT = Path("reports/audit_2026_09_08")
BASE_REPORT = ROOT / "crypto_independent_expert_agreement_oct7.json"
BASE_ROWS = ROOT / "crypto_independent_expert_agreement_oct7.pkl"
OUT = ROOT / "crypto_serial_regime_direction_router_oct7.json"
ROWS_OUT = ROOT / "crypto_serial_regime_direction_router_oct7.pkl"
CONFIGS = (
    {"leaves": 15, "leaf": 300, "l2": 3.0},
    {"leaves": 31, "leaf": 500, "l2": 5.0},
)
FLIP_QUANTILES = (0.0, .05, .10, .15, .20, .25, .30, .35, .40)
SERIAL_FEATURES = (
    "sym_ac72", "sym_ac288", "sym_vr12", "sym_vr72",
    "sym_sign_persistence72", "sym_efficiency288", "sym_rv_ratio",
    "btc_ac72", "btc_ac288", "btc_vr12", "btc_vr72",
    "btc_sign_persistence72", "btc_efficiency288", "btc_rv_ratio",
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def symbol_serial(candle: dict, prefix: str) -> pd.DataFrame:
    close = pd.Series(np.asarray(candle["close"], dtype=float))
    ret = np.log(close).diff()
    sign = np.sign(ret)
    one_std = ret.rolling(2016, min_periods=1000).std(ddof=0)
    ret12 = np.log(close / close.shift(12))
    ret72 = np.log(close / close.shift(72))
    path288 = close.diff().abs().rolling(288, min_periods=216).sum()
    result = pd.DataFrame({"signal_time": np.asarray(candle["time"], dtype=np.int64)})
    result[f"{prefix}_ac72"] = ret.rolling(72, min_periods=54).corr(ret.shift(1))
    result[f"{prefix}_ac288"] = ret.rolling(288, min_periods=216).corr(ret.shift(1))
    result[f"{prefix}_vr12"] = ret12.rolling(2016, min_periods=1000).std(ddof=0) / (np.sqrt(12) * one_std)
    result[f"{prefix}_vr72"] = ret72.rolling(2016, min_periods=1000).std(ddof=0) / (np.sqrt(72) * one_std)
    result[f"{prefix}_sign_persistence72"] = sign.rolling(72, min_periods=54).mean().abs()
    result[f"{prefix}_efficiency288"] = (close - close.shift(288)).abs() / path288
    result[f"{prefix}_rv_ratio"] = ret.rolling(72, min_periods=54).std(ddof=0) / one_std
    return result.replace([np.inf, -np.inf], np.nan).dropna()


def attach_serial(frame: pd.DataFrame, candles: dict) -> pd.DataFrame:
    btc = symbol_serial(candles["BTCUSDT"], "btc")
    parts = []
    for symbol, part in frame.groupby("symbol", sort=False):
        left = part.copy()
        left["signal_time"] = left.entry_time.astype(np.int64) - 300
        found = left.merge(symbol_serial(candles[symbol], "sym"),
                           on="signal_time", how="inner", validate="many_to_one")
        found = found.merge(btc, on="signal_time", how="inner",
                            validate="many_to_one")
        parts.append(found)
    return pd.concat(parts, ignore_index=True).replace(
        [np.inf, -np.inf], np.nan).dropna(subset=list(SERIAL_FEATURES))


def reverse_outcomes(frame: pd.DataFrame, candles: dict) -> pd.DataFrame:
    clean = frame.drop(columns=[name for name in (
        "signed_return", "exit_time", "target_hit", "exit_reason")
        if name in frame]).copy()
    clean["direction"] = -clean.direction
    return price.add_outcomes(clean, candles)


def matrix(frame: pd.DataFrame) -> np.ndarray:
    direction = frame.direction.to_numpy(float)
    columns = []
    for name in SERIAL_FEATURES:
        values = frame[name].to_numpy(float)
        columns.extend((values[:, None], (values * direction)[:, None]))
    columns.extend((
        frame.ret12_dispersion.to_numpy(float)[:, None],
        frame.vol_breadth_expanding.to_numpy(float)[:, None],
        frame.source_count.to_numpy(float)[:, None],
    ))
    for prefix, names in (
        ("family_", price.FAMILIES),
        ("flow_family_", flow.FAMILIES),
    ):
        for name in names:
            columns.append(frame[f"{prefix}{name}"].to_numpy(float)[:, None])
    return np.hstack(columns).astype(np.float32)


def hybrid(original: pd.DataFrame, reverse: pd.DataFrame,
           prediction: np.ndarray, threshold: float) -> pd.DataFrame:
    output = original.copy()
    flip = prediction < threshold
    for name in ("direction", "signed_return", "exit_time", "target_hit", "exit_reason"):
        output.loc[flip, name] = reverse.loc[flip, name].to_numpy()
    output["direction_edge"] = prediction
    output["direction_flipped"] = flip
    return output


def acceptable(value: dict, validation: bool = False) -> bool:
    return bool(value["n"] >= (500 if validation else 700)
                and value["return_pct"] > 0
                and value["literal_target_hit_rate"] >= (66 if validation else 69)
                and value["dd_pct"] <= 7
                and value["trades_per_calendar_day"] >= (2.0 if validation else 2.5))


def main() -> None:
    freeze = json.loads(BASE_REPORT.read_text(encoding="utf-8"))
    if not freeze.get("frozen_before_2024"):
        raise ValueError("base expert agreement was not frozen")
    candles, candle_sources = flow.load()
    original = attach_serial(pd.read_pickle(BASE_ROWS), candles).reset_index(drop=True)
    reverse = reverse_outcomes(original, candles).reset_index(drop=True)
    x = matrix(original)
    target = (original.signed_return - reverse.signed_return).to_numpy(float)
    fit = original.year.isin((2020, 2021)).to_numpy()
    selection = original.year.eq(2022).to_numpy()
    candidates = []
    diagnostics = []
    cache = {}
    for config in CONFIGS:
        model = HistGradientBoostingRegressor(
            max_iter=180, learning_rate=.06, max_leaf_nodes=config["leaves"],
            min_samples_leaf=config["leaf"], l2_regularization=config["l2"],
            random_state=20261007)
        model.fit(x[fit], target[fit])
        prediction = model.predict(x)
        for quantile in FLIP_QUANTILES:
            threshold = (-np.inf if quantile == 0 else
                         float(np.quantile(prediction[selection], quantile)))
            routed = hybrid(original, reverse, prediction, threshold)
            key = (config["leaves"], config["leaf"], quantile)
            cache[key] = routed
            value = price.evaluate(routed, (2022,), 0)
            item = {"model": config, "flip_quantile": quantile,
                    "threshold": threshold, "selection_2022": value,
                    "selection_flip_rate": round(100 * float(
                        routed.loc[routed.year.eq(2022), "direction_flipped"].mean()), 4)}
            diagnostics.append(item)
            if acceptable(value):
                candidates.append({**item, "_key": key})
    validated = []
    validation_diagnostics = []
    for item in candidates:
        routed = cache[item["_key"]]
        value = price.evaluate(routed, (2023,), 0)
        candidate = {**item, "validation_2023": value,
                     "validation_flip_rate": round(100 * float(
                         routed.loc[routed.year.eq(2023), "direction_flipped"].mean()), 4)}
        validation_diagnostics.append({k: v for k, v in candidate.items() if k != "_key"})
        if acceptable(value, validation=True):
            candidate["score"] = round(
                min(item["selection_2022"]["return_pct"], value["return_pct"])
                - 2 * max(item["selection_2022"]["dd_pct"], value["dd_pct"]), 8)
            validated.append(candidate)
    validated.sort(key=lambda item: item["score"], reverse=True)
    frozen = validated[0] if validated else None
    test_2024 = later = checks = None
    if frozen:
        routed = cache[frozen["_key"]]
        test_2024 = price.evaluate(routed, (2024,), 0)
        if acceptable(test_2024, validation=True):
            later = {str(year): price.evaluate(routed, (year,), 0)
                     for year in (2025, 2026)}
            checks = {year: acceptable(value, validation=True)
                      for year, value in later.items()}
        routed.to_pickle(ROWS_OUT)
    public_frozen = None if frozen is None else {
        key: value for key, value in frozen.items() if key != "_key"}
    report = {
        "status": ("POST_TEST_FORWARD_CANDIDATE"
                   if later and all(value["return_pct"] > 0 for value in later.values())
                   else "REJECTED"),
        "protocol": {
            "base_entry_filter": "frozen independent expert agreement",
            "router_fit": [2020, 2021], "selection": 2022,
            "validation_and_ranking": 2023, "test": 2024,
            "later_diagnostic": [2025, 2026],
            "warning": "This branch was conceived after the 2026 base failure was known; 2024-2026 are not pristine holdouts for the new router.",
            "entry": "next 5m market open", "target_pct": .5,
            "stop_pct": 1.0, "max_bars": 24,
            "money": "3% equity margin, fixed 10x gross",
        },
        "rows": len(original), "fit_rows": int(fit.sum()),
        "selection_diagnostics": diagnostics,
        "qualified_2022": len(candidates), "qualified_2023": len(validated),
        "validation_diagnostics": validation_diagnostics,
        "frozen_before_test": public_frozen, "test_2024": test_2024,
        "later": later, "later_checks": checks,
        "sources": {**candle_sources, str(BASE_REPORT): digest(BASE_REPORT),
                    str(BASE_ROWS): digest(BASE_ROWS),
                    str(Path(__file__)): digest(Path(__file__))},
        "limitations": [
            "The branch is a post-2026 diagnostic and needs new forward data.",
            "Direction reversal uses the same literal target/stop execution model as the base audit.",
        ],
    }
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({key: report[key] for key in (
        "status", "rows", "qualified_2022", "qualified_2023",
        "frozen_before_test", "test_2024", "later", "later_checks")},
        indent=2), flush=True)


if __name__ == "__main__":
    main()
