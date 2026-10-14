"""Causal literal-target audit of mark/index premium dislocations."""
from __future__ import annotations

import hashlib
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

import crypto_5m_regime_router_literal_oct6_audit as price
import crypto_futures_funding_regime_oct6_audit as funding


ROOT = Path("reports/audit_2026_09_08")
PREMIUM_ROOT = ROOT / "crypto_premium_index_15m_oct15"
MANIFEST = PREMIUM_ROOT / "manifest.json"
OUT = ROOT / "crypto_premium_dislocation_oct15.json"
FAMILIES = ("premium_snapback", "premium_sweep",
            "premium_lead", "relative_reversion")
QUANTILES = (.50, .60, .70, .75, .80, .85, .90)
CONFIGS = (
    {"max_leaf_nodes": 15, "min_samples_leaf": 300, "l2": 3.0},
    {"max_leaf_nodes": 31, "min_samples_leaf": 500, "l2": 5.0},
)
PREMIUM_FEATURES = (
    "premium_close", "premium_z672", "premium_delta4",
    "premium_delta4_z", "premium_delta16", "premium_high_z",
    "premium_low_z", "premium_range_ratio", "premium_position",
    "premium_cross_z", "btc_premium", "relative_premium",
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_premium() -> tuple[dict[str, dict], dict[str, str]]:
    manifest_raw = MANIFEST.read_bytes()
    manifest = json.loads(manifest_raw)
    data, sources = {}, {str(MANIFEST): hashlib.sha256(manifest_raw).hexdigest()}
    for symbol in funding.SYMBOLS:
        meta = manifest["symbols"][symbol]
        path = Path(meta["path"])
        found = digest(path)
        if found != meta["sha256"]:
            raise ValueError(f"premium source hash mismatch: {symbol}")
        data[symbol] = pickle.loads(path.read_bytes())
        sources[str(path)] = found
    return data, sources


def premium_features(data: dict) -> pd.DataFrame:
    frame = pd.DataFrame({key: np.asarray(data[key], dtype=float)
                          for key in ("open", "high", "low", "close")})
    frame["premium_time"] = np.asarray(data["time"], dtype=np.int64)
    frame["premium_available_time"] = frame.premium_time+900
    mean = frame.close.rolling(672, min_periods=480).mean()
    std = frame.close.rolling(672, min_periods=480).std().replace(0, np.nan)
    delta_std = frame.close.diff().rolling(
        672, min_periods=480).std().replace(0, np.nan)
    span = (frame.high-frame.low).abs()
    frame["premium_close"] = frame.close
    frame["premium_z672"] = (frame.close-mean)/std
    frame["premium_high_z"] = (frame.high-mean)/std
    frame["premium_low_z"] = (frame.low-mean)/std
    frame["premium_delta4"] = frame.close-frame.close.shift(4)
    frame["premium_delta4_z"] = frame.premium_delta4/delta_std/2
    frame["premium_delta16"] = frame.close-frame.close.shift(16)
    frame["premium_range_ratio"] = span/span.shift(1).rolling(
        192, min_periods=128).median().replace(0, np.nan)
    frame["premium_position"] = (frame.close-frame.low)/span.replace(0, np.nan)
    return frame[["premium_available_time", *PREMIUM_FEATURES[:9]]].dropna()


def attach(frame: pd.DataFrame, premium: dict[str, dict]) -> pd.DataFrame:
    pieces = []
    for symbol, part in frame.groupby("symbol", sort=False):
        left = part.sort_values("entry_time").copy()
        right = premium_features(premium[str(symbol)]).sort_values(
            "premium_available_time")
        found = pd.merge_asof(
            left, right, left_on="entry_time",
            right_on="premium_available_time", direction="backward",
            tolerance=1800)
        pieces.append(found)
    result = pd.concat(pieces, ignore_index=True).dropna(
        subset=list(PREMIUM_FEATURES[:9]))
    grouped = result.groupby("entry_time", sort=False)
    mean = grouped.premium_close.transform("mean")
    std = grouped.premium_close.transform("std").replace(0, np.nan)
    result["premium_cross_z"] = (result.premium_close-mean)/std
    btc = result[result.symbol.eq("BTCUSDT")][
        ["entry_time", "premium_close"]].rename(
            columns={"premium_close": "btc_premium"})
    result = result.merge(btc, on="entry_time", how="left",
                          validate="many_to_one")
    result["relative_premium"] = result.premium_close-result.btc_premium
    return result.replace([np.inf, -np.inf], np.nan).dropna(
        subset=list(PREMIUM_FEATURES))


def proposals(frame: pd.DataFrame) -> pd.DataFrame:
    pieces = []

    def add(name: str, direction: pd.Series, condition: pd.Series) -> None:
        found = frame.loc[condition & direction.ne(0)].copy()
        found["direction"] = direction.loc[found.index].astype(np.int8)
        found[f"premium_family_{name}"] = 1
        pieces.append(found)

    premium_sign = np.sign(frame.premium_z672)
    snapback_direction = -premium_sign
    add("premium_snapback", snapback_direction,
        frame.premium_z672.abs().ge(1.25)
        & (frame.premium_delta4*premium_sign).lt(0)
        & (frame.ret12*premium_sign).ge(.002))

    sweep_direction = pd.Series(0, index=frame.index, dtype=np.int8)
    sweep_direction.loc[(frame.premium_low_z <= -2)
                        & (frame.premium_position >= .65)] = 1
    sweep_direction.loc[(frame.premium_high_z >= 2)
                        & (frame.premium_position <= .35)] = -1
    add("premium_sweep", sweep_direction,
        frame.premium_range_ratio.ge(1.5))

    lead_direction = np.sign(frame.premium_delta4_z)
    add("premium_lead", lead_direction,
        frame.premium_delta4_z.abs().ge(1.5)
        & (frame.ret3*lead_direction).between(-.002, .0015)
        & (frame.ret72*lead_direction).ge(0))

    relative_sign = np.sign(frame.premium_cross_z)
    relative_direction = -relative_sign
    add("relative_reversion", relative_direction,
        frame.premium_cross_z.abs().ge(1.25)
        & (frame.premium_delta4*relative_sign).lt(0))

    combined = pd.concat(pieces, ignore_index=True)
    keys = ["symbol", "entry_time", "direction"]
    for family in FAMILIES:
        column = f"premium_family_{family}"
        if column not in combined:
            combined[column] = 0
    flags = combined.groupby(keys, sort=False)[
        [f"premium_family_{family}" for family in FAMILIES]].max().reset_index()
    base = combined.sort_values(keys).drop_duplicates(keys).drop(
        columns=[f"premium_family_{family}" for family in FAMILIES])
    result = base.merge(flags, on=keys, how="left", validate="one_to_one")
    for family in FAMILIES:
        column = f"premium_family_{family}"
        result[column] = result[column].fillna(0).astype(np.int8)
    for family in price.FAMILIES:
        result[f"family_{family}"] = 0
    return result


def matrices(frame: pd.DataFrame, schema: list[str]) -> tuple[np.ndarray, np.ndarray]:
    control = price.matrix(frame, schema)
    direction = frame.direction.to_numpy(float)
    directed = [
        (frame[name].to_numpy(float)*direction)[:, None]
        for name in PREMIUM_FEATURES]
    absolute = [
        frame[name].abs().to_numpy(float)[:, None]
        for name in ("premium_close", "premium_z672", "premium_delta4_z",
                     "premium_cross_z", "relative_premium")]
    family = [frame[f"premium_family_{name}"].to_numpy(float)[:, None]
              for name in FAMILIES]
    extra = np.hstack([*directed, *absolute, *family]).astype(np.float32)
    return control, np.hstack((control, extra))


def selection_ok(value: dict) -> bool:
    return bool(value["n"] >= 500 and value["return_pct"] > 0
                and value["literal_target_hit_rate"] >= 65
                and value["dd_pct"] <= 12
                and value["trades_per_calendar_day"] >= 2.5)


def validation_ok(value: dict) -> bool:
    return bool(value["n"] >= 350 and value["return_pct"] > 0
                and value["literal_target_hit_rate"] >= 65
                and value["dd_pct"] <= 10
                and value["trades_per_calendar_day"] >= 2.0)


def public(item: dict | None) -> dict | None:
    if item is None:
        return None
    return {key: value for key, value in item.items()
            if key not in ("_model", "_frame")}


def main() -> None:
    candles, _, candle_sources = funding.load()
    premium, premium_sources = load_premium()
    raw = price.build_features(candles)
    enriched = attach(raw, premium)
    frame = price.add_outcomes(proposals(enriched), candles)
    family_baseline = {}
    for family in FAMILIES:
        subset = frame[frame[f"premium_family_{family}"].eq(1)].copy()
        subset["quality"] = 1.0
        family_baseline[family] = {
            str(year): price.evaluate(subset, (year,), .5)
            for year in (2020, 2021, 2022, 2023, 2024, 2025, 2026)}
    fit = frame.year.isin((2020, 2021)).to_numpy()
    selection_mask = frame.year.eq(2022).to_numpy()
    schema = price.columns(frame.loc[fit])
    control_matrix, premium_matrix = matrices(frame, schema)
    target = frame.target_hit.astype(int).to_numpy()
    candidates, frontier = [], []
    for kind, matrix in (("price_control", control_matrix),
                         ("premium_plus_price", premium_matrix)):
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
                threshold = float(scored.loc[selection_mask, "quality"].quantile(
                    quantile))
                selection = price.evaluate(scored, (2022,), threshold)
                item = {"model_kind": kind, "model": config,
                        "quantile": quantile, "threshold": threshold,
                        "selection_2022": selection}
                frontier.append(item)
                if not selection_ok(selection):
                    continue
                item["validation_2023"] = price.evaluate(
                    scored, (2023,), threshold)
                item["validation_2024"] = price.evaluate(
                    scored, (2024,), threshold)
                if not (validation_ok(item["validation_2023"])
                        and validation_ok(item["validation_2024"])):
                    continue
                item["score"] = round(sum(
                    item[name]["return_pct"]-item[name]["dd_pct"]
                    for name in ("selection_2022", "validation_2023",
                                 "validation_2024")), 8)
                item["_model"], item["_frame"] = model, scored
                candidates.append(item)
    candidates.sort(key=lambda item: item["score"], reverse=True)
    premium_candidates = [item for item in candidates
                          if item["model_kind"] == "premium_plus_price"]
    frozen = premium_candidates[0] if premium_candidates else None
    results = checks = None
    frozen_family_composition = None
    if frozen:
        results = {str(year): price.evaluate(
            frozen["_frame"], (year,), frozen["threshold"])
            for year in (2022, 2023, 2024, 2025, 2026)}
        checks = {year: price.objective_ok(value)
                  for year, value in results.items()}
        frozen_family_composition = {}
        for year in (2022, 2023, 2024, 2025, 2026):
            accepted = price.portfolio_rows(
                frozen["_frame"], (year,), frozen["threshold"])
            frozen_family_composition[str(year)] = {
                family: sum(int(row[f"premium_family_{family}"])
                            for row in accepted)
                for family in FAMILIES}
    status = ("OBJECTIVE_PASS" if checks and all(checks.values())
              else "REJECTED_FORWARD" if frozen else "REJECTED_PRETEST")
    frontier.sort(key=lambda item: (
        item["selection_2022"]["return_pct"],
        item["selection_2022"]["literal_target_hit_rate"],
        -item["selection_2022"]["dd_pct"]), reverse=True)
    report = {
        "status": status, "ready_for_live_orders": False,
        "decision": "No runtime change unless all chronological objectives pass.",
        "protocol": {
            "source": "official Binance UM premiumIndexKlines 15m archives",
            "fit": "2020-2021", "selection": "2022",
            "validation": ["2023", "2024"], "test": "2025",
            "observed": "2026", "entry": "next 5m futures market open",
            "target_pct": .5, "stop_pct": 1.0, "max_bars": 24,
            "money": "3% current-equity margin, fixed 10x gross",
            "causality": "premium bar joins only after its 15m close time",
        },
        "rows": {"enriched": len(enriched), "proposals": len(frame)},
        "qualified_before_test": len(candidates),
        "qualified_premium_before_test": len(premium_candidates),
        "frozen_before_test": public(frozen),
        "best_price_control": next((public(item) for item in candidates
                                    if item["model_kind"] == "price_control"), None),
        "family_baseline": family_baseline,
        "frozen_family_composition": frozen_family_composition,
        "results": results, "objective_checks": checks,
        "top_selection_frontier": [public(item) for item in frontier[:20]],
        "sources": {**candle_sources, **premium_sources,
                    str(Path(__file__)): digest(Path(__file__))},
        "limitations": [
            "Binance USDT perpetual premium is not OKX X-Perp premium.",
            "The bounded model and threshold grid creates selection risk.",
            "Any survivor still requires new direct X-Perp shadow evidence.",
        ],
    }
    OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({key: report[key] for key in (
        "status", "rows", "qualified_before_test",
        "qualified_premium_before_test", "frozen_before_test",
        "best_price_control", "results", "objective_checks")}, indent=2))


if __name__ == "__main__":
    main()
