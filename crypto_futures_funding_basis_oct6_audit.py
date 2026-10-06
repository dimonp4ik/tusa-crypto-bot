"""Literal-target derivative regimes using funding plus perpetual/spot basis."""
from __future__ import annotations

import hashlib
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

import crypto_5m_regime_router_literal_oct6_audit as price
import crypto_futures_funding_regime_oct6_audit as funding_source


SPOT_ROOT = Path("reports/audit_2026_09_08/crypto_5m_history")
OUT = Path("reports/audit_2026_09_08/crypto_futures_funding_basis_oct6.json")
BASIS_FEATURES = (
    "basis", "basis_mean48", "basis_z336", "basis_delta12",
    "basis_cross_z", "btc_basis", "relative_basis",
)
QUANTILES = funding_source.QUANTILES
CONFIGS = funding_source.CONFIGS


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_spot() -> tuple[dict, dict]:
    data, sources = {}, {}
    for symbol in funding_source.SYMBOLS:
        path = SPOT_ROOT / f"{symbol}_5min_2020_2026.pkl"
        data[symbol] = pickle.loads(path.read_bytes())
        sources[str(path)] = digest(path)
    return data, sources


def attach_basis(frame: pd.DataFrame, futures: dict,
                 spot: dict) -> pd.DataFrame:
    pieces = []
    for symbol, part in frame.groupby("symbol", sort=False):
        signal_time = part.entry_time.to_numpy(np.int64) - 300
        future_map = {int(t): float(c) for t, c in zip(
            futures[symbol]["time"], futures[symbol]["close"])}
        spot_map = {int(t): float(c) for t, c in zip(
            spot[symbol]["time"], spot[symbol]["close"])}
        found = part.sort_values("entry_time").copy()
        future_close = np.asarray([future_map.get(int(t), np.nan)
                                   for t in signal_time], dtype=float)
        spot_close = np.asarray([spot_map.get(int(t), np.nan)
                                 for t in signal_time], dtype=float)
        found["basis"] = future_close / spot_close - 1.0
        found["basis_mean48"] = found.basis.rolling(48, min_periods=36).mean()
        mean336 = found.basis.rolling(336, min_periods=240).mean()
        std336 = found.basis.rolling(336, min_periods=240).std().replace(0, np.nan)
        found["basis_z336"] = (found.basis - mean336) / std336
        found["basis_delta12"] = found.basis - found.basis.shift(12)
        pieces.append(found)
    frame = pd.concat(pieces, ignore_index=True).dropna(
        subset=["basis", "basis_mean48", "basis_z336", "basis_delta12"])
    grouped = frame.groupby("entry_time", sort=False)
    mean = grouped.basis.transform("mean")
    std = grouped.basis.transform("std").replace(0, np.nan)
    frame["basis_cross_z"] = (frame.basis - mean) / std
    btc = frame[frame.symbol.eq("BTCUSDT")][
        ["entry_time", "basis"]].rename(columns={"basis": "btc_basis"})
    frame = frame.merge(btc, on="entry_time", how="left", validate="many_to_one")
    frame["relative_basis"] = frame.basis - frame.btc_basis
    return frame.replace([np.inf, -np.inf], np.nan).dropna(
        subset=list(BASIS_FEATURES))


def proposals(frame: pd.DataFrame) -> pd.DataFrame:
    funding_rows = funding_source.proposals(frame)
    direction = -np.sign(frame.basis_z336)
    mask = (direction.ne(0) & frame.basis_z336.abs().ge(.75)
            & (frame.ret12 * -direction).ge(.002)
            & frame.cross_z12.abs().ge(.4))
    basis_rows = frame.loc[mask].copy()
    basis_rows["direction"] = direction.loc[mask].astype(np.int8)
    for family in funding_source.FAMILIES:
        basis_rows[f"funding_family_{family}"] = 0
    basis_rows["funding_family_basis_reversal"] = 1
    funding_rows["funding_family_basis_reversal"] = 0
    combined = pd.concat([funding_rows, basis_rows], ignore_index=True)
    families = (*funding_source.FAMILIES, "basis_reversal")
    keys = ["symbol", "entry_time", "direction"]
    flags = combined.groupby(keys, sort=False)[
        [f"funding_family_{family}" for family in families]].max().reset_index()
    base = combined.sort_values(keys).drop_duplicates(keys).drop(
        columns=[f"funding_family_{family}" for family in families])
    return base.merge(flags, on=keys, how="left", validate="one_to_one")


def matrices(frame: pd.DataFrame, schema: list[str]) -> tuple[np.ndarray, np.ndarray]:
    _, funding_matrix = funding_source.matrices(frame, schema)
    direction = frame.direction.to_numpy(float)
    extra = [(frame[name].to_numpy(float) * direction)[:, None]
             for name in BASIS_FEATURES]
    extra.extend([
        frame.basis.abs().to_numpy(float)[:, None],
        frame.basis_z336.abs().to_numpy(float)[:, None],
        frame.basis_cross_z.abs().to_numpy(float)[:, None],
        frame.funding_family_basis_reversal.to_numpy(float)[:, None],
    ])
    return funding_matrix, np.hstack((funding_matrix,
                                      np.hstack(extra).astype(np.float32)))


def public(item: dict) -> dict:
    return {key: value for key, value in item.items()
            if key not in ("_frame", "_model")}


def main() -> None:
    futures, funding, sources = funding_source.load()
    spot, spot_sources = load_spot()
    raw = price.build_features(futures)
    enriched = funding_source.attach_funding(raw, funding)
    enriched = attach_basis(enriched, futures, spot)
    frame = price.add_outcomes(proposals(enriched), futures)
    fit = frame.year.isin((2020, 2021)).to_numpy()
    selection = frame.year.eq(2022).to_numpy()
    schema = price.columns(frame[fit])
    x_funding, x_full = matrices(frame, schema)
    y = frame.target_hit.astype(int).to_numpy()
    candidates = []
    for model_kind, matrix in (("funding_control", x_funding),
                               ("funding_plus_basis", x_full)):
        for config in CONFIGS:
            model = HistGradientBoostingClassifier(
                max_iter=180, learning_rate=.06,
                max_leaf_nodes=config["max_leaf_nodes"],
                min_samples_leaf=config["min_samples_leaf"],
                l2_regularization=config["l2"], random_state=20261006)
            model.fit(matrix[fit], y[fit])
            scored = frame.copy()
            scored["quality"] = model.predict_proba(matrix)[:, 1]
            for quantile in QUANTILES:
                threshold = float(scored.loc[selection, "quality"].quantile(quantile))
                s22 = price.evaluate(scored, (2022,), threshold)
                if not funding_source.acceptable(s22, validation=False):
                    continue
                item = {"model_kind": model_kind, "model": config,
                        "quantile": quantile, "threshold": threshold,
                        "selection_2022": s22,
                        "validation_2023": price.evaluate(scored, (2023,), threshold),
                        "validation_2024": price.evaluate(scored, (2024,), threshold)}
                if not (funding_source.acceptable(
                            item["validation_2023"], validation=True)
                        and funding_source.acceptable(
                            item["validation_2024"], validation=True)):
                    continue
                item["score"] = round(sum(
                    item[name]["return_pct"] - item[name]["dd_pct"]
                    for name in ("selection_2022", "validation_2023",
                                 "validation_2024")), 8)
                item["_frame"], item["_model"] = scored, model
                candidates.append(item)
    candidates.sort(key=lambda item: item["score"], reverse=True)
    full = [item for item in candidates
            if item["model_kind"] == "funding_plus_basis"]
    frozen = full[0] if full else None
    results = None
    if frozen:
        results = {str(year): price.evaluate(
            frozen["_frame"], (year,), frozen["threshold"])
            for year in (2022, 2023, 2024, 2025, 2026)}
    passed = bool(results and all(
        funding_source.acceptable(value, validation=(year != "2022"))
        for year, value in results.items()))
    control = next((public(item) for item in candidates
                    if item["model_kind"] == "funding_control"), None)
    report = {
        "status": ("RESEARCH_CANDIDATE_AWAITING_VENUE_FORWARD"
                   if passed else "REJECTED"),
        "protocol": {"fit": "2020-2021", "selection": "2022",
                     "validation": ["2023", "2024"], "test": "2025",
                     "observed": "2026", "entry": "next 5m futures market open",
                     "target_pct": .5, "stop_pct": 1.0, "max_bars": 24,
                     "basis": "perpetual close / spot close - 1 on closed bar",
                     "money": "3% equity margin, fixed 10x gross"},
        "enriched_rows": len(enriched), "proposal_rows": len(frame),
        "qualified_before_test": len(candidates),
        "frozen_basis_before_test": public(frozen) if frozen else None,
        "best_funding_control_before_test": control,
        "results": results, "test_pass": passed,
        "top_before_test": [public(item) for item in candidates[:20]],
        "limitations": [
            "The proposal and bounded model grids create selection risk.",
            "Funding and spot archives end in September 2026.",
            "Any survivor requires new direct X-Perp shadow evidence.",
        ],
        "sources": {**sources, **spot_sources,
                    str(Path(__file__)): digest(Path(__file__))},
    }
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({key: report[key] for key in (
        "status", "enriched_rows", "proposal_rows", "qualified_before_test",
        "frozen_basis_before_test", "best_funding_control_before_test",
        "results", "test_pass")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
