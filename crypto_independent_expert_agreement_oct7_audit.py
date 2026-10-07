"""Causal agreement audit across independent crypto feature groups.

Semantic entry families provide a finite union of proposals.  Four shallow
experts see geometry, perpetual trade flow, spot/perpetual flow, or volatility
term structure.  Each expert is fitted only on 2020-2021.  Agreement rules are
selected on 2022, ranked on 2023, and 2024 remains untouched unless the frozen
rule passes both development years.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

import crypto_5m_regime_router_literal_oct6_audit as price
import crypto_futures_trade_flow_regime_oct7_audit as flow
import crypto_spot_perp_flow_divergence_oct7_audit as dual
import crypto_volatility_term_structure_oct7_audit as vol


OUT = Path("reports/audit_2026_09_08/crypto_independent_expert_agreement_oct7.json")
ROWS_OUT = Path("reports/audit_2026_09_08/crypto_independent_expert_agreement_oct7.pkl")
SOURCE_GROUPS = ("geometry", "perp_flow", "dual_flow", "volatility")
SCORE_MODES = ("mean", "median", "mean_minus_std", "second_lowest")
QUANTILES = (.70, .75, .80, .85, .90, .925, .95, .96, .97, .98, .985, .99)
MODEL = {"max_leaf_nodes": 15, "min_samples_leaf": 400,
         "l2_regularization": 4.0}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def flags(frame: pd.DataFrame, names: tuple[str, ...], prefix: str,
          source: str) -> pd.DataFrame:
    keys = ["symbol", "entry_time", "direction"]
    columns = [f"{prefix}{name}" for name in names]
    found = frame[keys + columns].copy()
    found[f"source_{source}"] = 1
    return found.groupby(keys, sort=False).max().reset_index()


def build_dataset() -> tuple[pd.DataFrame, dict]:
    candles, future_sources = flow.load()
    spot, spot_sources = dual.load_spot()
    raw = price.build_features(candles)

    geometry_proposals = price.proposals(raw)
    proposal_parts = [flags(geometry_proposals, price.FAMILIES,
                            "family_", "geometry")]

    flow_full = flow.attach_flow(raw, candles)
    proposal_parts.append(flags(flow.proposals(flow_full), flow.FAMILIES,
                                "flow_family_", "perp_flow"))
    flow_features = flow_full[["symbol", "entry_time", *flow.FLOW_FEATURES]].copy()

    dual_full = dual.attach(raw, candles, spot)
    proposal_parts.append(flags(dual.proposals(dual_full), dual.FAMILIES,
                                "dual_family_", "dual_flow"))
    dual_features = dual_full[["symbol", "entry_time", *dual.FEATURES]].copy()

    vol_full = vol.attach(raw, candles)
    proposal_parts.append(flags(vol.proposals(vol_full), vol.FAMILIES,
                                "vol_family_", "volatility"))
    vol_features = vol_full[["symbol", "entry_time", *vol.VOL_FEATURES]].copy()

    all_flag_columns = (
        [f"family_{name}" for name in price.FAMILIES]
        + [f"flow_family_{name}" for name in flow.FAMILIES]
        + [f"dual_family_{name}" for name in dual.FAMILIES]
        + [f"vol_family_{name}" for name in vol.FAMILIES]
        + [f"source_{name}" for name in SOURCE_GROUPS]
    )
    combined = pd.concat([part.reindex(
        columns=["symbol", "entry_time", "direction", *all_flag_columns],
        fill_value=0) for part in proposal_parts], ignore_index=True)
    candidate_flags = combined.groupby(
        ["symbol", "entry_time", "direction"], sort=False).max().reset_index()
    candidate_flags["source_count"] = candidate_flags[
        [f"source_{name}" for name in SOURCE_GROUPS]].sum(axis=1)

    dataset = candidate_flags.merge(raw, on=["symbol", "entry_time"],
                                    how="inner", validate="many_to_one")
    for feature_frame in (flow_features, dual_features, vol_features):
        dataset = dataset.merge(feature_frame, on=["symbol", "entry_time"],
                                how="inner", validate="many_to_one")
    dataset = dataset.replace([np.inf, -np.inf], np.nan).dropna()
    frame = price.add_outcomes(dataset, candles)
    return frame, {**future_sources, **spot_sources}


def percentile_against_fit(probability: np.ndarray,
                           fit_probability: np.ndarray) -> np.ndarray:
    reference = np.sort(fit_probability)
    return np.searchsorted(reference, probability, side="right") / len(reference)


def score_modes(ranks: np.ndarray) -> dict[str, np.ndarray]:
    ordered = np.sort(ranks, axis=1)
    return {
        "mean": ranks.mean(axis=1),
        "median": np.median(ranks, axis=1),
        "mean_minus_std": ranks.mean(axis=1)-ranks.std(axis=1),
        "second_lowest": ordered[:, 1],
    }


def selection_ok(value: dict) -> bool:
    return bool(value["n"] >= 700 and value["return_pct"] > 0
                and value["literal_target_hit_rate"] >= 69
                and value["dd_pct"] <= 7
                and value["trades_per_calendar_day"] >= 2.5)


def validation_ok(value: dict) -> bool:
    return bool(value["n"] >= 500 and value["return_pct"] > 0
                and value["literal_target_hit_rate"] >= 66
                and value["dd_pct"] <= 7
                and value["trades_per_calendar_day"] >= 2.0)


def main() -> None:
    frame, sources = build_dataset()
    fit = frame.year.isin((2020, 2021)).to_numpy()
    schema = price.columns(frame.loc[fit])
    matrices = {
        "geometry": price.matrix(frame, schema),
        "perp_flow": flow.matrices(frame, schema)[1],
        "dual_flow": dual.matrix(frame, schema),
        "volatility": vol.matrix(frame, schema),
    }
    labels = frame.target_hit.astype(int).to_numpy()
    ranks = []
    expert_diagnostics = {}
    for name in SOURCE_GROUPS:
        model = HistGradientBoostingClassifier(
            max_iter=180, learning_rate=.06,
            max_leaf_nodes=MODEL["max_leaf_nodes"],
            min_samples_leaf=MODEL["min_samples_leaf"],
            l2_regularization=MODEL["l2_regularization"],
            random_state=20261007)
        model.fit(matrices[name][fit], labels[fit])
        probability = model.predict_proba(matrices[name])[:, 1]
        ranks.append(percentile_against_fit(probability, probability[fit]))
        expert_diagnostics[name] = {
            "fit_auc_proxy": round(float(np.corrcoef(probability[fit], labels[fit])[0, 1]), 5),
            "selection_probability_mean": round(float(probability[frame.year.eq(2022)] .mean()), 6),
        }
    rank_matrix = np.column_stack(ranks)
    scored = frame.copy()
    for index, name in enumerate(SOURCE_GROUPS):
        scored[f"rank_{name}"] = rank_matrix[:, index]
    mode_scores = score_modes(rank_matrix)
    selection = []
    for mode in SCORE_MODES:
        scored["quality"] = mode_scores[mode]
        year_scores = scored.loc[scored.year.eq(2022), "quality"]
        for quantile in QUANTILES:
            threshold = float(year_scores.quantile(quantile))
            value = price.evaluate(scored, (2022,), threshold)
            if selection_ok(value):
                selection.append({"mode": mode, "quantile": quantile,
                                  "threshold": threshold,
                                  "selection_2022": value})
    validated = []
    validation_diagnostics = []
    for item in selection:
        scored["quality"] = mode_scores[item["mode"]]
        value = price.evaluate(scored, (2023,), item["threshold"])
        candidate = {**item, "validation_2023": value}
        validation_diagnostics.append(candidate)
        if validation_ok(value):
            candidate["score"] = round(
                min(item["selection_2022"]["return_pct"], value["return_pct"])
                - 2*max(item["selection_2022"]["dd_pct"], value["dd_pct"])
                + min(item["selection_2022"]["trades_per_calendar_day"],
                      value["trades_per_calendar_day"]), 8)
            validated.append(candidate)
    validated.sort(key=lambda item: item["score"], reverse=True)
    frozen = validated[0] if validated else None
    test_2024 = later = checks = None
    frozen_rows = None
    if frozen:
        scored["quality"] = mode_scores[frozen["mode"]]
        test_2024 = price.evaluate(scored, (2024,), frozen["threshold"])
        if validation_ok(test_2024):
            later = {str(year): price.evaluate(scored, (year,), frozen["threshold"])
                     for year in (2025, 2026)}
            checks = {year: validation_ok(value) for year, value in later.items()}
        frozen_rows = scored[scored.quality.ge(frozen["threshold"])].copy()
        frozen_rows.to_pickle(ROWS_OUT)
    report = {
        "status": ("RESEARCH_CANDIDATE_AWAITING_DIRECT_FORWARD"
                   if checks and all(checks.values()) else "REJECTED"),
        "protocol": {
            "fit": [2020, 2021], "selection": 2022,
            "validation_and_ranking": 2023, "untouched_test": 2024,
            "hidden_if_test_passes": [2025, 2026],
            "experts": list(SOURCE_GROUPS), "score_modes": list(SCORE_MODES),
            "model": MODEL, "quantiles": list(QUANTILES),
            "entry": "next 5m market open", "target_pct": .5,
            "stop_pct": 1.0, "max_bars": 24,
            "money": "3% equity margin, fixed 10x gross",
        },
        "candidate_rows": len(frame),
        "fit_literal_target_prevalence_pct": round(100*float(labels[fit].mean()), 4),
        "source_count": frame.source_count.value_counts().sort_index().to_dict(),
        "expert_diagnostics": expert_diagnostics,
        "qualified_2022": len(selection), "qualified_2023": len(validated),
        "selection_frontier": selection,
        "validation_diagnostics": validation_diagnostics,
        "frozen_before_2024": frozen, "test_2024": test_2024,
        "later": later, "later_checks": checks,
        "sources": {**sources, str(Path(__file__)): digest(Path(__file__))},
        "limitations": [
            "Expert agreement cannot create information absent from its component feature groups.",
            "The semantic proposal union and finite agreement grid still create selection risk.",
            "Any historical pass remains shadow-only pending direct X-Perp forward outcomes.",
        ],
    }
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({key: report[key] for key in (
        "status", "candidate_rows", "fit_literal_target_prevalence_pct",
        "source_count", "expert_diagnostics", "qualified_2022",
        "qualified_2023", "frozen_before_2024", "test_2024", "later",
        "later_checks")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
