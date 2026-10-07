"""Pre-entry market-state gate for the annual four-hour complement.

Small decision trees learn only from 2022-2023.  Gate thresholds are selected
on 2024, validated on 2025, and finally replayed on research-biased 2026.
All features are available before the next-hour market entry.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text

import crypto_okx_swap_1h_regime_oct7_audit as base
import crypto_okx_swap_1h_target_cascade_oct7_audit as target_engine
import crypto_okx_swap_4h_structure_complement_oct8_audit as four
import crypto_okx_swap_4h_online_health_oct8_audit as health
from crypto_runtime_shadow_portfolio_oct3_audit import runtime_gate


ROOT = Path("reports/audit_2026_09_08")
CASCADE_ROWS = ROOT / "crypto_okx_swap_1h_target_cascade_oct7.pkl"
CASCADE_REPORT = ROOT / "crypto_okx_swap_1h_target_cascade_oct7.json"
WALK_REPORT = ROOT / "crypto_okx_swap_4h_walkforward_complement_oct8.json"
HEALTH_REPORT = ROOT / "crypto_okx_swap_4h_online_health_oct8.json"
OUT = ROOT / "crypto_okx_swap_4h_exante_regime_gate_oct8.json"
FEATURES = (
    "directed_ret3", "directed_ret6", "directed_ret18", "directed_ret42",
    "directed_btc3", "directed_btc6", "directed_btc18", "directed_btc42",
    "directed_cross_z18", "directed_position18",
    "directed_breadth3", "directed_breadth6", "directed_breadth18",
    "dispersion6", "dispersion18", "eff6", "eff18", "eff42",
    "fast_slow", "volume_ratio", "range_ratio", "autocorr18", "autocorr42",
    "atr14_pct", "selected_target_pct",
)
CONFIGS = (
    {"depth": 2, "leaf": 60},
    {"depth": 2, "leaf": 100},
    {"depth": 3, "leaf": 60},
    {"depth": 3, "leaf": 100},
    {"depth": 3, "leaf": 150},
)
QUANTILES = (.20, .35, .50, .65, .75, .85, .90, .95)
SOURCE = "four_hour_exante_regime"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def regime_matrix(frame: pd.DataFrame) -> pd.DataFrame:
    direction = frame.direction.to_numpy(float)
    output = pd.DataFrame(index=frame.index)
    for window in (3, 6, 18, 42):
        output[f"directed_ret{window}"] = frame[f"ret{window}"] * direction
        output[f"directed_btc{window}"] = frame[f"btc_ret{window}"] * direction
    output["directed_cross_z18"] = frame.cross_z18 * direction
    output["directed_position18"] = np.where(
        direction > 0, frame.position18, 1 - frame.position18)
    for window in (3, 6, 18):
        output[f"directed_breadth{window}"] = np.where(
            direction > 0, frame[f"breadth{window}"],
            1 - frame[f"breadth{window}"])
    for name in ("dispersion6", "dispersion18", "eff6", "eff18", "eff42",
                 "fast_slow", "volume_ratio", "range_ratio", "autocorr18",
                 "autocorr42", "atr14_pct", "selected_target_pct"):
        output[name] = frame[name]
    return output.reindex(columns=FEATURES).replace(
        [np.inf, -np.inf], np.nan).fillna(0).astype(np.float32)


def summarize(rows: list[dict]) -> dict:
    return four.summarize(rows)


def raw_complement(frame: pd.DataFrame, years: tuple[int, ...],
                   threshold: float) -> list[dict]:
    selected = frame[frame.year.isin(years) & frame.regime_quality.ge(threshold)].copy()
    selected["quality"] = selected.regime_quality
    return four.raw_rows(selected, years, SOURCE, 1)


def static_rows(cascade: pd.DataFrame, candidate: pd.DataFrame,
                years: tuple[int, ...]) -> tuple[list[dict], list[dict]]:
    base_rows = four.raw_rows(cascade, years, "hourly_cascade", 0)
    extra = candidate[candidate.year.isin(years)].copy()
    extra["regime_quality"] = 1.0
    extra_rows = raw_complement(extra, years, .5)
    return base_rows, extra_rows


def evaluate(cascade: pd.DataFrame, candidate: pd.DataFrame,
             years: tuple[int, ...], threshold: float,
             static: tuple[list[dict], list[dict]] | None = None) -> dict:
    base_rows, ungated_extra = static or static_rows(cascade, candidate, years)
    gated_extra = raw_complement(candidate, years, threshold)
    baseline = runtime_gate(base_rows)
    ungated = runtime_gate([*base_rows, *ungated_extra])
    gated = runtime_gate([*base_rows, *gated_extra])
    return {
        "cascade": summarize(baseline),
        "ungated_union": summarize(ungated),
        "gated_union": summarize(gated),
        "accepted_regime_complement": sum(
            row["source"] == SOURCE for row in gated),
        "rejected_raw_complement": len(ungated_extra) - len(gated_extra),
    }


def passes(value: dict, selection: bool) -> bool:
    baseline, gated = value["cascade"], value["gated_union"]
    return bool(value["accepted_regime_complement"] >= (80 if selection else 60)
                and value["rejected_raw_complement"] >= (20 if selection else 0)
                and gated["return_pct"] > baseline["return_pct"]
                and gated["dd_pct"] <= baseline["dd_pct"] + 1.0
                and gated["literal_target_hit_rate"] >= 63
                and gated["trades_per_calendar_day"] >= 2.5)


def drift(x: pd.DataFrame, train: pd.Series, final: pd.Series) -> list[dict]:
    rows = []
    for name in FEATURES:
        reference = x.loc[train, name]
        scale = float(reference.std(ddof=0))
        delta = (float(x.loc[final, name].mean()) - float(reference.mean()))
        standardized = delta / scale if scale > 0 else 0.0
        rows.append({"feature": name,
                     "mean_2022_2025": float(reference.mean()),
                     "mean_2026": float(x.loc[final, name].mean()),
                     "standardized_shift": standardized})
    return sorted(rows, key=lambda item: abs(item["standardized_shift"]), reverse=True)


def main() -> None:
    candles, sources = base.load()
    raw = four.build_features(candles)
    frame = four.proposals(raw).reset_index(drop=True)
    for target in four.TARGETS:
        target_engine.add_target(frame, candles, target)
    columns = four.schema(frame[frame.year.isin((2020, 2021))])
    x_all = four.matrix(frame, columns)
    cascade = pd.read_pickle(CASCADE_ROWS)
    walk_report = json.loads(WALK_REPORT.read_text(encoding="utf-8"))
    _, routed = health.build_rows(frame, x_all, cascade, walk_report)
    candidate = pd.concat([
        routed[year][routed[year].year.eq(year)] for year in range(2022, 2027)
    ], ignore_index=True)
    x = regime_matrix(candidate)
    fit = candidate.year.isin((2022, 2023))
    labels = candidate.target_hit.astype(int)
    selection_static = static_rows(cascade, candidate, (2024,))
    validation_static = static_rows(cascade, candidate, (2025,))
    final_static = static_rows(cascade, candidate, (2026,))
    choices = []
    selection_frontier = []
    models = {}
    for config in CONFIGS:
        model = DecisionTreeClassifier(
            max_depth=config["depth"], min_samples_leaf=config["leaf"],
            class_weight="balanced", random_state=20261008)
        model.fit(x.loc[fit, list(FEATURES)], labels.loc[fit])
        scored = candidate.copy()
        scored["regime_quality"] = model.predict_proba(x[list(FEATURES)])[:, 1]
        models[(config["depth"], config["leaf"])] = (model, scored)
        selection_scores = scored.loc[scored.year.eq(2024), "regime_quality"]
        for quantile in QUANTILES:
            threshold = float(selection_scores.quantile(quantile))
            value = evaluate(
                cascade, scored, (2024,), threshold, selection_static)
            if not passes(value, True):
                continue
            gain = value["gated_union"]["return_pct"] - value["cascade"]["return_pct"]
            dd_cost = max(0.0, value["gated_union"]["dd_pct"]
                          - value["cascade"]["dd_pct"])
            item = {"model": config, "quantile": quantile,
                    "threshold": threshold, "selection_2024": value,
                    "selection_rank": gain - 2 * dd_cost}
            selection_frontier.append(item)
            choices.append(item)
    validated = []
    for item in choices:
        key = (item["model"]["depth"], item["model"]["leaf"])
        _, scored = models[key]
        value = evaluate(
            cascade, scored, (2025,), item["threshold"], validation_static)
        if not passes(value, False):
            continue
        gain24 = (item["selection_2024"]["gated_union"]["return_pct"]
                  - item["selection_2024"]["cascade"]["return_pct"])
        gain25 = value["gated_union"]["return_pct"] - value["cascade"]["return_pct"]
        dd_cost = max(
            item["selection_2024"]["gated_union"]["dd_pct"]
            - item["selection_2024"]["cascade"]["dd_pct"],
            value["gated_union"]["dd_pct"] - value["cascade"]["dd_pct"], 0.0)
        validated.append({**item, "validation_2025": value,
                          "robust_rank": min(gain24, gain25) - 2 * dd_cost})
    validated.sort(key=lambda item: item["robust_rank"], reverse=True)
    frozen = validated[0] if validated else None
    final = tree = None
    status = "REJECTED_SELECTION" if not choices else "REJECTED_2025"
    if frozen:
        key = (frozen["model"]["depth"], frozen["model"]["leaf"])
        model, scored = models[key]
        final = evaluate(
            cascade, scored, (2026,), frozen["threshold"], final_static)
        tree = export_text(model, feature_names=list(FEATURES), decimals=5)
        status = ("EXANTE_REGIME_RESEARCH_CANDIDATE_REQUIRES_FORWARD"
                  if passes(final, False) else "REJECTED_2026")
    report = {
        "status": status,
        "protocol": {"fit": [2022, 2023], "threshold_selection": 2024,
                     "validation": 2025, "research_biased_final": 2026,
                     "models": list(CONFIGS), "quantiles": list(QUANTILES),
                     "features": list(FEATURES),
                     "money": "3% current equity margin, fixed 10x gross"},
        "candidate_rows": len(candidate),
        "fit_literal_hit_rate": round(100 * float(labels.loc[fit].mean()), 4),
        "tested": len(CONFIGS) * len(QUANTILES),
        "qualified_selection": len(choices),
        "qualified_validation": len(validated),
        "selection_frontier": sorted(
            selection_frontier, key=lambda item: item["selection_rank"], reverse=True),
        "frozen_before_2026": frozen, "final_2026": final,
        "frozen_tree": tree,
        "feature_drift_2026": drift(
            x, candidate.year.isin((2022, 2023, 2024, 2025)),
            candidate.year.eq(2026)),
        "sources": {**sources, str(CASCADE_ROWS): digest(CASCADE_ROWS),
                    str(CASCADE_REPORT): digest(CASCADE_REPORT),
                    str(WALK_REPORT): digest(WALK_REPORT),
                    str(HEALTH_REPORT): digest(HEALTH_REPORT),
                    str(Path(__file__)): digest(Path(__file__))},
        "limitations": [
            "The ex-ante gate was designed after inspecting the 2026 failure and cannot prove readiness on 2026.",
            "A pass still requires fresh direct EU X-Perp forward shadow confirmation.",
        ],
    }
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({key: report[key] for key in (
        "status", "candidate_rows", "fit_literal_hit_rate", "tested",
        "qualified_selection", "qualified_validation", "frozen_before_2026",
        "final_2026", "frozen_tree", "feature_drift_2026")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
