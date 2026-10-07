"""Low-frequency precision satellite from the price-impact proposal set.

The parent frequent-system protocol discarded this candidate only because its
2023 rate was below two trades per calendar day.  This audit freezes a separate
satellite using 2022 selection and 2023 validation before inspecting its later
years.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from sklearn.ensemble import HistGradientBoostingClassifier

import crypto_5m_regime_router_literal_oct6_audit as price
import crypto_futures_trade_flow_regime_oct7_audit as perp
import crypto_spot_perp_flow_divergence_oct7_audit as dual
import crypto_price_impact_response_oct7_audit as impact


OUT = Path("reports/audit_2026_09_08/crypto_price_impact_precision_satellite_oct7.json")
ROWS_OUT = Path("reports/audit_2026_09_08/crypto_price_impact_precision_satellite_oct7.pkl")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def selection_ok(value: dict) -> bool:
    return bool(value["n"] >= 900 and value["return_pct"] > 0
                and value["literal_target_hit_rate"] >= 70
                and value["dd_pct"] <= 4
                and value["trades_per_calendar_day"] >= 2.5)


def validation_ok(value: dict) -> bool:
    return bool(value["n"] >= 500 and value["return_pct"] > 0
                and value["literal_target_hit_rate"] >= 68
                and value["dd_pct"] <= 4
                and value["trades_per_calendar_day"] >= 1.3)


def later_ok(value: dict) -> bool:
    return bool(value["n"] >= 300 and value["return_pct"] > 0
                and value["literal_target_hit_rate"] >= 67
                and value["dd_pct"] <= 5
                and value["trades_per_calendar_day"] >= 1.0)


def public(item: dict | None) -> dict | None:
    if item is None:
        return None
    return {key: value for key, value in item.items()
            if key not in ("_frame", "_model")}


def main() -> None:
    futures, future_sources = perp.load()
    spot, spot_sources = dual.load_spot()
    raw = price.build_features(futures)
    enriched = impact.attach(raw, futures, spot)
    frame = price.add_outcomes(impact.proposals(enriched), futures)
    fit = frame.year.isin((2020, 2021)).to_numpy()
    schema = price.columns(frame.loc[fit])
    x = impact.matrix(frame, schema)
    labels = frame.target_hit.astype(int).to_numpy()
    selection = []
    for config in impact.CONFIGS:
        model = HistGradientBoostingClassifier(
            max_iter=180, learning_rate=.06, max_leaf_nodes=config["leaves"],
            min_samples_leaf=config["leaf"], l2_regularization=config["l2"],
            random_state=20261007)
        model.fit(x[fit], labels[fit])
        scored = frame.copy()
        scored["quality"] = model.predict_proba(x)[:, 1]
        scores = scored.loc[scored.year.eq(2022), "quality"]
        for quantile in impact.QUANTILES:
            threshold = float(scores.quantile(quantile))
            value = price.evaluate(scored, (2022,), threshold)
            if selection_ok(value):
                selection.append({
                    "model": config, "quantile": quantile,
                    "threshold": threshold, "selection_2022": value,
                    "_frame": scored, "_model": model,
                })
    validated = []
    diagnostics = []
    for item in selection:
        value = price.evaluate(item["_frame"], (2023,), item["threshold"])
        diagnostics.append(public({**item, "validation_2023": value}))
        if validation_ok(value):
            candidate = {**item, "validation_2023": value}
            candidate["score"] = round(
                min(item["selection_2022"]["return_pct"], value["return_pct"])
                - 2 * max(item["selection_2022"]["dd_pct"], value["dd_pct"]), 8)
            validated.append(candidate)
    validated.sort(key=lambda item: item["score"], reverse=True)
    frozen = validated[0] if validated else None
    test_2024 = later = checks = None
    if frozen:
        test_2024 = price.evaluate(frozen["_frame"], (2024,), frozen["threshold"])
        if later_ok(test_2024):
            later = {str(year): price.evaluate(
                frozen["_frame"], (year,), frozen["threshold"])
                for year in (2025, 2026)}
            checks = {year: later_ok(value) for year, value in later.items()}
        frozen["_frame"].loc[
            frozen["_frame"].quality.ge(frozen["threshold"])].to_pickle(ROWS_OUT)
    report = {
        "status": ("RESEARCH_CANDIDATE_AWAITING_DIRECT_FORWARD"
                   if checks and all(checks.values()) else "REJECTED"),
        "protocol": {
            "relationship": "separate low-frequency satellite; not a fallback replacement for the parent frequent candidate",
            "fit": [2020, 2021], "selection": 2022,
            "validation_and_ranking": 2023, "satellite_test": 2024,
            "hidden_if_test_passes": [2025, 2026],
            "entry": "next 5m market open", "target_pct": .5,
            "stop_pct": 1.0, "max_bars": 24,
            "money": "3% equity margin, fixed 10x gross",
        },
        "qualified_2022": len(selection), "qualified_2023": len(validated),
        "validation_diagnostics": diagnostics,
        "frozen_before_2024": public(frozen), "test_2024": test_2024,
        "later": later, "later_checks": checks,
        "sources": {**future_sources, **spot_sources,
                    str(Path(impact.__file__)): digest(Path(impact.__file__)),
                    str(Path(__file__)): digest(Path(__file__))},
        "limitations": [
            "The satellite protocol was declared after the parent frequent route failed its 2024 gate.",
            "Its own 2024+ rows were not used for the 2022/2023 freeze.",
            "Any historical pass remains shadow-only pending direct X-Perp forward outcomes.",
        ],
    }
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({key: report[key] for key in (
        "status", "qualified_2022", "qualified_2023",
        "frozen_before_2024", "test_2024", "later", "later_checks")},
        indent=2), flush=True)


if __name__ == "__main__":
    main()
