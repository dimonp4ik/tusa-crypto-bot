"""Exploratory family transfer diagnostics; never a deployable selector."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

import crypto_5m_regime_router_literal_oct6_audit as price
import crypto_futures_funding_regime_oct6_audit as futures_source
import crypto_futures_positioning_regime_oct7_audit as study


OUT = Path("reports/audit_2026_09_08/crypto_futures_positioning_family_diagnostic_oct7.json")


def main() -> None:
    candles, _, _ = futures_source.load()
    metrics, _ = study.load_metrics()
    features = study.attach_metrics(price.build_features(candles), metrics)
    frame = price.add_outcomes(study.proposals(features), candles)
    fit = study.period_mask(frame, "fit_2023")
    selection_mask = study.period_mask(frame, "selection_2024_h1")
    schema = price.columns(frame.loc[fit])
    _, matrix = study.matrices(frame, schema)
    y = frame.target_hit.astype(int).to_numpy()
    diagnostics = []
    for config in study.CONFIGS:
        model = HistGradientBoostingClassifier(
            max_iter=180, learning_rate=.06,
            max_leaf_nodes=config["leaves"], min_samples_leaf=config["leaf"],
            l2_regularization=config["l2"], random_state=20261007)
        model.fit(matrix[fit], y[fit])
        scored = frame.copy()
        scored["quality"] = model.predict_proba(matrix)[:, 1]
        for quantile in study.QUANTILES:
            threshold = float(scored.loc[selection_mask, "quality"].quantile(quantile))
            selection = study.evaluate_period(scored, "selection_2024_h1", threshold)
            if not study.selection_ok(selection):
                continue
            item = {"model": config, "quantile": quantile,
                    "threshold": threshold, "selection": selection,
                    "full_transfer": {name: study.evaluate_period(scored, name, threshold)
                                      for name in ("validation_2024_h2", "test_2025",
                                                   "observed_2026")},
                    "families": {}}
            for family in study.FAMILIES:
                subset = scored[scored[f"positioning_family_{family}"].eq(1)].copy()
                item["families"][family] = {
                    name: study.evaluate_period(subset, name, threshold)
                    for name in ("selection_2024_h1", "validation_2024_h2",
                                 "test_2025", "observed_2026")}
            diagnostics.append(item)
    report = {
        "status": "EXPLORATORY_ONLY_NOT_A_SELECTOR",
        "rows": len(frame), "selection_passes": len(diagnostics),
        "diagnostics": diagnostics,
        "warning": ("Families are shown after hidden periods were inspected; "
                    "none may be promoted without a newly frozen forward test."),
    }
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    compact = []
    for item in diagnostics:
        compact.append({"model": item["model"], "quantile": item["quantile"],
                        "selection": item["selection"],
                        "full_transfer": item["full_transfer"]})
    print(json.dumps({"status": report["status"],
                      "selection_passes": len(diagnostics),
                      "full_transfer": compact}, indent=2), flush=True)


if __name__ == "__main__":
    main()
