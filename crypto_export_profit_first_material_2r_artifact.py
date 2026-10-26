"""Export the frozen material 0.5R/2R Profit-First HGB to portable JSON."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

import crypto_high_target_tree_oct5_audit as base
import crypto_profit_first_material_target_oct18_audit as audit
from crypto_export_okx_1h_hgb_artifact import export_nodes
from src.portable_hgb import predict_probability


SOURCE_REPORT = Path(
    "reports/audit_2026_09_08/crypto_profit_first_material_target_oct18.json")
OUT = Path("src/frozen_crypto_profit_first_material_2r.json")
VERIFY_OUT = Path(
    "reports/audit_2026_09_08/crypto_profit_first_material_2r_export.json")
EXPORTER = Path("crypto_export_profit_first_material_2r_artifact.py")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    report = json.loads(SOURCE_REPORT.read_text(encoding="utf-8"))
    frozen = report.get("frozen_before_2024")
    if not frozen or float(frozen["runner_target_r"]) != 2.0:
        raise ValueError("source report does not contain the frozen 2R policy")
    historical = audit.load_joined(2.0)
    fit = base.stage(historical, audit.FIT)
    schema = base.design(fit)
    model = HistGradientBoostingClassifier(
        **audit.MODEL, max_iter=150, l2_regularization=1.0,
        early_stopping=False, random_state=20260908)
    model.fit(base.matrix(fit, schema), fit.outcome.eq("TP2").astype(int))
    artifact = {
        "version": 1,
        "kind": "binary_hist_gradient_boosting",
        "features": schema["feature_names"],
        "medians": schema["medians"],
        "categories": schema["categories"],
        "baseline": float(model._baseline_prediction.ravel()[0]),
        "trees": [export_nodes(iteration[0]) for iteration in model._predictors],
        "quality_threshold": float(frozen["threshold"]),
        "bar": "15m",
        "entry": "next 15-minute X-Perp market price",
        "stop_atr": 2.0,
        "risk_pct_min": .01,
        "risk_pct_max": .02,
        "near_target_r": .5,
        "near_close_fraction": float(frozen["near_fraction"]),
        "runner_target_r": 2.0,
        "hold_bars": 48,
        "loss_pause": 3,
        "hourly_cap": 2,
        "source_report_sha256": digest(SOURCE_REPORT),
    }
    OUT.write_text(json.dumps(artifact, separators=(",", ":")),
                   encoding="utf-8")

    x = base.matrix(historical, schema)
    indices = np.linspace(0, len(x) - 1, min(5000, len(x)), dtype=int)
    expected = model.predict_proba(x[indices])[:, 1]
    observed = np.asarray([
        predict_probability(artifact, row.tolist()) for row in x[indices]],
        dtype=float)
    max_error = float(np.max(np.abs(expected - observed)))
    if max_error > 1e-12:
        raise ValueError(f"portable HGB parity failure: {max_error}")
    verification = {
        "status": "EXACT_PORTABLE_EXPORT",
        "artifact": str(OUT),
        "artifact_sha256": digest(OUT),
        "features": len(artifact["features"]),
        "trees": len(artifact["trees"]),
        "nodes": sum(len(tree) for tree in artifact["trees"]),
        "verified_rows": len(indices),
        "max_probability_error": max_error,
        "sources": {
            str(SOURCE_REPORT): digest(SOURCE_REPORT),
            str(EXPORTER): digest(EXPORTER),
        },
    }
    VERIFY_OUT.write_text(json.dumps(
        verification, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(verification, indent=2), flush=True)


if __name__ == "__main__":
    main()
