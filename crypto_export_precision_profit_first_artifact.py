"""Export the frozen 15-minute profit-first HGB to portable JSON."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

import crypto_2r_precision_barbell_oct5_audit as precision
import crypto_high_target_tree_oct5_audit as base
import crypto_precision_profit_first_oct18_audit as audit
from crypto_export_okx_1h_hgb_artifact import export_nodes
from src.portable_hgb import predict_probability


SOURCE_REPORT = Path("reports/audit_2026_09_08/crypto_precision_profit_first_oct18.json")
OUT = Path("src/frozen_crypto_precision_profit_first_hgb.json")
VERIFY_OUT = Path("reports/audit_2026_09_08/crypto_precision_profit_first_hgb_export.json")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    report = json.loads(SOURCE_REPORT.read_text(encoding="utf-8"))
    frozen = report.get("frozen_before_2024")
    if report.get("status") != "PROFIT_FIRST_5PCT_SHADOW_CANDIDATE" or not frozen:
        raise ValueError("precision model is not a frozen 5% shadow candidate")
    historical = audit.canonical_history(precision.load_joined())
    fit = base.stage(historical, precision.FIT)
    schema = base.design(fit)
    previous = json.loads(audit.OLD_REPORT.read_text(encoding="utf-8"))[
        "frozen_before_2024"]
    model = HistGradientBoostingClassifier(
        learning_rate=float(previous["learning_rate"]), max_iter=150,
        max_leaf_nodes=int(previous["max_leaf_nodes"]),
        min_samples_leaf=int(previous["min_samples_leaf"]),
        l2_regularization=1.0, early_stopping=False,
        random_state=20260908)
    model.fit(base.matrix(fit, schema), fit.outcome.eq("TP2").astype(int))
    artifact = {
        "version": 1,
        "kind": "binary_hist_gradient_boosting",
        "features": schema["feature_names"],
        "baseline": float(model._baseline_prediction.ravel()[0]),
        "trees": [export_nodes(iteration[0]) for iteration in model._predictors],
        "quality_threshold": float(frozen["threshold"]),
        "bar": "15m", "entry": "next 15-minute market open",
        "stop_atr": 2.0, "near_target_r": .25,
        "runner_target_r": 2.0, "near_close_fraction": .10,
        "hold_bars": 48, "loss_pause": int(frozen["loss_pause"]),
        "hourly_cap": 2,
        "history_overlap_duplicates_removed": report["history_integrity"][
            "overlap_duplicates_removed"],
        "source_report_sha256": digest(SOURCE_REPORT),
    }
    OUT.write_text(json.dumps(artifact, separators=(",", ":")), encoding="utf-8")
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
        "artifact": str(OUT), "artifact_sha256": digest(OUT),
        "features": len(artifact["features"]),
        "trees": len(artifact["trees"]),
        "nodes": sum(len(tree) for tree in artifact["trees"]),
        "verified_rows": len(indices), "max_probability_error": max_error,
        "sources": {
            str(SOURCE_REPORT): digest(SOURCE_REPORT),
            str(Path(__file__)): digest(Path(__file__)),
        },
    }
    VERIFY_OUT.write_text(json.dumps(
        verification, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(verification, indent=2), flush=True)


if __name__ == "__main__":
    main()
