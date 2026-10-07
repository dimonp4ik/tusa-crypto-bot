"""Export the frozen OKX one-hour HGB model to portable JSON."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

import crypto_okx_swap_1h_regime_oct7_audit as base
from src.portable_hgb import predict_probability


SOURCE_REPORT = Path("reports/audit_2026_09_08/crypto_okx_swap_1h_regime_oct7.json")
OUT = Path("src/frozen_crypto_okx_1h_hgb.json")
VERIFY_OUT = Path("reports/audit_2026_09_08/crypto_okx_swap_1h_hgb_export.json")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def export_nodes(predictor) -> list[list]:
    output = []
    for node in predictor.nodes:
        output.append([
            int(node["feature_idx"]), float(node["num_threshold"]),
            int(node["missing_go_to_left"]), int(node["left"]),
            int(node["right"]), float(node["value"]), int(node["is_leaf"]),
        ])
    return output


def main() -> None:
    report = json.loads(SOURCE_REPORT.read_text(encoding="utf-8"))
    frozen = report.get("frozen_before_2024")
    if report.get("status") != "RESEARCH_CANDIDATE_AWAITING_DIRECT_FORWARD" or not frozen:
        raise ValueError("OKX hourly model is not a frozen candidate")
    candles, sources = base.load()
    raw = base.build_features(candles)
    frame = base.add_outcomes(base.proposals(raw), candles)
    fit = frame.year.isin((2020, 2021)).to_numpy()
    columns = base.schema(frame.loc[fit])
    x = base.matrix(frame, columns)
    labels = frame.target_hit.astype(int).to_numpy()
    config = frozen["model"]
    model = HistGradientBoostingClassifier(
        max_iter=180, learning_rate=.06, max_leaf_nodes=config["leaves"],
        min_samples_leaf=config["leaf"], l2_regularization=config["l2"],
        random_state=20261007)
    model.fit(x[fit], labels[fit])
    artifact = {
        "version": 1, "kind": "binary_hist_gradient_boosting",
        "features": columns,
        "baseline": float(model._baseline_prediction.ravel()[0]),
        "trees": [export_nodes(iteration[0]) for iteration in model._predictors],
        "quality_threshold": float(frozen["threshold"]),
        "bar": "1H", "entry": "next hourly market open",
        "target_pct": .5, "stop_pct": 1.0, "hold_bars": 12,
        "training_manifest_sha256": digest(base.MANIFEST),
        "source_report_sha256": digest(SOURCE_REPORT),
    }
    OUT.write_text(json.dumps(artifact, separators=(",", ":")), encoding="utf-8")
    indices = np.linspace(0, len(x) - 1, min(3000, len(x)), dtype=int)
    expected = model.predict_proba(x[indices])[:, 1]
    observed = np.asarray([
        predict_probability(artifact, row.tolist()) for row in x[indices]],
        dtype=float)
    max_error = float(np.max(np.abs(expected - observed)))
    if max_error > 1e-12:
        raise ValueError(f"portable HGB parity failure: {max_error}")
    verification = {
        "status": "EXACT_PORTABLE_EXPORT", "artifact": str(OUT),
        "artifact_sha256": digest(OUT), "features": len(columns),
        "trees": len(artifact["trees"]),
        "nodes": sum(len(tree) for tree in artifact["trees"]),
        "verified_rows": len(indices), "max_probability_error": max_error,
        "sources": {**sources, str(SOURCE_REPORT): digest(SOURCE_REPORT),
                    str(Path(__file__)): digest(Path(__file__))},
    }
    VERIFY_OUT.write_text(json.dumps(
        verification, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(verification, indent=2), flush=True)


if __name__ == "__main__":
    main()
