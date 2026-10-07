"""Export the three frozen OKX one-hour target-upgrade models."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

import crypto_okx_swap_1h_regime_oct7_audit as base
import crypto_okx_swap_1h_target_cascade_oct7_audit as cascade
from crypto_export_okx_1h_hgb_artifact import export_nodes
from src.portable_hgb import predict_probability


SOURCE_REPORT = Path(
    "reports/audit_2026_09_08/crypto_okx_swap_1h_target_cascade_oct7.json")
OUT = Path("src/frozen_crypto_okx_1h_target_cascade.json")
VERIFY_OUT = Path(
    "reports/audit_2026_09_08/crypto_okx_swap_1h_target_cascade_export.json")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    report = json.loads(SOURCE_REPORT.read_text(encoding="utf-8"))
    frozen = report.get("frozen_before_2023")
    if report.get("status") not in ("CASCADE_RESEARCH_CANDIDATE", "REJECTED_LATER") or not frozen:
        raise ValueError("target cascade was not frozen before later years")
    candles, sources = base.load()
    raw = base.build_features(candles)
    frame = base.proposals(raw).reset_index(drop=True)
    for target in cascade.TARGETS:
        cascade.add_target(frame, candles, target)
    fit = frame.year.isin((2020, 2021)).to_numpy()
    columns = base.schema(frame.loc[fit])
    x = base.matrix(frame, columns)
    indices = np.linspace(0, len(x) - 1, min(3000, len(x)), dtype=int)
    models = {}
    max_error = 0.0
    total_nodes = 0
    for target in (1.0, 1.5, 2.0):
        suffix = str(target).replace(".", "_")
        model = cascade.estimator()
        labels = frame[f"hit_{suffix}"].astype(int).to_numpy()
        model.fit(x[fit], labels[fit])
        portable = {
            "features": columns,
            "baseline": float(model._baseline_prediction.ravel()[0]),
            "trees": [export_nodes(iteration[0]) for iteration in model._predictors],
        }
        expected = model.predict_proba(x[indices])[:, 1]
        observed = np.asarray([
            predict_probability(portable, row.tolist()) for row in x[indices]],
            dtype=float)
        error = float(np.max(np.abs(expected - observed)))
        max_error = max(max_error, error)
        total_nodes += sum(len(tree) for tree in portable["trees"])
        models[str(target)] = {
            **portable,
            "quality_threshold": float(frozen["thresholds"][str(target)]),
            "target_pct": target, "hold_bars": cascade.HOLD[target],
        }
    if max_error > 1e-12:
        raise ValueError(f"portable cascade parity failure: {max_error}")
    artifact = {
        "version": 1, "kind": "okx_1h_target_cascade",
        "base_target_pct": .5, "stop_pct": 1.0,
        "models": models,
        "source_report_sha256": digest(SOURCE_REPORT),
        "training_manifest_sha256": digest(base.MANIFEST),
    }
    OUT.write_text(json.dumps(artifact, separators=(",", ":")), encoding="utf-8")
    verification = {
        "status": "EXACT_PORTABLE_CASCADE_EXPORT", "artifact": str(OUT),
        "artifact_sha256": digest(OUT), "models": len(models),
        "features_per_model": len(columns), "trees_per_model": 180,
        "nodes": total_nodes, "verified_rows_per_model": len(indices),
        "max_probability_error": max_error,
        "sources": {**sources, str(SOURCE_REPORT): digest(SOURCE_REPORT),
                    str(Path(__file__)): digest(Path(__file__))},
    }
    VERIFY_OUT.write_text(json.dumps(
        verification, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(verification, indent=2), flush=True)


if __name__ == "__main__":
    main()
