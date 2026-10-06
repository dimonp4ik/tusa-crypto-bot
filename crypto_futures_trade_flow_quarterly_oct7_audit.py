"""Quarterly causal refit for the 0.5%-to-2% trade-flow ladder."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

import crypto_5m_regime_router_literal_oct6_audit as price
import crypto_futures_trade_flow_family_health_oct7_audit as family_health
import crypto_futures_trade_flow_health_gate_oct7_audit as health
import crypto_futures_trade_flow_ladder_oct7_audit as ladder
import crypto_futures_trade_flow_regime_oct7_audit as flow


OUT = Path("reports/audit_2026_09_08/crypto_futures_trade_flow_quarterly_oct7.json")
ROWS_OUT = Path("reports/audit_2026_09_08/crypto_futures_trade_flow_quarterly_oct7.pkl")
MODEL = {"leaves": 31, "leaf": 400, "l2": 5.0}
QUANTILE = .925
CALIBRATION_DAYS = 180
FAMILY_SPEC = {"review_days": 1, "lookback_days": 14,
               "minimum_completed": 20, "hit_min": .64,
               "mean_min": -.00025}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    candles, sources = flow.load()
    features = flow.attach_flow(price.build_features(candles), candles)
    proposals = flow.proposals(features)
    training = price.add_outcomes(proposals, candles)
    schema = price.columns(training[training.year.isin((2020, 2021))])
    _, matrix = flow.matrices(training, schema)
    labels = training.target_hit.astype(int).to_numpy()
    stamps = pd.to_datetime(training.entry_time, unit="s", utc=True)
    selected = np.zeros(len(training), dtype=bool)
    periods = []
    for start in pd.date_range("2022-01-01", "2026-10-01",
                               freq="QS", tz="UTC"):
        end = start + pd.offsets.QuarterBegin(startingMonth=1)
        calibration_start = start-pd.Timedelta(days=CALIBRATION_DAYS)
        train = (stamps < calibration_start).to_numpy()
        calibration = ((stamps >= calibration_start) & (stamps < start)).to_numpy()
        target = ((stamps >= start) & (stamps < end)).to_numpy()
        if not target.any():
            continue
        model = HistGradientBoostingClassifier(
            max_iter=180, learning_rate=.06, max_leaf_nodes=MODEL["leaves"],
            min_samples_leaf=MODEL["leaf"], l2_regularization=MODEL["l2"],
            random_state=20261007+int(start.strftime("%Y%m")))
        model.fit(matrix[train], labels[train])
        calibration_scores = model.predict_proba(matrix[calibration])[:, 1]
        threshold = float(np.quantile(calibration_scores, QUANTILE))
        target_scores = model.predict_proba(matrix[target])[:, 1]
        selected[target] = target_scores >= threshold
        periods.append({
            "start": start.isoformat(), "end": end.isoformat(),
            "train_rows": int(train.sum()),
            "calibration_rows": int(calibration.sum()),
            "target_rows": int(target.sum()),
            "threshold": threshold, "selected_rows": int(selected[target].sum()),
        })

    resolved = ladder.add_ladder(proposals, candles, **health.EXIT)
    keys = ["symbol", "entry_time", "direction"]
    if not resolved[keys].reset_index(drop=True).equals(
            training[keys].reset_index(drop=True)):
        raise ValueError("ladder/model row alignment mismatch")
    resolved["quality"] = selected.astype(float)
    pieces = []
    for year in range(2022, 2027):
        rows = price.portfolio_rows(resolved, (year,), .5)
        if rows:
            pieces.append(pd.DataFrame(rows))
    base = pd.concat(pieces, ignore_index=True)
    featured = family_health.add_family_health(
        base, FAMILY_SPEC["review_days"], FAMILY_SPEC["lookback_days"])
    gated = family_health.apply_gate(featured, FAMILY_SPEC)
    gated.to_pickle(ROWS_OUT, compression="gzip")
    base_results = {str(year): health.evaluate(base, year)
                    for year in range(2022, 2027)}
    results = {str(year): health.evaluate(gated, year)
               for year in range(2022, 2027)}
    checks = {year: family_health.validation_ok(value)
              for year, value in results.items()}
    report = {
        "status": ("RESEARCH_ONLY_NEEDS_NEW_FORWARD"
                   if all(checks.values()) else "REJECTED"),
        "protocol": {
            "refit": "calendar-quarter start",
            "training": "all rows before calibration window",
            "calibration": f"previous {CALIBRATION_DAYS} days",
            "quantile": QUANTILE, "model": MODEL,
            "family_health": FAMILY_SPEC,
            "exit": {"near_pct": .5, **health.EXIT,
                     "initial_stop_pct": 1.0},
            "money": "3% equity margin, fixed 10x gross",
            "warning": "architecture proposed after observing 2026; all history is development evidence",
        },
        "raw_proposals": len(proposals), "base_rows": len(base),
        "gated_rows": len(gated), "periods": periods,
        "base_results": base_results, "results": results, "checks": checks,
        "sources": {**sources,
                    str(Path(family_health.__file__)): digest(Path(family_health.__file__)),
                    str(Path(__file__)): digest(Path(__file__))},
        "limitations": [
            "Quarterly refitting was proposed after annual 2026 stress failed.",
            "No historical period remains untouched for this architecture.",
            "Passing history would require a new direct X-Perp shadow forward.",
        ],
    }
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"status": report["status"],
                      "base_results": base_results,
                      "results": results, "checks": checks}, indent=2), flush=True)


if __name__ == "__main__":
    main()
