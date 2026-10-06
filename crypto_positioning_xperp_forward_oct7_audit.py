"""Frozen top-trader divergence on Oct 1-5 direct X-Perp candles."""
from __future__ import annotations

import hashlib
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

import crypto_5m_regime_router_literal_oct6_audit as price
import crypto_futures_funding_regime_oct6_audit as futures_source
import crypto_futures_positioning_regime_oct7_audit as study


ROOT = Path("reports/audit_2026_09_08")
FORWARD_ROOT = ROOT / "crypto_positioning_oct7_forward"
XPERP_ROOT = ROOT / "xperp_5m_oct6_forward"
OUT = ROOT / "crypto_positioning_xperp_forward_oct7.json"
ROWS_OUT = ROOT / "crypto_positioning_xperp_forward_oct7.csv"
THRESHOLD = .7050843607886652
START = pd.Timestamp("2026-10-01", tz="UTC")
END = pd.Timestamp("2026-10-06", tz="UTC")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def combine(left: dict, right: dict) -> dict:
    rows = {}
    fields = [key for key in left if key != "time"]
    for source in (left, right):
        for index, stamp in enumerate(source["time"]):
            rows[int(stamp)] = tuple(source[field][index] for field in fields)
    ordered = sorted(rows)
    result = {"time": ordered}
    for index, field in enumerate(fields):
        result[field] = [rows[stamp][index] for stamp in ordered]
    return result


def load_forward_metrics(history: dict) -> tuple[dict, dict]:
    manifest_path = FORWARD_ROOT / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    sources = {str(manifest_path): digest(manifest_path),
               str(FORWARD_ROOT / "freeze.json"): digest(FORWARD_ROOT / "freeze.json")}
    result = {}
    for symbol in futures_source.SYMBOLS:
        path = Path(manifest["symbols"][symbol]["path"])
        raw = path.read_bytes()
        found = hashlib.sha256(raw).hexdigest()
        if found != manifest["symbols"][symbol]["sha256"]:
            raise ValueError(f"forward metrics hash mismatch: {symbol}")
        result[symbol] = combine(history[symbol], pickle.loads(raw))
        sources[str(path)] = found
    return result, sources


def load_xperp() -> tuple[dict, dict]:
    manifest_path = XPERP_ROOT / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    result, sources = {}, {str(manifest_path): digest(manifest_path)}
    for symbol in futures_source.SYMBOLS:
        path = Path(manifest["symbols"][symbol]["path"])
        raw = path.read_bytes()
        found = hashlib.sha256(raw).hexdigest()
        if found != manifest["symbols"][symbol]["sha256"]:
            raise ValueError(f"X-Perp hash mismatch: {symbol}")
        result[symbol] = pickle.loads(raw)
        sources[str(path)] = found
    return result, sources


def main() -> None:
    binance, _, price_sources = futures_source.load()
    historical_metrics, metric_sources = study.load_metrics()
    combined_metrics, forward_sources = load_forward_metrics(historical_metrics)
    xperp, xperp_sources = load_xperp()

    training_features = study.attach_metrics(price.build_features(binance),
                                             historical_metrics)
    training = price.add_outcomes(study.proposals(training_features), binance)
    fit = study.period_mask(training, "fit_2023")
    schema = price.columns(training.loc[fit])
    _, training_matrix = study.matrices(training, schema)
    model = HistGradientBoostingClassifier(
        max_iter=180, learning_rate=.06, max_leaf_nodes=31,
        min_samples_leaf=400, l2_regularization=6.0, random_state=20261007)
    model.fit(training_matrix[fit], training.target_hit.astype(int).to_numpy()[fit])

    direct_features = study.attach_metrics(price.build_features(xperp), combined_metrics)
    direct = price.add_outcomes(study.proposals(direct_features), xperp)
    _, direct_matrix = study.matrices(direct, schema)
    direct["quality"] = model.predict_proba(direct_matrix)[:, 1]
    stamp = pd.to_datetime(direct.entry_time, unit="s", utc=True)
    eligible = direct[(stamp >= START) & (stamp < END)
                      & direct.positioning_family_top_divergence.eq(1)].copy()
    eligible.to_csv(ROWS_OUT, index=False)
    result = price.evaluate(eligible, (2026,), THRESHOLD)
    selected = price.portfolio_rows(eligible, (2026,), THRESHOLD)
    by_day = {}
    for row in selected:
        day = pd.Timestamp(row["entry_time"], unit="s", tz="UTC").strftime("%Y-%m-%d")
        item = by_day.setdefault(day, {"n": 0, "target_hits": 0,
                                      "stops": 0, "timeouts": 0})
        item["n"] += 1
        item["target_hits"] += int(bool(row["target_hit"]))
        item["stops"] += int(row["exit_reason"] == 0)
        item["timeouts"] += int(row["exit_reason"] == 2)
    report = {
        "status": "INSUFFICIENT_FORWARD_EVIDENCE",
        "protocol": {"family": "top_divergence", "threshold": THRESHOLD,
                     "model": {"leaves": 31, "leaf": 400, "l2": 6.0},
                     "period": [START.isoformat(), END.isoformat()],
                     "execution": "direct X-Perp next 5m market open",
                     "target_pct": .5, "stop_pct": 1.0, "max_bars": 24,
                     "money": "3% equity margin, fixed 10x gross",
                     "win_definition": "literal target touch"},
        "rows": {"direct_features": len(direct_features),
                 "all_proposals": len(direct), "eligible_family": len(eligible),
                 "selected": len(selected)},
        "result": result, "by_day": by_day,
        "minimum_forward_gate": {"trades": 100, "days": 21,
                                 "literal_target_hit_rate": 70,
                                 "positive_return": True, "maximum_dd_pct": 5},
        "sources": {**price_sources, **metric_sources, **forward_sources,
                    **xperp_sources,
                    str(Path(study.__file__)): digest(Path(study.__file__)),
                    str(Path(__file__)): digest(Path(__file__))},
        "limitations": [
            "Only five complete forward days are available.",
            "The family was selected after historical subgroup inspection.",
            "This evidence cannot activate runtime trading.",
        ],
    }
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"status": report["status"], "rows": report["rows"],
                      "result": result, "by_day": by_day}, indent=2), flush=True)


if __name__ == "__main__":
    main()
