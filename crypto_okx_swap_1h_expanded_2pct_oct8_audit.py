"""Cross-symbol 2% OKX one-hour regime audit on an expanded universe.

The model deliberately omits symbol identity.  It must learn transferable
directional regimes on 2020-2021, select one probability threshold on 2022,
and confirm it on 2023 before any later year may be inspected.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

import crypto_5m_regime_router_literal_oct6_audit as portfolio
import crypto_okx_swap_1h_regime_oct7_audit as base
import crypto_okx_swap_1h_target_cascade_oct7_audit as target_engine
from src.research_profit_gate import economic_gate, minimum_binary_hit_rate_pct


ROOT = Path("reports/audit_2026_09_08")
MANIFEST = ROOT / "crypto_okx_swap_1h_expanded_oct8" / "manifest.json"
OUT = ROOT / "crypto_okx_swap_1h_expanded_2pct_oct8.json"
TARGET_PCT = 2.0
STOP_PCT = 1.0
HOLD_BARS = 48
CONFIGS = (
    {"leaves": 15, "leaf": 900, "l2": 6.0},
    {"leaves": 31, "leaf": 1600, "l2": 10.0},
)
QUANTILES = (.50, .60, .70, .75, .80, .85, .875, .90, .925, .94,
             .95, .96, .97, .975, .98, .985, .99, .995)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def evaluate(frame, years: tuple[int, ...], threshold: float) -> dict:
    metrics = portfolio.evaluate(frame, years, threshold)
    rows = portfolio.portfolio_rows(frame, years, threshold)
    gate = economic_gate(
        [float(row["signed_price_return"]) for row in rows],
        metrics["trades_per_calendar_day"], market_days=30)
    return {"portfolio": metrics, "economic_gate": gate}


def development_pass(value: dict) -> bool:
    metrics, gate = value["portfolio"], value["economic_gate"]
    return bool(metrics["n"] >= 900
                and metrics["literal_target_hit_rate"] >= 65
                and metrics["dd_pct"] <= 8
                and metrics["negative_months"] == 0
                and gate["passes"])


def rank(value: dict) -> tuple[float, float, float, float]:
    metrics, gate = value["portfolio"], value["economic_gate"]
    return (gate["projected_monthly_return_pct"],
            metrics["literal_target_hit_rate"],
            -metrics["negative_months"], -metrics["dd_pct"])


def main() -> None:
    base.MANIFEST = MANIFEST
    candles, sources = base.load()
    raw = base.build_features(candles)
    frame = base.proposals(raw).reset_index(drop=True)
    target_engine.add_target(frame, candles, TARGET_PCT)
    frame["signed_return"] = frame["signed_2_0"]
    frame["exit_time"] = frame["exit_2_0"]
    frame["target_hit"] = frame["hit_2_0"]
    frame["exit_reason"] = frame["reason_2_0"]
    frame["selected_target_pct"] = TARGET_PCT
    fit = frame.year.isin((2020, 2021)).to_numpy()
    # No symbol one-hot columns: later instruments must transfer by regime.
    columns = [*base.FEATURES,
               *[f"family_{name}" for name in base.FAMILIES]]
    x = base.matrix(frame, columns)
    labels = frame.target_hit.astype(int).to_numpy()
    candidates = []
    model_reports = []
    for config in CONFIGS:
        model = HistGradientBoostingClassifier(
            max_iter=180, learning_rate=.06,
            max_leaf_nodes=config["leaves"],
            min_samples_leaf=config["leaf"],
            l2_regularization=config["l2"], random_state=20261008)
        model.fit(x[fit], labels[fit])
        scored = frame.copy()
        scored["quality"] = model.predict_proba(x)[:, 1]
        selection_scores = scored.loc[scored.year.eq(2022), "quality"]
        frontier = []
        for quantile in QUANTILES:
            threshold = float(selection_scores.quantile(quantile))
            selection = evaluate(scored, (2022,), threshold)
            frontier.append({
                "quantile": quantile, "threshold": threshold,
                "selection_2022": selection,
            })
        frontier.sort(key=lambda item: rank(item["selection_2022"]), reverse=True)
        for item in frontier[:10]:
            validation = evaluate(scored, (2023,), item["threshold"])
            candidates.append({
                "model": config, **item, "validation_2023": validation,
                "development_pass": (
                    development_pass(item["selection_2022"])
                    and development_pass(validation)),
                "_frame": scored,
            })
        model_reports.append({
            "model": config, "fit_rows": int(fit.sum()),
            "fit_target_prevalence_pct": round(100 * float(labels[fit].mean()), 4),
            "frontier_top5": frontier[:5],
        })
        print(json.dumps({"model": config, "best": frontier[0]}, default=float),
              flush=True)
    eligible = [item for item in candidates if item["development_pass"]]
    eligible.sort(key=lambda item: (
        min(rank(item["selection_2022"])[0],
            rank(item["validation_2023"])[0]),
        min(rank(item["selection_2022"])[1],
            rank(item["validation_2023"])[1])), reverse=True)
    frozen = eligible[0] if eligible else None
    hidden = None
    if frozen is not None:
        hidden = {str(year): evaluate(
            frozen["_frame"], (year,), frozen["threshold"])
            for year in (2024, 2025, 2026)}
    candidates.sort(key=lambda item: min(
        rank(item["selection_2022"])[0],
        rank(item["validation_2023"])[0]), reverse=True)
    public = [{key: value for key, value in item.items() if key != "_frame"}
              for item in candidates[:20]]
    status = "REJECTED_DEVELOPMENT"
    if frozen is not None:
        status = ("RESEARCH_CANDIDATE" if all(
            development_pass(value) for value in hidden.values())
                  else "REJECTED_HIDDEN")
    report = {
        "status": status, "ready_for_live_orders": False,
        "protocol": {
            "venue": "OKX global USDT swap", "bar": "1H",
            "symbols": sorted(candles), "symbol_identity_feature": False,
            "target_pct": TARGET_PCT, "stop_pct": STOP_PCT,
            "hold_bars": HOLD_BARS, "entry": "next hourly market open",
            "fit": [2020, 2021], "selection": 2022,
            "validation": 2023,
            "hidden_only_after_development_pass": [2024, 2025, 2026],
            "money": "3% current equity margin, fixed 10x gross",
            "fees_and_slippage": "excluded per client research scope",
            "ambiguous_bar": "stop first",
        },
        "feature_rows": len(raw), "proposal_rows": len(frame),
        "idealized_minimum_hit_rate_pct": minimum_binary_hit_rate_pct(
            TARGET_PCT, STOP_PCT, market_days=30),
        "models": model_reports, "development_top20": public,
        "frozen_before_hidden": ({key: value for key, value in frozen.items()
                                  if key != "_frame"} if frozen else None),
        "hidden": hidden,
        "decision": "No runtime change unless all chronological stages pass.",
        "sources": {**sources, str(MANIFEST): digest(MANIFEST),
                    str(Path(__file__)): digest(Path(__file__))},
    }
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"status": status,
                      "proposal_rows": len(frame),
                      "best_development": public[:3],
                      "hidden": hidden}, indent=2, default=float), flush=True)


if __name__ == "__main__":
    main()
