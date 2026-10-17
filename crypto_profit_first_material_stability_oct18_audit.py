"""Causal BTC regime-stability gate for the frozen material ensemble."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

import crypto_high_target_tree_oct5_audit as base
import crypto_profit_first_material_barbell_oct18_audit as prior


ROOT = Path("reports/audit_2026_09_08")
SOURCE = ROOT / "crypto_profit_first_material_ensemble_oct18.pkl"
FRESH_SOURCE = ROOT / "crypto_profit_first_material_ensemble_xperp_oct17_fresh.pkl"
FORWARD_SOURCE = ROOT / "crypto_profit_first_material_ensemble_xperp_oct17_forward.pkl"
OUT = ROOT / "crypto_profit_first_material_stability_oct18.json"
ROWS_OUT = ROOT / "crypto_profit_first_material_stability_oct18.pkl"
FRESH_OUT = ROOT / "crypto_profit_first_material_stability_xperp_oct17_fresh.pkl"
FORWARD_OUT = ROOT / "crypto_profit_first_material_stability_xperp_oct17_forward.pkl"
SELECT = prior.SELECT
CONFIRM = prior.CONFIRM
VALIDATION = prior.VALIDATION
LATER = prior.LATER
MARGINS_ATR = (0.0, .25, .50, .75, 1.0, 1.5)
EFFICIENCIES = (0.0, .10, .20, .30)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def enrich(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["btc_regime_margin_atr"] = result[
        ["btc_fast_slow_atr", "btc_slow_long_atr"]].abs().min(axis=1)
    return result


def apply_gate(frame: pd.DataFrame, margin_atr: float,
               efficiency: float) -> pd.DataFrame:
    return frame[
        (frame.btc_regime_margin_atr >= margin_atr)
        & (frame.btc_eff20 >= efficiency)].copy()


def detailed(frame: pd.DataFrame, periods: tuple[str, ...], margin_atr: float,
             efficiency: float) -> dict:
    def one(labels: tuple[str, ...]) -> dict:
        part = apply_gate(frame[frame.period.isin(labels)], margin_atr, efficiency)
        return base.evaluate(part.to_dict("records"))
    return {"all": one(periods),
            "periods": {period: one((period,)) for period in periods}}


def development_ok(selection: dict, confirmation: dict) -> bool:
    a, b = selection["all"], confirmation["all"]
    return bool(a["return_pct"] > 0 and b["return_pct"] > 0
                and a["dd_pct"] <= 10 and b["dd_pct"] <= 10
                and a["trades_per_calendar_day"] >= 3
                and b["trades_per_calendar_day"] >= 2
                and all(item["return_pct"] > 0
                        for item in confirmation["periods"].values()))


def direct(frame: pd.DataFrame, margin_atr: float,
           efficiency: float, out: Path) -> dict:
    rows = apply_gate(frame, margin_atr, efficiency)
    rows.to_pickle(out)
    return base.evaluate(rows.to_dict("records"))


def main() -> None:
    frame = enrich(pd.read_pickle(SOURCE))
    candidates = []
    for margin_atr in MARGINS_ATR:
        for efficiency in EFFICIENCIES:
            selection = detailed(frame, SELECT, margin_atr, efficiency)
            confirmation = detailed(frame, CONFIRM, margin_atr, efficiency)
            if not development_ok(selection, confirmation):
                continue
            candidates.append({
                "minimum_regime_margin_atr": margin_atr,
                "minimum_btc_eff20": efficiency,
                "selection_2022h1": selection,
                "confirmation_2022h2_2023": confirmation,
                "score": prior.development_score(selection, confirmation),
            })
    candidates.sort(key=lambda item: item["score"], reverse=True)
    frozen = candidates[0] if candidates else None
    validation = later = fresh = forward = checks = None
    status = "REJECTED_DEVELOPMENT"
    if frozen:
        margin_atr = frozen["minimum_regime_margin_atr"]
        efficiency = frozen["minimum_btc_eff20"]
        validation = detailed(frame, VALIDATION, margin_atr, efficiency)
        later = detailed(frame, LATER, margin_atr, efficiency)
        apply_gate(frame, margin_atr, efficiency).to_pickle(ROWS_OUT)
        fresh = direct(enrich(pd.read_pickle(FRESH_SOURCE)), margin_atr,
                       efficiency, FRESH_OUT)
        forward = direct(enrich(pd.read_pickle(FORWARD_SOURCE)), margin_atr,
                         efficiency, FORWARD_OUT)
        historical = {"selection": frozen["selection_2022h1"],
                      "confirmation": frozen["confirmation_2022h2_2023"],
                      "validation": validation, "later": later}
        checks = {name: prior.objective_ok(value)
                  for name, value in historical.items()}
        status = ("MATERIAL_STABILITY_RESEARCH_CANDIDATE"
                  if all(checks.values()) else "REJECTED_OBJECTIVE")
    report = {
        "status": status, "ready_for_live_orders": False,
        "protocol": {
            "priority": "account return first; win rate diagnostic only",
            "selection": [*SELECT, *CONFIRM],
            "frozen_validation": list(VALIDATION), "later": list(LATER),
            "regime_margin": "min(abs(BTC fast-slow ATR), abs(BTC slow-long ATR))",
            "causality": "all BTC features use candles closed before entry",
            "entry": "market only", "money": "3% current-equity margin, fixed 10x gross",
            "objective": ">=30% account return every month; >=3 trades/day; DD <=10%",
        },
        "candidate_count": len(candidates),
        "development_frontier": candidates,
        "frozen_before_2024": frozen,
        "validation_2024": validation, "later_2025_2026": later,
        "fresh_direct_xperp": fresh, "untouched_forward": forward,
        "objective_checks": checks,
        "sources": {str(path): digest(path) for path in (
            SOURCE, FRESH_SOURCE, FORWARD_SOURCE, Path(__file__))},
        "limitations": [
            "Filtering an already gated portfolio is conservative because suppressed alternatives are not backfilled.",
            "The post-prior-file direct forward is only about two weeks.",
            "Gross returns exclude fees at user request.",
        ],
    }
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"status": status, "frozen": frozen,
                      "validation": validation, "later": later,
                      "fresh_direct": fresh, "untouched_forward": forward},
                     indent=2), flush=True)


if __name__ == "__main__":
    main()
