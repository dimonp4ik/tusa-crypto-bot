"""Causal cluster-stop rails for the frozen material crypto signal policy.

Only loss-pause and rolling-hour entry caps are chosen on 2022-2023.  The
signal model, targets, and union regime/family routes remain frozen.  Later
years and the refreshed X-Perp window are evaluation only.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import pandas as pd

import crypto_05_2r_precision_barbell_oct5_audit as legacy
import crypto_high_target_tree_oct5_audit as base
import crypto_profit_first_material_barbell_oct18_audit as prior
import crypto_profit_first_material_regime_family_oct18_audit as family
import crypto_profit_first_material_target_oct18_audit as target
import crypto_profit_first_material_xperp_oct20_audit as refresh
from crypto_runtime_shadow_portfolio_oct3_audit import money, runtime_gate


ROOT = Path("reports/audit_2026_09_08")
OUT = ROOT / "crypto_profit_first_cluster_rails_oct20.json"
ROWS_OUT = ROOT / "crypto_profit_first_cluster_rails_xperp_oct20.pkl"
FRESH_OUT = ROOT / "crypto_profit_first_cluster_rails_xperp_oct20_fresh.pkl"
LOSS_PAUSES = (1, 2, 3)
HOURLY_CAPS = (1, 2)
FIT, SELECT, CONFIRM = prior.FIT, prior.SELECT, prior.CONFIRM
VALIDATION, LATER = prior.VALIDATION, prior.LATER


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def route_rows(frame: pd.DataFrame, periods: tuple[str, ...], routes: dict,
               loss_pause: int, hourly_cap: int) -> pd.DataFrame:
    staged = base.stage(frame, periods)
    parts = []
    for route in routes.values():
        part = staged[
            staged.btc_regime.eq(route["btc_regime"])
            & staged.family.eq(route["family"])
            & (staged.model_probability >= float(route["threshold"]))].copy()
        if len(part):
            parts.append(target.barbell_frame(
                part, float(route["near_fraction"]), family.TARGET_R))
    if not parts:
        return staged.iloc[0:0].copy()
    chosen = pd.concat(parts, ignore_index=True)
    return pd.DataFrame(runtime_gate(base.standardized(
        chosen, "profit_first_material_cluster_rails"),
        loss_pause=loss_pause, hourly_cap=hourly_cap))


def one(frame: pd.DataFrame, periods: tuple[str, ...], routes: dict,
        loss_pause: int, hourly_cap: int, margin: float) -> dict:
    rows = route_rows(frame, periods, routes, loss_pause, hourly_cap)
    return money(rows.to_dict("records"), margin)


def detailed(frame: pd.DataFrame, periods: tuple[str, ...], routes: dict,
             loss_pause: int, hourly_cap: int, margin: float) -> dict:
    return {"all": one(frame, periods, routes, loss_pause, hourly_cap, margin),
            "periods": {period: one(frame, (period,), routes, loss_pause,
                                    hourly_cap, margin) for period in periods}}


def monthly_equivalent(value: dict) -> float:
    months = max(1, len(value["all"].get("monthly_return_pct", {})))
    factor = 1 + value["all"]["return_pct"] / 100
    return 100 * math.expm1(math.log(factor) / months) if factor > 0 else -100


def eligible(selection: dict, confirmation: dict) -> bool:
    a, b = selection["all"], confirmation["all"]
    return bool(a["return_pct"] > 0 and b["return_pct"] > 0
                and a["dd_pct"] <= 15 and b["dd_pct"] <= 15
                and a["trades_per_calendar_day"] >= 3
                and b["trades_per_calendar_day"] >= 2
                and all(item["return_pct"] > 0
                        for item in confirmation["periods"].values()))


def development_score(selection: dict, confirmation: dict) -> float:
    monthly = min(monthly_equivalent(selection), monthly_equivalent(confirmation))
    drawdown = max(selection["all"]["dd_pct"], confirmation["all"]["dd_pct"])
    negative = selection["all"]["negative_months"] + confirmation["all"]["negative_months"]
    return round(monthly - .20 * drawdown - .20 * negative, 8)


def main() -> None:
    model, schema, routes, _ = refresh.train_frozen()
    historical = target.load_joined(family.TARGET_R)
    scored = legacy.scored(historical, model, schema)
    candidates = []
    for loss_pause in LOSS_PAUSES:
        for hourly_cap in HOURLY_CAPS:
            selection = detailed(scored, SELECT, routes, loss_pause,
                                 hourly_cap, .05)
            confirmation = detailed(scored, CONFIRM, routes, loss_pause,
                                    hourly_cap, .05)
            if eligible(selection, confirmation):
                candidates.append({
                    "loss_pause": loss_pause, "hourly_cap": hourly_cap,
                    "selection_2022h1": selection,
                    "confirmation_2022h2_2023": confirmation,
                    "score": development_score(selection, confirmation),
                })
    candidates.sort(key=lambda item: item["score"], reverse=True)
    frozen = candidates[0] if candidates else None
    if frozen is None:
        raise ValueError("no causal cluster-rail candidate")
    lp, cap = frozen["loss_pause"], frozen["hourly_cap"]
    validation = detailed(scored, VALIDATION, routes, lp, cap, .05)
    later = detailed(scored, LATER, routes, lp, cap, .05)

    old = prior.join_direct_outcomes(refresh.OLD_FAR, refresh.OLD_NEAR)
    new = prior.join_direct_outcomes(refresh.NEW_FAR, refresh.NEW_NEAR)
    cutoff = float(old.entry_time.max())
    new_scored = legacy.scored(new[
        new.symbol.isin(base.ROBUST_SYMBOLS)
        & new.risk_pct.between(legacy.RISK_PCT_MIN, legacy.RISK_PCT_MAX,
                               inclusive="both")].copy(), model, schema)
    new_scored["period"] = "direct"
    all_rows = route_rows(new_scored, ("direct",), routes, lp, cap)
    fresh_rows = all_rows[all_rows.entry_time > cutoff].copy()
    all_rows.to_pickle(ROWS_OUT)
    fresh_rows.to_pickle(FRESH_OUT)
    results = {
        "all_refreshed_xperp": {
            f"margin_{int(100*m)}pct": refresh.diagnostics(all_rows, m)
            for m in refresh.MARGINS},
        "untouched_after_oct17_raw_cutoff": {
            f"margin_{int(100*m)}pct": refresh.diagnostics(fresh_rows, m)
            for m in refresh.MARGINS},
    }
    all5 = results["all_refreshed_xperp"]["margin_5pct"]
    fresh5 = results["untouched_after_oct17_raw_cutoff"]["margin_5pct"]
    pass_gate = bool(
        len(fresh_rows) >= 30 and fresh5["return_pct"] > 0
        and all5["dd_pct"] <= 10 and all5["negative_months"] == 0
        and all5["economic_30pct_month_gate"])
    report = {
        "status": ("CLUSTER_RAILS_SHADOW_CANDIDATE" if pass_gate
                   else "CLUSTER_RAILS_REJECTED_OBJECTIVE"),
        "ready_for_live_orders": False,
        "protocol": {
            "signal_policy": "frozen Oct-18 union regime/family material route",
            "rail_selection": "2022-2023 only; 5% fixed client margin",
            "tested_loss_pauses": list(LOSS_PAUSES),
            "tested_hourly_caps": list(HOURLY_CAPS),
            "orders": "market only", "leverage": 10,
        },
        "development_candidates": candidates,
        "frozen_before_2024": {key: frozen[key] for key in (
            "loss_pause", "hourly_cap", "score")},
        "validation_2024": validation, "later_2025_2026": later,
        "results": results, "economic_pass": pass_gate,
        "sources": {str(path): digest(path) for path in (
            refresh.FROZEN_REPORT, refresh.NEW_NEAR, refresh.NEW_FAR,
            Path(__file__))},
        "limitations": [
            "The untouched refresh is only a few calendar days.",
            "The rail grid is small and fixed, but still selected on historical data.",
            "Gross returns exclude fees at user request.",
        ],
    }
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"status": report["status"],
                      "frozen": report["frozen_before_2024"],
                      "validation": validation["all"],
                      "later": later["all"], "results": results,
                      "economic_pass": pass_gate}, indent=2), flush=True)


if __name__ == "__main__":
    main()
