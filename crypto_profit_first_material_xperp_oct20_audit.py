"""Untouched Oct-20 refresh of the frozen profit-first material crypto route.

The model and regime/family routing policy come from the Oct-18 report.  This
script does not select a threshold on the refreshed X-Perp candles.  It checks
overlap parity, then scores only entries that were not observable in the prior
raw replay.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

import crypto_05_2r_precision_barbell_oct5_audit as legacy
import crypto_high_target_tree_oct5_audit as base
import crypto_profit_first_material_barbell_oct18_audit as barbell
import crypto_profit_first_material_ensemble_oct18_audit as ensemble
import crypto_profit_first_material_regime_family_oct18_audit as family
import crypto_profit_first_material_target_oct18_audit as target
from crypto_runtime_shadow_portfolio_oct3_audit import money


ROOT = Path("reports/audit_2026_09_08")
FROZEN_REPORT = ROOT / "crypto_profit_first_material_ensemble_oct18.json"
OLD_NEAR = ROOT / "crypto_xperp_raw_router_audit_0_5_oct17_material_raw_raw_families.csv"
OLD_FAR = ROOT / "crypto_xperp_raw_router_audit_2_0_oct17_material_raw_raw_families.csv"
NEW_NEAR = ROOT / "crypto_xperp_raw_router_audit_0_5_oct20_material_raw_raw_families.csv"
NEW_FAR = ROOT / "crypto_xperp_raw_router_audit_2_0_oct20_material_raw_raw_families.csv"
MANIFEST = ROOT / "xperp_15m_oct20_material" / "manifest.json"
EXACT_PATH_REPORT = ROOT / "crypto_material_exact_path_oct20.json"
OUT = ROOT / "crypto_profit_first_material_xperp_oct20.json"
ROWS_OUT = ROOT / "crypto_profit_first_material_xperp_oct20.pkl"
FRESH_ROWS_OUT = ROOT / "crypto_profit_first_material_xperp_oct20_fresh.pkl"
MARGINS = (0.03, 0.05)
LEVERAGE = 10.0
TARGET_MONTH = 0.30
TARGET_TRADES_PER_DAY = 3.0


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def required_signed_move(margin: float, trades_per_day: float,
                         days: int = 30) -> float:
    """Constant signed price return needed per trade for TARGET_MONTH."""
    periods = trades_per_day * days
    return math.expm1(math.log1p(TARGET_MONTH) / periods) / (margin * LEVERAGE)


def projected_month(mean_signed: float, trades_per_day: float,
                    margin: float, days: int = 30) -> float:
    factor = 1 + margin * LEVERAGE * mean_signed
    return 100 * math.expm1(trades_per_day * days * math.log(factor)) \
        if factor > 0 and trades_per_day > 0 else -100.0


def train_frozen() -> tuple[object, dict, dict, dict[str, dict]]:
    report = json.loads(FROZEN_REPORT.read_text(encoding="utf-8"))
    frozen = report["frozen_before_2024"]
    if not frozen or frozen.get("policy") != "union":
        raise ValueError("unexpected frozen material policy")
    frame = target.load_joined(ensemble.TARGET_R)
    fit = base.stage(frame, ensemble.FIT)
    schema = base.design(fit)
    model = HistGradientBoostingClassifier(
        **ensemble.MODEL, max_iter=150, l2_regularization=1.0,
        early_stopping=False, random_state=20260908)
    model.fit(base.matrix(fit, schema), fit.outcome.eq("TP2").astype(int))
    scored = legacy.scored(frame, model, schema)
    broad_report = json.loads(ensemble.BROAD_REPORT.read_text(encoding="utf-8"))
    family_report = json.loads(ensemble.FAMILY_REPORT.read_text(encoding="utf-8"))
    policies = ensemble.policy_maps(
        scored, broad_report["routes_frozen_before_2024"],
        family_report["routes_frozen_before_2024"])
    if policies["union"] != frozen["routes"]:
        raise ValueError("reconstructed union policy differs from frozen report")
    return model, schema, frozen["routes"], policies


def routed_rows(frame: pd.DataFrame, model, schema: dict,
                routes: dict) -> pd.DataFrame:
    eligible = frame[
        frame.symbol.isin(base.ROBUST_SYMBOLS)
        & frame.risk_pct.between(legacy.RISK_PCT_MIN, legacy.RISK_PCT_MAX,
                                 inclusive="both")].copy()
    scored = legacy.scored(eligible, model, schema)
    scored["period"] = "direct"
    return family.route_rows(scored, ("direct",), routes)


def diagnostics(rows: pd.DataFrame, margin: float) -> dict:
    records = rows.to_dict("records")
    value = money(records, margin)
    signed = rows.signed_price_return.to_numpy(float) if len(rows) else np.array([])
    mean_signed = float(signed.mean()) if len(signed) else 0.0
    rate = float(value["trades_per_calendar_day"])
    literal = int(rows.barbell_outcome.eq("TP_RUNNER").sum()) if len(rows) else 0
    value.update({
        "mean_signed_price_return_pct": round(100 * mean_signed, 6),
        "median_signed_price_return_pct": round(
            100 * float(np.median(signed)), 6) if len(signed) else 0.0,
        "literal_runner_hits": literal,
        "literal_runner_hit_rate_pct": round(100 * literal / len(rows), 4)
            if len(rows) else 0.0,
        "projected_30d_return_pct": round(
            projected_month(mean_signed, rate, margin), 4),
        "required_signed_move_at_3_trades_day_pct": round(
            100 * required_signed_move(margin, TARGET_TRADES_PER_DAY), 6),
        "required_signed_move_at_observed_frequency_pct": round(
            100 * required_signed_move(margin, rate), 6) if rate else None,
    })
    value["economic_30pct_month_gate"] = bool(
        value["projected_30d_return_pct"] >= 30
        and value["trades_per_calendar_day"] >= TARGET_TRADES_PER_DAY)
    return value


def parity(old: pd.DataFrame, new: pd.DataFrame, cutoff: float) -> dict:
    keys = list(barbell.DIRECT_KEYS)
    fields = [*keys, "outcome", "near_outcome", "exit_time", "near_exit_time",
              "risk_pct", "net_r"]
    # The refresh is a fixed-length rolling cache, so a few oldest days leave
    # the left edge when new bars arrive.  Parity is defined on the complete
    # common time interval, never on rows that only one snapshot can contain.
    common_start = max(float(old.entry_time.min()), float(new.entry_time.min()))
    a = old[old.entry_time.between(common_start, cutoff, inclusive="both")][
        fields].sort_values(keys).reset_index(drop=True)
    b = new[new.entry_time.between(common_start, cutoff, inclusive="both")][
        fields].sort_values(keys).reset_index(drop=True)
    joined = a.merge(b, on=keys, suffixes=("_old", "_new"), validate="one_to_one")
    comparisons = [field for field in fields if field not in keys]
    mismatch = 0
    for field in comparisons:
        left, right = joined[f"{field}_old"], joined[f"{field}_new"]
        if pd.api.types.is_numeric_dtype(left):
            mismatch += int((~np.isclose(left, right, rtol=0, atol=1e-12,
                                         equal_nan=True)).sum())
        else:
            mismatch += int((left.fillna("") != right.fillna("")).sum())
    return {"common_start": int(common_start),
            "old_rows": len(a), "new_rows": len(b), "matched_rows": len(joined),
            "value_mismatches": mismatch,
            "passes": bool(len(a) == len(b) == len(joined) and mismatch == 0)}


def main() -> None:
    model, schema, routes, policies = train_frozen()
    old = barbell.join_direct_outcomes(OLD_FAR, OLD_NEAR)
    new = barbell.join_direct_outcomes(NEW_FAR, NEW_NEAR)
    prior_cutoff = float(old.entry_time.max())
    overlap = parity(old, new, prior_cutoff)
    if not overlap["passes"]:
        raise ValueError(f"refreshed raw replay drifted: {overlap}")
    all_rows = routed_rows(new, model, schema, routes)
    fresh_rows = all_rows[all_rows.entry_time > prior_cutoff].copy()
    all_rows.to_pickle(ROWS_OUT)
    fresh_rows.to_pickle(FRESH_ROWS_OUT)
    results = {
        "all_refreshed_xperp": {
            f"margin_{int(100*m)}pct": diagnostics(all_rows, m)
            for m in MARGINS},
        "untouched_after_oct17_raw_cutoff": {
            f"margin_{int(100*m)}pct": diagnostics(fresh_rows, m)
            for m in MARGINS},
    }
    policy_comparison = {}
    for name, policy_routes in policies.items():
        policy_all = routed_rows(new, model, schema, policy_routes)
        policy_fresh = policy_all[policy_all.entry_time > prior_cutoff].copy()
        policy_comparison[name] = {
            "route_count": len(policy_routes),
            "all_refreshed_xperp": {
                f"margin_{int(100*m)}pct": diagnostics(policy_all, m)
                for m in MARGINS},
            "untouched_after_oct17_raw_cutoff": {
                f"margin_{int(100*m)}pct": diagnostics(policy_fresh, m)
                for m in MARGINS},
        }
    fresh3 = results["untouched_after_oct17_raw_cutoff"]["margin_3pct"]
    all3 = results["all_refreshed_xperp"]["margin_3pct"]
    pass_gate = bool(
        len(fresh_rows) >= 30 and fresh3["return_pct"] > 0
        and fresh3["dd_pct"] <= 10 and all3["negative_months"] == 0
        and all3["economic_30pct_month_gate"])
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    exact_path = json.loads(EXACT_PATH_REPORT.read_text(encoding="utf-8"))
    exit_path_valid = exact_path.get("status") != "MATERIAL_ROUTE_INVALIDATED_EXIT_PATH"
    report = {
        "status": ("MATERIAL_ROUTE_INVALIDATED_EXIT_PATH" if not exit_path_valid
                   else "MATERIAL_ROUTE_SHADOW_CANDIDATE" if pass_gate
                   else "MATERIAL_ROUTE_REJECTED_OBJECTIVE"),
        "ready_for_live_orders": False,
        "protocol": {
            "selection": "model and union regime/family routes frozen in the Oct-18 report",
            "refresh_use": "overlap parity only, then untouched entries after the prior raw cutoff",
            "entry": "next 15-minute market open", "orders": "market only",
            "money": "client 3% or 5% current-equity margin; fixed 10x gross",
            "objective": ">=30% projected 30-day return, >=3 trades/day, no negative observed months",
        },
        "prior_raw_cutoff": int(prior_cutoff),
        "prior_raw_cutoff_iso": pd.Timestamp(prior_cutoff, unit="s", tz="UTC").isoformat(),
        "snapshot_end": max(item["end"] for item in manifest["symbols"].values()
                            if item.get("status") == "OK"),
        "overlap_parity": overlap,
        "route_count": len(routes), "results": results,
        "frozen_policy_comparison": policy_comparison,
        "economic_pass": pass_gate and exit_path_valid,
        "exit_path_valid": exit_path_valid,
        "exact_path_report": str(EXACT_PATH_REPORT),
        "sources": {str(path): digest(path) for path in (
            FROZEN_REPORT, OLD_NEAR, OLD_FAR, NEW_NEAR, NEW_FAR, MANIFEST,
            EXACT_PATH_REPORT, Path(__file__))},
        "limitations": [
            "Independent near/far labels overstate returns after the live breakeven stop; the exact path report supersedes these metrics.",
            "The untouched refresh is only a few calendar days.",
            "X-Perp history covers roughly six months, not a full market cycle.",
            "Gross returns exclude fees at user request.",
            "The 30-day value is a diagnostic projection, not a guarantee.",
        ],
    }
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"status": report["status"], "overlap": overlap,
                      "cutoff": report["prior_raw_cutoff_iso"],
                      "fresh_rows": len(fresh_rows), "results": results,
                      "economic_pass": pass_gate}, indent=2), flush=True)


if __name__ == "__main__":
    main()
