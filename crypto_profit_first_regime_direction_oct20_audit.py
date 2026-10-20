"""Causal BTC-regime direction guard for the material crypto route.

Countertrend entries are optionally suppressed when the already-closed BTC
trend has a sufficient ATR margin.  Scope and margin are selected on
2022-2023, then frozen before 2024-2026 and refreshed X-Perp evaluation.
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
import crypto_profit_first_route_health_oct20_audit as health
from crypto_runtime_shadow_portfolio_oct3_audit import money, runtime_gate


ROOT = Path("reports/audit_2026_09_08")
OUT = ROOT / "crypto_profit_first_regime_direction_oct20.json"
ROWS_OUT = ROOT / "crypto_profit_first_regime_direction_xperp_oct20.pkl"
FRESH_OUT = ROOT / "crypto_profit_first_regime_direction_xperp_oct20_fresh.pkl"
SCOPES = ("all", "trend_families")
MARGINS_ATR = (0.0, 0.25, 0.5, 1.0, math.inf)
TREND_FAMILIES = frozenset(("breakout", "trend_pullback"))
FIT, SELECT, CONFIRM = prior.FIT, prior.SELECT, prior.CONFIRM
VALIDATION, LATER = prior.VALIDATION, prior.LATER


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def is_aligned(regime: str, direction: str) -> bool:
    regime, direction = str(regime), str(direction).upper()
    return ((regime.startswith("bull") and direction == "LONG")
            or (regime.startswith("bear") and direction == "SHORT"))


def apply_guard(frame: pd.DataFrame, scope: str,
                minimum_margin_atr: float) -> pd.DataFrame:
    result = frame.copy()
    aligned = pd.Series([
        is_aligned(regime, direction)
        for regime, direction in zip(result.btc_regime, result.direction)],
        index=result.index)
    margin = result[["btc_fast_slow_atr", "btc_slow_long_atr"]].abs().min(axis=1)
    in_scope = (pd.Series(True, index=result.index) if scope == "all"
                else result.family.isin(TREND_FAMILIES))
    blocked = (~aligned) & in_scope & (margin >= minimum_margin_atr)
    result["regime_direction_aligned"] = aligned
    result["regime_margin_atr"] = margin
    return result[~blocked].copy()


def accepted(frame: pd.DataFrame, periods: tuple[str, ...], scope: str,
             minimum_margin_atr: float) -> pd.DataFrame:
    part = frame[frame.period.isin(periods)].copy()
    guarded = apply_guard(part, scope, minimum_margin_atr)
    return pd.DataFrame(runtime_gate(guarded.to_dict("records"),
                                     loss_pause=3, hourly_cap=2))


def detailed(frame: pd.DataFrame, periods: tuple[str, ...], scope: str,
             minimum_margin_atr: float, margin: float) -> dict:
    def evaluate(labels: tuple[str, ...]) -> dict:
        rows = accepted(frame, labels, scope, minimum_margin_atr)
        return money(rows.to_dict("records"), margin)
    return {"all": evaluate(periods),
            "periods": {period: evaluate((period,)) for period in periods}}


def monthly_equivalent(value: dict) -> float:
    months = max(1, len(value["all"].get("monthly_return_pct", {})))
    factor = 1 + value["all"]["return_pct"] / 100
    return 100 * math.expm1(math.log(factor) / months) if factor > 0 else -100


def development_ok(selection: dict, confirmation: dict) -> bool:
    a, b = selection["all"], confirmation["all"]
    return bool(a["return_pct"] > 0 and b["return_pct"] > 0
                and a["dd_pct"] <= 15 and b["dd_pct"] <= 15
                and a["trades_per_calendar_day"] >= 3
                and b["trades_per_calendar_day"] >= 2
                and all(item["return_pct"] > 0
                        for item in confirmation["periods"].values()))


def score(selection: dict, confirmation: dict) -> float:
    monthly = min(monthly_equivalent(selection), monthly_equivalent(confirmation))
    drawdown = max(selection["all"]["dd_pct"], confirmation["all"]["dd_pct"])
    negative = selection["all"]["negative_months"] + confirmation["all"]["negative_months"]
    return round(monthly - .25 * drawdown - .25 * negative, 8)


def main() -> None:
    model, schema, routes, _ = refresh.train_frozen()
    history = target.load_joined(family.TARGET_R)
    historical_scored = legacy.scored(history, model, schema)
    historical_candidates = health.candidates(historical_scored, routes)
    choices = []
    for scope in SCOPES:
        for minimum_margin_atr in MARGINS_ATR:
            # infinity is the unfiltered baseline; keep only one copy.
            if math.isinf(minimum_margin_atr) and scope != "all":
                continue
            selection = detailed(historical_candidates, SELECT, scope,
                                 minimum_margin_atr, .05)
            confirmation = detailed(historical_candidates, CONFIRM, scope,
                                    minimum_margin_atr, .05)
            if development_ok(selection, confirmation):
                choices.append({
                    "scope": scope,
                    "minimum_margin_atr": (None if math.isinf(minimum_margin_atr)
                                           else minimum_margin_atr),
                    "selection_2022h1": selection,
                    "confirmation_2022h2_2023": confirmation,
                    "score": score(selection, confirmation),
                })
    choices.sort(key=lambda item: item["score"], reverse=True)
    frozen = choices[0] if choices else None
    if frozen is None:
        raise ValueError("no stable regime-direction policy")
    scope = frozen["scope"]
    threshold = math.inf if frozen["minimum_margin_atr"] is None \
        else float(frozen["minimum_margin_atr"])
    validation = detailed(historical_candidates, VALIDATION, scope,
                          threshold, .05)
    later = detailed(historical_candidates, LATER, scope, threshold, .05)

    old = prior.join_direct_outcomes(refresh.OLD_FAR, refresh.OLD_NEAR)
    new = prior.join_direct_outcomes(refresh.NEW_FAR, refresh.NEW_NEAR)
    cutoff = float(old.entry_time.max())
    eligible = new[
        new.symbol.isin(base.ROBUST_SYMBOLS)
        & new.risk_pct.between(legacy.RISK_PCT_MIN, legacy.RISK_PCT_MAX,
                               inclusive="both")].copy()
    direct_scored = legacy.scored(eligible, model, schema)
    direct_scored["period"] = "direct"
    direct_candidates = health.candidates(direct_scored, routes)
    all_rows = accepted(direct_candidates, ("direct",), scope, threshold)
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
        "status": ("REGIME_DIRECTION_SHADOW_CANDIDATE" if pass_gate
                   else "REGIME_DIRECTION_REJECTED_OBJECTIVE"),
        "ready_for_live_orders": False,
        "protocol": {
            "signal_policy": "frozen Oct-18 union regime/family material route",
            "guard": "countertrend versus already-closed BTC regime and ATR margin",
            "selection": "2022-2023 only at fixed 5% margin",
            "scopes": list(SCOPES),
            "margin_thresholds_atr": [None if math.isinf(x) else x for x in MARGINS_ATR],
            "orders": "market only", "leverage": 10,
        },
        "development_frontier": choices,
        "frozen_before_2024": {key: frozen[key] for key in (
            "scope", "minimum_margin_atr", "score")},
        "validation_2024": validation, "later_2025_2026": later,
        "results": results, "economic_pass": pass_gate,
        "sources": {str(path): digest(path) for path in (
            refresh.FROZEN_REPORT, refresh.NEW_NEAR, refresh.NEW_FAR,
            Path(__file__))},
        "limitations": [
            "The untouched refresh is only a few calendar days.",
            "BTC regime alignment is a causal proxy, not order-book liquidity.",
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
