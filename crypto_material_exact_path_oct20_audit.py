"""Exact candle-path verification of the frozen 0.5R/2R material route.

The older material reports joined independent 0.5R and fixed-stop 2R labels.
This verifier replays one path: original stop until +0.5R, then the runner
stop moves to entry from the next candle.  A route-specific fraction is taken
at +0.5R; the remainder can only reach 2R before the breakeven stop.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

import crypto_05_2r_precision_barbell_oct5_audit as legacy
import crypto_high_target_tree_oct5_audit as base
import crypto_precision_be_runner_oct18_audit as exact
import crypto_profit_first_material_barbell_oct18_audit as prior
import crypto_profit_first_material_regime_family_oct18_audit as family
import crypto_profit_first_material_target_oct18_audit as target
import crypto_profit_first_material_xperp_oct20_audit as refresh
from crypto_runtime_shadow_portfolio_oct3_audit import money, runtime_gate


ROOT = Path("reports/audit_2026_09_08")
OUT = ROOT / "crypto_material_exact_path_oct20.json"
ROWS_OUT = ROOT / "crypto_material_exact_path_xperp_oct20.pkl"
FRESH_OUT = ROOT / "crypto_material_exact_path_xperp_oct20_fresh.pkl"
ACTIVATION_R = 0.5
RUNNER_R = 2.0


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def selected_candidates(scored: pd.DataFrame, routes: dict) -> pd.DataFrame:
    parts = []
    for route in routes.values():
        part = scored[
            scored.btc_regime.eq(route["btc_regime"])
            & scored.family.eq(route["family"])
            & (scored.model_probability >= float(route["threshold"]))].copy()
        if len(part):
            part["near_fraction"] = float(route["near_fraction"])
            parts.append(part)
    if not parts:
        return scored.iloc[0:0].copy()
    return pd.concat(parts, ignore_index=True)


def partial_net_r(runner_outcome: str, runner_net_r: float,
                  near_fraction: float) -> float:
    if runner_outcome == "TP2":
        return near_fraction * ACTIVATION_R + (1 - near_fraction) * RUNNER_R
    if runner_outcome == "BE":
        return near_fraction * ACTIVATION_R
    return runner_net_r


def exact_candidates(selected: pd.DataFrame, *, direct: bool) -> pd.DataFrame:
    replayed = exact.attach_runtime_runner(
        selected, direct=direct, activation_r=ACTIVATION_R,
        runner_r=RUNNER_R)
    replayed["runner_net_r"] = replayed.net_r.astype(float)
    replayed["net_r"] = [
        partial_net_r(str(outcome), float(value), float(fraction))
        for outcome, value, fraction in zip(
            replayed.runner_outcome, replayed.runner_net_r,
            replayed.near_fraction)]
    replayed["barbell_outcome"] = replayed.runner_outcome.map(
        {"TP2": "TP_RUNNER", "BE": "TP_BE"}).fillna(
            replayed.runner_outcome)
    return replayed


def accepted(frame: pd.DataFrame, periods: tuple[str, ...]) -> pd.DataFrame:
    chosen = base.stage(frame, periods)
    rows = base.standardized(chosen, "material_exact_path")
    for row in rows:
        row["outcome"] = "SL" if float(row["net_r"]) < 0 else "TP"
    return pd.DataFrame(runtime_gate(rows, loss_pause=3, hourly_cap=2))


def metrics(rows: pd.DataFrame, margin: float) -> dict:
    value = refresh.diagnostics(rows, margin)
    count = len(rows)
    value.update({
        "tp2_rate_pct": round(100 * rows.runner_outcome.eq("TP2").sum() / count, 4)
            if count else 0.0,
        "breakeven_runner_rate_pct": round(
            100 * rows.runner_outcome.eq("BE").sum() / count, 4)
            if count else 0.0,
        "non_loss_rate_pct": round(100 * rows.net_r.ge(0).sum() / count, 4)
            if count else 0.0,
    })
    return value


def main() -> None:
    model, schema, routes, _ = refresh.train_frozen()

    historical = target.load_joined(family.TARGET_R)
    historical_scored = legacy.scored(historical, model, schema)
    historical_selected = selected_candidates(historical_scored, routes)
    historical_exact = exact_candidates(historical_selected, direct=False)
    holdouts = {}
    for name, periods in (("validation_2024", prior.VALIDATION),
                          ("later_2025_2026", prior.LATER)):
        rows = accepted(historical_exact, periods)
        holdouts[name] = {
            f"margin_{int(100*m)}pct": metrics(rows, m)
            for m in refresh.MARGINS}

    old = prior.join_direct_outcomes(refresh.OLD_FAR, refresh.OLD_NEAR)
    new = prior.join_direct_outcomes(refresh.NEW_FAR, refresh.NEW_NEAR)
    cutoff = float(old.entry_time.max())
    eligible = new[
        new.symbol.isin(base.ROBUST_SYMBOLS)
        & new.risk_pct.between(legacy.RISK_PCT_MIN, legacy.RISK_PCT_MAX,
                               inclusive="both")].copy()
    direct_scored = legacy.scored(eligible, model, schema)
    direct_scored["period"] = "direct"
    direct_selected = selected_candidates(direct_scored, routes)
    original_cache = exact.DIRECT_CACHE
    exact.DIRECT_CACHE = ROOT / "xperp_15m_oct20_material"
    try:
        direct_exact = exact_candidates(direct_selected, direct=True)
    finally:
        exact.DIRECT_CACHE = original_cache
    all_rows = accepted(direct_exact, ("direct",))
    fresh_rows = all_rows[all_rows.entry_time > cutoff].copy()
    all_rows.to_pickle(ROWS_OUT)
    fresh_rows.to_pickle(FRESH_OUT)
    direct_results = {
        "all_refreshed_xperp": {
            f"margin_{int(100*m)}pct": metrics(all_rows, m)
            for m in refresh.MARGINS},
        "untouched_after_oct17_raw_cutoff": {
            f"margin_{int(100*m)}pct": metrics(fresh_rows, m)
            for m in refresh.MARGINS},
    }
    exact3 = direct_results["all_refreshed_xperp"]["margin_3pct"]
    original = json.loads(refresh.OUT.read_text(encoding="utf-8"))[
        "results"]["all_refreshed_xperp"]["margin_3pct"]
    objective = bool(
        exact3["negative_months"] == 0 and exact3["dd_pct"] <= 10
        and exact3["economic_30pct_month_gate"]
        and direct_results["untouched_after_oct17_raw_cutoff"][
            "margin_3pct"]["n"] >= 30)
    report = {
        "status": ("MATERIAL_EXACT_PATH_SHADOW_CANDIDATE" if objective
                   else "MATERIAL_ROUTE_INVALIDATED_EXIT_PATH"),
        "ready_for_live_orders": False,
        "protocol": {
            "signal_policy": "frozen Oct-18 union regime/family material route",
            "entry": "next 15-minute market open",
            "exit": ("route fraction at +0.5R; remaining stop moves to entry "
                     "from next candle; runner target +2R"),
            "collision": "original stop first on activation bar; breakeven stop first afterwards",
            "orders": "market only", "money": "client 3% or 5%; fixed 10x",
        },
        "selected_candidate_rows": {
            "historical": len(historical_selected),
            "direct": len(direct_selected)},
        "holdouts": holdouts,
        "direct": direct_results,
        "comparison_to_independent_label_replay_3pct": {
            "optimistic_return_pct": original["return_pct"],
            "exact_return_pct": exact3["return_pct"],
            "optimistic_dd_pct": original["dd_pct"],
            "exact_dd_pct": exact3["dd_pct"],
        },
        "objective_pass": objective,
        "sources": {str(path): digest(path) for path in (
            refresh.FROZEN_REPORT, refresh.NEW_NEAR, refresh.NEW_FAR,
            refresh.MANIFEST, Path(__file__))},
        "limitations": [
            "This verifies the frozen material policy; it does not refit on exact outcomes.",
            "The untouched refresh is only a few calendar days.",
            "Gross returns exclude fees at user request.",
        ],
    }
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"status": report["status"],
                      "holdouts": holdouts, "direct": direct_results,
                      "comparison": report[
                          "comparison_to_independent_label_replay_3pct"],
                      "objective_pass": objective}, indent=2), flush=True)


if __name__ == "__main__":
    main()
