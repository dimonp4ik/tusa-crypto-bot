"""Profit-first replay of the causal 0.25R/2R precision strategy.

The classifier family is inherited from the older precision audit and remains
fit on 2020-2021 only.  Threshold and exit fraction are chosen on 2022-2023,
then frozen before replaying 2024, 2025-2026, and direct X-Perp history.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

import crypto_2r_precision_barbell_oct5_audit as precision
import crypto_high_target_tree_oct5_audit as base
from crypto_runtime_shadow_portfolio_oct3_audit import money, runtime_gate


ROOT = Path("reports/audit_2026_09_08")
OUT = ROOT / "crypto_precision_profit_first_oct18.json"
ROWS_OUT = ROOT / "crypto_precision_profit_first_xperp_oct17.pkl"
OLD_REPORT = ROOT / "crypto_2r_precision_barbell_oct5.json"
FRESH_FAR = ROOT / "crypto_xperp_raw_router_audit_2_0_oct17_material_raw_raw_families.csv"
FRESH_NEAR = ROOT / "crypto_xperp_raw_router_audit_0_25_oct17_material_raw_raw_families.csv"
QUANTILES = tuple(float(value) for value in np.arange(.50, .976, .025))
FRACTIONS = (0.0, .10, .20, .30, .40, .50, .60, .70)
DIRECT_KEYS = ("symbol", "entry_time", "direction", "family")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def evaluate(scored: pd.DataFrame, periods: tuple[str, ...], threshold: float,
             fraction: float, margin: float) -> dict:
    chosen = base.stage(scored, periods)
    chosen = chosen[chosen.model_probability >= threshold].copy()
    chosen = precision.barbell_frame(chosen, fraction)
    rows = runtime_gate(base.standardized(chosen, "precision_profit_first"))
    return money(rows, margin)


def qualifies(selection: dict, confirmation: dict) -> bool:
    return bool(
        selection["n"] >= 100 and confirmation["n"] >= 100
        and selection["trades_per_calendar_day"] >= 3
        and confirmation["trades_per_calendar_day"] >= 3
        and selection["return_pct"] > 0 and confirmation["return_pct"] > 0
        and selection["dd_pct"] <= 10 and confirmation["dd_pct"] <= 10
        and selection["negative_months"] <= 1
        and confirmation["negative_months"] <= 2
    )


def development_score(selection: dict, confirmation: dict) -> float:
    return round(
        selection["return_pct"] + 2 * confirmation["return_pct"]
        - .5 * selection["dd_pct"] - confirmation["dd_pct"], 8)


def direct_frame() -> pd.DataFrame:
    far = pd.read_csv(FRESH_FAR)
    near = pd.read_csv(FRESH_NEAR)[[*DIRECT_KEYS, "outcome", "exit_time"]]
    near = near.rename(columns={
        "outcome": "near_outcome", "exit_time": "near_exit_time"})
    if far.duplicated(list(DIRECT_KEYS)).any() or near.duplicated(list(DIRECT_KEYS)).any():
        raise ValueError("direct replay has duplicate signal identities")
    joined = far.merge(near, on=list(DIRECT_KEYS), how="inner",
                       validate="one_to_one")
    if len(joined) != len(far) or len(joined) != len(near):
        raise ValueError("0.25R and 2R direct replays do not have exact parity")
    return joined[joined.symbol.isin(base.ROBUST_SYMBOLS)].copy()


def direct_rows(scored: pd.DataFrame, threshold: float,
                fraction: float) -> list[dict]:
    chosen = scored[scored.model_probability >= threshold].copy()
    chosen = precision.barbell_frame(chosen, fraction)
    return runtime_gate(base.standardized(chosen, "precision_profit_first"))


def main() -> None:
    previous = json.loads(OLD_REPORT.read_text())
    previous_frozen = previous["frozen_before_2024"]
    historical = precision.load_joined()
    fit = base.stage(historical, precision.FIT)
    schema = base.design(fit)
    model = HistGradientBoostingClassifier(
        learning_rate=float(previous_frozen["learning_rate"]), max_iter=150,
        max_leaf_nodes=int(previous_frozen["max_leaf_nodes"]),
        min_samples_leaf=int(previous_frozen["min_samples_leaf"]),
        l2_regularization=1.0, early_stopping=False,
        random_state=20260908)
    model.fit(base.matrix(fit, schema), fit.outcome.eq("TP2").astype(int))
    scored = precision.scored(historical, model, schema)
    selection_scores = base.stage(scored, precision.SELECT).model_probability

    candidates = []
    for quantile in QUANTILES:
        threshold = float(selection_scores.quantile(quantile))
        for fraction in FRACTIONS:
            selection = evaluate(
                scored, precision.SELECT, threshold, fraction, .05)
            confirmation = evaluate(
                scored, precision.CONFIRM, threshold, fraction, .05)
            if not qualifies(selection, confirmation):
                continue
            candidates.append({
                "quantile": quantile, "threshold": threshold,
                "near_fraction": fraction, "selection_2022h1": selection,
                "confirmation_2022h2_2023": confirmation,
                "score": development_score(selection, confirmation),
            })
    candidates.sort(key=lambda item: item["score"], reverse=True)
    frozen = candidates[0] if candidates else None
    report = {
        "status": "REJECTED_DEVELOPMENT",
        "ready_for_live_orders": False,
        "protocol": {
            "model_fit": list(precision.FIT),
            "profit_policy_development": [*precision.SELECT, *precision.CONFIRM],
            "first_holdout": list(precision.VALIDATION),
            "later_holdout": list(precision.HIDDEN),
            "direct_reveal": "X-Perp 2026-04-16 through 2026-10-16",
            "fixed_model_family": {
                key: previous_frozen[key] for key in (
                    "max_leaf_nodes", "min_samples_leaf", "learning_rate")
            },
            "threshold_quantiles": list(QUANTILES),
            "near_target_r": .25, "runner_target_r": 2.0,
            "near_fractions": list(FRACTIONS),
            "entry": "next 15-minute market open",
            "money": "client 3% or 5% current-equity margin; fixed 10x gross",
            "priority": "account return first; win rate diagnostic only",
        },
        "eligible_development_candidates": len(candidates),
        "top_development_candidates": candidates[:10],
        "frozen_before_2024": frozen,
    }
    if frozen:
        threshold = float(frozen["threshold"])
        fraction = float(frozen["near_fraction"])
        validation = {
            "margin_3pct": evaluate(scored, precision.VALIDATION,
                                    threshold, fraction, .03),
            "margin_5pct": evaluate(scored, precision.VALIDATION,
                                    threshold, fraction, .05),
        }
        later = {
            "margin_3pct": evaluate(scored, precision.HIDDEN,
                                    threshold, fraction, .03),
            "margin_5pct": evaluate(scored, precision.HIDDEN,
                                    threshold, fraction, .05),
        }
        direct = direct_frame()
        direct_scored = precision.scored(direct, model, schema)
        rows = direct_rows(direct_scored, threshold, fraction)
        pd.DataFrame(rows).to_pickle(ROWS_OUT)
        old_end = float(pd.read_csv(base.direct_path(2.0)).entry_time.max())
        forward_rows = [row for row in rows
                        if float(row["entry_time"]) > old_end]
        accepted = pd.DataFrame(rows)
        target_move = 2 * pd.to_numeric(accepted.risk_pct, errors="coerce")
        first = pd.to_datetime(direct.entry_time.min(), unit="s", utc=True)
        last = pd.to_datetime(direct.entry_time.max(), unit="s", utc=True)
        fresh = {
            "margin_3pct": money(rows, .03),
            "margin_5pct": money(rows, .05),
        }
        forward = {
            "margin_3pct": money(forward_rows, .03),
            "margin_5pct": money(forward_rows, .05),
        }
        full_months = {key: value for key, value in
                       fresh["margin_5pct"]["monthly_return_pct"].items()
                       if key in ("2026-05", "2026-06", "2026-07",
                                  "2026-08", "2026-09")}
        objective = {
            "all_complete_direct_months_at_least_30pct": bool(
                full_months and all(value >= 30 for value in full_months.values())),
            "direct_drawdown_at_most_10pct": fresh["margin_5pct"]["dd_pct"] <= 10,
            "direct_at_least_3_trades_per_day": (
                fresh["margin_5pct"]["trades_per_calendar_day"] >= 3),
            "later_drawdown_at_most_10pct": later["margin_5pct"]["dd_pct"] <= 10,
        }
        report.update({
            "status": ("PROFIT_FIRST_SHADOW_CANDIDATE_OBJECTIVE_NOT_MET"
                       if all((objective["direct_drawdown_at_most_10pct"],
                               objective["direct_at_least_3_trades_per_day"]))
                       else "REJECTED_RISK"),
            "validation_2024": validation,
            "later_2025_2026": later,
            "fresh_direct_xperp": fresh,
            "untouched_forward_after_oct2": forward,
            "direct_coverage": {
                "first_entry": first.isoformat(), "last_entry": last.isoformat(),
                "raw_joined_rows": len(direct), "accepted": len(rows),
            },
            "price_geometry": {
                "runner_target_price_move_quantiles_pct": {
                    str(q): round(100 * float(target_move.quantile(q)), 4)
                    for q in (0.0, .10, .25, .50, .75, .90, 1.0)
                },
                "accepted_runner_targets_at_least_0_5pct": round(
                    float((target_move >= .005).mean()), 6),
                "accepted_runner_targets_at_most_2pct": round(
                    float((target_move <= .02).mean()), 6),
            },
            "objective_checks": objective,
            "complete_direct_months_5pct": full_months,
        })
    report["sources"] = {
        str(OLD_REPORT): digest(OLD_REPORT),
        str(FRESH_FAR): digest(FRESH_FAR),
        str(FRESH_NEAR): digest(FRESH_NEAR),
        str(Path(__file__)): digest(Path(__file__)),
    }
    report["limitations"] = [
        "The April and October direct months are partial calendar months.",
        "The same historical model family was previously inspected; 2024 and later are transfer evidence, not pristine data.",
        "The result is a shadow candidate because May remains below the 30% monthly goal.",
        "Large total returns are theoretical compounding results, not a live-profit guarantee.",
        "Gross returns exclude fees at user request.",
    ]
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({
        "status": report["status"], "frozen": frozen,
        "validation": report.get("validation_2024"),
        "later": report.get("later_2025_2026"),
        "fresh": report.get("fresh_direct_xperp"),
        "forward": report.get("untouched_forward_after_oct2"),
        "geometry": report.get("price_geometry"),
        "objective": report.get("objective_checks"),
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
