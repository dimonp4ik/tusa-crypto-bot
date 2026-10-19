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
EXACT_REPLAY_REPORT = ROOT / "crypto_exact_be_expected_value_oct19.json"
QUANTILES = tuple(float(value) for value in np.arange(.50, .976, .025))
FRACTIONS = (0.0, .10, .20, .30, .40, .50, .60, .70)
LOSS_PAUSES = (1, 2, 3, 4, 5)
DIRECT_KEYS = ("symbol", "entry_time", "direction", "family")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_history(frame: pd.DataFrame) -> pd.DataFrame:
    """Remove overlap between adjacent cache periods without leaking into fit.

    Some six-month source caches overlap by roughly two days.  Keeping the
    later period assignment ensures a row shared by FIT and SELECT belongs to
    SELECT only, so its outcome cannot enter model fitting.
    """
    order = {period: index for index, period in enumerate(base.PERIODS)}
    result = frame.copy()
    result["_period_order"] = result.period.map(order)
    if result["_period_order"].isna().any():
        raise ValueError("unknown historical period during canonicalization")
    result = result.sort_values([*DIRECT_KEYS, "_period_order"])
    duplicate_extra = int(result.duplicated(list(DIRECT_KEYS)).sum())
    result = result.drop_duplicates(list(DIRECT_KEYS), keep="last")
    result = result.drop(columns="_period_order").reset_index(drop=True)
    result.attrs["duplicate_rows_removed"] = duplicate_extra
    return result


def evaluate(scored: pd.DataFrame, periods: tuple[str, ...], threshold: float,
             fraction: float, margin: float, loss_pause: int = 3) -> dict:
    chosen = base.stage(scored, periods)
    chosen = chosen[chosen.model_probability >= threshold].copy()
    chosen = precision.barbell_frame(chosen, fraction)
    rows = runtime_gate(base.standardized(chosen, "precision_profit_first"),
                        loss_pause=loss_pause, hourly_cap=2)
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


def invalidate_with_exact_replay(report: dict, exact_report: dict) -> dict:
    """Retire the optimistic two-label replay after exact candle replay."""
    if exact_report.get("status") == "EXACT_BE_EXPECTED_VALUE_SHADOW_CANDIDATE":
        return report
    legacy_status = str(report.get("status", "UNKNOWN"))
    report["legacy_status_before_exact_replay"] = legacy_status
    report["status"] = "INVALIDATED_EXIT_PATH"
    report["ready_for_live_orders"] = False
    checks = report.setdefault("objective_checks", {})
    checks["exact_runtime_exit_path_validated"] = False
    report["invalidation"] = {
        "reason": (
            "Independent near/far terminal labels overcount 2R wins after the "
            "live full-position stop has already moved to breakeven."
        ),
        "superseding_report": str(EXACT_REPLAY_REPORT),
        "superseding_status": exact_report.get("status"),
        "exact_direct_xperp": exact_report.get("direct_xperp"),
    }
    return report


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
                fraction: float, loss_pause: int) -> list[dict]:
    chosen = scored[scored.model_probability >= threshold].copy()
    chosen = precision.barbell_frame(chosen, fraction)
    return runtime_gate(base.standardized(chosen, "precision_profit_first"),
                        loss_pause=loss_pause, hourly_cap=2)


def main() -> None:
    previous = json.loads(OLD_REPORT.read_text())
    previous_frozen = previous["frozen_before_2024"]
    historical_raw = precision.load_joined()
    historical = canonical_history(historical_raw)
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
    rail_candidates = []
    if frozen:
        threshold = float(frozen["threshold"])
        fraction = float(frozen["near_fraction"])
        for loss_pause in LOSS_PAUSES:
            selection = evaluate(scored, precision.SELECT, threshold, fraction,
                                 .03, loss_pause)
            confirmation = evaluate(scored, precision.CONFIRM, threshold,
                                    fraction, .03, loss_pause)
            if not qualifies(selection, confirmation):
                continue
            rail_candidates.append({
                "loss_pause": loss_pause,
                "selection_2022h1": selection,
                "confirmation_2022h2_2023": confirmation,
                "score": round(selection["return_pct"]
                               - .5 * selection["dd_pct"], 8),
            })
        rail_candidates.sort(key=lambda item: item["score"], reverse=True)
        if rail_candidates:
            rail = rail_candidates[0]
            frozen = {**frozen, "loss_pause": rail["loss_pause"],
                      "rail_selection": rail["selection_2022h1"],
                      "rail_confirmation": rail[
                          "confirmation_2022h2_2023"]}
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
            "loss_pause_candidates": list(LOSS_PAUSES),
            "loss_pause_selection": (
                "maximize 2022h1 return minus half drawdown; require the same "
                "profit, drawdown, frequency and nonnegative-month gates on "
                "2022h1 and 2022h2-2023"),
            "entry": "next 15-minute market open",
            "money": "client 3% or 5% current-equity margin; fixed 10x gross",
            "priority": "account return first; win rate diagnostic only",
        },
        "eligible_development_candidates": len(candidates),
        "top_development_candidates": candidates[:10],
        "eligible_rail_candidates": rail_candidates,
        "frozen_before_2024": frozen,
        "history_integrity": {
            "raw_rows": len(historical_raw),
            "canonical_rows": len(historical),
            "overlap_duplicates_removed": historical.attrs[
                "duplicate_rows_removed"],
            "dedupe_identity": list(DIRECT_KEYS),
            "overlap_assignment": "later chronological period",
        },
    }
    if frozen:
        threshold = float(frozen["threshold"])
        fraction = float(frozen["near_fraction"])
        loss_pause = int(frozen["loss_pause"])
        validation = {
            "margin_3pct": evaluate(scored, precision.VALIDATION,
                                    threshold, fraction, .03, loss_pause),
            "margin_5pct": evaluate(scored, precision.VALIDATION,
                                    threshold, fraction, .05, loss_pause),
        }
        later = {
            "margin_3pct": evaluate(scored, precision.HIDDEN,
                                    threshold, fraction, .03, loss_pause),
            "margin_5pct": evaluate(scored, precision.HIDDEN,
                                    threshold, fraction, .05, loss_pause),
        }
        direct = direct_frame()
        direct_scored = precision.scored(direct, model, schema)
        rows = direct_rows(direct_scored, threshold, fraction, loss_pause)
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
        full_months_3pct = {key: value for key, value in
                            fresh["margin_3pct"]["monthly_return_pct"].items()
                            if key in full_months}
        holdout_complete_months = {
            **{key: validation["margin_5pct"]["monthly_return_pct"][key]
               for key in ("2024-02", "2024-03", "2024-04", "2024-05",
                           "2024-06")},
            **{key: later["margin_5pct"]["monthly_return_pct"][key]
               for key in ("2025-02", "2025-03", "2025-04", "2025-05",
                           "2025-06", "2026-03", "2026-04", "2026-05",
                           "2026-06", "2026-07")},
        }
        objective = {
            "all_complete_direct_months_at_least_30pct": bool(
                full_months and all(value >= 30 for value in full_months.values())),
            "direct_drawdown_at_most_10pct": fresh["margin_5pct"]["dd_pct"] <= 10,
            "direct_at_least_3_trades_per_day": (
                fresh["margin_5pct"]["trades_per_calendar_day"] >= 3),
            "later_drawdown_at_most_10pct": later["margin_5pct"]["dd_pct"] <= 10,
            "all_complete_holdout_months_at_least_30pct_5pct_margin": bool(
                holdout_complete_months and all(
                    value >= 30 for value in holdout_complete_months.values())),
            "all_complete_direct_months_at_least_30pct_3pct_margin": bool(
                full_months_3pct and all(
                    value >= 30 for value in full_months_3pct.values())),
        }
        report.update({
            "status": ("PROFIT_FIRST_5PCT_SHADOW_CANDIDATE"
                       if all((objective[
                           "all_complete_direct_months_at_least_30pct"],
                           objective[
                           "all_complete_holdout_months_at_least_30pct_5pct_margin"],
                           objective["direct_drawdown_at_most_10pct"],
                           objective["later_drawdown_at_most_10pct"],
                           objective["direct_at_least_3_trades_per_day"]))
                       else "PROFIT_FIRST_SHADOW_CANDIDATE_OBJECTIVE_NOT_MET"),
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
            "complete_direct_months_3pct": full_months_3pct,
            "complete_holdout_months_5pct": holdout_complete_months,
        })
    if EXACT_REPLAY_REPORT.exists():
        exact_report = json.loads(EXACT_REPLAY_REPORT.read_text())
        report = invalidate_with_exact_replay(report, exact_report)
    report["sources"] = {
        str(OLD_REPORT): digest(OLD_REPORT),
        str(FRESH_FAR): digest(FRESH_FAR),
        str(FRESH_NEAR): digest(FRESH_NEAR),
        str(Path(__file__)): digest(Path(__file__)),
    }
    if EXACT_REPLAY_REPORT.exists():
        report["sources"][str(EXACT_REPLAY_REPORT)] = digest(EXACT_REPLAY_REPORT)
    report["limitations"] = [
        "This legacy two-label replay is invalidated by the exact candle-path replay and must not be used for strategy selection.",
        "The April and October direct months are partial calendar months.",
        "The same historical model family was previously inspected; 2024 and later are transfer evidence, not pristine data.",
        "At 3% margin May remains below the 30% monthly goal; the complete-month gate passes at 5% only.",
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
