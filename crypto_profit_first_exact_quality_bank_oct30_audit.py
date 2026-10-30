"""Exact audit of signal-quality-dependent TP1 banking.

The entry route and exact 0.5R -> next-bar breakeven -> 2R sequence remain
frozen.  Only the amount banked at 0.5R changes: lower-probability entries may
bank a defensive fraction while all other entries keep the frozen 5% bank.
The policy grid is selected on 2022-2023 before 2024+, and the newest exact
X-Perp replay, are inspected.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import pandas as pd

import crypto_profit_first_exact_health_router_oct30_audit as health
import crypto_profit_first_exact_oct30_2030_forward as latest
import crypto_profit_first_material_target_oct18_audit as frozen
import crypto_profit_first_runner_protection_oct30_audit as exact
from crypto_runtime_shadow_portfolio_oct3_audit import money


ROOT = Path("reports/audit_2026_09_08")
OUT = ROOT / "crypto_profit_first_exact_quality_bank_oct30.json"
HISTORICAL_CACHE = ROOT / "_tmp_exact_historical_rows.pkl"
DIRECT_ROWS = ROOT / "crypto_profit_first_exact_oct30_2030.pkl"
THRESHOLDS = (.335, .345, .355, .370, .400)
DEFENSIVE_FRACTIONS = (.10, .25, .50, .75, 1.0)
BASE_FRACTION = .05
DIRECT_CUTOFF = int(pd.Timestamp(json.loads(
    latest.SOURCE_REPORT.read_text(encoding="utf-8"))
    ["fresh_direct_coverage"]["last_entry"]).timestamp())


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def historical_rows() -> pd.DataFrame:
    if HISTORICAL_CACHE.exists():
        raw = pd.read_pickle(HISTORICAL_CACHE)
    else:
        model, schema = health.fit_model()
        entry_threshold = float(json.loads(
            latest.SOURCE_REPORT.read_text(encoding="utf-8"))
            ["frozen_before_2024"]["threshold"])
        raw, _ = exact.prepare_historical(model, schema, entry_threshold)
    current = health.current_outcomes(raw)
    return pd.DataFrame(health.accepted(current, "quality_bank_baseline"))


def apply_policy(frame: pd.DataFrame, *, threshold: float,
                 defensive_fraction: float, direct: bool = False) -> pd.DataFrame:
    result = frame.copy()
    fraction = pd.Series(BASE_FRACTION, index=result.index, dtype=float)
    fraction.loc[result.model_probability.lt(threshold)] = defensive_fraction
    if direct:
        runner = result.sequential_runner_r.astype(float)
        banked = result.sequential_banked.astype(bool)
    else:
        prefix = exact.policy_key(.5, 0.0)
        runner = result[f"{prefix}_runner_r"].astype(float)
        banked = result[f"{prefix}_banked"].astype(bool)
    result["first_fraction"] = fraction
    result["net_r"] = runner.where(
        ~banked, fraction * .5 + (1.0 - fraction) * runner)
    result["signed_price_return"] = result.net_r * result.risk_pct
    return result


def metric(frame: pd.DataFrame, periods: tuple[str, ...]) -> dict:
    return money(frame[frame.period.isin(periods)].to_dict("records"), .03)


def compact(value: dict) -> dict:
    keys = ("n", "wr", "return_pct", "dd_pct", "max_loss_streak",
            "negative_months", "trades_per_calendar_day",
            "monthly_return_pct")
    return {key: value[key] for key in keys}


def development_pass(candidate: dict, baseline: dict) -> bool:
    return bool(
        candidate["return_pct"] >= baseline["return_pct"]
        and candidate["dd_pct"] <= 1.05 * baseline["dd_pct"]
        and candidate["negative_months"] <= baseline["negative_months"]
        and candidate["max_loss_streak"] <= baseline["max_loss_streak"])


def dominates(candidate: dict, baseline: dict) -> bool:
    return bool(
        candidate["return_pct"] >= baseline["return_pct"]
        and candidate["dd_pct"] <= baseline["dd_pct"]
        and candidate["negative_months"] <= baseline["negative_months"]
        and candidate["max_loss_streak"] <= baseline["max_loss_streak"])


def score(selection: dict, confirmation: dict,
          base_selection: dict, base_confirmation: dict) -> float:
    return round(min(
        selection["return_pct"] / max(base_selection["return_pct"], 1e-9),
        confirmation["return_pct"] / max(base_confirmation["return_pct"], 1e-9),
    ) + .10 * min(
        base_selection["dd_pct"] / max(selection["dd_pct"], 1e-9),
        base_confirmation["dd_pct"] / max(confirmation["dd_pct"], 1e-9),
    ), 8)


def direct_metric(frame: pd.DataFrame) -> dict:
    october = frame[frame.entry_time.ge(pd.Timestamp(
        "2026-10-01", tz="UTC").timestamp())]
    unseen = frame[frame.entry_time.gt(DIRECT_CUTOFF)]
    return {
        "october": latest.metrics(october, .03),
        "after_oct16": latest.metrics(unseen, .03),
    }


def main() -> None:
    historical = historical_rows()
    base_selection = metric(historical, frozen.SELECT)
    base_confirmation = metric(historical, frozen.CONFIRM)
    frontier = []
    eligible = []
    for threshold in THRESHOLDS:
        for fraction in DEFENSIVE_FRACTIONS:
            candidate = apply_policy(
                historical, threshold=threshold,
                defensive_fraction=fraction)
            selection = metric(candidate, frozen.SELECT)
            confirmation = metric(candidate, frozen.CONFIRM)
            passed = (
                development_pass(selection, base_selection)
                and development_pass(confirmation, base_confirmation))
            item = {
                "probability_below": threshold,
                "defensive_first_fraction": fraction,
                "selection_2022h1": compact(selection),
                "confirmation_2022h2_2023": compact(confirmation),
                "development_pass": passed,
                "score": score(selection, confirmation,
                               base_selection, base_confirmation),
            }
            frontier.append(item)
            if passed:
                eligible.append(item)
    frontier.sort(key=lambda item: (item["development_pass"], item["score"]),
                  reverse=True)
    eligible.sort(key=lambda item: item["score"], reverse=True)
    selected = eligible[0] if eligible else None

    base_validation = metric(historical, frozen.VALIDATION)
    base_later = metric(historical, frozen.LATER)
    direct = pd.read_pickle(DIRECT_ROWS)
    base_direct = direct_metric(direct)
    validation = later = candidate_direct = None
    historical_pass = fresh_pass = False
    if selected:
        candidate = apply_policy(
            historical,
            threshold=float(selected["probability_below"]),
            defensive_fraction=float(selected["defensive_first_fraction"]))
        validation = metric(candidate, frozen.VALIDATION)
        later = metric(candidate, frozen.LATER)
        historical_pass = (
            dominates(validation, base_validation)
            and dominates(later, base_later))
        candidate_current = apply_policy(
            direct,
            threshold=float(selected["probability_below"]),
            defensive_fraction=float(selected["defensive_first_fraction"]),
            direct=True)
        candidate_direct = direct_metric(candidate_current)
        fresh_pass = bool(
            candidate_direct["october"]["return_pct"]
                >= base_direct["october"]["return_pct"]
            and candidate_direct["after_oct16"]["return_pct"]
                >= base_direct["after_oct16"]["return_pct"]
            and candidate_direct["after_oct16"]["dd_pct"]
                <= base_direct["after_oct16"]["dd_pct"])

    robust = bool(selected and historical_pass and fresh_pass)
    report = {
        "status": ("QUALITY_BANK_SHADOW_CHALLENGER_AWAIT_FORWARD" if robust
                   else "REJECTED_FRESH_EXACT_FORWARD" if selected
                   else "REJECTED_DEVELOPMENT"),
        "ready_for_live_orders": False,
        "orders_possible": False,
        "runtime_change_allowed": False,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "protocol": {
            "entry_and_exit": "frozen exact 0.5R -> next-bar BE -> 2R",
            "baseline_first_fraction": BASE_FRACTION,
            "probability_thresholds": list(THRESHOLDS),
            "defensive_first_fractions": list(DEFENSIVE_FRACTIONS),
            "selection": [*frozen.SELECT, *frozen.CONFIRM],
            "holdouts": [*frozen.VALIDATION, *frozen.LATER,
                         "direct through 2026-10-30"],
            "orders": "market only",
            "position_fraction": "client setting",
        },
        "candidate_count": len(frontier),
        "development_pass_count": len(eligible),
        "baseline": {
            "selection_2022h1": compact(base_selection),
            "confirmation_2022h2_2023": compact(base_confirmation),
            "validation_2024": compact(base_validation),
            "later_2025_2026": compact(base_later),
            "direct_xperp": base_direct,
        },
        "selected_before_holdouts": selected,
        "validation_2024": compact(validation) if validation else None,
        "later_2025_2026": compact(later) if later else None,
        "direct_xperp": candidate_direct,
        "historical_holdout_pass": historical_pass,
        "fresh_exact_forward_pass": fresh_pass,
        "robust_improvement": robust,
        "development_frontier": frontier,
        "sources": {str(path): digest(path) for path in (
            health.OUT, latest.OUT, DIRECT_ROWS, Path(__file__).resolve())},
        "limitations": [
            "The chosen quality cutoff failed the newest exact X-Perp replay.",
            "A higher profitable-exit rate can still reduce compounded return.",
            "Gross returns exclude fees at user request.",
        ],
    }
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({
        "status": report["status"],
        "selected": selected,
        "validation": report["validation_2024"],
        "later": report["later_2025_2026"],
        "direct": candidate_direct,
        "base_direct": base_direct,
        "robust": robust,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
