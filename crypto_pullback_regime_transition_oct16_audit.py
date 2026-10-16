"""Causal audit of stale-trend transitions in the frozen OKX 1H pullback family.

The live append showed repeated pullback stops when the frozen 168-hour trend
opposed the most recent 24-hour move.  This audit tests a small predeclared set
of entry-time regime-transition guards.  It selects on 2022, validates on 2023,
and opens later years only after each earlier gate passes.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

import crypto_okx_swap_1h_target_cascade_oct7_audit as cascade
from src.research_profit_gate import economic_gate


ROOT = Path("reports/audit_2026_09_08")
SOURCE = ROOT / "crypto_okx_swap_1h_target_cascade_oct7.pkl"
OUT = ROOT / "crypto_pullback_regime_transition_oct16.json"
SELECTION = (2022,)
VALIDATION = (2023,)
TEST = (2024,)
CONFIRMATION = (2025, 2026)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def enrich(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    direction = out.direction.astype(float)
    out["pullback_entry"] = out.family_pullback.fillna(0).eq(1)
    out["aligned_ret24"] = direction * out.ret24
    out["aligned_btc24"] = direction * out.btc_ret24
    out["aligned_cross_z24"] = direction * out.cross_z24
    out["breadth_alignment"] = out.breadth24.where(
        direction > 0, 1 - out.breadth24)
    out["recent_week_ratio"] = (
        out.aligned_ret24 / out.ret168.abs().clip(lower=.005))
    out["recent_velocity"] = direction * (out.ret24 - out.ret72 / 3)
    return out


def candidate_risks(frame: pd.DataFrame) -> list[tuple[str, pd.Series]]:
    """Return fixed hypotheses; all masks are known before the entry open."""
    symbol_against = frame.aligned_ret24 < 0
    btc_against = frame.aligned_btc24 < 0
    breadth_against = frame.breadth_alignment < .5
    return [
        ("recent_24h_opposes", symbol_against),
        ("recent_24h_opposes_by_0_5pct", frame.aligned_ret24 <= -.005),
        ("recent_24h_opposes_by_1pct", frame.aligned_ret24 <= -.01),
        ("btc_24h_opposes", btc_against),
        ("breadth_majority_opposes", breadth_against),
        ("symbol_and_btc_24h_oppose", symbol_against & btc_against),
        ("symbol_and_breadth_oppose", symbol_against & breadth_against),
        ("btc_and_breadth_oppose", btc_against & breadth_against),
        ("symbol_btc_breadth_oppose",
         symbol_against & btc_against & breadth_against),
        ("cross_section_and_symbol_oppose",
         symbol_against & frame.aligned_cross_z24.le(-.5)),
        ("recent_move_reverses_quarter_of_168h",
         frame.recent_week_ratio <= -.25),
        ("recent_velocity_break",
         symbol_against & frame.recent_velocity.le(-.005)),
    ]


def accepted(frame: pd.DataFrame, years: tuple[int, ...]) -> pd.DataFrame:
    return pd.DataFrame(cascade.portfolio.portfolio_rows(frame, years, .5))


def metrics(frame: pd.DataFrame, years: tuple[int, ...]) -> dict:
    value = cascade.evaluate(frame, years)
    rows = accepted(frame, years)
    returns = rows.signed_return.astype(float).tolist() if len(rows) else []
    value["economic_gate"] = economic_gate(
        returns, float(value.get("trades_per_calendar_day", 0)), market_days=30)
    return value


def flag_stats(frame: pd.DataFrame, risk: pd.Series, year: int) -> dict:
    tagged = frame.copy()
    tagged["transition_risk"] = (
        tagged.pullback_entry & risk.reindex(tagged.index, fill_value=False))
    rows = accepted(tagged, (year,))
    flagged = rows.transition_risk.astype(bool)
    stopped = rows.exit_reason.eq(0)
    baseline_rate = float(stopped.mean()) if len(rows) else 0.0
    flagged_rate = float(stopped[flagged].mean()) if flagged.any() else 0.0
    return {
        "accepted_n": len(rows),
        "flagged_n": int(flagged.sum()),
        "flagged_stops": int((flagged & stopped).sum()),
        "baseline_stop_rate_pct": round(100 * baseline_rate, 4),
        "flagged_stop_rate_pct": round(100 * flagged_rate, 4),
        "stop_rate_lift": round(flagged_rate / baseline_rate, 4)
        if baseline_rate else None,
    }


def dominates(candidate: dict, baseline: dict, minimum_n: int) -> bool:
    return bool(
        candidate["n"] >= minimum_n
        and candidate["trades_per_calendar_day"] >= 2.0
        and candidate["return_pct"] >= baseline["return_pct"]
        and candidate["literal_target_hit_rate"] >= baseline["literal_target_hit_rate"]
        and candidate["dd_pct"] <= baseline["dd_pct"]
        and candidate["negative_months"] <= baseline["negative_months"]
        and candidate["minimum_month_pct"] >= baseline["minimum_month_pct"]
        and candidate["max_loss_streak"] <= baseline["max_loss_streak"])


def objective_pass(value: dict) -> bool:
    return bool(
        75 <= value["literal_target_hit_rate"] <= 95
        and value["trades_per_calendar_day"] >= 3
        and value["negative_months"] == 0
        and value["dd_pct"] <= 5
        and value["economic_gate"]["passes"])


def main() -> None:
    frame = enrich(pd.read_pickle(SOURCE))
    baseline_selection = metrics(frame, SELECTION)
    frontier = []
    eligible = []
    for name, risk in candidate_risks(frame):
        flagged = flag_stats(frame, risk, SELECTION[0])
        blocked = frame.pullback_entry & risk
        candidate = frame.loc[~blocked].copy()
        score = metrics(candidate, SELECTION)
        stable_risk = bool(
            flagged["flagged_n"] >= 30
            and flagged["stop_rate_lift"] is not None
            and flagged["stop_rate_lift"] >= 1.10)
        passed = stable_risk and dominates(score, baseline_selection, 700)
        item = {
            "guard": name,
            "selection_flag_stats": flagged,
            "stable_stop_risk": stable_risk,
            "selection_2022": score,
            "selection_pass": passed,
        }
        frontier.append(item)
        if passed:
            rank = (
                baseline_selection["negative_months"] - score["negative_months"],
                score["minimum_month_pct"], score["return_pct"],
                score["literal_target_hit_rate"], -score["dd_pct"])
            eligible.append((rank, name, candidate, item))

    frontier.sort(key=lambda item: (
        item["selection_pass"], item["stable_stop_risk"],
        item["selection_2022"]["return_pct"],
        -item["selection_2022"]["dd_pct"]), reverse=True)

    frozen = validation = test = confirmation = None
    baseline_validation = baseline_test = None
    validation_passed = test_passed = None
    status = "REJECTED_SELECTION"
    if eligible:
        _, name, candidate, selected = max(eligible, key=lambda item: item[0])
        frozen = {
            "guard": name,
            "eligible_guards": len(eligible),
            "selection_flag_stats": selected["selection_flag_stats"],
            "selection_2022": selected["selection_2022"],
            "selection_baseline_2022": baseline_selection,
        }
        baseline_validation = metrics(frame, VALIDATION)
        validation = metrics(candidate, VALIDATION)
        validation_passed = dominates(validation, baseline_validation, 500)
        status = "REJECTED_VALIDATION"
        if validation_passed:
            baseline_test = metrics(frame, TEST)
            test = metrics(candidate, TEST)
            test_passed = dominates(test, baseline_test, 500)
            status = "REJECTED_TEST"
            if test_passed:
                confirmation = {
                    str(year): metrics(candidate, (year,))
                    for year in CONFIRMATION}
                status = ("PULLBACK_TRANSITION_RESEARCH_CANDIDATE"
                          if all(objective_pass(value)
                                 for value in confirmation.values())
                          else "REJECTED_OBJECTIVE")

    report = {
        "status": status,
        "ready_for_live_orders": False,
        "protocol": {
            "input": "exact frozen OKX hourly target-cascade rows",
            "hypothesis": (
                "the 168h pullback direction becomes stale when closed 24h symbol, "
                "BTC or breadth state reverses before entry"),
            "candidate_count": len(frontier),
            "selection": list(SELECTION),
            "validation": list(VALIDATION),
            "conditional_test": list(TEST),
            "published_confirmation_only": list(CONFIRMATION),
            "entry": "next hourly market open",
            "targets_pct": [.5, 1.0, 1.5, 2.0],
            "stop_pct": 1.0,
            "money": "3% current-equity margin, fixed 10x gross",
            "no_fit_or_target_retuning": True,
        },
        "selection_baseline_2022": baseline_selection,
        "frontier": frontier,
        "frozen_before_validation": frozen,
        "validation_baseline_2023": baseline_validation,
        "validation_2023": validation,
        "validation_pass": validation_passed,
        "test_baseline_2024": baseline_test,
        "test_2024": test,
        "test_pass": test_passed,
        "confirmation_2025_2026": confirmation,
        "sources": {
            str(SOURCE): digest(SOURCE),
            str(Path(cascade.__file__)): digest(Path(cascade.__file__)),
            str(Path(__file__)): digest(Path(__file__)),
        },
        "limitations": [
            "The guard uses closed hourly OHLCV and cross-sectional returns, not historical order-book depth.",
            "The small candidate list is predeclared to limit multiple testing.",
            "A historical pass would still require independent shadow outcomes before activation.",
        ],
    }
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({
        "status": status,
        "eligible": len(eligible),
        "frozen": frozen,
        "validation": validation,
        "test": test,
        "confirmation": confirmation,
        "top_frontier": frontier[:6],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
