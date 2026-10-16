"""Chronological profit-first audit of literal 2.5-4% crypto moves."""
from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

import crypto_5m_regime_router_literal_oct6_audit as portfolio
import crypto_okx_swap_1h_regime_oct7_audit as base
import crypto_profit_first_literal_runner_oct16_audit as literal
from crypto_profit_first_runner_oct16_audit import enrich


ROOT = Path("reports/audit_2026_09_08")
OUT = ROOT / "crypto_profit_first_high_move_oct17.json"
ROWS_OUT = ROOT / "crypto_profit_first_high_move_oct17.pkl"
TARGETS_PCT = (2.5, 3.0, 4.0)
HOLDS = (24, 48, 96)
QUANTILES = (.75, .80, .85, .875, .90, .925, .95, .96, .97, .98)
FIT_YEARS = (2020, 2021)
SELECTION_YEAR = 2022
LATER_YEARS = (2023, 2024, 2025, 2026)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def evaluate(frame: pd.DataFrame, years: tuple[int, ...], threshold: float) -> dict:
    return enrich(portfolio.evaluate(frame, years, threshold))


def selection_ok(value: dict) -> bool:
    return bool(value["return_pct"] > 0 and value["dd_pct"] <= 18
                and value["trades_per_calendar_day"] >= 3)


def score(value: dict) -> float:
    return round(value["equivalent_monthly_return_pct"]
                 - .10 * value["dd_pct"]
                 - .20 * value["negative_months"], 8)


def objective_ok(value: dict) -> bool:
    return bool(value["months"] > 0 and value["dd_pct"] <= 10
                and value["trades_per_calendar_day"] >= 3
                and value["months_ge_30pct"] == value["months"])


def public(item: dict) -> dict:
    return {key: value for key, value in item.items() if key != "_quality"}


def main() -> None:
    candles, sources = base.load()
    features = base.build_features(candles)
    proposals = base.proposals(features)
    candidates = []
    profile_reports = []
    for target_pct, hold in itertools.product(TARGETS_PCT, HOLDS):
        frame = literal.add_outcomes(proposals, candles, target_pct, hold)
        fit = frame.year.isin(FIT_YEARS)
        columns = base.schema(frame.loc[fit])
        estimator = literal.model()
        estimator.fit(base.matrix(frame.loc[fit], columns),
                      frame.loc[fit, "target_hit"].to_numpy(np.int8))
        quality = estimator.predict_proba(base.matrix(frame, columns))[:, 1]
        frame["quality"] = quality
        scores = frame.loc[frame.year.eq(SELECTION_YEAR), "quality"]
        frontier = []
        for quantile in QUANTILES:
            threshold = float(scores.quantile(quantile))
            value = evaluate(frame, (SELECTION_YEAR,), threshold)
            item = {
                "target_pct": target_pct, "hold_bars": hold,
                "selection_quantile": quantile, "threshold": threshold,
                "selection_2022": value, "selection_pass": selection_ok(value),
                "score": score(value), "_quality": quality,
            }
            candidates.append(item)
            frontier.append(public(item))
        best = max(frontier, key=lambda item: item["score"])
        profile_reports.append({
            "target_pct": target_pct, "hold_bars": hold,
            "fit_rows": int(fit.sum()),
            "fit_target_prevalence_pct": round(100 * float(frame.loc[fit, "target_hit"].mean()), 4),
            "best_2022": best,
        })
        print(json.dumps(profile_reports[-1], default=float), flush=True)
    candidates.sort(key=lambda item: (item["selection_pass"], item["score"]),
                    reverse=True)
    frozen = candidates[0] if candidates and candidates[0]["selection_pass"] else None
    later = checks = None
    status = "REJECTED_SELECTION"
    if frozen:
        selected_frame = literal.add_outcomes(
            proposals, candles, frozen["target_pct"], frozen["hold_bars"])
        selected_frame["quality"] = frozen["_quality"]
        later = {str(year): evaluate(selected_frame, (year,), frozen["threshold"])
                 for year in LATER_YEARS}
        all_years = {str(SELECTION_YEAR): frozen["selection_2022"], **later}
        checks = {year: objective_ok(value) for year, value in all_years.items()}
        status = ("HIGH_MOVE_RESEARCH_CANDIDATE"
                  if all(checks.values()) else "REJECTED_OBJECTIVE")
        accepted = portfolio.portfolio_rows(
            selected_frame, (SELECTION_YEAR, *LATER_YEARS), frozen["threshold"])
        pd.DataFrame(accepted).to_pickle(ROWS_OUT)
    report = {
        "status": status, "ready_for_live_orders": False,
        "protocol": {
            "priority": "account return first; win rate diagnostic only",
            "model_fit": list(FIT_YEARS),
            "profile_and_cutoff_selection": SELECTION_YEAR,
            "frozen_later_years": list(LATER_YEARS),
            "targets_pct": list(TARGETS_PCT), "hold_hours": list(HOLDS),
            "stop_pct": 1.0, "entry": "next one-hour market open",
            "orders": "market only",
            "money": "3% current-equity margin, fixed 10x gross",
            "objective": ">=30% account return in every month; >=3 trades/day; DD <=10%",
        },
        "feature_rows": len(features), "proposal_rows": len(proposals),
        "candidate_count": len(candidates), "profiles": profile_reports,
        "selection_top20": [public(item) for item in candidates[:20]],
        "frozen_before_2023": public(frozen) if frozen else None,
        "later_2023_2026": later, "objective_checks": checks,
        "sources": {**sources,
                    str(Path(base.__file__)): digest(Path(base.__file__)),
                    str(Path(literal.__file__)): digest(Path(literal.__file__)),
                    str(Path(__file__)): digest(Path(__file__))},
        "limitations": [
            "The larger target profiles and numeric cutoff are selected on 2022 only.",
            "One-hour target/stop collisions are resolved pessimistically as stops.",
            "Historical OKX global results still require direct EU X-Perp shadow transfer.",
        ],
    }
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"status": status,
                      "frozen": public(frozen) if frozen else None,
                      "later_2023_2026": later,
                      "objective_checks": checks}, indent=2), flush=True)


if __name__ == "__main__":
    main()
