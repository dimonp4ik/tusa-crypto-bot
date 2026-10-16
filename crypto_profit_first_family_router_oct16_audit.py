"""Freeze a stable semantic-family subset for the literal two-percent runner.

The parent two-percent target, 24-hour hold, model and probability cutoff were
chosen on 2022.  This audit uses 2022-2023 only to choose which of the seven
pre-entry semantic families may trade, then opens 2024 and finally 2025-2026.
Family overlap is intentional: a proposal survives when any enabled semantic
setup independently describes it.
"""
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
OUT = ROOT / "crypto_profit_first_family_router_oct16.json"
ROWS_OUT = ROOT / "crypto_profit_first_family_router_oct16.pkl"
PARENT_REPORT = ROOT / "crypto_profit_first_literal_runner_oct16.json"
SELECTION_YEARS = (2022, 2023)
VALIDATION_YEAR = 2024
REVEAL_YEARS = (2025, 2026)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def family_mask(frame: pd.DataFrame, enabled: tuple[str, ...]) -> pd.Series:
    columns = [f"family_{name}" for name in enabled]
    if not columns:
        return pd.Series(False, index=frame.index)
    return frame[columns].fillna(0).eq(1).any(axis=1)


def evaluate(frame: pd.DataFrame, years: tuple[int, ...], threshold: float,
             enabled: tuple[str, ...]) -> dict:
    allowed = frame[family_mask(frame, enabled)]
    return enrich(portfolio.evaluate(allowed, years, threshold))


def selection_ok(values: dict[str, dict]) -> bool:
    return all(value["return_pct"] > 0 and value["dd_pct"] <= 15
               and value["trades_per_calendar_day"] >= 3
               for value in values.values())


def subset_score(values: dict[str, dict]) -> float:
    equivalent = [value["equivalent_monthly_return_pct"]
                  for value in values.values()]
    return round(min(equivalent) + .25 * sum(equivalent)
                 - .10 * max(value["dd_pct"] for value in values.values())
                 - .20 * sum(value["negative_months"]
                             for value in values.values()), 8)


def objective_ok(value: dict) -> bool:
    return bool(value["months"] > 0 and value["dd_pct"] <= 10
                and value["trades_per_calendar_day"] >= 3
                and value["months_ge_30pct"] == value["months"])


def main() -> None:
    parent = json.loads(PARENT_REPORT.read_text(encoding="utf-8"))
    frozen = parent.get("frozen_before_validation")
    if not frozen:
        raise ValueError("literal-runner parent has no frozen profile")
    target = float(frozen["target_pct"])
    hold = int(frozen["hold_bars"])
    threshold = float(frozen["threshold"])
    candles, sources = base.load()
    features = base.build_features(candles)
    proposals = base.proposals(features)
    frame = literal.add_outcomes(proposals, candles, target, hold)
    train = frame[frame.year.isin(literal.FIT_YEARS)]
    columns = base.schema(train)
    estimator = literal.model()
    estimator.fit(base.matrix(train, columns),
                  train.target_hit.to_numpy(np.int8))
    frame["quality"] = estimator.predict_proba(base.matrix(frame, columns))[:, 1]

    candidates = []
    for size in range(1, len(base.FAMILIES) + 1):
        for enabled in itertools.combinations(base.FAMILIES, size):
            values = {str(year): evaluate(frame, (year,), threshold, enabled)
                      for year in SELECTION_YEARS}
            candidates.append({
                "enabled_families": list(enabled),
                "selection": values,
                "selection_pass": selection_ok(values),
                "score": subset_score(values),
            })
    candidates.sort(key=lambda item: (
        item["selection_pass"], item["score"]), reverse=True)
    chosen = candidates[0] if candidates and candidates[0]["selection_pass"] else None
    validation = reveal = checks = None
    status = "REJECTED_SELECTION"
    if chosen:
        enabled = tuple(chosen["enabled_families"])
        validation = evaluate(frame, (VALIDATION_YEAR,), threshold, enabled)
        reveal = {str(year): evaluate(frame, (year,), threshold, enabled)
                  for year in REVEAL_YEARS}
        yearly = {**chosen["selection"], str(VALIDATION_YEAR): validation,
                  **reveal}
        checks = {year: objective_ok(value) for year, value in yearly.items()}
        status = ("FAMILY_ROUTER_RESEARCH_CANDIDATE"
                  if all(checks.values()) else "REJECTED_OBJECTIVE")
        allowed = frame[family_mask(frame, enabled)]
        accepted = portfolio.portfolio_rows(
            allowed, (*SELECTION_YEARS, VALIDATION_YEAR, *REVEAL_YEARS),
            threshold)
        pd.DataFrame(accepted).to_pickle(ROWS_OUT)
    report = {
        "status": status,
        "ready_for_live_orders": False,
        "protocol": {
            "parent_profile_frozen_on": 2022,
            "family_selection": list(SELECTION_YEARS),
            "validation": VALIDATION_YEAR,
            "revealed_after_freeze": list(REVEAL_YEARS),
            "family_subsets": 2 ** len(base.FAMILIES) - 1,
            "target_pct": target,
            "stop_pct": 1.0,
            "hold_hours": hold,
            "orders": "market only",
            "money": "3% current-equity margin, fixed 10x gross",
            "objective": ">=30% account return in every month; >=3 trades/day; DD <=10%",
        },
        "candidate_count": len(candidates),
        "top_selection": candidates[:20],
        "frozen_before_2024": chosen,
        "validation_2024": validation,
        "reveal_2025_2026": reveal,
        "objective_checks": checks,
        "sources": {
            **sources,
            str(PARENT_REPORT): digest(PARENT_REPORT),
            str(Path(base.__file__)): digest(Path(base.__file__)),
            str(Path(literal.__file__)): digest(Path(literal.__file__)),
            str(Path(__file__)): digest(Path(__file__)),
        },
        "limitations": [
            "Overlapping semantic flags use an any-enabled-family rule.",
            "The family subset is selected before 2024 is inspected.",
            "The result remains historical research until direct shadow transfer passes.",
        ],
    }
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({
        "status": status,
        "candidate_count": len(candidates),
        "chosen": chosen,
        "validation_2024": validation,
        "reveal_2025_2026": reveal,
        "objective_checks": checks,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
