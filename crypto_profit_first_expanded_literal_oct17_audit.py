"""Profit-first 2% crypto runner on the expanded 19-symbol universe.

The target, stop and 24-hour hold are fixed before this audit.  Models are fit
on 2020-2021 and only the symbol-identity choice and numeric cutoff are selected
on 2022.  That frozen choice is then reported separately on 2023-2026.  Win
rate is diagnostic and is not an acceptance gate.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

import crypto_5m_regime_router_literal_oct6_audit as portfolio
import crypto_okx_swap_1h_regime_oct7_audit as base
import crypto_profit_first_literal_runner_oct16_audit as literal
from crypto_profit_first_runner_oct16_audit import enrich


ROOT = Path("reports/audit_2026_09_08")
MANIFEST = ROOT / "crypto_okx_swap_1h_expanded_oct16_11z" / "manifest.json"
OUT = ROOT / "crypto_profit_first_expanded_literal_oct17.json"
ROWS_OUT = ROOT / "crypto_profit_first_expanded_literal_oct17.pkl"
TARGET_PCT = 2.0
HOLD_BARS = 24
FIT_YEARS = (2020, 2021)
SELECTION_YEAR = 2022
LATER_YEARS = (2023, 2024, 2025, 2026)
QUANTILES = (.70, .75, .80, .85, .875, .90, .925, .95, .96, .97, .98, .99)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def evaluate(frame: pd.DataFrame, years: tuple[int, ...], threshold: float) -> dict:
    return enrich(portfolio.evaluate(frame, years, threshold))


def selection_ok(value: dict) -> bool:
    return bool(value["return_pct"] > 0 and value["dd_pct"] <= 15
                and value["trades_per_calendar_day"] >= 3)


def score(value: dict) -> float:
    return round(value["equivalent_monthly_return_pct"]
                 - .10 * value["dd_pct"]
                 - .20 * value["negative_months"], 8)


def objective_ok(value: dict) -> bool:
    return bool(value["months"] > 0 and value["dd_pct"] <= 10
                and value["trades_per_calendar_day"] >= 3
                and value["months_ge_30pct"] == value["months"])


def feature_columns(frame: pd.DataFrame, include_symbol: bool) -> list[str]:
    columns = [*base.FEATURES, *[f"family_{name}" for name in base.FAMILIES]]
    if include_symbol:
        columns.extend(f"symbol_{name}" for name in sorted(frame.symbol.unique()))
    return columns


def main() -> None:
    base.MANIFEST = MANIFEST
    candles, sources = base.load()
    features = base.build_features(candles)
    proposals = base.proposals(features)
    frame = literal.add_outcomes(proposals, candles, TARGET_PCT, HOLD_BARS)
    fit = frame.year.isin(FIT_YEARS).to_numpy()
    candidates = []
    quality_by_variant: dict[str, np.ndarray] = {}
    model_reports = []
    for include_symbol in (False, True):
        variant = "with_symbol_identity" if include_symbol else "transferable_no_symbol"
        columns = feature_columns(frame.loc[fit], include_symbol)
        estimator = literal.model()
        estimator.fit(base.matrix(frame.loc[fit], columns),
                      frame.loc[fit, "target_hit"].to_numpy(np.int8))
        quality = estimator.predict_proba(base.matrix(frame, columns))[:, 1]
        quality_by_variant[variant] = quality
        scored = frame.copy()
        scored["quality"] = quality
        selection_scores = scored.loc[scored.year.eq(SELECTION_YEAR), "quality"]
        frontier = []
        for quantile in QUANTILES:
            threshold = float(selection_scores.quantile(quantile))
            value = evaluate(scored, (SELECTION_YEAR,), threshold)
            item = {
                "variant": variant,
                "include_symbol_identity": include_symbol,
                "selection_quantile": quantile,
                "threshold": threshold,
                "selection_2022": value,
                "selection_pass": selection_ok(value),
                "score": score(value),
            }
            candidates.append(item)
            frontier.append(item)
        model_reports.append({
            "variant": variant,
            "include_symbol_identity": include_symbol,
            "feature_count": len(columns),
            "fit_rows": int(fit.sum()),
            "fit_target_prevalence_pct": round(
                100 * float(frame.loc[fit, "target_hit"].mean()), 4),
            "selection_frontier": frontier,
        })
        print(json.dumps({"variant": variant,
                          "best_2022": max(frontier, key=lambda x: x["score"])},
                         default=float), flush=True)
    candidates.sort(key=lambda item: (item["selection_pass"], item["score"]),
                    reverse=True)
    frozen = candidates[0] if candidates and candidates[0]["selection_pass"] else None
    later = checks = None
    status = "REJECTED_SELECTION"
    if frozen:
        frame["quality"] = quality_by_variant[frozen["variant"]]
        later = {str(year): evaluate(frame, (year,), frozen["threshold"])
                 for year in LATER_YEARS}
        all_years = {str(SELECTION_YEAR): frozen["selection_2022"], **later}
        checks = {year: objective_ok(value) for year, value in all_years.items()}
        status = ("EXPANDED_LITERAL_RESEARCH_CANDIDATE"
                  if all(checks.values()) else "REJECTED_OBJECTIVE")
        accepted = portfolio.portfolio_rows(
            frame, (SELECTION_YEAR, *LATER_YEARS), frozen["threshold"])
        pd.DataFrame(accepted).to_pickle(ROWS_OUT)
    report = {
        "status": status,
        "ready_for_live_orders": False,
        "protocol": {
            "priority": "account return first; win rate diagnostic only",
            "symbols": sorted(candles),
            "model_fit": list(FIT_YEARS),
            "variant_and_cutoff_selection": SELECTION_YEAR,
            "frozen_later_years": list(LATER_YEARS),
            "target_pct": TARGET_PCT,
            "stop_pct": 1.0,
            "hold_hours": HOLD_BARS,
            "entry": "next one-hour market open",
            "orders": "market only",
            "money": "3% current-equity margin, fixed 10x gross",
            "objective": ">=30% account return in every month; >=3 trades/day; DD <=10%",
        },
        "feature_rows": len(features),
        "proposal_rows": len(frame),
        "candidate_count": len(candidates),
        "models": model_reports,
        "selection_top20": candidates[:20],
        "frozen_before_2023": frozen,
        "later_2023_2026": later,
        "objective_checks": checks,
        "sources": {
            **sources,
            str(MANIFEST): digest(MANIFEST),
            str(Path(base.__file__)): digest(Path(base.__file__)),
            str(Path(literal.__file__)): digest(Path(literal.__file__)),
            str(Path(__file__)): digest(Path(__file__)),
        },
        "limitations": [
            "The 2022 selection chooses between symbol identity and transfer-only features.",
            "One-hour target/stop collisions are resolved pessimistically as stops.",
            "Historical OKX global results still require direct EU X-Perp shadow transfer.",
        ],
    }
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"status": status, "frozen": frozen,
                      "later_2023_2026": later,
                      "objective_checks": checks}, indent=2), flush=True)


if __name__ == "__main__":
    main()
