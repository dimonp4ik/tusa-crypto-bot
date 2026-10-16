"""Route frozen 2% and 2.5% crypto runners with simple causal regimes.

Both entry models are fit on 2020-2021 and use cutoffs already selected on
2022.  A small, predefined set of interpretable regime rules is developed on
2022-2023, frozen, and then reported on 2024-2026.  The router operates before
the shared exposure, cooldown and daily-loss-pause controls.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

import crypto_okx_swap_1h_regime_oct7_audit as base
import crypto_profit_first_literal_runner_oct16_audit as literal
from crypto_profit_first_runner_oct16_audit import enrich
from crypto_runtime_shadow_portfolio_oct3_audit import money, runtime_gate


ROOT = Path("reports/audit_2026_09_08")
OUT = ROOT / "crypto_profit_first_regime_target_router_oct17.json"
ROWS_OUT = ROOT / "crypto_profit_first_regime_target_router_oct17.pkl"
FIT_YEARS = (2020, 2021)
DEVELOPMENT_YEARS = (2022, 2023)
CONFIRMATION_YEARS = (2024, 2025, 2026)
PROFILES = {
    2.0: {"hold_bars": 24, "threshold": 0.39302933351899905},
    2.5: {"hold_bars": 24, "threshold": 0.34531812541857465},
}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rules() -> dict[str, Callable[[pd.DataFrame], pd.Series]]:
    output: dict[str, Callable[[pd.DataFrame], pd.Series]] = {
        "always_2pct": lambda frame: pd.Series(False, index=frame.index),
        "always_2_5pct": lambda frame: pd.Series(True, index=frame.index),
        "aligned_btc24": lambda frame: frame.direction * frame.btc_ret24 > 0,
        "counter_btc24": lambda frame: frame.direction * frame.btc_ret24 < 0,
    }
    for threshold in (.006, .008, .010, .012, .015):
        output[f"atr14_ge_{threshold:g}"] = (
            lambda frame, value=threshold: frame.atr14_pct >= value)
        output[f"atr14_ge_{threshold:g}_aligned_btc24"] = (
            lambda frame, value=threshold:
            (frame.atr14_pct >= value) & (frame.direction * frame.btc_ret24 > 0))
    for threshold in (.008, .012, .020, .030):
        output[f"abs_btc24_ge_{threshold:g}"] = (
            lambda frame, value=threshold: frame.btc_ret24.abs() >= value)
    for threshold in (.02, .04, .06, .10):
        output[f"abs_btc168_ge_{threshold:g}"] = (
            lambda frame, value=threshold: frame.btc_ret168.abs() >= value)
    for family in ("trend", "pullback", "breakout", "compression_release",
                   "breadth", "relative_leader"):
        output[f"family_{family}"] = (
            lambda frame, name=family: frame[f"family_{name}"].fillna(0).astype(bool))
    return output


def routed_rows(profile_frames: dict[float, pd.DataFrame], years: tuple[int, ...],
                prefer_2_5: Callable[[pd.DataFrame], pd.Series]) -> list[dict]:
    pieces = []
    for target, frame in profile_frames.items():
        profile = PROFILES[target]
        part = frame[
            frame.year.isin(years) & frame.quality.ge(profile["threshold"])
        ].copy()
        part["route_target_pct"] = target
        part["normalized_quality"] = (
            (part.quality - profile["threshold"]) / (1 - profile["threshold"])
        )
        pieces.append(part)
    union = pd.concat(pieces, ignore_index=True)
    desired_2_5 = prefer_2_5(union)
    union["preferred_target"] = np.where(desired_2_5, 2.5, 2.0)
    union["is_preferred"] = union.route_target_pct.eq(union.preferred_target)
    keys = ["symbol", "entry_time", "direction"]
    union = union.sort_values(
        [*keys, "is_preferred", "normalized_quality"],
        ascending=[True, True, True, False, False],
    ).drop_duplicates(keys, keep="first")
    union = union.sort_values(
        ["entry_time", "normalized_quality", "symbol"],
        ascending=[True, False, True],
    )
    union["source_priority"] = union.groupby("entry_time").cumcount()
    union["module_priority"] = 0
    union["source"] = "regime_target_router"
    union["module"] = "regime_target_router"
    union["direction"] = np.where(union.direction > 0, "LONG", "SHORT")
    union["outcome"] = np.where(union.signed_return < 0, "SL", "TP")
    union["signed_price_return"] = union.signed_return
    return runtime_gate(union.to_dict("records"))


def evaluate(profile_frames: dict[float, pd.DataFrame], years: tuple[int, ...],
             prefer_2_5: Callable[[pd.DataFrame], pd.Series]) -> dict:
    rows = routed_rows(profile_frames, years, prefer_2_5)
    value = enrich(money(rows, .03))
    counts = pd.Series([float(row["route_target_pct"]) for row in rows]).value_counts()
    value["routed_target_counts"] = {
        str(target): int(counts.get(target, 0)) for target in PROFILES}
    return value


def development_ok(value: dict) -> bool:
    return bool(value["return_pct"] > 0 and value["dd_pct"] <= 15
                and value["trades_per_calendar_day"] >= 3)


def development_score(values: dict[str, dict]) -> float:
    monthly = min(value["equivalent_monthly_return_pct"] for value in values.values())
    drawdown = max(value["dd_pct"] for value in values.values())
    negative = sum(value["negative_months"] for value in values.values())
    return round(monthly - .08 * drawdown - .10 * negative, 8)


def objective_ok(value: dict) -> bool:
    return bool(value["months"] > 0 and value["dd_pct"] <= 10
                and value["trades_per_calendar_day"] >= 3
                and value["months_ge_30pct"] == value["months"])


def main() -> None:
    candles, sources = base.load()
    features = base.build_features(candles)
    proposals = base.proposals(features)
    profile_frames = {}
    model_reports = []
    for target, profile in PROFILES.items():
        frame = literal.add_outcomes(
            proposals, candles, target, profile["hold_bars"])
        fit = frame.year.isin(FIT_YEARS)
        columns = base.schema(frame.loc[fit])
        estimator = literal.model()
        estimator.fit(base.matrix(frame.loc[fit], columns),
                      frame.loc[fit, "target_hit"].to_numpy(np.int8))
        frame["quality"] = estimator.predict_proba(base.matrix(frame, columns))[:, 1]
        profile_frames[target] = frame
        model_reports.append({
            "target_pct": target, **profile,
            "fit_rows": int(fit.sum()),
            "fit_target_prevalence_pct": round(
                100 * float(frame.loc[fit, "target_hit"].mean()), 4),
        })
    candidates = []
    available_rules = rules()
    for name, rule in available_rules.items():
        development = {
            str(year): evaluate(profile_frames, (year,), rule)
            for year in DEVELOPMENT_YEARS
        }
        candidate = {
            "rule": name,
            "development": development,
            "development_pass": all(development_ok(value)
                                    for value in development.values()),
            "score": development_score(development),
        }
        candidates.append(candidate)
        print(json.dumps(candidate, default=float), flush=True)
    candidates.sort(key=lambda item: (item["development_pass"], item["score"]),
                    reverse=True)
    frozen = candidates[0] if candidates and candidates[0]["development_pass"] else None
    confirmation = checks = None
    accepted_rows = []
    status = "REJECTED_DEVELOPMENT"
    if frozen:
        rule = available_rules[frozen["rule"]]
        confirmation = {
            str(year): evaluate(profile_frames, (year,), rule)
            for year in CONFIRMATION_YEARS
        }
        all_years = {**frozen["development"], **confirmation}
        checks = {year: objective_ok(value) for year, value in all_years.items()}
        status = ("REGIME_TARGET_RESEARCH_CANDIDATE"
                  if all(checks.values()) else "REJECTED_OBJECTIVE")
        accepted_rows = routed_rows(
            profile_frames, (*DEVELOPMENT_YEARS, *CONFIRMATION_YEARS), rule)
        pd.DataFrame(accepted_rows).to_pickle(ROWS_OUT)
    report = {
        "status": status, "ready_for_live_orders": False,
        "protocol": {
            "priority": "account return first; win rate diagnostic only",
            "entry_model_fit": list(FIT_YEARS),
            "profile_cutoffs_preselected_on": 2022,
            "regime_rule_development": list(DEVELOPMENT_YEARS),
            "confirmation_only": list(CONFIRMATION_YEARS),
            "profiles": PROFILES,
            "orders": "market only",
            "money": "3% current-equity margin, fixed 10x gross",
            "objective": ">=30% account return in every month; >=3 trades/day; DD <=10%",
        },
        "feature_rows": len(features), "proposal_rows": len(proposals),
        "models": model_reports, "candidate_count": len(candidates),
        "development_frontier": candidates,
        "frozen_before_confirmation": frozen,
        "confirmation_2024_2026": confirmation,
        "objective_checks": checks,
        "accepted_rows": len(accepted_rows),
        "sources": {**sources,
                    str(Path(base.__file__)): digest(Path(base.__file__)),
                    str(Path(literal.__file__)): digest(Path(literal.__file__)),
                    str(Path(__file__)): digest(Path(__file__))},
        "limitations": [
            "The regime rule is developed on 2022-2023 after profile cutoffs were selected.",
            "The confirmation years were seen by earlier related audits and are not pristine hidden data.",
            "Historical OKX global results still require direct EU X-Perp shadow transfer.",
        ],
    }
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"status": status, "frozen": frozen,
                      "confirmation_2024_2026": confirmation,
                      "objective_checks": checks}, indent=2), flush=True)


if __name__ == "__main__":
    main()
