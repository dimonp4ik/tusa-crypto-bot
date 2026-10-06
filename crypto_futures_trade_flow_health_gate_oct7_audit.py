"""Causal health gate for the 0.5%-to-2% trade-flow ladder.

The gate is deliberately trained only on 2022.  It reviews completed shadow
outcomes at fixed boundaries, so a decision never sees the result of an open
trade.  The chosen specification is then transferred unchanged to 2023 and
2024 before either later year is inspected for acceptance.
"""
from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

import crypto_5m_regime_router_literal_oct6_audit as price
import crypto_futures_trade_flow_ladder_oct7_audit as ladder
import crypto_futures_trade_flow_regime_oct7_audit as flow
import crypto_futures_trade_flow_walkforward_oct7_audit as walk
from crypto_runtime_shadow_portfolio_oct3_audit import money


OUT = Path("reports/audit_2026_09_08/crypto_futures_trade_flow_health_gate_oct7.json")
BASE_ROWS_OUT = Path("reports/audit_2026_09_08/crypto_futures_trade_flow_ladder_base_oct7.pkl")
MODEL = {"leaves": 31, "leaf": 400, "l2": 5.0}
QUANTILE = .925
EXIT = {"far_pct": 2.0, "lock_fraction": .75, "max_bars": 24}
REVIEW_DAYS = (1, 3, 7)
LOOKBACK_DAYS = (14, 30, 60, 90)
MINIMUMS = (20, 40, 80)
HIT_MINS = (.60, .62, .64, .66)
MEAN_MINS = (-.0005, -.00025, 0.0)
GROUPS = ("global", "direction", "symbol")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_ladder_rows() -> tuple[pd.DataFrame, dict]:
    candles, sources = flow.load()
    features = flow.attach_flow(price.build_features(candles), candles)
    proposals = flow.proposals(features)
    training = price.add_outcomes(proposals, candles)
    schema = price.columns(training[training.year.isin((2020, 2021))])
    _, matrix = flow.matrices(training, schema)
    labels = training.target_hit.astype(int).to_numpy()
    quality = np.full(len(training), np.nan)
    thresholds = {}
    for year in walk.YEARS:
        train, calibration, target = walk.masks(training, year)
        model = HistGradientBoostingClassifier(
            max_iter=180, learning_rate=.06, max_leaf_nodes=MODEL["leaves"],
            min_samples_leaf=MODEL["leaf"], l2_regularization=MODEL["l2"],
            random_state=20261007+year)
        model.fit(matrix[train], labels[train])
        calibration_scores = model.predict_proba(matrix[calibration])[:, 1]
        thresholds[str(year)] = float(np.quantile(calibration_scores, QUANTILE))
        quality[target] = model.predict_proba(matrix[target])[:, 1]
    resolved = ladder.add_ladder(proposals, candles, **EXIT)
    keys = ["symbol", "entry_time", "direction"]
    if not resolved[keys].reset_index(drop=True).equals(
            training[keys].reset_index(drop=True)):
        raise ValueError("ladder/model row alignment mismatch")
    resolved["quality"] = quality

    # First apply the real portfolio's exposure/cooldown/hourly rails.  The
    # health gate observes the resulting candidate stream in shadow mode.
    pieces = []
    for year in walk.YEARS:
        records = price.portfolio_rows(resolved, (year,), thresholds[str(year)])
        part = pd.DataFrame(records)
        if not part.empty:
            pieces.append(part)
    return pd.concat(pieces, ignore_index=True), {
        "sources": sources, "thresholds": thresholds,
        "raw_proposals": int(len(proposals)),
    }


def add_health(frame: pd.DataFrame, review_days: int, lookback_days: int,
               group: str) -> pd.DataFrame:
    source = frame.sort_values(["entry_time", "quality", "symbol"],
                               ascending=[True, False, True]).reset_index(drop=True)
    seconds = review_days * 86400
    source["review_start"] = (
        source.entry_time.astype(np.int64) // seconds) * seconds
    if group == "global":
        source["health_group"] = "all"
    elif group == "direction":
        source["health_group"] = source.direction.astype(str)
    elif group == "symbol":
        source["health_group"] = source.symbol.astype(str)
    else:
        raise ValueError(group)
    pieces = []
    for (start, bucket), current in source.groupby(
            ["review_start", "health_group"], sort=True, observed=True):
        history = source[
            source.health_group.eq(bucket)
            & source.exit_time.lt(start)
            & source.exit_time.ge(start-lookback_days*86400)]
        found = current.copy()
        found["health_n"] = len(history)
        found["health_mean"] = (history.signed_price_return.mean()
                                if len(history) else np.nan)
        found["health_hit"] = (history.target_hit.mean()
                               if len(history) else np.nan)
        pieces.append(found)
    return pd.concat(pieces, ignore_index=True).sort_values(
        ["entry_time", "quality", "symbol"], ascending=[True, False, True])


def gate(frame: pd.DataFrame, spec: dict) -> pd.DataFrame:
    ready = frame.health_n.ge(spec["minimum_completed"])
    healthy = (frame.health_mean.ge(spec["mean_min"])
               & frame.health_hit.ge(spec["hit_min"]))
    # Warm-up is permissive.  Once enough closed observations exist, a weak
    # bucket is disabled only until the next fixed review boundary.
    return frame.loc[~ready | healthy].copy()


def evaluate(frame: pd.DataFrame, year: int) -> dict:
    chosen = frame[frame.year.eq(year)].to_dict("records")
    result = money(chosen, .03)
    n = len(chosen)
    months = list(result.get("monthly_return_pct", {}).values())
    result.update({
        "literal_target_hits": sum(bool(row["target_hit"]) for row in chosen),
        "literal_target_hit_rate": round(
            100*sum(bool(row["target_hit"]) for row in chosen)/n, 4) if n else 0.0,
        "far_target_hits": sum(bool(row["far_target_hit"]) for row in chosen),
        "far_target_hit_rate": round(
            100*sum(bool(row["far_target_hit"]) for row in chosen)/n, 4) if n else 0.0,
        "months": len(months),
        "minimum_month_pct": round(min(months), 4) if months else 0.0,
        "median_month_pct": round(float(np.median(months)), 4) if months else 0.0,
    })
    return result


def selection_ok(value: dict) -> bool:
    return bool(value["n"] >= 900 and value["return_pct"] > 0
                and value["literal_target_hit_rate"] >= 67
                and value["dd_pct"] <= 6
                and value["trades_per_calendar_day"] >= 2.5)


def validation_ok(value: dict) -> bool:
    return bool(value["n"] >= 750 and value["return_pct"] > 0
                and value["literal_target_hit_rate"] >= 65
                and value["dd_pct"] <= 7
                and value["trades_per_calendar_day"] >= 2.0)


def main() -> None:
    base, build = build_ladder_rows()
    base.to_pickle(BASE_ROWS_OUT, compression="gzip")
    base_results = {str(year): evaluate(base, year)
                    for year in (2022, 2023, 2024)}
    cache = {}
    selection = []
    for group, review, lookback in itertools.product(
            GROUPS, REVIEW_DAYS, LOOKBACK_DAYS):
        if lookback < review:
            continue
        featured = add_health(base, review, lookback, group)
        cache[(group, review, lookback)] = featured
        for minimum, hit_min, mean_min in itertools.product(
                MINIMUMS, HIT_MINS, MEAN_MINS):
            spec = {"group": group, "review_days": review,
                    "lookback_days": lookback,
                    "minimum_completed": minimum,
                    "hit_min": hit_min, "mean_min": mean_min}
            value = evaluate(gate(featured, spec), 2022)
            if selection_ok(value):
                selection.append({"spec": spec, "selection_2022": value})
    selection.sort(key=lambda item: (
        item["selection_2022"]["return_pct"]
        - 2*item["selection_2022"]["dd_pct"],
        item["selection_2022"]["literal_target_hit_rate"],
        item["selection_2022"]["trades_per_calendar_day"]), reverse=True)

    # Freeze exactly one specification before validation.  Later years do not
    # participate in ranking or fallback selection.
    frozen = selection[0] if selection else None
    validation = checks = later = None
    if frozen:
        spec = frozen["spec"]
        featured = cache[(spec["group"], spec["review_days"],
                          spec["lookback_days"])]
        chosen = gate(featured, spec)
        validation = {str(year): evaluate(chosen, year)
                      for year in (2023, 2024)}
        checks = {year: validation_ok(value)
                  for year, value in validation.items()}
        if all(checks.values()):
            later = {str(year): evaluate(chosen, year)
                     for year in (2025, 2026)}
    status = ("RESEARCH_CANDIDATE_AWAITING_VENUE_FORWARD"
              if later and all(validation_ok(value) for value in later.values())
              else "REJECTED")
    report = {
        "status": status,
        "protocol": {
            "signal": "annual causal walk-forward trade-flow model",
            "model": MODEL, "quantile": QUANTILE,
            "exit": {"near_pct": .5, **EXIT, "initial_stop_pct": 1.0,
                     "runner_stop": "entry after near target"},
            "health_selection": "2022 only",
            "validation": [2023, 2024], "test_if_validated": [2025, 2026],
            "health_information": "completed shadow outcomes before fixed review boundary",
            "money": "3% equity margin, fixed 10x gross",
        },
        "grid_size": len(GROUPS)*len(REVIEW_DAYS)*len(LOOKBACK_DAYS)
                     *len(MINIMUMS)*len(HIT_MINS)*len(MEAN_MINS),
        "base_rows": len(base), "raw_proposals": build["raw_proposals"],
        "thresholds": build["thresholds"], "base_pretest": base_results,
        "qualified_on_2022": len(selection),
        "frozen_before_validation": frozen,
        "validation": validation, "validation_checks": checks,
        "later_years": later,
        "selection_frontier": selection[:20],
        "sources": {**build["sources"],
                    str(Path(ladder.__file__)): digest(Path(ladder.__file__)),
                    str(Path(__file__)): digest(Path(__file__))},
        "limitations": [
            "The health grid is finite but still creates 2022 selection risk.",
            "Shadow health requires recording hypothetical outcomes while a bucket is paused.",
            "Any historical pass still requires untouched direct X-Perp shadow evidence.",
        ],
    }
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({key: report[key] for key in (
        "status", "base_pretest", "qualified_on_2022",
        "frozen_before_validation", "validation", "validation_checks",
        "later_years")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
