"""Partial-lock 0.5%-to-2% runner for causal walk-forward flow signals."""
from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path

import numba as nb
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

import crypto_5m_regime_router_literal_oct6_audit as price
import crypto_futures_trade_flow_regime_oct7_audit as flow
import crypto_futures_trade_flow_walkforward_oct7_audit as walk


OUT = Path("reports/audit_2026_09_08/crypto_futures_trade_flow_ladder_oct7.json")
ROWS_OUT = Path("reports/audit_2026_09_08/crypto_futures_trade_flow_ladder_oct7.pkl")
MODEL = {"leaves": 31, "leaf": 400, "l2": 5.0}
QUANTILE = .925
NEAR_PCT = .5
FAR_TARGETS = (1.0, 1.5, 2.0)
LOCK_FRACTIONS = (.25, .50, .75)
MAX_BARS_VALUES = (24, 48, 144)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@nb.njit(cache=True)
def resolve_ladder(open_, high, low, close, times, indices, directions,
                   far_pct, lock_fraction, max_bars):
    n = len(indices)
    returns = np.empty(n, dtype=np.float64)
    exits = np.empty(n, dtype=np.int64)
    near_hits = np.zeros(n, dtype=np.int8)
    far_hits = np.zeros(n, dtype=np.int8)
    reasons = np.empty(n, dtype=np.int8)
    near_move = NEAR_PCT/100.0
    far_move = far_pct/100.0
    stop_move = .01
    for row in range(n):
        first = indices[row]
        direction = directions[row]
        entry = open_[first]
        near = entry*(1+direction*near_move)
        far = entry*(1+direction*far_move)
        stop = entry*(1-direction*stop_move)
        locked = 0.0
        result = 0.0
        reason = 2
        last = first
        near_done = False
        resolved = False
        end = min(len(times), first+max_bars)
        for index in range(first, end):
            if index > first and times[index]-times[index-1] != 300:
                break
            last = index
            if not near_done:
                stop_hit = low[index] <= stop if direction > 0 else high[index] >= stop
                near_hit = high[index] >= near if direction > 0 else low[index] <= near
                if stop_hit:
                    fill = min(open_[index], stop) if direction > 0 else max(open_[index], stop)
                    result = direction*(fill-entry)/entry
                    reason, resolved = 0, True
                    break
                if near_hit:
                    near_done = True
                    near_hits[row] = 1
                    locked = lock_fraction*near_move
                    continue  # far/BE collision in the same bar is unknowable
            else:
                be_hit = low[index] <= entry if direction > 0 else high[index] >= entry
                far_hit = high[index] >= far if direction > 0 else low[index] <= far
                if be_hit:
                    result = locked
                    reason, resolved = 3, True
                    break
                if far_hit:
                    result = locked+(1-lock_fraction)*far_move
                    far_hits[row] = 1
                    reason, resolved = 1, True
                    break
        if not resolved:
            move = direction*(close[last]-entry)/entry
            if near_done:
                move = max(0.0, min(far_move, move))
                result = locked+(1-lock_fraction)*move
            else:
                result = max(-stop_move, min(near_move, move))
        returns[row] = result
        exits[row] = times[last]+300
        reasons[row] = reason
    return returns, exits, near_hits, far_hits, reasons


def add_ladder(frame, candles, far_pct: float, lock_fraction: float,
               max_bars: int):
    pieces = []
    for symbol, part in frame.groupby("symbol", sort=False):
        candle = candles[symbol]
        values = resolve_ladder(
            np.asarray(candle["open"], dtype=float),
            np.asarray(candle["high"], dtype=float),
            np.asarray(candle["low"], dtype=float),
            np.asarray(candle["close"], dtype=float),
            np.asarray(candle["time"], dtype=np.int64),
            part.entry_index.to_numpy(np.int64),
            part.direction.to_numpy(np.int8), far_pct, lock_fraction, max_bars)
        found = part.copy()
        found["signed_return"] = values[0]
        found["exit_time"] = values[1]
        found["target_hit"] = values[2].astype(bool)
        found["far_target_hit"] = values[3].astype(bool)
        found["exit_reason"] = values[4]
        pieces.append(found)
    return pd.concat(pieces, ignore_index=True)


def pretest_ok(value: dict) -> bool:
    return bool(value["n"] >= 500 and value["return_pct"] > 0
                and value["literal_target_hit_rate"] >= 64
                and value["dd_pct"] <= 8
                and value["trades_per_calendar_day"] >= 2.5)


def objective_ok(value: dict) -> bool:
    return bool(value["n"] >= 500 and value["return_pct"] > 0
                and 75 <= value["literal_target_hit_rate"] <= 95
                and value["dd_pct"] <= 5 and value["negative_months"] == 0
                and value["trades_per_calendar_day"] >= 3
                and value["months_ge_30pct"] == value["months"])


def main() -> None:
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

    candidates, frontier = [], []
    for far_pct, lock_fraction, max_bars in itertools.product(
            FAR_TARGETS, LOCK_FRACTIONS, MAX_BARS_VALUES):
        resolved = add_ladder(proposals, candles, far_pct, lock_fraction, max_bars)
        keys = ["symbol", "entry_time", "direction"]
        if not resolved[keys].reset_index(drop=True).equals(
                training[keys].reset_index(drop=True)):
            raise ValueError("ladder/model row alignment mismatch")
        resolved["quality"] = quality
        pretest = {str(year): price.evaluate(
            resolved, (year,), thresholds[str(year)]) for year in (2022, 2023, 2024)}
        far = {}
        for year in (2022, 2023, 2024):
            rows = price.portfolio_rows(resolved, (year,), thresholds[str(year)])
            far[str(year)] = {"hits": sum(bool(row["far_target_hit"]) for row in rows),
                              "rate": round(100*sum(bool(row["far_target_hit"])
                                                    for row in rows)/len(rows), 4)
                              if rows else 0.0}
        item = {"exit": {"near_pct": NEAR_PCT, "far_pct": far_pct,
                         "lock_fraction": lock_fraction,
                         "max_bars": max_bars},
                "pretest": pretest, "far_target": far}
        item["score"] = round(sum(
            value["return_pct"]-2*value["dd_pct"]
            for value in pretest.values()), 8)
        frontier.append(item)
        if pretest_ok(pretest["2022"]) and pretest_ok(pretest["2023"]) \
                and pretest_ok(pretest["2024"]):
            item["_frame"] = resolved
            candidates.append(item)
    candidates.sort(key=lambda item: item["score"], reverse=True)
    frontier.sort(key=lambda item: item["score"], reverse=True)
    frozen = candidates[0] if candidates else None
    results = checks = None
    if frozen:
        results = {str(year): price.evaluate(
            frozen["_frame"], (year,), thresholds[str(year)]) for year in walk.YEARS}
        checks = {year: objective_ok(value) for year, value in results.items()}
        selected = []
        for year in walk.YEARS:
            part = frozen["_frame"]
            selected.append(part[part.year.eq(year)
                                 & part.quality.ge(thresholds[str(year)])])
        pd.concat(selected, ignore_index=True).to_pickle(
            ROWS_OUT, compression="gzip")
    public_frozen = ({key: value for key, value in frozen.items() if key != "_frame"}
                     if frozen else None)
    report = {
        "status": ("RESEARCH_CANDIDATE_AWAITING_VENUE_FORWARD"
                   if checks and all(checks.values()) else "REJECTED"),
        "protocol": {"signal": "annual causal walk-forward trade-flow model",
                     "model": MODEL, "quantile": QUANTILE,
                     "selection": "exit grid on 2022-2024",
                     "test": "2025", "observed": "2026",
                     "near_pct": NEAR_PCT, "far_pct": FAR_TARGETS,
                     "lock_fraction": LOCK_FRACTIONS,
                     "max_bars": MAX_BARS_VALUES, "initial_stop_pct": 1.0,
                     "runner_stop": "entry after literal near target",
                     "collision": "stop/BE first; no same-bar far after near",
                     "money": "3% equity margin, fixed 10x gross"},
        "thresholds": thresholds, "qualified_before_test": len(candidates),
        "frozen_before_test": public_frozen,
        "top_before_test": [{key: value for key, value in item.items()
                             if key != "_frame"} for item in candidates[:12]],
        "pretest_frontier": frontier[:20],
        "results": results, "checks": checks,
        "sources": {**sources, str(Path(flow.__file__)): digest(Path(flow.__file__)),
                    str(Path(walk.__file__)): digest(Path(walk.__file__)),
                    str(Path(__file__)): digest(Path(__file__))},
        "limitations": [
            "Bar data cannot order intrabar near/far/BE touches; handling is pessimistic.",
            "The bounded 27-exit grid creates selection risk.",
            "Any historical pass still requires untouched direct X-Perp evidence.",
        ],
    }
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"status": report["status"],
                      "qualified_before_test": len(candidates),
                      "frozen": public_frozen, "results": results,
                      "checks": checks}, indent=2), flush=True)


if __name__ == "__main__":
    main()
