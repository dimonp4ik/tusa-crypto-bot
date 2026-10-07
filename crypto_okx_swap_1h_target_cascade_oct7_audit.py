"""Chronological audit of learned 0.5/1/1.5/2% targets on OKX 1H entries.

The entry universe and base 0.5% classifier are identical to the frozen
one-hour candidate.  Separate target-touch classifiers are fitted on
2020-2021 only.  Upgrade thresholds are selected on 2022 only, checked on
2023, and 2024-2026 are revealed only if that validation passes.
"""
from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path

import numba as nb
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

import crypto_okx_swap_1h_regime_oct7_audit as base
import crypto_5m_regime_router_literal_oct6_audit as portfolio


OUT = Path("reports/audit_2026_09_08/crypto_okx_swap_1h_target_cascade_oct7.json")
ROWS_OUT = Path("reports/audit_2026_09_08/crypto_okx_swap_1h_target_cascade_oct7.pkl")
TARGETS = (.5, 1.0, 1.5, 2.0)
HOLD = {.5: 12, 1.0: 24, 1.5: 36, 2.0: 48}
STOP_PCT = 1.0
QUANTILES = (.75, .85, .90, .94, .97, .985, .995)
BASE_THRESHOLD = .7303150537752118
MODEL_CONFIG = {"leaves": 31, "leaf": 280, "l2": 5.0}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@nb.njit(cache=True)
def resolve(open_, high, low, close, times, entries, directions,
            target_pct, hold_bars):
    n = len(entries)
    signed = np.empty(n, dtype=np.float64)
    exits = np.empty(n, dtype=np.int64)
    hits = np.zeros(n, dtype=np.int8)
    reasons = np.empty(n, dtype=np.int8)
    target_move = target_pct / 100.0
    stop_move = STOP_PCT / 100.0
    for row in range(n):
        first = entries[row]
        direction = directions[row]
        entry = open_[first]
        target = entry * (1 + direction * target_move)
        stop = entry * (1 - direction * stop_move)
        last = first
        value = 0.0
        reason = 2
        resolved = False
        end = min(len(times), first + hold_bars)
        for index in range(first, end):
            if index > first and times[index] - times[index - 1] != 3600:
                break
            last = index
            stop_hit = low[index] <= stop if direction > 0 else high[index] >= stop
            target_hit = high[index] >= target if direction > 0 else low[index] <= target
            # Conservative ambiguity rule, identical to the frozen audit.
            if stop_hit:
                fill = min(open_[index], stop) if direction > 0 else max(open_[index], stop)
                value = direction * (fill / entry - 1)
                reason = 0
                resolved = True
                break
            if target_hit:
                value = target_move
                hits[row] = 1
                reason = 1
                resolved = True
                break
        if not resolved:
            value = direction * (close[last] / entry - 1)
            value = max(-stop_move, min(target_move, value))
        signed[row] = value
        exits[row] = times[last] + 3600
        reasons[row] = reason
    return signed, exits, hits, reasons


def add_target(frame: pd.DataFrame, candles: dict, target: float) -> None:
    suffix = str(target).replace(".", "_")
    signed = np.empty(len(frame), dtype=float)
    exits = np.empty(len(frame), dtype=np.int64)
    hits = np.empty(len(frame), dtype=np.int8)
    reasons = np.empty(len(frame), dtype=np.int8)
    for symbol, indices in frame.groupby("symbol", sort=False).groups.items():
        positions = np.asarray(list(indices), dtype=np.int64)
        part = frame.loc[positions]
        candle = candles[symbol]
        arrays = [np.asarray(candle[key], dtype=float)
                  for key in ("open", "high", "low", "close")]
        values = resolve(
            *arrays, np.asarray(candle["time"], dtype=np.int64),
            part.entry_index.to_numpy(np.int64),
            part.direction.to_numpy(np.int8), target, HOLD[target])
        signed[positions], exits[positions], hits[positions], reasons[positions] = values
    frame[f"signed_{suffix}"], frame[f"exit_{suffix}"] = signed, exits
    frame[f"hit_{suffix}"], frame[f"reason_{suffix}"] = hits.astype(bool), reasons


def estimator() -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        max_iter=180, learning_rate=.06,
        max_leaf_nodes=MODEL_CONFIG["leaves"],
        min_samples_leaf=MODEL_CONFIG["leaf"],
        l2_regularization=MODEL_CONFIG["l2"], random_state=20261007)


def routed(frame: pd.DataFrame, thresholds: dict[float, float]) -> pd.DataFrame:
    found = frame[frame.quality_0_5.ge(BASE_THRESHOLD)].copy()
    target = np.full(len(found), .5, dtype=float)
    for value in (1.0, 1.5, 2.0):
        target[found[f"quality_{str(value).replace('.', '_')}"]
               .to_numpy() >= thresholds[value]] = value
    found["selected_target_pct"] = target
    found["signed_return"] = 0.0
    found["exit_time"] = 0
    found["target_hit"] = False
    found["exit_reason"] = 2
    for value in TARGETS:
        mask = found.selected_target_pct.eq(value)
        suffix = str(value).replace(".", "_")
        found.loc[mask, "signed_return"] = found.loc[mask, f"signed_{suffix}"]
        found.loc[mask, "exit_time"] = found.loc[mask, f"exit_{suffix}"]
        found.loc[mask, "target_hit"] = found.loc[mask, f"hit_{suffix}"]
        found.loc[mask, "exit_reason"] = found.loc[mask, f"reason_{suffix}"]
    found["quality"] = 1.0
    return found


def evaluate(frame: pd.DataFrame, years: tuple[int, ...]) -> dict:
    value = portfolio.evaluate(frame, years, 0.5)
    rows = portfolio.portfolio_rows(frame, years, 0.5)
    distribution = {str(target): sum(
        float(row.get("selected_target_pct", 0)) == target for row in rows)
        for target in TARGETS}
    hits = {str(target): sum(
        float(row.get("selected_target_pct", 0)) == target
        and bool(row.get("target_hit")) for row in rows) for target in TARGETS}
    value["target_distribution"] = distribution
    value["target_hits"] = hits
    value["average_selected_target_pct"] = round(
        sum(float(row.get("selected_target_pct", 0)) for row in rows) / len(rows), 4
        if rows else 0.0)
    return value


def passes(value: dict, selection: bool = False) -> bool:
    return bool(value["n"] >= (700 if selection else 500)
                and value["return_pct"] > 0
                and value["literal_target_hit_rate"] >= (68 if selection else 65)
                and value["dd_pct"] <= 7
                and value["trades_per_calendar_day"] >= 2.0)


def main() -> None:
    candles, sources = base.load()
    raw = base.build_features(candles)
    frame = base.proposals(raw).reset_index(drop=True)
    for target in TARGETS:
        add_target(frame, candles, target)
    fit = frame.year.isin((2020, 2021)).to_numpy()
    columns = base.schema(frame.loc[fit])
    x = base.matrix(frame, columns)
    prevalence = {}
    for target in TARGETS:
        suffix = str(target).replace(".", "_")
        model = estimator()
        labels = frame[f"hit_{suffix}"].astype(int).to_numpy()
        model.fit(x[fit], labels[fit])
        frame[f"quality_{suffix}"] = model.predict_proba(x)[:, 1]
        prevalence[str(target)] = round(100 * float(labels[fit].mean()), 4)

    selection_rows = frame[frame.year.eq(2022)
                           & frame.quality_0_5.ge(BASE_THRESHOLD)]
    grids = {}
    for target in (1.0, 1.5, 2.0):
        column = f"quality_{str(target).replace('.', '_')}"
        grids[target] = sorted(set(float(selection_rows[column].quantile(q))
                                   for q in QUANTILES))
    frontier = []
    best = None
    for values in itertools.product(*(grids[target] for target in (1.0, 1.5, 2.0))):
        thresholds = dict(zip((1.0, 1.5, 2.0), values))
        candidate = routed(frame, thresholds)
        score = evaluate(candidate, (2022,))
        item = {"thresholds": {str(k): v for k, v in thresholds.items()},
                "selection_2022": score}
        frontier.append(item)
        if not passes(score, selection=True):
            continue
        rank = (score["return_pct"] - 2 * score["dd_pct"]
                + .2 * score["minimum_month_pct"]
                + score["average_selected_target_pct"])
        if best is None or rank > best[0]:
            best = (rank, thresholds, candidate, score)
    frontier.sort(key=lambda item: (
        item["selection_2022"]["return_pct"]
        - 2 * item["selection_2022"]["dd_pct"]), reverse=True)

    validation = test_2024 = later = None
    status = "REJECTED_SELECTION"
    frozen_public = None
    if best is not None:
        rank, thresholds, candidate, selection = best
        validation = evaluate(candidate, (2023,))
        frozen_public = {
            "rank": rank,
            "thresholds": {str(k): v for k, v in thresholds.items()},
            "selection_2022": selection,
        }
        if passes(validation):
            test_2024 = evaluate(candidate, (2024,))
            if passes(test_2024):
                later = {str(year): evaluate(candidate, (year,))
                         for year in (2025, 2026)}
                status = ("CASCADE_RESEARCH_CANDIDATE"
                          if all(passes(value) for value in later.values())
                          else "REJECTED_LATER")
                candidate.to_pickle(ROWS_OUT)
            else:
                status = "REJECTED_2024"
        else:
            status = "REJECTED_2023"
    report = {
        "status": status,
        "protocol": {
            "venue": "OKX global USDT swap", "bar": "1H",
            "fit": [2020, 2021], "selection": 2022, "validation": 2023,
            "untouched_test": 2024, "hidden_if_test_passes": [2025, 2026],
            "targets_pct": list(TARGETS), "hold_bars": HOLD,
            "stop_pct": STOP_PCT, "entry": "next hourly market open",
            "money": "3% current equity margin, fixed 10x gross",
            "base_threshold": BASE_THRESHOLD, "model": MODEL_CONFIG,
        },
        "fit_target_prevalence_pct": prevalence,
        "selection_combinations": len(frontier),
        "frontier_top20": frontier[:20], "frozen_before_2023": frozen_public,
        "validation_2023": validation, "test_2024": test_2024, "later": later,
        "sources": {**sources, str(Path(__file__)): digest(Path(__file__))},
    }
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({key: report[key] for key in (
        "status", "fit_target_prevalence_pct", "selection_combinations",
        "frozen_before_2023", "validation_2023", "test_2024", "later")},
        indent=2), flush=True)


if __name__ == "__main__":
    main()
