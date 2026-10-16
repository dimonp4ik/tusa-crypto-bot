"""Select literal 1-2% crypto moves with a profit-first causal classifier.

The entry families and features use only completed one-hour candles.  Each
profile has a literal target, a one-percent stop and a fixed maximum hold.  A
classifier fitted on 2020-2021 ranks the probability of reaching that larger
target first.  Profile and numeric score cutoff are selected on 2022, then
frozen before the 2023-2026 checks.  Win rate is diagnostic only.
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

import crypto_5m_regime_router_literal_oct6_audit as portfolio
import crypto_okx_swap_1h_regime_oct7_audit as base
from crypto_profit_first_runner_oct16_audit import enrich


ROOT = Path("reports/audit_2026_09_08")
OUT = ROOT / "crypto_profit_first_literal_runner_oct16.json"
ROWS_OUT = ROOT / "crypto_profit_first_literal_runner_oct16.pkl"
TARGETS_PCT = (1.0, 1.5, 2.0)
HOLDS = (24, 48, 96)
SELECTION_QUANTILES = (.90, .925, .95, .96, .97, .98)
FIT_YEARS = (2020, 2021)
SELECTION_YEAR = 2022
VALIDATION_YEAR = 2023
TEST_YEAR = 2024
CONFIRMATION_YEARS = (2025, 2026)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@nb.njit(cache=True)
def resolve(open_, high, low, close, times, entries, directions, target_pct,
            hold):
    count = len(entries)
    signed = np.empty(count, dtype=np.float64)
    exits = np.empty(count, dtype=np.int64)
    hits = np.zeros(count, dtype=np.int8)
    reasons = np.empty(count, dtype=np.int8)
    target_move = target_pct / 100.0
    stop_move = .01
    for row in range(count):
        first = entries[row]
        direction = directions[row]
        entry = open_[first]
        target = entry * (1.0 + direction * target_move)
        stop = entry * (1.0 - direction * stop_move)
        last = first
        result = 0.0
        reason = 2
        resolved = False
        end = min(len(times), first + hold)
        for index in range(first, end):
            if index > first and times[index] - times[index - 1] != 3600:
                break
            last = index
            stop_hit = low[index] <= stop if direction > 0 else high[index] >= stop
            target_hit = high[index] >= target if direction > 0 else low[index] <= target
            if stop_hit:  # Pessimistic when both levels occur inside one bar.
                fill = min(open_[index], stop) if direction > 0 else max(open_[index], stop)
                result = direction * (fill / entry - 1.0)
                reason = 0
                resolved = True
                break
            if target_hit:
                result = target_move
                hits[row] = 1
                reason = 1
                resolved = True
                break
        if not resolved:
            result = direction * (close[last] / entry - 1.0)
            result = max(-stop_move, min(target_move, result))
        signed[row] = result
        exits[row] = times[last] + 3600
        reasons[row] = reason
    return signed, exits, hits, reasons


def add_outcomes(frame: pd.DataFrame, candles: dict, target_pct: float,
                 hold: int) -> pd.DataFrame:
    pieces = []
    for symbol, part in frame.groupby("symbol", sort=False):
        candle = candles[symbol]
        arrays = [np.asarray(candle[name], dtype=float)
                  for name in ("open", "high", "low", "close")]
        values, exits, hits, reasons = resolve(
            *arrays, np.asarray(candle["time"], dtype=np.int64),
            part.entry_index.to_numpy(np.int64),
            part.direction.to_numpy(np.int8), target_pct, hold)
        found = part.copy()
        found["signed_return"] = values
        found["exit_time"] = exits
        found["target_hit"] = hits.astype(bool)
        found["exit_reason"] = reasons
        found["literal_target_pct"] = target_pct
        found["hold_bars"] = hold
        pieces.append(found)
    return pd.concat(pieces, ignore_index=True)


def model() -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        loss="log_loss", max_iter=160, learning_rate=.05,
        max_leaf_nodes=31, min_samples_leaf=250,
        l2_regularization=5.0, random_state=20261016)


def evaluate(frame: pd.DataFrame, years: tuple[int, ...], threshold: float) -> dict:
    return enrich(portfolio.evaluate(frame, years, threshold))


def selection_ok(value: dict) -> bool:
    return bool(value["return_pct"] > 0 and value["dd_pct"] <= 15
                and value["trades_per_calendar_day"] >= 3)


def profile_score(value: dict) -> float:
    return round(value["equivalent_monthly_return_pct"]
                 - .10 * value["dd_pct"]
                 - .20 * value["negative_months"], 8)


def objective_ok(value: dict) -> bool:
    return bool(value["months"] > 0 and value["dd_pct"] <= 10
                and value["trades_per_calendar_day"] >= 3
                and value["months_ge_30pct"] == value["months"])


def public(item: dict) -> dict:
    return {key: value for key, value in item.items()
            if key not in {"_frame", "_model"}}


def main() -> None:
    candles, sources = base.load()
    features = base.build_features(candles)
    proposals = base.proposals(features)
    candidates = []
    for target_pct, hold in itertools.product(TARGETS_PCT, HOLDS):
        frame = add_outcomes(proposals, candles, target_pct, hold)
        train = frame[frame.year.isin(FIT_YEARS)].copy()
        columns = base.schema(train)
        estimator = model()
        estimator.fit(base.matrix(train, columns),
                      train.target_hit.to_numpy(np.int8))
        frame["quality"] = estimator.predict_proba(
            base.matrix(frame, columns))[:, 1]
        selection_scores = frame.loc[
            frame.year.eq(SELECTION_YEAR), "quality"]
        for quantile in SELECTION_QUANTILES:
            threshold = float(selection_scores.quantile(quantile))
            value = evaluate(frame, (SELECTION_YEAR,), threshold)
            candidates.append({
                "target_pct": target_pct,
                "hold_bars": hold,
                "selection_quantile": quantile,
                "threshold": threshold,
                "selection_2022": value,
                "selection_pass": selection_ok(value),
                "score": profile_score(value),
                "_frame": frame,
                "_model": estimator,
            })
    candidates.sort(key=lambda item: (
        item["selection_pass"], item["score"]), reverse=True)
    frozen = candidates[0] if candidates and candidates[0]["selection_pass"] else None
    validation = test = confirmation = checks = None
    status = "REJECTED_SELECTION"
    if frozen:
        frame = frozen["_frame"]
        threshold = frozen["threshold"]
        validation = evaluate(frame, (VALIDATION_YEAR,), threshold)
        test = evaluate(frame, (TEST_YEAR,), threshold)
        confirmation = {str(year): evaluate(frame, (year,), threshold)
                        for year in CONFIRMATION_YEARS}
        yearly = {
            str(SELECTION_YEAR): frozen["selection_2022"],
            str(VALIDATION_YEAR): validation,
            str(TEST_YEAR): test,
            **confirmation,
        }
        checks = {year: objective_ok(value) for year, value in yearly.items()}
        status = ("LITERAL_RUNNER_RESEARCH_CANDIDATE"
                  if all(checks.values()) else "REJECTED_OBJECTIVE")
        accepted = portfolio.portfolio_rows(
            frame, (SELECTION_YEAR, VALIDATION_YEAR, TEST_YEAR,
                    *CONFIRMATION_YEARS), threshold)
        pd.DataFrame(accepted).to_pickle(ROWS_OUT)
    report = {
        "status": status,
        "ready_for_live_orders": False,
        "protocol": {
            "priority": "account return first; win rate diagnostic only",
            "model_fit": list(FIT_YEARS),
            "profile_and_cutoff_selection": SELECTION_YEAR,
            "frozen_validation": VALIDATION_YEAR,
            "untouched_test": TEST_YEAR,
            "confirmation_only": list(CONFIRMATION_YEARS),
            "literal_targets_pct": list(TARGETS_PCT),
            "maximum_holds_hours": list(HOLDS),
            "initial_stop_pct": 1.0,
            "entry": "next one-hour market open",
            "orders": "market only",
            "money": "3% current-equity margin, fixed 10x gross",
            "objective": ">=30% account return in every month; >=3 trades/day; DD <=10%",
        },
        "feature_rows": len(features),
        "proposal_rows": len(proposals),
        "candidate_count": len(candidates),
        "top_selection": [public(item) for item in candidates[:15]],
        "frozen_before_validation": public(frozen) if frozen else None,
        "validation_2023": validation,
        "test_2024": test,
        "confirmation_2025_2026": confirmation,
        "objective_checks": checks,
        "sources": {
            **sources,
            str(Path(base.__file__)): digest(Path(base.__file__)),
            str(Path(__file__)): digest(Path(__file__)),
        },
        "limitations": [
            "The selected numeric probability cutoff uses 2022 and is frozen afterward.",
            "A one-hour candle cannot reveal target/stop order, so collisions count as stops.",
            "Historical OKX global results require direct EU X-Perp shadow transfer.",
        ],
    }
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({
        "status": status,
        "candidate_count": len(candidates),
        "frozen": public(frozen) if frozen else None,
        "validation_2023": validation,
        "test_2024": test,
        "confirmation_2025_2026": confirmation,
        "objective_checks": checks,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
