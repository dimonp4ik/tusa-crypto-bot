"""Profit-first ATR-scaled runner for frozen causal OKX one-hour entries.

The runner target is fixed at entry as a multiple of the already-known ATR,
clamped to a literal 2-8% move.  One profile is selected on 2022, then frozen
before 2023 validation and the untouched 2024 test.  Win rate is diagnostic.
"""
from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path

import numba as nb
import numpy as np
import pandas as pd

import crypto_5m_regime_router_literal_oct6_audit as portfolio
import crypto_okx_swap_1h_regime_oct7_audit as base
from crypto_profit_first_runner_oct16_audit import enrich


ROOT = Path("reports/audit_2026_09_08")
SOURCE_REPORT = ROOT / "crypto_okx_swap_1h_regime_oct7.json"
SOURCE_ROWS = ROOT / "crypto_okx_swap_1h_regime_oct7.pkl"
OUT = ROOT / "crypto_profit_first_atr_runner_oct16.json"
ROWS_OUT = ROOT / "crypto_profit_first_atr_runner_oct16.pkl"
PARTIALS = (.10, .25)
ATR_MULTIPLIERS = (2.0, 3.0, 4.0, 5.0)
HOLDS = (96, 168)
MIN_TARGET_PCT = 2.0
MAX_TARGET_PCT = 8.0


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def runner_targets(frame: pd.DataFrame, atr_multiplier: float) -> np.ndarray:
    return np.clip(
        frame.atr14_pct.to_numpy(float) * 100 * atr_multiplier,
        MIN_TARGET_PCT, MAX_TARGET_PCT)


@nb.njit(cache=True)
def resolve(open_, high, low, close, times, entries, directions, targets,
            partial, hold):
    n = len(entries)
    signed = np.empty(n, dtype=np.float64)
    exits = np.empty(n, dtype=np.int64)
    first_hits = np.zeros(n, dtype=np.int8)
    runner_hits = np.zeros(n, dtype=np.int8)
    reasons = np.empty(n, dtype=np.int8)
    first_move = .005
    stop_move = .01
    for row in range(n):
        first = entries[row]
        direction = directions[row]
        entry = open_[first]
        runner_move = targets[row] / 100
        first_price = entry * (1 + direction * first_move)
        runner_price = entry * (1 + direction * runner_move)
        stop = entry * (1 - direction * stop_move)
        last = first
        end = min(len(times), first + hold)
        first_index = -1
        result = 0.0
        reason = 2
        for index in range(first, end):
            if index > first and times[index] - times[index - 1] != 3600:
                break
            last = index
            stop_hit = low[index] <= stop if direction > 0 else high[index] >= stop
            target_hit = high[index] >= first_price if direction > 0 else low[index] <= first_price
            if stop_hit:
                fill = min(open_[index], stop) if direction > 0 else max(open_[index], stop)
                result = direction * (fill / entry - 1)
                reason = 0
                break
            if target_hit:
                first_hits[row] = 1
                first_index = index
                result = partial * first_move
                reason = 3
                break
        if first_index >= 0:
            for index in range(first_index + 1, end):
                if times[index] - times[index - 1] != 3600:
                    break
                last = index
                breakeven = low[index] <= entry if direction > 0 else high[index] >= entry
                runner_hit = high[index] >= runner_price if direction > 0 else low[index] <= runner_price
                if breakeven:
                    fill = min(open_[index], entry) if direction > 0 else max(open_[index], entry)
                    result = partial * first_move + (1 - partial) * direction * (fill / entry - 1)
                    reason = 4
                    break
                if runner_hit:
                    result = partial * first_move + (1 - partial) * runner_move
                    runner_hits[row] = 1
                    reason = 1
                    break
            else:
                runner_return = direction * (close[last] / entry - 1)
                result = partial * first_move + (1 - partial) * max(
                    0.0, min(runner_move, runner_return))
        elif reason == 2:
            result = direction * (close[last] / entry - 1)
            result = max(-stop_move, min(first_move, result))
        signed[row] = result
        exits[row] = times[last] + 3600
        reasons[row] = reason
    return signed, exits, first_hits, runner_hits, reasons


def add_outcomes(frame: pd.DataFrame, candles: dict, partial: float,
                 atr_multiplier: float, hold: int) -> pd.DataFrame:
    pieces = []
    for symbol, part in frame.groupby("symbol", sort=False):
        candle = candles[symbol]
        arrays = [np.asarray(candle[key], dtype=float)
                  for key in ("open", "high", "low", "close")]
        targets = runner_targets(part, atr_multiplier)
        values, exits, hits, runner_hits, reasons = resolve(
            *arrays, np.asarray(candle["time"], dtype=np.int64),
            part.entry_index.to_numpy(np.int64),
            part.direction.to_numpy(np.int8), targets, partial, hold)
        found = part.drop(columns=[name for name in (
            "signed_return", "exit_time", "target_hit", "runner_hit",
            "exit_reason") if name in part]).copy()
        found["runner_target_pct"] = targets
        found["signed_return"] = values
        found["exit_time"] = exits
        found["target_hit"] = hits.astype(bool)
        found["runner_hit"] = runner_hits.astype(bool)
        found["exit_reason"] = reasons
        pieces.append(found)
    return pd.concat(pieces, ignore_index=True)


def evaluate(frame: pd.DataFrame, years: tuple[int, ...]) -> dict:
    value = enrich(portfolio.evaluate(frame, years, 0))
    selected = portfolio.portfolio_rows(frame, years, 0)
    targets = [float(row["runner_target_pct"]) for row in selected]
    value["mean_runner_target_pct"] = round(float(np.mean(targets)), 4) if targets else 0.0
    value["runner_hit_rate"] = round(
        100 * sum(bool(row["runner_hit"]) for row in selected) / len(selected), 4
    ) if selected else 0.0
    return value


def selection_ok(value: dict) -> bool:
    return bool(value["n"] >= 700 and value["return_pct"] > 0
                and value["dd_pct"] <= 15
                and value["trades_per_calendar_day"] >= 2)


def validation_ok(value: dict) -> bool:
    return bool(value["n"] >= 500 and value["return_pct"] > 0
                and value["dd_pct"] <= 15
                and value["trades_per_calendar_day"] >= 2)


def profile_score(value: dict) -> float:
    return round(value["equivalent_monthly_return_pct"]
                 - .10 * value["dd_pct"]
                 - .20 * value["negative_months"], 8)


def objective_ok(value: dict) -> bool:
    return bool(value["months"] > 0 and value["dd_pct"] <= 10
                and value["months_ge_30pct"] == value["months"])


def public(item: dict | None) -> dict | None:
    if item is None:
        return None
    return {key: value for key, value in item.items()
            if key not in {"_rows"}}


def main() -> None:
    source = json.loads(SOURCE_REPORT.read_text(encoding="utf-8"))
    if source.get("status") != "RESEARCH_CANDIDATE_AWAITING_DIRECT_FORWARD":
        raise ValueError("source OKX one-hour entry candidate is not frozen")
    candles, sources = base.load()
    entries = pd.read_pickle(SOURCE_ROWS)
    candidates = []
    for partial, multiplier, hold in itertools.product(
            PARTIALS, ATR_MULTIPLIERS, HOLDS):
        rows = add_outcomes(entries, candles, partial, multiplier, hold)
        selection = evaluate(rows, (2022,))
        candidates.append({
            "partial_fraction_at_0_5pct": partial,
            "atr_multiplier": multiplier,
            "hold_bars": hold,
            "selection_2022": selection,
            "selection_pass": selection_ok(selection),
            "score": profile_score(selection),
            "_rows": rows,
        })
    candidates.sort(key=lambda item: (
        item["selection_pass"], item["score"]), reverse=True)
    frozen = candidates[0] if candidates and candidates[0]["selection_pass"] else None
    validation = test = confirmation = checks = None
    status = "REJECTED_SELECTION"
    if frozen:
        validation = evaluate(frozen["_rows"], (2023,))
        status = "REJECTED_VALIDATION"
        if validation_ok(validation):
            test = evaluate(frozen["_rows"], (2024,))
            status = "REJECTED_TEST"
            if validation_ok(test):
                confirmation = {
                    str(year): evaluate(frozen["_rows"], (year,))
                    for year in (2025, 2026)}
                checks = {
                    "2022": objective_ok(frozen["selection_2022"]),
                    "2023": objective_ok(validation),
                    "2024": objective_ok(test),
                    **{str(year): objective_ok(confirmation[str(year)])
                       for year in (2025, 2026)},
                }
                status = ("ATR_RUNNER_RESEARCH_CANDIDATE"
                          if all(checks.values()) else "REJECTED_OBJECTIVE")
            frozen["_rows"].to_pickle(ROWS_OUT)

    report = {
        "status": status,
        "ready_for_live_orders": False,
        "protocol": {
            "priority": "account return first; win rate diagnostic only",
            "parent_entry_model_fit": [2020, 2021],
            "profile_selection": 2022,
            "frozen_validation": 2023,
            "untouched_test": 2024,
            "confirmation_only": [2025, 2026],
            "target": "entry ATR multiple clamped to literal 2-8%",
            "profiles": {
                "partials": list(PARTIALS),
                "atr_multipliers": list(ATR_MULTIPLIERS),
                "holds": list(HOLDS),
            },
            "initial_stop_pct": 1.0,
            "entry": "frozen causal signal; next 1h market open",
            "orders": "market only",
            "money": "3% current-equity margin, fixed 10x gross",
            "objective": ">=30% account return in every month; DD <=10%",
        },
        "selection_frontier": [public(item) for item in candidates],
        "frozen_before_validation": public(frozen),
        "validation_2023": validation,
        "test_2024": test,
        "confirmation_2025_2026": confirmation,
        "objective_checks": checks,
        "sources": {
            **sources,
            str(SOURCE_REPORT): digest(SOURCE_REPORT),
            str(SOURCE_ROWS): digest(SOURCE_ROWS),
            str(Path(__file__)): digest(Path(__file__)),
        },
        "limitations": [
            "ATR scaling adapts payoff distance but does not create entry edge.",
            "Historical OKX global results require EU X-Perp forward transfer.",
        ],
    }
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({key: report[key] for key in (
        "status", "frozen_before_validation", "validation_2023",
        "test_2024", "confirmation_2025_2026", "objective_checks")},
        indent=2), flush=True)


if __name__ == "__main__":
    main()
