"""Runner ladder for the frozen independent-expert crypto entries."""
from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path

import numba as nb
import numpy as np
import pandas as pd

import crypto_5m_regime_router_literal_oct6_audit as price
import crypto_futures_trade_flow_regime_oct7_audit as flow


ROOT = Path("reports/audit_2026_09_08")
FREEZE_REPORT = ROOT / "crypto_independent_expert_agreement_oct7.json"
FREEZE_ROWS = ROOT / "crypto_independent_expert_agreement_oct7.pkl"
OUT = ROOT / "crypto_expert_agreement_ladder_oct7.json"
ROWS_OUT = ROOT / "crypto_expert_agreement_ladder_oct7.pkl"
FIRST_TARGET = .5
INITIAL_STOP = 1.0
PARTIALS = (.25, .50, .75)
RUNNER_TARGETS = (1.0, 1.5, 2.0)
HOLDS = (24, 48, 72)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@nb.njit(cache=True)
def resolve_ladder(open_, high, low, close, times, entries, directions,
                   partial, runner_target, hold):
    n = len(entries)
    signed = np.empty(n, dtype=np.float64)
    exits = np.empty(n, dtype=np.int64)
    first_hits = np.zeros(n, dtype=np.int8)
    runner_hits = np.zeros(n, dtype=np.int8)
    reasons = np.empty(n, dtype=np.int8)
    first_move = FIRST_TARGET / 100
    stop_move = INITIAL_STOP / 100
    runner_move = runner_target / 100
    for row in range(n):
        first = entries[row]
        direction = directions[row]
        entry = open_[first]
        first_price = entry * (1 + direction * first_move)
        runner_price = entry * (1 + direction * runner_move)
        initial_stop = entry * (1 - direction * stop_move)
        last = first
        end = min(len(times), first + hold)
        first_index = -1
        result = 0.0
        reason = 2
        for index in range(first, end):
            if index > first and times[index] - times[index - 1] != 300:
                break
            last = index
            stop_hit = low[index] <= initial_stop if direction > 0 else high[index] >= initial_stop
            first_hit = high[index] >= first_price if direction > 0 else low[index] <= first_price
            if stop_hit:  # pessimistic collision ordering
                fill = min(open_[index], initial_stop) if direction > 0 else max(open_[index], initial_stop)
                result = direction * (fill / entry - 1)
                reason = 0
                break
            if first_hit:
                first_hits[row] = 1
                first_index = index
                result = partial * first_move
                reason = 3
                break
        if first_index >= 0:
            # Start managing the runner on the following candle.  This avoids
            # assuming a favourable intrabar path after the first target touch.
            for index in range(first_index + 1, end):
                if times[index] - times[index - 1] != 300:
                    break
                last = index
                breakeven_hit = low[index] <= entry if direction > 0 else high[index] >= entry
                runner_hit = high[index] >= runner_price if direction > 0 else low[index] <= runner_price
                if breakeven_hit:  # pessimistic if both touch
                    fill = min(open_[index], entry) if direction > 0 else max(open_[index], entry)
                    runner_return = direction * (fill / entry - 1)
                    result = partial * first_move + (1 - partial) * runner_return
                    reason = 4
                    break
                if runner_hit:
                    result = partial * first_move + (1 - partial) * runner_move
                    runner_hits[row] = 1
                    reason = 1
                    break
            else:
                runner_return = direction * (close[last] / entry - 1)
                runner_return = max(0.0, min(runner_move, runner_return))
                result = partial * first_move + (1 - partial) * runner_return
        elif reason == 2:
            result = direction * (close[last] / entry - 1)
            result = max(-stop_move, min(first_move, result))
        signed[row] = result
        exits[row] = times[last] + 300
        reasons[row] = reason
    return signed, exits, first_hits, runner_hits, reasons


def add_ladder(frame: pd.DataFrame, candles: dict, partial: float,
               runner_target: float, hold: int) -> pd.DataFrame:
    parts = []
    for symbol, part in frame.groupby("symbol", sort=False):
        candle = candles[symbol]
        arrays = [np.asarray(candle[key], dtype=float)
                  for key in ("open", "high", "low", "close")]
        values, exits, first_hits, runner_hits, reasons = resolve_ladder(
            *arrays, np.asarray(candle["time"], dtype=np.int64),
            part.entry_index.to_numpy(np.int64),
            part.direction.to_numpy(np.int8), partial, runner_target, hold)
        found = part.drop(columns=[name for name in (
            "signed_return", "exit_time", "target_hit", "exit_reason",
            "runner_hit") if name in part]).copy()
        found["signed_return"] = values
        found["exit_time"] = exits
        found["target_hit"] = first_hits.astype(bool)
        found["runner_hit"] = runner_hits.astype(bool)
        found["exit_reason"] = reasons
        parts.append(found)
    return pd.concat(parts, ignore_index=True)


def runner_rate(frame: pd.DataFrame, year: int) -> float:
    found = frame[frame.year.eq(year)]
    return round(100 * float(found.runner_hit.mean()), 4) if len(found) else 0.0


def acceptable(value: dict, validation: bool = False) -> bool:
    return bool(value["n"] >= (500 if validation else 700)
                and value["return_pct"] > 0
                and value["literal_target_hit_rate"] >= (66 if validation else 69)
                and value["dd_pct"] <= 7
                and value["trades_per_calendar_day"] >= (2.0 if validation else 2.5))


def main() -> None:
    freeze = json.loads(FREEZE_REPORT.read_text(encoding="utf-8"))
    if not freeze.get("frozen_before_2024"):
        raise ValueError("independent expert entry filter was not frozen")
    base = pd.read_pickle(FREEZE_ROWS)
    candles, candle_sources = flow.load()
    selection = []
    cache = {}
    for partial, runner_target, hold in itertools.product(
            PARTIALS, RUNNER_TARGETS, HOLDS):
        key = (partial, runner_target, hold)
        resolved = add_ladder(base, candles, *key)
        cache[key] = resolved
        value = price.evaluate(resolved, (2022,), 0)
        if acceptable(value):
            selection.append({
                "partial_fraction": partial, "runner_target_pct": runner_target,
                "hold_bars": hold, "selection_2022": value,
                "selection_runner_hit_rate": runner_rate(resolved, 2022),
                "_key": key,
            })
    validated = []
    diagnostics = []
    for item in selection:
        resolved = cache[item["_key"]]
        value = price.evaluate(resolved, (2023,), 0)
        public = {**item, "validation_2023": value,
                  "validation_runner_hit_rate": runner_rate(resolved, 2023)}
        public.pop("_key")
        diagnostics.append(public)
        if acceptable(value, validation=True):
            candidate = {**item, "validation_2023": value,
                         "validation_runner_hit_rate": runner_rate(resolved, 2023)}
            candidate["score"] = round(
                min(item["selection_2022"]["return_pct"], value["return_pct"])
                - 2 * max(item["selection_2022"]["dd_pct"], value["dd_pct"]), 8)
            validated.append(candidate)
    validated.sort(key=lambda item: item["score"], reverse=True)
    frozen = validated[0] if validated else None
    test_2024 = later = checks = None
    if frozen:
        resolved = cache[frozen["_key"]]
        test_2024 = {**price.evaluate(resolved, (2024,), 0),
                     "runner_hit_rate": runner_rate(resolved, 2024)}
        if acceptable(test_2024, validation=True):
            later = {str(year): {**price.evaluate(resolved, (year,), 0),
                                 "runner_hit_rate": runner_rate(resolved, year)}
                     for year in (2025, 2026)}
            checks = {year: acceptable(value, validation=True)
                      for year, value in later.items()}
        resolved.to_pickle(ROWS_OUT)
    public_frozen = None if frozen is None else {
        key: value for key, value in frozen.items() if key != "_key"}
    report = {
        "status": ("RESEARCH_CANDIDATE_AWAITING_DIRECT_FORWARD"
                   if checks and all(checks.values()) else "REJECTED"),
        "protocol": {
            "entry_filter": "frozen independent expert median agreement",
            "exit_selection": 2022, "validation_and_ranking": 2023,
            "untouched_test": 2024, "hidden_if_test_passes": [2025, 2026],
            "first_target_pct": FIRST_TARGET, "initial_stop_pct": INITIAL_STOP,
            "partial_fractions": list(PARTIALS),
            "runner_targets_pct": list(RUNNER_TARGETS),
            "hold_bars": list(HOLDS),
            "runner_stop_after_first_target": "entry price from next candle",
            "entry": "next 5m market open",
            "money": "3% equity margin, fixed 10x gross",
        },
        "qualified_2022": len(selection), "qualified_2023": len(validated),
        "validation_diagnostics": diagnostics,
        "frozen_before_2024": public_frozen, "test_2024": test_2024,
        "later": later, "later_checks": checks,
        "sources": {**candle_sources, str(FREEZE_REPORT): digest(FREEZE_REPORT),
                    str(FREEZE_ROWS): digest(FREEZE_ROWS),
                    str(Path(__file__)): digest(Path(__file__))},
        "limitations": [
            "First-target and runner fills are touch-based market-exit proxies.",
            "The runner starts on the next candle to avoid favourable intrabar path assumptions.",
            "Any historical pass remains shadow-only pending direct X-Perp forward outcomes.",
        ],
    }
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({key: report[key] for key in (
        "status", "qualified_2022", "qualified_2023",
        "frozen_before_2024", "test_2024", "later", "later_checks")},
        indent=2), flush=True)


if __name__ == "__main__":
    main()
