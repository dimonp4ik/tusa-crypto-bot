"""Causal stop-attribution audit for the frozen OKX one-hour target router.

All explanatory fields are computed from candles closed before the market
entry.  Guard selection uses 2022 only, validation uses 2023, and 2024 is
evaluated only after validation passes.  The already-published 2025-2026
period is confirmation evidence, never an untouched holdout.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import pickle
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

import crypto_okx_swap_1h_target_cascade_oct7_audit as cascade
from src import crypto_okx_1h_shadow_router as router


ROOT = Path("reports/audit_2026_09_08")
MANIFEST = ROOT / "crypto_okx_swap_1h_expanded_oct11" / "manifest.json"
ROWS = ROOT / "crypto_okx_swap_1h_target_cascade_oct7.pkl"
OUT = ROOT / "crypto_stop_cause_oct11.json"
STOP_PCT = 1.0

CAUSES = (
    "stop_inside_hourly_atr",
    "higher_timeframe_conflict",
    "btc_direction_conflict",
    "market_breadth_conflict",
    "opposing_level_before_target",
    "no_protective_level_before_stop",
    "failed_directional_breakout",
    "opposing_order_block",
    "abnormal_quote_volume",
    "crowded_same_hour",
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def resolve_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else Path.cwd() / path


def load_candles() -> tuple[dict, dict]:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    candles = {}
    sources = {str(MANIFEST): digest(MANIFEST)}
    for symbol in sorted(router.SUPPORTED_SYMBOLS):
        item = manifest["symbols"][symbol]
        path = resolve_path(item["path"])
        raw = path.read_bytes()
        found = hashlib.sha256(raw).hexdigest()
        if found != item["sha256"]:
            raise ValueError(f"source hash mismatch: {symbol}")
        candle = pickle.loads(raw)
        times = np.asarray(candle["time"], dtype=np.int64)
        if len(times) < 200 or np.any(np.diff(times) != 3600):
            raise ValueError(f"non-contiguous candles: {symbol}")
        candles[symbol] = candle
        sources[str(path)] = found
    return candles, sources


def valid_order_block_flags(
        open_: np.ndarray, high: np.ndarray, low: np.ndarray,
        close: np.ndarray, entry_index: int, direction: int,
        entry_price: float, target_pct: float) -> tuple[bool, bool]:
    """Return (opposing block in target path, supportive block near stop).

    A supply block is a bullish candle followed within two bars by a >=1%
    bearish displacement that breaks its low.  Demand is the mirror image.
    A block is invalid once a later close crosses its far edge.  Formation and
    invalidation are both restricted to candles closed before entry.
    """
    signal = entry_index - 1
    start = max(1, signal - 72)
    opposing = False
    supportive = False
    for index in range(start, signal - 1):
        end = index + 2
        if end > signal:
            break
        bullish = close[index] > open_[index]
        bearish = close[index] < open_[index]
        down_displacement = close[end] / close[index] - 1 <= -.01
        up_displacement = close[end] / close[index] - 1 >= .01
        supply = bullish and down_displacement and low[end] < low[index]
        demand = bearish and up_displacement and high[end] > high[index]
        if supply:
            valid = np.max(close[end:signal + 1]) <= high[index] * 1.001
            near = min(open_[index], close[index])
            distance = 100 * (near / entry_price - 1)
            if valid and direction > 0 and 0 <= distance <= target_pct:
                opposing = True
            distance = 100 * (high[index] / entry_price - 1)
            if valid and direction < 0 and 0 <= distance <= 1.2 * STOP_PCT:
                supportive = True
        if demand:
            valid = np.min(close[end:signal + 1]) >= low[index] * .999
            near = max(open_[index], close[index])
            distance = 100 * (entry_price / near - 1)
            if valid and direction < 0 and 0 <= distance <= target_pct:
                opposing = True
            distance = 100 * (entry_price / low[index] - 1)
            if valid and direction > 0 and 0 <= distance <= 1.2 * STOP_PCT:
                supportive = True
    return opposing, supportive


def enrich(frame: pd.DataFrame, candles: dict) -> pd.DataFrame:
    out = frame.copy()
    out["raw_same_hour"] = out.groupby("entry_time")["entry_time"].transform("size")
    values: dict[str, list] = {
        "opposing_distance_24_pct": [],
        "opposing_distance_72_pct": [],
        "protective_distance_24_pct": [],
        "failed_breakout_flag": [],
        "opposing_ob_flag": [],
        "supportive_ob_flag": [],
    }
    ordered_indices = []
    for symbol, part in out.groupby("symbol", sort=False):
        candle = candles[symbol]
        open_ = np.asarray(candle["open"], dtype=float)
        high = np.asarray(candle["high"], dtype=float)
        low = np.asarray(candle["low"], dtype=float)
        close = np.asarray(candle["close"], dtype=float)
        times = np.asarray(candle["time"], dtype=np.int64)
        for row_index, row in part.iterrows():
            entry_index = int(row.entry_index)
            if entry_index <= 168 or entry_index >= len(times):
                raise ValueError(f"invalid entry index: {symbol} {entry_index}")
            if int(times[entry_index]) != int(row.entry_time):
                raise ValueError(f"entry-time mismatch: {symbol} {row.entry_time}")
            signal = entry_index - 1
            entry = float(open_[entry_index])
            direction = int(row.direction)
            distances = {}
            for window in (24, 72):
                start = signal - window + 1
                prior_high = float(np.max(high[start:signal + 1]))
                prior_low = float(np.min(low[start:signal + 1]))
                opposing = (100 * (prior_high / entry - 1) if direction > 0
                             else 100 * (entry / prior_low - 1))
                distances[window] = opposing
                if window == 24:
                    protective = (100 * (entry / prior_low - 1)
                                  if direction > 0
                                  else 100 * (prior_high / entry - 1))
            prior_high = float(np.max(high[signal - 24:signal]))
            prior_low = float(np.min(low[signal - 24:signal]))
            failed_breakout = (
                high[signal] > prior_high and close[signal] < prior_high
                if direction > 0 else
                low[signal] < prior_low and close[signal] > prior_low
            )
            opposing_ob, supportive_ob = valid_order_block_flags(
                open_, high, low, close, entry_index, direction, entry,
                float(row.selected_target_pct))
            ordered_indices.append(row_index)
            values["opposing_distance_24_pct"].append(distances[24])
            values["opposing_distance_72_pct"].append(distances[72])
            values["protective_distance_24_pct"].append(protective)
            values["failed_breakout_flag"].append(bool(failed_breakout))
            values["opposing_ob_flag"].append(bool(opposing_ob))
            values["supportive_ob_flag"].append(bool(supportive_ob))
    derived = pd.DataFrame(values, index=ordered_indices)
    for column in derived:
        out.loc[derived.index, column] = derived[column]

    direction = out.direction.astype(float)
    breadth_alignment = out.breadth24.where(direction > 0, 1 - out.breadth24)
    out["stop_inside_hourly_atr"] = out.atr14_pct * 100 >= STOP_PCT
    out["higher_timeframe_conflict"] = direction * out.ret168 <= 0
    out["btc_direction_conflict"] = direction * out.btc_ret24 <= -.005
    out["market_breadth_conflict"] = breadth_alignment < .375
    out["opposing_level_before_target"] = (
        out.opposing_distance_72_pct.ge(0)
        & out.opposing_distance_72_pct.le(out.selected_target_pct))
    out["no_protective_level_before_stop"] = (
        out.protective_distance_24_pct > 1.5 * STOP_PCT)
    out["failed_directional_breakout"] = out.failed_breakout_flag.astype(bool)
    out["opposing_order_block"] = out.opposing_ob_flag.astype(bool)
    out["abnormal_quote_volume"] = (
        (out.volume_ratio < .6) | (out.volume_ratio > 3.0))
    out["crowded_same_hour"] = out.raw_same_hour >= 3
    return out


def accepted_frame(frame: pd.DataFrame, years: tuple[int, ...]) -> pd.DataFrame:
    rows = cascade.portfolio.portfolio_rows(frame, years, .5)
    return pd.DataFrame(rows)


def metrics(frame: pd.DataFrame, years: tuple[int, ...]) -> dict:
    return cascade.evaluate(frame, years)


def attribution(frame: pd.DataFrame, year: int) -> dict:
    accepted = accepted_frame(frame, (year,))
    stopped = accepted.exit_reason.eq(0)
    baseline = float(stopped.mean()) if len(accepted) else 0.0
    causes = {}
    for cause in CAUSES:
        flagged = accepted[cause].astype(bool)
        count = int(flagged.sum())
        flagged_stops = int((flagged & stopped).sum())
        precision = flagged_stops / count if count else 0.0
        causes[cause] = {
            "flagged_trades": count,
            "flagged_stops": flagged_stops,
            "stop_capture_pct": round(100 * flagged_stops / max(1, int(stopped.sum())), 4),
            "stop_rate_pct": round(100 * precision, 4),
            "stop_rate_lift": round(precision / baseline, 4) if baseline else None,
        }
    cause_count = accepted[list(CAUSES)].astype(bool).sum(axis=1)
    overlap = Counter(int(value) for value in cause_count[stopped].tolist())
    return {
        "accepted_trades": len(accepted),
        "stops": int(stopped.sum()),
        "baseline_stop_rate_pct": round(100 * baseline, 4),
        "stop_cause_overlap": {str(key): value for key, value in sorted(overlap.items())},
        "causes": causes,
    }


def candidate_masks(frame: pd.DataFrame) -> list[tuple[str, pd.Series]]:
    masks = []
    for cause in CAUSES:
        masks.append((cause, frame[cause].astype(bool)))
    for size in (2, 3):
        for names in itertools.combinations(CAUSES, size):
            mask = frame[names[0]].astype(bool)
            for name in names[1:]:
                mask &= frame[name].astype(bool)
            masks.append((" AND ".join(names), mask))
    return masks


def dominates(candidate: dict, baseline: dict) -> bool:
    return bool(
        candidate["n"] >= 700
        and candidate["trades_per_calendar_day"] >= 2.0
        and candidate["return_pct"] >= baseline["return_pct"]
        and candidate["literal_target_hit_rate"] >= baseline["literal_target_hit_rate"]
        and candidate["dd_pct"] <= baseline["dd_pct"]
        and candidate["negative_months"] <= baseline["negative_months"]
        and candidate["minimum_month_pct"] >= baseline["minimum_month_pct"]
    )


def validation_pass(candidate: dict, baseline: dict) -> bool:
    return bool(
        candidate["n"] >= 500
        and candidate["trades_per_calendar_day"] >= 2.0
        and candidate["return_pct"] >= baseline["return_pct"]
        and candidate["literal_target_hit_rate"] >= baseline["literal_target_hit_rate"]
        and candidate["dd_pct"] <= baseline["dd_pct"]
        and candidate["negative_months"] <= baseline["negative_months"]
        and candidate["minimum_month_pct"] >= baseline["minimum_month_pct"]
    )


def objective_pass(value: dict) -> bool:
    return bool(
        75 <= value["literal_target_hit_rate"] <= 95
        and value["trades_per_calendar_day"] >= 3
        and value["negative_months"] == 0
        and value["months_ge_30pct"] == value["months"]
    )


def main() -> None:
    candles, sources = load_candles()
    raw = pd.read_pickle(ROWS)
    required = {"symbol", "entry_index", "entry_time", "direction",
                "selected_target_pct", "signed_return", "exit_time",
                "target_hit", "exit_reason"}
    if not required.issubset(raw.columns):
        raise ValueError(f"frozen rows missing: {sorted(required - set(raw.columns))}")
    frame = enrich(raw, candles)

    baseline_2022 = metrics(frame, (2022,))
    frontier = []
    eligible = []
    for expression, blocked in candidate_masks(frame):
        candidate = frame.loc[~blocked].copy()
        score = metrics(candidate, (2022,))
        item = {"block_when": expression, "selection_2022": score,
                "dominates_baseline": dominates(score, baseline_2022)}
        frontier.append(item)
        if item["dominates_baseline"]:
            eligible.append((
                (baseline_2022["negative_months"] - score["negative_months"],
                 score["minimum_month_pct"], score["return_pct"],
                 score["literal_target_hit_rate"], -score["dd_pct"]),
                expression, candidate, score))
    frontier.sort(key=lambda item: (
        item["dominates_baseline"],
        -item["selection_2022"]["negative_months"],
        item["selection_2022"]["minimum_month_pct"],
        item["selection_2022"]["return_pct"]), reverse=True)

    frozen = validation = test_2024 = confirmation = None
    status = "REJECTED_SELECTION"
    if eligible:
        _, expression, candidate, selection = max(eligible, key=lambda item: item[0])
        baseline_2023 = metrics(frame, (2023,))
        validation = metrics(candidate, (2023,))
        passed_validation = validation_pass(validation, baseline_2023)
        frozen = {
            "block_when": expression,
            "selection_2022": selection,
            "selection_baseline_2022": baseline_2022,
            "eligible_candidates": len(eligible),
        }
        status = "REJECTED_VALIDATION"
        if passed_validation:
            baseline_2024 = metrics(frame, (2024,))
            test_2024 = metrics(candidate, (2024,))
            if validation_pass(test_2024, baseline_2024):
                confirmation = {str(year): metrics(candidate, (year,))
                                for year in (2025, 2026)}
                status = ("STOP_GUARD_RESEARCH_CANDIDATE"
                          if all(objective_pass(value)
                                 for value in confirmation.values())
                          else "REJECTED_OBJECTIVE")
            else:
                status = "REJECTED_TEST_2024"

    report = {
        "status": status,
        "ready_for_live_orders": False,
        "protocol": {
            "input": "frozen target-cascade rows; no entry or target retuning",
            "all_features_available_before_entry": True,
            "selection": 2022,
            "validation": 2023,
            "conditional_test": 2024,
            "published_confirmation_only": [2025, 2026],
            "market_entry": "next hourly open",
            "stop_pct": STOP_PCT,
            "targets_pct": [.5, 1.0, 1.5, 2.0],
            "money": "3% current equity margin, fixed 10x gross",
            "fees": "excluded by user instruction",
        },
        "cause_definitions": {
            "stop_inside_hourly_atr": "14h ATR is at least the fixed 1% stop",
            "higher_timeframe_conflict": "trade direction opposes prior 168h return",
            "btc_direction_conflict": "trade direction opposes BTC prior 24h return by at least 0.5%",
            "market_breadth_conflict": "fewer than 37.5% of tracked symbols align over prior 24h",
            "opposing_level_before_target": "prior 72h high/low lies between entry and selected target",
            "no_protective_level_before_stop": "prior 24h low/high is more than 1.5% behind entry",
            "failed_directional_breakout": "last closed candle pierced prior 24h extreme but closed back inside",
            "opposing_order_block": "uninvalidated 1% two-bar displacement block lies in target path",
            "abnormal_quote_volume": "last-bar quote volume is below 0.6x or above 3x trailing median",
            "crowded_same_hour": "at least three raw frozen entries share the entry hour",
        },
        "attribution": {str(year): attribution(frame, year)
                        for year in (2022, 2023, 2024, 2025, 2026)},
        "selection_baseline_2022": baseline_2022,
        "evaluated_guards": len(frontier),
        "selection_frontier_top20": frontier[:20],
        "frozen_before_validation": frozen,
        "validation_2023": validation,
        "test_2024": test_2024,
        "confirmation_2025_2026": confirmation,
        "sources": {
            **sources,
            str(ROWS): digest(ROWS),
            str(Path(cascade.__file__).name): digest(Path(cascade.__file__)),
            str(Path(router.__file__)): digest(Path(router.__file__)),
            str(Path(__file__)): digest(Path(__file__)),
        },
        "limitations": [
            "OHLCV sweeps, levels and quote volume are liquidity proxies, not historical order-book depth.",
            "Order blocks use one fixed reproducible definition; discretionary chart labels were not used.",
            "Cause flags may overlap and describe pre-entry risk, not prove a unique causal mechanism.",
            "The objective requires independent forward outcomes even if a historical guard passes.",
        ],
    }
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({
        "status": status,
        "evaluated_guards": len(frontier),
        "frozen_before_validation": frozen,
        "validation_2023": validation,
        "test_2024": test_2024,
        "confirmation_2025_2026": confirmation,
        "strongest_stop_lifts": {
            str(year): sorted(
                report["attribution"][str(year)]["causes"].items(),
                key=lambda item: item[1]["stop_rate_lift"] or 0,
                reverse=True)[:4]
            for year in (2022, 2023, 2024, 2025, 2026)
        },
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
