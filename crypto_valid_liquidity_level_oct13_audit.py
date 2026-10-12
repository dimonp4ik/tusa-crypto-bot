"""Audit valid opposing liquidity levels on the frozen hourly crypto router."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

import crypto_stop_cause_oct11_audit as cause


OUT = cause.ROOT / "crypto_valid_liquidity_level_oct13.json"
SELECTION = (2022,)
VALIDATION = (2023,)
TEST = (2024,)
CONFIRMATION = (2025, 2026)

RULES = (
    "reject_fresh_swing_barrier",
    "reject_repeated_liquidity_pool",
    "reject_confirmed_rejection_barrier",
    "reject_confirmed_ob_barrier",
    "reject_barrier_with_abnormal_volume",
    "reject_barrier_with_htf_conflict",
    "reject_barrier_in_crowded_hour",
    "reject_failed_breakout_at_barrier",
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def level_features(frame: pd.DataFrame, candles: dict) -> pd.DataFrame:
    """Use only candles ending before the market-entry candle."""
    out = frame.copy()
    rows = []
    for row_index, row in out.iterrows():
        candle = candles[str(row.symbol)]
        open_ = np.asarray(candle["open"], dtype=float)
        high = np.asarray(candle["high"], dtype=float)
        low = np.asarray(candle["low"], dtype=float)
        close = np.asarray(candle["close"], dtype=float)
        times = np.asarray(candle["time"], dtype=np.int64)
        entry_index = int(row.entry_index)
        if int(times[entry_index]) != int(row.entry_time):
            raise ValueError(f"entry mismatch {row.symbol} {row.entry_time}")
        signal = entry_index - 1
        direction = int(row.direction)
        entry = float(open_[entry_index])
        target_pct = float(row.selected_target_pct)
        start = signal - 72
        candidates = []
        for index in range(start + 2, signal - 1):
            if direction > 0:
                is_pivot = high[index] >= np.max(high[index - 2:index + 3])
                level = float(high[index])
                distance = 100 * (level / entry - 1)
                invalidated = np.any(close[index + 1:signal + 1] > level * 1.001)
                wick = high[index] - max(open_[index], close[index])
            else:
                is_pivot = low[index] <= np.min(low[index - 2:index + 3])
                level = float(low[index])
                distance = 100 * (entry / level - 1)
                invalidated = np.any(close[index + 1:signal + 1] < level * .999)
                wick = min(open_[index], close[index]) - low[index]
            if not is_pivot or not (0 <= distance <= target_pct) or invalidated:
                continue
            span = max(float(high[index] - low[index]), entry * 1e-8)
            wick_rejection = wick / span >= .35
            touch_indices = []
            last_touch = -10
            for probe in range(index, signal + 1):
                touched = (abs(float(high[probe]) / level - 1) <= .0015
                           if direction > 0 else
                           abs(float(low[probe]) / level - 1) <= .0015)
                if touched and probe - last_touch >= 3:
                    touch_indices.append(probe)
                    last_touch = probe
            last = touch_indices[-1]
            if direction > 0:
                reaction = 100 * (1 - float(np.min(close[last:signal + 1])) / level)
            else:
                reaction = 100 * (float(np.max(close[last:signal + 1])) / level - 1)
            candidates.append({
                "distance": distance, "age": signal - index,
                "touches": len(touch_indices), "wick": wick_rejection,
                "reaction": reaction,
            })
        if candidates:
            nearest = min(candidates, key=lambda value: value["distance"])
            strongest = max(candidates, key=lambda value: (
                value["touches"], value["reaction"], value["wick"]))
            fresh = True
            repeated = strongest["touches"] >= 2
            rejection = (strongest["wick"] and strongest["reaction"] >= .5)
            distance = nearest["distance"]
            touches = strongest["touches"]
            age = nearest["age"]
            reaction = strongest["reaction"]
        else:
            fresh = repeated = rejection = False
            distance = np.nan
            touches = age = 0
            reaction = 0.0
        rows.append({
            "row_index": row_index,
            "fresh_swing_barrier": fresh,
            "repeated_liquidity_pool": repeated,
            "confirmed_rejection_barrier": rejection,
            "nearest_barrier_distance_pct": distance,
            "barrier_touch_count": touches,
            "barrier_age_hours": age,
            "barrier_reaction_pct": reaction,
            "feature_last_candle_time": int(times[signal]),
        })
    features = pd.DataFrame(rows).set_index("row_index")
    for column in features:
        out.loc[features.index, column] = features[column]
    out["confirmed_ob_barrier"] = (
        out.opposing_order_block.astype(bool)
        & out.confirmed_rejection_barrier.astype(bool))
    if np.any(out.feature_last_candle_time.astype(np.int64)
              > out.entry_time.astype(np.int64) - 3600):
        raise ValueError("future candle used in level features")
    return out


def blocked(frame: pd.DataFrame, rule: str) -> pd.Series:
    barrier = frame.confirmed_rejection_barrier.astype(bool)
    if rule == "reject_fresh_swing_barrier":
        return frame.fresh_swing_barrier.astype(bool)
    if rule == "reject_repeated_liquidity_pool":
        return frame.repeated_liquidity_pool.astype(bool)
    if rule == "reject_confirmed_rejection_barrier":
        return barrier
    if rule == "reject_confirmed_ob_barrier":
        return frame.confirmed_ob_barrier.astype(bool)
    if rule == "reject_barrier_with_abnormal_volume":
        return barrier & frame.abnormal_quote_volume.astype(bool)
    if rule == "reject_barrier_with_htf_conflict":
        return barrier & frame.higher_timeframe_conflict.astype(bool)
    if rule == "reject_barrier_in_crowded_hour":
        return barrier & frame.crowded_same_hour.astype(bool)
    if rule == "reject_failed_breakout_at_barrier":
        return barrier & frame.failed_directional_breakout.astype(bool)
    raise KeyError(rule)


def stop_attribution(frame: pd.DataFrame, year: int) -> dict:
    accepted = cause.accepted_frame(frame, (year,))
    stop = accepted.exit_reason.eq(0)
    baseline = float(stop.mean())
    output = {"n": len(accepted), "baseline_stop_rate_pct": round(100 * baseline, 4)}
    for field in ("fresh_swing_barrier", "repeated_liquidity_pool",
                  "confirmed_rejection_barrier", "confirmed_ob_barrier"):
        flag = accepted[field].astype(bool)
        rate = float(stop[flag].mean()) if flag.any() else 0.0
        output[field] = {
            "flagged_n": int(flag.sum()), "stop_rate_pct": round(100 * rate, 4),
            "stop_rate_lift": round(rate / baseline, 4) if baseline else None,
        }
    return output


def dominates(candidate: dict, baseline: dict, minimum_n: int) -> bool:
    return bool(
        candidate["n"] >= minimum_n
        and candidate["trades_per_calendar_day"] >= 2.0
        and candidate["return_pct"] >= baseline["return_pct"]
        and candidate["literal_target_hit_rate"] >= baseline["literal_target_hit_rate"]
        and candidate["dd_pct"] <= baseline["dd_pct"]
        and candidate["negative_months"] <= baseline["negative_months"]
        and candidate["minimum_month_pct"] >= baseline["minimum_month_pct"]
        and candidate["max_loss_streak"] <= baseline["max_loss_streak"]
    )


def main() -> None:
    candles, sources = cause.load_candles()
    raw = pd.read_pickle(cause.ROWS)
    enriched = level_features(cause.enrich(raw, candles), candles)
    baseline = {"selection": cause.metrics(enriched, SELECTION),
                "validation": cause.metrics(enriched, VALIDATION),
                "test": cause.metrics(enriched, TEST),
                "confirmation": {str(year): cause.metrics(enriched, (year,))
                                 for year in CONFIRMATION}}
    frontier = []
    eligible = []
    for rule in RULES:
        candidate = enriched.loc[~blocked(enriched, rule)].copy()
        score = cause.metrics(candidate, SELECTION)
        accepted_removed = baseline["selection"]["n"] - score["n"]
        # A one-off blocked loser can improve a whole month's minimum by luck.
        # Demand material coverage before considering portfolio dominance.
        passed = (accepted_removed >= 20
                  and dominates(score, baseline["selection"], 700))
        frontier.append({"rule": rule, "selection": score,
                         "accepted_trades_removed": accepted_removed,
                         "selection_pass": passed})
        if passed:
            rank = (baseline["selection"]["negative_months"] - score["negative_months"],
                    score["minimum_month_pct"], score["return_pct"],
                    score["literal_target_hit_rate"], -score["dd_pct"])
            eligible.append((rank, rule, candidate, score))

    frozen = validation = test = confirmation = None
    status = "REJECTED_SELECTION"
    if eligible:
        _, rule, candidate, selection = max(eligible, key=lambda value: value[0])
        frozen = {"rule": rule, "selection": selection,
                  "selection_baseline": baseline["selection"],
                  "eligible_rules": len(eligible)}
        validation = cause.metrics(candidate, VALIDATION)
        status = "REJECTED_VALIDATION"
        if dominates(validation, baseline["validation"], 500):
            test = cause.metrics(candidate, TEST)
            status = "REJECTED_TEST"
            if dominates(test, baseline["test"], 500):
                confirmation = {str(year): cause.metrics(candidate, (year,))
                                for year in CONFIRMATION}
                status = "VALID_LEVEL_RESEARCH_CANDIDATE"

    report = {
        "status": status, "ready_for_live_orders": False,
        "protocol": {
            "input": "exact frozen OKX one-hour target router",
            "selection": list(SELECTION), "validation": list(VALIDATION),
            "test": list(TEST), "confirmation_only": list(CONFIRMATION),
            "entry": "market at next hourly open",
            "targets_pct": [.5, 1.0, 1.5, 2.0], "stop_pct": cause.STOP_PCT,
            "money": "3% current-equity margin, fixed 10x gross",
        },
        "definition": {
            "pivot": "two closed candles on each side of a local high/low",
            "fresh": "level lies between entry and target and no later pre-entry close invalidates it",
            "repeated_pool": "at least two touches separated by three hours within 0.15%",
            "confirmed_rejection": "pivot wick >=35% of range and subsequent move away >=0.5%",
            "confirmed_ob": "rejection barrier overlaps the independently defined valid order block",
        },
        "causal_timestamp_violations": 0,
        "baseline": baseline, "frontier": frontier,
        "frozen_before_validation": frozen, "validation": validation,
        "test": test, "confirmation": confirmation,
        "attribution_by_year": {str(year): stop_attribution(enriched, year)
                                for year in (2022, 2023, 2024, 2025, 2026)},
        "sources": {**sources, str(cause.ROWS): digest(cause.ROWS),
                    str(Path(cause.__file__)): digest(Path(cause.__file__)),
                    str(Path(__file__)): digest(Path(__file__))},
        "limitations": [
            "Levels are causal OHLCV proxies; historical order-book depth is unavailable.",
            "Repeated highs/lows can be either barriers or liquidity magnets; validation decides the effect.",
            "Published 2025-2026 data is confirmation evidence rather than a pristine holdout.",
        ],
    }
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"status": status, "frozen": frozen,
                      "validation": validation, "test": test,
                      "confirmation": confirmation,
                      "attribution": report["attribution_by_year"]},
                     indent=2), flush=True)


if __name__ == "__main__":
    main()
