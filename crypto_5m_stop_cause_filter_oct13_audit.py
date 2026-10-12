"""Causal stop-cause audit for the full-history 5m trend-pullback route.

The inspected route was frozen by older 2020-2024 work.  This audit replaces
its tiny ATR target with literal price objectives: bank 70% at +0.5%, move the
runner to breakeven, and leave 30% for +2.0%, with a fixed 1.0% initial stop.
Level, liquidity-pool, order-block proxy and crowding features use only bars
closed before the next-market-open entry.
"""
from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

import crypto_5m_barbell_regime_oct3_audit as portfolio
import crypto_5m_oi_actual_move_precision_oct6_audit as literal
import crypto_5m_rotation_barbell_transfer_oct4_audit as rotation
import crypto_5m_tree_filter_audit as events_source


ROOT = Path("reports/audit_2026_09_08")
EVENTS = ROOT / "crypto_5m_events_cache.pkl.gz"
TREE = ROOT / "crypto_5m_tree_filter_audit.json"
OUT = ROOT / "crypto_5m_stop_cause_filter_oct13.json"
ROUTE_INDEX = 2
DEVELOPMENT = (2020, 2021, 2022)
SELECTION = (2023,)
VALIDATION = (2024,)
TEST = (2025,)
CONFIRMATION = (2026,)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def candle_indicators(candle: dict) -> dict[str, np.ndarray]:
    frame = pd.DataFrame({key: np.asarray(candle[key], dtype=float)
                          for key in ("open", "high", "low", "close")})
    previous = frame.close.shift(1)
    tr = pd.concat(((frame.high - frame.low),
                    (frame.high - previous).abs(),
                    (frame.low - previous).abs()), axis=1).max(axis=1)
    return {
        "open": frame.open.to_numpy(), "high": frame.high.to_numpy(),
        "low": frame.low.to_numpy(), "close": frame.close.to_numpy(),
        "atr": tr.rolling(14).mean().to_numpy(),
        "prior20_high": frame.high.shift(1).rolling(20).max().to_numpy(),
        "prior20_low": frame.low.shift(1).rolling(20).min().to_numpy(),
    }


def causal_structure(rows: pd.DataFrame, candle: dict) -> pd.DataFrame:
    """Measure the structure visible at each signal close.

    The OB proxy is deliberately mechanical: the most recent opposite-colour
    candle immediately preceding a >=1 ATR displacement close through the
    previous 20-bar extreme.  It is valid only while no later close invalidates
    its far edge.  This is an auditable proxy, not a claim that discretionary
    order blocks can be reconstructed from OHLCV.
    """
    arrays = candle_indicators(candle)
    open_, high, low, close = (arrays[key] for key in
                               ("open", "high", "low", "close"))
    atr = arrays["atr"]
    prior20_high = arrays["prior20_high"]
    prior20_low = arrays["prior20_low"]
    indices = rows.entry_index.to_numpy(dtype=np.int64) - 1
    directions = rows.direction.to_numpy(dtype=np.int8)
    output = {
        key: np.full(len(rows), np.nan, dtype=float)
        for key in ("room_to_level_pct", "room_to_level_atr",
                    "liquidity_touches_96", "level_recency_bars",
                    "wick_rejection", "ob_found", "ob_valid",
                    "ob_retest", "false_ob")
    }
    for position, (signal, direction) in enumerate(zip(indices, directions)):
        if signal < 100 or signal >= len(close) or not np.isfinite(atr[signal]):
            continue
        entry = float(open_[signal + 1])
        unit = float(atr[signal])
        start = max(0, signal - 96)
        if direction > 0:
            level = float(np.max(high[start:signal]))
            distance = (level - entry) / entry
            tolerance = max(.0005 * entry, .15 * unit)
            touches = np.flatnonzero(np.abs(high[start:signal] - level) <= tolerance)
            upper_wick = high[signal] - max(open_[signal], close[signal])
            span = max(high[signal] - low[signal], 1e-12)
            rejection = (high[signal] >= level - tolerance
                         and close[signal] < level and upper_wick / span >= .45)
        else:
            level = float(np.min(low[start:signal]))
            distance = (entry - level) / entry
            tolerance = max(.0005 * entry, .15 * unit)
            touches = np.flatnonzero(np.abs(low[start:signal] - level) <= tolerance)
            lower_wick = min(open_[signal], close[signal]) - low[signal]
            span = max(high[signal] - low[signal], 1e-12)
            rejection = (low[signal] <= level + tolerance
                         and close[signal] > level and lower_wick / span >= .45)
        output["room_to_level_pct"][position] = distance
        output["room_to_level_atr"][position] = distance * entry / unit
        output["liquidity_touches_96"][position] = len(touches)
        output["level_recency_bars"][position] = (
            signal - (start + int(touches[-1])) if len(touches) else 999)
        output["wick_rejection"][position] = float(rejection)

        found = valid = retest = False
        for displacement in range(signal, max(20, signal - 48), -1):
            body = close[displacement] - open_[displacement]
            if not np.isfinite(atr[displacement]):
                continue
            if direction > 0:
                displaced = (body >= atr[displacement]
                             and close[displacement] > prior20_high[displacement])
            else:
                displaced = (-body >= atr[displacement]
                             and close[displacement] < prior20_low[displacement])
            if not displaced:
                continue
            origin = None
            for candidate in range(displacement - 1,
                                   max(-1, displacement - 4), -1):
                opposite = (close[candidate] < open_[candidate] if direction > 0
                            else close[candidate] > open_[candidate])
                if opposite:
                    origin = candidate
                    break
            if origin is None:
                continue
            found = True
            if direction > 0:
                valid = bool(np.min(close[displacement + 1:signal + 1],
                                    initial=np.inf) >= low[origin])
                retest = bool(valid and low[signal] <= high[origin]
                              and close[signal] >= low[origin])
            else:
                valid = bool(np.max(close[displacement + 1:signal + 1],
                                    initial=-np.inf) <= high[origin])
                retest = bool(valid and high[signal] >= low[origin]
                              and close[signal] <= high[origin])
            break
        output["ob_found"][position] = float(found)
        output["ob_valid"][position] = float(valid)
        output["ob_retest"][position] = float(retest)
        output["false_ob"][position] = float(found and not valid)
    return pd.DataFrame(output, index=rows.index)


def add_literal_outcomes(frame: pd.DataFrame, candles: dict[str, dict]) -> pd.DataFrame:
    pieces = []
    for symbol, part in frame.groupby("symbol", sort=False):
        candle = candles[symbol]
        times = np.asarray(candle["time"], dtype=np.int64)
        indices = np.searchsorted(times, part.entry_time.to_numpy(np.int64))
        values, exits, near, far, reasons = literal.resolve_literal(
            *[np.asarray(candle[key], dtype=float)
              for key in ("open", "high", "low", "close")],
            times, indices.astype(np.int64),
            part.direction.to_numpy(np.int8))
        found = part.copy()
        found["entry_index"] = indices
        found["signed_return"] = values
        found["exit_time"] = exits
        found["near_hit"] = near.astype(bool)
        found["far_hit"] = far.astype(bool)
        found["outcome_code"] = reasons
        structure = causal_structure(found, candle)
        for column in structure:
            found[column] = structure[column].to_numpy()
        pieces.append(found)
    result = pd.concat(pieces, ignore_index=True)
    crowd = result.groupby("entry_time").symbol.transform("nunique")
    same = result.groupby(["entry_time", "direction"]).symbol.transform("nunique")
    result["crowded_symbols"] = crowd.astype(int)
    result["same_direction_symbols"] = same.astype(int)
    return result


def metrics(frame: pd.DataFrame, years: tuple[int, ...]) -> dict:
    chosen = frame[frame.year.isin(years)].copy()
    accepted = portfolio.portfolio_gate(chosen)
    value = portfolio.metrics(accepted)
    value.update({
        "literal_near_hit_rate": round(100 * float(accepted.near_hit.mean()), 3)
            if len(accepted) else 0.0,
        "literal_far_hit_rate": round(100 * float(accepted.far_hit.mean()), 3)
            if len(accepted) else 0.0,
        "initial_stop_rate": round(100 * float((accepted.outcome_code == 0).mean()), 3)
            if len(accepted) else 0.0,
        "timeouts_without_near_rate": round(
            100 * float((accepted.outcome_code == 4).mean()), 3)
            if len(accepted) else 0.0,
    })
    return value


def detailed(frame: pd.DataFrame, years: tuple[int, ...]) -> dict:
    return {"all": metrics(frame, years),
            "years": {str(year): metrics(frame, (year,)) for year in years}}


def attribution(frame: pd.DataFrame, condition: pd.Series,
                label: str) -> dict:
    rows = []
    for year in range(2020, 2027):
        period = frame[frame.year == year]
        exposed = period[condition.loc[period.index]]
        other = period[~condition.loc[period.index]]
        stop = 100 * float((exposed.outcome_code == 0).mean()) if len(exposed) else 0
        control = 100 * float((other.outcome_code == 0).mean()) if len(other) else 0
        rows.append({"year": year, "exposed_n": int(len(exposed)),
                     "stop_rate": round(stop, 3),
                     "control_stop_rate": round(control, 3),
                     "stop_rate_lift": round(stop / control, 4) if control else None,
                     "near_hit_rate": round(100 * float(exposed.near_hit.mean()), 3)
                         if len(exposed) else 0.0})
    return {"factor": label, "by_year": rows}


def atom_masks(frame: pd.DataFrame) -> dict[str, pd.Series]:
    return {
        "room_ge_0_5pct": frame.room_to_level_pct >= .005,
        "room_ge_0_75pct": frame.room_to_level_pct >= .0075,
        "room_ge_1pct": frame.room_to_level_pct >= .01,
        "no_repeated_pool_nearby": ~((frame.liquidity_touches_96 >= 2)
                                      & (frame.room_to_level_pct <= .0075)),
        "no_wick_rejection": frame.wick_rejection == 0,
        "no_false_ob": frame.false_ob == 0,
        "valid_ob_retest": frame.ob_retest == 1,
        "same_direction_symbols_le_2": frame.same_direction_symbols <= 2,
        "same_direction_symbols_le_3": frame.same_direction_symbols <= 3,
        "btc_12bar_aligned": frame.btc_ret12_dir_atr >= 0,
        "btc_48bar_aligned": frame.btc_ret48_dir_atr >= 0,
        "own_12bar_aligned": frame.ret12_dir_atr >= 0,
        "relative_strength_aligned": frame.relative_ret12_dir_atr >= 0,
        "own_atr_ge_0_20pct": frame.atr_pct >= .0020,
        "own_atr_ge_0_25pct": frame.atr_pct >= .0025,
        "own_atr_ge_0_30pct": frame.atr_pct >= .0030,
        "btc_atr_ge_0_12pct": frame.btc_atr_pct >= .0012,
        "btc_atr_ge_0_15pct": frame.btc_atr_pct >= .0015,
        "btc_atr_ge_0_20pct": frame.btc_atr_pct >= .0020,
        "volume_z_ge_minus_0_5": frame.volume_z >= -.5,
        "volume_z_ge_minus_0_25": frame.volume_z >= -.25,
        "volume_not_weak": frame.volume_z >= 0,
        "volume_not_extreme": frame.volume_z <= 3,
    }


def passes_development(value: dict, baseline: dict) -> bool:
    all_ = value["all"]
    return bool(
        all_["n"] >= 1000 and all_["trades_per_day"] >= 1.0
        and all_["return_pct"] > 0
        # A defensive filter may trade some old-period return for stability,
        # but it must retain at least half of the baseline profit and improve
        # literal target precision without increasing drawdown.
        and all_["return_pct"] >= .5 * baseline["all"]["return_pct"]
        and all_["dd_pct"] <= baseline["all"]["dd_pct"]
        and all_["literal_near_hit_rate"] >= baseline["all"]["literal_near_hit_rate"] + 1
        and all(item["return_pct"] > 0 for item in value["years"].values())
    )


def passes_selection(value: dict) -> bool:
    all_ = value["all"]
    return bool(all_["n"] >= 150 and all_["return_pct"] > 0
                and all_["dd_pct"] <= 7 and all_["trades_per_day"] >= .5
                and all_["literal_near_hit_rate"] >= 58
                and all_["negative_months"] <= 6)


def objective(value: dict) -> dict:
    all_ = value["all"]
    return {
        "win_rate_75_to_95": bool(75 <= all_["wr"] <= 95),
        "at_least_3_trades_per_day": bool(all_["trades_per_day"] >= 3),
        "no_negative_months": bool(all_["negative_months"] == 0),
        "return_positive": bool(all_["return_pct"] > 0),
        "drawdown_at_most_5pct": bool(all_["dd_pct"] <= 5),
    }


def main() -> None:
    raw_tree = json.loads(TREE.read_text(encoding="utf-8"))
    route = raw_tree["top_candidates"][ROUTE_INDEX]
    expected = ("trend_pullback", 2.0, .25)
    actual = (route["family"], float(route["stop_atr"]),
              float(route["target_r"]))
    if actual != expected:
        raise AssertionError(f"frozen route changed: {actual}")
    raw_events = pd.read_pickle(EVENTS, compression="gzip")
    selected = rotation.selected_route(raw_events, route)
    candles, candle_sources = events_source.load()
    frame = add_literal_outcomes(selected, candles)

    baseline = {
        "development": detailed(frame, DEVELOPMENT),
        "selection": detailed(frame, SELECTION),
        "validation": detailed(frame, VALIDATION),
        "test": detailed(frame, TEST),
        "confirmation": detailed(frame, CONFIRMATION),
    }
    repeated_near = ((frame.liquidity_touches_96 >= 2)
                     & (frame.room_to_level_pct <= .0075))
    factors = [
        attribution(frame, frame.room_to_level_pct.between(0, .005),
                    "prior 96-bar level inside +0.5% path"),
        attribution(frame, repeated_near,
                    "repeated high/low pool inside +0.75% path"),
        attribution(frame, frame.wick_rejection == 1,
                    "signal candle rejects the directional level"),
        attribution(frame, frame.false_ob == 1,
                    "most recent displacement-origin OB proxy already invalidated"),
        attribution(frame, frame.ob_retest == 1,
                    "valid same-direction OB proxy retested"),
        attribution(frame, frame.same_direction_symbols >= 3,
                    "three or more route symbols signal same direction together"),
    ]

    masks = atom_masks(frame)
    searched = []
    names = list(masks)
    for size in (1, 2):
        for conditions in itertools.combinations(names, size):
            if size == 2 and conditions == ("same_direction_symbols_le_2",
                                             "same_direction_symbols_le_3"):
                continue
            mask = pd.Series(True, index=frame.index)
            for condition in conditions:
                mask &= masks[condition]
            chosen = frame[mask]
            development = detailed(chosen, DEVELOPMENT)
            if not passes_development(development, baseline["development"]):
                continue
            selection = detailed(chosen, SELECTION)
            passed = passes_selection(selection)
            score = (selection["all"]["return_pct"]
                     - selection["all"]["dd_pct"]
                     + .05 * selection["all"]["literal_near_hit_rate"]
                     + .001 * selection["all"]["n"])
            searched.append({"conditions": list(conditions),
                             "development": development,
                             "selection_2023": selection,
                             "selection_pass": passed,
                             "score": round(score, 8)})
    searched.sort(key=lambda item: (item["selection_pass"], item["score"]),
                  reverse=True)
    frozen = next((item for item in searched if item["selection_pass"]), None)
    validation = test = confirmation = None
    if frozen:
        mask = pd.Series(True, index=frame.index)
        for condition in frozen["conditions"]:
            mask &= masks[condition]
        chosen = frame[mask]
        validation = detailed(chosen, VALIDATION)
        test = detailed(chosen, TEST)
        confirmation = detailed(chosen, CONFIRMATION)
    status = "REJECTED_NO_CAUSAL_FILTER"
    if frozen:
        status = "REJECTED_OBJECTIVES"
        if all(all(objective(item).values())
               for item in (validation, test, confirmation)):
            status = "RESEARCH_CANDIDATE_REQUIRES_NEW_FORWARD"

    report = {
        "status": status, "ready_for_live_orders": False,
        "route": {key: route[key] for key in
                  ("family", "stop_atr", "target_r", "selected_leaves")},
        "protocol": {
            "source_history": "complete Binance spot 5m history by actual calendar year",
            "entry": "next 5-minute market open",
            "exit": "70% at literal +0.5%, remainder breakeven/2%; literal 1% stop; 6h expiry",
            "development": list(DEVELOPMENT), "policy_selection": list(SELECTION),
            "validation": list(VALIDATION), "test": list(TEST),
            "confirmation": list(CONFIRMATION),
            "filters_tested": len(names) + len(list(itertools.combinations(names, 2))),
            "money": "3% current-equity margin, fixed 10x, gross",
        },
        "baseline": baseline,
        "stop_cause_attribution": factors,
        "eligible_after_development": len(searched),
        "frozen_before_2024": frozen,
        "validation_2024": validation, "test_2025": test,
        "confirmation_2026": confirmation,
        "objective_checks": {
            "validation": objective(validation) if validation else None,
            "test": objective(test) if test else None,
            "confirmation": objective(confirmation) if confirmation else None,
        },
        "top_selection_candidates": searched[:20],
        "sources": {str(EVENTS): digest(EVENTS), str(TREE): digest(TREE),
                    **candle_sources, str(Path(__file__)): digest(Path(__file__))},
        "limitations": [
            "The OB measure is a fixed OHLCV displacement-origin proxy; historical candles cannot prove institutional order blocks.",
            "Binance spot provides structural history and does not prove OKX X-Perp execution transfer.",
            "Attribution is association; only chronological filter validation is used for decisions.",
            "Fees are excluded by user instruction.",
        ],
    }
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({
        "status": status,
        "baseline": {key: value["all"] for key, value in baseline.items()},
        "stop_cause_attribution": factors,
        "eligible_after_development": len(searched),
        "frozen": frozen, "validation": validation,
        "test": test, "confirmation": confirmation,
        "objective_checks": report["objective_checks"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
