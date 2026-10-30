"""Causal audit of a small post-TP2 trailing remainder.

The frozen Profit-First contract remains 5% at 0.5R and 95% at 2R.  A
candidate banks most of that 95% at 2R and keeps only a bounded tail.  The
tail stop is computed from a completed candle and becomes active on the next
candle, so a candle never supplies both the new favorable extreme and the
stop fill.  Selection uses 2022-2023 only; 2024, 2025-2026 and direct X-Perp
remain holdouts.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

import crypto_profit_first_exact_oct30_2030_forward as latest
import crypto_profit_first_exact_quality_bank_oct30_audit as quality
import crypto_profit_first_material_target_oct18_audit as frozen
import crypto_profit_first_runner_protection_oct30_audit as exact
from crypto_runtime_shadow_portfolio_oct3_audit import money


ROOT = Path("reports/audit_2026_09_08")
OUT = ROOT / "crypto_profit_first_exact_post_tp2_trail_oct31.json"
ROWS_OUT = ROOT / "crypto_profit_first_exact_post_tp2_trail_oct31.pkl"
FIRST_FRACTION = .05
FIRST_TARGET_R = .5
RUNNER_FRACTION = 1.0 - FIRST_FRACTION
TP2_R = 2.0

# Tail fraction is relative to the 95% runner, not to the original position.
TAIL_FRACTIONS = (.10, .20, .30)
TRAIL_DISTANCES_R = (.25, .50, .75, 1.0)
TRAIL_FLOORS_R = (1.50, 1.75, 2.0)
TAIL_HORIZONS = (12, 24, 48)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_candle(path: Path) -> dict:
    return pickle.loads(path.read_bytes())


def _stop_fill(opened: float, level: float, sign: int) -> float:
    return min(opened, level) if sign > 0 else max(opened, level)


def post_tp2_trail_exit_r(candle: dict, tp2_bar: int, *, direction: str,
                          entry: float, stop: float, distance_r: float,
                          floor_r: float, horizon: int) -> tuple[float, bool, int]:
    """Replay a trail that starts on the candle after the observed TP2 hit."""
    sign = 1 if direction == "LONG" else -1
    risk = abs(float(entry) - float(stop))
    if risk <= 0 or distance_r <= 0 or horizon <= 0:
        raise ValueError("invalid post-TP2 trail geometry")
    if not 0 <= tp2_bar < len(candle["time"]):
        raise IndexError("TP2 bar is outside the candle series")

    floor = float(entry) + sign * risk * float(floor_r)
    best = (float(candle["high"][tp2_bar]) if sign > 0
            else float(candle["low"][tp2_bar]))

    def level() -> float:
        raw = best - risk * float(distance_r) if sign > 0 \
            else best + risk * float(distance_r)
        return max(floor, raw) if sign > 0 else min(floor, raw)

    active_stop = level()
    last = tp2_bar + int(horizon)
    if last >= len(candle["time"]):
        last = len(candle["time"]) - 1
        complete = False
    else:
        complete = True

    previous_time = int(candle["time"][tp2_bar])
    for bar in range(tp2_bar + 1, last + 1):
        timestamp = int(candle["time"][bar])
        if timestamp - previous_time != exact.INTERVAL:
            return TP2_R, False, previous_time + exact.INTERVAL
        previous_time = timestamp
        opened = float(candle["open"][bar])
        high = float(candle["high"][bar])
        low = float(candle["low"][bar])
        stop_hit = low <= active_stop if sign > 0 else high >= active_stop
        if stop_hit:
            fill = _stop_fill(opened, active_stop, sign)
            return (sign * (fill - float(entry)) / risk,
                    True, timestamp + exact.INTERVAL)
        best = max(best, high) if sign > 0 else min(best, low)
        # This candle's new extreme can move the stop only for the next bar.
        active_stop = level()

    close = float(candle["close"][last])
    return (sign * (close - float(entry)) / risk,
            complete, int(candle["time"][last]) + exact.INTERVAL)


def policy_key(distance_r: float, floor_r: float, horizon: int) -> str:
    value = f"d{distance_r:g}_f{floor_r:g}_h{horizon}"
    return value.replace(".", "p")


def attach_tail_paths(frame: pd.DataFrame, paths: dict, *,
                      direct: bool) -> pd.DataFrame:
    """Attach every trail geometry in one grouped raw-candle pass."""
    result = frame.copy().reset_index(drop=True)
    label_col = "sequential_outcome" if direct else "a0p5_l0_label"
    exit_col = "sequential_exit_time" if direct else "a0p5_l0_exit_time"
    records = []
    candle_cache = {}
    for row in result.to_dict("records"):
        output = {
            "tp2_hit_close_r": np.nan,
            "tp2_hit_excursion_r": np.nan,
            "tp2_hit_directional_body_r": np.nan,
            "tp2_hit_volume_ratio": np.nan,
        }
        hit_tp2 = str(row[label_col]) == "TP_RUNNER"
        if hit_tp2:
            key = (("direct", str(row["symbol"])) if direct else
                   (str(row["period"]), str(row["symbol"])))
            if key not in candle_cache:
                candle = load_candle(paths[key])
                candle_cache[key] = (
                    candle,
                    {int(value): index
                     for index, value in enumerate(candle["time"])},
                )
            candle, lookup = candle_cache[key]
            tp2_start = int(float(row[exit_col])) - exact.INTERVAL
            tp2_bar = lookup.get(tp2_start)
            if tp2_bar is None:
                raise RuntimeError(
                    f"missing exact TP2 bar: {row['symbol']} {row['entry_time']}")
            sign = 1 if str(row["direction"]) == "LONG" else -1
            entry = float(row["entry"])
            risk = abs(entry - float(row["sl"]))
            opened = float(candle["open"][tp2_bar])
            close = float(candle["close"][tp2_bar])
            extreme = (float(candle["high"][tp2_bar]) if sign > 0
                       else float(candle["low"][tp2_bar]))
            prior_volume = [
                float(value) for value in candle["volume"][
                    max(0, tp2_bar - 20):tp2_bar]
            ]
            median_volume = float(np.median(prior_volume))
            output.update({
                "tp2_hit_close_r": sign * (close - entry) / risk,
                "tp2_hit_excursion_r": sign * (extreme - entry) / risk,
                "tp2_hit_directional_body_r": sign * (close - opened) / risk,
                "tp2_hit_volume_ratio": (
                    float(candle["volume"][tp2_bar]) / median_volume
                    if median_volume > 0 else 1.0),
            })
            for distance_r in TRAIL_DISTANCES_R:
                for floor_r in TRAIL_FLOORS_R:
                    for horizon in TAIL_HORIZONS:
                        key_name = policy_key(distance_r, floor_r, horizon)
                        tail_r, complete, exit_time = post_tp2_trail_exit_r(
                            candle, tp2_bar,
                            direction=str(row["direction"]),
                            entry=float(row["entry"]), stop=float(row["sl"]),
                            distance_r=distance_r, floor_r=floor_r,
                            horizon=horizon)
                        output[f"{key_name}_r"] = tail_r
                        output[f"{key_name}_complete"] = complete
                        output[f"{key_name}_exit_time"] = exit_time
        else:
            for distance_r in TRAIL_DISTANCES_R:
                for floor_r in TRAIL_FLOORS_R:
                    for horizon in TAIL_HORIZONS:
                        key_name = policy_key(distance_r, floor_r, horizon)
                        output[f"{key_name}_r"] = np.nan
                        output[f"{key_name}_complete"] = True
                        output[f"{key_name}_exit_time"] = float(row["exit_time"])
        records.append(output)
    return pd.concat([result, pd.DataFrame.from_records(records)], axis=1)


def apply_policy(frame: pd.DataFrame, *, tail_fraction: float,
                 distance_r: float, floor_r: float, horizon: int,
                 direct: bool = False) -> pd.DataFrame:
    result = frame.copy()
    label_col = "sequential_outcome" if direct else "a0p5_l0_label"
    key_name = policy_key(distance_r, floor_r, horizon)
    tp2 = result[label_col].astype(str).eq("TP_RUNNER")
    complete = result[f"{key_name}_complete"].astype(bool)
    result["post_tp2_tail_complete"] = complete
    result["post_tp2_tail_exit_r"] = result[f"{key_name}_r"]
    result["post_tp2_tail_fraction"] = 0.0
    result.loc[tp2, "post_tp2_tail_fraction"] = float(tail_fraction)
    tail_r = result[f"{key_name}_r"].astype(float)
    candidate_r = (
        FIRST_FRACTION * FIRST_TARGET_R
        + RUNNER_FRACTION * (
            (1.0 - float(tail_fraction)) * TP2_R
            + float(tail_fraction) * tail_r
        )
    )
    result.loc[tp2 & complete, "net_r"] = candidate_r.loc[tp2 & complete]
    result["signed_price_return"] = result.net_r * result.risk_pct
    return result


def resolved(frame: pd.DataFrame) -> pd.DataFrame:
    return frame[frame.post_tp2_tail_complete.astype(bool)].copy()


def metric(frame: pd.DataFrame, periods: tuple[str, ...]) -> dict:
    part = resolved(frame[frame.period.isin(periods)])
    return money(part.to_dict("records"), .03)


def compact(value: dict) -> dict:
    keys = ("n", "wr", "return_pct", "dd_pct", "max_loss_streak",
            "negative_months", "trades_per_calendar_day",
            "monthly_return_pct")
    return {key: value[key] for key in keys}


def development_pass(candidate: dict, baseline: dict) -> bool:
    return bool(
        candidate["return_pct"] >= baseline["return_pct"]
        and candidate["dd_pct"] <= 1.05 * baseline["dd_pct"]
        and candidate["negative_months"] <= baseline["negative_months"]
        and candidate["max_loss_streak"] <= baseline["max_loss_streak"])


def dominates(candidate: dict, baseline: dict) -> bool:
    return bool(
        candidate["return_pct"] >= baseline["return_pct"]
        and candidate["dd_pct"] <= baseline["dd_pct"]
        and candidate["negative_months"] <= baseline["negative_months"]
        and candidate["max_loss_streak"] <= baseline["max_loss_streak"])


def score(selection: dict, confirmation: dict,
          base_selection: dict, base_confirmation: dict) -> float:
    return round(min(
        selection["return_pct"] / max(base_selection["return_pct"], 1e-9),
        confirmation["return_pct"] / max(base_confirmation["return_pct"], 1e-9),
    ) + .1 * min(
        base_selection["dd_pct"] / max(selection["dd_pct"], 1e-9),
        base_confirmation["dd_pct"] / max(confirmation["dd_pct"], 1e-9),
    ), 8)


def base_metric(frame: pd.DataFrame, periods: tuple[str, ...]) -> dict:
    return money(frame[frame.period.isin(periods)].to_dict("records"), .03)


def direct_slices(frame: pd.DataFrame, cutoff: int) -> dict:
    frame = resolved(frame)
    october = frame[frame.entry_time.ge(pd.Timestamp(
        "2026-10-01", tz="UTC").timestamp())]
    unseen = frame[frame.entry_time.gt(cutoff)]
    return {
        "october": {
            "margin_3pct": latest.metrics(october, .03),
            "margin_5pct": latest.metrics(october, .05),
        },
        "after_oct16": {
            "margin_3pct": latest.metrics(unseen, .03),
            "margin_5pct": latest.metrics(unseen, .05),
        },
    }


def main() -> None:
    historical_paths = exact.candle_paths((
        *frozen.SELECT, *frozen.CONFIRM, *frozen.VALIDATION, *frozen.LATER))
    historical = attach_tail_paths(
        quality.historical_rows(), historical_paths, direct=False)
    base_selection = base_metric(historical, frozen.SELECT)
    base_confirmation = base_metric(historical, frozen.CONFIRM)

    frontier = []
    eligible = []
    for tail_fraction in TAIL_FRACTIONS:
        for distance_r in TRAIL_DISTANCES_R:
            for floor_r in TRAIL_FLOORS_R:
                for horizon in TAIL_HORIZONS:
                    candidate = apply_policy(
                        historical, tail_fraction=tail_fraction,
                        distance_r=distance_r, floor_r=floor_r,
                        horizon=horizon)
                    selection = metric(candidate, frozen.SELECT)
                    confirmation = metric(candidate, frozen.CONFIRM)
                    passed = (development_pass(selection, base_selection)
                              and development_pass(
                                  confirmation, base_confirmation))
                    item = {
                        "tail_fraction_of_95pct_runner": tail_fraction,
                        "tail_fraction_of_initial_position": round(
                            RUNNER_FRACTION * tail_fraction, 4),
                        "closed_at_tp2_fraction_of_initial_position": round(
                            RUNNER_FRACTION * (1 - tail_fraction), 4),
                        "trail_distance_r": distance_r,
                        "trail_floor_r": floor_r,
                        "tail_horizon_bars": horizon,
                        "selection_2022h1": compact(selection),
                        "confirmation_2022h2_2023": compact(confirmation),
                        "development_pass": passed,
                        "score": score(selection, confirmation,
                                       base_selection, base_confirmation),
                    }
                    frontier.append(item)
                    if passed:
                        eligible.append(item)
    frontier.sort(key=lambda item: (item["development_pass"], item["score"]),
                  reverse=True)
    eligible.sort(key=lambda item: item["score"], reverse=True)
    selected = eligible[0] if eligible else None

    base_validation = base_metric(historical, frozen.VALIDATION)
    base_later = base_metric(historical, frozen.LATER)
    source = json.loads(latest.SOURCE_REPORT.read_text(encoding="utf-8"))
    cutoff = int(pd.Timestamp(
        source["fresh_direct_coverage"]["last_entry"]).timestamp())
    direct_paths = latest.paths(latest.NEW_MANIFEST.parent)
    direct = attach_tail_paths(
        pd.read_pickle(quality.DIRECT_ROWS), direct_paths, direct=True)

    validation = later = base_direct = candidate_direct = None
    historical_pass = direct_pass = False
    candidate_current = None
    if selected:
        kwargs = {
            "tail_fraction": float(selected["tail_fraction_of_95pct_runner"]),
            "distance_r": float(selected["trail_distance_r"]),
            "floor_r": float(selected["trail_floor_r"]),
            "horizon": int(selected["tail_horizon_bars"]),
        }
        candidate = apply_policy(historical, **kwargs)
        validation = metric(candidate, frozen.VALIDATION)
        later = metric(candidate, frozen.LATER)
        historical_pass = (
            dominates(validation, base_validation)
            and dominates(later, base_later))
        candidate_current = apply_policy(direct, direct=True, **kwargs)
        common = resolved(candidate_current)
        baseline_common = common.copy()
        baseline_common["net_r"] = pd.read_pickle(quality.DIRECT_ROWS).loc[
            common.index, "net_r"].astype(float)
        baseline_common["signed_price_return"] = (
            baseline_common.net_r * baseline_common.risk_pct)
        base_direct = direct_slices(baseline_common, cutoff)
        candidate_direct = direct_slices(candidate_current, cutoff)
        base_oct = base_direct["october"]["margin_3pct"]
        cand_oct = candidate_direct["october"]["margin_3pct"]
        base_new = base_direct["after_oct16"]["margin_3pct"]
        cand_new = candidate_direct["after_oct16"]["margin_3pct"]
        direct_pass = bool(
            cand_oct["return_pct"] >= base_oct["return_pct"]
            and cand_oct["dd_pct"] <= base_oct["dd_pct"]
            and cand_new["return_pct"] >= base_new["return_pct"]
            and cand_new["dd_pct"] <= base_new["dd_pct"]
            and cand_new["negative_months"] <= base_new["negative_months"])

    robust = bool(selected and historical_pass and direct_pass)
    if robust and candidate_current is not None:
        candidate_current.to_pickle(ROWS_OUT)

    report = {
        "status": ("EXACT_POST_TP2_TRAIL_SHADOW_CHALLENGER_AWAIT_FORWARD"
                   if robust else "REJECTED_EXACT_POST_TP2_TRAIL"),
        "ready_for_live_orders": False,
        "orders_possible": False,
        "runtime_change_allowed": False,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "protocol": {
            "base_exit": "5% at 0.5R; remaining 95% at 2R; next-bar BE",
            "candidate": (
                "bank most of the 95% runner at 2R and trail a small tail; "
                "each completed-bar extreme moves the stop for the next bar"),
            "tail_fractions": list(TAIL_FRACTIONS),
            "trail_distances_r": list(TRAIL_DISTANCES_R),
            "trail_floors_r": list(TRAIL_FLOORS_R),
            "tail_horizons_bars": list(TAIL_HORIZONS),
            "selection": [*frozen.SELECT, *frozen.CONFIRM],
            "holdouts": [*frozen.VALIDATION, *frozen.LATER,
                         "direct through 2026-10-30"],
            "ordering": "old stop first; new trail level active next bar",
            "position_fraction": "client setting",
        },
        "candidate_count": len(frontier),
        "development_pass_count": len(eligible),
        "baseline": {
            "selection_2022h1": compact(base_selection),
            "confirmation_2022h2_2023": compact(base_confirmation),
            "validation_2024": compact(base_validation),
            "later_2025_2026": compact(base_later),
            "direct_common_rows": base_direct,
        },
        "selected_before_holdouts": selected,
        "validation_2024": compact(validation) if validation else None,
        "later_2025_2026": compact(later) if later else None,
        "direct_xperp": candidate_direct,
        "historical_holdout_pass": historical_pass,
        "fresh_exact_forward_pass": direct_pass,
        "robust_improvement": robust,
        "development_frontier": frontier,
        "sources": {str(path): digest(path) for path in (
            *historical_paths.values(), *direct_paths.values(),
            quality.OUT, latest.OUT, quality.DIRECT_ROWS,
            Path(__file__).resolve())},
        "limitations": [
            "The selected policy still needs genuinely new shadow-forward trades.",
            "A candle replay cannot reproduce market-order latency or gaps beyond its OHLC.",
            "Historical negative months remain disqualifying for any profit guarantee.",
        ],
    }
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True),
                   encoding="utf-8")
    print(json.dumps({
        "status": report["status"],
        "selected": selected,
        "validation": report["validation_2024"],
        "later": report["later_2025_2026"],
        "baseline_direct": base_direct,
        "direct": candidate_direct,
        "robust": robust,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
