"""Exact, executable TP1-candle volume banking audit.

The frozen route banks 5% at 0.5R and leaves 95% for 2R with breakeven active
on the next bar.  This audit may bank an additional part at the *close* of the
TP1-hit bar when that completed bar has weak volume.  Both the volume and close
price are therefore observable before the market action; no passed TP1 price or
intrabar order is invented.

The policy grid is selected on 2022-2023.  Only the selected policy is opened on
2024, 2025-2026 and the latest exact X-Perp replay.
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
OUT = ROOT / "crypto_profit_first_exact_tp1_volume_bank_oct30.json"
ROWS_OUT = ROOT / "crypto_profit_first_exact_tp1_volume_bank_oct30.pkl"
VOLUME_THRESHOLDS = (.75, 1.0, 1.25, 1.5, 2.0)
EXTRA_CLOSE_FRACTIONS = (.10, .25, .50, .75)
BASE_CLOSE_FRACTION = .05
VOLUME_LOOKBACK = 20


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_candle(path: Path) -> dict:
    return pickle.loads(path.read_bytes())


def attach_tp1_path(frame: pd.DataFrame, paths: dict, *,
                    direct: bool) -> pd.DataFrame:
    """Attach values observable at the completed TP1-hit candle."""
    result = frame.copy().reset_index(drop=True)
    bank_col = "sequential_banked" if direct else "a0p5_l0_banked"
    exit_col = "sequential_exit_time" if direct else "a0p5_l0_exit_time"
    candle_cache = {}
    features = []
    found = 0
    for row in result.to_dict("records"):
        item = {
            "bars_to_tp1": np.nan,
            "hit_volume_ratio": np.nan,
            "hit_close_r": np.nan,
            "hit_bar_end": np.nan,
            "runner_open_after_hit_close": False,
        }
        if bool(row[bank_col]):
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
            start = lookup[int(row["entry_time"])]
            sign = 1 if str(row["direction"]) == "LONG" else -1
            entry = float(row["entry"])
            stop = float(row["sl"])
            risk = abs(entry - stop)
            target = entry + sign * .5 * risk
            hit = None
            for bar in range(start, min(start + exact.HORIZON,
                                        len(candle["time"]))):
                stop_hit = (float(candle["low"][bar]) <= stop if sign > 0
                            else float(candle["high"][bar]) >= stop)
                if stop_hit:
                    break
                target_hit = (float(candle["high"][bar]) >= target if sign > 0
                              else float(candle["low"][bar]) <= target)
                if target_hit:
                    hit = bar
                    break
            if hit is None:
                raise RuntimeError(
                    f"exact banked row has no stop-first TP1 hit: "
                    f"{row['symbol']} {row['entry_time']}")
            prior_volume = [
                float(candle["volume"][bar])
                for bar in range(max(0, hit - VOLUME_LOOKBACK), hit)
            ]
            volume_median = float(np.median(prior_volume))
            hit_end = int(candle["time"][hit]) + exact.INTERVAL
            item = {
                "bars_to_tp1": hit - start + 1,
                "hit_volume_ratio": (
                    float(candle["volume"][hit]) / volume_median
                    if volume_median > 0 else 1.0),
                "hit_close_r": (
                    sign * (float(candle["close"][hit]) - entry) / risk),
                "hit_bar_end": hit_end,
                # If the 2R runner resolved on the TP1 bar, there is no open
                # remainder to close after that candle completes.
                "runner_open_after_hit_close": float(row[exit_col]) > hit_end,
            }
            found += 1
        features.append(item)
    attached = pd.concat([result, pd.DataFrame(features)], axis=1)
    expected = int(attached[bank_col].astype(bool).sum())
    if found != expected or attached.loc[
            attached[bank_col].astype(bool), "hit_volume_ratio"].isna().any():
        raise RuntimeError(
            f"TP1 path feature coverage mismatch: found={found} expected={expected}")
    return attached


def apply_policy(frame: pd.DataFrame, *, volume_threshold: float,
                 extra_close_fraction: float, direct: bool = False) -> pd.DataFrame:
    """Apply a market close at the observed TP1-candle close."""
    result = frame.copy()
    bank_col = "sequential_banked" if direct else "a0p5_l0_banked"
    runner_col = "sequential_runner_r" if direct else "a0p5_l0_runner_r"
    runner = result[runner_col].astype(float)
    banked = result[bank_col].astype(bool)
    weak = (
        banked
        & result.runner_open_after_hit_close.astype(bool)
        & result.hit_volume_ratio.lt(volume_threshold)
    )
    result["tp1_extra_close_fraction"] = 0.0
    result.loc[weak, "tp1_extra_close_fraction"] = extra_close_fraction
    result["net_r"] = runner.where(
        ~banked, BASE_CLOSE_FRACTION * .5 + (1 - BASE_CLOSE_FRACTION) * runner)
    result.loc[weak, "net_r"] = (
        BASE_CLOSE_FRACTION * .5
        + extra_close_fraction * result.loc[weak, "hit_close_r"]
        + (1 - BASE_CLOSE_FRACTION - extra_close_fraction) * runner.loc[weak]
    )
    result["signed_price_return"] = result.net_r * result.risk_pct
    return result


def metric(frame: pd.DataFrame, periods: tuple[str, ...]) -> dict:
    return money(frame[frame.period.isin(periods)].to_dict("records"), .03)


def compact(value: dict) -> dict:
    keys = ("n", "wr", "return_pct", "dd_pct", "max_loss_streak",
            "negative_months", "trades_per_calendar_day",
            "monthly_return_pct")
    return {key: value[key] for key in keys}


def development_pass(candidate: dict, baseline: dict) -> bool:
    return bool(
        candidate["return_pct"] >= baseline["return_pct"]
        and candidate["dd_pct"] <= baseline["dd_pct"]
        and candidate["negative_months"] <= baseline["negative_months"]
        and candidate["max_loss_streak"] <= baseline["max_loss_streak"])


def dominates(candidate: dict, baseline: dict) -> bool:
    return development_pass(candidate, baseline)


def score(selection: dict, confirmation: dict,
          base_selection: dict, base_confirmation: dict) -> float:
    return round(min(
        selection["return_pct"] / max(base_selection["return_pct"], 1e-9),
        confirmation["return_pct"] / max(base_confirmation["return_pct"], 1e-9),
    ) + .10 * min(
        base_selection["dd_pct"] / max(selection["dd_pct"], 1e-9),
        base_confirmation["dd_pct"] / max(confirmation["dd_pct"], 1e-9),
    ), 8)


def direct_metrics(frame: pd.DataFrame, cutoff: int) -> dict:
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
    historical = attach_tp1_path(
        quality.historical_rows(), historical_paths, direct=False)
    base_selection = metric(historical, frozen.SELECT)
    base_confirmation = metric(historical, frozen.CONFIRM)

    frontier = []
    eligible = []
    for threshold in VOLUME_THRESHOLDS:
        for fraction in EXTRA_CLOSE_FRACTIONS:
            candidate = apply_policy(
                historical, volume_threshold=threshold,
                extra_close_fraction=fraction)
            selection = metric(candidate, frozen.SELECT)
            confirmation = metric(candidate, frozen.CONFIRM)
            passed = (
                development_pass(selection, base_selection)
                and development_pass(confirmation, base_confirmation))
            item = {
                "volume_ratio_below": threshold,
                "extra_close_fraction_at_hit_bar_close": fraction,
                "total_closed_after_weak_tp1_bar": BASE_CLOSE_FRACTION + fraction,
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

    base_validation = metric(historical, frozen.VALIDATION)
    base_later = metric(historical, frozen.LATER)
    source = json.loads(latest.SOURCE_REPORT.read_text(encoding="utf-8"))
    cutoff = int(pd.Timestamp(
        source["fresh_direct_coverage"]["last_entry"]).timestamp())
    direct_paths = latest.paths(latest.NEW_MANIFEST.parent)
    direct = attach_tp1_path(
        pd.read_pickle(quality.DIRECT_ROWS), direct_paths, direct=True)
    baseline_direct = direct_metrics(direct, cutoff)

    validation = later = candidate_direct = None
    historical_pass = direct_pass = False
    candidate_current = None
    if selected:
        candidate = apply_policy(
            historical,
            volume_threshold=float(selected["volume_ratio_below"]),
            extra_close_fraction=float(
                selected["extra_close_fraction_at_hit_bar_close"]))
        validation = metric(candidate, frozen.VALIDATION)
        later = metric(candidate, frozen.LATER)
        historical_pass = (
            dominates(validation, base_validation)
            and dominates(later, base_later))
        candidate_current = apply_policy(
            direct,
            volume_threshold=float(selected["volume_ratio_below"]),
            extra_close_fraction=float(
                selected["extra_close_fraction_at_hit_bar_close"]),
            direct=True)
        candidate_direct = direct_metrics(candidate_current, cutoff)
        base_oct = baseline_direct["october"]["margin_3pct"]
        cand_oct = candidate_direct["october"]["margin_3pct"]
        base_new = baseline_direct["after_oct16"]["margin_3pct"]
        cand_new = candidate_direct["after_oct16"]["margin_3pct"]
        direct_pass = bool(
            cand_oct["return_pct"] >= base_oct["return_pct"]
            and cand_oct["dd_pct"] <= base_oct["dd_pct"]
            and cand_new["return_pct"] >= base_new["return_pct"]
            and cand_new["dd_pct"] <= base_new["dd_pct"]
            and cand_new["negative_months"] <= base_new["negative_months"]
            and cand_new["trades_per_calendar_day"] >= 3)

    robust = bool(selected and historical_pass and direct_pass)
    if robust and candidate_current is not None:
        candidate_current.to_pickle(ROWS_OUT)
    report = {
        "status": (
            "EXACT_TP1_VOLUME_BANK_SHADOW_CHALLENGER_AWAIT_NEW_FORWARD"
            if robust else "REJECTED_EXACT_TP1_VOLUME_BANK"),
        "ready_for_live_orders": False,
        "orders_possible": False,
        "runtime_change_allowed": False,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "protocol": {
            "entry": "frozen exact Profit-First route",
            "base_exit": "5% at 0.5R, 95% to 2R, BE active next bar",
            "candidate_action": (
                "after the TP1-hit candle closes, use a market close at that "
                "observed close when its volume is weak"),
            "volume_ratio": "hit-candle volume / median prior 20 completed bars",
            "same_bar_2r": "excluded because no runner remains at candle close",
            "grid": {
                "volume_thresholds": list(VOLUME_THRESHOLDS),
                "extra_close_fractions": list(EXTRA_CLOSE_FRACTIONS),
            },
            "selection": [*frozen.SELECT, *frozen.CONFIRM],
            "holdouts": [*frozen.VALIDATION, *frozen.LATER,
                         "direct through 2026-10-30"],
            "orders": "market only",
            "position_fraction": "client setting",
        },
        "candidate_count": len(frontier),
        "development_pass_count": len(eligible),
        "baseline": {
            "selection_2022h1": compact(base_selection),
            "confirmation_2022h2_2023": compact(base_confirmation),
            "validation_2024": compact(base_validation),
            "later_2025_2026": compact(base_later),
            "direct_xperp": baseline_direct,
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
            "The candidate is frozen only for genuinely new shadow forward data.",
            "Historical months remain negative; the 30% monthly target at 3% margin is not proven.",
            "The extra close assumes execution at the completed candle close and excludes fees at user request.",
        ],
    }
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({
        "status": report["status"],
        "selected": selected,
        "validation": report["validation_2024"],
        "later": report["later_2025_2026"],
        "baseline_direct": baseline_direct,
        "direct": candidate_direct,
        "robust": robust,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
