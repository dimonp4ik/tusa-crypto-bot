"""Causal TP1-volume bank audit executed at the next 15-minute open.

The earlier experiment observed the completed TP1 candle but credited the
extra market close at that candle's close.  This version makes the decision
only after the candle is complete and executes at the next candle open.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

import crypto_profit_first_exact_oct31_1700_forward as latest
import crypto_profit_first_exact_quality_bank_oct30_audit as quality
import crypto_profit_first_exact_tp1_volume_bank_oct30_audit as prior
import crypto_profit_first_material_target_oct18_audit as frozen
import crypto_profit_first_runner_protection_oct30_audit as exact
from crypto_runtime_shadow_portfolio_oct3_audit import money


ROOT = Path("reports/audit_2026_09_08")
OUT = ROOT / "crypto_profit_first_tp1_volume_next_open_nov1.json"
ROWS_OUT = ROOT / "crypto_profit_first_tp1_volume_next_open_nov1.pkl"
OLD_DIRECT_ROWS = quality.DIRECT_ROWS
NEW_DIRECT_ROWS = latest.ROWS_OUT


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def attach_next_open(frame: pd.DataFrame, paths: dict, *,
                     direct: bool) -> pd.DataFrame:
    result = prior.attach_tp1_path(frame, paths, direct=direct)
    bank_col = "sequential_banked" if direct else "a0p5_l0_banked"
    exit_col = "sequential_exit_time" if direct else "a0p5_l0_exit_time"
    caches = {}
    features = []
    for row in result.to_dict("records"):
        item = {"action_time": np.nan, "action_open_r": np.nan,
                "runner_open_at_action": False}
        if bool(row[bank_col]):
            key = (("direct", str(row["symbol"])) if direct else
                   (str(row["period"]), str(row["symbol"])))
            if key not in caches:
                candle = prior.load_candle(paths[key])
                caches[key] = (candle, {
                    int(value): index for index, value in enumerate(candle["time"])
                })
            candle, lookup = caches[key]
            action_time = int(row["hit_bar_end"])
            action_index = lookup.get(action_time)
            if action_index is not None:
                sign = 1 if str(row["direction"]) == "LONG" else -1
                entry = float(row["entry"])
                stop = float(row["sl"])
                risk = abs(entry - stop)
                runner_target = entry + sign * 2.0 * risk
                action_open = float(candle["open"][action_index])
                inside = (stop < action_open < runner_target if sign > 0
                          else runner_target < action_open < stop)
                item = {
                    "action_time": action_time,
                    "action_open_r": sign * (action_open - entry) / risk,
                    "runner_open_at_action": bool(
                        float(row[exit_col]) >= action_time and inside),
                }
        features.append(item)
    attached = pd.concat([result.reset_index(drop=True),
                          pd.DataFrame(features)], axis=1)
    actionable = attached[bank_col].astype(bool) & attached.action_time.notna()
    if attached.loc[actionable, "action_open_r"].isna().any():
        raise RuntimeError("next-open feature coverage mismatch")
    return attached


def apply_policy(frame: pd.DataFrame, *, volume_threshold: float,
                 extra_close_fraction: float, direct: bool = False) -> pd.DataFrame:
    result = frame.copy()
    bank_col = "sequential_banked" if direct else "a0p5_l0_banked"
    runner_col = "sequential_runner_r" if direct else "a0p5_l0_runner_r"
    banked = result[bank_col].astype(bool)
    runner = result[runner_col].astype(float)
    weak = (banked & result.runner_open_at_action.astype(bool)
            & result.hit_volume_ratio.lt(volume_threshold))
    result["tp1_extra_close_fraction"] = 0.0
    result.loc[weak, "tp1_extra_close_fraction"] = extra_close_fraction
    result["net_r"] = runner.where(
        ~banked, prior.BASE_CLOSE_FRACTION * .5
        + (1 - prior.BASE_CLOSE_FRACTION) * runner)
    result.loc[weak, "net_r"] = (
        prior.BASE_CLOSE_FRACTION * .5
        + extra_close_fraction * result.loc[weak, "action_open_r"]
        + (1 - prior.BASE_CLOSE_FRACTION - extra_close_fraction)
        * runner.loc[weak])
    result["signed_price_return"] = result.net_r * result.risk_pct
    return result


def metric(frame: pd.DataFrame, periods: tuple[str, ...]) -> dict:
    return money(frame[frame.period.isin(periods)].to_dict("records"), .03)


def compact(value: dict) -> dict:
    return {key: value[key] for key in (
        "n", "wr", "return_pct", "dd_pct", "max_loss_streak",
        "negative_months", "trades_per_calendar_day", "monthly_return_pct")}


def direct_metrics(frame: pd.DataFrame, cutoff: int,
                   old_snapshot_end: int) -> dict:
    scopes = {
        "october": frame[frame.entry_time.ge(pd.Timestamp(
            "2026-10-01", tz="UTC").timestamp())],
        "after_oct16": frame[frame.entry_time.gt(cutoff)],
        "after_old_snapshot": frame[frame.entry_time.gt(old_snapshot_end)],
    }
    return {scope: {
        f"margin_{int(margin * 100)}pct": latest.metrics(rows, margin)
        for margin in (.03, .05)
    } for scope, rows in scopes.items()}


def dominates(candidate: dict, baseline: dict) -> bool:
    return bool(candidate["return_pct"] >= baseline["return_pct"]
                and candidate["dd_pct"] <= baseline["dd_pct"]
                and candidate["negative_months"] <= baseline["negative_months"]
                and candidate["max_loss_streak"] <= baseline["max_loss_streak"])


def main() -> None:
    historical_paths = exact.candle_paths((
        *frozen.SELECT, *frozen.CONFIRM, *frozen.VALIDATION, *frozen.LATER))
    historical = attach_next_open(
        quality.historical_rows(), historical_paths, direct=False)
    base_selection = metric(historical, frozen.SELECT)
    base_confirmation = metric(historical, frozen.CONFIRM)
    frontier = []
    eligible = []
    for threshold in prior.VOLUME_THRESHOLDS:
        for fraction in prior.EXTRA_CLOSE_FRACTIONS:
            candidate = apply_policy(
                historical, volume_threshold=threshold,
                extra_close_fraction=fraction)
            selection = metric(candidate, frozen.SELECT)
            confirmation = metric(candidate, frozen.CONFIRM)
            passed = (dominates(selection, base_selection)
                      and dominates(confirmation, base_confirmation))
            item = {
                "volume_ratio_below": threshold,
                "extra_close_fraction_at_next_open": fraction,
                "total_closed_after_action": prior.BASE_CLOSE_FRACTION + fraction,
                "selection_2022h1": compact(selection),
                "confirmation_2022h2_2023": compact(confirmation),
                "development_pass": passed,
                "score": prior.score(selection, confirmation,
                                     base_selection, base_confirmation),
            }
            frontier.append(item)
            if passed:
                eligible.append(item)
    eligible.sort(key=lambda item: item["score"], reverse=True)
    selected = eligible[0] if eligible else None

    base_validation = metric(historical, frozen.VALIDATION)
    base_later = metric(historical, frozen.LATER)
    old_direct = pd.read_pickle(OLD_DIRECT_ROWS)
    old_snapshot_end = int(old_direct.entry_time.max())
    source = json.loads(latest.SOURCE_REPORT.read_text(encoding="utf-8"))
    cutoff = int(pd.Timestamp(source["fresh_direct_coverage"]["last_entry"]).timestamp())
    direct_paths = latest.paths(latest.NEW_MANIFEST.parent)
    direct = attach_next_open(
        pd.read_pickle(NEW_DIRECT_ROWS), direct_paths, direct=True)
    baseline_direct = direct_metrics(direct, cutoff, old_snapshot_end)

    validation = later = candidate_direct = None
    historical_pass = direct_pass = False
    candidate_current = None
    if selected:
        candidate = apply_policy(
            historical, volume_threshold=float(selected["volume_ratio_below"]),
            extra_close_fraction=float(
                selected["extra_close_fraction_at_next_open"]))
        validation = metric(candidate, frozen.VALIDATION)
        later = metric(candidate, frozen.LATER)
        historical_pass = (dominates(validation, base_validation)
                           and dominates(later, base_later))
        candidate_current = apply_policy(
            direct, volume_threshold=float(selected["volume_ratio_below"]),
            extra_close_fraction=float(
                selected["extra_close_fraction_at_next_open"]), direct=True)
        candidate_direct = direct_metrics(
            candidate_current, cutoff, old_snapshot_end)
        comparisons = []
        for scope in ("october", "after_oct16"):
            base_value = baseline_direct[scope]["margin_3pct"]
            candidate_value = candidate_direct[scope]["margin_3pct"]
            comparisons.append(
                candidate_value["return_pct"] >= base_value["return_pct"]
                and candidate_value["dd_pct"] <= base_value["dd_pct"]
                and candidate_value["negative_months"] <= base_value["negative_months"])
        direct_pass = all(comparisons)

    sealed_evidence_pass = bool(selected and historical_pass and direct_pass)
    forward_base = baseline_direct["after_old_snapshot"]["margin_3pct"]
    forward_candidate = (candidate_direct["after_old_snapshot"]["margin_3pct"]
                         if candidate_direct else None)
    forward_gate = {
        "minimum_resolved_rows": 30,
        "observed_resolved_rows": int(forward_base["n"]),
        "candidate_return_not_below_baseline": bool(
            forward_candidate
            and forward_candidate["return_pct"] >= forward_base["return_pct"]),
        "candidate_drawdown_not_above_baseline": bool(
            forward_candidate
            and forward_candidate["dd_pct"] <= forward_base["dd_pct"]),
    }
    forward_gate["passes"] = bool(
        forward_gate["observed_resolved_rows"]
        >= forward_gate["minimum_resolved_rows"]
        and forward_gate["candidate_return_not_below_baseline"]
        and forward_gate["candidate_drawdown_not_above_baseline"])
    if sealed_evidence_pass and candidate_current is not None:
        candidate_current.to_pickle(ROWS_OUT)
    report = {
        "status": (
            "NEXT_OPEN_VOLUME_BANK_FORWARD_GATE_PASSED"
            if sealed_evidence_pass and forward_gate["passes"]
            else "NEXT_OPEN_VOLUME_BANK_AWAIT_MORE_FORWARD"
            if sealed_evidence_pass else "REJECTED_NEXT_OPEN_VOLUME_BANK"),
        "ready_for_live_orders": False,
        "orders_possible": False,
        "runtime_change_allowed": False,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "protocol": {
            "decision": "after the completed TP1-hit candle",
            "execution": "market close at the next 15-minute open",
            "volume_ratio": "TP1 candle volume / median prior 20 completed candles",
            "selection": [*frozen.SELECT, *frozen.CONFIRM],
            "sealed_holdouts": [*frozen.VALIDATION, *frozen.LATER,
                                "direct X-Perp through Oct31"],
            "position_fraction": "client setting",
        },
        "candidate_count": len(frontier),
        "development_pass_count": len(eligible),
        "selected_before_holdouts": selected,
        "baseline": {
            "selection_2022h1": compact(base_selection),
            "confirmation_2022h2_2023": compact(base_confirmation),
            "validation_2024": compact(base_validation),
            "later_2025_2026": compact(base_later),
            "direct_xperp": baseline_direct,
        },
        "validation_2024": compact(validation) if validation else None,
        "later_2025_2026": compact(later) if later else None,
        "direct_xperp": candidate_direct,
        "historical_holdout_pass": historical_pass,
        "direct_pass": direct_pass,
        "sealed_evidence_pass": sealed_evidence_pass,
        "forward_gate": forward_gate,
        "development_frontier": sorted(
            frontier, key=lambda item: (item["development_pass"], item["score"]),
            reverse=True),
        "sources": {str(path): digest(path) for path in (
            *historical_paths.values(), *direct_paths.values(),
            prior.OUT, latest.OUT, NEW_DIRECT_ROWS, Path(__file__).resolve())},
        "limitations": [
            "The newest append-only slice is too small to activate a live-order change.",
            "Partial PnL is conservatively compounded at the final runner exit.",
            "Public OHLC cannot reproduce market-order slippage.",
            "Gross returns exclude fees at user request.",
        ],
    }
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({
        "status": report["status"], "selected": selected,
        "historical_holdout_pass": historical_pass,
        "baseline_direct": baseline_direct,
        "candidate_direct": candidate_direct,
        "direct_pass": direct_pass,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
