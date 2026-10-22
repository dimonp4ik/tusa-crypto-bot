"""Material fixed-target exit for the frozen stable OKX 1H route."""
from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

import crypto_5m_regime_router_literal_oct6_audit as portfolio
import crypto_okx_swap_1h_forward_append_oct8_audit as legacy
import crypto_okx_swap_1h_forward_append_oct16_audit as append
import crypto_okx_swap_1h_regime_oct7_audit as base
import crypto_okx_swap_1h_stable_direction_oct22_audit as stable
import crypto_okx_swap_1h_stable_direction_forward_oct22_audit as forward
import crypto_okx_swap_1h_target_cascade_oct7_audit as cascade
import crypto_okx_swap_1h_xperp_transfer_oct7_audit as transfer


ROOT = Path("reports/audit_2026_09_08")
FREQUENCY_ROWS = ROOT / "crypto_okx_swap_1h_frequency_oct22.pkl"
STABLE_REPORT = ROOT / "crypto_okx_swap_1h_stable_direction_oct22.json"
FIXED_FORWARD_REPORT = ROOT / "crypto_okx_swap_1h_stable_direction_forward_oct22.json"
OUT = ROOT / "crypto_okx_swap_1h_stable_material_exit_oct22.json"
ROWS_OUT = ROOT / "crypto_okx_swap_1h_stable_material_exit_oct22.pkl"
TARGETS = (.75, 1.0, 1.5, 2.0)
HOLDS_1H = (12, 24, 48)
STOP_PCT = 1.0


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def add_hourly_outcomes(frame: pd.DataFrame, candles: dict, target: float,
                        hold: int) -> pd.DataFrame:
    parts = []
    for symbol, part in frame.groupby("symbol", sort=False):
        candle = candles[symbol]
        arrays = [np.asarray(candle[field], dtype=float)
                  for field in ("open", "high", "low", "close")]
        values, exits, hits, reasons = cascade.resolve(
            *arrays, np.asarray(candle["time"], dtype=np.int64),
            part.entry_index.to_numpy(np.int64),
            part.direction.to_numpy(np.int8), target, hold)
        found = part.drop(columns=[name for name in (
            "signed_return", "exit_time", "target_hit", "exit_reason")
            if name in part]).copy()
        found["signed_return"] = values
        found["exit_time"] = exits
        found["target_hit"] = hits.astype(bool)
        found["exit_reason"] = reasons
        found["selected_target_pct"] = target
        found["quality"] = 1.0
        parts.append(found)
    return pd.concat(parts, ignore_index=True)


def metrics(frame: pd.DataFrame, year: int) -> dict:
    return portfolio.evaluate(frame, (year,), 0)


def eligible(value: dict) -> bool:
    return bool(value["n"] >= 500 and value["return_pct"] > 0
                and value["dd_pct"] <= 10
                and value["trades_per_calendar_day"] >= 1.75)


def select_exit(entries: pd.DataFrame, candles: dict) -> tuple[dict, pd.DataFrame, list]:
    candidates = []
    cache = {}
    for target, hold in itertools.product(TARGETS, HOLDS_1H):
        resolved = add_hourly_outcomes(entries, candles, target, hold)
        cache[(target, hold)] = resolved
        development = {str(year): metrics(resolved, year) for year in (2022, 2023)}
        if all(eligible(value) for value in development.values()):
            candidates.append({
                "target_pct": target,
                "stop_pct": STOP_PCT,
                "hold_hours": hold,
                "development": development,
                "rank": [
                    min(value["return_pct"] for value in development.values()),
                    sum(value["return_pct"] for value in development.values()),
                    -max(value["dd_pct"] for value in development.values()),
                ],
                "_key": (target, hold),
            })
    if not candidates:
        raise RuntimeError("no material exit passed development rails")
    selected = max(candidates, key=lambda item: tuple(item["rank"]))
    diagnostics = [{key: value for key, value in item.items() if key != "_key"}
                   for item in sorted(
                       candidates, key=lambda value: tuple(value["rank"]), reverse=True)]
    return selected, cache[selected["_key"]], diagnostics


def resolve_available(entries: pd.DataFrame, candles: dict, target: float,
                      hold: int) -> pd.DataFrame:
    parts = []
    for symbol, part in entries.groupby("symbol", sort=False):
        candle = candles.get(symbol)
        if candle is None:
            continue
        times = np.asarray(candle["time"], dtype=np.int64)
        lookup = {int(stamp): index for index, stamp in enumerate(times)}
        found = part.copy()
        found["entry_index"] = found.entry_time.map(lookup)
        found = found.dropna(subset=["entry_index"]).copy()
        if found.empty:
            continue
        found["entry_index"] = found.entry_index.astype(np.int64)
        arrays = [np.asarray(candle[field], dtype=float)
                  for field in ("open", "high", "low", "close")]
        values, exits, hits, reasons = transfer.resolve_fixed(
            *arrays, times, found.entry_index.to_numpy(np.int64),
            found.direction.to_numpy(np.int8), target, STOP_PCT, hold)
        terminal = np.isin(reasons, np.asarray((0, 1), dtype=np.int8))
        full_path = found.entry_index.to_numpy(np.int64) + hold <= len(times)
        complete = terminal | full_path
        found["outcome_state"] = np.where(complete, "RESOLVED", "PENDING")
        found["signed_return"] = np.where(complete, values, np.nan)
        found["exit_time"] = np.where(
            complete, exits, found.entry_time.to_numpy(np.int64) + hold * 300)
        found["target_hit"] = np.where(complete, hits.astype(bool), False)
        found["exit_reason"] = np.where(complete, reasons, -1)
        found["selected_target_pct"] = target
        found["quality"] = found.get("model_quality", 1.0)
        parts.append(found)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def main() -> None:
    stable_report = json.loads(STABLE_REPORT.read_text(encoding="utf-8"))
    fixed_report = json.loads(FIXED_FORWARD_REPORT.read_text(encoding="utf-8"))
    policy = stable_report["selected_before_direct"]
    historical_entries = stable.apply_policy(pd.read_pickle(FREQUENCY_ROWS), policy)
    global_candles, _, _ = legacy.load_merged(
        (append.OLD_GLOBAL, forward.GLOBAL_MANIFEST), legacy.GLOBAL_FIELDS, 3600)
    selected, historical_resolved, diagnostics = select_exit(
        historical_entries, global_candles)
    later = {str(year): metrics(historical_resolved, year)
             for year in (2024, 2025, 2026)}
    later_passed = all(eligible(value) for value in later.values())

    raw = base.build_features(global_candles)
    entries = forward.frozen_entries(base.proposals(raw).reset_index(drop=True))
    xperp_candles, xperp_sources, xperp_coverage = legacy.load_merged(
        (append.OLD_XPERP, forward.XPERP_MANIFEST), legacy.XPERP_FIELDS, 300)
    hold_5m = int(selected["hold_hours"] * 12)
    measured = resolve_available(
        entries, xperp_candles, selected["target_pct"], hold_5m)
    accepted = append.runtime_gate_pending(append.emission_rows(measured))
    cutoff = int(pd.Timestamp(
        stable_report["direct_xperp"]["last_entry_utc"]).timestamp())
    completed = append.complete(accepted)
    prior = [row for row in completed if int(row["entry_time"]) <= cutoff]
    fresh = [row for row in completed if int(row["entry_time"]) > cutoff]
    pending = [row for row in accepted if row["outcome"] == "PENDING"]
    prior_stats = append.stats(prior)
    fresh_stats = append.stats(fresh)
    cumulative_stats = append.stats(completed)
    fixed_fresh = fixed_report["fresh_completed"]
    fixed_cumulative = fixed_report["cumulative_completed"]
    direct_passed = bool(
        cumulative_stats["n"] >= 100
        and cumulative_stats["return_pct"] > 0
        and cumulative_stats["dd_pct"] <= 10
        and cumulative_stats["negative_months"] <= 2)
    fresh_passed = bool(
        fresh_stats["n"] >= 20 and fresh_stats["return_pct"] > 0
        and fresh_stats["dd_pct"] <= 5)
    improves_fixed = bool(
        fresh_stats["return_pct"] > fixed_fresh["return_pct"]
        and cumulative_stats["return_pct"] > fixed_cumulative["return_pct"])
    pd.DataFrame(accepted).to_pickle(ROWS_OUT)

    public_selected = {key: value for key, value in selected.items()
                       if key != "_key"}
    passed = later_passed and direct_passed and fresh_passed and improves_fixed
    report = {
        "status": ("MATERIAL_EXIT_FORWARD_GATE_PASSED_AWAIT_MORE_TIME"
                   if passed else "MATERIAL_EXIT_REJECTED"),
        "ready_for_live_orders": False,
        "order_eligible": False,
        "protocol": {
            "entry_route": policy,
            "exit_development": [2022, 2023],
            "untouched_calendar_checks": [2024, 2025, 2026],
            "direct_transfer": "X-Perp 5-minute candles",
            "fresh_cutoff_utc": stable_report["direct_xperp"]["last_entry_utc"],
            "stop_pct": STOP_PCT,
            "collision_order": "stop before target",
            "retune_on_later_years_or_direct": False,
            "money": "client-selected current-equity margin, fixed 10x gross",
        },
        "exit_universe": {"target_pct": list(TARGETS), "hold_hours": list(HOLDS_1H)},
        "qualified_exit_profiles": len(diagnostics),
        "selected_before_later_years_and_direct": public_selected,
        "selection_diagnostics": diagnostics,
        "later_calendar_checks": later,
        "later_calendar_checks_passed": later_passed,
        "direct_pre_cutoff": prior_stats,
        "fresh_post_cutoff": fresh_stats,
        "pending_rows": len(pending),
        "cumulative_direct": cumulative_stats,
        "fixed_0_5_comparator": {
            "fresh_post_cutoff": fixed_fresh,
            "cumulative_direct": fixed_cumulative,
            "material_exit_improves_return": improves_fixed,
        },
        "gates": {
            "later_years_passed": later_passed,
            "direct_passed": direct_passed,
            "fresh_passed": fresh_passed,
            "improves_fixed_0_5_exit": improves_fixed,
            "ready_for_live_orders": False,
        },
        "coverage": {"xperp": xperp_coverage},
        "sources": {
            **xperp_sources,
            str(FREQUENCY_ROWS): digest(FREQUENCY_ROWS),
            str(STABLE_REPORT): digest(STABLE_REPORT),
            str(FIXED_FORWARD_REPORT): digest(FIXED_FORWARD_REPORT),
            str(forward.GLOBAL_MANIFEST): digest(forward.GLOBAL_MANIFEST),
            str(forward.XPERP_MANIFEST): digest(forward.XPERP_MANIFEST),
            str(Path(__file__).resolve()): digest(Path(__file__).resolve()),
        },
        "limitations": [
            "The genuinely new forward interval is only about two weeks.",
            "Public candles cannot reproduce order-book slippage or latency.",
            "Gross results exclude fees at user request.",
            "Historical or short-forward results cannot guarantee a monthly return.",
        ],
    }
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({
        "status": report["status"], "selected": public_selected,
        "later": later, "direct_pre_cutoff": prior_stats,
        "fresh_post_cutoff": fresh_stats, "pending_rows": len(pending),
        "cumulative_direct": cumulative_stats, "gates": report["gates"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
