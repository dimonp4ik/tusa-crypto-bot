"""Frozen runner exit for the stable OKX 1H direction route.

The exit profile is selected only on 2022-2023 global OKX candles, checked on
each later calendar year, and then transferred once to direct X-Perp 5-minute
candles.  The portfolio gate is replayed after changing the exit duration so
that overlapping positions are not inherited from the fixed-target result.
"""
from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

import crypto_5m_regime_router_literal_oct6_audit as portfolio
import crypto_expert_agreement_ladder_oct7_audit as ladder
import crypto_okx_swap_1h_forward_append_oct8_audit as legacy
import crypto_okx_swap_1h_forward_append_oct16_audit as append
import crypto_okx_swap_1h_regime_oct7_audit as base
import crypto_okx_swap_1h_runner_oct7_audit as hourly_runner
import crypto_okx_swap_1h_stable_direction_oct22_audit as stable
import crypto_okx_swap_1h_stable_direction_forward_oct22_audit as forward
import crypto_okx_swap_1h_xperp_transfer_oct7_audit as transfer


ROOT = Path("reports/audit_2026_09_08")
FREQUENCY_ROWS = ROOT / "crypto_okx_swap_1h_frequency_oct22.pkl"
STABLE_REPORT = ROOT / "crypto_okx_swap_1h_stable_direction_oct22.json"
FIXED_FORWARD_REPORT = (
    ROOT / "crypto_okx_swap_1h_stable_direction_forward_oct22.json")
OUT = ROOT / "crypto_okx_swap_1h_stable_runner_oct22.json"
ROWS_OUT = ROOT / "crypto_okx_swap_1h_stable_runner_oct22.pkl"
PARTIALS = (.25, .50, .75)
RUNNER_TARGETS = (1.0, 1.5, 2.0)
HOLDS_1H = (12, 24, 48)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def historical_stats(frame: pd.DataFrame, year: int) -> dict:
    value = portfolio.evaluate(frame, (year,), 0)
    selected = portfolio.portfolio_rows(frame, (year,), 0)
    value["runner_hit_rate"] = round(
        100 * sum(bool(row["runner_hit"]) for row in selected) / len(selected), 4
    ) if selected else 0.0
    return value


def eligible(value: dict) -> bool:
    return bool(
        value["n"] >= 500
        and value["return_pct"] > 0
        and value["literal_target_hit_rate"] >= 65
        and value["dd_pct"] <= 7
        and value["trades_per_calendar_day"] >= 2
    )


def select_exit(entries: pd.DataFrame, candles: dict) -> tuple[dict, pd.DataFrame, list]:
    candidates = []
    cache = {}
    for key in itertools.product(PARTIALS, RUNNER_TARGETS, HOLDS_1H):
        resolved = hourly_runner.add_outcomes(entries, candles, *key)
        cache[key] = resolved
        development = {str(year): historical_stats(resolved, year)
                       for year in (2022, 2023)}
        if all(eligible(value) for value in development.values()):
            candidates.append({
                "partial_fraction": key[0],
                "runner_target_pct": key[1],
                "hold_hours": key[2],
                "development": development,
                "rank": [
                    min(value["return_pct"] for value in development.values()),
                    sum(value["return_pct"] for value in development.values()),
                    -max(value["dd_pct"] for value in development.values()),
                ],
                "_key": key,
            })
    if not candidates:
        raise RuntimeError("no exit profile passed 2022-2023 development rails")
    selected = max(candidates, key=lambda item: tuple(item["rank"]))
    resolved = cache[selected["_key"]]
    diagnostics = []
    for item in sorted(candidates, key=lambda value: tuple(value["rank"]), reverse=True):
        diagnostics.append({key: value for key, value in item.items() if key != "_key"})
    return selected, resolved, diagnostics


def resolve_available(entries: pd.DataFrame, candles: dict, partial: float,
                      runner_target: float, hold: int) -> pd.DataFrame:
    """Resolve terminal runner paths and retain incomplete paths as pending."""
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
        values, exits, first_hits, runner_hits, reasons = ladder.resolve_ladder(
            *arrays, times, found.entry_index.to_numpy(np.int64),
            found.direction.to_numpy(np.int8), partial, runner_target, hold)
        terminal = np.isin(reasons, np.asarray((0, 1, 4), dtype=np.int8))
        full_path = found.entry_index.to_numpy(np.int64) + hold <= len(times)
        complete = terminal | full_path
        found["outcome_state"] = np.where(complete, "RESOLVED", "PENDING")
        found["signed_return"] = np.where(complete, values, np.nan)
        found["exit_time"] = np.where(
            complete, exits, found.entry_time.to_numpy(np.int64) + hold * 300)
        found["target_hit"] = np.where(complete, first_hits.astype(bool), False)
        found["runner_hit"] = np.where(complete, runner_hits.astype(bool), False)
        found["exit_reason"] = np.where(complete, reasons, -1)
        found["selected_target_pct"] = .5
        found["quality"] = found.get("model_quality", 1.0)
        parts.append(found)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def stats(rows: list[dict]) -> dict:
    value = append.stats(rows)
    value["runner_hits"] = sum(bool(row.get("runner_hit")) for row in rows)
    value["runner_hit_rate"] = round(
        100 * value["runner_hits"] / len(rows), 4) if rows else 0.0
    value["exit_reasons"] = {
        str(reason): sum(int(row["exit_reason"]) == reason for row in rows)
        for reason in (0, 1, 2, 3, 4)
    }
    return value


def main() -> None:
    stable_report = json.loads(STABLE_REPORT.read_text(encoding="utf-8"))
    fixed_report = json.loads(FIXED_FORWARD_REPORT.read_text(encoding="utf-8"))
    selected_policy = stable_report["selected_before_direct"]
    historical_entries = stable.apply_policy(
        pd.read_pickle(FREQUENCY_ROWS), selected_policy)
    global_candles, _, _ = legacy.load_merged(
        (append.OLD_GLOBAL, forward.GLOBAL_MANIFEST), legacy.GLOBAL_FIELDS, 3600)
    selected, historical_resolved, diagnostics = select_exit(
        historical_entries, global_candles)
    later = {str(year): historical_stats(historical_resolved, year)
             for year in (2024, 2025, 2026)}
    later_passed = all(eligible(value) for value in later.values())

    raw = base.build_features(global_candles)
    all_entries = forward.frozen_entries(base.proposals(raw).reset_index(drop=True))
    xperp_candles, xperp_sources, xperp_coverage = legacy.load_merged(
        (append.OLD_XPERP, forward.XPERP_MANIFEST), legacy.XPERP_FIELDS, 300)
    hold_5m = int(selected["hold_hours"] * 12)
    measured = resolve_available(
        all_entries, xperp_candles, selected["partial_fraction"],
        selected["runner_target_pct"], hold_5m)
    accepted = append.runtime_gate_pending(append.emission_rows(measured))
    cutoff = int(pd.Timestamp(
        stable_report["direct_xperp"]["last_entry_utc"]).timestamp())
    completed = append.complete(accepted)
    prior = [row for row in completed if int(row["entry_time"]) <= cutoff]
    fresh = [row for row in completed if int(row["entry_time"]) > cutoff]
    pending = [row for row in accepted if row["outcome"] == "PENDING"]
    prior_stats, fresh_stats, cumulative_stats = (
        stats(prior), stats(fresh), stats(completed))
    direct_passed = bool(
        cumulative_stats["n"] >= 150
        and cumulative_stats["return_pct"] > 0
        and cumulative_stats["literal_target_hit_rate"] >= 65
        and cumulative_stats["dd_pct"] <= 7
        and cumulative_stats["negative_months"] <= 2)
    fresh_passed = bool(
        fresh_stats["n"] >= 20
        and fresh_stats["return_pct"] > 0
        and fresh_stats["literal_target_hit_rate"] >= 65
        and fresh_stats["dd_pct"] <= 5)
    fixed_fresh = fixed_report["fresh_completed"]
    fixed_cumulative = fixed_report["cumulative_completed"]
    improves_fixed = bool(
        fresh_stats["return_pct"] >= fixed_fresh["return_pct"]
        and cumulative_stats["return_pct"] >= fixed_cumulative["return_pct"]
        and fresh_stats["dd_pct"] <= fixed_fresh["dd_pct"]
        and cumulative_stats["dd_pct"] <= fixed_cumulative["dd_pct"])
    pd.DataFrame(accepted).to_pickle(ROWS_OUT)

    public_selected = {key: value for key, value in selected.items()
                       if key != "_key"}
    report = {
        "status": (
            "RUNNER_FORWARD_GATE_PASSED_AWAIT_MORE_TIME"
            if later_passed and direct_passed and fresh_passed and improves_fixed
            else "RUNNER_REJECTED_DIRECT_UNDERPERFORMANCE"),
        "ready_for_live_orders": False,
        "order_eligible": False,
        "protocol": {
            "entry_route": selected_policy,
            "exit_development": [2022, 2023],
            "untouched_calendar_checks": [2024, 2025, 2026],
            "direct_transfer": "X-Perp 5-minute candles",
            "fresh_cutoff_utc": stable_report["direct_xperp"]["last_entry_utc"],
            "first_target_pct": .5,
            "initial_stop_pct": 1.0,
            "runner_stop": "entry price from the next 5-minute candle",
            "collision_order": "initial stop first; breakeven before runner",
            "retune_on_direct_or_fresh": False,
            "money": "client-selected current-equity margin, fixed 10x gross",
        },
        "exit_universe": {
            "partial_fractions": list(PARTIALS),
            "runner_targets_pct": list(RUNNER_TARGETS),
            "hold_hours": list(HOLDS_1H),
        },
        "qualified_exit_profiles": len(diagnostics),
        "selected_before_later_years_and_direct": public_selected,
        "selection_diagnostics": diagnostics,
        "later_calendar_checks": later,
        "later_calendar_checks_passed": later_passed,
        "direct_pre_cutoff": prior_stats,
        "fresh_post_cutoff": fresh_stats,
        "pending_rows": len(pending),
        "cumulative_direct": cumulative_stats,
        "fixed_exit_comparator": {
            "fresh_post_cutoff": fixed_fresh,
            "cumulative_direct": fixed_cumulative,
            "runner_improves_return_and_drawdown": improves_fixed,
        },
        "gates": {
            "direct_passed": direct_passed,
            "fresh_passed": fresh_passed,
            "improves_fixed_exit": improves_fixed,
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
        "status": report["status"],
        "selected": public_selected,
        "later": later,
        "direct_pre_cutoff": prior_stats,
        "fresh_post_cutoff": fresh_stats,
        "pending_rows": len(pending),
        "cumulative_direct": cumulative_stats,
        "gates": report["gates"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
