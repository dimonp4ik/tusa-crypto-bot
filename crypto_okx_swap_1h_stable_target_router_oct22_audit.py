"""Coarse 0.5%/2% target router for the frozen stable hourly entries."""
from __future__ import annotations

import hashlib
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
import crypto_okx_swap_1h_stable_material_exit_oct22_audit as material


ROOT = Path("reports/audit_2026_09_08")
FREQUENCY_ROWS = ROOT / "crypto_okx_swap_1h_frequency_oct22.pkl"
STABLE_REPORT = ROOT / "crypto_okx_swap_1h_stable_direction_oct22.json"
MATERIAL_REPORT = ROOT / "crypto_okx_swap_1h_stable_material_exit_oct22.json"
OUT = ROOT / "crypto_okx_swap_1h_stable_target_router_oct22.json"
ROWS_OUT = ROOT / "crypto_okx_swap_1h_stable_target_router_oct22.pkl"
POLICIES = (
    "all_2pct",
    "stable_symbols_2pct",
    "asia_us_2pct",
    "soft_regime_2pct",
    "short_soft_regime_2pct",
    "short_soft_or_asia_us_2pct",
    "core_families_2pct",
)
MATERIAL_SYMBOLS = frozenset({
    "AAVEUSDT", "AVAXUSDT", "ETHUSDT", "LINKUSDT", "SOLUSDT", "XRPUSDT",
})


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def target_mask(frame: pd.DataFrame, policy: str) -> np.ndarray:
    hour = pd.to_datetime(frame.entry_time, unit="s", utc=True).dt.hour
    asia_us = (hour <= 5) | ((hour >= 12) & (hour <= 17))
    soft = frame.btc_ret168.ge(-.05) & frame.btc_ret168.lt(.05)
    primary_family = stable.family(frame)
    masks = {
        "all_2pct": np.ones(len(frame), dtype=bool),
        "stable_symbols_2pct": frame.symbol.isin(MATERIAL_SYMBOLS).to_numpy(),
        "asia_us_2pct": asia_us.to_numpy(),
        "soft_regime_2pct": soft.to_numpy(),
        "short_soft_regime_2pct": (frame.direction.lt(0) & soft).to_numpy(),
        "short_soft_or_asia_us_2pct": (
            (frame.direction.lt(0) & soft) | asia_us).to_numpy(),
        "core_families_2pct": primary_family.isin(
            ("pullback", "trend", "breadth", "breakout")).to_numpy(),
    }
    if policy not in masks:
        raise ValueError(f"unknown target policy: {policy}")
    return masks[policy]


def mix(fixed: pd.DataFrame, material_rows: pd.DataFrame,
        mask: np.ndarray) -> pd.DataFrame:
    result = fixed.copy()
    for column in ("signed_return", "exit_time", "target_hit", "exit_reason",
                   "selected_target_pct"):
        result.loc[mask, column] = material_rows.loc[mask, column].to_numpy()
    return result


def eligible(value: dict) -> bool:
    return bool(value["n"] >= 500 and value["return_pct"] > 0
                and value["dd_pct"] <= 10
                and value["trades_per_calendar_day"] >= 1.75)


def main() -> None:
    stable_report = json.loads(STABLE_REPORT.read_text(encoding="utf-8"))
    material_report = json.loads(MATERIAL_REPORT.read_text(encoding="utf-8"))
    route = stable_report["selected_before_direct"]
    base_entries = stable.apply_policy(pd.read_pickle(FREQUENCY_ROWS), route)
    candles, _, _ = legacy.load_merged(
        (append.OLD_GLOBAL, forward.GLOBAL_MANIFEST), legacy.GLOBAL_FIELDS, 3600)
    fixed = material.add_hourly_outcomes(base_entries, candles, .5, 12)
    two = material.add_hourly_outcomes(base_entries, candles, 2.0, 12)
    candidates = []
    routed = {}
    for policy in POLICIES:
        frame = mix(fixed, two, target_mask(base_entries, policy))
        routed[policy] = frame
        development = {
            str(year): portfolio.evaluate(frame, (year,), 0)
            for year in (2022, 2023)
        }
        if all(eligible(value) for value in development.values()):
            candidates.append({
                "policy": policy,
                "development": development,
                "rank": [
                    min(value["return_pct"] for value in development.values()),
                    sum(value["return_pct"] for value in development.values()),
                    -max(value["dd_pct"] for value in development.values()),
                ],
            })
    if not candidates:
        raise RuntimeError("no target router passed development rails")
    candidates.sort(key=lambda item: tuple(item["rank"]), reverse=True)
    selected = candidates[0]
    selected_frame = routed[selected["policy"]]
    later = {str(year): portfolio.evaluate(selected_frame, (year,), 0)
             for year in (2024, 2025, 2026)}
    later_passed = all(eligible(value) for value in later.values())

    raw = base.build_features(candles)
    entries = forward.frozen_entries(base.proposals(raw).reset_index(drop=True))
    mask = target_mask(entries, selected["policy"])
    entries = entries.copy()
    entries["routed_target_pct"] = np.where(mask, 2.0, .5)
    xperp, xperp_sources, coverage = legacy.load_merged(
        (append.OLD_XPERP, forward.XPERP_MANIFEST), legacy.XPERP_FIELDS, 300)
    measured_parts = []
    for target, part in entries.groupby("routed_target_pct", sort=True):
        measured_parts.append(material.resolve_available(
            part, xperp, float(target), 12 * 12))
    measured = pd.concat(measured_parts, ignore_index=True)
    accepted = append.runtime_gate_pending(append.emission_rows(measured))
    cutoff = int(pd.Timestamp(
        stable_report["direct_xperp"]["last_entry_utc"]).timestamp())
    complete = append.complete(accepted)
    prior = [row for row in complete if int(row["entry_time"]) <= cutoff]
    fresh = [row for row in complete if int(row["entry_time"]) > cutoff]
    pending = [row for row in accepted if row["outcome"] == "PENDING"]
    prior_stats, fresh_stats, cumulative_stats = (
        append.stats(prior), append.stats(fresh), append.stats(complete))
    all2_fresh = material_report["fresh_post_cutoff"]
    all2_cumulative = material_report["cumulative_direct"]
    direct_passed = bool(cumulative_stats["n"] >= 100
                         and cumulative_stats["return_pct"] > 0
                         and cumulative_stats["dd_pct"] <= 10)
    fresh_passed = bool(fresh_stats["n"] >= 20
                        and fresh_stats["return_pct"] > 0
                        and fresh_stats["dd_pct"] <= 5)
    improves_all2 = bool(
        fresh_stats["return_pct"] >= all2_fresh["return_pct"]
        and cumulative_stats["return_pct"] >= all2_cumulative["return_pct"]
        and fresh_stats["dd_pct"] <= all2_fresh["dd_pct"]
        and cumulative_stats["dd_pct"] <= all2_cumulative["dd_pct"])
    passed = later_passed and direct_passed and fresh_passed and improves_all2
    pd.DataFrame(accepted).to_pickle(ROWS_OUT)
    report = {
        "status": ("TARGET_ROUTER_FORWARD_GATE_PASSED_AWAIT_MORE_TIME"
                   if passed else "TARGET_ROUTER_REJECTED"),
        "ready_for_live_orders": False,
        "order_eligible": False,
        "protocol": {
            "entry_route": route,
            "target_policy_development": [2022, 2023],
            "untouched_calendar_checks": [2024, 2025, 2026],
            "targets_pct": [.5, 2.0],
            "stop_pct": 1.0,
            "hold_hours": 12,
            "retune_on_later_years_or_direct": False,
            "money": "client-selected current-equity margin, fixed 10x gross",
        },
        "policy_universe": list(POLICIES),
        "qualified_policies": candidates,
        "selected_before_later_years_and_direct": selected,
        "later_calendar_checks": later,
        "later_calendar_checks_passed": later_passed,
        "direct_pre_cutoff": prior_stats,
        "fresh_post_cutoff": fresh_stats,
        "pending_rows": len(pending),
        "cumulative_direct": cumulative_stats,
        "all_2pct_comparator": {
            "fresh_post_cutoff": all2_fresh,
            "cumulative_direct": all2_cumulative,
            "router_improves_return_and_drawdown": improves_all2,
        },
        "gates": {
            "later_years_passed": later_passed,
            "direct_passed": direct_passed,
            "fresh_passed": fresh_passed,
            "improves_all_2pct": improves_all2,
            "ready_for_live_orders": False,
        },
        "coverage": {"xperp": coverage},
        "sources": {
            **xperp_sources,
            str(FREQUENCY_ROWS): digest(FREQUENCY_ROWS),
            str(STABLE_REPORT): digest(STABLE_REPORT),
            str(MATERIAL_REPORT): digest(MATERIAL_REPORT),
            str(forward.GLOBAL_MANIFEST): digest(forward.GLOBAL_MANIFEST),
            str(forward.XPERP_MANIFEST): digest(forward.XPERP_MANIFEST),
            str(Path(__file__).resolve()): digest(Path(__file__).resolve()),
        },
        "limitations": [
            "The policy universe is deliberately coarse and fixed before direct replay.",
            "The genuinely new forward interval is only about two weeks.",
            "Gross results exclude fees at user request.",
            "No historical or short-forward result guarantees a monthly return.",
        ],
    }
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({
        "status": report["status"], "selected": selected,
        "later": later, "direct_pre_cutoff": prior_stats,
        "fresh_post_cutoff": fresh_stats, "pending_rows": len(pending),
        "cumulative_direct": cumulative_stats, "gates": report["gates"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
