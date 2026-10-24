"""Append-only Oct-24 monitor for the frozen stable 2% OKX 1H route.

This module does not fit or select anything.  It replays the production-shadow
entry policy and fixed exit against a later immutable public-data snapshot,
checks that every previously completed observation is reproduced, and reports
only the additional forward evidence.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

import crypto_okx_swap_1h_forward_append_oct8_audit as legacy
import crypto_okx_swap_1h_forward_append_oct16_audit as append
import crypto_okx_swap_1h_regime_oct7_audit as base
import crypto_okx_swap_1h_stable_direction_forward_oct22_audit as entries
import crypto_okx_swap_1h_stable_material_exit_oct22_audit as frozen
from crypto_okx_swap_1h_forward_append_oct9_audit import cluster_diagnostics
from src import crypto_okx_1h_shadow_router as router


ROOT = Path("reports/audit_2026_09_08")
GLOBAL_MANIFEST = ROOT / "crypto_okx_swap_1h_expanded_oct24" / "manifest.json"
XPERP_MANIFEST = ROOT / "xperp_5m_oct24_forward" / "manifest.json"
PRIOR_REPORT = ROOT / "crypto_okx_swap_1h_stable_material_exit_oct22.json"
PRIOR_ROWS = ROOT / "crypto_okx_swap_1h_stable_material_exit_oct22.pkl"
OUT = ROOT / "crypto_okx_swap_1h_stable_material_exit_oct24_forward.json"
ROWS_OUT = ROOT / "crypto_okx_swap_1h_stable_material_exit_oct24_forward.pkl"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def row_key(row: dict) -> tuple[str, int]:
    return str(row["symbol"]), int(row["entry_time"])


def assert_frozen_runtime(report: dict) -> None:
    selected = report["selected_before_later_years_and_direct"]
    expected = {
        "route_policy": "short_plus_long_pullback_or_avax",
        "target_pct": 2.0,
        "stop_pct": 1.0,
        "hold_hours": 12,
    }
    observed = {
        "route_policy": str(router._POLICY["route_policy"]),
        "target_pct": float(router.TARGET_PCT),
        "stop_pct": float(router.STOP_PCT),
        "hold_hours": int(router.HOLD_BARS),
    }
    if observed != expected:
        raise RuntimeError(f"runtime is not the frozen material route: {observed}")
    selected_observed = {
        "target_pct": float(selected["target_pct"]),
        "stop_pct": float(selected["stop_pct"]),
        "hold_hours": int(selected["hold_hours"]),
    }
    if selected_observed != {key: expected[key] for key in selected_observed}:
        raise RuntimeError(f"source exit profile changed: {selected_observed}")


def verify_prior_rows(prior: pd.DataFrame, refreshed: pd.DataFrame) -> dict:
    old = {row_key(row): row for row in prior.to_dict("records")}
    new = {row_key(row): row for row in refreshed.to_dict("records")}
    missing = sorted(set(old) - set(new))
    if missing:
        raise RuntimeError(f"previous accepted rows disappeared: {missing[:5]}")
    fields = ("direction", "outcome", "exit_time", "target_hit",
              "signed_price_return", "selected_target_pct")
    mismatches = []
    for key, left in old.items():
        right = new[key]
        for field in fields:
            if left[field] != right[field]:
                mismatches.append((key, field, left[field], right[field]))
    if mismatches:
        raise RuntimeError(f"previous observation changed: {mismatches[:5]}")
    return {
        "prior_rows_verified": len(old),
        "prior_row_fields_verified": list(fields),
        "missing_rows": 0,
        "mismatched_fields": 0,
    }


def main() -> None:
    prior_report = json.loads(PRIOR_REPORT.read_text(encoding="utf-8"))
    assert_frozen_runtime(prior_report)

    global_candles, global_sources, global_coverage = legacy.load_merged(
        (append.OLD_GLOBAL, GLOBAL_MANIFEST), legacy.GLOBAL_FIELDS, 3600)
    xperp_candles, xperp_sources, xperp_coverage = legacy.load_merged(
        (append.OLD_XPERP, XPERP_MANIFEST), legacy.XPERP_FIELDS, 300)

    raw = base.build_features(global_candles)
    proposed = base.proposals(raw).reset_index(drop=True)
    selected_entries = entries.frozen_entries(proposed)
    measured = frozen.resolve_available(
        selected_entries, xperp_candles, router.TARGET_PCT,
        int(router.HOLD_BARS * 12))
    accepted = append.runtime_gate_pending(append.emission_rows(measured))
    completed = append.complete(accepted)
    refreshed = pd.DataFrame(accepted)
    prior = pd.read_pickle(PRIOR_ROWS)
    integrity = verify_prior_rows(prior, refreshed)

    prior_keys = {row_key(row) for row in prior.to_dict("records")}
    additional_all = [row for row in accepted if row_key(row) not in prior_keys]
    additional_completed = append.complete(additional_all)
    additional_pending = [row for row in additional_all
                          if row["outcome"] == "PENDING"]
    cumulative_stats = append.stats(completed)
    incremental_stats = append.stats(additional_completed)
    clusters = cluster_diagnostics(additional_completed)
    cluster_share = (100 * sum(
        count for size, count in clusters["same_direction_hour_size_distribution"].items()
        if int(size) > 1) / max(1, clusters["distinct_direction_hours"]))
    clusters["multi_entry_direction_hour_share_pct"] = round(cluster_share, 4)

    refreshed.to_pickle(ROWS_OUT)
    objective_pass = bool(cumulative_stats["objective_pass"])
    report = {
        "status": ("FROZEN_FORWARD_OBJECTIVE_PASSED_AWAIT_MORE_TIME"
                   if objective_pass else "FROZEN_FORWARD_OBJECTIVE_NOT_MET"),
        "ready_for_live_orders": False,
        "order_eligible": False,
        "protocol": {
            "entry_policy": str(router._POLICY["route_policy"]),
            "quality_threshold": router.QUALITY_THRESHOLD,
            "exit": "fixed 2% target, 1% stop, 12-hour maximum, stop-first",
            "execution": "same hourly timestamp X-Perp market open",
            "retune_on_forward": False,
            "money": "client-selected current-equity margin; diagnostics use 3% and 10x",
            "prior_checkpoint_last_entry_utc": prior_report[
                "cumulative_direct"]["last_entry_utc"],
        },
        "integrity": integrity,
        "additional_forward": {
            "completed": incremental_stats,
            "completed_rows": len(additional_completed),
            "pending_rows": len(additional_pending),
            "clusters": clusters,
        },
        "cumulative_direct": cumulative_stats,
        "objective_pass": objective_pass,
        "coverage": {"global": global_coverage, "xperp": xperp_coverage},
        "sources": {
            **global_sources,
            **xperp_sources,
            str(GLOBAL_MANIFEST): digest(GLOBAL_MANIFEST),
            str(XPERP_MANIFEST): digest(XPERP_MANIFEST),
            str(PRIOR_REPORT): digest(PRIOR_REPORT),
            str(PRIOR_ROWS): digest(PRIOR_ROWS),
            str(router.MODEL_PATH): digest(router.MODEL_PATH),
            str(router.POLICY_PATH): digest(router.POLICY_PATH),
            str(Path(__file__).resolve()): digest(Path(__file__).resolve()),
        },
        "limitations": [
            "The appended interval is only about three calendar days.",
            "Public candles cannot reproduce order-book slippage or latency.",
            "Gross results exclude fees at user request.",
            "The economic gate is a diagnostic projection, not a return forecast.",
        ],
    }
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({
        "status": report["status"],
        "integrity": integrity,
        "additional_forward": report["additional_forward"],
        "cumulative_direct": cumulative_stats,
        "objective_pass": objective_pass,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
