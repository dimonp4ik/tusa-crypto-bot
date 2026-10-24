"""Oct-24 append for the frozen October venue-adaptive target router.

The October model was fitted before September and its threshold/portfolio
rails were selected on September.  This audit keeps that policy fixed, checks
the original October replay exactly, verifies overlapping raw rows against a
later X-Perp snapshot, and appends only newly observable October outcomes.
"""
from __future__ import annotations

import collections
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

import crypto_profit_first_walkforward_target_router_oct18_audit as frozen
from crypto_runtime_shadow_portfolio_oct3_audit import money
from src.research_profit_gate import economic_gate


ROOT = Path("reports/audit_2026_09_08")
SOURCE_REPORT = ROOT / "crypto_profit_first_walkforward_target_router_oct18.json"
NEW_NEAR = ROOT / "crypto_xperp_raw_router_audit_0_25_oct24_material_raw_raw_families.csv"
NEW_FAR = ROOT / "crypto_xperp_raw_router_audit_2_0_oct24_material_raw_raw_families.csv"
NEW_CACHE = ROOT / "xperp_15m_oct24_material"
NEW_MANIFEST = NEW_CACHE / "manifest.json"
OUT = ROOT / "crypto_walkforward_target_router_oct24_forward.json"
ROWS_OUT = ROOT / "crypto_walkforward_target_router_oct24_forward.pkl"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def direct_wide(near: Path, far: Path, cache: Path) -> pd.DataFrame:
    old_near, old_far = frozen.prior.FRESH_NEAR, frozen.prior.FRESH_FAR
    old_cache = frozen.be.DIRECT_CACHE
    frozen.prior.FRESH_NEAR = near
    frozen.prior.FRESH_FAR = far
    frozen.be.DIRECT_CACHE = cache
    try:
        return frozen.direct_wide()
    finally:
        frozen.prior.FRESH_NEAR = old_near
        frozen.prior.FRESH_FAR = old_far
        frozen.be.DIRECT_CACHE = old_cache


def month(frame: pd.DataFrame) -> pd.Series:
    return pd.to_datetime(
        frame.entry_time, unit="s", utc=True).dt.strftime("%Y-%m")


def policy_from_report(report: dict) -> dict:
    step = next(item for item in report["venue_adaptive_walkforward"]
                if item["test_month"] == "2026-10")
    return step["policy"], step["test_metrics"]


def metrics(rows: list[dict], margin_fraction: float) -> dict:
    value = money(rows, margin_fraction)
    counts = collections.Counter(float(row["target_r"]) for row in rows)
    value.update({
        "target_counts": {
            str(target): counts[target] for target in frozen.TARGETS},
        "target_hit_rate": round(100 * sum(
            row["target_outcome"] == "TP2" for row in rows) / len(rows), 2)
            if rows else 0.0,
        "non_loss_rate": round(100 * sum(
            float(row["net_r"]) >= 0 for row in rows) / len(rows), 2)
            if rows else 0.0,
        "max_entries_same_timestamp": max(collections.Counter(
            int(float(row["entry_time"])) for row in rows).values())
            if rows else 0,
    })
    value["economic_gate"] = economic_gate(
        [float(row["signed_price_return"]) for row in rows],
        float(value["trades_per_calendar_day"]),
        margin_fraction=margin_fraction)
    return value


def accepted(frame: pd.DataFrame, policy: dict) -> list[dict]:
    return frozen.be.ranked_runtime_gate(
        frozen.candidate_rows(frame, float(policy["threshold"])),
        loss_pause=int(policy["loss_pause"]),
        hourly_cap=int(policy["hourly_cap"]))


def assert_previous(value: dict, expected: dict) -> None:
    for field in ("n", "return_pct", "dd_pct", "wr",
                  "trades_per_calendar_day", "negative_months"):
        if value[field] != expected[field]:
            raise RuntimeError(
                f"prior October replay mismatch {field}: "
                f"{value[field]} != {expected[field]}")


def overlap_parity(old: pd.DataFrame, new: pd.DataFrame) -> dict:
    keys = list(frozen.KEYS)
    old = old.copy()
    new = new.copy()
    old["month"] = month(old)
    new["month"] = month(new)
    old = old[old.month.isin(("2026-09", "2026-10"))]
    new = new[new.month.isin(("2026-09", "2026-10"))]
    common_end = int(old.entry_time.max())
    new = new[new.entry_time <= common_end]
    fields = [*keys, "entry", "sl", "risk_pct"]
    for target in frozen.TARGETS:
        suffix = str(target).replace(".", "_")
        fields.extend((f"net_r_{suffix}", f"outcome_{suffix}",
                       f"exit_time_{suffix}"))
    a = old[fields].sort_values(keys).reset_index(drop=True)
    b = new[fields].sort_values(keys).reset_index(drop=True)
    joined = a.merge(b, on=keys, suffixes=("_old", "_new"),
                     validate="one_to_one")
    mismatch = 0
    for field in fields:
        if field in keys:
            continue
        left, right = joined[f"{field}_old"], joined[f"{field}_new"]
        if pd.api.types.is_numeric_dtype(left):
            mismatch += int((~np.isclose(
                left, right, rtol=0, atol=1e-12, equal_nan=True)).sum())
        else:
            mismatch += int((left.fillna("") != right.fillna("")).sum())
    passed = bool(len(a) == len(b) == len(joined) and mismatch == 0)
    if not passed:
        raise RuntimeError({
            "old_rows": len(a), "new_rows": len(b),
            "matched_rows": len(joined), "value_mismatches": mismatch})
    return {
        "through_entry_time": common_end,
        "through_entry_utc": pd.Timestamp(
            common_end, unit="s", tz="UTC").isoformat(),
        "old_rows": len(a), "new_rows": len(b),
        "matched_rows": len(joined), "value_mismatches": mismatch,
        "passes": True,
    }


def row_key(row: dict) -> tuple:
    return tuple(row[key] for key in frozen.KEYS)


def main() -> None:
    source = json.loads(SOURCE_REPORT.read_text(encoding="utf-8"))
    policy, expected_october = policy_from_report(source)
    historical = frozen.historical_wide()
    historical_fit = historical[~historical.period.eq("2026")].copy()

    old_direct = direct_wide(
        frozen.prior.FRESH_NEAR, frozen.prior.FRESH_FAR,
        frozen.be.DIRECT_CACHE)
    new_direct = direct_wide(NEW_NEAR, NEW_FAR, NEW_CACHE)
    parity = overlap_parity(old_direct, new_direct)
    old_direct["month"] = month(old_direct)
    new_direct["month"] = month(new_direct)

    older_direct = old_direct[old_direct.month < "2026-09"].copy()
    train = pd.concat([historical_fit, older_direct], ignore_index=True)
    old_scoring = old_direct[old_direct.month.isin(
        ("2026-09", "2026-10"))].copy().reset_index(drop=True)
    new_scoring = new_direct[new_direct.month.isin(
        ("2026-09", "2026-10"))].copy().reset_index(drop=True)
    old_routed = frozen.route(
        old_scoring, frozen.fit_models_on_rows(train, old_scoring))
    new_routed = frozen.route(
        new_scoring, frozen.fit_models_on_rows(train, new_scoring))

    old_october = old_routed[old_routed.month.eq("2026-10")].copy()
    new_october = new_routed[new_routed.month.eq("2026-10")].copy()
    old_rows = accepted(old_october, policy)
    old_metrics = metrics(old_rows, .03)
    assert_previous(old_metrics, expected_october)
    new_rows = accepted(new_october, policy)

    old_keys = {row_key(row) for row in old_rows}
    new_prefix = [row for row in new_rows
                  if int(float(row["entry_time"])) <= parity["through_entry_time"]]
    new_prefix_keys = {row_key(row) for row in new_prefix}
    if old_keys != new_prefix_keys:
        raise RuntimeError(
            f"accepted prefix drift: old={len(old_keys)} new={len(new_prefix_keys)}")
    incremental = [row for row in new_rows if row_key(row) not in old_keys]
    all_results = {f"margin_{int(margin * 100)}pct": metrics(new_rows, margin)
                   for margin in (.03, .05)}
    incremental_results = {
        f"margin_{int(margin * 100)}pct": metrics(incremental, margin)
        for margin in (.03, .05)}
    pd.DataFrame(new_rows).to_pickle(ROWS_OUT)

    objective = bool(
        all_results["margin_3pct"]["economic_gate"]["passes"]
        and all_results["margin_3pct"]["negative_months"] == 0
        and all_results["margin_3pct"]["dd_pct"] <= 10)
    report = {
        "status": ("OCTOBER_TARGET_ROUTER_OBJECTIVE_PASSED_AWAIT_MORE_TIME"
                   if objective else "OCTOBER_TARGET_ROUTER_OBJECTIVE_NOT_MET"),
        "ready_for_live_orders": False,
        "order_eligible": False,
        "protocol": {
            "fit": "historical through 2025 plus X-Perp before September",
            "selection": "September 2026 only",
            "test": "October 2026, append-only through Oct 24",
            "policy": {key: policy[key] for key in (
                "quantile", "threshold", "loss_pause", "hourly_cap")},
            "targets_r": list(frozen.TARGETS),
            "target_price_move_range_pct": [
                100 * frozen.MIN_TARGET_MOVE, 100 * frozen.MAX_TARGET_MOVE],
            "entry": "next 15-minute market open",
            "exit": "single stop-first OHLC path, fixed selected target",
            "orders": "market entry and market close",
            "retune_on_october": False,
        },
        "integrity": {
            "raw_overlap": parity,
            "prior_october_reconstruction": old_metrics,
            "accepted_prefix_rows_verified": len(old_keys),
        },
        "october_to_date": all_results,
        "incremental_after_prior_snapshot": incremental_results,
        "incremental_rows": len(incremental),
        "objective_pass": objective,
        "sources": {
            str(SOURCE_REPORT): digest(SOURCE_REPORT),
            str(NEW_NEAR): digest(NEW_NEAR),
            str(NEW_FAR): digest(NEW_FAR),
            str(NEW_MANIFEST): digest(NEW_MANIFEST),
            str(Path(__file__).resolve()): digest(Path(__file__).resolve()),
        },
        "limitations": [
            "October is one partial out-of-sample month.",
            "Public candles cannot reproduce order-book slippage or latency.",
            "Gross results exclude fees at user request.",
            "The economic gate is a diagnostic projection, not a guarantee.",
        ],
    }
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({
        "status": report["status"], "integrity": report["integrity"],
        "october_to_date": all_results,
        "incremental_after_prior_snapshot": incremental_results,
        "objective_pass": objective,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
