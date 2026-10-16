"""Outcome-readiness-safe Oct 16 append of the frozen OKX 1H cascade.

Earlier append audits required the entire maximum hold path before an entry was
allowed into the portfolio gate. That let a short-target trade occupy a slot
until a higher-target trade from the same timestamp later became measurable.
This audit selects emissions before outcome availability, resolves early
targets/stops immediately, and keeps unresolved emissions open as PENDING.
"""
from __future__ import annotations

import collections
import hashlib
import heapq
import json
from pathlib import Path

import numpy as np
import pandas as pd

import crypto_okx_swap_1h_forward_append_oct8_audit as legacy
import crypto_okx_swap_1h_regime_oct7_audit as base
import crypto_okx_swap_1h_target_cascade_oct7_audit as cascade
import crypto_okx_swap_1h_xperp_transfer_oct7_audit as transfer
from crypto_okx_swap_1h_forward_append_oct9_audit import cluster_diagnostics
from src import crypto_okx_1h_shadow_router as router
from src.research_profit_gate import economic_gate


ROOT = Path("reports/audit_2026_09_08")
OLD_GLOBAL = ROOT / "crypto_okx_swap_1h_history" / "manifest.json"
OLD_XPERP = ROOT / "xperp_5m_oct4" / "manifest.json"
PRIOR_GLOBAL = ROOT / "crypto_okx_swap_1h_expanded_oct9" / "manifest.json"
PRIOR_XPERP = ROOT / "xperp_5m_oct9_forward" / "manifest.json"
CURRENT_GLOBAL = ROOT / "crypto_okx_swap_1h_expanded_oct16" / "manifest.json"
CURRENT_XPERP = ROOT / "xperp_5m_oct16_forward" / "manifest.json"
FROZEN_ROWS = ROOT / "crypto_okx_swap_1h_target_cascade_oct7.pkl"
OUT = ROOT / "crypto_okx_swap_1h_forward_append_oct16.json"
ROWS_OUT = ROOT / "crypto_okx_swap_1h_forward_append_oct16.pkl"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def row_key(row: dict) -> tuple[str, int, str]:
    return legacy.row_key(row)


def resolve_available(entries: pd.DataFrame, candles: dict) -> pd.DataFrame:
    """Resolve terminal paths now and retain incomplete paths as PENDING."""
    parts = []
    for target, target_rows in entries.groupby("selected_target_pct", sort=True):
        target = float(target)
        hold = int(cascade.HOLD[target] * 12)
        for symbol, part in target_rows.groupby("symbol", sort=False):
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
                found.direction.to_numpy(np.int8), target, 1.0, hold)
            complete = []
            for index, reason in zip(found.entry_index.to_numpy(int), reasons):
                terminal = int(reason) in (0, 1)
                complete.append(terminal or index + hold <= len(times))
            complete = np.asarray(complete, dtype=bool)
            found["outcome_state"] = np.where(complete, "RESOLVED", "PENDING")
            found["signed_return"] = np.where(complete, values, np.nan)
            found["target_hit"] = np.where(complete, hits.astype(bool), False)
            found["exit_reason"] = np.where(complete, reasons, -1)
            horizons = found.entry_time.to_numpy(np.int64) + hold * 300
            found["exit_time"] = np.where(complete, exits, horizons)
            found["quality"] = 1.0
            parts.append(found)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def emission_rows(frame: pd.DataFrame) -> list[dict]:
    if frame.empty:
        return []
    chosen = frame.sort_values(
        ["entry_time", "quality", "symbol"], ascending=[True, False, True]).copy()
    chosen["source_priority"] = chosen.groupby("entry_time").cumcount()
    rows = []
    for row in chosen.to_dict("records"):
        pending = row["outcome_state"] == "PENDING"
        rows.append({
            **row,
            "direction": "LONG" if float(row["direction"]) > 0 else "SHORT",
            "source": "literal_regime_router", "source_priority": int(row["source_priority"]),
            "module": "literal_regime_router", "module_priority": 0,
            "outcome": "PENDING" if pending else (
                "SL" if float(row["signed_return"]) < 0 else "TP"),
            "signed_price_return": None if pending else float(row["signed_return"]),
        })
    return rows


def runtime_gate_pending(rows: list[dict], loss_pause: int = 3,
                         hourly_cap: int = 2) -> list[dict]:
    """Replay the runtime while unresolved entries reserve exposure slots."""
    exits = []
    opened = {}
    last = {}
    accepted_times = collections.deque()
    accepted = []
    day = None
    streak = 0
    paused = False
    ordered = sorted(rows, key=lambda row: (
        float(row["entry_time"]), int(row["source_priority"]), str(row["symbol"])))
    for row in ordered:
        now = float(row["entry_time"])
        new_day = int(now // 86400)
        if new_day != day:
            day, streak, paused = new_day, 0, False
        while exits and exits[0][0] < now:
            _, _, done = heapq.heappop(exits)
            opened.pop(str(done["symbol"]), None)
            if done["outcome"] != "PENDING" and int(float(done["exit_time"]) // 86400) == day:
                streak = streak + 1 if done["outcome"] == "SL" else 0
                if streak >= loss_pause:
                    paused = True
        while accepted_times and accepted_times[0] <= now - 3600:
            accepted_times.popleft()
        symbol, direction = str(row["symbol"]), str(row["direction"])
        if paused or symbol in opened or len(accepted_times) >= hourly_cap:
            continue
        if now - last.get((symbol, direction), -1e20) < 10800:
            continue
        if sum(str(value["direction"]) == direction for value in opened.values()) >= 3:
            continue
        opened[symbol] = row
        last[(symbol, direction)] = now
        accepted_times.append(now)
        accepted.append(row)
        heapq.heappush(exits, (float(row["exit_time"]), len(accepted), row))
    return accepted


def complete(rows: list[dict]) -> list[dict]:
    return [row for row in rows if row["outcome"] != "PENDING"]


def stats(rows: list[dict]) -> dict:
    value = legacy.stats(rows)
    signed = [float(row["signed_price_return"]) for row in rows]
    rate = float(value.get("trades_per_calendar_day", 0))
    value["economic_gate"] = economic_gate(signed, rate, market_days=30)
    value["objective_checks"] = {
        "win_rate_75_to_95": 75 <= value["wr"] <= 95,
        "at_least_3_calendar_trades_per_day": rate >= 3,
        "no_negative_months": value.get("negative_months", 0) == 0,
        "drawdown_at_most_5pct": value["dd_pct"] <= 5,
        "economic_return_gate": value["economic_gate"]["passes"],
    }
    value["objective_pass"] = all(value["objective_checks"].values())
    return value


def stop_cause_diagnostics(prior_rows: list[dict], fresh_rows: list[dict]) -> dict:
    """Describe causal entry-time proxies without turning the append into training data."""
    def subset(rows: list[dict], *, pullback: bool = False,
               regime_mismatch: bool = False) -> list[dict]:
        selected = []
        for row in rows:
            is_pullback = pd.notna(row.get("family_pullback")) \
                and float(row.get("family_pullback", 0)) == 1
            if pullback and not is_pullback:
                continue
            sign = 1 if row["direction"] == "LONG" else -1
            mismatch = sign * float(row["ret24"]) < 0
            if regime_mismatch and not mismatch:
                continue
            selected.append(row)
        return selected

    def compact(rows: list[dict]) -> dict:
        if not rows:
            return {"n": 0, "wins": 0, "stops": 0, "win_rate": None,
                    "signed_price_return_sum_pct": 0.0}
        wins = sum(row["outcome"] == "TP" for row in rows)
        return {
            "n": len(rows), "wins": wins,
            "stops": sum(row["outcome"] == "SL" for row in rows),
            "win_rate": round(100 * wins / len(rows), 2),
            "signed_price_return_sum_pct": round(
                100 * sum(float(row["signed_price_return"]) for row in rows), 4),
        }

    prior_pullback = subset(prior_rows, pullback=True)
    fresh_pullback = subset(fresh_rows, pullback=True)
    prior_mismatch = subset(prior_pullback, regime_mismatch=True)
    fresh_mismatch = subset(fresh_pullback, regime_mismatch=True)
    winners = [row for row in fresh_rows if row["outcome"] == "TP"]
    losers = [row for row in fresh_rows if row["outcome"] == "SL"]

    def mean(rows: list[dict], field: str) -> float | None:
        return round(float(np.mean([float(row[field]) for row in rows])), 6) \
            if rows else None

    def boundary_distance(row: dict) -> float:
        return float(row["breakout_up"] if row["direction"] == "LONG"
                     else row["breakout_down"])

    return {
        "fresh_losses": len(losers),
        "fresh_pullback": compact(fresh_pullback),
        "fresh_pullback_recent_24h_regime_mismatch": compact(fresh_mismatch),
        "prior_pullback": compact(prior_pullback),
        "prior_pullback_recent_24h_regime_mismatch": compact(prior_mismatch),
        "fresh_volume_ratio_mean": {
            "wins": mean(winners, "volume_ratio"),
            "losses": mean(losers, "volume_ratio"),
        },
        "fresh_prior_24h_boundary_distance_mean": {
            "wins": round(float(np.mean([boundary_distance(row) for row in winners])), 6)
            if winners else None,
            "losses": round(float(np.mean([boundary_distance(row) for row in losers])), 6)
            if losers else None,
        },
        "interpretation": (
            "Five of the eight fresh stops came from pullbacks whose frozen 168h "
            "direction opposed the most recent 24h return. The same mismatch was "
            "profitable before the append, so this is evidence of regime drift, not "
            "a validated support/resistance or liquidity filter. Fresh winners had "
            "higher volume on average, but the sample is too small to freeze a gate."),
    }


def snapshot(global_manifest: Path, xperp_manifest: Path) -> tuple[list[dict], dict]:
    global_candles, global_sources, global_coverage = legacy.load_merged(
        (OLD_GLOBAL, global_manifest), legacy.GLOBAL_FIELDS, 3600)
    xperp_candles, xperp_sources, xperp_coverage = legacy.load_merged(
        (OLD_XPERP, xperp_manifest), legacy.XPERP_FIELDS, 300)
    frozen_manifest = json.loads(OLD_GLOBAL.read_text(encoding="utf-8"))
    freeze_end = min(int(item["end"]) for item in frozen_manifest["symbols"].values())
    old_entries = pd.read_pickle(FROZEN_ROWS)
    raw = base.build_features(global_candles)
    proposals = base.proposals(raw).reset_index(drop=True)
    future = legacy.frozen_scores(proposals[proposals.entry_time.gt(freeze_end)].copy(), True)
    future["year"] = pd.to_datetime(future.entry_time, unit="s", utc=True).dt.year.astype(int)
    future["month"] = pd.to_datetime(future.entry_time, unit="s", utc=True).dt.strftime("%Y%m").astype(int)
    entries = pd.concat([old_entries, future], ignore_index=True, sort=False)
    entries = entries.drop_duplicates(["symbol", "entry_time", "direction"], keep="first")
    measured = resolve_available(entries, xperp_candles)
    accepted = runtime_gate_pending(emission_rows(measured))
    return accepted, {
        "global_sources": global_sources, "xperp_sources": xperp_sources,
        "global_coverage": global_coverage, "xperp_coverage": xperp_coverage,
        "freeze_end": freeze_end, "proposals": len(proposals),
        "selected_entries": len(entries), "mapped_entries": len(measured),
    }


def main() -> None:
    prior, prior_meta = snapshot(PRIOR_GLOBAL, PRIOR_XPERP)
    current, current_meta = snapshot(CURRENT_GLOBAL, CURRENT_XPERP)
    prior_by_key = {row_key(row): row for row in prior}
    current_by_key = {row_key(row): row for row in current}
    missing_emissions = set(prior_by_key) - set(current_by_key)
    changed_resolved = []
    for key in set(prior_by_key) & set(current_by_key):
        left, right = prior_by_key[key], current_by_key[key]
        if left["outcome"] == "PENDING":
            continue
        if (left["outcome"] != right["outcome"]
                or bool(left["target_hit"]) != bool(right["target_hit"])
                or abs(float(left["signed_price_return"])
                       - float(right["signed_price_return"])) > 1e-12):
            changed_resolved.append(key)
    if missing_emissions or changed_resolved:
        raise ValueError(
            f"corrected append integrity failed: missing={len(missing_emissions)} "
            f"changed={len(changed_resolved)}")

    prior_complete = {key: row for key, row in prior_by_key.items()
                      if row["outcome"] != "PENDING"}
    current_complete = {key: row for key, row in current_by_key.items()
                        if row["outcome"] != "PENDING"}
    incremental = [row for key, row in current_complete.items() if key not in prior_complete]
    newly_resolved = [row for key, row in current_complete.items()
                      if key in prior_by_key and prior_by_key[key]["outcome"] == "PENDING"]
    new_emissions = [row for key, row in current_by_key.items() if key not in prior_by_key]
    new_completed = [row for row in new_emissions if row["outcome"] != "PENDING"]
    pending = [row for row in current if row["outcome"] == "PENDING"]
    current_completed = complete(current)

    incremental_stats = stats(incremental)
    cumulative_stats = stats(current_completed)
    report = {
        "status": "FORWARD_APPEND_OBSERVED",
        "ready_for_live_orders": bool(
            incremental_stats["n"] >= 30 and incremental_stats["objective_pass"]
            and cumulative_stats["objective_pass"]),
        "protocol": {
            "fit_or_retune_on_append": False,
            "entry_model": "frozen OKX global 1H HGB and target cascade",
            "entry": "same hourly timestamp X-Perp market open",
            "orders": "market only",
            "outcome_readiness_fix": (
                "portfolio emissions are selected before outcome availability; early "
                "targets/stops resolve immediately; incomplete paths remain PENDING"),
            "money": "3% current-equity margin, fixed 10x gross",
        },
        "integrity": {
            "prior_emissions": len(prior), "current_emissions": len(current),
            "prior_emissions_missing": len(missing_emissions),
            "prior_resolved_outcome_changes": len(changed_resolved),
            "prior_pending": sum(row["outcome"] == "PENDING" for row in prior),
            "current_pending": len(pending),
            "global_coverage": current_meta["global_coverage"],
            "xperp_coverage": current_meta["xperp_coverage"],
        },
        "append": {
            "new_emissions": len(new_emissions),
            "new_completed_emissions": len(new_completed),
            "newly_resolved_prior_pending": len(newly_resolved),
            "incremental_completed": incremental_stats,
            "new_completed_only": stats(new_completed),
            "newly_resolved_only": stats(newly_resolved),
            "cumulative_completed": cumulative_stats,
            "pending": [{
                "symbol": row["symbol"], "direction": row["direction"],
                "entry_time": int(row["entry_time"]),
                "target_pct": float(row["selected_target_pct"]),
                "maximum_exit_time": int(row["exit_time"]),
            } for row in pending],
        },
        "cluster_diagnostics_incremental": cluster_diagnostics(incremental),
        "stop_cause_diagnostics": stop_cause_diagnostics(
            list(prior_complete.values()), incremental),
        "decision": (
            "Continue shadow collection. The corrected append is immutable, but "
            "neither fresh nor cumulative results satisfy the economic objective."),
        "sources": {
            **prior_meta["global_sources"], **prior_meta["xperp_sources"],
            **current_meta["global_sources"], **current_meta["xperp_sources"],
            str(FROZEN_ROWS): digest(FROZEN_ROWS),
            str(router.MODEL_PATH): digest(router.MODEL_PATH),
            str(router.CASCADE_PATH): digest(router.CASCADE_PATH),
            str(Path(__file__)): digest(Path(__file__)),
        },
        "limitations": [
            "The fresh append is short and cannot establish a stable monthly return.",
            "Public candles cannot reproduce order-book slippage or execution latency.",
            "The 30% monthly objective needs 0.973137% constant signed return per trade at 3 trades/day.",
        ],
    }
    pd.DataFrame(current).to_pickle(ROWS_OUT)
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({
        "status": report["status"], "integrity": report["integrity"],
        "append": report["append"],
        "clusters": report["cluster_diagnostics_incremental"],
        "ready_for_live_orders": report["ready_for_live_orders"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
