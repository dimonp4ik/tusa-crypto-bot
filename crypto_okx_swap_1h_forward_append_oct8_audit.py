"""Append-only forward audit of the frozen OKX 1H cascade on EU X-Perp.

This script never refits or retunes a model.  It merges immutable public
snapshots, checks every overlapping candle for revisions, reproduces the
already-frozen decisions, and then scores only observations that became fully
observable after the previous direct-transfer snapshot.
"""
from __future__ import annotations

import hashlib
import json
import pickle
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

import crypto_5m_regime_router_literal_oct6_audit as portfolio
import crypto_okx_swap_1h_regime_oct7_audit as base
import crypto_okx_swap_1h_target_cascade_oct7_audit as cascade
import crypto_okx_swap_1h_xperp_transfer_oct7_audit as transfer
from src import crypto_okx_1h_shadow_router as router
from src.portable_hgb import score_named


ROOT = Path("reports/audit_2026_09_08")
OLD_GLOBAL_MANIFEST = ROOT / "crypto_okx_swap_1h_history" / "manifest.json"
NEW_GLOBAL_MANIFEST = ROOT / "crypto_okx_swap_1h_oct8_forward" / "manifest.json"
OLD_XPERP_MANIFEST = ROOT / "xperp_5m_oct4" / "manifest.json"
NEW_XPERP_MANIFEST = ROOT / "xperp_5m_oct8_forward" / "manifest.json"
FROZEN_ROWS = ROOT / "crypto_okx_swap_1h_target_cascade_oct7.pkl"
PRIOR_DIRECT_ROWS = ROOT / "crypto_okx_swap_1h_target_cascade_xperp_oct7.pkl"
OUT = ROOT / "crypto_okx_swap_1h_forward_append_oct8.json"
ROWS_OUT = ROOT / "crypto_okx_swap_1h_forward_append_oct8.pkl"
SYMBOLS = tuple(sorted(router.SUPPORTED_SYMBOLS))
GLOBAL_FIELDS = ("open", "high", "low", "close", "volume",
                 "volume_ccy", "volume_quote")
XPERP_FIELDS = ("open", "high", "low", "close", "volume")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def resolve_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else Path.cwd() / path


def load_merged(manifests: tuple[Path, ...], fields: tuple[str, ...],
                step: int) -> tuple[dict, dict, dict]:
    """Load hashed snapshots and fail if an overlapping confirmed bar changed."""
    merged = {symbol: {} for symbol in SYMBOLS}
    sources: dict[str, str] = {}
    overlap = {symbol: 0 for symbol in SYMBOLS}
    for manifest_path in manifests:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        sources[str(manifest_path)] = digest(manifest_path)
        for symbol in SYMBOLS:
            item = manifest["symbols"][symbol]
            path = resolve_path(item["path"])
            raw = path.read_bytes()
            found = hashlib.sha256(raw).hexdigest()
            if found != item["sha256"]:
                raise ValueError(f"source hash mismatch: {path}")
            candle = pickle.loads(raw)
            sources[str(path)] = found
            for index, timestamp in enumerate(candle["time"]):
                values = tuple(float(candle[field][index]) for field in fields)
                old = merged[symbol].get(int(timestamp))
                if old is not None:
                    overlap[symbol] += 1
                    if old != values:
                        changed = [field for field, left, right in zip(
                            fields, old, values) if left != right]
                        raise ValueError(
                            f"confirmed source revision: {symbol} {timestamp} {changed}")
                merged[symbol][int(timestamp)] = values
    candles = {}
    coverage = {}
    for symbol, values in merged.items():
        times = sorted(values)
        if not times or any(right - left != step
                            for left, right in zip(times, times[1:])):
            raise ValueError(f"non-contiguous merged history: {symbol}")
        candles[symbol] = {"time": times, **{
            field: [values[timestamp][position] for timestamp in times]
            for position, field in enumerate(fields)
        }}
        coverage[symbol] = {
            "bars": len(times), "start": times[0], "end": times[-1],
            "overlap_bars_verified": overlap[symbol],
        }
    return candles, sources, coverage


def frozen_scores(frame: pd.DataFrame, collapse_directions: bool) -> pd.DataFrame:
    """Apply the exported production models without fitting any estimator."""
    selected = []
    for row in frame.to_dict("records"):
        families = {name for name in base.FAMILIES
                    if pd.notna(row.get(f"family_{name}", 0))
                    and float(row.get(f"family_{name}", 0)) == 1.0}
        model_row = router._model_row(
            str(row["symbol"]), row, int(row["direction"]), families)
        base_quality = float(score_named(router._MODEL, model_row))
        if base_quality < router.QUALITY_THRESHOLD:
            continue
        target = float(router.TARGET_PCT)
        hold = int(router.HOLD_BARS)
        target_scores = {}
        for name, model in sorted(router._CASCADE["models"].items(),
                                  key=lambda item: float(item[0])):
            quality = float(score_named(model, model_row))
            target_scores[name] = quality
            if quality >= float(model["quality_threshold"]):
                target = float(model["target_pct"])
                hold = int(model["hold_bars"])
        selected.append({**row, "model_quality": base_quality,
                         "target_model_scores": target_scores,
                         "selected_target_pct": target,
                         "hold_bars": hold, "quality": 1.0})
    output = pd.DataFrame(selected)
    if output.empty or not collapse_directions:
        return output
    # The production router emits at most one direction per symbol and hour.
    return (output.sort_values(
        ["symbol", "entry_time", "model_quality"],
        ascending=[True, True, False])
        .drop_duplicates(["symbol", "entry_time"], keep="first")
        .reset_index(drop=True))


def row_key(row: dict) -> tuple[str, int, str]:
    direction = row["direction"]
    if not isinstance(direction, str):
        direction = "LONG" if float(direction) > 0 else "SHORT"
    return str(row["symbol"]), int(float(row["entry_time"])), direction


def accepted_rows(frame: pd.DataFrame) -> list[dict]:
    if frame.empty:
        return []
    years = tuple(sorted(int(year) for year in frame.year.unique()))
    return portfolio.portfolio_rows(frame, years, .5)


def stats(rows: list[dict]) -> dict:
    if not rows:
        return {
            "n": 0, "wr": 0.0, "return_pct": 0.0, "dd_pct": 0.0,
            "literal_target_hits": 0, "literal_target_hit_rate": 0.0,
            "stops": 0, "timeouts": 0, "target_distribution": {},
            "target_hits": {}, "first_entry_utc": None,
            "last_entry_utc": None,
        }
    result = portfolio.money(rows, portfolio.MARGIN_FRACTION)
    hits = sum(bool(row["target_hit"]) for row in rows)
    targets = Counter(str(float(row["selected_target_pct"])) for row in rows)
    target_hits = Counter(str(float(row["selected_target_pct"]))
                          for row in rows if bool(row["target_hit"]))
    result.update({
        "literal_target_hits": hits,
        "literal_target_hit_rate": round(100 * hits / len(rows), 4),
        "stops": sum(int(row["exit_reason"]) == 0 for row in rows),
        "timeouts": sum(int(row["exit_reason"]) == 2 for row in rows),
        "target_distribution": dict(sorted(targets.items())),
        "target_hits": dict(sorted(target_hits.items())),
        "first_entry_utc": pd.Timestamp(
            min(float(row["entry_time"]) for row in rows), unit="s",
            tz="UTC").isoformat(),
        "last_entry_utc": pd.Timestamp(
            max(float(row["entry_time"]) for row in rows), unit="s",
            tz="UTC").isoformat(),
    })
    return result


def breakdown(rows: list[dict], frequency: str) -> dict:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        stamp = pd.Timestamp(float(row["entry_time"]), unit="s", tz="UTC")
        key = stamp.strftime("%Y-%m-%d" if frequency == "day" else "%Y-%m")
        grouped[key].append(row)
    return {key: stats(value) for key, value in sorted(grouped.items())}


def global_outcomes(entries: pd.DataFrame, candles: dict) -> pd.DataFrame:
    parts = []
    for target, target_rows in entries.groupby("selected_target_pct", sort=True):
        target = float(target)
        hold = cascade.HOLD[target]
        mapped = transfer.map_entries(target_rows, candles, hold)
        for symbol, part in mapped.groupby("symbol", sort=False):
            candle = candles[symbol]
            values, exits, hits, reasons = cascade.resolve(
                *[np.asarray(candle[field], dtype=float)
                  for field in ("open", "high", "low", "close")],
                np.asarray(candle["time"], dtype=np.int64),
                part.entry_index.to_numpy(np.int64),
                part.direction.to_numpy(np.int8), target, hold)
            found = part.copy()
            found["signed_return"], found["exit_time"] = values, exits
            found["target_hit"], found["exit_reason"] = hits.astype(bool), reasons
            found["quality"] = 1.0
            parts.append(found)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def venue_comparison(xperp_rows: list[dict], global_frame: pd.DataFrame) -> dict:
    global_by_key = {row_key(row): row for row in global_frame.to_dict("records")}
    pairs = [(row, global_by_key[row_key(row)]) for row in xperp_rows
             if row_key(row) in global_by_key]
    if not pairs:
        return {"comparable": 0}
    disagreements = sum(bool(x["target_hit"]) != bool(g["target_hit"])
                        for x, g in pairs)
    x_hits = sum(bool(x["target_hit"]) for x, _ in pairs)
    g_hits = sum(bool(g["target_hit"]) for _, g in pairs)
    return {
        "comparable": len(pairs), "target_hit_disagreements": disagreements,
        "xperp_literal_hits": x_hits, "global_literal_hits": g_hits,
        "xperp_literal_hit_rate": round(100 * x_hits / len(pairs), 4),
        "global_literal_hit_rate": round(100 * g_hits / len(pairs), 4),
        "xperp_mean_signed_move_pct": round(
            100 * float(np.mean([x["signed_price_return"] for x, _ in pairs])), 6),
        "global_mean_signed_move_pct": round(
            100 * float(np.mean([g["signed_return"] for _, g in pairs])), 6),
    }


def main() -> None:
    global_candles, global_sources, global_coverage = load_merged(
        (OLD_GLOBAL_MANIFEST, NEW_GLOBAL_MANIFEST), GLOBAL_FIELDS, 3600)
    xperp_candles, xperp_sources, xperp_coverage = load_merged(
        (OLD_XPERP_MANIFEST, NEW_XPERP_MANIFEST), XPERP_FIELDS, 300)

    old_global_manifest = json.loads(
        OLD_GLOBAL_MANIFEST.read_text(encoding="utf-8"))
    freeze_end = min(int(item["end"])
                     for item in old_global_manifest["symbols"].values())
    old_entries = pd.read_pickle(FROZEN_ROWS)
    old_direct = pd.read_pickle(PRIOR_DIRECT_ROWS)

    raw = base.build_features(global_candles)
    proposals = base.proposals(raw).reset_index(drop=True)
    parity_start = int(pd.Timestamp("2026-09-01", tz="UTC").timestamp())
    overlap_proposals = proposals[proposals.entry_time.between(
        parity_start, freeze_end, inclusive="both")].copy()
    rescored_overlap = frozen_scores(overlap_proposals, False)
    frozen_overlap = old_entries[old_entries.entry_time.between(
        parity_start, freeze_end, inclusive="both")].copy()
    old_keys = {row_key(row) for row in frozen_overlap.to_dict("records")}
    rescored_keys = {row_key(row) for row in rescored_overlap.to_dict("records")}
    old_target = {row_key(row): float(row["selected_target_pct"])
                  for row in frozen_overlap.to_dict("records")}
    new_target = {row_key(row): float(row["selected_target_pct"])
                  for row in rescored_overlap.to_dict("records")}
    common_scored = old_keys & rescored_keys
    target_mismatches = sum(old_target[key] != new_target[key]
                            for key in common_scored)
    dual_overlap = int((rescored_overlap.groupby(
        ["symbol", "entry_time"]).size() > 1).sum()) if len(rescored_overlap) else 0
    scoring_parity = {
        "period_start": parity_start, "period_end_inclusive": freeze_end,
        "frozen_selected": len(frozen_overlap),
        "rescored_selected": len(rescored_overlap),
        "missing_keys": len(old_keys - rescored_keys),
        "unexpected_keys": len(rescored_keys - old_keys),
        "target_mismatches": target_mismatches,
        "dual_direction_symbol_hours": dual_overlap,
    }
    if any(scoring_parity[key] for key in (
            "missing_keys", "unexpected_keys", "target_mismatches")):
        raise ValueError(f"frozen scoring parity failed: {scoring_parity}")

    future_proposals = proposals[proposals.entry_time.gt(freeze_end)].copy()
    future_entries = frozen_scores(future_proposals, True)
    future_entries["year"] = pd.to_datetime(
        future_entries.entry_time, unit="s", utc=True).dt.year.astype(int)
    future_entries["month"] = pd.to_datetime(
        future_entries.entry_time, unit="s", utc=True).dt.strftime(
            "%Y%m").astype(int)
    all_entries = pd.concat([old_entries, future_entries], ignore_index=True,
                            sort=False)
    all_entries = all_entries.drop_duplicates(
        ["symbol", "entry_time", "direction"], keep="first")

    refreshed_direct = transfer_direct_rows(all_entries, xperp_candles)
    refreshed_direct["year"] = pd.to_datetime(
        refreshed_direct.entry_time, unit="s", utc=True).dt.year.astype(int)
    refreshed_accepted = accepted_rows(refreshed_direct)
    prior_accepted = accepted_rows(old_direct)
    prior_by_key = {row_key(row): row for row in prior_accepted}
    refreshed_by_key = {row_key(row): row for row in refreshed_accepted}
    missing_prior = set(prior_by_key) - set(refreshed_by_key)
    changed_prior = []
    for key in set(prior_by_key) & set(refreshed_by_key):
        left, right = prior_by_key[key], refreshed_by_key[key]
        if (bool(left["target_hit"]) != bool(right["target_hit"])
                or int(left["exit_reason"]) != int(right["exit_reason"])
                or abs(float(left["signed_price_return"])
                       - float(right["signed_price_return"])) > 1e-12):
            changed_prior.append(key)
    if missing_prior or changed_prior:
        raise ValueError(
            f"append changed prior accepted rows: missing={len(missing_prior)} "
            f"changed={len(changed_prior)}")

    incremental = [row for key, row in refreshed_by_key.items()
                   if key not in prior_by_key]
    newly_resolved = [row for row in incremental
                      if float(row["entry_time"]) <= freeze_end]
    post_freeze = [row for row in incremental
                   if float(row["entry_time"]) > freeze_end]
    global_direct = global_outcomes(all_entries, global_candles)

    raw_old_keys = {row_key(row) for row in old_direct.to_dict("records")}
    raw_refreshed_keys = {
        row_key(row) for row in refreshed_direct.to_dict("records")}
    report = {
        "status": "FORWARD_APPEND_OBSERVED",
        "protocol": {
            "fit_or_retune_on_append": False,
            "entry_model": "frozen OKX global 1H HGB",
            "execution_venue": "EU X-Perp public 5m candles",
            "entry": "same hourly timestamp X-Perp market-open proxy",
            "targets_pct": list(cascade.TARGETS),
            "stop_pct": cascade.STOP_PCT,
            "hold_hours": cascade.HOLD,
            "money": "3% current equity margin, fixed 10x gross",
            "fees": "excluded by user instruction",
        },
        "freeze": {
            "last_old_global_bar": freeze_end,
            "last_old_global_bar_utc": pd.Timestamp(
                freeze_end, unit="s", tz="UTC").isoformat(),
            "prior_mapped_rows": len(old_direct),
            "prior_accepted_rows": len(prior_accepted),
        },
        "integrity": {
            "global_coverage": global_coverage,
            "xperp_coverage": xperp_coverage,
            "frozen_scoring_parity": scoring_parity,
            "prior_accepted_missing_after_append": len(missing_prior),
            "prior_accepted_outcome_changes": len(changed_prior),
        },
        "new_global_candidates": {
            "proposal_rows": len(future_proposals),
            "selected_rows_before_full_outcome_requirement": len(future_entries),
            "dual_direction_rows_removed_by_production_semantics": int(
                len(frozen_scores(future_proposals, False)) - len(future_entries)),
        },
        "append": {
            "refreshed_mapped_rows": len(refreshed_direct),
            "newly_mapped_rows": len(raw_refreshed_keys - raw_old_keys),
            "refreshed_accepted_rows": len(refreshed_accepted),
            "new_accepted_rows": len(incremental),
            "newly_resolved_frozen_entries": stats(newly_resolved),
            "post_freeze_entries": stats(post_freeze),
            "all_incremental": stats(incremental),
            "incremental_by_day": breakdown(incremental, "day"),
            "incremental_by_month": breakdown(incremental, "month"),
            "refreshed_total": stats(refreshed_accepted),
        },
        "venue_discrepancy_incremental": venue_comparison(
            incremental, global_direct),
        "readiness": {
            "fresh_accepted_minimum": 30,
            "fresh_accepted_observed": len(incremental),
            "statistically_ready": len(incremental) >= 30,
            "live_activation_allowed": False,
            "reason": ("At least 30 genuinely new accepted X-Perp outcomes are "
                       "required before another decision; this threshold is an "
                       "observation gate, not proof of profitability."),
        },
        "sources": {
            **global_sources, **xperp_sources,
            str(FROZEN_ROWS): digest(FROZEN_ROWS),
            str(PRIOR_DIRECT_ROWS): digest(PRIOR_DIRECT_ROWS),
            str(router.MODEL_PATH): digest(router.MODEL_PATH),
            str(router.CASCADE_PATH): digest(router.CASCADE_PATH),
            str(Path(__file__)): digest(Path(__file__)),
        },
        "limitations": [
            "The append spans only about two new calendar days.",
            "Small-sample profit or hit rate is not evidence of stable live performance.",
            "Public candles cannot reproduce order-book slippage or execution latency.",
            "The user targets of 30-200% monthly return and no losing months remain unproven.",
        ],
    }
    refreshed_direct.to_pickle(ROWS_OUT)
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({
        "status": report["status"],
        "frozen_scoring_parity": scoring_parity,
        "new_global_candidates": report["new_global_candidates"],
        "append": report["append"],
        "venue_discrepancy_incremental": report["venue_discrepancy_incremental"],
        "readiness": report["readiness"],
    }, indent=2), flush=True)


def transfer_direct_rows(entries: pd.DataFrame, candles: dict) -> pd.DataFrame:
    """Local copy of the cascade transfer to keep the source manifest explicit."""
    parts = []
    for target, target_rows in entries.groupby("selected_target_pct", sort=True):
        target = float(target)
        hold = cascade.HOLD[target] * 12
        mapped = transfer.map_entries(target_rows, candles, hold)
        for symbol, part in mapped.groupby("symbol", sort=False):
            candle = candles[symbol]
            values, exits, hits, reasons = transfer.resolve_fixed(
                *[np.asarray(candle[field], dtype=float)
                  for field in ("open", "high", "low", "close")],
                np.asarray(candle["time"], dtype=np.int64),
                part.entry_index.to_numpy(np.int64),
                part.direction.to_numpy(np.int8), target, 1.0, hold)
            found = part.copy()
            found["signed_return"], found["exit_time"] = values, exits
            found["target_hit"], found["exit_reason"] = hits.astype(bool), reasons
            found["quality"] = 1.0
            parts.append(found)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


if __name__ == "__main__":
    main()
