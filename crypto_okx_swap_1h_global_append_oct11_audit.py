"""Frozen global-OKX forward check on the verified October 11 snapshot.

The audit does not fit or tune anything.  It uses only the eight symbols and
the exported models used by the shadow runtime, verifies historical scoring
parity, and keeps incomplete forward outcomes separate from resolved trades.
"""
from __future__ import annotations

import hashlib
import json
import pickle
from collections import Counter
from pathlib import Path

import pandas as pd

import crypto_okx_swap_1h_forward_append_oct8_audit as prior
import crypto_okx_swap_1h_regime_oct7_audit as base
from src import crypto_okx_1h_shadow_router as router


ROOT = Path("reports/audit_2026_09_08")
OLD_MANIFEST = ROOT / "crypto_okx_swap_1h_expanded_oct9" / "manifest.json"
NEW_MANIFEST = ROOT / "crypto_okx_swap_1h_expanded_oct11" / "manifest.json"
FROZEN_ROWS = ROOT / "crypto_okx_swap_1h_target_cascade_oct7.pkl"
OUT = ROOT / "crypto_okx_swap_1h_global_append_oct11.json"
FIELDS = ("open", "high", "low", "close", "volume",
          "volume_ccy", "volume_quote")
SYMBOLS = tuple(sorted(router.SUPPORTED_SYMBOLS))


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def resolve_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else Path.cwd() / path


def load_verified(manifest_path: Path) -> tuple[dict, dict, dict]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    candles, coverage, sources = {}, {}, {str(manifest_path): digest(manifest_path)}
    for symbol in SYMBOLS:
        item = manifest["symbols"][symbol]
        path = resolve_path(item["path"])
        raw = path.read_bytes()
        found = hashlib.sha256(raw).hexdigest()
        if found != item["sha256"]:
            raise ValueError(f"source hash mismatch: {symbol} {path}")
        candle = pickle.loads(raw)
        lengths = {len(candle[field]) for field in ("time", *FIELDS)}
        if len(lengths) != 1:
            raise ValueError(f"field length mismatch: {symbol}")
        times = [int(value) for value in candle["time"]]
        if any(right - left != 3600 for left, right in zip(times, times[1:])):
            raise ValueError(f"non-contiguous source: {symbol}")
        if times[0] != int(item["start"]) or times[-1] != int(item["end"]):
            raise ValueError(f"manifest coverage mismatch: {symbol}")
        candles[symbol] = candle
        coverage[symbol] = {"bars": len(times), "start": times[0], "end": times[-1]}
        sources[str(path)] = found
    return candles, coverage, sources


def key(row: dict) -> tuple[str, int, int]:
    direction = row["direction"]
    if isinstance(direction, str):
        direction = 1 if direction == "LONG" else -1
    return str(row["symbol"]), int(float(row["entry_time"])), int(direction)


def compact_rows(frame: pd.DataFrame) -> list[dict]:
    fields = ("symbol", "entry_time", "direction", "model_quality",
              "selected_target_pct", "hold_bars")
    rows = []
    for row in frame.sort_values(["entry_time", "symbol"]).to_dict("records"):
        direction = row["direction"]
        rows.append({
            field: (int(row[field]) if field in ("entry_time", "hold_bars")
                    else float(row[field]) if field in (
                        "model_quality", "selected_target_pct")
                    else "LONG" if field == "direction" and float(direction) > 0
                    else "SHORT" if field == "direction"
                    else str(row[field]))
            for field in fields
        })
    return rows


def compact_outcomes(frame: pd.DataFrame) -> list[dict]:
    rows = []
    for row in frame.sort_values(["entry_time", "symbol"]).to_dict("records"):
        rows.append({
            "symbol": str(row["symbol"]),
            "entry_time": int(float(row["entry_time"])),
            "direction": "LONG" if float(row["direction"]) > 0 else "SHORT",
            "selected_target_pct": float(row["selected_target_pct"]),
            "signed_price_return_pct": round(100 * float(row["signed_return"]), 6),
            "target_hit": bool(row["target_hit"]),
            "exit_reason": int(row["exit_reason"]),
            "exit_time": int(float(row["exit_time"])),
        })
    return rows


def main() -> None:
    old_manifest = json.loads(OLD_MANIFEST.read_text(encoding="utf-8"))
    new_manifest = json.loads(NEW_MANIFEST.read_text(encoding="utf-8"))
    cutoff = min(int(old_manifest["symbols"][symbol]["end"]) for symbol in SYMBOLS)
    end = min(int(new_manifest["symbols"][symbol]["end"]) for symbol in SYMBOLS)
    if end <= cutoff:
        raise ValueError("new snapshot has no closed bars after the cutoff")

    candles, coverage, sources = load_verified(NEW_MANIFEST)
    raw = base.build_features(candles)
    proposals = base.proposals(raw).reset_index(drop=True)

    frozen = pd.read_pickle(FROZEN_ROWS)
    parity_start = int(pd.Timestamp("2026-09-01", tz="UTC").timestamp())
    parity_end = int(frozen.entry_time.max())
    overlap = proposals[proposals.entry_time.between(
        parity_start, parity_end, inclusive="both")]
    rescored = prior.frozen_scores(overlap, False)
    expected = frozen[frozen.entry_time.between(
        parity_start, parity_end, inclusive="both")]
    expected_by_key = {key(row): float(row["selected_target_pct"])
                       for row in expected.to_dict("records")}
    rescored_by_key = {key(row): float(row["selected_target_pct"])
                       for row in rescored.to_dict("records")}
    common = set(expected_by_key) & set(rescored_by_key)
    parity = {
        "period_start": parity_start,
        "period_end_inclusive": parity_end,
        "expected_selected": len(expected_by_key),
        "rescored_selected": len(rescored_by_key),
        "missing_keys": len(set(expected_by_key) - set(rescored_by_key)),
        "unexpected_keys": len(set(rescored_by_key) - set(expected_by_key)),
        "target_mismatches": sum(expected_by_key[item] != rescored_by_key[item]
                                 for item in common),
    }
    if any(parity[field] for field in (
            "missing_keys", "unexpected_keys", "target_mismatches")):
        raise ValueError(f"frozen scoring parity failed: {parity}")

    # Reuse the immutable historical selections and score only proposals after
    # their final timestamp.  This is equivalent to rescoring the full history
    # after the parity assertion above, but much faster with the portable model.
    newer_proposals = proposals[proposals.entry_time.gt(parity_end)].copy()
    newer_selected = prior.frozen_scores(newer_proposals, True)
    all_selected = pd.concat([frozen, newer_selected], ignore_index=True,
                             sort=False).drop_duplicates(
        ["symbol", "entry_time", "direction"], keep="first")
    all_selected["year"] = pd.to_datetime(
        all_selected.entry_time, unit="s", utc=True).dt.year.astype(int)
    fresh = all_selected[all_selected.entry_time.gt(cutoff)].copy()

    resolved_all = prior.global_outcomes(all_selected, candles)
    resolved_keys = {key(row) for row in resolved_all.to_dict("records")}
    fresh_resolved = resolved_all[resolved_all.entry_time.gt(cutoff)].copy()
    fresh_pending = fresh[
        ~fresh.apply(lambda row: key(row.to_dict()) in resolved_keys, axis=1)
    ].copy()

    accepted_all = prior.accepted_rows(resolved_all)
    fresh_accepted = [row for row in accepted_all
                      if float(row["entry_time"]) > cutoff]
    target_counts = Counter(str(float(value))
                            for value in fresh.selected_target_pct.tolist())
    same_hour = Counter(int(float(value)) for value in fresh.entry_time.tolist())
    max_same_hour = max(same_hour.values(), default=0)
    clustered_hours = {
        str(timestamp): count for timestamp, count in sorted(same_hour.items())
        if count > 1
    }
    resolved_summary = prior.stats(fresh_accepted)

    report = {
        "status": "FROZEN_GLOBAL_FORWARD_DIAGNOSTIC",
        "protocol": {
            "fit_or_retune": False,
            "universe": list(SYMBOLS),
            "entry_model": "exported frozen OKX global 1H HGB and target cascade",
            "execution_data": "OKX global USDT swap 1H candles",
            "targets_pct": [0.5, 1.0, 1.5, 2.0],
            "stop_pct": 1.0,
            "money": "3% current equity margin, fixed 10x gross",
            "fees": "excluded by user instruction",
        },
        "window": {
            "old_snapshot_last_closed_bar": cutoff,
            "old_snapshot_last_closed_bar_utc": pd.Timestamp(
                cutoff, unit="s", tz="UTC").isoformat(),
            "new_snapshot_last_closed_bar": end,
            "new_snapshot_last_closed_bar_utc": pd.Timestamp(
                end, unit="s", tz="UTC").isoformat(),
            "fresh_closed_hours": int((end - cutoff) // 3600),
        },
        "integrity": {
            "frozen_scoring_parity": parity,
            "coverage": coverage,
            "old_manifest_sha256": digest(OLD_MANIFEST),
            "new_manifest_sha256": digest(NEW_MANIFEST),
        },
        "fresh_candidates": {
            "proposal_rows": int((proposals.entry_time > cutoff).sum()),
            "selected_rows": len(fresh),
            "resolved_selected_rows": len(fresh_resolved),
            "pending_selected_rows": len(fresh_pending),
            "selected_target_distribution": dict(sorted(target_counts.items())),
            "max_selected_same_entry_hour": max_same_hour,
            "clustered_entry_hours": clustered_hours,
            "rows": compact_rows(fresh),
            "resolved_rows": compact_outcomes(fresh_resolved),
            "pending_rows": compact_rows(fresh_pending),
        },
        "fresh_resolved_after_portfolio_rails": resolved_summary,
        "readiness": {
            "fresh_resolved_accepted_minimum": 30,
            "fresh_resolved_accepted_observed": len(fresh_accepted),
            "statistically_ready": len(fresh_accepted) >= 30,
            "objective_proven": False,
            "live_activation_allowed": False,
        },
        "sources": {
            **sources,
            str(FROZEN_ROWS): digest(FROZEN_ROWS),
            str(router.MODEL_PATH): digest(router.MODEL_PATH),
            str(router.CASCADE_PATH): digest(router.CASCADE_PATH),
            str(Path(__file__)): digest(Path(__file__)),
        },
        "limitations": [
            "This is a global-OKX signal diagnostic, not direct EU X-Perp execution evidence.",
            "Incomplete trades are reported as pending and excluded from performance.",
            "The forward window remains too short to establish a stable win rate.",
            "The 30-200% monthly objective remains unproven.",
        ],
    }
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({
        "status": report["status"],
        "window": report["window"],
        "parity": parity,
        "fresh_candidates": report["fresh_candidates"],
        "fresh_resolved_after_portfolio_rails": resolved_summary,
        "readiness": report["readiness"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
