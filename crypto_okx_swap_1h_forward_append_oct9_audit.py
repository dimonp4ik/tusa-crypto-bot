"""Immutable 9 October forward append for the frozen OKX 1H cascade.

The underlying audit owns all scoring and execution semantics.  This entry
point only advances the public snapshots and the prior direct-observation
checkpoint.  It then records clustering of genuinely new accepted trades.
No model is fitted or retuned here.
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

import crypto_okx_swap_1h_forward_append_oct8_audit as audit


ROOT = Path("reports/audit_2026_09_08")
OUT = ROOT / "crypto_okx_swap_1h_forward_append_oct9.json"
ROWS_OUT = ROOT / "crypto_okx_swap_1h_forward_append_oct9.pkl"
PRIOR_ROWS = ROOT / "crypto_okx_swap_1h_forward_append_oct8.pkl"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def key(row: dict) -> tuple[str, int, str]:
    return audit.row_key(row)


def cluster_diagnostics(rows: list[dict]) -> dict:
    groups: dict[tuple[int, str], list[dict]] = defaultdict(list)
    for row in rows:
        direction = str(row["direction"])
        if direction not in {"LONG", "SHORT"}:
            direction = "LONG" if float(row["direction"]) > 0 else "SHORT"
        groups[(int(float(row["entry_time"])), direction)].append(row)
    clusters = []
    for (entry_time, direction), values in sorted(groups.items()):
        if len(values) < 2:
            continue
        clusters.append({
            "entry_time": entry_time,
            "entry_utc": pd.Timestamp(entry_time, unit="s", tz="UTC").isoformat(),
            "direction": direction,
            "size": len(values),
            "symbols": sorted(str(row["symbol"]) for row in values),
            "target_hits": sum(bool(row["target_hit"]) for row in values),
            "stops": sum(int(row["exit_reason"]) == 0 for row in values),
            "signed_price_return_sum": round(sum(
                float(row["signed_price_return"]) for row in values), 8),
        })
    sizes = Counter(len(values) for values in groups.values())
    return {
        "accepted_rows": len(rows),
        "distinct_direction_hours": len(groups),
        "same_direction_hour_size_distribution": {
            str(size): count for size, count in sorted(sizes.items())
        },
        "multi_entry_direction_hours": len(clusters),
        "max_same_direction_hour": max((item["size"] for item in clusters), default=1 if rows else 0),
        "clusters": clusters,
    }


def main() -> None:
    audit.NEW_GLOBAL_MANIFEST = (
        ROOT / "crypto_okx_swap_1h_expanded_oct9" / "manifest.json")
    audit.NEW_XPERP_MANIFEST = ROOT / "xperp_5m_oct9_forward" / "manifest.json"
    audit.PRIOR_DIRECT_ROWS = PRIOR_ROWS
    audit.OUT = OUT
    audit.ROWS_OUT = ROWS_OUT
    audit.main()

    report = json.loads(OUT.read_text(encoding="utf-8"))
    refreshed = pd.read_pickle(ROWS_OUT)
    prior = pd.read_pickle(PRIOR_ROWS)
    refreshed_accepted = audit.accepted_rows(refreshed)
    prior_keys = {key(row) for row in audit.accepted_rows(prior)}
    incremental = [row for row in refreshed_accepted if key(row) not in prior_keys]
    report["cluster_diagnostics_incremental"] = cluster_diagnostics(incremental)
    report["protocol"]["append_checkpoint"] = (
        "8 October direct X-Perp observation set")
    report["sources"][str(Path(__file__))] = digest(Path(__file__))
    report["limitations"] = [
        "This is a short forward append and cannot establish a stable win rate.",
        "Simultaneous symbols can express one correlated market bet; cluster diagnostics must be reviewed.",
        "Public candles cannot reproduce order-book slippage or execution latency.",
        "The requested monthly return and no-loss-month objectives remain unproven.",
    ]
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({
        "status": report["status"],
        "integrity": report["integrity"],
        "append": report["append"],
        "cluster_diagnostics_incremental": report["cluster_diagnostics_incremental"],
        "readiness": report["readiness"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
