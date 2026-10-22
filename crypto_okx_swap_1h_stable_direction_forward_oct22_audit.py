"""Genuinely new Oct 7-21 forward replay of the frozen stable 1H route."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

import crypto_okx_swap_1h_forward_append_oct8_audit as legacy
import crypto_okx_swap_1h_forward_append_oct16_audit as append
import crypto_okx_swap_1h_regime_oct7_audit as base
import crypto_5m_regime_router_literal_oct6_audit as portfolio
from src import crypto_okx_1h_shadow_router as router
from src.portable_hgb import score_named


ROOT = Path("reports/audit_2026_09_08")
GLOBAL_MANIFEST = ROOT / "crypto_okx_swap_1h_expanded_oct22" / "manifest.json"
XPERP_MANIFEST = ROOT / "xperp_5m_oct22_forward" / "manifest.json"
SOURCE_REPORT = ROOT / "crypto_okx_swap_1h_stable_direction_oct22.json"
SOURCE_ROWS = ROOT / "crypto_okx_swap_1h_stable_direction_oct22.pkl"
OUT = ROOT / "crypto_okx_swap_1h_stable_direction_forward_oct22.json"
ROWS_OUT = ROOT / "crypto_okx_swap_1h_stable_direction_forward_oct22.pkl"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def frozen_entries(frame: pd.DataFrame) -> pd.DataFrame:
    selected = []
    for row in frame.to_dict("records"):
        families = {
            name for name in base.FAMILIES
            if pd.notna(row.get(f"family_{name}", 0))
            and float(row.get(f"family_{name}", 0)) == 1.0
        }
        symbol = str(row["symbol"])
        direction = int(row["direction"])
        if not router._route_allowed(symbol, direction, families):
            continue
        model_row = router._model_row(symbol, row, direction, families)
        quality = float(score_named(router._MODEL, model_row))
        if quality < router.QUALITY_THRESHOLD:
            continue
        selected.append({
            **row,
            "model_quality": quality,
            "selected_target_pct": router.TARGET_PCT,
            "hold_bars": router.HOLD_BARS,
            "quality": quality,
        })
    output = pd.DataFrame(selected)
    if output.empty:
        return output
    return (output.sort_values(
        ["symbol", "entry_time", "model_quality"],
        ascending=[True, True, False])
        .drop_duplicates(["symbol", "entry_time"], keep="first")
        .reset_index(drop=True))


def main() -> None:
    source = json.loads(SOURCE_REPORT.read_text(encoding="utf-8"))
    freeze_time = int(pd.Timestamp(
        source["direct_xperp"]["last_entry_utc"]).timestamp())
    global_candles, global_sources, global_coverage = legacy.load_merged(
        (append.OLD_GLOBAL, GLOBAL_MANIFEST), legacy.GLOBAL_FIELDS, 3600)
    xperp_candles, xperp_sources, xperp_coverage = legacy.load_merged(
        (append.OLD_XPERP, XPERP_MANIFEST), legacy.XPERP_FIELDS, 300)

    raw = base.build_features(global_candles)
    proposals = base.proposals(raw).reset_index(drop=True)
    entries = frozen_entries(proposals)
    entries = entries[entries.entry_time.gt(freeze_time)].copy()
    measured = append.resolve_available(entries, xperp_candles)
    accepted = append.runtime_gate_pending(append.emission_rows(measured))
    fresh = append.complete(accepted)
    pending = [row for row in accepted if row["outcome"] == "PENDING"]

    source_rows = pd.read_pickle(SOURCE_ROWS)
    prior = portfolio.portfolio_rows(source_rows, (2026,), 0)
    for row in prior:
        row["selected_target_pct"] = router.TARGET_PCT
    prior_stats = append.stats(prior)
    expected = source["direct_xperp"]
    for field in ("n", "return_pct", "dd_pct", "literal_target_hit_rate"):
        if prior_stats[field] != expected[field]:
            raise RuntimeError(
                f"pre-freeze reconstruction mismatch {field}: "
                f"{prior_stats[field]} != {expected[field]}")

    fresh_stats = append.stats(fresh)
    cumulative_stats = append.stats([*prior, *fresh])
    forward_passed = bool(
        fresh_stats["n"] >= 20
        and fresh_stats["return_pct"] > 0
        and fresh_stats["literal_target_hit_rate"] >= 65
        and fresh_stats["dd_pct"] <= 5)
    pd.DataFrame(accepted).to_pickle(ROWS_OUT)
    report = {
        "status": ("FORWARD_GATE_PASSED_AWAIT_MORE_TIME"
                   if forward_passed else "FORWARD_GATE_NOT_MET"),
        "ready_for_live_orders": False,
        "order_eligible": False,
        "protocol": {
            "frozen_after": source["direct_xperp"]["last_entry_utc"],
            "retune_on_forward": False,
            "entry": "same hourly timestamp X-Perp market open",
            "exit": "literal 0.5% target, 1% stop, 12-hour maximum, stop-first",
            "money": "client-selected 3% current-equity margin, fixed 10x gross",
            "route_policy": source["selected_before_direct"],
        },
        "pre_freeze_reconstruction": prior_stats,
        "fresh_completed": fresh_stats,
        "fresh_completed_rows": len(fresh),
        "pending_rows": len(pending),
        "cumulative_completed": cumulative_stats,
        "forward_gate": {
            "minimum_completed": 20,
            "positive_return": True,
            "minimum_literal_target_hit_rate": 65,
            "maximum_drawdown_pct": 5,
            "passed": forward_passed,
        },
        "sources": {
            **global_sources,
            **xperp_sources,
            str(GLOBAL_MANIFEST): digest(GLOBAL_MANIFEST),
            str(XPERP_MANIFEST): digest(XPERP_MANIFEST),
            str(SOURCE_REPORT): digest(SOURCE_REPORT),
            str(SOURCE_ROWS): digest(SOURCE_ROWS),
            str(router.MODEL_PATH): digest(router.MODEL_PATH),
            str(router.POLICY_PATH): digest(router.POLICY_PATH),
            str(Path(__file__).resolve()): digest(Path(__file__).resolve()),
        },
        "coverage": {
            "global": global_coverage,
            "xperp": xperp_coverage,
        },
        "limitations": [
            "The genuinely new forward interval is only about two weeks.",
            "Public candles cannot reproduce order-book slippage or latency.",
            "Gross results exclude fees at user request.",
            "No short forward sample can guarantee future monthly return.",
        ],
    }
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({
        "status": report["status"],
        "fresh_completed": fresh_stats,
        "pending_rows": len(pending),
        "cumulative_completed": cumulative_stats,
        "forward_gate": report["forward_gate"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
