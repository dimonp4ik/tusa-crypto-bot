"""Unseen October archive holdout for the frozen cross-sectional confirmation."""
from __future__ import annotations

import hashlib
import json
import pickle
from pathlib import Path

import numpy as np

import crypto_5m_cross_section_confirmation_xperp_oct16_audit as transfer
import crypto_5m_sweep_oi_flush_oct14_audit as oi_source
import crypto_5m_sweep_taker_flow_oct14_audit as flow_source
from src import research_crypto_replay as replay


ROOT = Path("reports/audit_2026_09_08")
HOLDOUT_MANIFEST = ROOT / "crypto_futures_oct16_holdout" / "manifest.json"
OUT = ROOT / "crypto_5m_cross_section_confirmation_holdout_oct16.json"
HOLDOUT_START = 1_790_812_800  # 2026-10-01 00:00 UTC


def merge_time_series(base: dict, tail: dict) -> dict:
    base_times = [int(value) for value in base["time"]]
    tail_times = [int(value) for value in tail["time"]]
    fields = set(base) & set(tail) - {"time"}
    if set(base) != set(tail):
        raise ValueError("base and holdout fields differ")
    rows = {
        stamp: {field: base[field][index] for field in fields}
        for index, stamp in enumerate(base_times)
    }
    rows.update({
        stamp: {field: tail[field][index] for field in fields}
        for index, stamp in enumerate(tail_times)
    })
    ordered = sorted(rows)
    return {"time": ordered, **{
        field: [rows[stamp][field] for stamp in ordered] for field in fields
    }}


def load_holdout(path: Path = HOLDOUT_MANIFEST) -> tuple[dict, dict, dict]:
    raw = path.read_bytes()
    manifest = json.loads(raw)
    candles, metrics = {}, {}
    sources = {str(path): hashlib.sha256(raw).hexdigest()}
    for symbol, items in manifest["symbols"].items():
        for kind, output in (("candles", candles), ("metrics", metrics)):
            meta = items[kind]
            source = Path(meta["path"])
            payload = source.read_bytes()
            observed = hashlib.sha256(payload).hexdigest()
            if observed != meta["sha256"]:
                raise ValueError(f"holdout hash mismatch: {symbol} {kind}")
            value = pickle.loads(payload)
            times = np.asarray(value["time"], dtype=np.int64)
            if len(times) < 2 or np.any(np.diff(times) != 300):
                raise ValueError(f"holdout gap: {symbol} {kind}")
            output[symbol] = value
            sources[str(source)] = observed
    return candles, metrics, sources


def combine_inputs() -> tuple[dict, dict, dict]:
    old_candles, old_sources = flow_source.load()
    old_metrics, old_metric_sources = oi_source.load_metrics()
    tail_candles, tail_metrics, tail_sources = load_holdout()
    candles = {symbol: merge_time_series(old_candles[symbol], tail_candles[symbol])
               for symbol in old_candles}
    metrics = {symbol: merge_time_series(old_metrics[symbol], tail_metrics[symbol])
               for symbol in old_metrics}
    return candles, metrics, {**old_sources, **old_metric_sources, **tail_sources}


def main() -> None:
    candles, metrics, sources = combine_inputs()
    all_signals = transfer.signal_frame(candles, metrics)
    holdout = all_signals[all_signals.entry_time.ge(HOLDOUT_START)].copy()
    # Require a complete 24-bar path in the source before comparing venues.
    holdout = transfer.remap_entries(holdout, candles)
    xperp_candles, venue_sources = transfer.load_xperp()
    xperp_entries = transfer.remap_entries(holdout, xperp_candles)
    keys = {(str(row.symbol), int(row.entry_time))
            for row in xperp_entries.itertuples()}
    source_entries = holdout[[
        (str(row.symbol), int(row.entry_time)) in keys
        for row in holdout.itertuples()
    ]].copy()

    source_outcomes = replay.add_outcomes(
        source_entries, candles, transfer.TARGET_PCT,
        transfer.STOP_PCT, transfer.HOLD_BARS)
    xperp_outcomes = replay.add_outcomes(
        xperp_entries, xperp_candles, transfer.TARGET_PCT,
        transfer.STOP_PCT, transfer.HOLD_BARS)
    _, source_exact = transfer.exact(source_outcomes)
    _, xperp_exact = transfer.exact(xperp_outcomes)
    positive_limited = bool(
        xperp_exact["n"] >= 20
        and xperp_exact["return_pct"] > 0
        and xperp_exact["dd_pct"] <= 5
        and xperp_exact["max_loss_streak"] <= 5
    )
    objective = {
        "win_rate_75_to_95": 75 <= xperp_exact["wr"] <= 95,
        "at_least_3_calendar_trades_per_day": xperp_exact["trades_per_calendar_day"] >= 3,
        "no_negative_months": xperp_exact["negative_months"] == 0,
        "drawdown_at_most_5pct": xperp_exact["dd_pct"] <= 5,
        "return_at_least_30pct_per_month": bool(
            xperp_exact.get("monthly_return_pct")) and all(
                value >= 30 for value in xperp_exact["monthly_return_pct"].values()),
    }
    report = {
        "status": "HOLDOUT_POSITIVE_LIMITED" if positive_limited else "REJECTED_HOLDOUT",
        "ready_for_live_orders": False,
        "selection_provenance": {
            "variant": transfer.VARIANT,
            "frozen_commit": "a364f80",
            "frozen_report": str(ROOT / "crypto_5m_cross_section_confirmation_oct14.json"),
            "selection_data_ended": "2026-09-30",
        },
        "protocol": {
            "holdout": "2026-10-01 through last complete Binance archive day",
            "loaded_after_rule_was_frozen": True,
            "qualification": (
                "unseen archive holdout, not live-forward time after the rule was formulated"),
            "execution": "native OKX X-Perp next 5m market open",
            "exit": "0.5% target, 1.0% stop, stop-first, 24 bars",
            "portfolio": "causal shared rails; 3% current-equity margin; fixed 10x",
        },
        "raw_holdout_signals": len(holdout),
        "paired_holdout_signals": len(xperp_entries),
        "holdout_start": int(xperp_entries.entry_time.min()) if len(xperp_entries) else None,
        "holdout_end": int(xperp_entries.entry_time.max()) if len(xperp_entries) else None,
        "paired_raw_outcomes": transfer.paired_comparison(source_outcomes, xperp_outcomes),
        "source_exact_portfolio": source_exact,
        "xperp_exact_portfolio": xperp_exact,
        "objective_checks": objective,
        "decision": (
            "Keep as a shadow candidate only; the holdout is positive but too short "
            "and does not satisfy the requested frequency, win rate or return."
            if positive_limited else
            "Reject the confirmation because it failed the unseen October holdout."
        ),
        "sources": {**sources, **venue_sources,
                    str(Path(__file__)): hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
    }
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({
        "status": report["status"], "raw": len(holdout), "paired": len(xperp_entries),
        "paired_outcomes": report["paired_raw_outcomes"],
        "source": source_exact, "xperp": xperp_exact,
        "objective_checks": objective,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
