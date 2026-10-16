"""Cross-venue execution transfer of the post-hoc OI reversion confirmation.

Signals and causal features come from the frozen Binance futures archive.  Their
entry timestamps are remapped to native OKX X-Perp 5m candles, where market-open
entries, stops, targets and expiries are resolved independently.  The exercise
is an execution-venue transfer test, not untouched temporal validation.
"""
from __future__ import annotations

import hashlib
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

import crypto_5m_cross_section_confirmation_oct14_audit as confirmation
import crypto_5m_cross_section_oi_router_oct14_audit as base
import crypto_5m_cross_section_regime_oct14_audit as regime
import crypto_5m_sweep_oi_flush_oct14_audit as oi_source
import crypto_5m_sweep_taker_flow_oct14_audit as flow_source
from src import research_crypto_replay as replay


ROOT = Path("reports/audit_2026_09_08")
MANIFEST = ROOT / "xperp_5m_oct16_forward" / "manifest.json"
OUT = ROOT / "crypto_5m_cross_section_confirmation_xperp_oct16.json"
VARIANT = "price_001_no_extreme"
TARGET_PCT = .5
STOP_PCT = 1.0
HOLD_BARS = 24


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_xperp(manifest_path: Path = MANIFEST) -> tuple[dict, dict]:
    raw = manifest_path.read_bytes()
    manifest = json.loads(raw)
    if manifest.get("bar") != "5m":
        raise ValueError("X-Perp manifest must contain native 5m candles")
    candles, sources = {}, {str(manifest_path): hashlib.sha256(raw).hexdigest()}
    for symbol, meta in manifest["symbols"].items():
        if meta.get("status") != "OK":
            continue
        path = Path(meta["path"])
        payload = path.read_bytes()
        observed = hashlib.sha256(payload).hexdigest()
        if observed != meta.get("sha256"):
            raise ValueError(f"hash mismatch: {symbol}")
        candle = pickle.loads(payload)
        times = np.asarray(candle["time"], dtype=np.int64)
        if len(times) < 2 or np.any(np.diff(times) <= 0):
            raise ValueError(f"non-monotonic candles: {symbol}")
        candles[symbol] = candle
        sources[str(path)] = observed
    return candles, sources


def remap_entries(frame: pd.DataFrame, candles: dict,
                  min_future_bars: int = HOLD_BARS) -> pd.DataFrame:
    """Map exact entry timestamps; reject missing bars and short/gapped paths."""
    pieces = []
    for symbol, part in frame.groupby("symbol", sort=False):
        candle = candles.get(symbol)
        if candle is None:
            continue
        times = np.asarray(candle["time"], dtype=np.int64)
        by_time = {int(stamp): index for index, stamp in enumerate(times)}
        found = part[part.entry_time.isin(by_time)].copy()
        if found.empty:
            continue
        found["entry_index"] = found.entry_time.map(by_time).astype(int)
        keep = []
        for index in found.entry_index.to_numpy(int):
            end = min(len(times), index + min_future_bars)
            path = times[index:end]
            keep.append(len(path) >= min_future_bars
                        and bool(np.all(np.diff(path) == 300)))
        pieces.append(found[np.asarray(keep, dtype=bool)])
    if not pieces:
        return frame.iloc[:0].copy()
    return pd.concat(pieces, ignore_index=True).sort_values(
        ["entry_time", "symbol"]).reset_index(drop=True)


def signal_frame(candles: dict, metrics: dict) -> pd.DataFrame:
    feature_frame = base.features(candles, metrics)
    _, reversion = base.directional_frames(feature_frame)
    reversion = regime.enrich(reversion, candles)
    reversion = regime.add_directional_context(reversion)
    parent = reversion[
        reversion.oi_change_3.le(-.0025)
        & reversion.flow_aligned.ge(.1)
    ].copy()
    mask = confirmation.masks(parent)[VARIANT].fillna(False)
    return parent[mask].copy().sort_values(["entry_time", "symbol"])


def paired_comparison(source: pd.DataFrame, venue: pd.DataFrame) -> dict:
    columns = ["symbol", "entry_time", "signed_return", "outcome_code"]
    paired = source[columns].merge(
        venue[columns], on=["symbol", "entry_time"], suffixes=("_source", "_xperp"),
        validate="one_to_one")
    if paired.empty:
        return {"n": 0}
    source_values = paired.signed_return_source.to_numpy(float)
    venue_values = paired.signed_return_xperp.to_numpy(float)
    return {
        "n": len(paired),
        "positive_sign_agreement_pct": round(
            100 * float(np.mean((source_values > 0) == (venue_values > 0))), 4),
        "literal_target_agreement_pct": round(100 * float(np.mean(
            paired.outcome_code_source.eq(2) == paired.outcome_code_xperp.eq(2))), 4),
        "source_mean_signed_price_return_pct": round(100 * float(source_values.mean()), 6),
        "xperp_mean_signed_price_return_pct": round(100 * float(venue_values.mean()), 6),
        "mean_xperp_minus_source_bps": round(
            10_000 * float(np.mean(venue_values - source_values)), 4),
    }


def exact(frame: pd.DataFrame) -> tuple[list[dict], dict]:
    rows = base.normalize(frame)
    accepted = replay.runtime_gate(rows)
    return accepted, replay.money(accepted, .03)


def main() -> None:
    source_candles, candle_sources = flow_source.load()
    metrics, metric_sources = oi_source.load_metrics()
    signals = signal_frame(source_candles, metrics)
    xperp_candles, venue_sources = load_xperp()
    mapped = remap_entries(signals, xperp_candles)

    keys = pd.MultiIndex.from_frame(mapped[["symbol", "entry_time"]])
    source_overlap = signals.set_index(["symbol", "entry_time"]).loc[keys].reset_index()
    source_outcomes = replay.add_outcomes(
        source_overlap, source_candles, TARGET_PCT, STOP_PCT, HOLD_BARS)
    xperp_outcomes = replay.add_outcomes(
        mapped, xperp_candles, TARGET_PCT, STOP_PCT, HOLD_BARS)
    source_accepted, source_exact = exact(source_outcomes)
    xperp_accepted, xperp_exact = exact(xperp_outcomes)

    start = int(mapped.entry_time.min()) if len(mapped) else None
    end = int(mapped.entry_time.max()) if len(mapped) else None
    transfer_component = bool(
        xperp_exact["n"] >= 50
        and xperp_exact["return_pct"] > 0
        and xperp_exact["dd_pct"] <= 8
    )
    report = {
        "status": "TRANSFER_COMPONENT" if transfer_component else "REJECTED_TRANSFER",
        "ready_for_live_orders": False,
        "variant": VARIANT,
        "protocol": {
            "signal_features": "frozen Binance futures archive through 2026-09-30",
            "execution_transfer": "native OKX X-Perp next 5m market open",
            "exit": "0.5% literal target, 1.0% stop, stop-first, 24 bars",
            "portfolio": "shared causal runtime gate, 3% current-equity margin, 10x",
            "important": (
                "The variant was derived after viewing 2025-2026 source outcomes; "
                "this is cross-venue transfer, not untouched forward validation."
            ),
        },
        "candidate_signals_all_source_history": len(signals),
        "mapped_overlap_signals": len(mapped),
        "overlap_start": start,
        "overlap_end": end,
        "paired_raw_outcomes": paired_comparison(source_outcomes, xperp_outcomes),
        "source_exact_portfolio": source_exact,
        "xperp_exact_portfolio": xperp_exact,
        "source_accepted_entries": len(source_accepted),
        "xperp_accepted_entries": len(xperp_accepted),
        "objective_checks": {
            "win_rate_75_to_95": 75 <= xperp_exact["wr"] <= 95,
            "at_least_3_calendar_trades_per_day": (
                xperp_exact["trades_per_calendar_day"] >= 3),
            "no_negative_months": xperp_exact["negative_months"] == 0,
            "drawdown_at_most_5pct": xperp_exact["dd_pct"] <= 5,
            "return_at_least_30pct_per_month": all(
                value >= 30 for value in xperp_exact.get("monthly_return_pct", {}).values()),
        },
        "decision": (
            "Do not change runtime. A positive transfer only preserves this variant "
            "as a shadow candidate; a new untouched time window is still required."
            if transfer_component else
            "Reject the post-hoc confirmation as a venue-dependent result."
        ),
        "sources": {
            **candle_sources, **metric_sources, **venue_sources,
            str(Path(confirmation.__file__)): digest(Path(confirmation.__file__)),
            str(Path(base.__file__)): digest(Path(base.__file__)),
            str(Path(regime.__file__)): digest(Path(regime.__file__)),
            str(Path(replay.__file__)): digest(Path(replay.__file__)),
            str(Path(__file__)): digest(Path(__file__)),
        },
    }
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({
        "status": report["status"],
        "mapped": len(mapped),
        "paired": report["paired_raw_outcomes"],
        "source": source_exact,
        "xperp": xperp_exact,
        "objective_checks": report["objective_checks"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
