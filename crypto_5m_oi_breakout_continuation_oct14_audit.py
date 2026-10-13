"""Bounded OI/flow continuation audit after closed 5m range breakouts."""
from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

import crypto_5m_liquidation_sweep_actual_oct5_audit as exits
import crypto_5m_sweep_oi_flush_oct14_audit as oi_source
import crypto_5m_sweep_taker_flow_oct14_audit as flow_source


ROOT = Path("reports/audit_2026_09_08")
OUT = ROOT / "crypto_5m_oi_breakout_continuation_oct14.json"
TARGETS = (.5, 1.0, 1.5, 2.0)
STOPS = (.5, .75, 1.0)
HOLDS = (24, 48, 72)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def events(candles: dict, metrics: dict) -> pd.DataFrame:
    pieces = []
    for symbol, candle in candles.items():
        frame = pd.DataFrame({
            key: np.asarray(candle[key], dtype=float)
            for key in ("open", "high", "low", "close", "quote_volume",
                        "taker_buy_quote_volume")})
        times = np.asarray(candle["time"], dtype=np.int64)
        previous = frame.close.shift(1)
        tr = pd.concat(((frame.high - frame.low),
                        (frame.high - previous).abs(),
                        (frame.low - previous).abs()), axis=1).max(axis=1)
        atr = tr.rolling(14).mean()
        prior_high = frame.high.shift(1).rolling(24).max()
        prior_low = frame.low.shift(1).rolling(24).min()
        direction = pd.Series(0, index=frame.index, dtype=np.int8)
        direction.loc[frame.close > prior_high] = 1
        direction.loc[frame.close < prior_low] = -1
        boundary = prior_high.where(direction > 0, prior_low)
        depth = direction * (frame.close - boundary) / atr
        span = (frame.high - frame.low).replace(0, np.nan)
        close_location = ((frame.close - frame.low) / span).where(
            direction > 0, (frame.high - frame.close) / span)
        range_atr = span / atr
        volume_ratio = (frame.quote_volume
                        / frame.quote_volume.shift(1).rolling(20).median())
        flow_delta = (2 * frame.taker_buy_quote_volume
                      / frame.quote_volume - 1)
        aligned_flow = direction * flow_delta
        entry_index = np.arange(len(frame)) + 1
        valid = ((direction != 0).to_numpy()
                 & (entry_index < len(frame))
                 & np.isfinite(depth.to_numpy())
                 & np.isfinite(volume_ratio.to_numpy()))
        valid[-1] = False
        valid &= (times[np.minimum(entry_index, len(times) - 1)]
                  - times == 300)
        signal = np.flatnonzero(valid)
        entry = signal + 1
        stamp = pd.to_datetime(times[entry], unit="s", utc=True)
        found = pd.DataFrame({
            "symbol": symbol, "entry_index": entry,
            "entry_time": times[entry], "year": stamp.year.astype(int),
            "month": (stamp.year * 100 + stamp.month).astype(int),
            "direction": direction.iloc[signal].to_numpy(np.int8),
            "breakout_depth_atr": depth.iloc[signal].to_numpy(float),
            "close_location": close_location.iloc[signal].to_numpy(float),
            "range_atr": range_atr.iloc[signal].to_numpy(float),
            "volume_ratio": volume_ratio.iloc[signal].to_numpy(float),
            "flow_aligned": aligned_flow.iloc[signal].to_numpy(float),
        })
        raw_metric = metrics[symbol]
        metric = pd.DataFrame({
            key: np.asarray(value, dtype=float)
            for key, value in raw_metric.items()}).set_index("time")
        open_interest = metric.sum_open_interest.replace(0, np.nan)
        oi_features = pd.DataFrame({
            "oi_change_1": open_interest.pct_change(1, fill_method=None),
            "oi_change_3": open_interest.pct_change(3, fill_method=None),
        })
        found[["oi_change_1", "oi_change_3"]] = oi_features.reindex(
            times[signal]).to_numpy()
        pieces.append(found)
    combined = pd.concat(pieces, ignore_index=True).replace(
        [np.inf, -np.inf], np.nan).dropna()
    return combined[
        combined.breakout_depth_atr.ge(.1)
        & combined.close_location.ge(.7)
        & combined.range_atr.ge(1.0)
        & combined.volume_ratio.ge(1.25)
    ].reset_index(drop=True)


def variant_masks(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    return {
        "base": np.ones(len(frame), dtype=bool),
        "aligned_flow_ge_02": frame.flow_aligned.ge(.2).to_numpy(),
        "oi1_ge_001": frame.oi_change_1.ge(.001).to_numpy(),
        "oi1_ge_001_and_flow_ge_0": (
            frame.oi_change_1.ge(.001)
            & frame.flow_aligned.ge(0)).to_numpy(),
    }


def annual(values: np.ndarray, years: np.ndarray) -> dict:
    output = {}
    for year in range(2022, 2027):
        found = values[years == year]
        output[str(year)] = {
            "n": len(found),
            "win_rate_pct": round(100 * float(np.mean(found > 0)), 4)
                if len(found) else 0.0,
            "mean_signed_price_return_pct": round(
                100 * float(found.mean()), 6) if len(found) else 0.0,
        }
    return output


def main() -> None:
    candles, sources = flow_source.load()
    metrics, metric_sources = oi_source.load_metrics()
    frame = events(candles, metrics)
    frame["_row_id"] = np.arange(len(frame))
    masks = variant_masks(frame)
    frontier = []
    for target, stop, hold in itertools.product(TARGETS, STOPS, HOLDS):
        resolved = exits.add_outcomes(
            frame.copy(), candles, target, stop, hold).sort_values("_row_id")
        values = resolved.signed_return.to_numpy(float)
        years = resolved.year.to_numpy(int)
        for name, mask in masks.items():
            yearly = annual(values[mask], years[mask])
            qualified = all(
                yearly[str(year)]["n"] >= 500
                and yearly[str(year)]["mean_signed_price_return_pct"] > 0
                for year in (2022, 2023, 2024))
            frontier.append({
                "variant": name, "target_pct": target,
                "stop_pct": stop, "hold_bars": hold,
                "annual": yearly, "qualified_before_2025": qualified,
                "pretest_minimum_mean_pct": min(
                    yearly[str(year)]["mean_signed_price_return_pct"]
                    for year in (2022, 2023, 2024)),
                "pretest_minimum_win_rate_pct": min(
                    yearly[str(year)]["win_rate_pct"]
                    for year in (2022, 2023, 2024)),
            })
    qualified = [item for item in frontier
                 if item["qualified_before_2025"]]
    frontier.sort(key=lambda item: (
        item["pretest_minimum_mean_pct"],
        item["pretest_minimum_win_rate_pct"]), reverse=True)
    report = {
        "status": "DEVELOPMENT_PASS" if qualified else "REJECTED_PRETEST",
        "ready_for_live_orders": False,
        "protocol": {
            "venue": "Binance USD-M futures",
            "event": (
                "closed 24-bar range breakout with >=0.1 ATR depth, >=0.7 "
                "directional close location, >=1 ATR range and >=1.25x volume"),
            "entry": "next 5m market-open proxy",
            "variants": list(masks), "targets_pct": list(TARGETS),
            "stops_pct": list(STOPS), "hold_bars": list(HOLDS),
            "selection": [2022, 2023], "validation": 2024,
            "test_if_pretest_passes": 2025, "observed": 2026,
            "pretest_gate": (
                ">=500 rows and positive raw expectancy in every 2022-2024 year"),
            "money": "price-path diagnostic; no position-size optimization",
        },
        "entry_rows": len(frame),
        "tested_combinations": len(frontier),
        "qualified_before_2025": len(qualified),
        "top_frontier": frontier[:20],
        "decision": (
            "No runtime change. The entire OI-expansion continuation family "
            "is rejected when no bounded exit geometry survives pretest."),
        "sources": {
            **sources, **metric_sources,
            str(Path(flow_source.__file__)): digest(Path(flow_source.__file__)),
            str(Path(oi_source.__file__)): digest(Path(oi_source.__file__)),
            str(Path(exits.__file__)): digest(Path(exits.__file__)),
            str(Path(__file__)): digest(Path(__file__)),
        },
        "limitations": [
            "Open interest and taker flow are five-minute aggregates.",
            "The bounded 144-combination grid still creates selection risk.",
            "2025-2026 are reported transparently after prior diagnostic access.",
        ],
    }
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({
        "status": report["status"], "entry_rows": len(frame),
        "tested": len(frontier), "qualified": len(qualified),
        "top": frontier[:5],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
