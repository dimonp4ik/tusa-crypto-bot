"""Causal regime and level diagnosis for the frozen cross-sectional reversion."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

import crypto_5m_cross_section_oi_router_oct14_audit as base
import crypto_5m_sweep_oi_flush_oct14_audit as oi_source
import crypto_5m_sweep_taker_flow_oct14_audit as flow_source
from src import research_crypto_replay as replay


ROOT = Path("reports/audit_2026_09_08")
OUT = ROOT / "crypto_5m_cross_section_regime_oct14.json"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def enrich(frame: pd.DataFrame, candles: dict) -> pd.DataFrame:
    chunks = []
    market_rows = []
    for symbol, candle in candles.items():
        close = pd.Series(np.asarray(candle["close"], dtype=float))
        high = pd.Series(np.asarray(candle["high"], dtype=float))
        low = pd.Series(np.asarray(candle["low"], dtype=float))
        quote = pd.Series(np.asarray(candle["quote_volume"], dtype=float))
        times = np.asarray(candle["time"], dtype=np.int64)
        ret1 = close.pct_change(fill_method=None)
        ret48 = close.pct_change(48, fill_method=None)
        market_rows.append(pd.DataFrame({
            "signal_time": times, "symbol": symbol,
            "market_ret48": ret48.to_numpy(float),
        }))
        part = frame[frame.symbol.eq(symbol)].copy()
        idx = part.entry_index.to_numpy(int) - 1
        prior_low = low.shift(1).rolling(288, min_periods=144).min()
        prior_high = high.shift(1).rolling(288, min_periods=144).max()
        median_volume = quote.shift(1).rolling(288, min_periods=144).median()
        part["ret1"] = ret1.iloc[idx].to_numpy(float)
        part["ret_prior36"] = (
            close.shift(12) / close.shift(48) - 1).iloc[idx].to_numpy(float)
        part["rv48"] = ret1.rolling(48, min_periods=24).std().iloc[
            idx].to_numpy(float)
        part["volume_ratio"] = (quote / median_volume).iloc[idx].to_numpy(float)
        part["prior_low_24h"] = prior_low.iloc[idx].to_numpy(float)
        part["prior_high_24h"] = prior_high.iloc[idx].to_numpy(float)
        part["signal_close"] = close.iloc[idx].to_numpy(float)
        part["signal_low"] = low.iloc[idx].to_numpy(float)
        part["signal_high"] = high.iloc[idx].to_numpy(float)
        chunks.append(part)
    found = pd.concat(chunks, ignore_index=True)
    market = pd.concat(market_rows, ignore_index=True).dropna()
    context = market.groupby("signal_time").market_ret48.agg(
        market_median_ret48="median",
        market_dispersion48="std",
        market_breadth_positive=lambda value: float(np.mean(value > 0)),
    ).reset_index()
    found = found.merge(context, on="signal_time", how="left")

    btc = candles["BTCUSDT"]
    btc_close = pd.Series(np.asarray(btc["close"], dtype=float))
    btc_ret1 = btc_close.pct_change(fill_method=None)
    btc_context = pd.DataFrame({
        "signal_time": np.asarray(btc["time"], dtype=np.int64),
        "btc_ret48": btc_close.pct_change(48, fill_method=None),
        "btc_ret288": btc_close.pct_change(288, fill_method=None),
        "btc_rv48": btc_ret1.rolling(48, min_periods=24).std(),
    })
    return found.merge(btc_context, on="signal_time", how="left")


def add_directional_context(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["aligned_ret1"] = frame.direction * frame.ret1
    frame["aligned_ret12"] = frame.direction * frame.ret12
    frame["aligned_ret48"] = frame.direction * frame.ret48
    frame["aligned_btc48"] = frame.direction * frame.btc_ret48
    frame["aligned_market48"] = frame.direction * frame.market_median_ret48
    long = frame.direction.gt(0)
    frame["level_distance24"] = np.where(
        long,
        (frame.signal_close - frame.prior_low_24h) / frame.signal_close,
        (frame.prior_high_24h - frame.signal_close) / frame.signal_close,
    )
    frame["level_reclaim24"] = np.where(
        long,
        frame.signal_low.lt(frame.prior_low_24h)
        & frame.signal_close.gt(frame.prior_low_24h),
        frame.signal_high.gt(frame.prior_high_24h)
        & frame.signal_close.lt(frame.prior_high_24h),
    )
    return frame.replace([np.inf, -np.inf], np.nan)


def candidate_masks(frame: pd.DataFrame) -> dict[str, pd.Series]:
    valid = pd.Series(True, index=frame.index)
    masks = {"base": valid}
    for threshold in (-.02, -.01, -.005, 0.0):
        masks[f"btc48_ge_{threshold}"] = frame.aligned_btc48.ge(threshold)
        masks[f"market48_ge_{threshold}"] = frame.aligned_market48.ge(threshold)
        masks[f"ret12_ge_{threshold}"] = frame.aligned_ret12.ge(threshold)
    for threshold in (.03, .05, .08):
        masks[f"btc24_abs_le_{threshold}"] = frame.btc_ret288.abs().le(threshold)
        masks[f"extreme4h_abs_le_{threshold}"] = frame.ret48.abs().le(threshold)
    for threshold in (.01, .02, .04):
        masks[f"level_distance_le_{threshold}"] = (
            frame.level_distance24.ge(-.002)
            & frame.level_distance24.le(threshold))
        masks[f"dispersion_le_{threshold}"] = frame.market_dispersion48.le(threshold)
    for threshold in (1.0, 1.5, 2.0):
        masks[f"volume_ratio_le_{threshold}"] = frame.volume_ratio.le(threshold)
    masks["level_reclaim24"] = frame.level_reclaim24.fillna(False)
    masks["long_only"] = frame.direction.gt(0)
    masks["short_only"] = frame.direction.lt(0)

    btc_guard = frame.aligned_btc48.ge(-.005)
    market_guard = frame.aligned_market48.ge(-.005)
    impulse_guard = frame.aligned_ret12.ge(-.01)
    level_guard = (frame.level_distance24.ge(-.002)
                   & frame.level_distance24.le(.02))
    range_guard = (frame.btc_ret288.abs().le(.05)
                   & frame.market_dispersion48.le(.04))
    volume_guard = frame.volume_ratio.le(2.0)
    masks.update({
        "market_guard": btc_guard & market_guard,
        "impulse_level_guard": impulse_guard & level_guard,
        "range_guard": range_guard,
        "market_impulse_guard": btc_guard & market_guard & impulse_guard,
        "market_level_guard": btc_guard & market_guard & level_guard,
        "market_range_guard": btc_guard & market_guard & range_guard,
        "market_impulse_level_guard": (
            btc_guard & market_guard & impulse_guard & level_guard),
        "full_causal_guard": (
            btc_guard & market_guard & impulse_guard & level_guard
            & range_guard & volume_guard),
    })
    return masks


def profile(frame: pd.DataFrame) -> dict:
    columns = [
        "aligned_ret1", "aligned_ret12", "aligned_btc48",
        "aligned_market48", "btc_ret288", "market_dispersion48",
        "level_distance24", "volume_ratio", "rv48",
    ]
    output = {}
    for year in range(2022, 2027):
        year_frame = frame[frame.year.eq(year)]
        output[str(year)] = {}
        for label, subset in {
                "literal_target": year_frame[year_frame.outcome_code.eq(2)],
                "stop_or_nonpositive": year_frame[year_frame.signed_return.le(0)],
        }.items():
            output[str(year)][label] = {
                "n": len(subset),
                "medians": {
                    column: round(float(subset[column].median()), 7)
                    if len(subset) else None for column in columns},
                "level_reclaim_rate_pct": round(
                    100 * float(subset.level_reclaim24.mean()), 4)
                    if len(subset) else None,
            }
    return output


def main() -> None:
    candles, candle_sources = flow_source.load()
    metrics, metric_sources = oi_source.load_metrics()
    feature_frame = base.features(candles, metrics)
    _, reversion = base.directional_frames(feature_frame)
    reversion = enrich(reversion, candles)
    reversion = add_directional_context(reversion)
    base_mask = (reversion.oi_change_3.le(-.0025)
                 & reversion.flow_aligned.ge(.1))
    selected = reversion[base_mask].copy()
    selected = replay.add_outcomes(selected, candles, .5, 1.0, 24)
    masks = candidate_masks(selected)
    candidates = []
    for name, mask in masks.items():
        rows = selected[mask.fillna(False)].copy()
        pretest = {str(year): base.exact_year(rows, year)
                   for year in range(2022, 2025)}
        stable = all(base.component(value) for value in pretest.values())
        candidates.append({
            "variant": name,
            "rows": len(rows),
            "exact_pretest": pretest,
            "stable_component_before_2025": stable,
            "minimum_pretest_return_pct": min(
                value["return_pct"] for value in pretest.values()),
            "maximum_pretest_drawdown_pct": max(
                value["dd_pct"] for value in pretest.values()),
            "total_pretest_return_pct": sum(
                value["return_pct"] for value in pretest.values()),
        })
    candidates.sort(key=lambda item: (
        item["stable_component_before_2025"],
        item["minimum_pretest_return_pct"],
        item["total_pretest_return_pct"],
        -item["maximum_pretest_drawdown_pct"]), reverse=True)
    frozen = candidates[0]
    frozen_rows = selected[masks[frozen["variant"]].fillna(False)].copy()
    exact = dict(frozen["exact_pretest"])
    exact.update({str(year): base.exact_year(frozen_rows, year)
                  for year in range(2025, 2027)})
    component_checks = {year: base.component(value)
                        for year, value in exact.items()}
    objective_checks = {year: base.objective(value)
                        for year, value in exact.items()}
    status = ("OBJECTIVE_PASS" if all(objective_checks.values())
              else "STABLE_COMPONENT" if all(component_checks.values())
              else "REJECTED_FORWARD")
    report = {
        "status": status,
        "ready_for_live_orders": False,
        "protocol": {
            "parent_route": (
                "cross-sectional 4h reversion; OI change <= -0.25%; "
                "directional taker-flow >= 0.10"),
            "entry": "next 5m market-open proxy",
            "exit": "0.5% literal target, 1.0% stop, 24 bars",
            "selection": "candidate filters ranked only on exact 2022-2024 portfolios",
            "test": 2025,
            "observed": 2026,
            "position": "3% current-equity margin and fixed 10x gross",
        },
        "candidate_count": len(candidates),
        "stable_components_before_2025": sum(
            item["stable_component_before_2025"] for item in candidates),
        "frozen_before_2025": frozen,
        "exact_frozen_portfolio": exact,
        "component_checks": component_checks,
        "objective_checks": objective_checks,
        "top_candidates": candidates[:15],
        "raw_cause_profile": profile(selected),
        "decision": (
            "Do not change runtime unless the frozen causal guard remains positive "
            "after the untouched 2025 and observed 2026 reveal."),
        "sources": {
            **candle_sources, **metric_sources,
            str(Path(base.__file__)): digest(Path(base.__file__)),
            str(Path(replay.__file__)): digest(Path(replay.__file__)),
            str(Path(__file__)): digest(Path(__file__)),
        },
        "limitations": [
            "Five-minute candles cannot reconstruct order-book liquidity depth.",
            "Rolling 24h highs and lows are causal support/resistance proxies.",
            "Candidate families were specified before revealing 2025-2026 results.",
        ],
    }
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({
        "status": status,
        "candidate_count": len(candidates),
        "stable_before_2025": report["stable_components_before_2025"],
        "frozen": frozen,
        "exact": exact,
        "component_checks": component_checks,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
