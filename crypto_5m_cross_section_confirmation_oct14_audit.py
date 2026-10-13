"""Post-hoc confirmation study for cross-sectional OI reversion losses."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

import crypto_5m_cross_section_oi_router_oct14_audit as base
import crypto_5m_cross_section_regime_oct14_audit as regime
import crypto_5m_sweep_oi_flush_oct14_audit as oi_source
import crypto_5m_sweep_taker_flow_oct14_audit as flow_source
from src import research_crypto_replay as replay


ROOT = Path("reports/audit_2026_09_08")
OUT = ROOT / "crypto_5m_cross_section_confirmation_oct14.json"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def masks(frame: pd.DataFrame) -> dict[str, pd.Series]:
    output = {"base": pd.Series(True, index=frame.index)}
    for threshold in (.0005, .001, .0015, .002, .003):
        output[f"price_confirmation_ge_{threshold}"] = (
            frame.aligned_ret1.ge(threshold))
    for threshold in (1.0, 1.25, 1.5, 2.0):
        output[f"volume_floor_ge_{threshold}"] = (
            frame.volume_ratio.ge(threshold))
    for threshold in (.0015, .002, .0025, .003):
        output[f"volatility_floor_ge_{threshold}"] = frame.rv48.ge(threshold)
    for threshold in (.2, .3, .4):
        output[f"flow_confirmation_ge_{threshold}"] = (
            frame.flow_aligned.ge(threshold))
    for threshold in (-.005, -.0075, -.01):
        output[f"oi_flush_le_{threshold}"] = frame.oi_change_3.le(threshold)
    output.update({
        "price_0005_volume_1": (
            frame.aligned_ret1.ge(.0005) & frame.volume_ratio.ge(1.0)),
        "price_001_volume_1": (
            frame.aligned_ret1.ge(.001) & frame.volume_ratio.ge(1.0)),
        "price_001_volume_125": (
            frame.aligned_ret1.ge(.001) & frame.volume_ratio.ge(1.25)),
        "price_0015_volume_125": (
            frame.aligned_ret1.ge(.0015) & frame.volume_ratio.ge(1.25)),
        "price_0015_volume_15": (
            frame.aligned_ret1.ge(.0015) & frame.volume_ratio.ge(1.5)),
        "price_001_rv_002": (
            frame.aligned_ret1.ge(.001) & frame.rv48.ge(.002)),
        "price_0015_rv_0025": (
            frame.aligned_ret1.ge(.0015) & frame.rv48.ge(.0025)),
        "price_volume_rv_loose": (
            frame.aligned_ret1.ge(.0005) & frame.volume_ratio.ge(1.0)
            & frame.rv48.ge(.002)),
        "price_volume_rv_medium": (
            frame.aligned_ret1.ge(.001) & frame.volume_ratio.ge(1.25)
            & frame.rv48.ge(.002)),
        "price_volume_flow_medium": (
            frame.aligned_ret1.ge(.001) & frame.volume_ratio.ge(1.25)
            & frame.flow_aligned.ge(.2)),
        "price_001_no_extreme": (
            frame.aligned_ret1.ge(.001) & frame.ret48.abs().le(.05)),
        "price_001_market_guard": (
            frame.aligned_ret1.ge(.001) & frame.aligned_btc48.ge(-.005)
            & frame.aligned_market48.ge(-.005)),
    })
    return output


def evaluate(frame: pd.DataFrame, candidate_masks: dict) -> list[dict]:
    results = []
    for name, mask in candidate_masks.items():
        rows = frame[mask.fillna(False)].copy()
        annual = {str(year): base.exact_year(rows, year)
                  for year in range(2022, 2027)}
        pretest_stable = all(base.component(annual[str(year)])
                             for year in range(2022, 2025))
        forward_positive = all(
            annual[str(year)]["return_pct"] > 0 for year in (2025, 2026))
        results.append({
            "variant": name,
            "rows": len(rows),
            "annual": annual,
            "stable_component_2022_2024": pretest_stable,
            "positive_2025_2026_posthoc": forward_positive,
            "minimum_pretest_return_pct": min(
                annual[str(year)]["return_pct"] for year in range(2022, 2025)),
            "minimum_forward_return_pct": min(
                annual[str(year)]["return_pct"] for year in (2025, 2026)),
            "minimum_all_year_win_rate_pct": min(
                annual[str(year)]["wr"] for year in range(2022, 2027)),
            "maximum_all_year_drawdown_pct": max(
                annual[str(year)]["dd_pct"] for year in range(2022, 2027)),
        })
    return sorted(results, key=lambda item: (
        item["stable_component_2022_2024"],
        item["positive_2025_2026_posthoc"],
        item["minimum_forward_return_pct"],
        item["minimum_pretest_return_pct"]), reverse=True)


def main() -> None:
    candles, candle_sources = flow_source.load()
    metrics, metric_sources = oi_source.load_metrics()
    feature_frame = base.features(candles, metrics)
    _, reversion = base.directional_frames(feature_frame)
    reversion = regime.enrich(reversion, candles)
    reversion = regime.add_directional_context(reversion)
    parent = reversion[
        reversion.oi_change_3.le(-.0025)
        & reversion.flow_aligned.ge(.1)
    ].copy()
    parent = replay.add_outcomes(parent, candles, .5, 1.0, 24)
    results = evaluate(parent, masks(parent))
    durable = [item for item in results
               if item["stable_component_2022_2024"]
               and item["positive_2025_2026_posthoc"]]
    report = {
        "status": "POSTHOC_DIAGNOSTIC",
        "ready_for_live_orders": False,
        "protocol": {
            "hypothesis": (
                "Require the final closed 5m candle, volume or realized volatility "
                "to confirm reversal after OI flush and aligned taker flow."),
            "parent": (
                "cross-sectional 4h reversion, OI3 <= -0.25%, flow >= 0.10, "
                "TP 0.5%, SL 1%, hold 24 bars"),
            "important": (
                "Hypothesis was derived after inspecting 2025-2026 cause profiles; "
                "forward columns are diagnostic and not an untouched validation."),
        },
        "tested_candidates": len(results),
        "positive_all_forward_posthoc": len(durable),
        "best_diagnostic": results[0],
        "durable_diagnostics": durable[:10],
        "all_candidates": results,
        "decision": (
            "A surviving confirmation may be shadowed, but cannot be promoted "
            "without new forward observations or an independent venue test."),
        "sources": {
            **candle_sources, **metric_sources,
            str(Path(base.__file__)): digest(Path(base.__file__)),
            str(Path(regime.__file__)): digest(Path(regime.__file__)),
            str(Path(replay.__file__)): digest(Path(replay.__file__)),
            str(Path(__file__)): digest(Path(__file__)),
        },
    }
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({
        "status": report["status"],
        "tested": len(results),
        "durable_posthoc": len(durable),
        "best": results[0],
        "durable_summary": [{
            "variant": item["variant"],
            "min_pretest": item["minimum_pretest_return_pct"],
            "min_forward": item["minimum_forward_return_pct"],
            "min_wr": item["minimum_all_year_win_rate_pct"],
            "max_dd": item["maximum_all_year_drawdown_pct"],
        } for item in durable[:10]],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
