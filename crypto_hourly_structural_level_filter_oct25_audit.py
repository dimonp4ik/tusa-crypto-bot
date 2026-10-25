"""Causal higher-timeframe level filter for the frozen hourly crypto route.

Only pivot levels confirmed before the signal close are visible.  The filter
tests whether a proposed trade has room to the next level, a nearby structural
anchor behind the stop, a repeated liquidity pool in its path, or a genuine
sweep/reclaim on the signal candle.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

import crypto_5m_regime_router_literal_oct6_audit as portfolio
import crypto_okx_swap_1h_forward_append_oct8_audit as legacy
import crypto_okx_swap_1h_forward_append_oct16_audit as append
import crypto_okx_swap_1h_regime_oct7_audit as base
import crypto_okx_swap_1h_stable_direction_oct22_audit as stable
import crypto_okx_swap_1h_stable_direction_forward_oct22_audit as forward
import crypto_okx_swap_1h_stable_material_exit_oct22_audit as material
from src import crypto_okx_1h_shadow_router as router


ROOT = Path("reports/audit_2026_09_08")
FREQUENCY_ROWS = ROOT / "crypto_okx_swap_1h_frequency_oct22.pkl"
STABLE_REPORT = ROOT / "crypto_okx_swap_1h_stable_direction_oct22.json"
GLOBAL_MANIFEST = ROOT / "crypto_okx_swap_1h_expanded_oct24" / "manifest.json"
XPERP_MANIFEST = ROOT / "xperp_5m_oct24_forward" / "manifest.json"
OUT = ROOT / "crypto_hourly_structural_level_filter_oct25.json"
ROWS_OUT = ROOT / "crypto_hourly_structural_level_filter_oct25.pkl"
LOOKBACK = 168


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def confirmed_pivots(high: np.ndarray, low: np.ndarray,
                     start: int, signal: int) -> tuple[np.ndarray, np.ndarray]:
    """Return 2-left/2-right pivots whose confirmation predates signal."""
    highs, lows = [], []
    lower = max(start, 2)
    upper = min(signal-2, len(high)-2)
    for index in range(lower, upper):
        if high[index] >= np.max(high[index-2:index+3]):
            highs.append(float(high[index]))
        if low[index] <= np.min(low[index-2:index+3]):
            lows.append(float(low[index]))
    return np.asarray(highs, dtype=float), np.asarray(lows, dtype=float)


def enrich_levels(frame: pd.DataFrame, candles: dict[str, dict]) -> pd.DataFrame:
    records = []
    for symbol, part in frame.groupby("symbol", sort=False):
        candle = candles[symbol]
        open_ = np.asarray(candle["open"], dtype=float)
        high = np.asarray(candle["high"], dtype=float)
        low = np.asarray(candle["low"], dtype=float)
        close = np.asarray(candle["close"], dtype=float)
        for row in part.to_dict("records"):
            entry_index = int(row["entry_index"])
            signal = entry_index-1
            if signal < 26 or entry_index >= len(open_):
                continue
            entry = float(open_[entry_index])
            direction = int(row["direction"])
            start = max(0, signal-LOOKBACK)
            piv_high, piv_low = confirmed_pivots(high, low, start, signal)
            if direction > 0:
                ahead = piv_high[piv_high > entry]
                behind = piv_low[piv_low < entry]
                ahead_level = float(np.min(ahead)) if len(ahead) else np.nan
                anchor_level = float(np.max(behind)) if len(behind) else np.nan
                prior_extreme = float(np.min(low[max(0, signal-24):signal]))
                sweep = bool(low[signal] < prior_extreme and close[signal] > prior_extreme)
            else:
                ahead = piv_low[piv_low < entry]
                behind = piv_high[piv_high > entry]
                ahead_level = float(np.max(ahead)) if len(ahead) else np.nan
                anchor_level = float(np.min(behind)) if len(behind) else np.nan
                prior_extreme = float(np.max(high[max(0, signal-24):signal]))
                sweep = bool(high[signal] > prior_extreme and close[signal] < prior_extreme)
            atr = max(float(row.get("atr14_pct", 0))*entry, 1e-12)
            tolerance = max(.001*entry, .15*atr)
            room = (direction*(ahead_level-entry)/entry
                    if np.isfinite(ahead_level) else .10)
            anchor = (direction*(entry-anchor_level)/entry
                      if np.isfinite(anchor_level) else .10)
            if np.isfinite(ahead_level):
                touch_source = high[start:signal] if direction > 0 else low[start:signal]
                ahead_touches = int(np.sum(np.abs(touch_source-ahead_level) <= tolerance))
                rejection = bool(
                    (high[signal] >= ahead_level-tolerance and close[signal] < ahead_level)
                    if direction > 0 else
                    (low[signal] <= ahead_level+tolerance and close[signal] > ahead_level))
            else:
                ahead_touches, rejection = 0, False
            if np.isfinite(anchor_level):
                touch_source = low[start:signal] if direction > 0 else high[start:signal]
                anchor_touches = int(np.sum(np.abs(touch_source-anchor_level) <= tolerance))
            else:
                anchor_touches = 0
            window_high = float(np.max(high[max(0, signal-72):signal+1]))
            window_low = float(np.min(low[max(0, signal-72):signal+1]))
            span = max(window_high-window_low, 1e-12)
            extension = ((entry-window_low)/span if direction > 0
                         else (window_high-entry)/span)
            row.update({
                "level_room_ahead_pct": float(max(room, 0.0)),
                "level_anchor_behind_pct": float(max(anchor, 0.0)),
                "level_room_ahead_atr": float(max(room, 0.0)*entry/atr),
                "level_anchor_behind_atr": float(max(anchor, 0.0)*entry/atr),
                "level_ahead_touches": ahead_touches,
                "level_anchor_touches": anchor_touches,
                "level_rejection_ahead": rejection,
                "level_sweep_reclaim_24": sweep,
                "level_directional_extension_72": float(extension),
            })
            records.append(row)
    return pd.DataFrame(records)


def masks(frame: pd.DataFrame) -> dict[str, pd.Series]:
    room = frame.level_room_ahead_pct
    anchor = frame.level_anchor_behind_pct
    extension = frame.level_directional_extension_72
    repeated_ahead = frame.level_ahead_touches.ge(2) & room.le(.02)
    anchored = frame.level_anchor_touches.ge(2)
    result = {"baseline": pd.Series(True, index=frame.index)}
    for level in (.005, .01, .015, .02, .03):
        result[f"room_ge_{level:.3f}"] = room.ge(level)
        result[f"room_le_{level:.3f}"] = room.le(level)
    for level in (.005, .0075, .01, .015, .02):
        result[f"anchor_le_{level:.4f}"] = anchor.le(level)
        result[f"anchor_ge_{level:.4f}"] = anchor.ge(level)
    for level in (.60, .70, .80, .90):
        result[f"extension_le_{level:.2f}"] = extension.le(level)
        result[f"extension_ge_{level:.2f}"] = extension.ge(level)
    result.update({
        "no_repeated_pool_ahead": ~repeated_ahead,
        "repeated_pool_ahead": repeated_ahead,
        "no_rejection_ahead": ~frame.level_rejection_ahead,
        "rejection_ahead": frame.level_rejection_ahead,
        "confirmed_anchor_inside_stop": anchored & anchor.le(.01),
        "sweep_reclaim_24": frame.level_sweep_reclaim_24,
        "room_015_anchor_015": room.ge(.015) & anchor.le(.015),
        "room_020_anchor_015": room.ge(.02) & anchor.le(.015),
        "room_015_not_extended": room.ge(.015) & extension.le(.80),
        "anchor_010_not_extended": anchor.le(.01) & extension.le(.80),
        "clean_path": room.ge(.015) & ~repeated_ahead
                      & ~frame.level_rejection_ahead,
        "structural_reference": frame.level_sweep_reclaim_24
                                | (anchored & anchor.le(.015)),
        "reference_with_room": room.ge(.015) & (
            frame.level_sweep_reclaim_24 | (anchored & anchor.le(.015))),
        "liquidity_magnet": room.le(.02) & frame.level_ahead_touches.ge(2),
        "level_on_either_side": room.le(.02) | anchor.le(.015),
        "avoid_unreferenced_open_air": ~(
            room.gt(.02) & anchor.gt(.02)
            & ~frame.level_sweep_reclaim_24),
    })
    return result


def apply_rule(frame: pd.DataFrame, name: str) -> pd.DataFrame:
    lookup = masks(frame)
    if name not in lookup:
        raise KeyError(name)
    return frame.loc[lookup[name]].copy()


def yearly(frame: pd.DataFrame, years: tuple[int, ...]) -> dict[str, dict]:
    return {str(year): material.metrics(frame, year) for year in years}


def ratio(value: float, baseline: float) -> float:
    return value/baseline if baseline else 0.0


def development_ok(values: dict[str, dict], baseline: dict[str, dict]) -> bool:
    return all(
        item["return_pct"] > 0
        and ratio(item["return_pct"], baseline[year]["return_pct"]) >= .70
        and item["dd_pct"] <= baseline[year]["dd_pct"]
        and ratio(item["trades_per_calendar_day"],
                  baseline[year]["trades_per_calendar_day"]) >= .65
        and item["negative_months"] <= baseline[year]["negative_months"]
        for year, item in values.items())


def validation_ok(value: dict, baseline: dict) -> bool:
    return bool(
        value["return_pct"] >= .85*baseline["return_pct"]
        and value["dd_pct"] <= baseline["dd_pct"]
        and value["negative_months"] <= baseline["negative_months"]
        and value["trades_per_calendar_day"] >= .65*baseline["trades_per_calendar_day"])


def rank(values: dict[str, dict], validation: dict) -> tuple:
    all_values = [*values.values(), validation]
    return (
        -sum(item["negative_months"] for item in all_values),
        min(item["return_pct"] for item in all_values),
        sum(item["return_pct"] for item in all_values),
        -max(item["dd_pct"] for item in all_values),
    )


def compact(value: dict) -> dict:
    return {key: value[key] for key in (
        "n", "wr", "return_pct", "dd_pct", "max_loss_streak",
        "negative_months", "trades_per_calendar_day",
        "literal_target_hit_rate")}


def main() -> None:
    stable_report = json.loads(STABLE_REPORT.read_text(encoding="utf-8"))
    policy = stable_report["selected_before_direct"]
    entries = stable.apply_policy(pd.read_pickle(FREQUENCY_ROWS), policy)
    global_candles, global_sources, global_coverage = legacy.load_merged(
        (append.OLD_GLOBAL, GLOBAL_MANIFEST), legacy.GLOBAL_FIELDS, 3600)
    resolved = material.add_hourly_outcomes(
        entries, global_candles, router.TARGET_PCT, router.HOLD_BARS)
    enriched = enrich_levels(resolved, global_candles)

    development_years = (2021, 2022)
    baseline_development = yearly(enriched, development_years)
    baseline_validation = material.metrics(enriched, 2023)
    candidates = []
    for name in masks(enriched):
        if name == "baseline":
            continue
        filtered = apply_rule(enriched, name)
        development = yearly(filtered, development_years)
        validation = material.metrics(filtered, 2023)
        item = {
            "rule": name,
            "development": development,
            "validation_2023": validation,
            "development_pass": development_ok(
                development, baseline_development),
        }
        item["validation_pass"] = bool(
            item["development_pass"]
            and validation_ok(validation, baseline_validation))
        item["rank"] = list(rank(development, validation))
        candidates.append(item)
    eligible = [item for item in candidates if item["validation_pass"]]
    eligible.sort(key=lambda item: tuple(item["rank"]), reverse=True)
    frozen = eligible[0] if eligible else None

    baseline_later = yearly(enriched, (2024, 2025, 2026))
    later = direct = baseline_direct = None
    accepted_rows: list[dict] = []
    if frozen:
        filtered = apply_rule(enriched, frozen["rule"])
        later = yearly(filtered, (2024, 2025, 2026))

        raw = base.build_features(global_candles)
        proposed = base.proposals(raw).reset_index(drop=True)
        selected_entries = forward.frozen_entries(proposed)
        enriched_entries = enrich_levels(selected_entries, global_candles)
        xperp_candles, _, _ = legacy.load_merged(
            (append.OLD_XPERP, XPERP_MANIFEST), legacy.XPERP_FIELDS, 300)
        measured = material.resolve_available(
            enriched_entries, xperp_candles, router.TARGET_PCT,
            int(router.HOLD_BARS*12))
        baseline_rows = append.runtime_gate_pending(append.emission_rows(measured))
        filtered_measured = apply_rule(measured, frozen["rule"])
        accepted_rows = append.runtime_gate_pending(
            append.emission_rows(filtered_measured))
        baseline_direct = append.stats(append.complete(baseline_rows))
        direct = append.stats(append.complete(accepted_rows))

    improves_later = bool(frozen and all(
        later[year]["return_pct"] >= .90*baseline_later[year]["return_pct"]
        and later[year]["dd_pct"] <= baseline_later[year]["dd_pct"]
        and later[year]["negative_months"] <= baseline_later[year]["negative_months"]
        for year in later))
    improves_direct = bool(
        direct and baseline_direct
        and direct["return_pct"] > baseline_direct["return_pct"]
        and direct["dd_pct"] <= baseline_direct["dd_pct"])
    passed = bool(improves_later and improves_direct)
    pd.DataFrame(accepted_rows).to_pickle(ROWS_OUT)
    report = {
        "status": ("STRUCTURAL_LEVEL_FILTER_AWAITING_FORWARD"
                   if passed else "STRUCTURAL_LEVEL_FILTER_REJECTED"),
        "ready_for_live_orders": False, "order_eligible": False,
        "protocol": {
            "entry_route": policy,
            "development": list(development_years),
            "validation": 2023,
            "later_checks": [2024, 2025, 2026],
            "levels": (
                "2-left/2-right hourly pivots confirmed before signal close; "
                "168-hour lookback"),
            "entry": "next hourly market open; direct execution uses X-Perp 5m market proxy",
            "exit": "fixed 2% target, 1% stop, 12-hour maximum, stop-first",
            "money": "client-selected current-equity margin; fixed 10x",
        },
        "feature_rows": len(enriched),
        "baseline_development": baseline_development,
        "baseline_validation_2023": baseline_validation,
        "eligible_before_later_checks": len(eligible),
        "frozen_before_later_checks": frozen,
        "candidate_diagnostics": sorted(
            candidates, key=lambda item: tuple(item["rank"]), reverse=True),
        "baseline_later": baseline_later,
        "later_checks": later,
        "baseline_direct_xperp": baseline_direct,
        "direct_xperp": direct,
        "gates": {
            "preserves_and_reduces_risk_in_later_years": improves_later,
            "improves_direct_xperp": improves_direct,
            "passed": passed,
        },
        "coverage": {"global": global_coverage},
        "sources": {
            **global_sources,
            str(FREQUENCY_ROWS): digest(FREQUENCY_ROWS),
            str(STABLE_REPORT): digest(STABLE_REPORT),
            str(GLOBAL_MANIFEST): digest(GLOBAL_MANIFEST),
            str(XPERP_MANIFEST): digest(XPERP_MANIFEST),
            str(Path(__file__).resolve()): digest(Path(__file__).resolve()),
        },
        "limitations": [
            "The frozen entry route itself was selected with all historical years.",
            "Pivot levels are an OHLC proxy for liquidity, not an order-book measurement.",
            "The recent direct interval is diagnostic after the observed loss cluster.",
        ],
    }
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({
        "status": report["status"],
        "eligible_before_later_checks": len(eligible),
        "frozen": frozen,
        "later": later,
        "direct": compact(direct) if direct else None,
        "baseline_direct": compact(baseline_direct) if baseline_direct else None,
        "gates": report["gates"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
