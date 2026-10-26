"""Stability selection for the full native X-Perp cross-sectional grid.

July 20-31 is the development slice.  August and September select a stable
fixed rule by their worst month; partial October is touched only afterwards.
"""
from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path

import pandas as pd

import crypto_raw_geometry_walkforward_oct20_audit as geometry_audit
import crypto_raw_hourly_direction_oct19_audit as parent
import crypto_xperp_cross_section_reversal_oct20_audit as prior
import crypto_xperp_dynamic_universe_cross_section_oct26_audit as base


OUT = parent.ROOT / "crypto_xperp_dynamic_universe_stability_oct26.json"
ROWS_OUT = parent.ROOT / "crypto_xperp_dynamic_universe_stability_oct26.pkl"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stable(metrics: dict) -> bool:
    return bool(metrics["return_pct"] >= 0
                and metrics["negative_months"] == 0
                and metrics["dd_pct"] <= 10
                and metrics["trades_per_calendar_day"] >= 3)


def rank_key(item: dict) -> tuple:
    periods = (item["august"], item["september"])
    return (
        base.development_ok(item["development"])
        and all(value["return_pct"] >= 30 for value in periods),
        base.development_ok(item["development"])
        and all(stable(value) for value in periods),
        min(float(value["return_pct"]) for value in periods),
        sum(float(value["return_pct"]) for value in periods),
        float(item["development"]["return_pct"]),
        -max(float(value["dd_pct"])
             for value in (item["development"], *periods)),
    )


def main() -> None:
    candles, _ = base.load_universe()
    old_cache = parent.rails.DIRECT_CACHE
    parent.rails.DIRECT_CACHE = base.MATERIAL
    candidates = []
    try:
        for lookback, style in itertools.product(base.LOOKBACKS, base.STYLES):
            for geometry in base.GEOMETRIES:
                features = prior.feature_rows(candles, lookback, style)
                labeled = geometry_audit.relabel(
                    features, geometry, direct=True)
                for minimum_universe, z_threshold, move_threshold in itertools.product(
                        base.MINIMUM_UNIVERSES, base.Z_THRESHOLDS,
                        base.MOVE_THRESHOLDS):
                    filtered = prior.select_entries(
                        labeled, minimum_universe, z_threshold, move_threshold)
                    for loss_pause, hourly_cap in itertools.product(
                            base.LOSS_PAUSES, base.HOURLY_CAPS):
                        development, _ = base.evaluate(
                            filtered, base.DEVELOPMENT, loss_pause, hourly_cap)
                        august, _ = base.evaluate(
                            filtered, base.SELECTION, loss_pause, hourly_cap)
                        september, _ = base.evaluate(
                            filtered, base.VALIDATION, loss_pause, hourly_cap)
                        candidates.append({
                            "lookback_bars": lookback,
                            "style": style,
                            "minimum_universe": minimum_universe,
                            "z_threshold": z_threshold,
                            "minimum_move_pct": 100 * move_threshold,
                            "target_pct": 100 * geometry[0],
                            "stop_pct": 100 * geometry[1],
                            "hold_bars": geometry[2],
                            "loss_pause": loss_pause,
                            "hourly_cap": hourly_cap,
                            "development": development,
                            "august": august,
                            "september": september,
                        })
        candidates.sort(key=rank_key, reverse=True)
        selected = candidates[0]
        frozen = base.selected_frame(candles, selected)
        october = {}
        output_rows = []
        for margin in (.03, .05):
            key = f"margin_{round(100 * margin)}pct"
            october[key], rows = base.evaluate(
                frozen, base.HOLDOUT, int(selected["loss_pause"]),
                int(selected["hourly_cap"]), margin)
            if margin == .03:
                output_rows.extend(rows)
        pd.DataFrame(output_rows).to_pickle(ROWS_OUT)
    finally:
        parent.rails.DIRECT_CACHE = old_cache

    three = october["margin_3pct"]
    objective = bool(
        base.development_ok(selected["development"])
        and selected["august"]["return_pct"] >= 30
        and selected["september"]["return_pct"] >= 30
        and three["economic_gate"]["projected_monthly_return_pct"] >= 30
        and three["dd_pct"] <= 10
        and three["trades_per_calendar_day"] >= 3)
    report = {
        "status": ("XPERP_DYNAMIC_STABILITY_SHADOW_CANDIDATE"
                   if objective else "REJECTED_OBJECTIVE"),
        "ready_for_live_orders": False,
        "objective_pass": objective,
        "protocol": {
            "symbols": list(candles),
            "dynamic_universe": True,
            "development": list(base.DEVELOPMENT),
            "stability_selection": [list(base.SELECTION),
                                    list(base.VALIDATION)],
            "holdout": list(base.HOLDOUT),
            "candidate_count": len(candidates),
            "selection_objective": "maximum worst August/September return",
            "entry": "next 15-minute market open",
            "orders": "market only",
            "money": "client-selected 3% or 5% current-equity margin; fixed 10x",
        },
        "selected_before_october": selected,
        "top20_before_october": candidates[:20],
        "october_partial_holdout": october,
        "objective_pass": objective,
        "sources": {
            str(base.MANIFEST): digest(base.MANIFEST),
            str(Path(__file__).resolve()): digest(Path(__file__).resolve()),
        },
        "limitations": [
            "Venue-native data begin in April 2026.",
            "August and September are selection months, not holdouts.",
            "October is the only holdout and ends on October 24.",
            "Gross results exclude fees at user request.",
        ],
    }
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True),
                   encoding="utf-8")
    print(json.dumps({
        "status": report["status"],
        "selected": selected,
        "october_partial_holdout": october,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
