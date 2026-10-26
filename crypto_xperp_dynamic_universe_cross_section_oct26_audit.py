"""Chronological cross-sectional audit over the full native X-Perp universe.

Unlike the earlier five-symbol audit, this experiment keeps every symbol that
has venue-native candles at a timestamp.  The rule is developed on July 20-31,
selected on August, and then frozen for September and partial October.
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


ROOT = parent.ROOT
MATERIAL = ROOT / "xperp_15m_oct24_material"
MANIFEST = MATERIAL / "manifest.json"
OUT = ROOT / "crypto_xperp_dynamic_universe_cross_section_oct26.json"
ROWS_OUT = ROOT / "crypto_xperp_dynamic_universe_cross_section_oct26.pkl"

LOOKBACKS = (1, 2, 4, 8)
STYLES = ("reversal", "momentum")
Z_THRESHOLDS = (1.0, 1.5, 2.0)
MOVE_THRESHOLDS = (.002, .004, .006)
MINIMUM_UNIVERSES = (8, 10, 12)
GEOMETRIES = (
    (.005, .005, 32),
    (.0075, .005, 32),
    (.010, .005, 48),
    (.010, .0075, 48),
    (.015, .0075, 64),
    (.015, .010, 64),
    (.020, .010, 64),
    (.020, .010, 96),
)
LOSS_PAUSES = (3, 5)
HOURLY_CAPS = (1, 2)

DEVELOPMENT = ("2026-07-20", "2026-08-01")
SELECTION = ("2026-08-01", "2026-09-01")
VALIDATION = ("2026-09-01", "2026-10-01")
HOLDOUT = ("2026-10-01", "2026-10-25")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def time_slice(frame: pd.DataFrame, bounds: tuple[str, str]) -> pd.DataFrame:
    start = int(pd.Timestamp(bounds[0], tz="UTC").timestamp())
    end = int(pd.Timestamp(bounds[1], tz="UTC").timestamp())
    return frame[frame.entry_time.ge(start) & frame.entry_time.lt(end)].copy()


def evaluate(frame: pd.DataFrame, bounds: tuple[str, str], loss_pause: int,
             hourly_cap: int, margin: float = .03) -> tuple[dict, list[dict]]:
    return parent.evaluate(
        time_slice(frame, bounds), float("-inf"), loss_pause, hourly_cap,
        margin)


def development_ok(metrics: dict) -> bool:
    return bool(metrics["return_pct"] > 0
                and metrics["dd_pct"] <= 10
                and metrics["trades_per_calendar_day"] >= 3
                and metrics["n"] >= 36)


def selection_key(item: dict) -> tuple:
    selected = item["selection"]
    developed = item["development"]
    return (
        development_ok(developed),
        float(selected["return_pct"]),
        float(developed["return_pct"]),
        -max(float(selected["dd_pct"]), float(developed["dd_pct"])),
        int(selected["n"]),
    )


def load_universe() -> tuple[dict[str, pd.DataFrame], dict]:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    candles = {}
    for symbol, metadata in manifest["symbols"].items():
        if metadata.get("status") != "OK":
            continue
        path = Path(metadata["path"])
        if not path.is_absolute():
            path = Path.cwd() / path
        candles[symbol] = parent.load_candles(path)
    return candles, manifest


def labeled_frame(candles: dict[str, pd.DataFrame], lookback: int,
                  style: str, geometry: tuple[float, float, int]
                  ) -> pd.DataFrame:
    features = prior.feature_rows(candles, lookback, style)
    return geometry_audit.relabel(features, geometry, direct=True)


def selected_frame(candles: dict[str, pd.DataFrame], item: dict
                   ) -> pd.DataFrame:
    geometry = (item["target_pct"] / 100, item["stop_pct"] / 100,
                int(item["hold_bars"]))
    labeled = labeled_frame(candles, int(item["lookback_bars"]),
                            str(item["style"]), geometry)
    return prior.select_entries(
        labeled, int(item["minimum_universe"]), float(item["z_threshold"]),
        float(item["minimum_move_pct"]) / 100)


def main() -> None:
    candles, manifest = load_universe()
    old_cache = parent.rails.DIRECT_CACHE
    parent.rails.DIRECT_CACHE = MATERIAL
    candidates = []
    try:
        for lookback, style in itertools.product(LOOKBACKS, STYLES):
            for geometry in GEOMETRIES:
                labeled = labeled_frame(candles, lookback, style, geometry)
                for minimum_universe, z_threshold, move_threshold in itertools.product(
                        MINIMUM_UNIVERSES, Z_THRESHOLDS, MOVE_THRESHOLDS):
                    filtered = prior.select_entries(
                        labeled, minimum_universe, z_threshold, move_threshold)
                    for loss_pause, hourly_cap in itertools.product(
                            LOSS_PAUSES, HOURLY_CAPS):
                        development, _ = evaluate(
                            filtered, DEVELOPMENT, loss_pause, hourly_cap)
                        selection, _ = evaluate(
                            filtered, SELECTION, loss_pause, hourly_cap)
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
                            "selection": selection,
                        })
        candidates.sort(key=selection_key, reverse=True)
        selected = candidates[0]
        frozen = selected_frame(candles, selected)
        validation = {}
        holdout = {}
        output_rows = []
        for margin in (.03, .05):
            key = f"margin_{round(100 * margin)}pct"
            validation[key], rows = evaluate(
                frozen, VALIDATION, int(selected["loss_pause"]),
                int(selected["hourly_cap"]), margin)
            holdout[key], holdout_rows = evaluate(
                frozen, HOLDOUT, int(selected["loss_pause"]),
                int(selected["hourly_cap"]), margin)
            if margin == .03:
                output_rows.extend(rows)
                output_rows.extend(holdout_rows)
        pd.DataFrame(output_rows).to_pickle(ROWS_OUT)
    finally:
        parent.rails.DIRECT_CACHE = old_cache

    three_validation = validation["margin_3pct"]
    three_holdout = holdout["margin_3pct"]
    objective = bool(
        development_ok(selected["development"])
        and selected["selection"]["return_pct"] >= 30
        and three_validation["return_pct"] >= 30
        and three_validation["dd_pct"] <= 10
        and three_validation["trades_per_calendar_day"] >= 3
        and three_holdout["economic_gate"]["projected_monthly_return_pct"] >= 30
        and three_holdout["dd_pct"] <= 10)
    report = {
        "status": ("XPERP_DYNAMIC_CROSS_SECTION_SHADOW_CANDIDATE"
                   if objective else "REJECTED_OBJECTIVE"),
        "ready_for_live_orders": False,
        "objective_pass": objective,
        "protocol": {
            "symbols": list(candles),
            "dynamic_universe": True,
            "minimum_universes": list(MINIMUM_UNIVERSES),
            "lookbacks_bars": list(LOOKBACKS),
            "styles": list(STYLES),
            "z_thresholds": list(Z_THRESHOLDS),
            "minimum_move_pct": [100 * value for value in MOVE_THRESHOLDS],
            "geometries": [{"target_pct": 100 * item[0],
                             "stop_pct": 100 * item[1],
                             "hold_bars": item[2]} for item in GEOMETRIES],
            "development": list(DEVELOPMENT),
            "selection": list(SELECTION),
            "validation": list(VALIDATION),
            "holdout": list(HOLDOUT),
            "entry": "next 15-minute market open",
            "exit": "fixed target and stop; stop first on ambiguous bars",
            "orders": "market only",
            "money": "client-selected 3% or 5% current-equity margin; fixed 10x",
            "model": "none; deterministic relative-extreme direction",
        },
        "candidate_count": len(candidates),
        "development_survivor_count": sum(
            development_ok(item["development"]) for item in candidates),
        "selected_before_validation": selected,
        "top20_before_validation": candidates[:20],
        "validation": validation,
        "october_partial_holdout": holdout,
        "sources": {
            str(MANIFEST): digest(MANIFEST),
            str(Path(__file__).resolve()): digest(Path(__file__).resolve()),
        },
        "limitations": [
            "Venue-native data begin in April 2026 and symbols list at different dates.",
            "Only September is a complete untouched validation month.",
            "October ends on October 24 and its monthly figure is a projection.",
            "Public candles cannot reproduce order-book slippage or latency.",
            "Gross results exclude fees at user request.",
        ],
    }
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True),
                   encoding="utf-8")
    print(json.dumps({
        "status": report["status"],
        "candidate_count": report["candidate_count"],
        "development_survivor_count": report["development_survivor_count"],
        "selected": selected,
        "validation": validation,
        "october_partial_holdout": holdout,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
