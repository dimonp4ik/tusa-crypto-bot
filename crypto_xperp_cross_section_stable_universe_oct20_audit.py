"""Cross-sectional X-Perp audit on symbols with full pre-May coverage.

The universe is selected only from data availability: every symbol starts in
April 2026 before the May selection month.  The trading grid and chronological
May / June-July / August-October protocol match the prior cross-section audit.
"""
from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path

import pandas as pd

import crypto_raw_geometry_walkforward_oct20_audit as geometry_audit
import crypto_raw_hourly_direction_oct19_audit as parent
import crypto_xperp_cross_section_reversal_oct20_audit as base


OUT = parent.ROOT / "crypto_xperp_cross_section_stable_universe_oct20.json"
ROWS_OUT = parent.ROOT / "crypto_xperp_cross_section_stable_universe_oct20.pkl"
SYMBOLS = ("BTCUSDT", "ETHUSDT", "SOLUSDT", "SUIUSDT", "XRPUSDT")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    candles = {}
    sources = {}
    coverage = {}
    for symbol in SYMBOLS:
        path = parent.rails.DIRECT_CACHE / f"{symbol}_15min_current.pkl"
        frame = parent.load_candles(path)
        candles[symbol] = frame.to_dict("list")
        sources[str(path)] = digest(path)
        coverage[symbol] = {
            "start": int(frame.time.min()), "end": int(frame.time.max()),
            "bars": len(frame),
        }

    candidates = []
    labeled_cache = {}
    for lookback, style in itertools.product(base.LOOKBACKS, base.STYLES):
        features = base.feature_rows(candles, lookback, style)
        for geometry in geometry_audit.GEOMETRIES:
            labeled_cache[(lookback, style, geometry)] = geometry_audit.relabel(
                features, geometry, direct=True)
        for minimum_universe, z_threshold, move_threshold, geometry in itertools.product(
                base.MINIMUM_UNIVERSES, base.Z_THRESHOLDS,
                base.MOVE_THRESHOLDS, geometry_audit.GEOMETRIES):
            selected_frame = base.select_entries(
                labeled_cache[(lookback, style, geometry)],
                minimum_universe, z_threshold, move_threshold)
            for loss_pause in parent.LOSS_PAUSES:
                for hourly_cap in parent.HOURLY_CAPS:
                    selection, _ = base.period_metrics(
                        selected_frame, base.SELECTION_MONTH,
                        loss_pause, hourly_cap)
                    confirmation = {
                        month: base.period_metrics(
                            selected_frame, month,
                            loss_pause, hourly_cap)[0]
                        for month in base.CONFIRMATION_MONTHS
                    }
                    values = [selection, *confirmation.values()]
                    candidates.append({
                        "lookback_bars": lookback, "style": style,
                        "minimum_universe": minimum_universe,
                        "z_threshold": z_threshold,
                        "minimum_move_pct": 100 * move_threshold,
                        "target_pct": 100 * geometry[0],
                        "stop_pct": 100 * geometry[1],
                        "hold_bars": geometry[2],
                        "loss_pause": loss_pause, "hourly_cap": hourly_cap,
                        "selection": selection, "confirmation": confirmation,
                        "admissible": all(base.period_ok(value)
                                          for value in values),
                    })
    candidates.sort(key=base.rank_key, reverse=True)
    selected = candidates[0]
    geometry = (float(selected["target_pct"]) / 100,
                float(selected["stop_pct"]) / 100,
                int(selected["hold_bars"]))
    chosen = base.select_entries(
        labeled_cache[(int(selected["lookback_bars"]),
                       str(selected["style"]), geometry)],
        int(selected["minimum_universe"]),
        float(selected["z_threshold"]),
        float(selected["minimum_move_pct"]) / 100)
    holdouts = {}
    output_rows = []
    for margin in (.03, .05):
        name = f"margin_{round(100 * margin)}pct"
        holdouts[name] = {}
        for month in base.HOLDOUT_MONTHS:
            metrics, rows = base.period_metrics(
                chosen, month, int(selected["loss_pause"]),
                int(selected["hourly_cap"]), margin)
            holdouts[name][month] = metrics
            if margin == .03:
                output_rows.extend(rows)
    pd.DataFrame(output_rows).to_pickle(ROWS_OUT)

    three = holdouts["margin_3pct"]
    objective = bool(
        selected["admissible"]
        and all(three[month]["return_pct"] >= 30
                and three[month]["negative_months"] == 0
                and three[month]["dd_pct"] <= 10
                and three[month]["trades_per_calendar_day"] >= 3
                for month in ("2026-08", "2026-09"))
        and three["2026-10"]["return_pct"] > 0)
    report = {
        "status": ("XPERP_STABLE_UNIVERSE_SHADOW_CANDIDATE"
                   if objective else "REJECTED_OBJECTIVE"),
        "ready_for_live_orders": False,
        "protocol": {
            "symbols": list(SYMBOLS),
            "universe_rule": "all symbols have direct history before May selection",
            "selection": base.SELECTION_MONTH,
            "confirmation": list(base.CONFIRMATION_MONTHS),
            "holdouts": list(base.HOLDOUT_MONTHS),
            "entry": "next 15-minute market open",
            "orders": "market only",
            "money": "client 3% or 5% current-equity margin; fixed 10x",
        },
        "coverage": coverage,
        "candidate_count": len(candidates),
        "admissible_count": sum(item["admissible"] for item in candidates),
        "selected_before_holdouts": selected,
        "top20_before_holdouts": candidates[:20],
        "holdouts": holdouts,
        "objective_pass": objective,
        "sources": {
            **sources,
            str(Path(__file__).resolve()): digest(Path(__file__).resolve()),
        },
        "limitations": [
            "Direct venue history begins in partial April 2026.",
            "The universe rule is availability-based, not performance-based.",
            "October is incomplete and remains diagnostic.",
            "Gross returns exclude fees at user request.",
            "No backtest guarantees a future monthly return.",
        ],
    }
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True),
                   encoding="utf-8")
    print(json.dumps({
        "status": report["status"],
        "candidate_count": report["candidate_count"],
        "admissible_count": report["admissible_count"],
        "selected": selected, "holdouts": holdouts,
        "objective_pass": objective,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
