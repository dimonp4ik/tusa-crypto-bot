"""Pre-2025 exit-geometry selection for the confirmed cross-sectional route."""
from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path

import crypto_5m_cross_section_confirmation_holdout_oct16_audit as holdout_source
import crypto_5m_cross_section_confirmation_xperp_oct16_audit as transfer
import crypto_5m_cross_section_oi_router_oct14_audit as base
import crypto_5m_sweep_oi_flush_oct14_audit as oi_source
import crypto_5m_sweep_taker_flow_oct14_audit as flow_source
from src import research_crypto_replay as replay


ROOT = Path("reports/audit_2026_09_08")
OUT = ROOT / "crypto_5m_cross_section_confirmation_geometry_oct16.json"
TARGETS = (.5, .75, 1.0, 1.5, 2.0)
STOPS = (.5, .75, 1.0)
HOLDS = (12, 24, 48)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pretest_candidate(exact: dict) -> bool:
    return all(base.component(exact[str(year)]) for year in (2022, 2023, 2024))


def score(exact: dict) -> tuple:
    years = [exact[str(year)] for year in (2022, 2023, 2024)]
    return (
        min(value["economic_gate"]["projected_monthly_return_pct"] for value in years),
        min(value["return_pct"] for value in years),
        min(value["mean_signed_price_return_pct"] for value in years),
        -max(value["dd_pct"] for value in years),
    )


def public(item: dict) -> dict:
    return {key: value for key, value in item.items() if not key.startswith("_")}


def resolve(frame, candles, target: float, stop: float, hold: int):
    return replay.add_outcomes(frame.copy(), candles, target, stop, hold)


def main() -> None:
    source_candles, candle_sources = flow_source.load()
    source_metrics, metric_sources = oi_source.load_metrics()
    frozen_signals = transfer.signal_frame(source_candles, source_metrics)
    candidates = []
    for target, stop, bars in itertools.product(TARGETS, STOPS, HOLDS):
        outcomes = resolve(frozen_signals, source_candles, target, stop, bars)
        exact = {str(year): base.exact_year(outcomes, year)
                 for year in (2022, 2023, 2024)}
        candidates.append({
            "target_pct": target, "stop_pct": stop, "hold_bars": bars,
            "development": exact, "pretest_component": pretest_candidate(exact),
            "selection_score": score(exact), "_outcomes": outcomes,
        })
    qualified = sorted(
        (item for item in candidates if item["pretest_component"]),
        key=lambda item: item["selection_score"], reverse=True)
    selected = qualified[0] if qualified else None
    if selected is None:
        report = {
            "status": "REJECTED_DEVELOPMENT", "ready_for_live_orders": False,
            "tested_geometries": len(candidates), "qualified_pretest": 0,
            "decision": "No exit geometry survived all 2022-2024 component gates.",
            "frontier": [public(item) for item in sorted(
                candidates, key=lambda item: item["selection_score"], reverse=True)[:10]],
        }
        OUT.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
        print(json.dumps(report, indent=2), flush=True)
        return

    source_outcomes = selected["_outcomes"]
    revealed = {str(year): base.exact_year(source_outcomes, year)
                for year in (2025, 2026)}

    combined_candles, combined_metrics, combined_sources = holdout_source.combine_inputs()
    combined_signals = transfer.signal_frame(combined_candles, combined_metrics)
    october = combined_signals[combined_signals.entry_time.ge(
        holdout_source.HOLDOUT_START)].copy()
    october = transfer.remap_entries(october, combined_candles,
                                     min_future_bars=selected["hold_bars"])
    xperp_candles, venue_sources = transfer.load_xperp()
    xperp_entries = transfer.remap_entries(
        october, xperp_candles, min_future_bars=selected["hold_bars"])
    keys = {(str(row.symbol), int(row.entry_time)) for row in xperp_entries.itertuples()}
    source_entries = october[[
        (str(row.symbol), int(row.entry_time)) in keys
        for row in october.itertuples()
    ]].copy()
    source_october = resolve(
        source_entries, combined_candles, selected["target_pct"],
        selected["stop_pct"], selected["hold_bars"])
    xperp_october = resolve(
        xperp_entries, xperp_candles, selected["target_pct"],
        selected["stop_pct"], selected["hold_bars"])
    _, source_october_exact = transfer.exact(source_october)
    _, xperp_october_exact = transfer.exact(xperp_october)

    revealed_positive = all(
        value["return_pct"] > 0 for value in revealed.values())
    october_positive = bool(
        xperp_october_exact["n"] >= 20 and xperp_october_exact["return_pct"] > 0)
    objective_checks = {
        "every_revealed_year_positive": revealed_positive,
        "october_xperp_positive": october_positive,
        "october_win_rate_75_to_95": 75 <= xperp_october_exact["wr"] <= 95,
        "october_at_least_3_calendar_trades_per_day": (
            xperp_october_exact["trades_per_calendar_day"] >= 3),
        "october_no_negative_month": xperp_october_exact["negative_months"] == 0,
        "october_drawdown_at_most_5pct": xperp_october_exact["dd_pct"] <= 5,
        "october_return_at_least_30pct": xperp_october_exact["return_pct"] >= 30,
    }
    report = {
        "status": (
            "GEOMETRY_HOLDOUT_COMPONENT" if revealed_positive and october_positive
            else "REJECTED_REVEAL"),
        "ready_for_live_orders": False,
        "protocol": {
            "entry_variant": transfer.VARIANT,
            "entry_variant_warning": "entry mask was derived post-hoc on 2025-2026",
            "geometry_selection_only": [2022, 2023, 2024],
            "year_reveal": [2025, 2026],
            "unseen_archive_holdout": "2026-10-01 through 2026-10-14",
            "orders": "next-bar market open only",
            "collision": "stop first",
            "portfolio": "causal shared rails, 3% current-equity margin, fixed 10x",
        },
        "tested_geometries": len(candidates),
        "qualified_pretest": len(qualified),
        "selected_before_reveal": public(selected),
        "revealed_years": revealed,
        "october_paired_raw": transfer.paired_comparison(source_october, xperp_october),
        "october_source_exact": source_october_exact,
        "october_xperp_exact": xperp_october_exact,
        "objective_checks": objective_checks,
        "frontier": [public(item) for item in qualified[:10]],
        "decision": (
            "Preserve as a shadow component only. Promotion still requires the "
            "complete objective and live-forward evidence after the entry rule freeze."
            if revealed_positive and october_positive else
            "Reject the selected geometry because it failed a revealed year or October OKX."),
        "sources": {
            **candle_sources, **metric_sources, **combined_sources, **venue_sources,
            str(Path(__file__)): digest(Path(__file__)),
        },
    }
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({
        "status": report["status"],
        "tested": len(candidates), "qualified": len(qualified),
        "selected": {key: selected[key] for key in (
            "target_pct", "stop_pct", "hold_bars", "selection_score")},
        "revealed": revealed,
        "october_source": source_october_exact,
        "october_xperp": xperp_october_exact,
        "checks": objective_checks,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
