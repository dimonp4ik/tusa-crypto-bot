"""Audit a causal market-turn veto for the frozen stable hourly route.

A trade is vetoed only when both BTC return and cross-market breadth oppose
the proposed direction.  Parameters are ranked on 2022-2023, then fixed for
the 2024-2026 and direct X-Perp checks.  The recent direct stop cluster
motivated this family and is therefore diagnostic rather than a clean holdout.
"""
from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path

import pandas as pd

import crypto_5m_regime_router_literal_oct6_audit as portfolio
import crypto_okx_swap_1h_forward_append_oct8_audit as legacy
import crypto_okx_swap_1h_forward_append_oct16_audit as append
import crypto_okx_swap_1h_regime_oct7_audit as base
import crypto_okx_swap_1h_stable_direction_oct22_audit as stable
import crypto_okx_swap_1h_stable_direction_forward_oct22_audit as entries
import crypto_okx_swap_1h_stable_material_exit_oct22_audit as material
from src import crypto_okx_1h_shadow_router as router


ROOT = Path("reports/audit_2026_09_08")
FREQUENCY_ROWS = ROOT / "crypto_okx_swap_1h_frequency_oct22.pkl"
STABLE_REPORT = ROOT / "crypto_okx_swap_1h_stable_direction_oct22.json"
GLOBAL_MANIFEST = ROOT / "crypto_okx_swap_1h_expanded_oct24" / "manifest.json"
XPERP_MANIFEST = ROOT / "xperp_5m_oct24_forward" / "manifest.json"
REFRESH_REPORT = ROOT / "crypto_okx_swap_1h_stable_material_exit_oct24_forward.json"
OUT = ROOT / "crypto_okx_swap_1h_market_turn_veto_oct24.json"
ROWS_OUT = ROOT / "crypto_okx_swap_1h_market_turn_veto_oct24.pkl"
HORIZONS = (6, 24)
RETURN_THRESHOLDS = (0.0, .0025, .005)
BREADTH_MARGINS = (0.0, .125, .25)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def apply_veto(frame: pd.DataFrame, horizon: int, return_threshold: float,
               breadth_margin: float) -> pd.DataFrame:
    direction = frame.direction.astype(float)
    btc_opposes = direction * frame[f"btc_ret{horizon}"].astype(float) < -return_threshold
    breadth_signed = direction * (
        frame[f"breadth{horizon}"].astype(float) - .5)
    breadth_opposes = breadth_signed < -breadth_margin
    return frame.loc[~(btc_opposes & breadth_opposes)].copy()


def metrics(frame: pd.DataFrame, year: int) -> dict:
    return portfolio.evaluate(frame, (year,), 0)


def development_ok(years: dict[str, dict]) -> bool:
    return all(value["n"] >= 500 and value["return_pct"] > 0
               and value["dd_pct"] <= 8
               and value["trades_per_calendar_day"] >= 1.4
               for value in years.values())


def rank(years: dict[str, dict]) -> tuple[float, float, float, float]:
    values = list(years.values())
    return (min(value["return_pct"] for value in values),
            sum(value["return_pct"] for value in values),
            -max(value["dd_pct"] for value in values),
            min(value["trades_per_calendar_day"] for value in values))


def compact(value: dict) -> dict:
    fields = ("n", "wr", "return_pct", "dd_pct", "max_loss_streak",
              "negative_months", "trades_per_calendar_day",
              "literal_target_hit_rate")
    return {field: value[field] for field in fields}


def main() -> None:
    source = json.loads(STABLE_REPORT.read_text(encoding="utf-8"))
    policy = source["selected_before_direct"]
    historical_entries = stable.apply_policy(pd.read_pickle(FREQUENCY_ROWS), policy)
    global_candles, global_sources, global_coverage = legacy.load_merged(
        (append.OLD_GLOBAL, GLOBAL_MANIFEST), legacy.GLOBAL_FIELDS, 3600)
    historical = material.add_hourly_outcomes(
        historical_entries, global_candles, router.TARGET_PCT, router.HOLD_BARS)
    baseline_development = {str(year): metrics(historical, year)
                            for year in (2022, 2023)}

    candidates = []
    for horizon, return_threshold, breadth_margin in itertools.product(
            HORIZONS, RETURN_THRESHOLDS, BREADTH_MARGINS):
        found = apply_veto(
            historical, horizon, return_threshold, breadth_margin)
        development = {str(year): metrics(found, year)
                       for year in (2022, 2023)}
        candidates.append({
            "horizon_hours": horizon,
            "return_threshold": return_threshold,
            "breadth_margin": breadth_margin,
            "development": development,
            "admissible": development_ok(development),
            "rank": list(rank(development)),
        })
    eligible = [candidate for candidate in candidates if candidate["admissible"]]
    if not eligible:
        raise RuntimeError("no market-turn veto passed development rails")
    selected = max(eligible, key=lambda value: tuple(value["rank"]))
    filtered_historical = apply_veto(
        historical, selected["horizon_hours"], selected["return_threshold"],
        selected["breadth_margin"])
    later = {str(year): metrics(filtered_historical, year)
             for year in (2024, 2025, 2026)}
    baseline_later = {str(year): metrics(historical, year)
                      for year in (2024, 2025, 2026)}

    raw = base.build_features(global_candles)
    proposed = base.proposals(raw).reset_index(drop=True)
    selected_entries = entries.frozen_entries(proposed)
    xperp_candles, xperp_sources, xperp_coverage = legacy.load_merged(
        (append.OLD_XPERP, XPERP_MANIFEST), legacy.XPERP_FIELDS, 300)
    measured = material.resolve_available(
        selected_entries, xperp_candles, router.TARGET_PCT,
        int(router.HOLD_BARS * 12))
    filtered_measured = apply_veto(
        measured, selected["horizon_hours"], selected["return_threshold"],
        selected["breadth_margin"])
    accepted = append.runtime_gate_pending(append.emission_rows(filtered_measured))
    completed = append.complete(accepted)
    direct = append.stats(completed)

    refresh = json.loads(REFRESH_REPORT.read_text(encoding="utf-8"))
    cutoff = int(pd.Timestamp(
        refresh["protocol"]["prior_checkpoint_last_entry_utc"]).timestamp())
    post_checkpoint = [row for row in completed if int(row["entry_time"]) > cutoff]
    post_stats = append.stats(post_checkpoint)
    blocked = 5 - post_stats["n"]

    improves_development = all(
        selected["development"][str(year)]["return_pct"]
        > baseline_development[str(year)]["return_pct"] for year in (2022, 2023))
    improves_later = all(
        later[str(year)]["return_pct"] > baseline_later[str(year)]["return_pct"]
        for year in (2024, 2025, 2026))
    stable_later = all(value["return_pct"] > 0 and value["dd_pct"] <= 8
                       for value in later.values())
    frequency_pass = all(value["trades_per_calendar_day"] >= 3
                         for value in (*selected["development"].values(), *later.values()))
    economic_pass = bool(direct["economic_gate"]["passes"])
    passed = bool(improves_development and improves_later and stable_later
                  and frequency_pass and economic_pass)

    pd.DataFrame(accepted).to_pickle(ROWS_OUT)
    report = {
        "status": ("MARKET_TURN_VETO_PASSED_AWAIT_NEW_FORWARD"
                   if passed else "MARKET_TURN_VETO_REJECTED"),
        "ready_for_live_orders": False,
        "order_eligible": False,
        "protocol": {
            "veto": "BTC return and market breadth must both oppose entry direction",
            "selection_and_ranking": [2022, 2023],
            "later_calendar_checks": [2024, 2025, 2026],
            "candidate_family": {
                "horizons_hours": list(HORIZONS),
                "btc_return_thresholds": list(RETURN_THRESHOLDS),
                "breadth_margins_from_half": list(BREADTH_MARGINS),
            },
            "direct_warning": "recent losses motivated the family and are not independent validation",
            "retune_on_later_or_direct": False,
        },
        "baseline_development": baseline_development,
        "eligible_candidates": len(eligible),
        "selected_before_later_checks": selected,
        "candidate_diagnostics": [{
            **{key: value for key, value in candidate.items()
               if key != "development"},
            "development": {year: compact(value)
                            for year, value in candidate["development"].items()},
        } for candidate in sorted(
            candidates, key=lambda value: tuple(value["rank"]), reverse=True)],
        "later_calendar_checks": later,
        "baseline_later_calendar_checks": baseline_later,
        "direct_xperp": direct,
        "baseline_direct_xperp": refresh["cumulative_direct"],
        "post_prior_checkpoint": post_stats,
        "recent_baseline_losses_blocked": blocked,
        "gates": {
            "improves_both_development_years": improves_development,
            "improves_every_later_year": improves_later,
            "positive_later_and_dd_at_most_8": stable_later,
            "three_trades_per_day_every_year": frequency_pass,
            "direct_economic_gate": economic_pass,
            "passed": passed,
        },
        "coverage": {"global": global_coverage, "xperp": xperp_coverage},
        "sources": {
            **global_sources, **xperp_sources,
            str(FREQUENCY_ROWS): digest(FREQUENCY_ROWS),
            str(STABLE_REPORT): digest(STABLE_REPORT),
            str(GLOBAL_MANIFEST): digest(GLOBAL_MANIFEST),
            str(XPERP_MANIFEST): digest(XPERP_MANIFEST),
            str(REFRESH_REPORT): digest(REFRESH_REPORT),
            str(Path(__file__).resolve()): digest(Path(__file__).resolve()),
        },
        "limitations": [
            "The family was designed after the Oct-22/23 stop cluster.",
            "Only a later untouched forward interval can validate it directly.",
            "Public candles cannot reproduce order-book slippage or latency.",
            "Gross results exclude fees at user request.",
        ],
    }
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({
        "status": report["status"], "eligible": len(eligible),
        "selected": selected,
        "later": {year: compact(value) for year, value in later.items()},
        "direct": compact(direct) | {"economic_gate": direct["economic_gate"]},
        "post_prior_checkpoint": compact(post_stats),
        "recent_baseline_losses_blocked": blocked,
        "gates": report["gates"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
