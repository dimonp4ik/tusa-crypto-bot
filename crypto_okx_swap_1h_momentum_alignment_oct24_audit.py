"""Audit causal fast-momentum alignment gates after the Oct-24 stop cluster.

The gate family and its winner are ranked on 2022-2023 only.  Calendar years
2024-2026 and direct X-Perp observations are reported after selection.  The
Oct-22/23 losses motivated this family, so their result is diagnostic and is
not represented as independent validation.
"""
from __future__ import annotations

import hashlib
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
OUT = ROOT / "crypto_okx_swap_1h_momentum_alignment_oct24.json"
ROWS_OUT = ROOT / "crypto_okx_swap_1h_momentum_alignment_oct24.pkl"
HORIZONS = (1, 3, 6, 12, 24)
SCOPES = ("all", "long_only", "short_only")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def apply_gate(frame: pd.DataFrame, horizon: int, scope: str) -> pd.DataFrame:
    aligned = frame.direction.astype(float) * frame[f"ret{horizon}"].astype(float) >= 0
    if scope == "long_only":
        keep = frame.direction.lt(0) | aligned
    elif scope == "short_only":
        keep = frame.direction.gt(0) | aligned
    elif scope == "all":
        keep = aligned
    else:
        raise ValueError(scope)
    return frame.loc[keep].copy()


def metrics(frame: pd.DataFrame, year: int) -> dict:
    return portfolio.evaluate(frame, (year,), 0)


def admissible(years: dict[str, dict]) -> bool:
    return all(value["n"] >= 350 and value["return_pct"] > 0
               and value["dd_pct"] <= 10
               and value["trades_per_calendar_day"] >= 1.0
               for value in years.values())


def rank(years: dict[str, dict]) -> tuple[float, float, float, float]:
    values = list(years.values())
    return (min(value["return_pct"] for value in values),
            sum(value["return_pct"] for value in values),
            -max(value["dd_pct"] for value in values),
            min(value["trades_per_calendar_day"] for value in values))


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

    diagnostics = []
    eligible = []
    for horizon in HORIZONS:
        for scope in SCOPES:
            filtered = apply_gate(historical, horizon, scope)
            development = {str(year): metrics(filtered, year)
                           for year in (2022, 2023)}
            candidate = {
                "horizon_hours": horizon,
                "scope": scope,
                "development": development,
                "admissible": admissible(development),
                "rank": list(rank(development)),
            }
            diagnostics.append(candidate)
            if candidate["admissible"]:
                eligible.append(candidate)
    if not eligible:
        raise RuntimeError("no alignment gate passed the declared development rails")
    selected = max(eligible, key=lambda value: tuple(value["rank"]))
    filtered_historical = apply_gate(
        historical, selected["horizon_hours"], selected["scope"])
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
    filtered_measured = apply_gate(
        measured, selected["horizon_hours"], selected["scope"])
    accepted = append.runtime_gate_pending(append.emission_rows(filtered_measured))
    completed = append.complete(accepted)
    direct = append.stats(completed)

    refresh = json.loads(REFRESH_REPORT.read_text(encoding="utf-8"))
    cutoff = int(pd.Timestamp(
        refresh["protocol"]["prior_checkpoint_last_entry_utc"]).timestamp())
    post_checkpoint = [row for row in completed if int(row["entry_time"]) > cutoff]
    post_checkpoint_stats = append.stats(post_checkpoint)
    baseline_direct = refresh["cumulative_direct"]
    blocked_recent = 5 - post_checkpoint_stats["n"]

    improves_development = all(
        selected["development"][str(year)]["return_pct"]
        > baseline_development[str(year)]["return_pct"] for year in (2022, 2023))
    improves_later = all(
        later[str(year)]["return_pct"] > baseline_later[str(year)]["return_pct"]
        for year in (2024, 2025, 2026))
    all_positive = all(value["return_pct"] > 0 for value in later.values())
    frequency_pass = all(value["trades_per_calendar_day"] >= 3
                         for value in (*selected["development"].values(), *later.values()))
    economic_pass = bool(direct["economic_gate"]["passes"])
    passed = bool(improves_development and improves_later and all_positive
                  and frequency_pass and economic_pass)

    pd.DataFrame(accepted).to_pickle(ROWS_OUT)
    report = {
        "status": ("MOMENTUM_ALIGNMENT_PASSED_AWAIT_NEW_FORWARD"
                   if passed else "MOMENTUM_ALIGNMENT_REJECTED"),
        "ready_for_live_orders": False,
        "order_eligible": False,
        "protocol": {
            "hypothesis": "entry direction should agree with causal trailing price momentum",
            "candidate_family": {"horizons_hours": list(HORIZONS),
                                 "scopes": list(SCOPES), "threshold": 0.0},
            "selection_and_ranking": [2022, 2023],
            "later_calendar_checks": [2024, 2025, 2026],
            "direct_warning": "Oct-22/23 losses motivated the family; direct blocking is not independent validation",
            "exit": "fixed 2% target, 1% stop, 12-hour maximum, stop-first",
            "retune_on_direct": False,
        },
        "baseline_development": baseline_development,
        "eligible_candidates": len(eligible),
        "selected_before_later_checks": selected,
        "development_diagnostics": sorted(
            diagnostics, key=lambda value: tuple(value["rank"]), reverse=True),
        "later_calendar_checks": later,
        "baseline_later_calendar_checks": baseline_later,
        "direct_xperp": direct,
        "baseline_direct_xperp": baseline_direct,
        "post_prior_checkpoint": post_checkpoint_stats,
        "recent_baseline_losses_blocked": blocked_recent,
        "gates": {
            "improves_both_development_years": improves_development,
            "improves_every_later_year": improves_later,
            "all_later_years_positive": all_positive,
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
            "The candidate family was proposed after observing the Oct-22/23 stop cluster.",
            "A blocked recent loss therefore cannot prove forward effectiveness.",
            "Public candles cannot reproduce order-book slippage or latency.",
            "Gross results exclude fees at user request.",
        ],
    }
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({
        "status": report["status"],
        "selected": selected,
        "later": later,
        "direct": direct,
        "post_prior_checkpoint": post_checkpoint_stats,
        "recent_baseline_losses_blocked": blocked_recent,
        "gates": report["gates"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
