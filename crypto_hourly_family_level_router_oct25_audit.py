"""Family-specific structural-level router for frozen hourly crypto entries."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

import crypto_okx_swap_1h_forward_append_oct8_audit as legacy
import crypto_okx_swap_1h_forward_append_oct16_audit as append
import crypto_okx_swap_1h_regime_oct7_audit as base
import crypto_okx_swap_1h_stable_direction_oct22_audit as stable
import crypto_okx_swap_1h_stable_direction_forward_oct22_audit as forward
import crypto_okx_swap_1h_stable_material_exit_oct22_audit as material
import crypto_hourly_structural_level_filter_oct25_audit as levels
from src import crypto_okx_1h_shadow_router as router


ROOT = Path("reports/audit_2026_09_08")
OUT = ROOT / "crypto_hourly_family_level_router_oct25.json"
ROWS_OUT = ROOT / "crypto_hourly_family_level_router_oct25.pkl"
FAMILIES = (
    "family_trend", "family_pullback", "family_breakout",
    "family_exhaustion", "family_compression_release", "family_breadth",
    "family_relative_leader",
)
DEVELOPMENT = (2021, 2022)
VALIDATION = 2023
LATER = (2024, 2025, 2026)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def family_active(frame: pd.DataFrame, family: str) -> pd.Series:
    return frame[family].notna() & frame[family].astype(float).ne(0)


def apply_one(frame: pd.DataFrame, family: str, rule: str) -> pd.DataFrame:
    keep = ~family_active(frame, family) | levels.masks(frame)[rule]
    return frame.loc[keep].copy()


def apply_policy(frame: pd.DataFrame, policy: list[dict]) -> pd.DataFrame:
    result = frame
    for item in policy:
        result = apply_one(result, item["family"], item["rule"])
    return result


def yearly(frame: pd.DataFrame, years: tuple[int, ...]) -> dict[str, dict]:
    return {str(year): material.metrics(frame, year) for year in years}


def ratio(value: float, baseline: float) -> float:
    return value/baseline if baseline else 0.0


def stage_ok(value: dict, baseline: dict, return_floor: float) -> bool:
    return bool(
        value["return_pct"] > 0
        and ratio(value["return_pct"], baseline["return_pct"]) >= return_floor
        and value["dd_pct"] <= baseline["dd_pct"]
        and value["negative_months"] <= baseline["negative_months"]
        and ratio(value["trades_per_calendar_day"],
                  baseline["trades_per_calendar_day"]) >= .85)


def candidate_ok(development: dict[str, dict], validation: dict,
                 baseline_development: dict[str, dict],
                 baseline_validation: dict) -> bool:
    return bool(
        all(stage_ok(value, baseline_development[year], .90)
            for year, value in development.items())
        and stage_ok(validation, baseline_validation, .90))


def candidate_rank(development: dict[str, dict], validation: dict) -> tuple:
    values = [*development.values(), validation]
    return (
        -sum(value["negative_months"] for value in values),
        min(value["return_pct"] for value in values),
        sum(value["return_pct"] for value in values),
        -max(value["dd_pct"] for value in values),
    )


def compact(value: dict) -> dict:
    return {key: value[key] for key in (
        "n", "wr", "return_pct", "dd_pct", "max_loss_streak",
        "negative_months", "trades_per_calendar_day",
        "literal_target_hit_rate")}


def main() -> None:
    stable_report = json.loads(levels.STABLE_REPORT.read_text(encoding="utf-8"))
    entry_policy = stable_report["selected_before_direct"]
    entries = stable.apply_policy(
        pd.read_pickle(levels.FREQUENCY_ROWS), entry_policy)
    global_candles, global_sources, global_coverage = legacy.load_merged(
        (append.OLD_GLOBAL, levels.GLOBAL_MANIFEST), legacy.GLOBAL_FIELDS, 3600)
    resolved = material.add_hourly_outcomes(
        entries, global_candles, router.TARGET_PCT, router.HOLD_BARS)
    enriched = levels.enrich_levels(resolved, global_candles)
    rule_names = [name for name in levels.masks(enriched) if name != "baseline"]

    baseline_development = yearly(enriched, DEVELOPMENT)
    baseline_validation = material.metrics(enriched, VALIDATION)
    frontier = []
    qualified = []
    for family in FAMILIES:
        family_rows = int(family_active(enriched, family).sum())
        for rule in rule_names:
            filtered = apply_one(enriched, family, rule)
            blocked = len(enriched)-len(filtered)
            if blocked < 10:
                continue
            development = yearly(filtered, DEVELOPMENT)
            validation = material.metrics(filtered, VALIDATION)
            item = {
                "family": family, "family_rows": family_rows,
                "rule": rule, "blocked_rows": blocked,
                "development": development,
                "validation_2023": validation,
                "rank": list(candidate_rank(development, validation)),
            }
            item["qualified"] = candidate_ok(
                development, validation,
                baseline_development, baseline_validation)
            frontier.append(item)
            if item["qualified"]:
                qualified.append(item)

    best_by_family = []
    for family in FAMILIES:
        found = [item for item in qualified if item["family"] == family]
        if found:
            best_by_family.append(max(found, key=lambda item: tuple(item["rank"])))
    combined_policy = [{"family": item["family"], "rule": item["rule"]}
                       for item in best_by_family]
    combined = apply_policy(enriched, combined_policy)
    combined_development = yearly(combined, DEVELOPMENT)
    combined_validation = material.metrics(combined, VALIDATION)
    combined_ok = bool(combined_policy and candidate_ok(
        combined_development, combined_validation,
        baseline_development, baseline_validation))

    qualified.sort(key=lambda item: tuple(item["rank"]), reverse=True)
    if combined_ok:
        frozen_policy = combined_policy
        frozen_source = "best_independent_rule_per_family"
    elif qualified:
        frozen_policy = [{"family": qualified[0]["family"],
                          "rule": qualified[0]["rule"]}]
        frozen_source = "best_single_family_rule"
    else:
        frozen_policy = []
        frozen_source = None

    baseline_later = yearly(enriched, LATER)
    later = direct = baseline_direct = None
    accepted_rows: list[dict] = []
    if frozen_policy:
        frozen_historical = apply_policy(enriched, frozen_policy)
        later = yearly(frozen_historical, LATER)

        raw = base.build_features(global_candles)
        proposed = base.proposals(raw).reset_index(drop=True)
        selected_entries = forward.frozen_entries(proposed)
        enriched_entries = levels.enrich_levels(selected_entries, global_candles)
        xperp_candles, _, _ = legacy.load_merged(
            (append.OLD_XPERP, levels.XPERP_MANIFEST), legacy.XPERP_FIELDS, 300)
        measured = material.resolve_available(
            enriched_entries, xperp_candles, router.TARGET_PCT,
            int(router.HOLD_BARS*12))
        baseline_rows = append.runtime_gate_pending(append.emission_rows(measured))
        routed = apply_policy(measured, frozen_policy)
        accepted_rows = append.runtime_gate_pending(append.emission_rows(routed))
        baseline_direct = append.stats(append.complete(baseline_rows))
        direct = append.stats(append.complete(accepted_rows))

    improves_later = bool(frozen_policy and all(
        later[year]["return_pct"] >= .95*baseline_later[year]["return_pct"]
        and later[year]["dd_pct"] <= baseline_later[year]["dd_pct"]
        and later[year]["negative_months"] <= baseline_later[year]["negative_months"]
        for year in later))
    improves_direct = bool(
        direct and baseline_direct
        and direct["return_pct"] > baseline_direct["return_pct"]
        and direct["dd_pct"] <= baseline_direct["dd_pct"]
        and direct["trades_per_calendar_day"] >= .9*baseline_direct[
            "trades_per_calendar_day"])
    passed = bool(improves_later and improves_direct)
    pd.DataFrame(accepted_rows).to_pickle(ROWS_OUT)

    report = {
        "status": ("FAMILY_LEVEL_ROUTER_AWAITING_FORWARD"
                   if passed else "FAMILY_LEVEL_ROUTER_REJECTED"),
        "ready_for_live_orders": False, "order_eligible": False,
        "protocol": {
            "entry_route": entry_policy,
            "development": list(DEVELOPMENT), "validation": VALIDATION,
            "later_checks": list(LATER),
            "mechanism": (
                "each signal family may use a different causal pivot-level rule; "
                "rules are applied only to rows carrying that family flag"),
            "entry": "next hourly market open; X-Perp execution replayed on 5m bars",
            "exit": "fixed 2% target, 1% stop, 12-hour maximum, stop-first",
            "money": "client-selected current-equity margin; fixed 10x",
        },
        "feature_rows": len(enriched),
        "candidate_count": len(frontier),
        "qualified_individual_rules": len(qualified),
        "baseline_development": baseline_development,
        "baseline_validation_2023": baseline_validation,
        "best_by_family": best_by_family,
        "combined_policy": combined_policy,
        "combined_development": combined_development,
        "combined_validation_2023": combined_validation,
        "combined_passed_before_later": combined_ok,
        "frozen_policy": frozen_policy,
        "frozen_source": frozen_source,
        "baseline_later": baseline_later,
        "later_checks": later,
        "baseline_direct_xperp": baseline_direct,
        "direct_xperp": direct,
        "gates": {
            "preserves_and_reduces_risk_in_later_years": improves_later,
            "improves_direct_xperp": improves_direct,
            "passed": passed,
        },
        "frontier": sorted(
            frontier, key=lambda item: tuple(item["rank"]), reverse=True)[:80],
        "coverage": {"global": global_coverage},
        "sources": {
            **global_sources,
            str(levels.FREQUENCY_ROWS): digest(levels.FREQUENCY_ROWS),
            str(levels.STABLE_REPORT): digest(levels.STABLE_REPORT),
            str(levels.GLOBAL_MANIFEST): digest(levels.GLOBAL_MANIFEST),
            str(levels.XPERP_MANIFEST): digest(levels.XPERP_MANIFEST),
            str(Path(__file__).resolve()): digest(Path(__file__).resolve()),
        },
        "limitations": [
            "Family flags overlap, so combined rules may interact.",
            "The frozen entry route itself was selected with all historical years.",
            "Pivot levels are OHLC proxies rather than order-book liquidity.",
        ],
    }
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({
        "status": report["status"],
        "qualified_individual_rules": len(qualified),
        "best_by_family": [{
            "family": item["family"], "rule": item["rule"],
            "blocked_rows": item["blocked_rows"],
        } for item in best_by_family],
        "combined_passed_before_later": combined_ok,
        "frozen_policy": frozen_policy,
        "later": later,
        "direct": compact(direct) if direct else None,
        "baseline_direct": compact(baseline_direct) if baseline_direct else None,
        "gates": report["gates"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
