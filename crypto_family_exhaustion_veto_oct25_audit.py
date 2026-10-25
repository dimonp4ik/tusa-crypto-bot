"""Chronological audit of family-specific exhaustion vetoes.

The Oct-22/23 direct cluster suggested two distinct failure modes: trend
entries chasing an already extended 24-hour move while the last closed hour
reverses, and pullback entries whose own 24-hour move still opposes the trade.
The direct cluster motivates the family but is diagnostic only.  Candidate
thresholds are selected on 2022, frozen, and checked sequentially thereafter.
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
OUT = ROOT / "crypto_family_exhaustion_veto_oct25.json"
ROWS_OUT = ROOT / "crypto_family_exhaustion_veto_oct25.pkl"

TREND_EXTENSIONS = (.02, .03, .04, .05)
COUNTER_HOURS = (0.0, .0025, .005)
PULLBACK_CONFLICTS = (.005, .01, .02)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def risk_masks(frame: pd.DataFrame) -> dict[str, pd.Series]:
    direction = frame.direction.astype(float)
    aligned_24 = direction * frame.ret24.astype(float)
    aligned_1 = direction * frame.ret1.astype(float)
    trend = frame.family_trend.fillna(0).eq(1)
    pullback = frame.family_pullback.fillna(0).eq(1)
    trend_masks = {
        f"trend_ext{extension:.4f}_counter1h{counter:.4f}": (
            trend & aligned_24.ge(extension) & aligned_1.le(-counter))
        for extension, counter in itertools.product(TREND_EXTENSIONS, COUNTER_HOURS)
    }
    pullback_masks = {
        f"pullback_24h_conflict{conflict:.4f}": (
            pullback & aligned_24.le(-conflict))
        for conflict in PULLBACK_CONFLICTS
    }
    combined = {
        f"{trend_name}__plus__{pullback_name}": trend_mask | pullback_mask
        for trend_name, trend_mask in trend_masks.items()
        for pullback_name, pullback_mask in pullback_masks.items()
    }
    return {**trend_masks, **pullback_masks, **combined}


def metrics(frame: pd.DataFrame, year: int) -> dict:
    return portfolio.evaluate(frame, (year,), 0)


def compact(value: dict) -> dict:
    fields = ("n", "wr", "return_pct", "dd_pct", "max_loss_streak",
              "negative_months", "minimum_month_pct",
              "trades_per_calendar_day", "literal_target_hit_rate")
    return {field: value[field] for field in fields}


def dominates(candidate: dict, baseline: dict, minimum_n: int) -> bool:
    return bool(
        candidate["n"] >= minimum_n
        and candidate["return_pct"] >= baseline["return_pct"]
        and candidate["dd_pct"] <= baseline["dd_pct"]
        and candidate["negative_months"] <= baseline["negative_months"]
        and candidate["minimum_month_pct"] >= baseline["minimum_month_pct"]
        and candidate["max_loss_streak"] <= baseline["max_loss_streak"]
        and candidate["literal_target_hit_rate"] >= baseline["literal_target_hit_rate"]
    )


def objective(value: dict) -> bool:
    return bool(
        value["trades_per_calendar_day"] >= 3
        and value["negative_months"] == 0
        and value["months_ge_30pct"] == value["months"]
        and value["dd_pct"] <= 10
    )


def main() -> None:
    source = json.loads(STABLE_REPORT.read_text(encoding="utf-8"))
    policy = source["selected_before_direct"]
    historical_entries = stable.apply_policy(pd.read_pickle(FREQUENCY_ROWS), policy)
    global_candles, global_sources, global_coverage = legacy.load_merged(
        (append.OLD_GLOBAL, GLOBAL_MANIFEST), legacy.GLOBAL_FIELDS, 3600)
    historical = material.add_hourly_outcomes(
        historical_entries, global_candles, router.TARGET_PCT, router.HOLD_BARS)

    baseline_2022 = metrics(historical, 2022)
    frontier = []
    eligible = []
    for name, blocked in risk_masks(historical).items():
        candidate = historical.loc[~blocked].copy()
        value = metrics(candidate, 2022)
        passed = dominates(value, baseline_2022, 650)
        item = {"guard": name, "blocked_raw_rows": int(blocked.sum()),
                "selection_2022": value, "selection_pass": passed}
        frontier.append(item)
        if passed:
            rank = (baseline_2022["negative_months"]-value["negative_months"],
                    value["minimum_month_pct"], value["return_pct"],
                    -value["dd_pct"], value["literal_target_hit_rate"])
            eligible.append((rank, name, candidate, item))

    frontier.sort(key=lambda item: (
        item["selection_pass"], item["selection_2022"]["minimum_month_pct"],
        item["selection_2022"]["return_pct"],
        -item["selection_2022"]["dd_pct"]), reverse=True)
    frozen = validation = test = confirmation = direct = None
    selected_frame = None
    status = "REJECTED_SELECTION"
    if eligible:
        _, name, selected_frame, selected = max(eligible, key=lambda item: item[0])
        baseline_2023 = metrics(historical, 2023)
        validation = {"baseline": baseline_2023,
                      "candidate": metrics(selected_frame, 2023)}
        validation["passes"] = dominates(
            validation["candidate"], baseline_2023, 550)
        frozen = {"guard": name, "eligible_guards": len(eligible),
                  "selection_baseline_2022": baseline_2022,
                  "selection_2022": selected["selection_2022"]}
        status = "REJECTED_VALIDATION"
        if validation["passes"]:
            baseline_2024 = metrics(historical, 2024)
            test = {"baseline": baseline_2024,
                    "candidate": metrics(selected_frame, 2024)}
            test["passes"] = dominates(test["candidate"], baseline_2024, 550)
            status = "REJECTED_TEST_2024"
            if test["passes"]:
                confirmation = {
                    str(year): {"baseline": metrics(historical, year),
                                "candidate": metrics(selected_frame, year)}
                    for year in (2025, 2026)
                }

                raw = base.build_features(global_candles)
                proposed = base.proposals(raw).reset_index(drop=True)
                direct_entries = entries.frozen_entries(proposed)
                direct_mask = risk_masks(direct_entries)[name]
                filtered_entries = direct_entries.loc[~direct_mask].copy()
                xperp_candles, xperp_sources, xperp_coverage = legacy.load_merged(
                    (append.OLD_XPERP, XPERP_MANIFEST), legacy.XPERP_FIELDS, 300)
                measured = material.resolve_available(
                    filtered_entries, xperp_candles, router.TARGET_PCT,
                    int(router.HOLD_BARS*12))
                accepted = append.runtime_gate_pending(append.emission_rows(measured))
                completed = append.complete(accepted)
                refresh = json.loads(REFRESH_REPORT.read_text(encoding="utf-8"))
                cutoff = int(pd.Timestamp(
                    refresh["protocol"]["prior_checkpoint_last_entry_utc"]).timestamp())
                recent = [row for row in completed if int(row["entry_time"]) > cutoff]
                direct = {
                    "cumulative": append.stats(completed),
                    "post_prior_checkpoint": append.stats(recent),
                    "recent_baseline_rows": 5,
                    "recent_rows_blocked": 5-len(recent),
                    "coverage": xperp_coverage,
                    "sources": xperp_sources,
                }
                passed_objective = all(
                    objective(value["candidate"]) for value in confirmation.values())
                status = ("FAMILY_EXHAUSTION_RESEARCH_CANDIDATE"
                          if passed_objective else "REJECTED_OBJECTIVE")
                pd.DataFrame(accepted).to_pickle(ROWS_OUT)

    report = {
        "status": status, "ready_for_live_orders": False,
        "order_eligible": False,
        "protocol": {
            "entry_route": "frozen stable direction hourly route",
            "exit": "fixed 2% target, 1% stop, 12-hour maximum, stop-first",
            "selection": 2022, "validation": 2023,
            "conditional_test": 2024, "confirmation_only": [2025, 2026],
            "direct_cluster": "diagnostic only; it motivated the family and is not validation",
            "trend_guard": "block extended direction-aligned 24h move when last closed hour reverses",
            "pullback_guard": "block when the symbol 24h move still opposes pullback direction",
            "market_entries_only": True,
            "money": "client-selected current-equity margin; diagnostics use 3% and fixed 10x",
        },
        "selection_baseline_2022": baseline_2022,
        "selection_frontier": frontier,
        "frozen_before_validation": frozen,
        "validation_2023": validation,
        "test_2024": test,
        "confirmation_2025_2026": confirmation,
        "direct_xperp_diagnostic": direct,
        "coverage": {"global": global_coverage},
        "sources": {
            **global_sources,
            str(FREQUENCY_ROWS): digest(FREQUENCY_ROWS),
            str(STABLE_REPORT): digest(STABLE_REPORT),
            str(GLOBAL_MANIFEST): digest(GLOBAL_MANIFEST),
            str(REFRESH_REPORT): digest(REFRESH_REPORT),
            str(Path(__file__).resolve()): digest(Path(__file__).resolve()),
        },
        "limitations": [
            "The candidate family was motivated by five already observed direct losses.",
            "Closed hourly features cannot reproduce order-book liquidity or tick order.",
            "A historical pass still requires a genuinely new forward interval before activation.",
            "Gross results exclude fees at user request.",
        ],
    }
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({
        "status": status, "eligible": len(eligible),
        "frozen": frozen,
        "validation": ({**validation, "baseline": compact(validation["baseline"]),
                        "candidate": compact(validation["candidate"])}
                       if validation else None),
        "test": ({**test, "baseline": compact(test["baseline"]),
                  "candidate": compact(test["candidate"])} if test else None),
        "confirmation": confirmation,
        "direct": direct,
        "top_frontier": [{**item,
                          "selection_2022": compact(item["selection_2022"])}
                         for item in frontier[:10]],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
