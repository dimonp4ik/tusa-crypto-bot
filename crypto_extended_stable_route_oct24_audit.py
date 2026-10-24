"""Stable route filter for the exact 0.5R-4R X-Perp target router.

The five-target model is fit from the original pre-September data snapshot.
Its cutoff and account rails are selected on September.  Coarse route groups
must be profitable in two disjoint pre-September blocks; September chooses
the group rule, then October is opened once.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

import crypto_extended_target_router_oct24_audit as extended
import crypto_profit_first_walkforward_target_router_oct18_audit as base
import crypto_stable_route_group_oct24_audit as stable
import crypto_walkforward_target_router_oct24_forward as refresh


ROOT = Path("reports/audit_2026_09_08")
BASELINE_REPORT = ROOT / "crypto_walkforward_target_router_oct24_forward.json"
OUT = ROOT / "crypto_extended_stable_route_oct24_audit.json"
ROWS_OUT = ROOT / "crypto_extended_stable_route_oct24_audit.pkl"
TARGETS = extended.TARGETS


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compact(value: dict) -> dict:
    return {name: value[name] for name in (
        "n", "return_pct", "dd_pct", "wr", "max_loss_streak",
        "trades_per_calendar_day", "negative_months")}


def main() -> None:
    original_targets = base.TARGETS
    base.TARGETS = TARGETS
    try:
        historical = extended.historical_wide()
        old_direct = refresh.direct_wide(
            base.prior.FRESH_NEAR, base.prior.FRESH_FAR,
            base.be.DIRECT_CACHE)
        new_direct = refresh.direct_wide(
            refresh.NEW_NEAR, refresh.NEW_FAR, refresh.NEW_CACHE)
        old_direct["month"] = refresh.month(old_direct)
        new_direct["month"] = refresh.month(new_direct)
        older = old_direct[old_direct.month < "2026-09"].copy()
        train = pd.concat([
            historical[~historical.period.eq("2026")], older
        ], ignore_index=True)
        scoring = pd.concat([
            older,
            new_direct[new_direct.month.isin(("2026-09", "2026-10"))]
        ], ignore_index=True)
        routed = base.route(
            scoring, base.fit_models_on_rows(train, scoring))
        routed["month"] = refresh.month(routed)
        september_routed = routed[routed.month.eq("2026-09")].copy()
        october_routed = routed[routed.month.eq("2026-10")].copy()
        policy, policy_frontier = base.choose_policy(september_routed)

        by_month = {}
        for month in sorted(routed.month.unique()):
            by_month[month] = base.candidate_rows(
                routed[routed.month.eq(month)].copy(),
                float(policy["threshold"]))
        early = [row for month in (
            "2026-02", "2026-03", "2026-04", "2026-05")
                 for row in by_month.get(month, [])]
        late = [row for month in ("2026-06", "2026-07", "2026-08")
                for row in by_month.get(month, [])]
        september = by_month["2026-09"]
        october = by_month["2026-10"]
        baseline_sep_rows = stable.gated(september, policy)
        baseline_oct_rows = stable.gated(october, policy)
        baseline_sep = extended.metrics(baseline_sep_rows, .03)
        baseline_oct3 = extended.metrics(baseline_oct_rows, .03)
        baseline_oct5 = extended.metrics(baseline_oct_rows, .05)

        frontier = []
        learned = {}
        for spec in stable.GROUP_SPECS:
            for mode in ("allow_stable_positive", "deny_stable_negative"):
                for min_count in stable.MIN_COUNTS:
                    for mean_floor in stable.MEAN_FLOORS:
                        groups = stable.learned_groups(
                            early, late, spec, min_count, mean_floor, mode)
                        if not groups:
                            continue
                        rows = stable.gated(stable.apply_filter(
                            september, spec, groups, mode), policy)
                        if not rows:
                            continue
                        value = extended.metrics(rows, .03)
                        item = {
                            "spec": list(spec), "mode": mode,
                            "min_count_per_development_block": min_count,
                            "mean_r_floor": mean_floor,
                            "learned_group_count": len(groups),
                            "selection_metrics": compact(value),
                            "selection_score": round(
                                value["return_pct"]
                                - .25 * value["dd_pct"], 8),
                        }
                        frontier.append(item)
                        learned[(spec, mode, min_count, mean_floor)] = groups
        eligible = [item for item in frontier
                    if item["selection_metrics"]["n"] >= 90
                    and item["selection_metrics"][
                        "trades_per_calendar_day"] >= 3
                    and item["selection_metrics"]["dd_pct"] <= 10]
        pool = eligible or frontier
        pool.sort(key=lambda item: (
            item["selection_score"],
            item["selection_metrics"]["return_pct"],
            -item["selection_metrics"]["dd_pct"]), reverse=True)
        selected = pool[0]
        spec = tuple(selected["spec"])
        groups = learned[(
            spec, selected["mode"],
            selected["min_count_per_development_block"],
            selected["mean_r_floor"])]
        test_rows = stable.gated(stable.apply_filter(
            october, spec, groups, selected["mode"]), policy)
        oct3 = extended.metrics(test_rows, .03)
        oct5 = extended.metrics(test_rows, .05)
    finally:
        base.TARGETS = original_targets

    pd.DataFrame(test_rows).to_pickle(ROWS_OUT)
    three_target = json.loads(
        BASELINE_REPORT.read_text(encoding="utf-8"))["october_to_date"]
    improved = bool(
        oct3["return_pct"] > three_target["margin_3pct"]["return_pct"]
        and oct3["dd_pct"] <= three_target["margin_3pct"]["dd_pct"]
        and oct3["trades_per_calendar_day"] >= 3)
    objective = bool(
        oct3["economic_gate"]["passes"]
        and oct3["negative_months"] == 0 and oct3["dd_pct"] <= 10)
    report = {
        "status": ("EXTENDED_STABLE_ROUTE_IMPROVED_OCTOBER"
                   if improved else "REJECTED_EXTENDED_STABLE_ROUTE"),
        "ready_for_live_orders": False,
        "order_eligible": False,
        "protocol": {
            "fit": "historical through 2025 plus original X-Perp before September",
            "base_policy_selection": "September 2026",
            "group_development_early": "February through May 2026",
            "group_development_late": "June through August 2026",
            "group_selection": "September 2026 with base rails fixed",
            "untouched_test": "October 2026 through Oct 24",
            "targets_r": list(TARGETS),
            "target_price_move_range_pct": [
                100 * base.MIN_TARGET_MOVE, 100 * base.MAX_TARGET_MOVE],
            "october_used_for_selection": False,
        },
        "base_policy": {key: policy[key] for key in (
            "quantile", "threshold", "loss_pause", "hourly_cap")},
        "base_policy_frontier": policy_frontier,
        "extended_baseline": {
            "september": compact(baseline_sep),
            "october_margin_3pct": baseline_oct3,
            "october_margin_5pct": baseline_oct5,
        },
        "selected_group_filter": {
            **selected,
            "learned_groups": [list(key) for key in sorted(groups, key=str)],
        },
        "october_test": {"margin_3pct": oct3, "margin_5pct": oct5},
        "three_target_october": three_target,
        "improves_three_target_with_drawdown_and_frequency_rails": improved,
        "objective_pass": objective,
        "selection_frontier": pool[:20],
        "sources": {
            str(BASELINE_REPORT): digest(BASELINE_REPORT),
            str(refresh.NEW_NEAR): digest(refresh.NEW_NEAR),
            str(refresh.NEW_FAR): digest(refresh.NEW_FAR),
            str(refresh.NEW_MANIFEST): digest(refresh.NEW_MANIFEST),
            str(Path(__file__).resolve()): digest(Path(__file__).resolve()),
        },
        "limitations": [
            "Only one partial October month is available as final validation.",
            "September selects the base policy and the coarse group rule.",
            "Gross results exclude fees at user request.",
        ],
    }
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True),
                   encoding="utf-8")
    print(json.dumps({
        "status": report["status"], "base_policy": report["base_policy"],
        "extended_baseline": report["extended_baseline"],
        "selected_group_filter": report["selected_group_filter"],
        "october_test": {
            "margin_3pct": compact(oct3), "margin_5pct": compact(oct5)},
        "objective_pass": objective,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
