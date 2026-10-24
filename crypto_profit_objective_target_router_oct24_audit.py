"""Select a target-router policy for the stated economic objective.

Models fit on data before September 2026.  September alone selects between
the exact 0.5/1/2R and 0.5/1/2/3/4R target universes and their threshold/
portfolio rails by projected geometric return, subject to frequency and
drawdown constraints.  October is revealed only after that choice is frozen.
"""
from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path

import pandas as pd

import crypto_extended_target_router_oct24_audit as extended
import crypto_profit_first_walkforward_target_router_oct18_audit as base
import crypto_walkforward_target_router_oct24_forward as refresh


ROOT = Path("reports/audit_2026_09_08")
OUT = ROOT / "crypto_profit_objective_target_router_oct24.json"
ROWS_OUT = ROOT / "crypto_profit_objective_target_router_oct24.pkl"
UNIVERSES = {
    "three_targets": (.5, 1.0, 2.0),
    "five_targets": (.5, 1.0, 2.0, 3.0, 4.0),
}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compact(value: dict) -> dict:
    fields = ("n", "wr", "return_pct", "dd_pct", "max_loss_streak",
              "negative_months", "trades_per_calendar_day",
              "target_counts", "target_hit_rate", "non_loss_rate",
              "economic_gate")
    return {field: value[field] for field in fields}


def build_wide(targets: tuple[float, ...]) -> pd.DataFrame:
    return base.historical_wide() if len(targets) == 3 else extended.historical_wide()


def route_universe(targets: tuple[float, ...]) -> tuple[pd.DataFrame, pd.DataFrame]:
    original = base.TARGETS
    base.TARGETS = targets
    try:
        historical = build_wide(targets)
        direct = refresh.direct_wide(
            refresh.NEW_NEAR, refresh.NEW_FAR, refresh.NEW_CACHE)
        direct["month"] = refresh.month(direct)
        train = pd.concat([
            historical[~historical.period.eq("2026")],
            direct[direct.month < "2026-09"],
        ], ignore_index=True)
        scoring = direct[direct.month.isin(
            ("2026-09", "2026-10"))].copy().reset_index(drop=True)
        routed = base.route(scoring, base.fit_models_on_rows(train, scoring))
        return (routed[routed.month.eq("2026-09")].copy(),
                routed[routed.month.eq("2026-10")].copy())
    finally:
        base.TARGETS = original


def candidates(frame: pd.DataFrame, targets: tuple[float, ...]) -> list[dict]:
    original = base.TARGETS
    base.TARGETS = targets
    try:
        scores = frame.model_quality.replace(
            [float("inf"), float("-inf")], pd.NA).dropna()
        found = []
        for quantile, loss_pause, hourly_cap in itertools.product(
                base.QUANTILES, base.LOSS_PAUSES, base.HOURLY_CAPS):
            threshold = float(scores.quantile(quantile))
            _, rows = base.evaluate(
                frame, threshold, loss_pause, hourly_cap)
            value = extended.metrics(rows, .03)
            qualified = bool(
                value["n"] >= 90
                and value["trades_per_calendar_day"] >= 3
                and value["dd_pct"] <= 10
                and value["negative_months"] == 0)
            found.append({
                "quantile": quantile, "threshold": threshold,
                "loss_pause": loss_pause, "hourly_cap": hourly_cap,
                "qualified": qualified, "metrics": value,
                "rank": [
                    value["economic_gate"]["projected_monthly_return_pct"],
                    value["return_pct"], -value["dd_pct"], value["n"],
                ],
            })
        return sorted(found, key=lambda item: tuple(item["rank"]), reverse=True)
    finally:
        base.TARGETS = original


def evaluate_policy(frame: pd.DataFrame, targets: tuple[float, ...],
                    policy: dict) -> tuple[dict, list[dict]]:
    original = base.TARGETS
    base.TARGETS = targets
    try:
        _, rows = base.evaluate(
            frame, float(policy["threshold"]), int(policy["loss_pause"]),
            int(policy["hourly_cap"]))
        return extended.metrics(rows, .03), rows
    finally:
        base.TARGETS = original


def main() -> None:
    routed = {name: route_universe(targets)
              for name, targets in UNIVERSES.items()}
    selections = []
    frontiers = {}
    for name, targets in UNIVERSES.items():
        calibration, _ = routed[name]
        frontier = candidates(calibration, targets)
        frontiers[name] = frontier[:20]
        qualified = [item for item in frontier if item["qualified"]]
        if qualified:
            selections.append({
                "universe": name, "targets": list(targets), **qualified[0]})
    if not selections:
        raise RuntimeError("no September policy met frequency/drawdown rails")
    selected = max(selections, key=lambda item: tuple(item["rank"]))
    targets = tuple(float(value) for value in selected["targets"])
    october = routed[selected["universe"]][1]
    october3, rows = evaluate_policy(october, targets, selected)
    october5 = extended.metrics(rows, .05)
    pd.DataFrame(rows).to_pickle(ROWS_OUT)

    passed = bool(
        october3["economic_gate"]["passes"]
        and october3["negative_months"] == 0
        and october3["dd_pct"] <= 10)
    public_selected = {key: value for key, value in selected.items()
                       if key != "metrics"}
    report = {
        "status": ("PROFIT_OBJECTIVE_ROUTER_PASSED_NEEDS_FULL_WALKFORWARD"
                   if passed else "PROFIT_OBJECTIVE_ROUTER_REJECTED"),
        "ready_for_live_orders": False,
        "order_eligible": False,
        "protocol": {
            "fit": "historical through 2025 plus X-Perp before September",
            "selection": "September 2026 only",
            "test": "October 2026 through Oct 24",
            "selection_objective": "maximum projected geometric return",
            "selection_rails": {"minimum_rows": 90,
                                "minimum_trades_per_day": 3,
                                "maximum_drawdown_pct": 10,
                                "negative_months": 0},
            "target_universes": {key: list(value)
                                 for key, value in UNIVERSES.items()},
            "target_price_move_range_pct": [
                100 * base.MIN_TARGET_MOVE, 100 * base.MAX_TARGET_MOVE],
            "entry": "next 15-minute market open",
            "exit": "single stop-first OHLC path, fixed selected target",
            "orders": "market entry and market close",
            "retune_on_october": False,
        },
        "selected_before_october": public_selected,
        "selection_metrics": compact(selected["metrics"]),
        "selection_frontiers": {
            name: [{**{key: value for key, value in item.items()
                       if key not in ("metrics",)},
                    "metrics": compact(item["metrics"])}
                   for item in values]
            for name, values in frontiers.items()},
        "october_holdout": {
            "margin_3pct": october3,
            "margin_5pct": october5,
        },
        "objective_pass": passed,
        "sources": {
            str(refresh.NEW_NEAR): digest(refresh.NEW_NEAR),
            str(refresh.NEW_FAR): digest(refresh.NEW_FAR),
            str(refresh.NEW_MANIFEST): digest(refresh.NEW_MANIFEST),
            str(Path(__file__).resolve()): digest(Path(__file__).resolve()),
        },
        "limitations": [
            "September is one policy-selection month and October one partial holdout month.",
            "A passing result would still require full historical walk-forward validation.",
            "Public candles cannot reproduce order-book slippage or latency.",
            "Gross results exclude fees at user request.",
        ],
    }
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({
        "status": report["status"],
        "selected_before_october": public_selected,
        "selection_metrics": report["selection_metrics"],
        "october_holdout": report["october_holdout"],
        "objective_pass": passed,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
