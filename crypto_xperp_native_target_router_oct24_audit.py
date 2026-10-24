"""Venue-native exact target router with an untouched October test.

Unlike the mixed historical/direct model, this router fits only on X-Perp
rows before September 2026.  September selects the target universe and
portfolio policy.  October remains untouched until the selection is frozen.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

import crypto_profit_first_walkforward_target_router_oct18_audit as base
import crypto_profit_objective_target_router_oct24_audit as objective
import crypto_walkforward_target_router_oct24_forward as refresh
import crypto_extended_target_router_oct24_audit as extended


ROOT = Path("reports/audit_2026_09_08")
OUT = ROOT / "crypto_xperp_native_target_router_oct24.json"
ROWS_OUT = ROOT / "crypto_xperp_native_target_router_oct24.pkl"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def route_native(targets: tuple[float, ...]) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    original = base.TARGETS
    base.TARGETS = targets
    try:
        direct = refresh.direct_wide(
            refresh.NEW_NEAR, refresh.NEW_FAR, refresh.NEW_CACHE)
        direct["month"] = refresh.month(direct)
        train = direct[direct.month < "2026-09"].copy()
        scoring = direct[direct.month.isin(
            ("2026-09", "2026-10"))].copy().reset_index(drop=True)
        routed = base.route(scoring, base.fit_models_on_rows(train, scoring))
        coverage = {
            "training_rows": len(train),
            "training_first_entry_utc": pd.Timestamp(
                train.entry_time.min(), unit="s", tz="UTC").isoformat(),
            "training_last_entry_utc": pd.Timestamp(
                train.entry_time.max(), unit="s", tz="UTC").isoformat(),
            "training_months": sorted(train.month.unique().tolist()),
        }
        return (routed[routed.month.eq("2026-09")].copy(),
                routed[routed.month.eq("2026-10")].copy(), coverage)
    finally:
        base.TARGETS = original


def main() -> None:
    routed = {name: route_native(targets)
              for name, targets in objective.UNIVERSES.items()}
    selections = []
    frontiers = {}
    for name, targets in objective.UNIVERSES.items():
        calibration, _, coverage = routed[name]
        frontier = objective.candidates(calibration, targets)
        frontiers[name] = frontier[:20]
        qualified = [item for item in frontier if item["qualified"]]
        if qualified:
            selections.append({
                "universe": name, "targets": list(targets),
                "training_coverage": coverage, **qualified[0]})
    if not selections:
        raise RuntimeError("no native September policy met the declared rails")
    selected = max(selections, key=lambda item: tuple(item["rank"]))
    targets = tuple(float(value) for value in selected["targets"])
    october = routed[selected["universe"]][1]
    october3, rows = objective.evaluate_policy(october, targets, selected)
    october5 = extended.metrics(rows, .05)
    pd.DataFrame(rows).to_pickle(ROWS_OUT)

    passed = bool(
        october3["economic_gate"]["passes"]
        and october3["negative_months"] == 0
        and october3["dd_pct"] <= 10)
    public_selected = {key: value for key, value in selected.items()
                       if key != "metrics"}
    report = {
        "status": ("XPERP_NATIVE_ROUTER_PASSED_NEEDS_LONGER_FORWARD"
                   if passed else "XPERP_NATIVE_ROUTER_REJECTED"),
        "ready_for_live_orders": False,
        "order_eligible": False,
        "protocol": {
            "fit": "X-Perp rows strictly before September 2026",
            "selection": "September 2026 only",
            "test": "October 2026 through Oct 24",
            "selection_objective": "maximum projected geometric return",
            "selection_rails": {"minimum_rows": 90,
                                "minimum_trades_per_day": 3,
                                "maximum_drawdown_pct": 10,
                                "negative_months": 0},
            "target_universes": {key: list(value)
                                 for key, value in objective.UNIVERSES.items()},
            "target_price_move_range_pct": [
                100 * base.MIN_TARGET_MOVE, 100 * base.MAX_TARGET_MOVE],
            "entry": "next 15-minute market open",
            "exit": "single stop-first OHLC path, fixed selected target",
            "orders": "market entry and market close",
            "retune_on_october": False,
        },
        "selected_before_october": public_selected,
        "selection_metrics": objective.compact(selected["metrics"]),
        "selection_frontiers": {
            name: [{**{key: value for key, value in item.items()
                       if key != "metrics"},
                    "metrics": objective.compact(item["metrics"])}
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
            "The venue-native fit spans only the available 2026 X-Perp history.",
            "September is one selection month and October one partial holdout month.",
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
