"""Pre-October extended-target audit for the venue-adaptive crypto router.

The original walk-forward router considered 0.5R, 1R and 2R.  This variant
adds exact 3R and 4R paths while retaining the 0.5%-2% price-move bounds.  It
fits on historical data plus X-Perp rows before September, selects its cutoff
and runtime rails on September, and reveals October only after freezing.
"""
from __future__ import annotations

import collections
import hashlib
import json
from pathlib import Path

import pandas as pd

import crypto_profit_first_walkforward_target_router_oct18_audit as base
import crypto_walkforward_target_router_oct24_forward as refresh
from crypto_runtime_shadow_portfolio_oct3_audit import money
from src.research_profit_gate import economic_gate


ROOT = Path("reports/audit_2026_09_08")
BASELINE_REPORT = ROOT / "crypto_walkforward_target_router_oct24_forward.json"
OUT = ROOT / "crypto_extended_target_router_oct24.json"
ROWS_OUT = ROOT / "crypto_extended_target_router_oct24.pkl"
TARGETS = (.5, 1.0, 2.0, 3.0, 4.0)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def metrics(rows: list[dict], margin_fraction: float) -> dict:
    value = money(rows, margin_fraction)
    counts = collections.Counter(float(row["target_r"]) for row in rows)
    value.update({
        "target_counts": {str(target): counts[target] for target in TARGETS},
        "target_hit_rate": round(100 * sum(
            row["target_outcome"] == "TP2" for row in rows) / len(rows), 2)
            if rows else 0.0,
        "non_loss_rate": round(100 * sum(
            float(row["net_r"]) >= 0 for row in rows) / len(rows), 2)
            if rows else 0.0,
    })
    value["economic_gate"] = economic_gate(
        [float(row["signed_price_return"]) for row in rows],
        float(value["trades_per_calendar_day"]),
        margin_fraction=margin_fraction)
    return value


def historical_wide() -> pd.DataFrame:
    frames = {target: base.canonical_target(target) for target in TARGETS}
    common = None
    for frame in frames.values():
        keys = pd.MultiIndex.from_frame(frame[list(base.KEYS)])
        common = keys if common is None else common.intersection(keys, sort=False)
    if common is None or common.empty:
        raise RuntimeError("extended targets have no common historical rows")
    reference = frames[1.0].set_index(list(base.KEYS)).loc[common].reset_index()
    for target, frame in frames.items():
        suffix = str(target).replace(".", "_")
        aligned = frame.set_index(list(base.KEYS)).loc[common]
        reference[f"net_r_{suffix}"] = aligned.net_r.to_numpy(float)
        reference[f"outcome_{suffix}"] = aligned.outcome.astype(str).to_numpy()
        reference[f"exit_time_{suffix}"] = aligned.exit_time.to_numpy(float)
    return reference


def main() -> None:
    original_targets = base.TARGETS
    base.TARGETS = TARGETS
    try:
        historical = historical_wide()
        direct = refresh.direct_wide(
            refresh.NEW_NEAR, refresh.NEW_FAR, refresh.NEW_CACHE)
        direct["month"] = refresh.month(direct)
        historical_fit = historical[~historical.period.eq("2026")].copy()
        older_direct = direct[direct.month < "2026-09"].copy()
        train = pd.concat([historical_fit, older_direct], ignore_index=True)
        scoring = direct[direct.month.isin(
            ("2026-09", "2026-10"))].copy().reset_index(drop=True)
        routed = base.route(scoring, base.fit_models_on_rows(train, scoring))
        calibration = routed[routed.month.eq("2026-09")].copy()
        october = routed[routed.month.eq("2026-10")].copy()
        policy, frontier = base.choose_policy(calibration)
        october_value, october_rows = base.evaluate(
            october, float(policy["threshold"]), int(policy["loss_pause"]),
            int(policy["hourly_cap"]))
    finally:
        base.TARGETS = original_targets

    results = {
        "margin_3pct": metrics(october_rows, .03),
        "margin_5pct": metrics(october_rows, .05),
    }
    baseline = json.loads(BASELINE_REPORT.read_text(encoding="utf-8"))[
        "october_to_date"]
    improves = bool(
        results["margin_3pct"]["return_pct"]
        > baseline["margin_3pct"]["return_pct"]
        and results["margin_3pct"]["dd_pct"]
        <= baseline["margin_3pct"]["dd_pct"]
        and results["margin_3pct"]["trades_per_calendar_day"] >= 3)
    economic_pass = bool(results["margin_3pct"]["economic_gate"]["passes"])
    passed = bool(improves and economic_pass)
    pd.DataFrame(october_rows).to_pickle(ROWS_OUT)
    report = {
        "status": ("EXTENDED_TARGET_DIRECT_GATE_PASSED_NEEDS_WALKFORWARD"
                   if passed else "EXTENDED_TARGET_ROUTER_REJECTED"),
        "ready_for_live_orders": False,
        "order_eligible": False,
        "protocol": {
            "fit": "historical through 2025 plus X-Perp before September",
            "selection": "September 2026 only",
            "test": "October 2026 through Oct 24",
            "targets_r": list(TARGETS),
            "target_price_move_range_pct": [
                100 * base.MIN_TARGET_MOVE, 100 * base.MAX_TARGET_MOVE],
            "entry": "next 15-minute market open",
            "exit": "single stop-first OHLC path, fixed selected target",
            "orders": "market entry and market close",
            "retune_on_october": False,
        },
        "selected_before_october": {
            key: policy[key] for key in (
                "quantile", "threshold", "loss_pause", "hourly_cap",
                "metrics", "score")},
        "selection_frontier": frontier,
        "october_holdout": results,
        "baseline_three_target_october": baseline,
        "gates": {
            "improves_return_without_higher_drawdown": improves,
            "economic_30pct_at_3pct_margin": economic_pass,
            "passed": passed,
        },
        "sources": {
            str(BASELINE_REPORT): digest(BASELINE_REPORT),
            str(refresh.NEW_NEAR): digest(refresh.NEW_NEAR),
            str(refresh.NEW_FAR): digest(refresh.NEW_FAR),
            str(refresh.NEW_MANIFEST): digest(refresh.NEW_MANIFEST),
            str(Path(__file__).resolve()): digest(Path(__file__).resolve()),
        },
        "limitations": [
            "October is one partial out-of-sample month.",
            "A passing direct result would still require full historical walk-forward validation.",
            "Public candles cannot reproduce order-book slippage or latency.",
            "Gross results exclude fees at user request.",
        ],
    }
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({
        "status": report["status"],
        "selected_before_october": report["selected_before_october"],
        "october_holdout": results,
        "baseline_three_target_october": baseline,
        "gates": report["gates"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
