"""Sequential holdout audit for the balanced exact-exit candidate.

The candidate universe and every development metric come from the October 20
exact-path search.  This audit chooses one candidate using 2022-2023 only,
rebuilds it from raw OHLC, and reveals 2024 first.  Later periods and direct
X-Perp are evaluated only when the preceding gate passes.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

import crypto_2r_precision_barbell_oct5_audit as precision
import crypto_exact_exit_geometry_oct20_audit as geometry
import crypto_high_target_tree_oct5_audit as base
import crypto_precision_be_runner_oct18_audit as exact
import crypto_precision_profit_first_oct18_audit as prior


ROOT = Path("reports/audit_2026_09_08")
SOURCE_REPORT = ROOT / "crypto_exact_exit_geometry_oct20.json"
OUT = ROOT / "crypto_exact_exit_balanced_holdout_oct22.json"
ROWS_OUT = ROOT / "crypto_exact_exit_balanced_holdout_oct22_rows.pkl"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def development_gate(candidate: dict) -> bool:
    """Require useful profit and capacity in both pre-holdout periods."""
    selection = candidate["selection"]
    confirmation = candidate["confirmation"]
    return bool(
        candidate["admissible"]
        and selection["n"] >= 500
        and confirmation["n"] >= 500
        and selection["trades_per_calendar_day"] >= 3
        and confirmation["trades_per_calendar_day"] >= 3
        and selection["return_pct"] >= 20
        and confirmation["return_pct"] >= 20
        and selection["negative_months"] <= 2
        and confirmation["negative_months"] <= 2
        and max(selection["dd_pct"], confirmation["dd_pct"]) <= 8
    )


def development_rank(candidate: dict) -> tuple[float, float, float]:
    selection = candidate["selection"]
    confirmation = candidate["confirmation"]
    return (
        min(selection["return_pct"], confirmation["return_pct"]),
        selection["return_pct"] + confirmation["return_pct"],
        -max(selection["dd_pct"], confirmation["dd_pct"]),
    )


def holdout_gate(value: dict, *, maximum_negative_months: int) -> bool:
    return bool(
        value["n"] >= 100
        and value["trades_per_calendar_day"] >= 3
        and value["return_pct"] > 0
        and value["dd_pct"] <= 10
        and value["negative_months"] <= maximum_negative_months
    )


def public_candidate(candidate: dict) -> dict:
    return {
        key: candidate[key]
        for key in (
            "target_r", "activation_r", "quantile", "threshold",
            "loss_pause", "hourly_cap", "selection", "confirmation",
            "development_floor_pct", "admissible",
        )
    }


def main() -> None:
    source = json.loads(SOURCE_REPORT.read_text(encoding="utf-8"))
    universe = source["top20_before_holdouts"]
    eligible = [candidate for candidate in universe if development_gate(candidate)]
    if not eligible:
        raise RuntimeError("balanced development gate produced no candidate")
    selected = max(eligible, key=development_rank)

    historical = prior.canonical_history(precision.load_joined())
    target_r = float(selected["target_r"])
    activation_r = (99.0 if selected["activation_r"] is None
                    else float(selected["activation_r"]))
    historical_target = geometry.eligible(historical, target_r)
    replayed = exact.attach_runtime_runner(
        historical_target, direct=False,
        activation_r=activation_r, runner_r=target_r)
    fit = base.stage(replayed, precision.FIT)
    design = base.design(fit)
    model = geometry.fit_model(fit, design)
    scored = geometry.score(replayed, model, design)

    threshold = float(selected["threshold"])
    loss_pause = int(selected["loss_pause"])
    hourly_cap = int(selected["hourly_cap"])
    rebuilt = {}
    development_rows: list[dict] = []
    for name, periods in (("selection", precision.SELECT),
                          ("confirmation", precision.CONFIRM)):
        rebuilt[name], rows = geometry.evaluate(
            scored, periods, threshold, loss_pause, hourly_cap, .03)
        for row in rows:
            row["audit_cohort"] = name
        development_rows.extend(rows)
        expected = selected[name]
        for field in ("n", "return_pct", "dd_pct", "negative_months",
                      "trades_per_calendar_day"):
            if rebuilt[name][field] != expected[field]:
                raise RuntimeError(
                    f"{name} reconstruction mismatch for {field}: "
                    f"{rebuilt[name][field]} != {expected[field]}")

    validation, validation_rows = geometry.evaluate(
        scored, precision.VALIDATION, threshold,
        loss_pause, hourly_cap, .03)
    validation_passed = holdout_gate(validation, maximum_negative_months=1)

    later = None
    later_rows: list[dict] = []
    later_passed = False
    direct = None
    direct_rows: list[dict] = []
    direct_passed = False
    if validation_passed:
        later, later_rows = geometry.evaluate(
            scored, precision.HIDDEN, threshold,
            loss_pause, hourly_cap, .03)
        later_passed = holdout_gate(later, maximum_negative_months=3)
    if later_passed:
        direct_frame = geometry.eligible(prior.direct_frame(), target_r)
        direct_replayed = exact.attach_runtime_runner(
            direct_frame, direct=True,
            activation_r=activation_r, runner_r=target_r)
        direct_scored = geometry.score(direct_replayed, model, design)
        direct, direct_rows = geometry.evaluate(
            direct_scored, ("xperp",), threshold,
            loss_pause, hourly_cap, .03)
        direct_passed = holdout_gate(direct, maximum_negative_months=1)

    for row in validation_rows:
        row["audit_cohort"] = "validation_2024"
    for row in later_rows:
        row["audit_cohort"] = "later_2025_2026"
    for row in direct_rows:
        row["audit_cohort"] = "direct_xperp"
    all_rows = development_rows + validation_rows + later_rows + direct_rows
    pd.DataFrame(all_rows).to_pickle(ROWS_OUT)
    report = {
        "status": ("AWAITING_NEW_FORWARD" if direct_passed
                   else "REJECTED_SEQUENTIAL_HOLDOUT"),
        "ready_for_live_orders": False,
        "order_eligible": False,
        "selection_protocol": {
            "candidate_universe": "October 20 top 20 ranked before holdouts",
            "development_only": ["2022h1", "2022h2", "2023"],
            "gate": {
                "minimum_trades_each_stage": 500,
                "minimum_trades_per_calendar_day_each_stage": 3,
                "minimum_return_pct_each_stage_at_3pct_margin": 20,
                "maximum_negative_months_each_stage": 2,
                "maximum_drawdown_pct_each_stage": 8,
            },
            "rank": "maximum of the lower development-stage return",
            "sequential_reveal": ["2024", "2025-2026", "direct X-Perp"],
            "holdout_gate": (
                "positive total return, at least 3 trades/calendar-day, "
                "drawdown <=10%, bounded negative months"),
        },
        "eligible_development_candidates": len(eligible),
        "selected_before_holdouts": public_candidate(selected),
        "rebuilt_development": rebuilt,
        "validation_2024": validation,
        "validation_2024_passed": validation_passed,
        "later_2025_2026": later,
        "later_2025_2026_passed": later_passed,
        "direct_xperp": direct,
        "direct_xperp_passed": direct_passed,
        "objective_checks": {
            "sequential_holdouts_passed": direct_passed,
            "every_observed_month_at_least_30pct": bool(
                direct_passed and all(
                    value >= 30 for value in direct["monthly_return_pct"].values())),
            "ready_for_live_orders": False,
        },
        "sources": {
            str(SOURCE_REPORT): digest(SOURCE_REPORT),
            str(Path(__file__).resolve()): digest(Path(__file__).resolve()),
            **exact.CANDLE_SOURCES,
        },
        "limitations": [
            "The balanced selection policy was introduced after the wider October 20 search, so 2024 is the first honest test for this policy.",
            "The sequential protocol intentionally leaves later evidence unread when an earlier gate fails.",
            "Gross results exclude fees at user request.",
            "A backtest cannot guarantee future monthly return.",
        ],
    }
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({
        "status": report["status"],
        "eligible_development_candidates": len(eligible),
        "selected": public_candidate(selected),
        "validation_2024": validation,
        "validation_2024_passed": validation_passed,
        "later_2025_2026": later,
        "direct_xperp": direct,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
