"""Audit directional crowding in the native X-Perp walk-forward.

The base model lost mainly on crowded longs in June and crowded shorts in
August.  This audit tests one causal mechanism: suppress a candidate when its
direction already dominates the preceding raw candidate stream.  June and
July select one fixed policy, August validates it, and later months are only
reported as confirmation diagnostics.
"""
from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path

import pandas as pd

import crypto_xperp_native_expanding_oct20_audit as parent
from crypto_runtime_shadow_portfolio_oct3_audit import money


OUT = parent.parent.ROOT / "crypto_xperp_direction_crowding_oct20.json"
DEV_MONTHS = ("2026-06", "2026-07")
VALIDATION_MONTH = "2026-08"
CONFIRMATION_MONTHS = ("2026-09", "2026-10")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def direction_crowding_filter(
        rows: list[dict], hours: int, max_share: float,
        min_candidates: int) -> list[dict]:
    """Use only earlier raw candidates to measure directional crowding."""
    accepted: list[dict] = []
    recent: list[dict] = []
    for row in sorted(rows, key=lambda item: int(item["entry_time"])):
        now = int(row["entry_time"])
        cutoff = now - hours * 3600
        recent = [old for old in recent if int(old["entry_time"]) >= cutoff]
        same = sum(old["direction"] == row["direction"] for old in recent)
        crowded = (len(recent) >= min_candidates
                   and same / len(recent) >= max_share)
        if not crowded:
            accepted.append(row)
        # Rejected signals remain part of the raw candidate stream.  This
        # prevents the filter from manufacturing balance through its own
        # previous rejections.
        recent.append(row)
    return accepted


def monthly_attribution(frame: pd.DataFrame, keys: list[str]) -> list[dict]:
    data = frame.copy()
    data["profitable"] = data["signed_price_return"] > 0
    grouped = data.groupby(["month", *keys], dropna=False).agg(
        n=("profitable", "size"),
        win_rate=("profitable", "mean"),
        signed_price_return=("signed_price_return", "sum"),
    ).reset_index()
    grouped["win_rate_pct"] = (100 * grouped.pop("win_rate")).round(2)
    grouped["signed_price_return_pct"] = (
        100 * grouped.pop("signed_price_return")).round(4)
    return grouped.to_dict("records")


def main() -> None:
    if not parent.ROWS_OUT.exists():
        parent.main()
    frame = pd.read_pickle(parent.ROWS_OUT).sort_values("entry_time")
    rows = frame.to_dict("records")

    frontier = []
    for hours, max_share, minimum in itertools.product(
            (6, 12, 24, 48, 72), (.60, .65, .70, .75, .80), (3, 5, 8, 12)):
        filtered = direction_crowding_filter(rows, hours, max_share, minimum)
        metrics = money(filtered, .03)
        monthly = metrics.get("monthly_return_pct", {})
        development = [float(monthly.get(month, -999.0))
                       for month in DEV_MONTHS]
        frontier.append({
            "hours": hours,
            "max_share": max_share,
            "min_candidates": minimum,
            "development_worst_month_pct": min(development),
            "development_mean_month_pct": sum(development) / len(development),
            "metrics_3pct": metrics,
        })

    selected = max(frontier, key=lambda item: (
        item["development_worst_month_pct"],
        item["development_mean_month_pct"],
        -item["metrics_3pct"]["dd_pct"],
    ))
    selected_rows = direction_crowding_filter(
        rows, int(selected["hours"]), float(selected["max_share"]),
        int(selected["min_candidates"]))
    selected_metrics = {
        "margin_3pct": money(selected_rows, .03),
        "margin_5pct": money(selected_rows, .05),
    }
    validation_return = selected_metrics["margin_3pct"][
        "monthly_return_pct"].get(VALIDATION_MONTH, -999.0)
    objective = bool(
        validation_return >= 30
        and selected_metrics["margin_3pct"]["dd_pct"] <= 10
        and selected_metrics["margin_3pct"]["trades_per_calendar_day"] >= 3)

    frame["hour"] = pd.to_datetime(
        frame.entry_time, unit="s", utc=True).dt.hour
    report = {
        "status": "REJECTED_OBJECTIVE" if not objective else "SHADOW_CANDIDATE",
        "ready_for_live_orders": False,
        "hypothesis": (
            "losses concentrate when one model direction dominates the recent "
            "raw candidate stream"),
        "protocol": {
            "development_months": list(DEV_MONTHS),
            "validation_month": VALIDATION_MONTH,
            "confirmation_months_already_seen_elsewhere": list(CONFIRMATION_MONTHS),
            "candidate_grid_count": len(frontier),
            "selection": "maximize worst development month, then mean, then DD",
            "causality": "only candidates with earlier entry_time enter the window",
            "orders": "market only",
            "money": "client 3% or 5% current-equity margin; fixed 10x",
        },
        "baseline": {
            "margin_3pct": money(rows, .03),
            "margin_5pct": money(rows, .05),
        },
        "attribution": {
            "by_symbol": monthly_attribution(frame, ["symbol"]),
            "by_direction": monthly_attribution(frame, ["direction"]),
            "by_hour": monthly_attribution(frame, ["hour"]),
        },
        "selected_policy": {
            key: selected[key] for key in (
                "hours", "max_share", "min_candidates",
                "development_worst_month_pct", "development_mean_month_pct")
        },
        "selected_metrics": selected_metrics,
        "validation_return_3pct": validation_return,
        "objective_pass": objective,
        "frontier": frontier,
        "sources": {
            str(parent.OUT): digest(parent.OUT),
            str(Path(__file__).resolve()): digest(Path(__file__).resolve()),
        },
        "limitations": [
            "The direct venue history begins in partial April 2026.",
            "September and October were already observed by earlier audits.",
            "Gross returns exclude fees at user request.",
            "No backtest guarantees a future monthly return.",
        ],
    }
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({
        "status": report["status"],
        "selected_policy": report["selected_policy"],
        "selected_metrics": selected_metrics,
        "validation_return_3pct": validation_return,
        "objective_pass": objective,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
