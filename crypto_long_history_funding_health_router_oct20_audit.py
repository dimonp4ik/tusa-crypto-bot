"""Monthly health router for development-stable funding/style experts.

At each month boundary the router scores only trades that have already exited.
The health policy is selected on 2023-2024 and then frozen for 2025-2026.
"""
from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

import crypto_long_history_funding_style_router_oct20_audit as funding
import crypto_long_history_hourly_cross_section_oct20_audit as base
import crypto_raw_hourly_direction_oct19_audit as portfolio


OUT = base.ROOT / "crypto_long_history_funding_health_router_oct20.json"
ROWS_OUT = base.ROOT / "crypto_long_history_funding_health_router_oct20.pkl"
SOURCE_REPORT = funding.OUT
LOOKBACK_MONTHS = (1, 3, 6, 12)
TOP_K = (1, 2, 3)
MIN_COUNTS = (20, 50)
MIN_MEANS = (0.0, .00025)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def expert_id(frame: pd.DataFrame, fields: tuple[str, ...]) -> pd.Series:
    return frame.loc[:, fields].astype(str).agg("|".join, axis=1)


def eligible_frame(frame: pd.DataFrame, fields: tuple[str, ...],
                   experts: pd.DataFrame) -> pd.DataFrame:
    if experts.empty:
        return frame.iloc[:0].copy()
    selected = frame.merge(
        experts[[*fields, "worst_year_mean", "mean_return"]],
        on=list(fields), how="inner", validate="many_to_one")
    selected["expert_id"] = expert_id(selected, fields)
    return selected


def trailing_scores(frame: pd.DataFrame, cutoff: pd.Timestamp,
                    lookback_months: int, min_count: int,
                    min_mean: float) -> pd.DataFrame:
    """Use outcomes known strictly before the month-start decision."""
    left = (cutoff - pd.DateOffset(months=lookback_months)).timestamp()
    right = cutoff.timestamp()
    history = frame[
        frame.exit_time.ge(left) & frame.exit_time.lt(right)]
    if history.empty:
        return pd.DataFrame(columns=["expert_id", "count", "mean_return"])
    scores = history.groupby("expert_id").signed_price_return.agg(
        count="size", mean_return="mean").reset_index()
    return scores[
        scores["count"].ge(min_count)
        & scores.mean_return.ge(min_mean)].sort_values(
            ["mean_return", "count", "expert_id"],
            ascending=[False, False, True], kind="mergesort")


def monthly_candidates(frame: pd.DataFrame, start: str, end: str, *,
                       lookback_months: int, top_k: int,
                       min_count: int, min_mean: float
                       ) -> tuple[pd.DataFrame, list[dict]]:
    pieces, decisions = [], []
    for month in pd.period_range(start, end, freq="M"):
        cutoff = month.start_time.tz_localize("UTC")
        finish = (month + 1).start_time.tz_localize("UTC")
        scores = trailing_scores(
            frame, cutoff, lookback_months, min_count, min_mean).head(top_k)
        chosen = dict(zip(scores.expert_id, scores.mean_return))
        decisions.append({
            "month": str(month),
            "selected": [
                {"expert_id": str(row.expert_id),
                 "completed_trades": int(row.count),
                 "mean_price_return_pct": 100 * float(row.mean_return)}
                for row in scores.itertuples(index=False)],
        })
        if not chosen:
            continue
        part = frame[
            frame.entry_time.ge(cutoff.timestamp())
            & frame.entry_time.lt(finish.timestamp())
            & frame.expert_id.isin(chosen)].copy()
        part["health_score"] = part.expert_id.map(chosen)
        pieces.append(part)
    if not pieces:
        return frame.iloc[:0].copy(), decisions
    return pd.concat(pieces, ignore_index=True), decisions


def route(candidates: pd.DataFrame, loss_pause: int,
          hourly_cap: int) -> list[dict]:
    if candidates.empty:
        return []
    selected = candidates.sort_values(
        ["entry_time", "health_score", "worst_year_mean", "symbol", "variant"],
        ascending=[True, False, False, True, True], kind="mergesort")
    selected = selected.drop_duplicates(["symbol", "entry_time"], keep="first")
    selected["model_quality"] = selected.health_score
    rows = portfolio.candidate_rows(selected, float("-inf"))
    for row in rows:
        row["module"] = f"funding_health:{row['variant']}"
        row["source"] = "long_history_funding_health"
    return portfolio.rails.ranked_runtime_gate(
        rows, loss_pause=loss_pause, hourly_cap=hourly_cap)


def rank(value: dict) -> tuple:
    return (
        value["dd_pct"] <= 15 and value["trades_per_calendar_day"] >= 3,
        -int(value["negative_months"]), base.month_floor(value),
        float(value["return_pct"]), -float(value["dd_pct"]),
    )


def public(item: dict) -> dict:
    return {key: value for key, value in item.items()
            if not key.startswith("_")}


def main() -> None:
    source = json.loads(SOURCE_REPORT.read_text(encoding="utf-8"))
    source_selected = source["selected_before_hidden_reveal"]
    fields = tuple(source_selected["fields"])
    candles, funding_data, sources = funding.load()
    base_source = json.loads(base.OUT.read_text(encoding="utf-8"))
    library = funding.build_library(
        candles, base_source["development_diverse_library"])
    frame = funding.attach_funding(library, funding_data)
    development = funding.stable_experts(
        frame, fields, base.DEVELOPMENT_YEARS,
        int(source_selected["development_minimum_count_each_year"]),
        float(source_selected["development_minimum_mean_each_year_pct"]) / 100)
    eligible = eligible_frame(frame, fields, development)
    policies = []
    policy_grid = itertools.product(
        LOOKBACK_MONTHS, TOP_K, MIN_COUNTS, MIN_MEANS)
    for lookback, top_k, min_count, min_mean in policy_grid:
        candidates, decisions = monthly_candidates(
            eligible, "2023-01", "2024-12",
            lookback_months=lookback, top_k=top_k,
            min_count=min_count, min_mean=min_mean)
        for loss_pause, hourly_cap in itertools.product(
                base.LOSS_PAUSES, base.HOURLY_CAPS):
            rows = route(candidates, loss_pause, hourly_cap)
            policies.append({
                "lookback_months": lookback, "top_k": top_k,
                "minimum_completed_trades": min_count,
                "minimum_mean_price_return_pct": 100 * min_mean,
                "loss_pause": loss_pause, "hourly_cap": hourly_cap,
                "confirmation_2023_2024_3pct": portfolio.money(rows, .03),
                "_decisions": decisions,
            })
    policies.sort(
        key=lambda item: rank(item["confirmation_2023_2024_3pct"]),
        reverse=True)
    selected = policies[0]
    hidden_candidates, hidden_decisions = monthly_candidates(
        eligible, "2025-01", "2026-09",
        lookback_months=int(selected["lookback_months"]),
        top_k=int(selected["top_k"]),
        min_count=int(selected["minimum_completed_trades"]),
        min_mean=float(selected["minimum_mean_price_return_pct"]) / 100)
    rows = route(
        hidden_candidates, int(selected["loss_pause"]),
        int(selected["hourly_cap"]))
    hidden = {
        "margin_3pct": portfolio.money(rows, .03),
        "margin_5pct": portfolio.money(rows, .05),
    }
    pd.DataFrame(rows).to_pickle(ROWS_OUT)
    three = hidden["margin_3pct"]
    objective = bool(
        three["negative_months"] == 0 and base.month_floor(three) >= 30
        and three["dd_pct"] <= 10
        and three["trades_per_calendar_day"] >= 3)
    report = {
        "status": ("FUNDING_HEALTH_SHADOW_CANDIDATE"
                   if objective else "REJECTED_OBJECTIVE"),
        "ready_for_live_orders": False,
        "protocol": {
            "expert_development": "funding/style buckets positive in each of 2020-2022",
            "policy_selection": "2023-2024 only",
            "health_decision": "month start using only exits strictly before cutoff",
            "hidden_reveal": "2025-01 through available 2026 archive",
            "orders": "next-bar market entry only",
            "money": "client 3% or 5% current-equity margin; fixed 10x",
        },
        "fields": list(fields), "development_expert_count": len(development),
        "eligible_rows": len(eligible), "policies_evaluated": len(policies),
        "selected_before_hidden_reveal": public(selected),
        "policy_frontier": [public(item) for item in policies[:20]],
        "selection_decisions": selected["_decisions"],
        "hidden_decisions": hidden_decisions,
        "hidden_2025_2026": hidden, "objective_pass": objective,
        "sources": {
            **sources, str(base.OUT): digest(base.OUT),
            str(SOURCE_REPORT): digest(SOURCE_REPORT),
            str(Path(__file__).resolve()): digest(Path(__file__).resolve()),
        },
        "limitations": [
            "The funding field family was selected on 2023-2024 before hidden reveal.",
            "Monthly health can react only after completed losses become observable.",
            "The 2026 archive ends in September.",
            "A survivor still requires direct X-Perp forward shadow evidence.",
            "Gross returns exclude fees at user request.",
            "No backtest guarantees a future monthly return.",
        ],
    }
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({
        "status": report["status"], "development_experts": len(development),
        "eligible_rows": len(eligible), "policies": len(policies),
        "selected": public(selected), "hidden": hidden,
        "objective_pass": objective,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
