"""Five-year stability confirmation for causal volume-regime experts."""
from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path

import pandas as pd

import crypto_long_history_hourly_cross_section_oct20_audit as base
import crypto_long_history_monthly_style_router_oct20_audit as monthly
import crypto_long_history_volume_regime_router_oct20_audit as volume
import crypto_raw_hourly_direction_oct19_audit as portfolio


OUT = base.ROOT / "crypto_long_history_five_year_regime_router_oct20.json"
ROWS_OUT = base.ROOT / "crypto_long_history_five_year_regime_router_oct20.pkl"
MIN_CONFIRM_COUNTS = (10, 25)
MIN_CONFIRM_MEANS = (0.0, .00025)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def confirmed_experts(frame: pd.DataFrame, fields: tuple[str, ...],
                      development_experts: pd.DataFrame,
                      min_count: int, min_mean: float) -> pd.DataFrame:
    """Keep development experts positive in 2023 and 2024 separately."""
    if development_experts.empty:
        return development_experts.copy()
    eligible = frame.merge(
        development_experts[list(fields)], on=list(fields),
        how="inner", validate="many_to_many")
    confirmation = eligible[
        eligible.year.isin(base.CONFIRMATION_YEARS)]
    stats = confirmation.groupby(
        ["year", *fields], dropna=False).signed_price_return.agg(
            count="size", mean_return="mean").reset_index()
    good = stats[
        stats["count"].ge(min_count)
        & stats.mean_return.ge(min_mean)]
    if good.empty:
        return development_experts.iloc[:0].copy()
    stable = good.groupby(list(fields), dropna=False).filter(
        lambda part: set(part.year.astype(int))
        == set(base.CONFIRMATION_YEARS))
    if stable.empty:
        return development_experts.iloc[:0].copy()
    keys = stable[list(fields)].drop_duplicates()
    confirmed = development_experts.merge(
        keys, on=list(fields), how="inner", validate="one_to_one")
    summary = stable.groupby(list(fields), dropna=False).agg(
        confirmation_worst_mean=("mean_return", "min"),
        confirmation_observations=("count", "sum")).reset_index()
    return confirmed.merge(
        summary, on=list(fields), how="inner", validate="one_to_one")


def rank(item: dict) -> tuple:
    value = item["confirmation_2023_2024_3pct"]
    return (
        value["dd_pct"] <= 15 and value["trades_per_calendar_day"] >= 3,
        -int(value["negative_months"]), base.month_floor(value),
        float(value["return_pct"]), -float(value["dd_pct"]),
        int(item["confirmed_expert_count"]),
    )


def public(item: dict) -> dict:
    return {key: value for key, value in item.items()
            if not key.startswith("_")}


def main() -> None:
    source = json.loads(volume.OUT.read_text(encoding="utf-8"))
    base_source = json.loads(base.OUT.read_text(encoding="utf-8"))
    items = base_source["development_diverse_library"]
    candles, candle_sources = base.load()
    library = monthly.build_library(candles, items)
    context, thresholds = volume.market_context(candles)
    frame = volume.enrich(library, items, context)
    candidates = []
    for family, min_count, min_mean, loss_pause, hourly_cap in itertools.product(
            source["development_frontier"], MIN_CONFIRM_COUNTS,
            MIN_CONFIRM_MEANS, base.LOSS_PAUSES, base.HOURLY_CAPS):
        fields = tuple(family["fields"])
        development_experts = pd.DataFrame(family["experts"])
        experts = confirmed_experts(
            frame, fields, development_experts, min_count, min_mean)
        rows = volume.route(
            frame, fields, experts, loss_pause, hourly_cap)
        candidates.append({
            "fields": list(fields),
            "development_minimum_count_each_year": family["minimum_count_each_year"],
            "development_minimum_mean_each_year_pct": family["minimum_mean_each_year_pct"],
            "confirmation_minimum_count_each_year": min_count,
            "confirmation_minimum_mean_each_year_pct": 100 * min_mean,
            "development_expert_count": len(development_experts),
            "confirmed_expert_count": len(experts),
            "loss_pause": loss_pause, "hourly_cap": hourly_cap,
            "development_2020_2022_3pct": family["development_2020_2022_3pct"],
            "confirmation_2023_2024_3pct": volume.stage(
                rows, base.CONFIRMATION_YEARS, .03),
            "experts": experts.to_dict("records"), "_rows": rows,
        })
    candidates.sort(key=rank, reverse=True)
    selected = candidates[0]
    hidden = {
        "margin_3pct": volume.stage(
            selected["_rows"], base.HIDDEN_YEARS, .03),
        "margin_5pct": volume.stage(
            selected["_rows"], base.HIDDEN_YEARS, .05),
    }
    hidden_rows = [row for row in selected["_rows"] if pd.Timestamp(
        float(row["entry_time"]), unit="s", tz="UTC").year
        in base.HIDDEN_YEARS]
    pd.DataFrame(hidden_rows).to_pickle(ROWS_OUT)
    three = hidden["margin_3pct"]
    objective = bool(
        three["negative_months"] == 0 and base.month_floor(three) >= 30
        and three["dd_pct"] <= 10
        and three["trades_per_calendar_day"] >= 3)
    report = {
        "status": ("FIVE_YEAR_REGIME_SHADOW_CANDIDATE"
                   if objective else "REJECTED_OBJECTIVE"),
        "ready_for_live_orders": False,
        "protocol": {
            "development_experts": "positive separately in 2020, 2021, 2022",
            "expert_confirmation": "positive separately in 2023 and 2024",
            "hidden_reveal": "2025-01 through available 2026 archive",
            "orders": "next-bar market entry only",
            "money": "client 3% or 5% current-equity margin; fixed 10x",
        },
        "context_thresholds_selected_on_2020_2022": thresholds,
        "library_size": len(library), "candidate_count": len(candidates),
        "selected_before_hidden_reveal": public(selected),
        "confirmation_frontier": [public(item) for item in candidates[:20]],
        "hidden_2025_2026": hidden, "objective_pass": objective,
        "sources": {
            **candle_sources, str(base.OUT): digest(base.OUT),
            str(volume.OUT): digest(volume.OUT),
            str(Path(__file__).resolve()): digest(Path(__file__).resolve()),
        },
        "limitations": [
            "Five-year expert filtering is conservative but still selected before one hidden reveal.",
            "Spot volume is venue-specific and requires X-Perp transfer validation.",
            "The 2026 archive is incomplete.",
            "Gross returns exclude fees at user request.",
            "No backtest guarantees a future monthly return.",
        ],
    }
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True),
                   encoding="utf-8")
    print(json.dumps({
        "status": report["status"], "candidate_count": len(candidates),
        "selected": public(selected), "hidden": hidden,
        "objective_pass": objective,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
