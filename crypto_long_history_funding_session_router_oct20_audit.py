"""Causal session and weekday router for long-history funding/style experts."""
from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path

import pandas as pd

import crypto_long_history_funding_style_router_oct20_audit as funding
import crypto_long_history_hourly_cross_section_oct20_audit as base


OUT = base.ROOT / "crypto_long_history_funding_session_router_oct20.json"
ROWS_OUT = base.ROOT / "crypto_long_history_funding_session_router_oct20.pkl"
FIELD_SETS = (
    ("variant", "direction", "session"),
    ("variant", "direction", "hour_block"),
    ("variant", "direction", "weekday"),
    ("variant", "direction", "session", "relative_crowding"),
    ("variant", "direction", "session", "funding_trend"),
    ("style", "lookback_hours", "target_pct", "direction",
     "session", "relative_crowding"),
)
MIN_COUNTS = (20, 50)
MIN_MEANS = (0.0, .00025)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def add_calendar(frame: pd.DataFrame) -> pd.DataFrame:
    stamp = pd.to_datetime(frame.entry_time, unit="s", utc=True)
    result = frame.copy()
    result["hour"] = stamp.dt.hour.astype(int)
    result["weekday"] = stamp.dt.dayofweek.astype(int)
    result["session"] = pd.cut(
        result.hour, [-1, 7, 15, 23],
        labels=["asia", "europe", "us"]).astype(str)
    result["hour_block"] = (result.hour // 3).astype(int)
    return result


def main() -> None:
    base_source = json.loads(base.OUT.read_text(encoding="utf-8"))
    candles, funding_data, sources = funding.load()
    library = funding.build_library(
        candles, base_source["development_diverse_library"])
    frame = add_calendar(funding.attach_funding(library, funding_data))
    candidates = []
    grid = itertools.product(
        FIELD_SETS, MIN_COUNTS, MIN_MEANS, MIN_COUNTS, MIN_MEANS,
        base.LOSS_PAUSES, base.HOURLY_CAPS)
    for (fields, dev_count, dev_mean, confirm_count, confirm_mean,
         loss_pause, hourly_cap) in grid:
        development = funding.stable_experts(
            frame, fields, base.DEVELOPMENT_YEARS, dev_count, dev_mean)
        experts = funding.confirmed_experts(
            frame, fields, development, confirm_count, confirm_mean)
        rows = funding.route(frame, fields, experts, loss_pause, hourly_cap)
        candidates.append({
            "fields": list(fields),
            "development_minimum_count_each_year": dev_count,
            "development_minimum_mean_each_year_pct": 100 * dev_mean,
            "confirmation_minimum_count_each_year": confirm_count,
            "confirmation_minimum_mean_each_year_pct": 100 * confirm_mean,
            "development_expert_count": len(development),
            "confirmed_expert_count": len(experts),
            "loss_pause": loss_pause, "hourly_cap": hourly_cap,
            "development_2020_2022_3pct": funding.stage(
                rows, base.DEVELOPMENT_YEARS, .03),
            "confirmation_2023_2024_3pct": funding.stage(
                rows, base.CONFIRMATION_YEARS, .03),
            "experts": experts.to_dict("records"), "_rows": rows,
        })
    candidates.sort(key=funding.rank, reverse=True)
    selected = candidates[0]
    hidden = {
        "margin_3pct": funding.stage(selected["_rows"], base.HIDDEN_YEARS, .03),
        "margin_5pct": funding.stage(selected["_rows"], base.HIDDEN_YEARS, .05),
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
        "status": ("FUNDING_SESSION_SHADOW_CANDIDATE"
                   if objective else "REJECTED_OBJECTIVE"),
        "ready_for_live_orders": False,
        "protocol": {
            "development_experts": "positive separately in 2020, 2021, 2022",
            "expert_confirmation": "positive separately in 2023 and 2024",
            "hidden_reveal": "2025-01 through available 2026 archive",
            "calendar": "UTC hour, three-hour block, weekday and 8h session",
            "entry": "next five-minute market open", "orders": "market only",
            "money": "client 3% or 5% current-equity margin; fixed 10x",
        },
        "library_size": len(base_source["development_diverse_library"]),
        "enriched_rows": len(frame), "candidate_count": len(candidates),
        "selected_before_hidden_reveal": funding.public(selected),
        "confirmation_frontier": [funding.public(item) for item in candidates[:20]],
        "hidden_2025_2026": hidden, "objective_pass": objective,
        "sources": {
            **sources, str(base.OUT): digest(base.OUT),
            str(funding.OUT): digest(funding.OUT),
            str(Path(__file__).resolve()): digest(Path(__file__).resolve()),
        },
        "limitations": [
            "Calendar buckets are UTC and do not reconstruct exchange microstructure.",
            "The finite calendar family was selected before the hidden reveal.",
            "The 2026 archive ends in September.",
            "A survivor still requires direct X-Perp forward shadow evidence.",
            "Gross returns exclude fees at user request.",
            "No backtest guarantees a future monthly return.",
        ],
    }
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({
        "status": report["status"], "enriched_rows": len(frame),
        "candidate_count": len(candidates),
        "selected": funding.public(selected), "hidden": hidden,
        "objective_pass": objective,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
