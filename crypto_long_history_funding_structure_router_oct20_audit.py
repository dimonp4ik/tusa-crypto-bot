"""Funding/style experts conditioned on causal support and liquidity context."""
from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

import crypto_long_history_funding_style_router_oct20_audit as funding
import crypto_long_history_hourly_cross_section_oct20_audit as base


OUT = base.ROOT / "crypto_long_history_funding_structure_router_oct20.json"
ROWS_OUT = base.ROOT / "crypto_long_history_funding_structure_router_oct20.pkl"
FIELD_SETS = (
    ("variant", "direction", "relative_crowding", "room_state"),
    ("variant", "direction", "relative_crowding", "backing_state"),
    ("variant", "direction", "relative_crowding", "stretch_state"),
    ("variant", "direction", "relative_crowding", "sweep_state"),
    ("variant", "direction", "room_state", "backing_state"),
    ("style", "lookback_hours", "target_pct", "direction",
     "relative_crowding", "room_state", "backing_state"),
)
MIN_COUNTS = (20, 50)
MIN_MEANS = (0.0, .00025)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _rolling(values: np.ndarray, window: int, operation: str) -> np.ndarray:
    series = pd.Series(values)
    if operation == "max":
        return series.rolling(window).max().shift(1).to_numpy(float)
    if operation == "min":
        return series.rolling(window).min().shift(1).to_numpy(float)
    raise ValueError(operation)


def add_structure(frame: pd.DataFrame,
                  candles: dict[str, dict]) -> pd.DataFrame:
    """Add levels and sweep state from bars fully closed before entry."""
    pieces = []
    for symbol, part in frame.groupby("symbol", sort=False):
        raw = candles[symbol]
        open_ = np.asarray(raw["open"], dtype=float)
        high = np.asarray(raw["high"], dtype=float)
        low = np.asarray(raw["low"], dtype=float)
        close = np.asarray(raw["close"], dtype=float)
        previous = np.r_[np.nan, close[:-1]]
        tr = np.maximum(high-low, np.maximum(np.abs(high-previous),
                                             np.abs(low-previous)))
        atr = pd.Series(tr).rolling(14).mean().to_numpy(float)
        high288, low288 = _rolling(high, 288, "max"), _rolling(low, 288, "min")
        high24, low24 = _rolling(high, 24, "max"), _rolling(low, 24, "min")
        ret4 = (pd.Series(close) / pd.Series(close).shift(48) - 1).to_numpy(float)
        indices = part.entry_index.to_numpy(np.int64)
        signal = indices - 1
        entry = open_[indices]
        direction = part.direction_n.to_numpy(float)
        target = part.target_pct.to_numpy(float) / 100
        stop = part.stop_pct.to_numpy(float) / 100
        room = np.where(
            direction > 0, (high288[indices] - entry) / entry,
            (entry - low288[indices]) / entry)
        backing = np.where(
            direction > 0, entry / low288[indices] - 1,
            high288[indices] / entry - 1)
        directed_stretch = direction * ret4[signal]
        long_sweep = (low[signal] < low24[signal]) & (close[signal] > low24[signal])
        short_sweep = (high[signal] > high24[signal]) & (close[signal] < high24[signal])
        supportive_sweep = np.where(direction > 0, long_sweep, short_sweep)
        found = part.copy()
        found["room_to_24h_level_pct"] = room
        found["backing_24h_distance_pct"] = backing
        found["atr_pct"] = atr[signal] / close[signal]
        found["directed_stretch_4h"] = directed_stretch
        found["supportive_sweep"] = supportive_sweep
        found["room_state"] = np.where(room >= target, "clear", "blocked")
        found["backing_state"] = np.select(
            [backing < 0, backing <= stop],
            ["crossed", "inside_stop"], default="beyond_stop")
        found["stretch_state"] = np.select(
            [directed_stretch < -.002, directed_stretch > .002],
            ["adverse", "extended"], default="flat")
        found["sweep_state"] = np.where(
            supportive_sweep, "supportive_sweep", "no_sweep")
        pieces.append(found)
    result = pd.concat(pieces, ignore_index=True)
    return result.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["room_to_24h_level_pct", "backing_24h_distance_pct",
                "atr_pct", "directed_stretch_4h"])


def rank(item: dict) -> tuple:
    return funding.rank(item)


def public(item: dict) -> dict:
    return funding.public(item)


def main() -> None:
    base_source = json.loads(base.OUT.read_text(encoding="utf-8"))
    candles, funding_data, sources = funding.load()
    library = funding.build_library(
        candles, base_source["development_diverse_library"])
    frame = add_structure(funding.attach_funding(library, funding_data), candles)
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
    candidates.sort(key=rank, reverse=True)
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
        "status": ("FUNDING_STRUCTURE_SHADOW_CANDIDATE"
                   if objective else "REJECTED_OBJECTIVE"),
        "ready_for_live_orders": False,
        "protocol": {
            "development_experts": "positive separately in 2020, 2021, 2022",
            "expert_confirmation": "positive separately in 2023 and 2024",
            "hidden_reveal": "2025-01 through available 2026 archive",
            "structure_cutoff": "high, low, ATR and sweep use bars before entry",
            "levels": "prior 24h high/low; room compared with literal target",
            "backing": "prior 24h opposite level compared with literal stop",
            "entry": "next five-minute market open", "orders": "market only",
            "money": "client 3% or 5% current-equity margin; fixed 10x",
        },
        "library_size": len(base_source["development_diverse_library"]),
        "enriched_rows": len(frame), "candidate_count": len(candidates),
        "selected_before_hidden_reveal": public(selected),
        "confirmation_frontier": [public(item) for item in candidates[:20]],
        "hidden_2025_2026": hidden, "objective_pass": objective,
        "sources": {
            **sources, str(base.OUT): digest(base.OUT),
            str(funding.OUT): digest(funding.OUT),
            str(Path(__file__).resolve()): digest(Path(__file__).resolve()),
        },
        "limitations": [
            "OHLCV levels and sweeps are liquidity proxies, not order-book history.",
            "The finite structure family was selected before the hidden reveal.",
            "The 2026 archive ends in September.",
            "A survivor still requires direct X-Perp forward shadow evidence.",
            "Gross returns exclude fees at user request.",
            "No backtest guarantees a future monthly return.",
        ],
    }
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({
        "status": report["status"], "enriched_rows": len(frame),
        "candidate_count": len(candidates), "selected": public(selected),
        "hidden": hidden, "objective_pass": objective,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
