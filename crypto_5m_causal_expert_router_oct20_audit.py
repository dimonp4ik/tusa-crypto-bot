"""Causal monthly router across crypto entry families and exit geometries.

The router learns only from complete prior calendar months.  For every
market-state bucket it selects the exit geometry with the best recent mean
signed price return, then trades matching events during the next month.
Development is 2021-2023, 2024 confirms the policy, and 2025-2026 are hidden.
"""
from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

import crypto_5m_tree_filter_audit as events_base
from crypto_runtime_shadow_portfolio_oct3_audit import money, runtime_gate


OUT = Path("reports/audit_2026_09_08/crypto_5m_causal_expert_router_oct20.json")
GEOMETRIES = {
    "s1_t1": (1.0, 1.0),
    "s1.5_t1": (1.5, 1.0),
    "s2_t1": (2.0, 1.0),
}
FIELD_SETS = (
    ("family", "direction_name", "session"),
    ("family", "direction_name", "session", "btc_alignment"),
    ("family", "direction_name", "btc_alignment"),
    ("family", "session", "btc_alignment", "self_alignment"),
)
WINDOW_MONTHS = (3, 6, 12)
MIN_COUNTS = (20, 50)
MIN_MEAN_RETURNS = (0.0, .0002)
MIN_WIN_RATES = (.50, .55)
MIN_TARGET_MOVE = .005
MAX_TARGET_MOVE = .02
DEVELOPMENT_YEARS = (2021, 2022, 2023)
CONFIRMATION_YEARS = (2024,)
HIDDEN_YEARS = (2025, 2026)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(events: pd.DataFrame) -> pd.DataFrame:
    result = events.copy()
    result["direction_name"] = np.where(result.direction > 0, "LONG", "SHORT")
    result["btc_alignment"] = np.where(
        result.btc_ma20_96_dir_atr >= 0, "aligned", "opposed")
    result["self_alignment"] = np.where(
        result.ma20_96_dir_atr >= 0, "aligned", "opposed")
    result["entry_month"] = pd.to_datetime(
        result.entry_time, unit="s", utc=True).dt.strftime("%Y-%m")
    return result


def geometry_frame(events: pd.DataFrame, name: str) -> pd.DataFrame:
    stop_atr, target_r = GEOMETRIES[name]
    key = f"s{stop_atr:g}_t{target_r:g}"
    move = stop_atr * target_r * events.atr_pct.astype(float)
    result = events[move.between(MIN_TARGET_MOVE, MAX_TARGET_MOVE)].copy()
    # build_events stores dollar PnL for $3 margin at 10x, hence /30 restores
    # the signed underlying price return used by current-equity sizing.
    result["signed_price_return"] = result[f"pnl_{key}"].astype(float) / 30.0
    result["exit_time_selected"] = result[f"exit_{key}"].astype(float)
    result["exit_month"] = pd.to_datetime(
        result.exit_time_selected, unit="s", utc=True).dt.strftime("%Y-%m")
    result["geometry"] = name
    return result


def prior_months(month: str, count: int) -> list[str]:
    current = pd.Period(month, freq="M")
    return [str(current - offset) for offset in range(count, 0, -1)]


def training_summary(frame: pd.DataFrame, fields: tuple[str, ...],
                     month: str, window_months: int) -> pd.DataFrame:
    monthly = monthly_statistics({str(frame.geometry.iloc[0]): frame}, fields)
    return training_summary_from_monthly(
        next(iter(monthly.values())), fields, month, window_months)


def monthly_statistics(frames: dict[str, pd.DataFrame],
                       fields: tuple[str, ...]) -> dict[str, pd.DataFrame]:
    result = {}
    for geometry, frame in frames.items():
        grouped = frame.groupby(
            ["exit_month", *fields], dropna=False).signed_price_return
        summary = grouped.agg(
            count="size", sum_return="sum",
            wins=lambda values: int((values > 0).sum())).reset_index()
        summary["geometry"] = geometry
        result[geometry] = summary
    return result


def training_summary_from_monthly(monthly: pd.DataFrame,
                                  fields: tuple[str, ...], month: str,
                                  window_months: int) -> pd.DataFrame:
    train = monthly[monthly.exit_month.isin(
        prior_months(month, window_months))]
    if train.empty:
        return pd.DataFrame()
    result = train.groupby(list(fields), dropna=False).agg(
        count=("count", "sum"), sum_return=("sum_return", "sum"),
        wins=("wins", "sum")).reset_index()
    result["mean_return"] = result.sum_return / result["count"]
    result["win_rate"] = result.wins / result["count"]
    result["geometry"] = str(monthly.geometry.iloc[0])
    return result


def choose_experts(frames: dict[str, pd.DataFrame], fields: tuple[str, ...],
                   month: str, window_months: int, min_count: int,
                   min_mean_return: float, min_win_rate: float,
                   monthly: dict[str, pd.DataFrame] | None = None) -> pd.DataFrame:
    monthly = monthly or monthly_statistics(frames, fields)
    summaries = [training_summary_from_monthly(
        frame, fields, month, window_months)
        for frame in monthly.values()]
    summaries = [frame for frame in summaries if not frame.empty]
    if not summaries:
        return pd.DataFrame()
    candidates = pd.concat(summaries, ignore_index=True)
    candidates = candidates[
        candidates["count"].ge(min_count)
        & candidates.mean_return.ge(min_mean_return)
        & candidates.win_rate.ge(min_win_rate)]
    if candidates.empty:
        return candidates
    candidates = candidates.sort_values(
        [*fields, "mean_return", "count", "geometry"],
        ascending=[*[True] * len(fields), False, False, True],
        kind="mergesort")
    return candidates.drop_duplicates(list(fields), keep="first")


def route(frames: dict[str, pd.DataFrame], fields: tuple[str, ...],
          window_months: int, min_count: int, min_mean_return: float,
          min_win_rate: float,
          monthly: dict[str, pd.DataFrame] | None = None,
          entries: dict[str, dict[str, pd.DataFrame]] | None = None) -> list[dict]:
    monthly = monthly or monthly_statistics(frames, fields)
    entries = entries or {
        geometry: {month: part for month, part in frame.groupby(
            "entry_month", sort=False)}
        for geometry, frame in frames.items()
    }
    months = sorted(set().union(*(set(frame.entry_month) for frame in frames.values())))
    output = []
    for month in months:
        if month < "2021-01":
            continue
        experts = choose_experts(
            frames, fields, month, window_months, min_count,
            min_mean_return, min_win_rate, monthly)
        if experts.empty:
            continue
        experts = experts.reset_index(drop=True)
        experts["expert_priority"] = np.arange(len(experts), dtype=int)
        for geometry, expert_part in experts.groupby("geometry", sort=False):
            current = entries[str(geometry)].get(month)
            if current is None:
                continue
            selected = current.merge(
                expert_part[[*fields, "mean_return", "expert_priority"]],
                on=list(fields), how="inner", validate="many_to_one")
            if selected.empty:
                continue
            selected = selected.sort_values(
                ["entry_time", "mean_return", "symbol", "family"],
                ascending=[True, False, True, True], kind="mergesort")
            for row in selected.to_dict("records"):
                output.append({
                    **row,
                    "direction": str(row["direction_name"]),
                    "exit_time": float(row["exit_time_selected"]),
                    "outcome": ("TP" if float(row["signed_price_return"]) > 0
                                else "SL"),
                    "source": "causal_5m_expert_router",
                    "source_priority": 0,
                    "module": f"{row['family']}:{row['geometry']}",
                    "module_priority": int(row["expert_priority"]),
                })
    return runtime_gate(output, loss_pause=3, hourly_cap=2)


def stage(rows: list[dict], years: tuple[int, ...], margin: float) -> dict:
    chosen = [row for row in rows if pd.Timestamp(
        float(row["entry_time"]), unit="s", tz="UTC").year in years]
    return money(chosen, margin)


def month_floor(value: dict) -> float:
    monthly = value.get("monthly_return_pct", {})
    return min(monthly.values()) if monthly else -999.0


def rank_key(item: dict) -> tuple:
    value = item["development_3pct"]
    return (
        value["n"] >= 300 and value["dd_pct"] <= 15,
        -value["negative_months"], month_floor(value),
        value["return_pct"], -value["dd_pct"],
        value["trades_per_calendar_day"],
    )


def main() -> None:
    candles, sources = events_base.load()
    events = prepare(events_base.build_events(candles))
    keep = set(itertools.chain.from_iterable(FIELD_SETS)) | {
        "symbol", "family", "entry_time", "entry_month", "direction",
        "direction_name", "atr_pct", "ma20_96_dir_atr",
        "btc_ma20_96_dir_atr",
    }
    for stop_atr, target_r in GEOMETRIES.values():
        key = f"s{stop_atr:g}_t{target_r:g}"
        keep.update((f"pnl_{key}", f"exit_{key}"))
    events = events[[column for column in events.columns if column in keep]].copy()
    frames = {name: geometry_frame(events, name) for name in GEOMETRIES}
    entry_cache = {
        geometry: {month: part for month, part in frame.groupby(
            "entry_month", sort=False)}
        for geometry, frame in frames.items()
    }
    stats_cache = {
        fields: monthly_statistics(frames, fields) for fields in FIELD_SETS}

    candidates = []
    configs = itertools.product(
        FIELD_SETS, WINDOW_MONTHS, MIN_COUNTS,
        MIN_MEAN_RETURNS, MIN_WIN_RATES)
    for index, (fields, window_months, min_count,
                min_mean_return, min_win_rate) in enumerate(configs, 1):
        rows = route(frames, fields, window_months, min_count,
                     min_mean_return, min_win_rate,
                     stats_cache[fields], entry_cache)
        development = stage(rows, DEVELOPMENT_YEARS, .03)
        candidates.append({
            "fields": list(fields), "window_months": window_months,
            "min_count": min_count,
            "min_mean_price_return_pct": 100 * min_mean_return,
            "min_win_rate": 100 * min_win_rate,
            "development_3pct": development,
            "_rows": rows,
        })
        if index % 12 == 0:
            print(json.dumps({"phase": "router", "done": index}), flush=True)
    candidates.sort(key=rank_key, reverse=True)

    # 2024 is policy confirmation.  Only the best twenty development rules
    # are exposed to it, and 2025-2026 remain unread until one is frozen.
    finalists = candidates[:20]
    for item in finalists:
        item["confirmation_2024_3pct"] = stage(
            item["_rows"], CONFIRMATION_YEARS, .03)
    finalists.sort(key=lambda item: (
        item["confirmation_2024_3pct"]["negative_months"] == 0,
        month_floor(item["confirmation_2024_3pct"]),
        item["confirmation_2024_3pct"]["return_pct"],
        *rank_key(item),
    ), reverse=True)
    frozen = finalists[0]
    hidden = {
        "margin_3pct": stage(frozen["_rows"], HIDDEN_YEARS, .03),
        "margin_5pct": stage(frozen["_rows"], HIDDEN_YEARS, .05),
    }
    confirmation_5pct = stage(
        frozen["_rows"], CONFIRMATION_YEARS, .05)
    development_5pct = stage(
        frozen["_rows"], DEVELOPMENT_YEARS, .05)
    observed = [
        frozen["development_3pct"], frozen["confirmation_2024_3pct"],
        hidden["margin_3pct"], development_5pct,
        confirmation_5pct, hidden["margin_5pct"],
    ]
    objective = {
        "no_negative_months_any_stage": all(
            value["negative_months"] == 0 for value in observed),
        "hidden_minimum_month_at_least_30pct_5pct": (
            month_floor(hidden["margin_5pct"]) >= 30),
        "hidden_drawdown_at_most_10pct_5pct": (
            hidden["margin_5pct"]["dd_pct"] <= 10),
        "hidden_at_least_3_trades_per_day": (
            hidden["margin_5pct"]["trades_per_calendar_day"] >= 3),
    }
    public_frozen = {key: value for key, value in frozen.items()
                     if key != "_rows"}
    report = {
        "status": ("CAUSAL_EXPERT_ROUTER_SHADOW_CANDIDATE"
                   if all(objective.values()) else "REJECTED_OBJECTIVE"),
        "ready_for_live_orders": False,
        "protocol": {
            "warmup": "2020", "development": list(DEVELOPMENT_YEARS),
            "confirmation": list(CONFIRMATION_YEARS),
            "hidden": list(HIDDEN_YEARS),
            "training_cutoff": "complete prior calendar months by exit time",
            "entry": "next 5-minute market open",
            "target_price_move_pct": [100 * MIN_TARGET_MOVE,
                                       100 * MAX_TARGET_MOVE],
            "geometries": GEOMETRIES,
            "money": "client 3% or 5% current-equity margin; fixed 10x",
            "priority": "negative months, worst month, return, drawdown; win rate diagnostic",
        },
        "event_count": len(events),
        "geometry_event_counts": {key: len(value) for key, value in frames.items()},
        "candidate_count": len(candidates),
        "frozen_after_2024_confirmation": public_frozen,
        "development_5pct": development_5pct,
        "confirmation_2024_5pct": confirmation_5pct,
        "hidden_2025_2026": hidden,
        "objective_checks": objective,
        "top20_development_before_confirmation": [
            {key: value for key, value in item.items() if key != "_rows"}
            for item in candidates[:20]],
        "sources": {
            **sources,
            str(Path(__file__).resolve()): digest(Path(__file__).resolve()),
        },
        "limitations": [
            "Gross results exclude fees at user request.",
            "The router updates monthly and cannot react inside the month.",
            "Historical Binance spot evidence still requires direct X-Perp transfer.",
            "No backtest guarantees a future monthly return.",
        ],
    }
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({
        "status": report["status"], "frozen": public_frozen,
        "development_5pct": development_5pct,
        "confirmation_2024_5pct": confirmation_5pct,
        "hidden_2025_2026": hidden,
        "objective_checks": objective,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
