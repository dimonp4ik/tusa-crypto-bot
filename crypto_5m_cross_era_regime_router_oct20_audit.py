"""Select contemporaneous crypto experts that survive every development era.

Unlike a trailing-performance router, this policy reads only current closed
bar state.  A bucket and exit geometry must have positive expectancy in each
of 2020, 2021, 2022 and 2023 independently.  2024 confirms the frozen policy;
2025-2026 are revealed last.
"""
from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

import crypto_5m_causal_expert_router_oct20_audit as causal
import crypto_5m_tree_filter_audit as events_base
from crypto_runtime_shadow_portfolio_oct3_audit import money, runtime_gate


OUT = Path("reports/audit_2026_09_08/crypto_5m_cross_era_regime_router_oct20.json")
DEVELOPMENT_YEARS = (2020, 2021, 2022, 2023)
CONFIRMATION_YEARS = (2024,)
HIDDEN_YEARS = (2025, 2026)
FIELD_SETS = (
    ("family", "direction_name", "btc_alignment", "self_alignment"),
    ("family", "direction_name", "session", "btc_alignment"),
    ("family", "direction_name", "btc_eff_regime", "vol_regime"),
    ("family", "session", "btc_alignment", "vol_regime"),
    ("family", "direction_name", "session", "btc_eff_regime", "vol_regime"),
    ("family", "direction_name", "session", "btc_trend", "btc_impulse"),
    ("family", "direction_name", "btc_trend", "btc_eff_regime", "vol_regime"),
    ("family", "session", "btc_trend", "btc_impulse", "vol_regime"),
    ("family", "direction_name", "session", "self_trend", "btc_trend"),
)
MIN_COUNTS = (10, 25)
MIN_YEAR_MEANS = (0.0, .0001)
MIN_YEAR_WIN_RATES = (.50, .55)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def add_regimes(events: pd.DataFrame) -> pd.DataFrame:
    result = causal.prepare(events)
    direction = result.direction.astype(float)
    raw_btc_trend = result.btc_ma20_96_dir_atr * direction
    raw_self_trend = result.ma20_96_dir_atr * direction
    raw_btc_impulse = result.btc_ret12_dir_atr * direction
    result["btc_trend"] = np.where(raw_btc_trend >= 0, "bull", "bear")
    result["self_trend"] = np.where(raw_self_trend >= 0, "bull", "bear")
    result["btc_impulse"] = pd.cut(
        raw_btc_impulse, [-np.inf, -1.0, 1.0, np.inf],
        labels=["down", "flat", "up"]).astype(str)
    result["btc_eff_regime"] = pd.cut(
        result.btc_eff20, [-np.inf, .2, .4, np.inf],
        labels=["range", "mixed", "efficient"]).astype(str)
    result["vol_regime"] = pd.cut(
        result.atr_pct, [-np.inf, .003, .006, np.inf],
        labels=["low", "normal", "high"]).astype(str)
    return result


def stable_experts(frames: dict[str, pd.DataFrame], fields: tuple[str, ...],
                   min_count: int, min_year_mean: float,
                   min_year_win_rate: float) -> pd.DataFrame:
    candidates = []
    for geometry, frame in frames.items():
        dev = frame[frame.year.isin(DEVELOPMENT_YEARS)]
        grouped = dev.groupby(["year", *fields], dropna=False).signed_price_return
        stats = grouped.agg(
            count="size", mean_return="mean",
            win_rate=lambda values: float((values > 0).mean())).reset_index()
        good = stats[
            stats["count"].ge(min_count)
            & stats.mean_return.ge(min_year_mean)
            & stats.win_rate.ge(min_year_win_rate)]
        if good.empty:
            continue
        stable = good.groupby(list(fields), dropna=False).filter(
            lambda part: set(part.year.astype(int)) == set(DEVELOPMENT_YEARS))
        if stable.empty:
            continue
        summary = stable.groupby(list(fields), dropna=False).agg(
            worst_year_mean=("mean_return", "min"),
            mean_return=("mean_return", "mean"),
            worst_year_win_rate=("win_rate", "min"),
            observations=("count", "sum")).reset_index()
        summary["geometry"] = geometry
        candidates.append(summary)
    if not candidates:
        return pd.DataFrame()
    result = pd.concat(candidates, ignore_index=True)
    result = result.sort_values(
        [*fields, "worst_year_mean", "mean_return", "observations", "geometry"],
        ascending=[*[True] * len(fields), False, False, False, True],
        kind="mergesort")
    return result.drop_duplicates(list(fields), keep="first")


def route(frames: dict[str, pd.DataFrame], fields: tuple[str, ...],
          experts: pd.DataFrame) -> list[dict]:
    if experts.empty:
        return []
    experts = experts.reset_index(drop=True)
    experts["expert_priority"] = np.arange(len(experts), dtype=int)
    output = []
    for geometry, expert_part in experts.groupby("geometry", sort=False):
        selected = frames[str(geometry)].merge(
            expert_part[[*fields, "worst_year_mean", "expert_priority"]],
            on=list(fields), how="inner", validate="many_to_one")
        selected = selected.sort_values(
            ["entry_time", "worst_year_mean", "symbol", "family"],
            ascending=[True, False, True, True], kind="mergesort")
        for row in selected.to_dict("records"):
            output.append({
                **row,
                "direction": str(row["direction_name"]),
                "exit_time": float(row["exit_time_selected"]),
                "outcome": ("TP" if float(row["signed_price_return"]) > 0
                            else "SL"),
                "source": "cross_era_5m_regime_router",
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


def development_rank(item: dict) -> tuple:
    value = item["development_3pct"]
    return (
        value["n"] >= 300 and value["dd_pct"] <= 15,
        -value["negative_months"], month_floor(value),
        value["return_pct"], -value["dd_pct"],
        value["trades_per_calendar_day"],
    )


def main() -> None:
    candles, sources = events_base.load()
    events = add_regimes(events_base.build_events(candles))
    frames = {name: causal.geometry_frame(events, name)
              for name in causal.GEOMETRIES}
    candidates = []
    for index, (fields, min_count, min_mean, min_win) in enumerate(
            itertools.product(FIELD_SETS, MIN_COUNTS, MIN_YEAR_MEANS,
                              MIN_YEAR_WIN_RATES), 1):
        experts = stable_experts(
            frames, fields, min_count, min_mean, min_win)
        rows = route(frames, fields, experts)
        candidates.append({
            "fields": list(fields), "min_count_each_year": min_count,
            "min_mean_price_return_each_year_pct": 100 * min_mean,
            "min_win_rate_each_year": 100 * min_win,
            "expert_count": len(experts),
            "experts": experts.to_dict("records"),
            "development_3pct": stage(rows, DEVELOPMENT_YEARS, .03),
            "_rows": rows,
        })
        if index % 10 == 0:
            print(json.dumps({"phase": "cross_era", "done": index}), flush=True)
    candidates.sort(key=development_rank, reverse=True)
    finalists = candidates[:10]
    for item in finalists:
        item["confirmation_2024_3pct"] = stage(
            item["_rows"], CONFIRMATION_YEARS, .03)
    finalists.sort(key=lambda item: (
        item["confirmation_2024_3pct"]["negative_months"] == 0,
        month_floor(item["confirmation_2024_3pct"]),
        item["confirmation_2024_3pct"]["return_pct"],
        *development_rank(item),
    ), reverse=True)
    frozen = finalists[0]
    hidden = {
        "margin_3pct": stage(frozen["_rows"], HIDDEN_YEARS, .03),
        "margin_5pct": stage(frozen["_rows"], HIDDEN_YEARS, .05),
    }
    development_5pct = stage(frozen["_rows"], DEVELOPMENT_YEARS, .05)
    confirmation_5pct = stage(frozen["_rows"], CONFIRMATION_YEARS, .05)
    all_values = [
        frozen["development_3pct"], frozen["confirmation_2024_3pct"],
        hidden["margin_3pct"], development_5pct, confirmation_5pct,
        hidden["margin_5pct"],
    ]
    objective = {
        "no_negative_months_any_stage": all(
            value["negative_months"] == 0 for value in all_values),
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
        "status": ("CROSS_ERA_REGIME_ROUTER_SHADOW_CANDIDATE"
                   if all(objective.values()) else "REJECTED_OBJECTIVE"),
        "ready_for_live_orders": False,
        "protocol": {
            "development": list(DEVELOPMENT_YEARS),
            "confirmation": list(CONFIRMATION_YEARS),
            "hidden": list(HIDDEN_YEARS),
            "expert_rule": "positive count, mean return and win rate in every development year",
            "entry": "next 5-minute market open",
            "target_price_move_pct": [100 * causal.MIN_TARGET_MOVE,
                                       100 * causal.MAX_TARGET_MOVE],
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
        "top10_development_before_confirmation": [
            {key: value for key, value in item.items() if key != "_rows"}
            for item in candidates[:10]],
        "sources": {
            **sources,
            str(Path(__file__).resolve()): digest(Path(__file__).resolve()),
        },
        "limitations": [
            "Gross results exclude fees at user request.",
            "Fixed state buckets may miss gradual regime transitions.",
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
