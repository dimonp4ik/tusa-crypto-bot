"""Cross-era crypto router gated by contemporaneous signal consensus.

The audit tests whether clustered entries are useful confirmation or correlated
noise.  Consensus uses only signals emitted on the same already-closed candle:
the number of distinct symbols/families agreeing with a direction and the
fraction of all current signals pointing that way.  Development is 2020-2023,
2024 selects one frozen policy, and 2025-2026 are revealed once at the end.
"""
from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

import crypto_5m_causal_expert_router_oct20_audit as causal
import crypto_5m_cross_era_regime_router_oct20_audit as base
import crypto_5m_tree_filter_audit as events_base


OUT = Path("reports/audit_2026_09_08/crypto_5m_consensus_regime_router_oct20.json")
FIELD_SETS = (
    ("family", "direction_name", "session", "consensus_symbols",
     "directional_consensus"),
    ("family", "direction_name", "btc_trend", "consensus_families",
     "directional_consensus"),
    ("family", "session", "breadth_alignment", "consensus_symbols",
     "directional_consensus"),
    ("family", "direction_name", "btc_eff_regime", "consensus_symbols",
     "directional_consensus"),
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def add_signal_consensus(events: pd.DataFrame) -> pd.DataFrame:
    """Attach same-candle agreement without reading outcomes or future bars."""
    keys = ["entry_time", "direction"]
    directional = events.groupby(keys, sort=False).agg(
        agreeing_symbols=("symbol", "nunique"),
        agreeing_families=("family", "nunique"),
        agreeing_signals=("family", "size"),
    ).reset_index()
    total = events.groupby("entry_time", sort=False).size().rename(
        "all_signals").reset_index()
    result = events.merge(directional, on=keys, how="left",
                          validate="many_to_one")
    result = result.merge(total, on="entry_time", how="left",
                          validate="many_to_one")
    ratio = result.agreeing_signals / result.all_signals.clip(lower=1)
    result["consensus_symbols"] = pd.cut(
        result.agreeing_symbols, [-np.inf, 1, 3, np.inf],
        labels=["solo", "group", "crowd"]).astype(str)
    result["consensus_families"] = pd.cut(
        result.agreeing_families, [-np.inf, 1, 2, np.inf],
        labels=["one", "two", "multi"]).astype(str)
    result["directional_consensus"] = pd.cut(
        ratio, [-np.inf, .60, .80, np.inf],
        labels=["conflicted", "majority", "strong"]).astype(str)
    return result


def rank(item: dict) -> tuple:
    value = item["confirmation_2024_3pct"]
    return (
        value["negative_months"] == 0,
        base.month_floor(value), value["return_pct"], -value["dd_pct"],
        *base.development_rank(item),
    )


def main() -> None:
    candles, sources = events_base.load()
    raw_events = add_signal_consensus(events_base.build_events(candles))
    events = base.add_regimes(raw_events, candles)
    frames = {name: causal.geometry_frame(events, name)
              for name in causal.GEOMETRIES}
    candidates = []
    configs = itertools.product(
        FIELD_SETS, base.MIN_COUNTS, base.MIN_YEAR_MEANS,
        base.MIN_YEAR_WIN_RATES)
    for index, (fields, min_count, min_mean, min_win) in enumerate(configs, 1):
        experts = base.stable_experts(
            frames, fields, min_count, min_mean, min_win)
        rows = base.route(frames, fields, experts, loss_pause=3, hourly_cap=2)
        candidates.append({
            "fields": list(fields), "min_count_each_year": min_count,
            "min_mean_price_return_each_year_pct": 100 * min_mean,
            "min_win_rate_each_year": 100 * min_win,
            "expert_count": len(experts),
            "experts": experts.to_dict("records"),
            "development_3pct": base.stage(
                rows, base.DEVELOPMENT_YEARS, .03),
            "_rows": rows,
        })
        if index % 8 == 0:
            print(json.dumps({"phase": "consensus", "done": index}), flush=True)

    candidates.sort(key=base.development_rank, reverse=True)
    finalists = candidates[:10]
    for item in finalists:
        item["confirmation_2024_3pct"] = base.stage(
            item["_rows"], base.CONFIRMATION_YEARS, .03)
    finalists.sort(key=rank, reverse=True)
    frozen = finalists[0]
    hidden = {
        "margin_3pct": base.stage(frozen["_rows"], base.HIDDEN_YEARS, .03),
        "margin_5pct": base.stage(frozen["_rows"], base.HIDDEN_YEARS, .05),
    }
    development_5pct = base.stage(
        frozen["_rows"], base.DEVELOPMENT_YEARS, .05)
    confirmation_5pct = base.stage(
        frozen["_rows"], base.CONFIRMATION_YEARS, .05)
    values = [
        frozen["development_3pct"], frozen["confirmation_2024_3pct"],
        development_5pct, confirmation_5pct, *hidden.values(),
    ]
    objective = {
        "no_negative_months_any_stage": all(
            value["negative_months"] == 0 for value in values),
        "hidden_minimum_month_at_least_30pct_5pct": (
            base.month_floor(hidden["margin_5pct"]) >= 30),
        "hidden_drawdown_at_most_10pct_5pct": (
            hidden["margin_5pct"]["dd_pct"] <= 10),
        "hidden_at_least_3_trades_per_day": (
            hidden["margin_5pct"]["trades_per_calendar_day"] >= 3),
    }
    public_frozen = {key: value for key, value in frozen.items()
                     if key != "_rows"}
    report = {
        "status": ("CONSENSUS_REGIME_ROUTER_SHADOW_CANDIDATE"
                   if all(objective.values()) else "REJECTED_OBJECTIVE"),
        "ready_for_live_orders": False,
        "protocol": {
            "development": list(base.DEVELOPMENT_YEARS),
            "confirmation": list(base.CONFIRMATION_YEARS),
            "hidden": list(base.HIDDEN_YEARS),
            "consensus": "same closed candle only; no outcome data",
            "entry": "next 5-minute market open",
            "target_price_move_pct": [100 * causal.MIN_TARGET_MOVE,
                                      100 * causal.MAX_TARGET_MOVE],
            "runtime_rails": {"daily_loss_pause": 3,
                              "maximum_entries_per_clock_hour": 2},
            "money": "client 3% or 5% current-equity margin; fixed 10x",
        },
        "event_count": len(events),
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
            str(Path(base.__file__).resolve()): digest(Path(base.__file__).resolve()),
        },
        "limitations": [
            "Gross results exclude fees at user request.",
            "Consensus can still reflect correlated signals from one market shock.",
            "Historical Binance spot evidence still requires direct X-Perp transfer.",
            "No backtest guarantees a future monthly return.",
        ],
    }
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True),
                   encoding="utf-8")
    print(json.dumps({
        "status": report["status"], "frozen": public_frozen,
        "development_5pct": development_5pct,
        "confirmation_2024_5pct": confirmation_5pct,
        "hidden_2025_2026": hidden, "objective_checks": objective,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
