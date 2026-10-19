"""Fresh-only replay of the frozen stable-universe X-Perp rule.

The strategy parameters come from the pre-existing October 20 audit.  This
script does not retune them.  It evaluates only entries after the previous
cache ended, while replaying the full refreshed history so daily pause and
open-position state at the boundary remain causal.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

import crypto_raw_geometry_walkforward_oct20_audit as geometry_audit
import crypto_raw_hourly_direction_oct19_audit as parent
import crypto_xperp_cross_section_reversal_oct20_audit as cross
import crypto_xperp_cross_section_stable_universe_oct20_audit as frozen
from crypto_runtime_shadow_portfolio_oct3_audit import money


ROOT = parent.ROOT
OLD_CACHE = parent.rails.DIRECT_CACHE
FRESH_CACHE = ROOT / "xperp_15m_oct20_forward"
OUT = ROOT / "crypto_xperp_stable_universe_fresh_oct20.json"
ROWS_OUT = ROOT / "crypto_xperp_stable_universe_fresh_oct20.pkl"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def after_cutoff(rows: list[dict], cutoff: int) -> list[dict]:
    return [row for row in rows if int(float(row["entry_time"])) > cutoff]


def main() -> None:
    prior = json.loads(frozen.OUT.read_text(encoding="utf-8"))
    selected = prior["selected_before_holdouts"]
    old_ends = []
    new_ends = []
    candles = {}
    sources = {str(frozen.OUT): digest(frozen.OUT)}
    for symbol in frozen.SYMBOLS:
        old_path = OLD_CACHE / f"{symbol}_15min_current.pkl"
        new_path = FRESH_CACHE / f"{symbol}_15min_current.pkl"
        old = parent.load_candles(old_path)
        new = parent.load_candles(new_path)
        old_ends.append(int(old.time.max()))
        new_ends.append(int(new.time.max()))
        candles[symbol] = new.to_dict("list")
        sources[str(old_path)] = digest(old_path)
        sources[str(new_path)] = digest(new_path)
    cutoff = min(old_ends)
    refreshed_end = min(new_ends)

    lookback = int(selected["lookback_bars"])
    style = str(selected["style"])
    geometry = (float(selected["target_pct"]) / 100,
                float(selected["stop_pct"]) / 100,
                int(selected["hold_bars"]))
    features = cross.feature_rows(candles, lookback, style)
    original_cache = parent.rails.DIRECT_CACHE
    try:
        parent.rails.DIRECT_CACHE = FRESH_CACHE
        labeled = geometry_audit.relabel(features, geometry, direct=True)
    finally:
        parent.rails.DIRECT_CACHE = original_cache
    chosen = cross.select_entries(
        labeled, int(selected["minimum_universe"]),
        float(selected["z_threshold"]),
        float(selected["minimum_move_pct"]) / 100)
    raw_fresh = chosen[chosen.entry_time.gt(cutoff)].copy()
    holdouts = {}
    output_rows = []
    for margin in (.03, .05):
        _, full_rows = parent.evaluate(
            chosen, float("-inf"), int(selected["loss_pause"]),
            int(selected["hourly_cap"]), margin)
        rows = after_cutoff(full_rows, cutoff)
        holdouts[f"margin_{round(100 * margin)}pct"] = money(rows, margin)
        if margin == .03:
            output_rows = rows
    pd.DataFrame(output_rows).to_pickle(ROWS_OUT)
    three = holdouts["margin_3pct"]
    report = {
        "status": ("FRESH_FORWARD_POSITIVE" if three["return_pct"] > 0
                   else "FRESH_FORWARD_NONPOSITIVE"),
        "ready_for_live_orders": False,
        "protocol": {
            "strategy_source": str(frozen.OUT),
            "parameters_frozen_before_fresh_data": selected,
            "fresh_entry_after": cutoff,
            "refreshed_history_end": refreshed_end,
            "entry": "next 15-minute market open",
            "orders": "market only",
            "collision": "stop first",
            "money": "client 3% or 5% current-equity margin; fixed 10x",
        },
        "raw_completed_fresh_signals": len(raw_fresh),
        "accepted_fresh_signals": len(output_rows),
        "fresh": holdouts,
        "objective_pass": False,
        "sources": {
            **sources,
            str(Path(__file__).resolve()): digest(Path(__file__).resolve()),
        },
        "limitations": [
            "The fresh window spans only about three days and cannot validate a monthly target.",
            "This strategy had already failed the earlier monthly objective.",
            "Gross returns exclude fees at user request.",
            "No backtest guarantees future returns.",
        ],
    }
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True),
                   encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
