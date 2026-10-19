"""Causal trailing-health router for fixed X-Perp trading styles.

Both component books remain shadow-scored.  At each entry time the router uses
only component outcomes whose exits are already complete and enables the style
with the higher positive trailing mean.  May selects the health window, June
and July confirm it, and August-October are diagnostics without retuning.
"""
from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

import crypto_raw_hourly_direction_oct19_audit as parent
import crypto_xperp_cross_section_reversal_oct20_audit as cross
import crypto_xperp_regime_style_router_oct20_audit as regime


OUT = parent.ROOT / "crypto_xperp_style_health_router_oct20.json"
ROWS_OUT = parent.ROOT / "crypto_xperp_style_health_router_oct20.pkl"
LOOKBACK_DAYS = (3, 7, 14, 30)
MIN_COUNTS = (5, 10, 20)
MIN_MEANS = (0.0, .0005, .001)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def trailing_health_route(frame: pd.DataFrame, lookback_days: int,
                          min_count: int, min_mean: float) -> pd.DataFrame:
    """Choose one style using only virtual trades exited before each entry."""
    ordered = frame.sort_values(
        ["entry_time", "style", "model_quality", "symbol"],
        ascending=[True, True, False, True], kind="mergesort").copy()
    history = {}
    for style in regime.COMPONENTS:
        past = ordered[ordered["style"].eq(style)].sort_values("exit_time")
        exits = past.exit_time.to_numpy(float)
        returns = past.signed_return.to_numpy(float)
        history[style] = (
            exits, np.concatenate(([0.0], np.cumsum(returns))))
    accepted = []
    window = lookback_days * 86400
    for now, current in ordered.groupby("entry_time", sort=True):
        scores = []
        for style, (exits, cumulative) in history.items():
            left = int(np.searchsorted(exits, now - window, side="left"))
            right = int(np.searchsorted(exits, now, side="left"))
            count = right - left
            mean = (float((cumulative[right] - cumulative[left]) / count)
                    if count else float("-inf"))
            if count >= min_count and mean >= min_mean:
                scores.append((mean, count, style))
        if not scores:
            continue
        chosen_style = max(scores)[2]
        accepted.append(current[current["style"].eq(chosen_style)])
    if not accepted:
        return ordered.iloc[:0].copy()
    return pd.concat(accepted, ignore_index=True)


def rank_key(item: dict) -> tuple:
    values = [item["selection"], *item["confirmation"].values()]
    return (
        bool(item["admissible"]),
        min(float(value["return_pct"]) for value in values),
        sum(float(value["return_pct"]) for value in values),
        -max(float(value["dd_pct"]) for value in values),
        sum(int(value["n"]) for value in values),
    )


def main() -> None:
    candles = {}
    sources = {}
    for symbol in parent.CORE_SYMBOLS:
        path = parent.rails.DIRECT_CACHE / f"{symbol}_15min_current.pkl"
        candles[symbol] = parent.load_candles(path).to_dict("list")
        sources[str(path)] = digest(path)
    context = regime.market_context(candles)
    components = pd.concat([
        regime.component_frame(candles, style, context)
        for style in regime.COMPONENTS
    ], ignore_index=True)

    candidates = []
    routed_cache = {}
    for lookback_days, min_count, min_mean in itertools.product(
            LOOKBACK_DAYS, MIN_COUNTS, MIN_MEANS):
        key = (lookback_days, min_count, min_mean)
        routed_cache[key] = trailing_health_route(
            components, lookback_days, min_count, min_mean)
        chosen = routed_cache[key]
        for loss_pause in parent.LOSS_PAUSES:
            for hourly_cap in parent.HOURLY_CAPS:
                selection, _ = cross.period_metrics(
                    chosen, cross.SELECTION_MONTH, loss_pause, hourly_cap)
                confirmation = {
                    month: cross.period_metrics(
                        chosen, month, loss_pause, hourly_cap)[0]
                    for month in cross.CONFIRMATION_MONTHS
                }
                values = [selection, *confirmation.values()]
                candidates.append({
                    "lookback_days": lookback_days,
                    "minimum_completed_trades": min_count,
                    "minimum_mean_price_return_pct": 100 * min_mean,
                    "loss_pause": loss_pause, "hourly_cap": hourly_cap,
                    "selection": selection, "confirmation": confirmation,
                    "admissible": all(cross.period_ok(value)
                                      for value in values),
                })
    candidates.sort(key=rank_key, reverse=True)
    selected = candidates[0]
    chosen = routed_cache[(
        int(selected["lookback_days"]),
        int(selected["minimum_completed_trades"]),
        float(selected["minimum_mean_price_return_pct"]) / 100,
    )]

    holdouts = {}
    output_rows = []
    for margin in (.03, .05):
        name = f"margin_{round(100 * margin)}pct"
        holdouts[name] = {}
        for month in cross.HOLDOUT_MONTHS:
            metrics, rows = cross.period_metrics(
                chosen, month, int(selected["loss_pause"]),
                int(selected["hourly_cap"]), margin)
            holdouts[name][month] = metrics
            if margin == .03:
                output_rows.extend(rows)
    pd.DataFrame(output_rows).to_pickle(ROWS_OUT)

    three = holdouts["margin_3pct"]
    objective = bool(
        selected["admissible"]
        and all(three[month]["return_pct"] >= 30
                and three[month]["negative_months"] == 0
                and three[month]["dd_pct"] <= 10
                and three[month]["trades_per_calendar_day"] >= 3
                for month in ("2026-08", "2026-09"))
        and three["2026-10"]["return_pct"] > 0)
    report = {
        "status": ("XPERP_STYLE_HEALTH_SHADOW_CANDIDATE"
                   if objective else "REJECTED_OBJECTIVE"),
        "ready_for_live_orders": False,
        "protocol": {
            "components": regime.COMPONENTS,
            "health_lookback_days": list(LOOKBACK_DAYS),
            "minimum_completed_trades": list(MIN_COUNTS),
            "minimum_mean_price_return_pct": [100 * value
                                               for value in MIN_MEANS],
            "selection": cross.SELECTION_MONTH,
            "confirmation": list(cross.CONFIRMATION_MONTHS),
            "diagnostic_holdouts": list(cross.HOLDOUT_MONTHS),
            "health_source": "resolved virtual component trades only",
            "entry": "next 15-minute market open",
            "orders": "market only",
            "money": "client 3% or 5% current-equity margin; fixed 10x",
        },
        "candidate_count": len(candidates),
        "admissible_count": sum(item["admissible"] for item in candidates),
        "selected_before_holdouts": selected,
        "top20_before_holdouts": candidates[:20],
        "holdouts": holdouts,
        "objective_pass": objective,
        "sources": {
            **sources,
            str(Path(__file__).resolve()): digest(Path(__file__).resolve()),
        },
        "limitations": [
            "The health router was conceived after earlier August-October results were observed.",
            "Virtual outcomes require continuously tracking both shadow books.",
            "Direct venue history begins in partial April 2026.",
            "October is incomplete and remains diagnostic.",
            "Gross returns exclude fees at user request.",
            "No backtest guarantees a future monthly return.",
        ],
    }
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True),
                   encoding="utf-8")
    print(json.dumps({
        "status": report["status"],
        "candidate_count": report["candidate_count"],
        "admissible_count": report["admissible_count"],
        "selected": selected, "holdouts": holdouts,
        "objective_pass": objective,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
