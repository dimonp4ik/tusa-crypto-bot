"""Require market-state and trailing-health X-Perp routers to agree.

The two causal routers are frozen from May-July evidence.  Their intersection
is replayed with shared portfolio rails; August-October remain diagnostic.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

import crypto_raw_hourly_direction_oct19_audit as parent
import crypto_xperp_cross_section_reversal_oct20_audit as cross
import crypto_xperp_regime_style_router_oct20_audit as market
import crypto_xperp_style_health_router_oct20_audit as health


OUT = parent.ROOT / "crypto_xperp_agreement_router_oct20.json"
ROWS_OUT = parent.ROOT / "crypto_xperp_agreement_router_oct20.pkl"
KEYS = ["symbol", "entry_time", "style", "direction_n"]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def intersection(left: pd.DataFrame, right: pd.DataFrame) -> pd.DataFrame:
    keys = right[KEYS].drop_duplicates()
    return left.merge(keys, on=KEYS, how="inner", validate="many_to_one")


def rank_key(item: dict) -> tuple:
    values = [item["selection"], *item["confirmation"].values()]
    return (
        bool(item["admissible"]),
        min(float(value["return_pct"]) for value in values),
        sum(float(value["return_pct"]) for value in values),
        -max(float(value["dd_pct"]) for value in values),
    )


def main() -> None:
    market_report = json.loads(market.OUT.read_text(encoding="utf-8"))
    health_report = json.loads(health.OUT.read_text(encoding="utf-8"))
    market_policy = market_report["selected_before_holdouts"]
    health_policy = health_report["selected_before_holdouts"]

    candles = {}
    sources = {}
    for symbol in parent.CORE_SYMBOLS:
        path = parent.rails.DIRECT_CACHE / f"{symbol}_15min_current.pkl"
        candles[symbol] = parent.load_candles(path).to_dict("list")
        sources[str(path)] = digest(path)
    context = market.market_context(candles)
    components = pd.concat([
        market.component_frame(candles, style, context)
        for style in market.COMPONENTS
    ], ignore_index=True)
    market_rows = market.routed(
        components, str(market_policy["field"]),
        float(market_policy["threshold"]),
        str(market_policy["high_style"]))
    health_rows = health.trailing_health_route(
        components, int(health_policy["lookback_days"]),
        int(health_policy["minimum_completed_trades"]),
        float(health_policy["minimum_mean_price_return_pct"]) / 100)
    agreed = intersection(market_rows, health_rows)

    candidates = []
    for loss_pause in parent.LOSS_PAUSES:
        for hourly_cap in parent.HOURLY_CAPS:
            selection, _ = cross.period_metrics(
                agreed, cross.SELECTION_MONTH, loss_pause, hourly_cap)
            confirmation = {
                month: cross.period_metrics(
                    agreed, month, loss_pause, hourly_cap)[0]
                for month in cross.CONFIRMATION_MONTHS
            }
            values = [selection, *confirmation.values()]
            candidates.append({
                "loss_pause": loss_pause, "hourly_cap": hourly_cap,
                "selection": selection, "confirmation": confirmation,
                "admissible": all(cross.period_ok(value)
                                  for value in values),
            })
    candidates.sort(key=rank_key, reverse=True)
    selected = candidates[0]
    holdouts = {}
    output_rows = []
    for margin_fraction in (.03, .05):
        name = f"margin_{round(100 * margin_fraction)}pct"
        holdouts[name] = {}
        for month in cross.HOLDOUT_MONTHS:
            metrics, rows = cross.period_metrics(
                agreed, month, int(selected["loss_pause"]),
                int(selected["hourly_cap"]), margin_fraction)
            holdouts[name][month] = metrics
            if margin_fraction == .03:
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
        "status": ("XPERP_AGREEMENT_SHADOW_CANDIDATE"
                   if objective else "REJECTED_OBJECTIVE"),
        "ready_for_live_orders": False,
        "protocol": {
            "market_policy": market_policy,
            "health_policy": health_policy,
            "selection": cross.SELECTION_MONTH,
            "confirmation": list(cross.CONFIRMATION_MONTHS),
            "diagnostic_holdouts": list(cross.HOLDOUT_MONTHS),
            "entry": "next 15-minute market open",
            "orders": "market only",
            "money": "client 3% or 5% current-equity margin; fixed 10x",
        },
        "candidate_count": len(candidates),
        "selected_before_holdouts": selected,
        "holdouts": holdouts,
        "objective_pass": objective,
        "sources": {
            **sources,
            str(market.OUT): digest(market.OUT),
            str(health.OUT): digest(health.OUT),
            str(Path(__file__).resolve()): digest(Path(__file__).resolve()),
        },
        "limitations": [
            "The agreement router was conceived after earlier holdout results were observed.",
            "Direct venue history begins in partial April 2026.",
            "October is incomplete and remains diagnostic.",
            "Gross returns exclude fees at user request.",
            "No backtest guarantees a future monthly return.",
        ],
    }
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True),
                   encoding="utf-8")
    print(json.dumps({
        "status": report["status"], "selected": selected,
        "holdouts": holdouts, "objective_pass": objective,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
