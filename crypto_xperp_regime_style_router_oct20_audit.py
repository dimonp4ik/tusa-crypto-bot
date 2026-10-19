"""Route between fixed cross-sectional reversal and momentum X-Perp styles.

Both component definitions were selected on May-July before the August-October
diagnostic window.  A small causal router chooses the style from synchronized
closed-candle market state.  May selects the switch, June and July must confirm
it independently, and August-October are reported without retuning.
"""
from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

import crypto_raw_geometry_walkforward_oct20_audit as geometry_audit
import crypto_raw_hourly_direction_oct19_audit as parent
import crypto_xperp_cross_section_reversal_oct20_audit as cross


OUT = parent.ROOT / "crypto_xperp_regime_style_router_oct20.json"
ROWS_OUT = parent.ROOT / "crypto_xperp_regime_style_router_oct20.pkl"
COMPONENTS = {
    "reversal": {
        "lookback": 2, "minimum_universe": 3, "z_threshold": 1.25,
        "move_threshold": .004, "geometry": (.015, .0075, 48),
    },
    "momentum": {
        "lookback": 4, "minimum_universe": 3, "z_threshold": 1.25,
        "move_threshold": .001, "geometry": (.010, .0075, 48),
    },
}
REGIME_FIELDS = (
    "btc_efficiency8", "btc_abs_return8", "market_abs_mean4",
    "market_dispersion4", "market_breadth_extreme4",
    "common_to_dispersion4",
)
QUANTILES = (.25, .50, .75)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def market_context(candles: dict[str, dict]) -> pd.DataFrame:
    """Compute regime state from the last closed 15-minute candles only."""
    closes = []
    for symbol, values in candles.items():
        index = pd.Index(np.asarray(values["time"], dtype=np.int64),
                         name="signal_time")
        closes.append(pd.Series(np.asarray(values["close"], dtype=float),
                                index=index, name=symbol))
    close = pd.concat(closes, axis=1).sort_index()
    returns4 = close / close.shift(4) - 1
    btc = close["BTCUSDT"]
    btc_change = btc.diff().abs()
    efficiency8 = ((btc - btc.shift(8)).abs()
                   / btc_change.rolling(8).sum().replace(0, np.nan))
    mean4 = returns4.mean(axis=1)
    dispersion4 = returns4.std(axis=1)
    breadth4 = (returns4 > 0).sum(axis=1) / returns4.notna().sum(
        axis=1).replace(0, np.nan)
    result = pd.DataFrame({
        "entry_time": close.index.to_numpy(np.int64) + 900,
        "btc_efficiency8": efficiency8.to_numpy(float),
        "btc_abs_return8": (btc / btc.shift(8) - 1).abs().to_numpy(float),
        "market_abs_mean4": mean4.abs().to_numpy(float),
        "market_dispersion4": dispersion4.to_numpy(float),
        "market_breadth_extreme4": (breadth4 - .5).abs().to_numpy(float),
        "common_to_dispersion4": (
            mean4.abs() / dispersion4.replace(0, np.nan)).to_numpy(float),
    })
    return result.replace([np.inf, -np.inf], np.nan).dropna()


def component_frame(candles: dict[str, dict], style: str,
                    context: pd.DataFrame) -> pd.DataFrame:
    config = COMPONENTS[style]
    raw = cross.feature_rows(candles, int(config["lookback"]), style)
    labeled = geometry_audit.relabel(
        raw, tuple(config["geometry"]), direct=True)
    selected = cross.select_entries(
        labeled, int(config["minimum_universe"]),
        float(config["z_threshold"]), float(config["move_threshold"]))
    selected["style"] = style
    return selected.merge(context, on="entry_time", how="inner",
                          validate="many_to_one")


def routed(components: pd.DataFrame, field: str, threshold: float,
           high_style: str) -> pd.DataFrame:
    low_style = "momentum" if high_style == "reversal" else "reversal"
    desired = np.where(components[field].ge(threshold),
                       high_style, low_style)
    return components[components["style"].eq(desired)].copy()


def period(frame: pd.DataFrame, month: str, loss_pause: int,
           hourly_cap: int, margin: float = .03) -> tuple[dict, list[dict]]:
    return cross.period_metrics(frame, month, loss_pause, hourly_cap, margin)


def admissible(item: dict) -> bool:
    values = [item["selection"], *item["confirmation"].values()]
    return all(cross.period_ok(value) for value in values)


def rank_key(item: dict) -> tuple:
    values = [item["selection"], *item["confirmation"].values()]
    return (
        bool(item["admissible"]),
        min(float(value["return_pct"]) for value in values),
        sum(float(value["return_pct"]) for value in values),
        -max(float(value["dd_pct"]) for value in values),
    )


def main() -> None:
    candles = {}
    sources = {}
    for symbol in parent.CORE_SYMBOLS:
        path = parent.rails.DIRECT_CACHE / f"{symbol}_15min_current.pkl"
        candles[symbol] = parent.load_candles(path).to_dict("list")
        sources[str(path)] = digest(path)
    context = market_context(candles)
    components = pd.concat([
        component_frame(candles, style, context) for style in COMPONENTS
    ], ignore_index=True)

    may_context = context[pd.to_datetime(
        context.entry_time, unit="s", utc=True).dt.strftime("%Y-%m").eq(
            cross.SELECTION_MONTH)]
    policies = []
    for field, quantile, high_style in itertools.product(
            REGIME_FIELDS, QUANTILES, COMPONENTS):
        threshold = float(may_context[field].quantile(quantile))
        chosen = routed(components, field, threshold, high_style)
        for loss_pause in parent.LOSS_PAUSES:
            for hourly_cap in parent.HOURLY_CAPS:
                selection, _ = period(
                    chosen, cross.SELECTION_MONTH, loss_pause, hourly_cap)
                confirmation = {
                    month: period(chosen, month, loss_pause, hourly_cap)[0]
                    for month in cross.CONFIRMATION_MONTHS
                }
                item = {
                    "field": field, "quantile": quantile,
                    "threshold": threshold, "high_style": high_style,
                    "low_style": ("momentum" if high_style == "reversal"
                                  else "reversal"),
                    "loss_pause": loss_pause, "hourly_cap": hourly_cap,
                    "selection": selection, "confirmation": confirmation,
                }
                item["admissible"] = admissible(item)
                policies.append(item)
    policies.sort(key=rank_key, reverse=True)
    selected = policies[0]
    chosen = routed(
        components, str(selected["field"]), float(selected["threshold"]),
        str(selected["high_style"]))

    holdouts = {}
    output_rows = []
    for margin in (.03, .05):
        name = f"margin_{round(100 * margin)}pct"
        holdouts[name] = {}
        for month in cross.HOLDOUT_MONTHS:
            metrics, rows = period(
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
        "status": ("XPERP_REGIME_STYLE_SHADOW_CANDIDATE"
                   if objective else "REJECTED_OBJECTIVE"),
        "ready_for_live_orders": False,
        "protocol": {
            "components": COMPONENTS,
            "regime_fields": list(REGIME_FIELDS),
            "quantiles": list(QUANTILES),
            "selection": cross.SELECTION_MONTH,
            "confirmation": list(cross.CONFIRMATION_MONTHS),
            "diagnostic_holdouts": list(cross.HOLDOUT_MONTHS),
            "entry": "next 15-minute market open",
            "orders": "market only",
            "money": "client 3% or 5% current-equity margin; fixed 10x",
            "causality": "router state uses synchronized closed candles only",
        },
        "candidate_count": len(policies),
        "admissible_count": sum(item["admissible"] for item in policies),
        "selected_before_holdouts": selected,
        "top20_before_holdouts": policies[:20],
        "holdouts": holdouts,
        "objective_pass": objective,
        "sources": {
            **sources,
            str(Path(__file__).resolve()): digest(Path(__file__).resolve()),
        },
        "limitations": [
            "The router was conceived after earlier August-October results were observed.",
            "Direct venue history begins in partial April 2026.",
            "Only five symbols have continuous native history.",
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
