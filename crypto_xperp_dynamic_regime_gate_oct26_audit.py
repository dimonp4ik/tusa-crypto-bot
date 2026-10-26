"""Causal regime gates for the stable native X-Perp reversal candidate.

The underlying geometry is fixed by the preceding July-September stability
audit.  This bounded search uses only features known at the signal close and
leaves partial October untouched until the gate is frozen.
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
import crypto_xperp_cross_section_reversal_oct20_audit as prior
import crypto_xperp_dynamic_universe_cross_section_oct26_audit as base


OUT = parent.ROOT / "crypto_xperp_dynamic_regime_gate_oct26.json"
ROWS_OUT = parent.ROOT / "crypto_xperp_dynamic_regime_gate_oct26.pkl"
BASE_RULE = {
    "lookback_bars": 4,
    "style": "reversal",
    "minimum_universe": 8,
    "z_threshold": 1.0,
    "minimum_move_pct": .6,
    "target_pct": 2.0,
    "stop_pct": 1.0,
    "hold_bars": 96,
    "loss_pause": 5,
    "hourly_cap": 1,
}
TREND_LAGS = (16, 32, 96)
TREND_MAXIMA = (None, .005, .010, .020, .040)
ALIGNMENTS = ("any", "with_btc", "against_btc")
DISPERSION_MAXIMA = (None, .005, .010, .015)
COMMON_MOVE_MAXIMA = (None, .001, .002, .004)
SESSIONS = ("all", "asia", "europe", "us")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def enrich(frame: pd.DataFrame, candles: dict[str, pd.DataFrame]) -> pd.DataFrame:
    output = frame.copy()
    btc = candles["BTCUSDT"].copy()
    close = pd.to_numeric(btc.close, errors="coerce")
    signal_time = pd.to_numeric(btc.time, errors="coerce").astype("int64") + 900
    lookup = pd.DataFrame({"entry_time": signal_time})
    for lag in TREND_LAGS:
        lookup[f"btc_return_{lag}"] = close / close.shift(lag) - 1
    output = output.merge(lookup, on="entry_time", how="left", validate="many_to_one")
    output["abs_common_move"] = output.cross_mean.abs()
    output["entry_hour"] = pd.to_datetime(
        output.entry_time, unit="s", utc=True).dt.hour
    return output.dropna(subset=[f"btc_return_{lag}" for lag in TREND_LAGS])


def gate(frame: pd.DataFrame, trend_lag: int, trend_max: float | None,
         alignment: str, dispersion_max: float | None,
         common_move_max: float | None, session: str) -> pd.DataFrame:
    mask = pd.Series(True, index=frame.index)
    trend = frame[f"btc_return_{trend_lag}"]
    if trend_max is not None:
        mask &= trend.abs().le(trend_max)
    if alignment == "with_btc":
        mask &= (frame.direction_n * trend).gt(0)
    elif alignment == "against_btc":
        mask &= (frame.direction_n * trend).lt(0)
    if dispersion_max is not None:
        mask &= frame.cross_std.le(dispersion_max)
    if common_move_max is not None:
        mask &= frame.abs_common_move.le(common_move_max)
    if session == "asia":
        mask &= frame.entry_hour.lt(8)
    elif session == "europe":
        mask &= frame.entry_hour.ge(8) & frame.entry_hour.lt(16)
    elif session == "us":
        mask &= frame.entry_hour.ge(16)
    return frame[mask].copy()


def stable(metrics: dict, minimum_n: int) -> bool:
    return bool(metrics["n"] >= minimum_n
                and metrics["return_pct"] >= 0
                and metrics["negative_months"] == 0
                and metrics["dd_pct"] <= 10
                and metrics["trades_per_calendar_day"] >= 3)


def rank_key(item: dict) -> tuple:
    periods = (item["development"], item["august"], item["september"])
    stable_all = (stable(periods[0], 36)
                  and stable(periods[1], 90)
                  and stable(periods[2], 90))
    return (
        stable_all and all(value["return_pct"] >= 30 for value in periods[1:]),
        stable_all,
        min(float(value["return_pct"]) for value in periods),
        sum(float(value["return_pct"]) for value in periods),
        -max(float(value["dd_pct"]) for value in periods),
        sum(int(value["n"]) for value in periods),
    )


def main() -> None:
    candles, _ = base.load_universe()
    old_cache = parent.rails.DIRECT_CACHE
    parent.rails.DIRECT_CACHE = base.MATERIAL
    try:
        features = prior.feature_rows(candles, BASE_RULE["lookback_bars"],
                                      BASE_RULE["style"])
        labeled = geometry_audit.relabel(
            features, (BASE_RULE["target_pct"] / 100,
                       BASE_RULE["stop_pct"] / 100,
                       BASE_RULE["hold_bars"]), direct=True)
        entries = prior.select_entries(
            labeled, BASE_RULE["minimum_universe"], BASE_RULE["z_threshold"],
            BASE_RULE["minimum_move_pct"] / 100)
        entries = enrich(entries, candles)
        candidates = []
        grid = itertools.product(
            TREND_LAGS, TREND_MAXIMA, ALIGNMENTS, DISPERSION_MAXIMA,
            COMMON_MOVE_MAXIMA, SESSIONS)
        for trend_lag, trend_max, alignment, dispersion_max, common_max, session in grid:
            filtered = gate(entries, trend_lag, trend_max, alignment,
                            dispersion_max, common_max, session)
            values = {}
            for name, bounds in (("development", base.DEVELOPMENT),
                                 ("august", base.SELECTION),
                                 ("september", base.VALIDATION)):
                values[name], _ = base.evaluate(
                    filtered, bounds, BASE_RULE["loss_pause"],
                    BASE_RULE["hourly_cap"])
            candidates.append({
                "trend_lag_bars": trend_lag,
                "maximum_abs_btc_trend_pct": (
                    None if trend_max is None else 100 * trend_max),
                "alignment": alignment,
                "maximum_cross_dispersion_pct": (
                    None if dispersion_max is None else 100 * dispersion_max),
                "maximum_abs_common_move_pct": (
                    None if common_max is None else 100 * common_max),
                "session": session,
                **values,
            })
        candidates.sort(key=rank_key, reverse=True)
        selected = candidates[0]
        frozen = gate(
            entries, int(selected["trend_lag_bars"]),
            (None if selected["maximum_abs_btc_trend_pct"] is None
             else selected["maximum_abs_btc_trend_pct"] / 100),
            str(selected["alignment"]),
            (None if selected["maximum_cross_dispersion_pct"] is None
             else selected["maximum_cross_dispersion_pct"] / 100),
            (None if selected["maximum_abs_common_move_pct"] is None
             else selected["maximum_abs_common_move_pct"] / 100),
            str(selected["session"]))
        october = {}
        output_rows = []
        for margin in (.03, .05):
            key = f"margin_{round(100 * margin)}pct"
            october[key], rows = base.evaluate(
                frozen, base.HOLDOUT, BASE_RULE["loss_pause"],
                BASE_RULE["hourly_cap"], margin)
            if margin == .03:
                output_rows.extend(rows)
        pd.DataFrame(output_rows).to_pickle(ROWS_OUT)
    finally:
        parent.rails.DIRECT_CACHE = old_cache

    three = october["margin_3pct"]
    objective = bool(
        all(value["return_pct"] >= 30
            for value in (selected["august"], selected["september"]))
        and three["economic_gate"]["projected_monthly_return_pct"] >= 30
        and three["dd_pct"] <= 10
        and three["trades_per_calendar_day"] >= 3)
    report = {
        "status": ("XPERP_CAUSAL_REGIME_GATE_SHADOW_CANDIDATE"
                   if objective else "REJECTED_OBJECTIVE"),
        "ready_for_live_orders": False,
        "objective_pass": objective,
        "base_rule": BASE_RULE,
        "protocol": {
            "candidate_count": len(candidates),
            "features": [
                "closed BTC trend", "closed cross-sectional dispersion",
                "closed common market move", "UTC session"],
            "development_and_selection": [list(base.DEVELOPMENT),
                                          list(base.SELECTION),
                                          list(base.VALIDATION)],
            "holdout": list(base.HOLDOUT),
            "entry": "next 15-minute market open",
            "orders": "market only",
        },
        "selected_before_october": selected,
        "top20_before_october": candidates[:20],
        "october_partial_holdout": october,
        "sources": {
            str(base.MANIFEST): digest(base.MANIFEST),
            str(Path(__file__).resolve()): digest(Path(__file__).resolve()),
        },
        "limitations": [
            "The base rule and gate both use July-September development data.",
            "October is the only holdout and ends on October 24.",
            "Public candles cannot reproduce order-book slippage or latency.",
            "Gross results exclude fees at user request.",
        ],
    }
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True),
                   encoding="utf-8")
    print(json.dumps({
        "status": report["status"],
        "selected": selected,
        "october_partial_holdout": october,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
