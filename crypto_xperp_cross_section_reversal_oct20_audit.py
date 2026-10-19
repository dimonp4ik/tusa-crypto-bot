"""Deterministic 15-minute X-Perp cross-sectional reversal audit.

The rule compares synchronized closed returns across the five continuous
native symbols and enters the relative extreme at the next market open.  May
selects the fixed rule, June-July confirm it, and August-October are holdout
diagnostics.  There is no fitted statistical model.
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
from crypto_runtime_shadow_portfolio_oct3_audit import money


OUT = parent.ROOT / "crypto_xperp_cross_section_reversal_oct20.json"
ROWS_OUT = parent.ROOT / "crypto_xperp_cross_section_reversal_oct20.pkl"
LOOKBACKS = (1, 2, 4, 8)
STYLES = ("reversal", "momentum")
Z_THRESHOLDS = (1.0, 1.25, 1.5)
MOVE_THRESHOLDS = (.001, .002, .004)
MINIMUM_UNIVERSES = (3, 4, 5)
SELECTION_MONTH = "2026-05"
CONFIRMATION_MONTHS = ("2026-06", "2026-07")
HOLDOUT_MONTHS = ("2026-08", "2026-09", "2026-10")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def feature_rows(candles: dict[str, dict], lookback: int,
                 style: str = "reversal") -> pd.DataFrame:
    if style not in STYLES:
        raise ValueError(f"unknown cross-sectional style: {style}")
    rows = []
    for symbol, values in candles.items():
        times = np.asarray(values["time"], dtype=np.int64)
        close = np.asarray(values["close"], dtype=float)
        for index in range(lookback, len(times) - 1):
            if times[index + 1] - times[index] != 900:
                continue
            move = close[index] / close[index - lookback] - 1
            rows.append({
                "period": "xperp", "symbol": symbol,
                "entry_time": int(times[index + 1]),
                "signal_return": move,
            })
    frame = pd.DataFrame(rows)
    grouped = frame.groupby("entry_time", sort=False).signal_return
    frame["universe_n"] = grouped.transform("size")
    frame["cross_mean"] = grouped.transform("mean")
    frame["cross_std"] = grouped.transform("std")
    frame["cross_z"] = ((frame.signal_return - frame.cross_mean)
                        / frame.cross_std.replace(0, np.nan))
    multiplier = -1 if style == "reversal" else 1
    frame["direction_n"] = (
        multiplier * np.sign(frame.cross_z).fillna(0).astype(np.int8))
    frame["direction"] = np.where(
        frame.direction_n > 0, "LONG", "SHORT")
    frame["model_quality"] = (
        frame.cross_z.abs() + 10 * frame.signal_return.abs())
    frame["month"] = pd.to_datetime(
        frame.entry_time, unit="s", utc=True).dt.strftime("%Y-%m")
    return frame.replace([np.inf, -np.inf], np.nan).dropna()


def select_entries(frame: pd.DataFrame, minimum_universe: int,
                   z_threshold: float,
                   move_threshold: float) -> pd.DataFrame:
    return frame[
        frame.universe_n.ge(minimum_universe)
        & frame.cross_z.abs().ge(z_threshold)
        & frame.signal_return.abs().ge(move_threshold)
        & frame.direction_n.ne(0)
    ].copy()


def period_metrics(frame: pd.DataFrame, month: str, loss_pause: int,
                   hourly_cap: int, margin: float = .03) -> tuple[dict,
                                                                  list[dict]]:
    found = frame[frame.month.eq(month)].copy()
    return parent.evaluate(
        found, float("-inf"), loss_pause, hourly_cap, margin)


def period_ok(value: dict) -> bool:
    return bool(value["return_pct"] >= 0
                and value["negative_months"] == 0
                and value["dd_pct"] <= 10
                and value["trades_per_calendar_day"] >= 3)


def rank_key(item: dict) -> tuple:
    values = [item["selection"], *item["confirmation"].values()]
    return (
        item["admissible"],
        min(float(value["return_pct"]) for value in values),
        sum(float(value["return_pct"]) for value in values),
        -max(float(value["dd_pct"]) for value in values),
    )


def main() -> None:
    candles = {}
    sources = {}
    for symbol in parent.CORE_SYMBOLS:
        path = parent.rails.DIRECT_CACHE / f"{symbol}_15min_current.pkl"
        candles[symbol] = parent.load_candles(path)
        sources[str(path)] = digest(path)

    candidates = []
    labeled_cache = {}
    for lookback, style in itertools.product(LOOKBACKS, STYLES):
        features = feature_rows(candles, lookback, style)
        for geometry in geometry_audit.GEOMETRIES:
            labeled_cache[(lookback, style, geometry)] = geometry_audit.relabel(
                features, geometry, direct=True)
        for minimum_universe, z_threshold, move_threshold, geometry in itertools.product(
                MINIMUM_UNIVERSES, Z_THRESHOLDS, MOVE_THRESHOLDS,
                geometry_audit.GEOMETRIES):
            selected_frame = select_entries(
                labeled_cache[(lookback, style, geometry)],
                minimum_universe, z_threshold, move_threshold)
            for loss_pause in parent.LOSS_PAUSES:
                for hourly_cap in parent.HOURLY_CAPS:
                    selection, _ = period_metrics(
                        selected_frame, SELECTION_MONTH,
                        loss_pause, hourly_cap)
                    confirmation = {
                        month: period_metrics(
                            selected_frame, month,
                            loss_pause, hourly_cap)[0]
                        for month in CONFIRMATION_MONTHS
                    }
                    values = [selection, *confirmation.values()]
                    candidates.append({
                        "lookback_bars": lookback,
                        "style": style,
                        "minimum_universe": minimum_universe,
                        "z_threshold": z_threshold,
                        "minimum_move_pct": 100 * move_threshold,
                        "target_pct": 100 * geometry[0],
                        "stop_pct": 100 * geometry[1],
                        "hold_bars": geometry[2],
                        "loss_pause": loss_pause,
                        "hourly_cap": hourly_cap,
                        "selection": selection,
                        "confirmation": confirmation,
                        "admissible": all(period_ok(value)
                                          for value in values),
                    })
    candidates.sort(key=rank_key, reverse=True)
    selected = candidates[0]
    geometry = (float(selected["target_pct"]) / 100,
                float(selected["stop_pct"]) / 100,
                int(selected["hold_bars"]))
    selected_frame = select_entries(
        labeled_cache[(int(selected["lookback_bars"]),
                       str(selected["style"]), geometry)],
        int(selected["minimum_universe"]),
        float(selected["z_threshold"]),
        float(selected["minimum_move_pct"]) / 100)
    loss_pause = int(selected["loss_pause"])
    hourly_cap = int(selected["hourly_cap"])
    holdouts = {}
    output_rows = []
    for margin in (.03, .05):
        name = f"margin_{round(100 * margin)}pct"
        holdouts[name] = {}
        for month in HOLDOUT_MONTHS:
            metrics, rows = period_metrics(
                selected_frame, month, loss_pause, hourly_cap, margin)
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
        "status": ("XPERP_CROSS_SECTION_SHADOW_CANDIDATE"
                   if objective else "REJECTED_OBJECTIVE"),
        "ready_for_live_orders": False,
        "protocol": {
            "symbols": list(parent.CORE_SYMBOLS),
            "minimum_universes": list(MINIMUM_UNIVERSES),
            "lookbacks_bars": list(LOOKBACKS),
            "styles": list(STYLES),
            "z_thresholds": list(Z_THRESHOLDS),
            "minimum_move_pct": [100 * value
                                 for value in MOVE_THRESHOLDS],
            "geometries": [{"target_pct": 100 * value[0],
                             "stop_pct": 100 * value[1],
                             "hold_bars": value[2]}
                            for value in geometry_audit.GEOMETRIES],
            "selection": SELECTION_MONTH,
            "confirmation": list(CONFIRMATION_MONTHS),
            "holdouts": list(HOLDOUT_MONTHS),
            "entry": "next 15-minute market open",
            "orders": "market only",
            "money": "client 3% or 5% current-equity margin; fixed 10x",
            "model": "none; deterministic relative-extreme direction",
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
        "selected": selected,
        "holdouts": holdouts,
        "objective_pass": objective,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
