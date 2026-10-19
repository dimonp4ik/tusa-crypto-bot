"""Long-history hourly cross-sectional audit with literal market exits.

Five Binance spot symbols with uninterrupted 2020-2026 archives form a
coverage-selected universe.  Signals use the last closed five-minute candle
of each hour and enter at the next five-minute market open.  2020-2022 rank
the grid, 2023-2024 select among the frozen finalists, and 2025-2026 are
revealed only after the rule is fixed.
"""
from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

import crypto_raw_hourly_direction_oct19_audit as portfolio
from src import research_crypto_replay as replay


ROOT = Path("reports/audit_2026_09_08")
HISTORY = ROOT / "crypto_5m_history"
OUT = ROOT / "crypto_long_history_hourly_cross_section_oct20.json"
ROWS_OUT = ROOT / "crypto_long_history_hourly_cross_section_oct20.pkl"
SYMBOLS = ("BTCUSDT", "ETHUSDT", "LINKUSDT", "XLMUSDT", "XRPUSDT")
LOOKBACK_HOURS = (1, 4, 12)
STYLES = ("reversal", "momentum")
Z_THRESHOLDS = (1.0, 1.5)
MOVE_THRESHOLDS = (.002, .004, .008)
# target %, stop %, maximum five-minute bars
GEOMETRIES = ((.5, .5, 144), (.75, .5, 288),
              (1.0, .5, 576), (1.5, .75, 576))
LOSS_PAUSES = (3, 5)
HOURLY_CAPS = (1, 2)
DEVELOPMENT_YEARS = (2020, 2021, 2022)
CONFIRMATION_YEARS = (2023, 2024)
HIDDEN_YEARS = (2025, 2026)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load() -> tuple[dict[str, dict], dict[str, str]]:
    candles = {}
    sources = {}
    for symbol in SYMBOLS:
        path = HISTORY / f"{symbol}_5min_2020_2026.pkl"
        candles[symbol] = pd.read_pickle(path)
        sources[str(path)] = digest(path)
    return candles, sources


def feature_rows(candles: dict[str, dict], lookback_hours: int,
                 style: str) -> pd.DataFrame:
    """Build synchronized cross-sectional features from closed bars only."""
    if style not in STYLES:
        raise ValueError(f"unknown style: {style}")
    pieces = []
    lag = lookback_hours * 12
    for symbol, raw in candles.items():
        times = np.asarray(raw["time"], dtype=np.int64)
        close = np.asarray(raw["close"], dtype=float)
        indices = np.arange(lag, len(times) - 1, dtype=np.int64)
        # The signal bar closes exactly when the next entry bar opens hourly.
        mask = ((times[indices] + 300) % 3600 == 0)
        mask &= times[indices + 1] - times[indices] == 300
        indices = indices[mask]
        pieces.append(pd.DataFrame({
            "symbol": symbol,
            "entry_index": indices + 1,
            "entry_time": times[indices + 1],
            "signal_return": close[indices] / close[indices - lag] - 1,
        }))
    frame = pd.concat(pieces, ignore_index=True)
    grouped = frame.groupby("entry_time", sort=False).signal_return
    frame["universe_n"] = grouped.transform("size")
    frame["cross_mean"] = grouped.transform("mean")
    frame["cross_std"] = grouped.transform("std")
    frame["cross_z"] = ((frame.signal_return - frame.cross_mean)
                        / frame.cross_std.replace(0, np.nan))
    multiplier = -1 if style == "reversal" else 1
    frame["direction_n"] = (
        multiplier * np.sign(frame.cross_z).fillna(0).astype(np.int8))
    frame["direction"] = np.where(frame.direction_n > 0, "LONG", "SHORT")
    frame["model_quality"] = (
        frame.cross_z.abs() + 10 * frame.signal_return.abs())
    stamp = pd.to_datetime(frame.entry_time, unit="s", utc=True)
    frame["year"] = stamp.dt.year
    frame["month"] = stamp.dt.strftime("%Y-%m")
    return frame.replace([np.inf, -np.inf], np.nan).dropna()


def label(frame: pd.DataFrame, candles: dict[str, dict],
          geometry: tuple[float, float, int]) -> pd.DataFrame:
    target, stop, hold = geometry
    replay_frame = frame.copy()
    replay_frame["direction_name"] = replay_frame["direction"]
    replay_frame["direction"] = replay_frame["direction_n"]
    result = replay.add_outcomes(replay_frame, candles, target, stop, hold)
    result["direction"] = result.pop("direction_name")
    result["outcome"] = np.select(
        [result.outcome_code.eq(2), result.outcome_code.eq(1)],
        ["TP", "TIME"], default="SL")
    return result


def entries(frame: pd.DataFrame, z_threshold: float,
            move_threshold: float) -> pd.DataFrame:
    return frame[
        frame.universe_n.eq(len(SYMBOLS))
        & frame.cross_z.abs().ge(z_threshold)
        & frame.signal_return.abs().ge(move_threshold)
        & frame.direction_n.ne(0)
    ].copy()


def stage(frame: pd.DataFrame, years: tuple[int, ...], loss_pause: int,
          hourly_cap: int, margin: float = .03) -> tuple[dict, list[dict]]:
    part = frame[frame.year.isin(years)].copy()
    return portfolio.evaluate(
        part, float("-inf"), loss_pause, hourly_cap, margin)


def month_floor(value: dict) -> float:
    monthly = value.get("monthly_return_pct", {})
    return min((float(item) for item in monthly.values()), default=-999.0)


def rank(value: dict) -> tuple:
    return (
        value["dd_pct"] <= 15 and value["trades_per_calendar_day"] >= 3,
        -int(value["negative_months"]), month_floor(value),
        float(value["return_pct"]), -float(value["dd_pct"]),
    )


def public(item: dict) -> dict:
    return {key: value for key, value in item.items()
            if not key.startswith("_")}


def main() -> None:
    candles, sources = load()
    candidates = []
    cache = {}
    combinations = list(itertools.product(LOOKBACK_HOURS, STYLES))
    for group_index, (lookback, style) in enumerate(combinations, 1):
        features = feature_rows(candles, lookback, style)
        for geometry in GEOMETRIES:
            labeled = label(features, candles, geometry)
            cache[(lookback, style, geometry)] = labeled
            for z_threshold, move_threshold in itertools.product(
                    Z_THRESHOLDS, MOVE_THRESHOLDS):
                chosen = entries(labeled, z_threshold, move_threshold)
                for loss_pause, hourly_cap in itertools.product(
                        LOSS_PAUSES, HOURLY_CAPS):
                    development, _ = stage(
                        chosen, DEVELOPMENT_YEARS, loss_pause, hourly_cap)
                    candidates.append({
                        "lookback_hours": lookback, "style": style,
                        "z_threshold": z_threshold,
                        "minimum_move_pct": 100 * move_threshold,
                        "target_pct": geometry[0], "stop_pct": geometry[1],
                        "hold_bars_5m": geometry[2],
                        "loss_pause": loss_pause, "hourly_cap": hourly_cap,
                        "development_2020_2022_3pct": development,
                    })
        print(json.dumps({"phase": "development",
                          "done": group_index,
                          "total": len(combinations)}), flush=True)

    candidates.sort(
        key=lambda item: rank(item["development_2020_2022_3pct"]),
        reverse=True)
    diverse = {}
    for item in candidates:
        family_key = (
            int(item["lookback_hours"]), str(item["style"]),
            float(item["target_pct"]), float(item["stop_pct"]),
            int(item["hold_bars_5m"]),
        )
        diverse.setdefault(family_key, item)
    finalists = []
    for item in candidates[:20]:
        geometry = (float(item["target_pct"]), float(item["stop_pct"]),
                    int(item["hold_bars_5m"]))
        chosen = entries(
            cache[(int(item["lookback_hours"]), str(item["style"]), geometry)],
            float(item["z_threshold"]),
            float(item["minimum_move_pct"]) / 100)
        confirmation, _ = stage(
            chosen, CONFIRMATION_YEARS, int(item["loss_pause"]),
            int(item["hourly_cap"]))
        finalists.append({**item,
                           "confirmation_2023_2024_3pct": confirmation,
                           "_chosen": chosen})
    finalists.sort(key=lambda item: (
        *rank(item["confirmation_2023_2024_3pct"]),
        *rank(item["development_2020_2022_3pct"])), reverse=True)
    selected = finalists[0]
    chosen = selected["_chosen"]
    holdouts = {}
    output_rows = []
    for margin in (.03, .05):
        metrics, rows = stage(
            chosen, HIDDEN_YEARS, int(selected["loss_pause"]),
            int(selected["hourly_cap"]), margin)
        holdouts[f"margin_{round(100 * margin)}pct"] = metrics
        if margin == .03:
            output_rows.extend(rows)
    pd.DataFrame(output_rows).to_pickle(ROWS_OUT)

    hidden = holdouts["margin_3pct"]
    objective = bool(
        hidden["negative_months"] == 0
        and month_floor(hidden) >= 30
        and hidden["dd_pct"] <= 10
        and hidden["trades_per_calendar_day"] >= 3)
    report = {
        "status": ("LONG_HISTORY_SHADOW_CANDIDATE"
                   if objective else "REJECTED_OBJECTIVE"),
        "ready_for_live_orders": False,
        "protocol": {
            "symbols": list(SYMBOLS),
            "universe_rule": "complete archive from 2020-01 before selection",
            "development": list(DEVELOPMENT_YEARS),
            "confirmation": list(CONFIRMATION_YEARS),
            "hidden_reveal": list(HIDDEN_YEARS),
            "signal": "last closed 5-minute candle before each hour",
            "entry": "next 5-minute market open",
            "orders": "market only",
            "collision": "stop first",
            "money": "client 3% or 5% current-equity margin; fixed 10x",
        },
        "candidate_count": len(candidates),
        "selected_before_hidden_reveal": public(selected),
        "development_frontier": [public(item) for item in candidates[:20]],
        "development_diverse_library": [
            public(item) for item in diverse.values()],
        "confirmation_frontier": [public(item) for item in finalists[:20]],
        "hidden_2025_2026": holdouts,
        "objective_pass": objective,
        "sources": {
            **sources,
            str(Path(__file__).resolve()): digest(Path(__file__).resolve()),
        },
        "limitations": [
            "Signals use Binance spot history and still require separate X-Perp transfer testing.",
            "The 2026 hidden period ends with the available archive in September.",
            "Gross returns exclude fees at user request.",
            "No backtest guarantees a future monthly return.",
        ],
    }
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True),
                   encoding="utf-8")
    print(json.dumps({
        "status": report["status"],
        "candidate_count": len(candidates),
        "selected": public(selected),
        "hidden": holdouts,
        "objective_pass": objective,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
