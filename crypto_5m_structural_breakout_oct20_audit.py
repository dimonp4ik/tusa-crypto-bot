"""Long-history volume-confirmed structural breakouts with 1%-2% targets."""
from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path

import numba as nb
import numpy as np
import pandas as pd

from crypto_5m_family_audit import load
from crypto_runtime_shadow_portfolio_oct3_audit import money, runtime_gate


ROOT = Path("reports/audit_2026_09_08")
OUT = ROOT / "crypto_5m_structural_breakout_oct20.json"
ROWS_OUT = ROOT / "crypto_5m_structural_breakout_oct20.pkl"
LOOKBACKS = (12, 24, 48)
VOLUME_MINS = (1.0, 1.5, 2.0)
RANGE_ATR_MINS = (.5, 1.0)
TRENDS = ("none", "ma96", "ma288")
STYLES = ("continuation", "false_breakout_fade")
TARGETS = (.01, .015, .02)
STOPS = (.005, .0075, .01)
HOLDS = (72, 144)
DEFAULT_GEOMETRY = (.015, .0075, 144)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@nb.njit(cache=True)
def resolve(open_, high, low, close, times, indices, directions,
            target_fraction, stop_fraction, max_bars):
    n = len(indices)
    signed = np.empty(n, dtype=np.float64)
    exits = np.empty(n, dtype=np.int64)
    codes = np.empty(n, dtype=np.int8)
    valid = np.ones(n, dtype=np.uint8)
    for pos in range(n):
        first = indices[pos]
        direction = directions[pos]
        entry = open_[first]
        if not np.isfinite(entry) or entry <= 0:
            valid[pos] = 0
            signed[pos] = 0.0
            exits[pos] = times[first]
            codes[pos] = 0
            continue
        target = entry * (1.0 + direction * target_fraction)
        stop = entry * (1.0 - direction * stop_fraction)
        result = 0.0
        code = 0
        last = first
        resolved = False
        end = min(len(times), first + max_bars)
        for bar in range(first, end):
            if bar > first and times[bar] - times[bar - 1] != 300:
                break
            last = bar
            stop_hit = low[bar] <= stop if direction > 0 else high[bar] >= stop
            target_hit = high[bar] >= target if direction > 0 else low[bar] <= target
            if stop_hit:
                fill = min(open_[bar], stop) if direction > 0 else max(open_[bar], stop)
                result = direction * (fill - entry) / entry
                code = -1
                resolved = True
                break
            if target_hit:
                result = target_fraction
                code = 1
                resolved = True
                break
        if not resolved:
            result = direction * (close[last] - entry) / entry
            result = max(-stop_fraction, min(target_fraction, result))
        signed[pos] = result
        exits[pos] = times[last] + 300
        codes[pos] = code
    return signed, exits, codes, valid


def base_entries(candles: dict[str, dict], lookback: int) -> pd.DataFrame:
    pieces = []
    for symbol, candle in candles.items():
        frame = pd.DataFrame({key: np.asarray(candle[key]) for key in (
            "time", "open", "high", "low", "close", "volume")})
        previous = frame.close.shift(1)
        tr = pd.concat(((frame.high - frame.low),
                        (frame.high - previous).abs(),
                        (frame.low - previous).abs()), axis=1).max(axis=1)
        atr = tr.shift(1).rolling(14).mean()
        prior_high = frame.high.shift(1).rolling(lookback).max()
        prior_low = frame.low.shift(1).rolling(lookback).min()
        span = (frame.high - frame.low).replace(0, np.nan)
        direction = np.zeros(len(frame), dtype=np.int8)
        direction[(frame.close > prior_high)
                  & ((frame.close - frame.low) / span >= .7)] = 1
        direction[(frame.close < prior_low)
                  & ((frame.high - frame.close) / span >= .7)] = -1
        ma96 = frame.close.shift(1).rolling(96).mean()
        ma288 = frame.close.shift(1).rolling(288).mean()
        volume_ratio = frame.volume / frame.volume.shift(1).rolling(20).median()
        range_atr = span / atr
        closed = np.flatnonzero(
            (direction != 0) & np.isfinite(volume_ratio) & np.isfinite(range_atr)
            & (np.arange(len(frame)) + 1 < len(frame)))
        entry = closed + 1
        contiguous = (frame.time.to_numpy(np.int64)[entry]
                      - frame.time.to_numpy(np.int64)[closed] == 300)
        closed, entry = closed[contiguous], entry[contiguous]
        found = pd.DataFrame({
            "symbol": symbol, "entry_index": entry,
            "entry_time": frame.time.to_numpy(np.int64)[entry],
            "direction_n": direction[closed],
            "volume_ratio": volume_ratio.to_numpy(float)[closed],
            "range_atr": range_atr.to_numpy(float)[closed],
            "trend_ma96": (direction[closed] * (
                frame.close.to_numpy(float)[closed] - ma96.to_numpy(float)[closed]) > 0),
            "trend_ma288": (direction[closed] * (
                frame.close.to_numpy(float)[closed] - ma288.to_numpy(float)[closed]) > 0),
        })
        found["score"] = found.volume_ratio * found.range_atr
        pieces.append(found)
    entries = pd.concat(pieces, ignore_index=True)
    entries["source_priority"] = entries.groupby("entry_time").score.rank(
        method="first", ascending=False).astype(int) - 1
    entries["direction"] = np.where(entries.direction_n > 0, "LONG", "SHORT")
    entries["year"] = pd.to_datetime(entries.entry_time, unit="s", utc=True).dt.year
    return entries


def outcomes(entries: pd.DataFrame, candles: dict[str, dict], target: float,
             stop: float, hold: int) -> list[dict]:
    output = []
    for symbol, part in entries.groupby("symbol", sort=False):
        candle = candles[symbol]
        signed, exits, codes, valid = resolve(
            np.asarray(candle["open"], dtype=float),
            np.asarray(candle["high"], dtype=float),
            np.asarray(candle["low"], dtype=float),
            np.asarray(candle["close"], dtype=float),
            np.asarray(candle["time"], dtype=np.int64),
            part.entry_index.to_numpy(np.int64),
            part.direction_n.to_numpy(np.int8), target, stop, hold)
        for row, value, exit_time, code, keep in zip(
                part.to_dict("records"), signed, exits, codes, valid):
            if keep:
                output.append({
                    **row, "signed_price_return": float(value),
                    "exit_time": int(exit_time),
                    "outcome": "SL" if code == -1 else "TP" if code == 1 else "EXPIRED",
                    "outcome_code": int(code), "module": "structural_breakout_5m",
                    "source": "structural_breakout_5m", "module_priority": 0,
                })
    return output


def apply_signal(rows: list[dict], volume_min: float, range_min: float,
                 trend: str) -> list[dict]:
    return [row for row in rows
            if row["volume_ratio"] >= volume_min
            and row["range_atr"] >= range_min
            and (trend == "none" or bool(row[f"trend_{trend}"]))]


def evaluate(rows: list[dict], years: tuple[int, ...], margin: float) -> dict:
    wanted = set(years)
    accepted = runtime_gate([row for row in rows if int(row["year"]) in wanted])
    result = money(accepted, margin)
    result["literal_target_rate_pct"] = round(
        100 * sum(row["outcome_code"] == 1 for row in accepted) / len(accepted), 4
    ) if accepted else 0.0
    result["mean_signed_move_pct"] = round(
        100 * float(np.mean([row["signed_price_return"] for row in accepted])), 5
    ) if accepted else 0.0
    return result


def floor_month(periods: dict[str, dict]) -> float:
    values = [month for period in periods.values()
              for month in period.get("monthly_return_pct", {}).values()]
    return min(map(float, values), default=-999.0)


def main() -> None:
    candles, sources = load()
    entries = {}
    for lookback in LOOKBACKS:
        continuation = base_entries(candles, lookback)
        entries[(lookback, "continuation")] = continuation
        fade = continuation.copy()
        fade["direction_n"] = -fade.direction_n
        fade["direction"] = np.where(fade.direction_n > 0, "LONG", "SHORT")
        fade["trend_ma96"] = ~fade.trend_ma96
        fade["trend_ma288"] = ~fade.trend_ma288
        entries[(lookback, "false_breakout_fade")] = fade
    default_rows = {
        (lookback, style): outcomes(entries[(lookback, style)], candles,
                                    *DEFAULT_GEOMETRY)
        for lookback, style in itertools.product(LOOKBACKS, STYLES)
    }
    signals = []
    for lookback, style, volume_min, range_min, trend in itertools.product(
            LOOKBACKS, STYLES, VOLUME_MINS, RANGE_ATR_MINS, TRENDS):
        rows = apply_signal(default_rows[(lookback, style)], volume_min,
                            range_min, trend)
        periods = {
            "selection_2020_2022": evaluate(rows, (2020, 2021, 2022), .03),
            "confirmation_2023_2024": evaluate(rows, (2023, 2024), .03),
        }
        admissible = all(value["n"] >= 300 and value["return_pct"] > 0
                         and value["dd_pct"] <= 12
                         and value["trades_per_calendar_day"] >= 2.5
                         for value in periods.values())
        signals.append({
            "lookback": lookback, "style": style, "volume_min": volume_min,
            "range_atr_min": range_min, "trend": trend,
            "development": periods, "admissible": admissible,
            "floor_month_pct": round(floor_month(periods), 4),
        })
    signals.sort(key=lambda item: (
        item["admissible"], item["floor_month_pct"],
        min(v["return_pct"] for v in item["development"].values()),
        -max(v["dd_pct"] for v in item["development"].values())), reverse=True)

    shortlist = signals[:12]
    del default_rows
    candidates = []
    for lookback, style in itertools.product(LOOKBACKS, STYLES):
        matching = [signal for signal in shortlist
                    if signal["lookback"] == lookback
                    and signal["style"] == style]
        if not matching:
            continue
        for target, stop, hold in itertools.product(TARGETS, STOPS, HOLDS):
            raw = outcomes(entries[(lookback, style)], candles,
                           target, stop, hold)
            for signal in matching:
                rows = apply_signal(raw, signal["volume_min"],
                                    signal["range_atr_min"], signal["trend"])
                periods = {
                    "selection_2020_2022": evaluate(
                        rows, (2020, 2021, 2022), .03),
                    "confirmation_2023_2024": evaluate(
                        rows, (2023, 2024), .03),
                }
                admissible = all(
                    value["n"] >= 300 and value["return_pct"] > 0
                    and value["dd_pct"] <= 12
                    and value["trades_per_calendar_day"] >= 2.5
                    for value in periods.values())
                candidates.append({
                    **{key: signal[key] for key in (
                        "lookback", "style", "volume_min",
                        "range_atr_min", "trend")},
                    "target_pct": 100 * target, "stop_pct": 100 * stop,
                    "hold_bars": hold, "development": periods,
                    "admissible": admissible,
                    "floor_month_pct": round(floor_month(periods), 4),
                })
    candidates.sort(key=lambda item: (
        item["admissible"], item["floor_month_pct"],
        min(v["return_pct"] for v in item["development"].values()),
        -max(v["dd_pct"] for v in item["development"].values())), reverse=True)
    selected = candidates[0]
    selected_rows = apply_signal(
        outcomes(entries[(selected["lookback"], selected["style"])], candles,
                 selected["target_pct"] / 100, selected["stop_pct"] / 100,
                 selected["hold_bars"]),
        selected["volume_min"], selected["range_atr_min"], selected["trend"])
    holdouts = {
        f"margin_{int(100 * margin)}pct": {
            str(year): evaluate(selected_rows, (year,), margin)
            for year in (2025, 2026)
        } for margin in (.03, .05)
    }
    pd.DataFrame(selected_rows).to_pickle(ROWS_OUT)
    passed = bool(
        selected["admissible"]
        and all(value["return_pct"] > 0 and value["dd_pct"] <= 10
                and value["negative_months"] == 0
                and value["trades_per_calendar_day"] >= 3
                for value in holdouts["margin_3pct"].values()))

    def public(item):
        return {key: value for key, value in item.items() if key != "_rows"}

    report = {
        "status": ("STRUCTURAL_BREAKOUT_AWAITING_DIRECT_VENUE_FORWARD" if passed
                   else "REJECTED_OBJECTIVE"),
        "ready_for_live_orders": False,
        "protocol": {
            "signal_screen": "2020-2024 with 2020-2022/2023-2024 split",
            "holdouts": [2025, 2026], "entry": "next 5m market open",
            "exits": "literal 1%-2% target, 0.5%-1% stop, stop first",
            "styles": list(STYLES),
            "money": "client 3% or 5% current-equity margin; fixed 10x",
        },
        "signal_count": len(signals), "candidate_count": len(candidates),
        "admissible_count": sum(item["admissible"] for item in candidates),
        "selected_before_holdouts": public(selected),
        "top20_before_holdouts": [public(item) for item in candidates[:20]],
        "holdouts": holdouts, "objective_pass": passed,
        "sources": {**sources,
                    str(Path(__file__).resolve()): digest(Path(__file__).resolve())},
        "limitations": [
            "The bounded signal and exit grid creates selection risk.",
            "Binance spot history still requires direct X-Perp transfer.",
            "Gross results exclude fees at user request.",
        ],
    }
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"status": report["status"],
                      "admissible_count": report["admissible_count"],
                      "selected": public(selected),
                      "holdouts": holdouts}, indent=2), flush=True)


if __name__ == "__main__":
    main()
