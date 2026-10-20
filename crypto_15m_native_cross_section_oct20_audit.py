"""Long-history crypto cross-sectional reversal/momentum with material exits.

Signals use only closed 15-minute bars.  The next 15-minute open is a market
entry, the literal target is +2% and the stop is -1%, with pessimistic
stop-first ordering.  Policy selection ends in 2023; 2024 and 2025-2026 are
revealed only after the policy is frozen.
"""
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
OUT = ROOT / "crypto_15m_native_cross_section_oct20.json"
ROWS_OUT = ROOT / "crypto_15m_native_cross_section_oct20.pkl"
LOOKBACKS = (4, 8, 16, 32)
Z_THRESHOLDS = (1.0, 1.5, 2.0, 2.5)
MINIMUM_MOVES = (0.004, 0.008, 0.012)
PICKS = (1, 2)
STYLES = ("reversion", "momentum")
LOSS_PAUSES = (2, 3, 4)
HOURLY_CAPS = (2, 3, 4)
TARGET_PCT = 2.0
STOP_PCT = 1.0
HOLD_BARS = 64
SYMBOLS = ("AAVEUSDT", "AVAXUSDT", "BTCUSDT", "ETHUSDT",
           "LINKUSDT", "SOLUSDT", "XLMUSDT", "XRPUSDT")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def resample_15m(candle: dict) -> pd.DataFrame:
    frame = pd.DataFrame({
        "time": np.asarray(candle["time"], dtype=np.int64),
        "open": np.asarray(candle["open"], dtype=float),
        "high": np.asarray(candle["high"], dtype=float),
        "low": np.asarray(candle["low"], dtype=float),
        "close": np.asarray(candle["close"], dtype=float),
    })
    frame["bucket"] = frame.time // 900 * 900
    grouped = frame.groupby("bucket", sort=True)
    result = grouped.agg(
        time=("bucket", "first"), open=("open", "first"),
        high=("high", "max"), low=("low", "min"), close=("close", "last"),
        bars=("time", "count"), first=("time", "min"), last=("time", "max"),
    ).reset_index(drop=True)
    result = result[
        result.bars.eq(3) & (result["last"] - result["first"]).eq(600)]
    return result[["time", "open", "high", "low", "close"]].reset_index(drop=True)


@nb.njit(cache=True)
def resolve_fixed(open_, high, low, close, times, indices, directions,
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
            if bar > first and times[bar] - times[bar - 1] != 900:
                break
            if (not np.isfinite(open_[bar]) or not np.isfinite(high[bar])
                    or not np.isfinite(low[bar]) or not np.isfinite(close[bar])):
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
        if last < first or not np.isfinite(close[last]):
            valid[pos] = 0
            result = 0.0
            last = first
        elif not resolved:
            result = direction * (close[last] - entry) / entry
            result = max(-stop_fraction, min(target_fraction, result))
            code = 0
        if first + 1 >= len(times):
            valid[pos] = 0
        signed[pos] = result
        exits[pos] = times[last] + 900
        codes[pos] = code
    return signed, exits, codes, valid


def aligned_frames(candles: dict[str, dict]) -> tuple[dict[str, pd.DataFrame], np.ndarray]:
    frames = {symbol: resample_15m(candles[symbol]) for symbol in SYMBOLS}
    times = sorted(set().union(*(set(frame.time.astype(int)) for frame in frames.values())))
    index = np.asarray(times, dtype=np.int64)
    aligned = {}
    for symbol, frame in frames.items():
        aligned[symbol] = frame.set_index("time").reindex(index)
    return aligned, index


def candidate_entries(frames: dict[str, pd.DataFrame], times: np.ndarray,
                      lookback: int, z_threshold: float, minimum_move: float,
                      picks: int, style: str) -> pd.DataFrame:
    closes = np.column_stack([frames[symbol].close.to_numpy(float)
                              for symbol in SYMBOLS])
    opens = np.column_stack([frames[symbol].open.to_numpy(float)
                             for symbol in SYMBOLS])
    returns = closes / np.roll(closes, lookback, axis=0) - 1.0
    returns[:lookback] = np.nan
    complete = (pd.DataFrame(np.isfinite(closes)).rolling(
        lookback + 1, min_periods=lookback + 1).sum().to_numpy()
        == lookback + 1)
    returns[~complete] = np.nan
    contiguous = np.zeros(len(times), dtype=bool)
    contiguous[lookback:] = (
        times[lookback:] - times[:-lookback] == lookback * 900)
    returns[~contiguous] = np.nan
    rows = []
    for index in range(lookback, len(times) - 1):
        if times[index + 1] - times[index] != 900:
            continue
        values = returns[index]
        available = (np.isfinite(values) & np.isfinite(opens[index + 1])
                     & (opens[index + 1] > 0))
        if available.sum() < 6:
            continue
        mean = float(values[available].mean())
        std = float(values[available].std())
        if not np.isfinite(std) or std <= 0:
            continue
        z = (values - mean) / std
        ranked = np.flatnonzero(available)
        ranked = ranked[np.argsort(z[ranked])]
        selected = []
        for side_rank, symbol_index in enumerate(ranked[:picks]):
            if z[symbol_index] <= -z_threshold and abs(values[symbol_index]) >= minimum_move:
                selected.append((symbol_index, 1 if style == "reversion" else -1,
                                 side_rank))
        for side_rank, symbol_index in enumerate(ranked[-picks:][::-1]):
            if z[symbol_index] >= z_threshold and abs(values[symbol_index]) >= minimum_move:
                selected.append((symbol_index, -1 if style == "reversion" else 1,
                                 side_rank))
        selected.sort(key=lambda item: (-abs(float(z[item[0]])), SYMBOLS[item[0]]))
        for priority, (symbol_index, direction, side_rank) in enumerate(selected):
            rows.append({
                "symbol": SYMBOLS[symbol_index], "entry_index": index + 1,
                "entry_time": int(times[index + 1]), "direction_n": direction,
                "direction": "LONG" if direction > 0 else "SHORT",
                "cross_z": float(z[symbol_index]),
                "signal_return": float(values[symbol_index]),
                "universe_n": int(available.sum()),
                "side_rank": int(side_rank),
                "module_priority": 0,
                "source_priority": priority,
            })
    return pd.DataFrame(rows)


def add_outcomes(entries: pd.DataFrame, frames: dict[str, pd.DataFrame],
                 times: np.ndarray) -> list[dict]:
    output = []
    for symbol, part in entries.groupby("symbol", sort=False):
        frame = frames[symbol]
        values, exits, codes, valid = resolve_fixed(
            frame.open.to_numpy(float), frame.high.to_numpy(float),
            frame.low.to_numpy(float), frame.close.to_numpy(float), times,
            part.entry_index.to_numpy(np.int64),
            part.direction_n.to_numpy(np.int8), TARGET_PCT / 100,
            STOP_PCT / 100, HOLD_BARS)
        for row, signed, exit_time, code, keep in zip(
                part.to_dict("records"), values, exits, codes, valid):
            if not keep:
                continue
            output.append({
                **row, "signed_price_return": float(signed),
                "net_r": float(signed) / (STOP_PCT / 100),
                "risk_pct": STOP_PCT / 100,
                "exit_time": int(exit_time), "outcome_code": int(code),
                "outcome": "SL" if code == -1 else "TP" if code == 1 else "EXPIRED",
                "module": "native_cross_section_15m", "source": "native_cross_section_15m",
            })
    return output


def years(rows: list[dict], wanted: tuple[int, ...]) -> list[dict]:
    allowed = set(wanted)
    return [row for row in rows if pd.Timestamp(
        float(row["entry_time"]), unit="s", tz="UTC").year in allowed]


def evaluate(rows: list[dict], wanted: tuple[int, ...], loss_pause: int,
             hourly_cap: int, margin: float) -> tuple[dict, list[dict]]:
    accepted = runtime_gate(
        years(rows, wanted), loss_pause=loss_pause, hourly_cap=hourly_cap)
    result = money(accepted, margin)
    if accepted:
        result.update({
            "mean_signed_price_return_pct": round(100 * float(np.mean([
                row["signed_price_return"] for row in accepted])), 6),
            "literal_tp_rate_pct": round(100 * sum(
                int(row["outcome_code"]) == 1 for row in accepted) / len(accepted), 4),
        })
    else:
        result.update({"mean_signed_price_return_pct": 0.0,
                       "literal_tp_rate_pct": 0.0})
    return result, accepted


def month_floor(value: dict) -> float:
    months = list(value.get("monthly_return_pct", {}).values())
    return min(map(float, months)) if months else -999.0


def filtered_rows(rows: list[dict], z_threshold: float,
                  minimum_move: float, picks: int) -> list[dict]:
    return [row for row in rows
            if abs(float(row["cross_z"])) >= z_threshold
            and abs(float(row["signal_return"])) >= minimum_move
            and int(row["side_rank"]) < picks]


def development_admissible(selection: dict, confirmation: dict) -> bool:
    return bool(
        selection["n"] >= 300 and confirmation["n"] >= 100
        and selection["return_pct"] > 0 and confirmation["return_pct"] > 0
        and selection["dd_pct"] <= 12 and confirmation["dd_pct"] <= 8
        and selection["trades_per_calendar_day"] >= 3
        and confirmation["trades_per_calendar_day"] >= 3)


def development_sort_key(item: dict) -> tuple:
    return (
        item["admissible"], item["development_floor_month_pct"],
        -max(item["selection_2020_2022"]["dd_pct"],
             item["confirmation_2023"]["dd_pct"]),
        item["selection_2020_2022"]["return_pct"]
        + item["confirmation_2023"]["return_pct"],
    )


def main() -> None:
    candles, sources = load()
    frames, times = aligned_frames(candles)
    base_cache = {}
    print("building eight exact outcome caches", flush=True)
    for lookback, style in itertools.product(LOOKBACKS, STYLES):
        entries = candidate_entries(
            frames, times, lookback, min(Z_THRESHOLDS), min(MINIMUM_MOVES),
            max(PICKS), style)
        base_cache[(lookback, style)] = add_outcomes(entries, frames, times)
        print(f"cache {lookback=} {style=} rows={len(entries)}", flush=True)

    signal_candidates = []
    print("stage one: signal family screening with fixed rails", flush=True)
    for lookback, z_threshold, minimum_move, picks, style in itertools.product(
            LOOKBACKS, Z_THRESHOLDS, MINIMUM_MOVES, PICKS, STYLES):
        rows = filtered_rows(base_cache[(lookback, style)], z_threshold,
                             minimum_move, picks)
        if not rows:
            continue
        selection, _ = evaluate(rows, (2020, 2021, 2022), 3, 3, 0.03)
        confirmation, _ = evaluate(rows, (2023,), 3, 3, 0.03)
        signal_candidates.append({
            "lookback_bars": lookback, "z_threshold": z_threshold,
            "minimum_move_pct": 100 * minimum_move, "picks": picks,
            "style": style, "loss_pause": 3, "hourly_cap": 3,
            "selection_2020_2022": selection,
            "confirmation_2023": confirmation,
            "development_floor_month_pct": round(min(
                month_floor(selection), month_floor(confirmation)), 4),
            "admissible": development_admissible(selection, confirmation),
        })
    signal_candidates.sort(key=development_sort_key, reverse=True)

    # Rails are tuned only after the signal definition shortlist is frozen.
    candidates = []
    shortlist = signal_candidates[:24]
    print(f"stage two: rail search for {len(shortlist)} signal definitions",
          flush=True)
    for signal in shortlist:
        lookback = int(signal["lookback_bars"])
        style = str(signal["style"])
        z_threshold = float(signal["z_threshold"])
        minimum_move = float(signal["minimum_move_pct"]) / 100
        picks = int(signal["picks"])
        rows = filtered_rows(base_cache[(lookback, style)], z_threshold,
                             minimum_move, picks)
        for loss_pause, hourly_cap in itertools.product(LOSS_PAUSES, HOURLY_CAPS):
            selection, _ = evaluate(rows, (2020, 2021, 2022),
                                    loss_pause, hourly_cap, 0.03)
            confirmation, _ = evaluate(rows, (2023,), loss_pause,
                                       hourly_cap, 0.03)
            candidates.append({
                "lookback_bars": lookback, "z_threshold": z_threshold,
                "minimum_move_pct": 100 * minimum_move, "picks": picks,
                "style": style, "loss_pause": loss_pause,
                "hourly_cap": hourly_cap, "selection_2020_2022": selection,
                "confirmation_2023": confirmation,
                "development_floor_month_pct": round(min(
                    month_floor(selection), month_floor(confirmation)), 4),
                "admissible": development_admissible(selection, confirmation),
            })
    candidates.sort(key=development_sort_key, reverse=True)
    selected = candidates[0]
    selected_rows = filtered_rows(
        base_cache[(int(selected["lookback_bars"]), str(selected["style"]))],
        float(selected["z_threshold"]),
        float(selected["minimum_move_pct"]) / 100, int(selected["picks"]))
    loss_pause = int(selected["loss_pause"])
    hourly_cap = int(selected["hourly_cap"])
    holdouts = {}
    saved_rows = []
    for margin in (0.03, 0.05):
        name = f"margin_{int(100 * margin)}pct"
        validation, rows_2024 = evaluate(
            selected_rows, (2024,), loss_pause, hourly_cap, margin)
        hidden, rows_hidden = evaluate(
            selected_rows, (2025, 2026), loss_pause, hourly_cap, margin)
        holdouts[name] = {"validation_2024": validation,
                          "hidden_2025_2026": hidden}
        if margin == 0.03:
            saved_rows = rows_2024 + rows_hidden
    pd.DataFrame(saved_rows).to_pickle(ROWS_OUT)
    public = {key: value for key, value in selected.items()
              if not key.startswith("_")}
    three = holdouts["margin_3pct"]
    objective = bool(
        selected["admissible"] and selected["development_floor_month_pct"] >= 30
        and all(value["negative_months"] == 0
                and value["trades_per_calendar_day"] >= 3
                for value in three.values())
        and all(month >= 30 for value in three.values()
                for month in value.get("monthly_return_pct", {}).values()))
    report = {
        "status": ("NATIVE_CROSS_SECTION_SHADOW_CANDIDATE" if objective
                   else "REJECTED_OBJECTIVE"),
        "ready_for_live_orders": False,
        "protocol": {
            "selection": "2020-2022", "confirmation": "2023",
            "validation": "2024", "hidden": [2025, 2026],
            "search": "192 signal definitions on fixed rails, then 9 rail variants for the top 24; all ranking ends in 2023",
            "entry": "next 15-minute market open", "orders": "market only",
            "exit": "+2% target, -1% stop, 64 bars, stop first",
            "money": "client 3% or 5% current-equity margin; fixed 10x",
        },
        "candidate_count": len(candidates),
        "signal_candidate_count": len(signal_candidates),
        "rail_shortlist_count": len(shortlist),
        "admissible_count": sum(item["admissible"] for item in candidates),
        "selected_before_holdouts": public,
        "top20_before_holdouts": [
            {key: value for key, value in item.items() if not key.startswith("_")}
            for item in candidates[:20]],
        "holdouts": holdouts,
        "objective_pass": objective,
        "sources": {**sources,
                    str(Path(__file__).resolve()): digest(Path(__file__).resolve())},
        "limitations": [
            "Binance spot history is structural evidence; direct X-Perp transfer is still required.",
            "The bounded policy grid creates selection risk.",
            "Gross results exclude fees at user request.",
        ],
    }
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({
        "status": report["status"], "candidate_count": len(candidates),
        "admissible_count": report["admissible_count"],
        "selected": public, "holdouts": holdouts,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
