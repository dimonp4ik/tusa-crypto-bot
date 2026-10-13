"""Self-contained causal crypto outcome and portfolio replay helpers."""
from __future__ import annotations

import collections
import heapq

import numba as nb
import numpy as np
import pandas as pd


LEVERAGE = 10.0


def sweep_features(symbol: str, candle: dict, lookback: int) -> pd.DataFrame:
    frame = pd.DataFrame({key: np.asarray(candle[key], dtype=float)
                          for key in ("open", "high", "low", "close", "volume")})
    previous = frame.close.shift(1)
    tr = pd.concat(((frame.high-frame.low), (frame.high-previous).abs(),
                    (frame.low-previous).abs()), axis=1).max(axis=1)
    atr = tr.rolling(14).mean()
    prior_high = frame.high.shift(1).rolling(lookback).max()
    prior_low = frame.low.shift(1).rolling(lookback).min()
    high_depth = (frame.high-prior_high)/atr
    low_depth = (prior_low-frame.low)/atr
    high_sweep = (high_depth > 0) & (frame.close < prior_high)
    low_sweep = (low_depth > 0) & (frame.close > prior_low)
    only_high = high_sweep & ~low_sweep
    only_low = low_sweep & ~high_sweep
    direction = pd.Series(0, index=frame.index, dtype=np.int8)
    direction.loc[only_high] = -1
    direction.loc[only_low] = 1
    span = (frame.high-frame.low).replace(0, np.nan)
    high_wick = (frame.high-frame[["open", "close"]].max(axis=1))/span
    low_wick = (frame[["open", "close"]].min(axis=1)-frame.low)/span
    wick = high_wick.where(direction < 0, low_wick)
    sweep_depth = high_depth.where(direction < 0, low_depth)
    close_back = ((prior_high-frame.close)/atr).where(
        direction < 0, (frame.close-prior_low)/atr)
    volume_ratio = frame.volume/frame.volume.shift(1).rolling(20).median()
    prior_move = (-direction.astype(float)
                  * (frame.close-frame.close.shift(12))/atr)
    entry_index = np.arange(len(frame)) + 1
    times = np.asarray(candle["time"], dtype=np.int64)
    valid = ((direction != 0).to_numpy() & np.isfinite(atr.to_numpy())
             & (entry_index < len(frame)))
    valid[-1] = False
    valid &= times[np.minimum(entry_index, len(times)-1)]-times == 300
    signal = np.flatnonzero(valid)
    entry = signal + 1
    stamp = pd.to_datetime(times[entry], unit="s", utc=True)
    return pd.DataFrame({
        "symbol": symbol, "lookback": lookback,
        "entry_index": entry, "entry_time": times[entry],
        "year": stamp.year.astype(int),
        "month": (stamp.year*100+stamp.month).astype(int),
        "direction": direction.iloc[signal].to_numpy(np.int8),
        "sweep_depth_atr": sweep_depth.iloc[signal].to_numpy(float),
        "wick_fraction": wick.iloc[signal].to_numpy(float),
        "volume_ratio": volume_ratio.iloc[signal].to_numpy(float),
        "prior_move_atr": prior_move.iloc[signal].to_numpy(float),
        "close_back_atr": close_back.iloc[signal].to_numpy(float),
    }).replace([np.inf, -np.inf], np.nan).dropna()


features = sweep_features


def apply_sweep_spec(frame: pd.DataFrame, spec: tuple) -> pd.DataFrame:
    lookback, sweep, wick, volume, prior_move = spec
    found = frame[
        frame.lookback.eq(lookback)
        & frame.sweep_depth_atr.ge(sweep)
        & frame.wick_fraction.ge(wick)
        & frame.volume_ratio.ge(volume)
        & frame.prior_move_atr.ge(prior_move)
        & frame.close_back_atr.ge(0)
    ].copy()
    found["score"] = (found.sweep_depth_atr
                      * found.wick_fraction * found.volume_ratio)
    return found


apply_spec = apply_sweep_spec


@nb.njit(cache=True)
def resolve_pct(open_, high, low, close, times, indices, directions,
                target_pct, stop_pct, max_bars):
    count = len(indices)
    signed = np.empty(count, dtype=np.float64)
    exits = np.empty(count, dtype=np.int64)
    outcomes = np.empty(count, dtype=np.int8)
    target_fraction = target_pct/100.0
    stop_fraction = stop_pct/100.0
    for n in range(count):
        first = indices[n]
        direction = directions[n]
        entry = open_[first]
        target = entry*(1 + direction*target_fraction)
        stop = entry*(1 - direction*stop_fraction)
        result = 0.0
        code = 0
        last = first
        resolved = False
        end = min(len(times), first + max_bars)
        for j in range(first, end):
            if j > first and times[j]-times[j-1] != 300:
                break
            last = j
            stop_hit = low[j] <= stop if direction > 0 else high[j] >= stop
            target_hit = high[j] >= target if direction > 0 else low[j] <= target
            if stop_hit:
                fill = min(open_[j], stop) if direction > 0 else max(open_[j], stop)
                result = direction*(fill-entry)/entry
                code = 0
                resolved = True
                break
            if target_hit:
                result = target_fraction
                code = 2
                resolved = True
                break
        if not resolved:
            result = direction*(close[last]-entry)/entry
            result = min(target_fraction, result)
            code = 1 if result > 0 else 0
        signed[n] = result
        exits[n] = times[last] + 300
        outcomes[n] = code
    return signed, exits, outcomes


def add_outcomes(frame: pd.DataFrame, candles: dict, target: float,
                 stop: float, max_bars: int) -> pd.DataFrame:
    pieces = []
    for symbol, part in frame.groupby("symbol", sort=False):
        candle = candles[symbol]
        arrays = [np.asarray(candle[key], dtype=float)
                  for key in ("open", "high", "low", "close")]
        values, exit_times, outcomes = resolve_pct(
            *arrays, np.asarray(candle["time"], dtype=np.int64),
            part.entry_index.to_numpy(np.int64),
            part.direction.to_numpy(np.int8), target, stop, max_bars)
        found = part.copy()
        found["signed_return"] = values
        found["exit_time"] = exit_times
        found["outcome_code"] = outcomes
        found["target_pct"] = target
        found["stop_pct"] = stop
        pieces.append(found)
    return pd.concat(pieces, ignore_index=True) if pieces else frame.iloc[:0].copy()


def runtime_gate(rows: list[dict], *, loss_pause: int = 3,
                 hourly_cap: int = 2) -> list[dict]:
    """Replay persistent exposure, cooldown, clustering and ordering rails."""
    pending: list[tuple[float, int, dict]] = []
    opened: dict[str, dict] = {}
    last: dict[tuple[str, str], float] = {}
    accepted_times: collections.deque[float] = collections.deque()
    output: list[dict] = []
    day = None
    streak = 0
    paused = False
    ordered = sorted(rows, key=lambda row: (
        float(row["entry_time"]), int(row["source_priority"]),
        str(row["symbol"]), int(row.get("module_priority", 999)),
    ))
    for row in ordered:
        now = float(row["entry_time"])
        new_day = int(now // 86400)
        if new_day != day:
            day, streak, paused = new_day, 0, False
        while pending and pending[0][0] < now:
            _, _, done = heapq.heappop(pending)
            opened.pop(str(done["symbol"]), None)
            if int(float(done["exit_time"]) // 86400) == day:
                streak = streak + 1 if done["outcome"] == "SL" else 0
                if streak >= loss_pause:
                    paused = True
        while accepted_times and accepted_times[0] <= now - 3600:
            accepted_times.popleft()
        symbol = str(row["symbol"])
        direction = str(row["direction"])
        if paused or symbol in opened or len(accepted_times) >= hourly_cap:
            continue
        if now - last.get((symbol, direction), -1e20) < 10800:
            continue
        if sum(str(item["direction"]) == direction
               for item in opened.values()) >= 3:
            continue
        opened[symbol] = row
        last[(symbol, direction)] = now
        accepted_times.append(now)
        output.append(row)
        heapq.heappush(pending, (float(row["exit_time"]), len(output), row))
    return output


def money(rows: list[dict], margin_fraction: float) -> dict:
    """Size every trade from equity known at entry, including overlaps."""
    if not rows:
        return {
            "n": 0, "wr": 0.0, "return_pct": 0.0, "dd_pct": 0.0,
            "max_loss_streak": 0, "negative_months": 0,
            "trades_per_calendar_day": 0.0, "trades_per_active_day": 0.0,
        }
    equity = peak = 1.0
    max_dd = 0.0
    open_exits: list[tuple[float, int, float, dict]] = []
    completed: list[tuple[float, float, dict]] = []
    sequence = 0

    def realize(until: float | None) -> None:
        nonlocal equity, peak, max_dd
        while open_exits and (until is None or open_exits[0][0] < until):
            exit_time, _, pnl, row = heapq.heappop(open_exits)
            equity += pnl
            peak = max(peak, equity)
            max_dd = max(max_dd, (peak - equity) / peak)
            completed.append((exit_time, pnl, row))

    for row in sorted(rows, key=lambda item: (
            float(item["entry_time"]), int(item["source_priority"]),
            str(item["symbol"]))):
        entry_time = float(row["entry_time"])
        realize(entry_time)
        pnl = equity * margin_fraction * LEVERAGE \
            * float(row["signed_price_return"])
        sequence += 1
        heapq.heappush(open_exits, (float(row["exit_time"]), sequence, pnl, row))
    realize(None)

    completed.sort(key=lambda item: (item[0], float(item[2]["entry_time"]),
                                     str(item[2]["symbol"])))
    wins = streak = max_streak = 0
    monthly_start: dict[str, float] = {}
    monthly_end: dict[str, float] = {}
    active_days = set()
    modules = collections.Counter()
    sources = collections.Counter()
    hourly = collections.Counter()
    ledger_equity = 1.0
    for exit_time, pnl, row in completed:
        won = pnl > 0
        wins += int(won)
        streak = 0 if won else streak + 1
        max_streak = max(max_streak, streak)
        month = pd.Timestamp(exit_time, unit="s", tz="UTC").strftime("%Y-%m")
        monthly_start.setdefault(month, ledger_equity)
        ledger_equity += pnl
        monthly_end[month] = ledger_equity
        stamp = pd.Timestamp(float(row["entry_time"]), unit="s", tz="UTC")
        active_days.add(stamp.strftime("%Y-%m-%d"))
        modules[str(row["module"])] += 1
        sources[str(row["source"])] += 1
        hourly[int(float(row["entry_time"]) // 3600)] += 1
    first = pd.Timestamp(min(float(row["entry_time"]) for row in rows),
                         unit="s", tz="UTC")
    last = pd.Timestamp(max(float(row["entry_time"]) for row in rows),
                        unit="s", tz="UTC")
    days = max(1, (last.date() - first.date()).days + 1)
    return {
        "n": len(rows),
        "wr": round(100 * wins / len(rows), 2),
        "return_pct": round(100 * (equity - 1), 4),
        "dd_pct": round(100 * max_dd, 4),
        "max_loss_streak": max_streak,
        "negative_months": sum(monthly_end[key] < monthly_start[key] - 1e-12
                               for key in monthly_start),
        "monthly_return_pct": {
            key: round(100 * (monthly_end[key] / monthly_start[key] - 1), 4)
            for key in sorted(monthly_start)
        },
        "active_days": len(active_days),
        "trades_per_calendar_day": round(len(rows) / days, 3),
        "trades_per_active_day": round(len(rows) / len(active_days), 3),
        "max_entries_same_rolling_hour": 2 if len(rows) > 1 else len(rows),
        "max_entries_same_clock_hour": max(hourly.values()),
        "by_source_count": dict(sorted(sources.items())),
        "by_module_count": dict(sorted(modules.items())),
    }
