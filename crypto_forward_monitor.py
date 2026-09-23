"""Replay the frozen crypto router only after the last audited entry."""
from __future__ import annotations

import argparse
import collections
import hashlib
import heapq
import json
import math
import pickle
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.backtest_integrity import simulate_exit
from src.crypto_venue_router import ROBUST_SYMBOLS, venue_setups


def metric(frame: pd.DataFrame) -> dict:
    if frame.empty:
        return {"n": 0, "wr": 0.0, "wr_lower_95": 0.0, "net_r": 0.0,
                "pf": 0.0, "dd_r": 0.0, "max_loss_streak": 0}
    values = frame.sort_values(["exit_time", "entry_time"])["net_r"].to_numpy(float)
    curve = np.cumsum(values)
    drawdown = np.maximum.accumulate(np.r_[0.0, curve])[1:] - curve
    wins, losses = values[values > 0].sum(), -values[values < 0].sum()
    count = len(values)
    win_count = int((values > 0).sum())
    rate = win_count / count
    z = 1.959963984540054
    denominator = 1 + z * z / count
    center = rate + z * z / (2 * count)
    spread = z * math.sqrt(
        rate * (1 - rate) / count + z * z / (4 * count * count))
    streak = maximum_streak = 0
    for value in values:
        streak = streak + 1 if value < 0 else 0
        maximum_streak = max(maximum_streak, streak)
    return {
        "n": int(count), "wr": float(100 * rate),
        "wr_lower_95": float(100 * (center - spread) / denominator),
        "net_r": float(values.sum()),
        "pf": float(wins / losses) if losses else None,
        "dd_r": float(drawdown.max(initial=0)),
        "max_loss_streak": maximum_streak,
    }


def _portfolio_gate(rows: list[dict], direction_cap: int) -> list[dict]:
    """Exact causal portfolio gate used by the frozen production replay."""
    closing: list[tuple[float, int, dict]] = []
    opened: dict[str, dict] = {}
    last: dict[tuple[str, str], float] = {}
    per_scan: collections.Counter[int] = collections.Counter()
    accepted = []
    day = None
    streak = 0
    paused = False
    for row in sorted(rows, key=lambda item: (item["entry_time"], item["symbol"])):
        now = row["entry_time"]
        new_day = int(now // 86400)
        if new_day != day:
            day, streak, paused = new_day, 0, False
        while closing and closing[0][0] < now:
            end, _, done = heapq.heappop(closing)
            opened.pop(done["symbol"], None)
            if int(end // 86400) == day:
                streak = streak + 1 if done["outcome"] == "SL" else 0
                if streak >= 3:
                    paused = True
        if paused or row["symbol"] in opened:
            continue
        key = (row["symbol"], row["direction"])
        if now - last.get(key, -1e20) < 10800:
            continue
        scan = int(now // 900)
        if per_scan[scan] >= 3:
            continue
        if sum(item["direction"] == row["direction"]
               for item in opened.values()) >= direction_cap:
            continue
        opened[row["symbol"]] = row
        last[key] = now
        per_scan[scan] += 1
        accepted.append(row)
        heapq.heappush(closing, (row["exit_time"], len(accepted), row))
    return accepted


def portfolio_gate_with_pending(completed: list[dict], pending: list[dict],
                                direction_cap: int = 3) -> tuple[list[dict], list[dict]]:
    """Run one causal gate over resolved and still-open candidates.

    A pending position has no known outcome and therefore stays open through
    the end of the snapshot.  It must consume the symbol and direction caps;
    gating only resolved trades can admit entries the live bot could not open.
    """
    tagged = [dict(row, _monitor_pending=False) for row in completed]
    tagged.extend(dict(row, _monitor_pending=True, exit_time=2 ** 62,
                       outcome="PENDING", net_r=0.0) for row in pending)
    accepted = _portfolio_gate(tagged, direction_cap)
    accepted_completed = []
    accepted_pending = []
    for row in accepted:
        clean = {key: value for key, value in row.items()
                 if key != "_monitor_pending"}
        if row["_monitor_pending"]:
            for key in ("exit_time", "outcome", "net_r"):
                clean.pop(key, None)
            accepted_pending.append(clean)
        else:
            accepted_completed.append(clean)
    return accepted_completed, accepted_pending


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-signals", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    reference_raw = args.reference_signals.read_bytes()
    reference = pd.read_csv(args.reference_signals)
    previous_max = int(reference["entry_time"].max())
    hashes = {str(args.reference_signals): hashlib.sha256(reference_raw).hexdigest()}
    candles = {}
    for symbol in sorted(ROBUST_SYMBOLS):
        path = args.cache_dir / f"{symbol}_15min_current.pkl"
        raw = path.read_bytes()
        hashes[str(path)] = hashlib.sha256(raw).hexdigest()
        series = pickle.loads(raw)
        times = list(map(int, series["time"]))
        if times != sorted(times) or len(times) != len(set(times)):
            raise ValueError(f"{symbol}: timestamps are not sorted and unique")
        if any(right - left != 900 for left, right in zip(times, times[1:])):
            raise ValueError(f"{symbol}: non-contiguous 15-minute candles")
        candles[symbol] = series

    btc = candles["BTCUSDT"]
    expected_times = list(map(int, btc["time"]))
    raw_completed = []
    pending = []
    for symbol in sorted(ROBUST_SYMBOLS):
        series = candles[symbol]
        if list(map(int, series["time"])) != expected_times:
            raise ValueError(f"{symbol}: timeline does not match BTC")
        for index, timestamp in enumerate(expected_times):
            if timestamp <= previous_max or index < 385:
                continue
            setup = venue_setups(
                {key: value[:index] for key, value in series.items()},
                {key: value[:index] for key, value in btc.items()},
                entry_time=timestamp, market_price=float(series["open"][index]),
                symbol=symbol,
            )
            if not setup:
                continue
            if len(setup) != 1:
                raise AssertionError(f"{symbol} {timestamp}: multiple setups")
            item = setup[0]
            base = {
                "symbol": symbol, "direction": item["direction"],
                "entry_time": timestamp, "entry": item["entry"],
                "sl": item["sl"], "tp1": item["tp"],
                "target_r": item["target_r"], "family": item["family"],
                "module": item["module"], "utc_session": item["utc_session"],
                "eff_ratio": item["eff_ratio"], "z": item["z"],
                "abs_z": item["abs_z"],
                "dir_z": (item["z"] if item["direction"] == "LONG"
                          else -item["z"]),
                "risk_pct": abs(item["entry"] - item["sl"]) / item["entry"],
                "btc_atr_pct": item["btc_atr_pct"],
                "btc_eff20": item["btc_eff20"],
                "btc_return20_atr": item["btc_return20_atr"],
            }
            if index + 48 > len(expected_times):
                pending.append(base)
                continue
            risk = abs(item["entry"] - item["sl"])
            result = simulate_exit(
                series, range(index, index + 48), direction=item["direction"],
                entry=item["entry"], sl=item["sl"], tp1=item["tp"], tp2=item["tp"],
                atr=risk / 2, tp1_fraction=0, trail=False, trail_mult=0,
                stop_on_close=False, backstop_r=1, choose_trail=lambda *_: 0,
            )
            raw_completed.append({
                **base, "outcome": result.outcome, "net_r": result.gross_r,
                "exit_time": int(series["time"][result.bar]) + 900,
            })

    raw_completed.sort(key=lambda row: (row["entry_time"], row["symbol"]))
    pending.sort(key=lambda row: (row["entry_time"], row["symbol"]))
    # Match MAX_SAME_DIRECTION_POSITIONS=3 from the production configuration,
    # including still-open entries in the same portfolio state.
    accepted, accepted_pending = portfolio_gate_with_pending(
        raw_completed, pending, 3)
    completed_frame = pd.DataFrame(accepted)
    if completed_frame.empty:
        completed_frame = pd.DataFrame(columns=["entry_time", "exit_time", "net_r"])
    report = {
        "status": "PASS",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "previous_max_entry_time": previous_max,
        "previous_max_entry_iso": datetime.fromtimestamp(
            previous_max, timezone.utc).isoformat(),
        "cache": {
            "bars": len(expected_times), "start": expected_times[0],
            "end": expected_times[-1],
            "latest_fully_observable_entry": expected_times[-48],
        },
        "raw_completed_signals": len(raw_completed),
        # Preserve the pre-portfolio stream so later ordering/cap audits can
        # reproduce the exact decision set instead of trying to infer rejected
        # candidates from the accepted CSV.
        "raw_completed": raw_completed,
        "portfolio_completed_signals": len(accepted),
        "completed_metrics": metric(completed_frame),
        "completed": accepted,
        "raw_pending_signals": len(pending),
        "raw_pending": pending,
        "pending_signals": len(accepted_pending),
        "pending": accepted_pending,
        "sources": hashes,
        "invariant": (
            "closed-bar router prefixes only; next-bar market open; 48-bar stop-first "
            "outcomes; resolved and pending entries share the frozen causal "
            "portfolio gate with direction cap 3"
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    pd.DataFrame(accepted).to_csv(args.out.with_suffix(".csv"), index=False)
    print(json.dumps({key: report[key] for key in (
        "status", "previous_max_entry_iso", "cache", "raw_completed_signals",
        "portfolio_completed_signals", "completed_metrics", "pending_signals",
        "pending",
    )}, indent=2))


if __name__ == "__main__":
    main()
