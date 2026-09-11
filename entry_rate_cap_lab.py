"""Causal replay of a rolling same-direction entry-rate cap.

The cap only reads entry timestamps that were already observed.  Closed trade
outcomes are used solely by the existing daily loss-streak brake and only after
their exit timestamp.  This is research output, not evidence of live profit.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import heapq
import json
import math
from collections import Counter, defaultdict, deque
from pathlib import Path


NUMERIC_FIELDS = ("entry_time", "exit_time", "net_r")


def load_rows(path: Path) -> list[dict]:
    rows = list(csv.DictReader(path.open(encoding="utf-8-sig", newline="")))
    for row in rows:
        for field in NUMERIC_FIELDS:
            try:
                row[field] = float(row[field])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"{path}: missing numeric {field}") from exc
        if row.get("direction") not in ("LONG", "SHORT"):
            raise ValueError(f"{path}: invalid direction {row.get('direction')!r}")
    return sorted(rows, key=lambda row: (row["entry_time"], row.get("symbol", "")))


def replay(
    rows: list[dict],
    *,
    same_direction_positions: int,
    loss_streak: int,
    rate_count: int | None = None,
    rate_window_hours: float | None = None,
) -> tuple[list[dict], Counter]:
    if same_direction_positions <= 0:
        raise ValueError("same_direction_positions must be positive")
    if loss_streak < 0:
        raise ValueError("loss_streak must be nonnegative")
    if (rate_count is None) != (rate_window_hours is None):
        raise ValueError("rate_count and rate_window_hours must be set together")
    if rate_count is not None and (rate_count <= 0 or rate_window_hours <= 0):
        raise ValueError("entry-rate limits must be positive")

    pending: list[tuple[float, int, dict]] = []
    open_by_symbol: dict[str, dict] = {}
    last_entry: dict[tuple[str, str], float] = {}
    entries_by_direction = {"LONG": deque(), "SHORT": deque()}
    per_scan: Counter = Counter()
    blocked: Counter = Counter()
    selected: list[dict] = []
    current_day: int | None = None
    streak = 0
    paused = False
    window_seconds = (rate_window_hours or 0.0) * 3600.0

    for row in rows:
        now = row["entry_time"]
        day = int(now // 86400)
        if day != current_day:
            current_day = day
            streak = 0
            paused = False

        while pending and pending[0][0] < now:
            exit_time, _, closed = heapq.heappop(pending)
            open_by_symbol.pop(closed.get("symbol", ""), None)
            if int(exit_time // 86400) == current_day and loss_streak:
                streak = streak + 1 if closed.get("outcome") == "SL" else 0
                if streak >= loss_streak:
                    paused = True

        if paused:
            blocked["loss_streak"] += 1
            continue
        symbol = row.get("symbol", "")
        if symbol in open_by_symbol:
            blocked["symbol_open"] += 1
            continue
        key = (symbol, row["direction"])
        if now - last_entry.get(key, -math.inf) < 10800:
            blocked["cooldown"] += 1
            continue
        scan = int(now // 900)
        if per_scan[scan] >= 3:
            blocked["scan_cap"] += 1
            continue
        if sum(opened["direction"] == row["direction"] for opened in open_by_symbol.values()) >= same_direction_positions:
            blocked["position_cap"] += 1
            continue

        recent = entries_by_direction[row["direction"]]
        if rate_count is not None:
            cutoff = now - window_seconds
            while recent and recent[0] <= cutoff:
                recent.popleft()
            if len(recent) >= rate_count:
                blocked["entry_rate_cap"] += 1
                continue

        selected.append(row)
        open_by_symbol[symbol] = row
        last_entry[key] = now
        per_scan[scan] += 1
        recent.append(now)
        heapq.heappush(pending, (row["exit_time"], len(selected), row))

    return selected, blocked


def metrics(rows: list[dict]) -> dict:
    values = [row["net_r"] for row in rows]
    gains = sum(value for value in values if value > 0)
    losses = -sum(value for value in values if value < 0)
    equity = peak = drawdown = 0.0
    for row in sorted(rows, key=lambda item: (item["exit_time"], item.get("symbol", ""))):
        equity += row["net_r"]
        peak = max(peak, equity)
        drawdown = max(drawdown, peak - equity)
    return {
        "n": len(values),
        "win_rate": 100.0 * sum(value > 0 for value in values) / len(values) if values else 0.0,
        "net_r": sum(values),
        "mean_r": sum(values) / len(values) if values else 0.0,
        "profit_factor": gains / losses if losses else None,
        "max_drawdown_r": drawdown,
    }


def evaluate(
    rows: list[dict],
    *,
    same_direction_positions: int,
    loss_streak: int,
    counts: list[int],
    windows: list[float],
) -> dict:
    baseline_rows, baseline_blocked = replay(
        rows,
        same_direction_positions=same_direction_positions,
        loss_streak=loss_streak,
    )
    variants = []
    for window in windows:
        for count in counts:
            selected, blocked = replay(
                rows,
                same_direction_positions=same_direction_positions,
                loss_streak=loss_streak,
                rate_count=count,
                rate_window_hours=window,
            )
            variants.append({
                "max_entries": count,
                "window_hours": window,
                "metrics": metrics(selected),
                "blocked": dict(blocked),
            })
    return {
        "baseline": {"metrics": metrics(baseline_rows), "blocked": dict(baseline_blocked)},
        "variants": variants,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("raw", type=Path)
    parser.add_argument("--same-direction-positions", type=int, required=True)
    parser.add_argument("--loss-streak", type=int, required=True)
    parser.add_argument("--counts", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--windows", type=float, nargs="+", default=[1, 2, 3, 4, 6])
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    rows = load_rows(args.raw)
    result = {
        "source": str(args.raw),
        "source_sha256": hashlib.sha256(args.raw.read_bytes()).hexdigest(),
        "arguments": {
            "same_direction_positions": args.same_direction_positions,
            "loss_streak": args.loss_streak,
            "counts": args.counts,
            "windows": args.windows,
        },
        **evaluate(
            rows,
            same_direction_positions=args.same_direction_positions,
            loss_streak=args.loss_streak,
            counts=args.counts,
            windows=args.windows,
        ),
        "status": "RESEARCH_ONLY",
        "limitations": [
            "Historical modeled outcomes are not exchange fills.",
            "The grid is exploratory; selecting its best row on this sample would overfit.",
            "The replay uses only past entries for the rate cap and only already closed outcomes for the loss-streak brake.",
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"baseline": result["baseline"], "variants": result["variants"]}))


if __name__ == "__main__":
    main()
