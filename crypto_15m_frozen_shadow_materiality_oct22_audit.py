"""Score frozen 15m shadow rules against material 0.5%-2% exits.

The compact input table contains exact public X-Perp outcomes plus hashes of
the source snapshots. Original rules predate the snapshot. Literal exits are
diagnostic because they were inspected after these entries had occurred.
"""
from __future__ import annotations

import collections
import hashlib
import heapq
import json
from pathlib import Path

import pandas as pd

from src.research_crypto_replay import money, runtime_gate


ROOT = Path("reports/audit_2026_09_08")
ROWS = ROOT / "crypto_15m_frozen_shadow_materiality_oct22_rows.pkl"
ONE_HOUR_ROWS = ROOT / "crypto_okx_swap_1h_forward_append_oct22.pkl"
OUT = ROOT / "crypto_15m_frozen_shadow_materiality_oct22.json"
TARGETS = (0.005, 0.01, 0.015, 0.02)
STOP = 0.01
HOLD_BARS = 64
DEV_END = int(pd.Timestamp("2026-07-01", tz="UTC").timestamp())
VALIDATION_END = int(pd.Timestamp("2026-09-01", tz="UTC").timestamp())


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def resolve_literal(frame: pd.DataFrame, candles: dict[str, dict], target: float,
                    module: str, priority: int = 1) -> list[dict]:
    """Resolve market-open entries with pessimistic stop-first ordering."""
    output = []
    lookups = {symbol: {int(stamp): index for index, stamp in enumerate(data["time"])}
               for symbol, data in candles.items()}
    for row in frame.to_dict("records"):
        symbol = str(row["symbol"])
        candle = candles.get(symbol)
        index = lookups.get(symbol, {}).get(int(row["entry_time"]))
        if candle is None or index is None or index + 1 >= len(candle["time"]):
            continue
        direction = 1 if row["direction"] == "LONG" else -1
        entry = float(candle["open"][index])
        target_price = entry * (1 + direction * target)
        stop_price = entry * (1 - direction * STOP)
        last = min(len(candle["time"]) - 1, index + HOLD_BARS - 1)
        signed_return = None
        outcome = "TIME"
        for bar in range(index, last + 1):
            stop_hit = (candle["low"][bar] <= stop_price if direction > 0
                        else candle["high"][bar] >= stop_price)
            target_hit = (candle["high"][bar] >= target_price if direction > 0
                          else candle["low"][bar] <= target_price)
            if stop_hit:
                fill = (min(float(candle["open"][bar]), stop_price)
                        if direction > 0
                        else max(float(candle["open"][bar]), stop_price))
                signed_return = direction * (fill - entry) / entry
                last, outcome = bar, "SL"
                break
            if target_hit:
                signed_return = target
                last, outcome = bar, "TP"
                break
        if signed_return is None:
            signed_return = direction * (float(candle["close"][last]) - entry) / entry
            signed_return = max(-STOP, min(target, signed_return))
        output.append({
            **row, "signed_price_return": float(signed_return),
            "exit_time": int(candle["time"][last]) + 900,
            "outcome": outcome, "source": module, "source_priority": priority,
            "module": module, "module_priority": priority,
        })
    return output


def candidate_gate(rows: list[dict]) -> list[dict]:
    """Replay the pre-existing 15m exposure, cooldown and daily loss rails."""
    pending = []
    opened = {}
    last = {}
    per_scan = collections.Counter()
    output = []
    day = None
    streak = 0
    paused = False
    for row in sorted(rows, key=lambda item: (
            float(item["entry_time"]), str(item["symbol"]),
            int(item.get("module_priority", 0)))):
        now = float(row["entry_time"])
        new_day = int(now // 86400)
        if new_day != day:
            day, streak, paused = new_day, 0, False
        while pending and pending[0][0] < now:
            _, _, done = heapq.heappop(pending)
            opened.pop(str(done["symbol"]), None)
            if int(float(done["exit_time"]) // 86400) == day:
                streak = streak + 1 if done["outcome"] == "SL" else 0
                if streak >= 3:
                    paused = True
        symbol, direction = str(row["symbol"]), str(row["direction"])
        if paused or symbol in opened:
            continue
        if now - last.get((symbol, direction), -1e20) < 10800:
            continue
        scan = int(now // 900)
        if per_scan[scan] >= 3:
            continue
        if sum(str(item["direction"]) == direction for item in opened.values()) >= 3:
            continue
        opened[symbol] = row
        last[(symbol, direction)] = now
        per_scan[scan] += 1
        output.append(row)
        heapq.heappush(pending, (float(row["exit_time"]), len(output), row))
    return output


def stage(rows: list[dict], start: int, end: int) -> dict:
    chosen = [row for row in rows if start <= float(row["entry_time"]) < end]
    return money(candidate_gate(chosen), 0.03)


def geometry_report(item: dict) -> dict:
    variants = {}
    for target in TARGETS:
        rows = item["resolved"][target]
        variants[f"{100 * target:g}%"] = {
            "development_apr_jun": stage(rows, 0, DEV_END),
            "validation_jul_aug": stage(rows, DEV_END, VALIDATION_END),
            "hidden_sep_oct": stage(rows, VALIDATION_END, 2**63 - 1),
            "all": stage(rows, 0, 2**63 - 1),
        }
    return {"original_target_materiality": item["original_target_materiality"],
            "literal_geometry": variants}


def normalize_hourly() -> list[dict]:
    rows = pd.read_pickle(ONE_HOUR_ROWS).to_dict("records")
    for row in rows:
        row.update(source="literal_regime_router", source_priority=0,
                   module="literal_regime_router", module_priority=0,
                   outcome="SL" if float(row["signed_price_return"]) < 0 else "TP")
    return rows


def main() -> None:
    payload = pd.read_pickle(ROWS)
    reports = {name: geometry_report(item)
               for name, item in payload["candidates"].items()}

    # Deliberately labelled post-hoc: inspection suggested these one-percent
    # variants, so this comparison cannot activate either rule.
    satellites = (payload["candidates"]["asia_pullback"]["resolved"][0.01]
                  + payload["candidates"]["historical_evening"]["resolved"][0.01])
    unique, seen = [], set()
    for row in sorted(satellites, key=lambda item: (
            float(item["entry_time"]), int(item["source_priority"]),
            str(item["symbol"]))):
        key = (str(row["symbol"]), int(row["entry_time"]))
        if key not in seen:
            seen.add(key)
            unique.append(row)
    hourly = normalize_hourly()
    start = min(float(row["entry_time"]) for row in hourly)
    unique = [row for row in unique if float(row["entry_time"]) >= start]
    baseline = runtime_gate(hourly, loss_pause=3, hourly_cap=2)
    combined = runtime_gate(hourly + unique, loss_pause=3, hourly_cap=2)
    accepted_satellites = [row for row in combined
                           if row["source"] != "literal_regime_router"]

    report = {
        "status": "REJECTED_MATERIALITY_AND_OBJECTIVE",
        "ready_for_live_orders": False, "orders_possible": False,
        "protocol": {
            "entries": "three frozen pre-Oct-22 shadow rules",
            "original_runtime_minimum_target_move_pct": 0.5,
            "literal_targets_pct": [0.5, 1.0, 1.5, 2.0],
            "literal_stop_pct": 1.0, "hold_15m_bars": HOLD_BARS,
            "entry": "next 15-minute X-Perp market open",
            "collision": "stop first",
            "money": "3% current-equity margin, fixed 10x",
        },
        "integrity": payload["integrity"], "candidates": reports,
        "posthoc_one_pct_portfolio_diagnostic": {
            "not_activation_evidence": True,
            "hourly_baseline": money(baseline, 0.03),
            "with_satellites": money(combined, 0.03),
            "accepted_satellites": money(accepted_satellites, 0.03),
            "october_with_satellites": money([
                row for row in combined
                if float(row["entry_time"]) >= pd.Timestamp(
                    "2026-10-01", tz="UTC").timestamp()], 0.03),
        },
        "decision": (
            "Do not integrate these filters. Their original high-win-rate exits never "
            "reach the runtime's 0.5% material-move floor. Literal retargeting is too "
            "sparse, misses the monthly objective, and the inspected 1% combination "
            "makes October worse. Keep the frozen 1H route in shadow collection."),
        "sources": {
            **payload["sources"], str(ROWS): digest(ROWS),
            str(ONE_HOUR_ROWS): digest(ONE_HOUR_ROWS),
            str(Path(__file__)): digest(Path(__file__)),
        },
        "limitations": [
            "Literal target variants are post-hoc diagnostics and cannot activate a rule.",
            "The direct X-Perp window is about 190 calendar days and some contracts start later.",
            "Public candles cannot reproduce order-book slippage or execution latency.",
        ],
    }
    if not all(item["exact"] for item in report["integrity"]["overlap"].values()):
        raise AssertionError("historical X-Perp overlap changed")
    OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({
        "status": report["status"], "candidates": reports,
        "portfolio": report["posthoc_one_pct_portfolio_diagnostic"],
    }, indent=2))


if __name__ == "__main__":
    main()
