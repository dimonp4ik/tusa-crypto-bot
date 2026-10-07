"""Direct EU X-Perp transfer of frozen OKX global one-hour entries."""
from __future__ import annotations

import hashlib
import json
import pickle
from pathlib import Path

import numba as nb
import numpy as np
import pandas as pd

import crypto_5m_regime_router_literal_oct6_audit as portfolio
import crypto_expert_agreement_ladder_oct7_audit as ladder


ROOT = Path("reports/audit_2026_09_08")
ENTRY_REPORT = ROOT / "crypto_okx_swap_1h_regime_oct7.json"
ENTRY_ROWS = ROOT / "crypto_okx_swap_1h_regime_oct7.pkl"
RUNNER_REPORT = ROOT / "crypto_okx_swap_1h_runner_oct7.json"
CONDITIONAL_REPORT = ROOT / "crypto_okx_swap_1h_conditional_exit_oct7.json"
CONDITIONAL_ROWS = ROOT / "crypto_okx_swap_1h_conditional_exit_oct7.pkl"
MANIFESTS = (
    ROOT / "xperp_5m_oct4" / "manifest.json",
    ROOT / "xperp_5m_oct6_forward" / "manifest.json",
)
OUT = ROOT / "crypto_okx_swap_1h_xperp_transfer_oct7.json"
ROWS_OUT = ROOT / "crypto_okx_swap_1h_xperp_transfer_oct7.pkl"
SYMBOLS = ("BTCUSDT", "ETHUSDT", "XRPUSDT", "XLMUSDT", "LINKUSDT",
           "SOLUSDT", "AVAXUSDT", "AAVEUSDT")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def resolve_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else Path.cwd() / path


def load_xperp() -> tuple[dict, dict]:
    merged = {symbol: {} for symbol in SYMBOLS}
    sources = {}
    fields = ("open", "high", "low", "close", "volume")
    for manifest_path in MANIFESTS:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        sources[str(manifest_path)] = digest(manifest_path)
        for symbol in SYMBOLS:
            item = manifest["symbols"][symbol]
            path = resolve_path(item["path"])
            raw = path.read_bytes()
            found = hashlib.sha256(raw).hexdigest()
            if found != item["sha256"]:
                raise ValueError(f"X-Perp hash mismatch: {symbol} {path}")
            candle = pickle.loads(raw)
            sources[str(path)] = found
            for index, timestamp in enumerate(candle["time"]):
                values = tuple(float(candle[field][index]) for field in fields)
                old = merged[symbol].get(int(timestamp))
                if old is not None and not np.allclose(old[:4], values[:4], rtol=0, atol=0):
                    raise ValueError(f"X-Perp source revision: {symbol} {timestamp}")
                merged[symbol][int(timestamp)] = values
    output = {}
    for symbol, rows in merged.items():
        times = sorted(rows)
        if any(right - left != 300 for left, right in zip(times, times[1:])):
            raise ValueError(f"non-contiguous merged X-Perp data: {symbol}")
        output[symbol] = {"time": times, **{
            field: [rows[timestamp][position] for timestamp in times]
            for position, field in enumerate(fields)
        }}
    return output, sources


@nb.njit(cache=True)
def resolve_fixed(open_, high, low, close, times, entries, directions,
                  target_pct, stop_pct, hold):
    n = len(entries)
    signed = np.empty(n, dtype=np.float64)
    exits = np.empty(n, dtype=np.int64)
    hits = np.zeros(n, dtype=np.int8)
    reasons = np.empty(n, dtype=np.int8)
    target_move = target_pct / 100
    stop_move = stop_pct / 100
    for row in range(n):
        first = entries[row]
        direction = directions[row]
        entry = open_[first]
        target = entry * (1 + direction * target_move)
        stop = entry * (1 - direction * stop_move)
        last = first
        result = 0.0
        reason = 2
        end = min(len(times), first + hold)
        resolved = False
        for index in range(first, end):
            if index > first and times[index] - times[index - 1] != 300:
                break
            last = index
            stop_hit = low[index] <= stop if direction > 0 else high[index] >= stop
            target_hit = high[index] >= target if direction > 0 else low[index] <= target
            if stop_hit:
                fill = min(open_[index], stop) if direction > 0 else max(open_[index], stop)
                result = direction * (fill / entry - 1)
                reason = 0
                resolved = True
                break
            if target_hit:
                result = target_move
                hits[row] = 1
                reason = 1
                resolved = True
                break
        if not resolved:
            result = direction * (close[last] / entry - 1)
            result = max(-stop_move, min(target_move, result))
        signed[row] = result
        exits[row] = times[last] + 300
        reasons[row] = reason
    return signed, exits, hits, reasons


def map_entries(entries: pd.DataFrame, candles: dict, hold: int) -> pd.DataFrame:
    parts = []
    for symbol, part in entries.groupby("symbol", sort=False):
        candle = candles[symbol]
        lookup = {int(value): index for index, value in enumerate(candle["time"])}
        found = part.copy()
        found["entry_index"] = found.entry_time.map(lookup)
        found = found.dropna(subset=["entry_index"]).copy()
        found["entry_index"] = found.entry_index.astype(np.int64)
        found = found[found.entry_index + hold <= len(candle["time"])].copy()
        parts.append(found)
    return pd.concat(parts, ignore_index=True)


def fixed_rows(entries: pd.DataFrame, candles: dict) -> pd.DataFrame:
    hold = 12 * 12
    mapped = map_entries(entries, candles, hold)
    parts = []
    for symbol, part in mapped.groupby("symbol", sort=False):
        candle = candles[symbol]
        arrays = [np.asarray(candle[key], dtype=float)
                  for key in ("open", "high", "low", "close")]
        values, exits, hits, reasons = resolve_fixed(
            *arrays, np.asarray(candle["time"], dtype=np.int64),
            part.entry_index.to_numpy(np.int64),
            part.direction.to_numpy(np.int8), .5, 1.0, hold)
        found = part.copy()
        found["signed_return"], found["exit_time"] = values, exits
        found["target_hit"], found["exit_reason"] = hits.astype(bool), reasons
        found["quality"] = 1.0
        parts.append(found)
    return pd.concat(parts, ignore_index=True)


def runner_rows(entries: pd.DataFrame, candles: dict) -> pd.DataFrame:
    hold = 48 * 12
    mapped = map_entries(entries, candles, hold)
    parts = []
    for symbol, part in mapped.groupby("symbol", sort=False):
        candle = candles[symbol]
        arrays = [np.asarray(candle[key], dtype=float)
                  for key in ("open", "high", "low", "close")]
        values, exits, hits, runners, reasons = ladder.resolve_ladder(
            *arrays, np.asarray(candle["time"], dtype=np.int64),
            part.entry_index.to_numpy(np.int64),
            part.direction.to_numpy(np.int8), .25, 2.0, hold)
        found = part.copy()
        found["signed_return"], found["exit_time"] = values, exits
        found["target_hit"], found["runner_hit"] = hits.astype(bool), runners.astype(bool)
        found["exit_reason"], found["quality"] = reasons, 1.0
        parts.append(found)
    return pd.concat(parts, ignore_index=True)


def conditional_rows(entries: pd.DataFrame, candles: dict) -> pd.DataFrame:
    hold = 12 * 12
    mapped = map_entries(entries, candles, hold)
    parts = []
    for symbol, group in mapped.groupby("symbol", sort=False):
        candle = candles[symbol]
        arrays = [np.asarray(candle[key], dtype=float)
                  for key in ("open", "high", "low", "close")]
        times = np.asarray(candle["time"], dtype=np.int64)
        fixed = group[group.exit_profile.eq("fixed_05")].copy()
        if len(fixed):
            values, exits, hits, reasons = resolve_fixed(
                *arrays, times, fixed.entry_index.to_numpy(np.int64),
                fixed.direction.to_numpy(np.int8), .5, 1.0, hold)
            fixed["signed_return"], fixed["exit_time"] = values, exits
            fixed["target_hit"], fixed["exit_reason"] = hits.astype(bool), reasons
            fixed["runner_hit"] = False
            fixed["quality"] = 1.0
            parts.append(fixed)
        routed = group[group.exit_profile.eq("p25_r20_h12")].copy()
        if len(routed):
            values, exits, hits, runners, reasons = ladder.resolve_ladder(
                *arrays, times, routed.entry_index.to_numpy(np.int64),
                routed.direction.to_numpy(np.int8), .25, 2.0, hold)
            routed["signed_return"], routed["exit_time"] = values, exits
            routed["target_hit"], routed["runner_hit"] = (
                hits.astype(bool), runners.astype(bool))
            routed["exit_reason"], routed["quality"] = reasons, 1.0
            parts.append(routed)
    return pd.concat(parts, ignore_index=True)


def evaluate(frame: pd.DataFrame) -> dict:
    value = portfolio.evaluate(frame, (2026,), 0)
    value["first_entry_utc"] = pd.Timestamp(
        int(frame.entry_time.min()), unit="s", tz="UTC").isoformat() if len(frame) else None
    value["last_entry_utc"] = pd.Timestamp(
        int(frame.entry_time.max()), unit="s", tz="UTC").isoformat() if len(frame) else None
    if "runner_hit" in frame:
        value["runner_hit_rate"] = round(100 * float(frame.runner_hit.mean()), 4)
    return value


def transfer_ok(value: dict) -> bool:
    return bool(value["n"] >= 150 and value["return_pct"] > 0
                and value["literal_target_hit_rate"] >= 65
                and value["dd_pct"] <= 7
                and value["trades_per_calendar_day"] >= 2.0)


def main() -> None:
    entry_report = json.loads(ENTRY_REPORT.read_text(encoding="utf-8"))
    runner_report = json.loads(RUNNER_REPORT.read_text(encoding="utf-8"))
    conditional_report = json.loads(CONDITIONAL_REPORT.read_text(encoding="utf-8"))
    if entry_report.get("status") != "RESEARCH_CANDIDATE_AWAITING_DIRECT_FORWARD":
        raise ValueError("global OKX entry candidate is not frozen")
    frozen_runner = runner_report.get("frozen_before_2024") or {}
    if (frozen_runner.get("partial_fraction"), frozen_runner.get("runner_target_pct"),
            frozen_runner.get("hold_bars")) != (.25, 2.0, 48):
        raise ValueError("unexpected frozen runner")
    if conditional_report.get("status") != "RESEARCH_CANDIDATE_AWAITING_DIRECT_FORWARD":
        raise ValueError("conditional exit candidate is not frozen")
    entries = pd.read_pickle(ENTRY_ROWS)
    candles, sources = load_xperp()
    fixed = fixed_rows(entries, candles)
    runner = runner_rows(entries, candles)
    conditional = conditional_rows(pd.read_pickle(CONDITIONAL_ROWS), candles)
    fixed_result, runner_result = evaluate(fixed), evaluate(runner)
    conditional_result = evaluate(conditional)
    pd.concat([fixed.assign(exit_profile="fixed_05"),
               runner.assign(exit_profile="runner_05_20"),
               conditional.assign(exit_profile="conditional")],
              ignore_index=True).to_pickle(ROWS_OUT)
    checks = {"fixed_05": transfer_ok(fixed_result),
              "runner_05_20": transfer_ok(runner_result),
              "conditional": transfer_ok(conditional_result)}
    report = {
        "status": ("DIRECT_XPERP_SHADOW_CANDIDATE"
                   if any(checks.values()) else "REJECTED"),
        "protocol": {
            "signal_venue": "OKX global USDT swap 1h",
            "execution_venue": "OKX EU X-Perp 5m public candles",
            "selection_on_transfer_rows": False,
            "entry": "same timestamp X-Perp 5m market open",
            "fixed_exit": "0.5% target, 1% stop, 12h maximum",
            "runner_exit": "25% at 0.5%, 75% to 2%, breakeven runner, 48h maximum",
            "money": "3% equity margin, fixed 10x gross",
        },
        "fixed_05": fixed_result, "runner_05_20": runner_result,
        "conditional": conditional_result,
        "checks": checks,
        "sources": {**sources, str(ENTRY_REPORT): digest(ENTRY_REPORT),
                    str(ENTRY_ROWS): digest(ENTRY_ROWS),
                    str(RUNNER_REPORT): digest(RUNNER_REPORT),
                    str(CONDITIONAL_REPORT): digest(CONDITIONAL_REPORT),
                    str(CONDITIONAL_ROWS): digest(CONDITIONAL_ROWS),
                    str(Path(__file__)): digest(Path(__file__))},
        "limitations": [
            "X-Perp contracts expose only a short public history beginning in July 2026.",
            "The transfer is independent of training but remains too short for live activation.",
            "All entries and exits are market-price proxies; no limit entry is assumed.",
        ],
    }
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({key: report[key] for key in (
        "status", "fixed_05", "runner_05_20", "conditional", "checks")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
