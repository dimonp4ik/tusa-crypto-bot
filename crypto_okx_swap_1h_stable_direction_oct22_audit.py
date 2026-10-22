"""Transfer a stable direction/symbol route to fresh X-Perp candles.

The frequency model and threshold are frozen by the preceding sequential
audit.  Five coarse direction routes are compared on complete 2022-2026
calendar years.  One route is selected before the direct X-Perp replay.
"""
from __future__ import annotations

import hashlib
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

import crypto_5m_regime_router_literal_oct6_audit as portfolio
import crypto_okx_swap_1h_xperp_transfer_oct7_audit as transfer


ROOT = Path("reports/audit_2026_09_08")
FREQUENCY_REPORT = ROOT / "crypto_okx_swap_1h_frequency_oct22.json"
FREQUENCY_ROWS = ROOT / "crypto_okx_swap_1h_frequency_oct22.pkl"
FRESH_MANIFEST = ROOT / "xperp_5m_oct22_forward" / "manifest.json"
OUT = ROOT / "crypto_okx_swap_1h_stable_direction_oct22.json"
ROWS_OUT = ROOT / "crypto_okx_swap_1h_stable_direction_oct22.pkl"
POLICIES = (
    "short_only",
    "short_plus_long_pullback",
    "short_plus_long_bull",
    "short_plus_long_avax_link",
    "short_plus_long_pullback_or_avax",
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def family(frame: pd.DataFrame) -> pd.Series:
    columns = [name for name in frame if name.startswith("family_")]
    return frame[columns].fillna(0).idxmax(axis=1).str[7:]


def apply_policy(frame: pd.DataFrame, policy: str) -> pd.DataFrame:
    short = frame.direction.eq(-1)
    long = frame.direction.eq(1)
    found_family = family(frame)
    bull = frame.btc_ret168.ge(0) & frame.btc_ret168.lt(.05)
    masks = {
        "short_only": short,
        "short_plus_long_pullback": short | (long & found_family.eq("pullback")),
        "short_plus_long_bull": short | (long & bull),
        "short_plus_long_avax_link": (
            short | (long & frame.symbol.isin(("AVAXUSDT", "LINKUSDT")))),
        "short_plus_long_pullback_or_avax": (
            short | (long & (found_family.eq("pullback")
                             | frame.symbol.eq("AVAXUSDT")))),
    }
    if policy not in masks:
        raise ValueError(f"unknown direction policy: {policy}")
    return frame[masks[policy]].copy()


def historical_metrics(frame: pd.DataFrame, policy: str) -> dict[str, dict]:
    routed = apply_policy(frame, policy)
    return {str(year): portfolio.evaluate(routed, (year,), 0)
            for year in range(2022, 2027)}


def admissible(metrics: dict[str, dict]) -> bool:
    values = list(metrics.values())
    return bool(
        min(value["n"] for value in values) >= 500
        and min(value["return_pct"] for value in values) > 0
        and min(value["literal_target_hit_rate"] for value in values) >= 65
        and max(value["dd_pct"] for value in values) <= 7
    )


def rank(metrics: dict[str, dict]) -> tuple[float, float, float]:
    values = list(metrics.values())
    return (
        min(value["return_pct"] for value in values),
        sum(value["return_pct"] for value in values),
        min(value["trades_per_calendar_day"] for value in values),
    )


def load_fresh_xperp() -> tuple[dict, dict]:
    manifests = (*transfer.MANIFESTS, FRESH_MANIFEST)
    merged = {symbol: {} for symbol in transfer.SYMBOLS}
    sources = {}
    fields = ("open", "high", "low", "close", "volume")
    for manifest_path in manifests:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        sources[str(manifest_path)] = digest(manifest_path)
        for symbol in transfer.SYMBOLS:
            item = manifest["symbols"][symbol]
            path = transfer.resolve_path(item["path"])
            raw = path.read_bytes()
            found = hashlib.sha256(raw).hexdigest()
            if found != item["sha256"]:
                raise ValueError(f"X-Perp hash mismatch: {symbol} {path}")
            candle = pickle.loads(raw)
            sources[str(path)] = found
            for index, timestamp in enumerate(candle["time"]):
                values = tuple(float(candle[field][index]) for field in fields)
                old = merged[symbol].get(int(timestamp))
                if old is not None and not np.allclose(
                        old[:4], values[:4], rtol=0, atol=0):
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


def main() -> None:
    frequency_report = json.loads(FREQUENCY_REPORT.read_text(encoding="utf-8"))
    frame = pd.read_pickle(FREQUENCY_ROWS)
    results = {policy: historical_metrics(frame, policy) for policy in POLICIES}
    accepted = [policy for policy in POLICIES if admissible(results[policy])]
    if not accepted:
        raise RuntimeError("no stable direction policy passes historical rails")
    selected = max(accepted, key=lambda policy: rank(results[policy]))

    routed = apply_policy(frame, selected)
    candles, sources = load_fresh_xperp()
    direct_rows = transfer.fixed_rows(routed, candles)
    direct = transfer.evaluate(direct_rows)
    direct_passed = bool(
        direct["n"] >= 100
        and direct["return_pct"] > 0
        and direct["literal_target_hit_rate"] >= 65
        and direct["dd_pct"] <= 7
        and direct["trades_per_calendar_day"] >= 1.8
        and direct["negative_months"] <= 2)
    direct_rows.to_pickle(ROWS_OUT)

    report = {
        "status": ("STABLE_DIRECTION_AWAITING_NEW_FORWARD_AFTER_2026_10_22"
                   if direct_passed else "REJECTED_DIRECT_TRANSFER"),
        "ready_for_live_orders": False,
        "order_eligible": False,
        "frozen_frequency_model": frequency_report["frozen_before_2024"],
        "selection_protocol": {
            "policy_universe": list(POLICIES),
            "historical_periods": [2022, 2023, 2024, 2025, 2026],
            "admission": (
                "positive return, >=65% target hits, <=7% drawdown, and "
                ">=500 trades in every calendar-year sample"),
            "rank": "maximize worst-year return, then total return and frequency",
            "direct_holdout": "X-Perp candles through October 21, 2026",
            "entry": "same-timestamp 5-minute market open",
            "exit": "literal 0.5% target, 1% stop, 12-hour maximum, stop-first",
            "money": "client-selected 3% current-equity margin, fixed 10x gross",
        },
        "historical_results": results,
        "admissible_policies": accepted,
        "selected_before_direct": selected,
        "direct_xperp": direct,
        "direct_xperp_passed": direct_passed,
        "objective_checks": {
            "direct_transfer_passed": direct_passed,
            "at_least_3_trades_per_calendar_day": (
                direct["trades_per_calendar_day"] >= 3),
            "every_direct_month_at_least_30pct": bool(
                direct["monthly_return_pct"]
                and min(direct["monthly_return_pct"].values()) >= 30),
            "ready_for_live_orders": False,
        },
        "sources": {
            **sources,
            str(FREQUENCY_REPORT): digest(FREQUENCY_REPORT),
            str(FREQUENCY_ROWS): digest(FREQUENCY_ROWS),
            str(Path(__file__).resolve()): digest(Path(__file__).resolve()),
        },
        "limitations": [
            "The direction policy uses all historical calendar years; X-Perp is the transfer holdout.",
            "Global OKX signal history currently ends before the newest X-Perp execution candles.",
            "Direct X-Perp is shorter than a full market cycle.",
            "Gross results exclude fees at user request.",
            "No backtest guarantees a future monthly return.",
        ],
    }
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({
        "status": report["status"],
        "admissible_policies": accepted,
        "selected_before_direct": selected,
        "historical": results[selected],
        "direct_xperp": direct,
        "objective_checks": report["objective_checks"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
