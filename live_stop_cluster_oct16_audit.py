"""Audit live-export stop clusters without fees or hindsight entry filters.

The export contains bot signals, not independently verified exchange fills.  This
script keeps diagnostics separate from strategy selection: rolling caps only see
earlier accepted entries, while same-window feature splits are labelled as
diagnostic and must not be promoted without cross-era validation.
"""
from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
import math
import statistics
from datetime import datetime, timezone
from pathlib import Path

try:
    from scipy.stats import fisher_exact, mannwhitneyu
except ImportError:  # pragma: no cover - the report still runs without scipy
    fisher_exact = mannwhitneyu = None


LOSS_STATUSES = {"SL", "SL_HIT"}
OPEN_STATUSES = {"OPEN", "TP1_PARTIAL"}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def number(value, default=None):
    try:
        if value in (None, ""):
            return default
        parsed = float(value)
        return parsed if math.isfinite(parsed) else default
    except (TypeError, ValueError):
        return default


def signed_price_return(row: dict[str, str]) -> float:
    entry = number(row.get("entry_price"))
    exit_price = number(row.get("exit_price"))
    if not entry or exit_price is None:
        raise ValueError(f"signal {row.get('id')} lacks a valid entry/exit")
    direction = (row.get("direction") or "").upper()
    if direction not in {"LONG", "SHORT"}:
        raise ValueError(f"signal {row.get('id')} has invalid direction")
    raw = (exit_price - entry) / entry
    return raw if direction == "LONG" else -raw


def prepare(signals: list[dict[str, str]]) -> list[dict]:
    result = []
    for raw in signals:
        if (raw.get("status") or "").upper() in OPEN_STATUSES:
            continue
        opened = number(raw.get("opened_at"))
        if opened is None or number(raw.get("exit_price")) is None:
            continue
        row = dict(raw)
        row["_opened"] = opened
        row["_return"] = signed_price_return(row)
        row["_loss"] = row["_return"] < 0
        row["_stop"] = (row.get("status") or "").upper() in LOSS_STATUSES
        row["_r"] = number(row.get("realized_r"))
        result.append(row)
    return sorted(result, key=lambda item: (item["_opened"], str(item.get("id"))))


def max_streak(rows: list[dict], field: str) -> int:
    best = run = 0
    for row in rows:
        run = run + 1 if row[field] else 0
        best = max(best, run)
    return best


def account_path(rows: list[dict], margin_fraction: float, leverage: float) -> dict:
    equity = peak = 1.0
    max_dd = 0.0
    for row in rows:
        equity *= 1.0 + margin_fraction * leverage * row["_return"]
        peak = max(peak, equity)
        max_dd = max(max_dd, 1.0 - equity / peak)
    return {
        "return_pct": (equity - 1.0) * 100.0,
        "max_drawdown_pct": max_dd * 100.0,
    }


def metrics(rows: list[dict], margin_fraction: float, leverage: float) -> dict:
    positives = sum(row["_return"] > 0 for row in rows)
    flats = sum(row["_return"] == 0 for row in rows)
    successful_statuses = sum(not row["_stop"] for row in rows)
    dates = {
        datetime.fromtimestamp(row["_opened"], tz=timezone.utc).date().isoformat()
        for row in rows
    }
    r_values = [row["_r"] for row in rows if row["_r"] is not None]
    result = {
        "n": len(rows),
        "positive_win_rate_pct": 100.0 * positives / len(rows) if rows else None,
        "successful_status_rate_pct": 100.0 * successful_statuses / len(rows) if rows else None,
        "negative_exit_rate_pct": 100.0 * sum(row["_loss"] for row in rows) / len(rows) if rows else None,
        "flat_exit_rate_pct": 100.0 * flats / len(rows) if rows else None,
        "signed_price_return_sum_pct": 100.0 * sum(row["_return"] for row in rows),
        "net_r": sum(r_values) if len(r_values) == len(rows) else None,
        "max_consecutive_negative_exits": max_streak(rows, "_loss"),
        "max_consecutive_stops": max_streak(rows, "_stop"),
        "active_utc_days": len(dates),
        "trades_per_active_day": len(rows) / len(dates) if dates else None,
    }
    result.update(account_path(rows, margin_fraction, leverage))
    return result


def causal_cap(rows: list[dict], cap: int, seconds: float = 3600.0,
               same_direction: bool = False) -> list[dict]:
    """Select rows using only entries already accepted at the decision time."""
    accepted = []
    for row in rows:
        recent = [prior for prior in accepted if row["_opened"] - prior["_opened"] < seconds]
        if same_direction:
            recent = [prior for prior in recent if prior.get("direction") == row.get("direction")]
        if len(recent) < cap:
            accepted.append(row)
    return accepted


def cluster_membership(rows: list[dict], seconds: float = 900.0) -> tuple[list[dict], list[dict]]:
    clustered, isolated = [], []
    for index, row in enumerate(rows):
        has_neighbor = any(
            index != other and abs(row["_opened"] - candidate["_opened"]) <= seconds
            for other, candidate in enumerate(rows)
        )
        (clustered if has_neighbor else isolated).append(row)
    return clustered, isolated


def prior_entry_bucket(rows: list[dict], seconds: float = 3600.0,
                       same_direction: bool = False) -> tuple[list[dict], list[dict]]:
    first, additional = [], []
    for index, row in enumerate(rows):
        priors = [p for p in rows[:index] if row["_opened"] - p["_opened"] < seconds]
        if same_direction:
            priors = [p for p in priors if p.get("direction") == row.get("direction")]
        (additional if priors else first).append(row)
    return first, additional


def contingency(clustered: list[dict], isolated: list[dict]):
    table = [
        [sum(r["_loss"] for r in clustered), sum(not r["_loss"] for r in clustered)],
        [sum(r["_loss"] for r in isolated), sum(not r["_loss"] for r in isolated)],
    ]
    if fisher_exact is None or not clustered or not isolated:
        return {"table_losses_nonlosses": table, "odds_ratio": None, "p_value": None}
    result = fisher_exact(table)
    return {"table_losses_nonlosses": table, "odds_ratio": float(result.statistic),
            "p_value": float(result.pvalue)}


def grouped(rows: list[dict], key, margin_fraction: float, leverage: float) -> dict:
    buckets = collections.defaultdict(list)
    for row in rows:
        buckets[str(key(row))].append(row)
    return {name: metrics(bucket, margin_fraction, leverage)
            for name, bucket in sorted(buckets.items())}


def distribution_test(rows: list[dict], field: str) -> dict:
    losses = [number(row.get(field)) for row in rows if row["_loss"]]
    nonlosses = [number(row.get(field)) for row in rows if not row["_loss"]]
    losses = [value for value in losses if value is not None]
    nonlosses = [value for value in nonlosses if value is not None]
    result = {
        "field": field,
        "loss_n": len(losses),
        "nonloss_n": len(nonlosses),
        "loss_median": statistics.median(losses) if losses else None,
        "nonloss_median": statistics.median(nonlosses) if nonlosses else None,
        "mann_whitney_p": None,
    }
    if mannwhitneyu is not None and losses and nonlosses:
        result["mann_whitney_p"] = float(mannwhitneyu(losses, nonlosses).pvalue)
    return result


def relation(row: dict, setup: dict | None) -> str:
    trend = ((setup or {}).get("trend") or "").lower()
    direction = (row.get("direction") or "").upper()
    if trend == "range" or not trend:
        return "range_or_missing"
    if (direction == "LONG" and trend == "bull") or (direction == "SHORT" and trend == "bear"):
        return "aligned"
    return "opposed"


def audit_pair(signals_path: Path, setups_path: Path, margin_fraction: float,
               leverage: float) -> dict:
    raw_signals, setups = read_csv(signals_path), read_csv(setups_path)
    rows = prepare(raw_signals)
    setup_by_signal = {str(s.get("signal_id")): s for s in setups if s.get("signal_id")}
    for row in rows:
        setup = setup_by_signal.get(str(row.get("id")))
        row["_relation"] = relation(row, setup)
        row["_oi_confirms"] = (setup or {}).get("oi_confirms") or "missing"
        for field in ("book_spread_bps", "book_depth_usd", "book_imbalance", "hours_to_event"):
            if setup and setup.get(field) not in (None, ""):
                row[field] = setup[field]

    clustered, isolated = cluster_membership(rows)
    first, additional = prior_entry_bucket(rows)
    first_dir, additional_dir = prior_entry_bucket(rows, same_direction=True)
    halves = [rows[:len(rows) // 2], rows[len(rows) // 2:]]
    caps = {}
    for cap in (1, 2, 3):
        caps[f"total_{cap}_per_rolling_hour"] = metrics(
            causal_cap(rows, cap), margin_fraction, leverage)
        caps[f"same_direction_{cap}_per_rolling_hour"] = metrics(
            causal_cap(rows, cap, same_direction=True), margin_fraction, leverage)

    hours = grouped(
        rows,
        lambda row: datetime.fromtimestamp(row["_opened"], tz=timezone.utc).strftime("%Y-%m-%dT%H:00Z"),
        margin_fraction,
        leverage,
    )
    worst_hours = sorted(
        ({"hour": hour, **values} for hour, values in hours.items()),
        key=lambda item: (item["net_r"] if item["net_r"] is not None else item["signed_price_return_sum_pct"]),
    )[:8]

    return {
        "sources": {
            signals_path.name: hashlib.sha256(signals_path.read_bytes()).hexdigest(),
            setups_path.name: hashlib.sha256(setups_path.read_bytes()).hexdigest(),
        },
        "raw_signal_count": len(raw_signals),
        "resolved_with_exit_count": len(rows),
        "baseline": metrics(rows, margin_fraction, leverage),
        "cluster_diagnostic": {
            "within_15m_neighbor": metrics(clustered, margin_fraction, leverage),
            "without_15m_neighbor": metrics(isolated, margin_fraction, leverage),
            "fisher_loss_association": contingency(clustered, isolated),
            "first_in_rolling_hour": metrics(first, margin_fraction, leverage),
            "additional_in_rolling_hour": metrics(additional, margin_fraction, leverage),
            "first_same_direction_in_rolling_hour": metrics(first_dir, margin_fraction, leverage),
            "additional_same_direction_in_rolling_hour": metrics(additional_dir, margin_fraction, leverage),
            "worst_utc_hours": worst_hours,
        },
        "causal_caps": caps,
        "chronological_half_check": {
            "early_baseline": metrics(halves[0], margin_fraction, leverage),
            "early_total_cap2": metrics(causal_cap(halves[0], 2), margin_fraction, leverage),
            "late_baseline": metrics(halves[1], margin_fraction, leverage),
            "late_total_cap2": metrics(causal_cap(halves[1], 2), margin_fraction, leverage),
        },
        "same_window_diagnostics": {
            "structure_relation": grouped(rows, lambda row: row["_relation"], margin_fraction, leverage),
            "entry_source": grouped(rows, lambda row: row.get("entry_source") or "missing", margin_fraction, leverage),
            "oi_confirms": grouped(rows, lambda row: row["_oi_confirms"], margin_fraction, leverage),
            "session": grouped(rows, lambda row: row.get("session") or "missing", margin_fraction, leverage),
            "loss_distribution_tests": [
                distribution_test(rows, field) for field in (
                    "book_spread_bps", "book_depth_usd", "book_imbalance",
                    "hours_to_event", "overhead_atr", "underfoot_atr",
                )
            ],
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--crypto-signals", type=Path, required=True)
    parser.add_argument("--crypto-setups", type=Path, required=True)
    parser.add_argument("--stock-signals", type=Path, required=True)
    parser.add_argument("--stock-setups", type=Path, required=True)
    parser.add_argument("--margin-fraction", type=float, default=0.03)
    parser.add_argument("--leverage", type=float, default=10.0)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if not 0 < args.margin_fraction <= 1 or args.leverage <= 0:
        parser.error("margin fraction must be in (0, 1] and leverage must be positive")

    result = {
        "method": {
            "margin_fraction_of_current_equity": args.margin_fraction,
            "leverage": args.leverage,
            "fees_and_funding": "excluded_by_request",
            "win_definition": "positive signed entry-to-exit price return",
            "time_order": "opened_at ascending",
        },
        "crypto": audit_pair(args.crypto_signals, args.crypto_setups,
                             args.margin_fraction, args.leverage),
        "stock": audit_pair(args.stock_signals, args.stock_setups,
                            args.margin_fraction, args.leverage),
        "limitations": [
            "Exports are bot signal accounting, not independently verified exchange fills.",
            "Overlapping positions are compounded in entry order; this approximates 3% of then-current equity.",
            "Feature splits use same-window labels and diagnose failures but are not validated filters.",
            "A rolling cap is causal but remains a risk brake, not evidence of trading edge.",
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({name: result[name]["baseline"] for name in ("crypto", "stock")},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
