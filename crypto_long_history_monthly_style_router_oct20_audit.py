"""Causal monthly router over a pre-hidden diverse cross-section library."""
from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

import crypto_long_history_hourly_cross_section_oct20_audit as base
import crypto_raw_hourly_direction_oct19_audit as portfolio


OUT = base.ROOT / "crypto_long_history_monthly_style_router_oct20.json"
ROWS_OUT = base.ROOT / "crypto_long_history_monthly_style_router_oct20.pkl"
BASE_REPORT = base.OUT
LOOKBACK_MONTHS = (1, 3, 6, 12)
TOP_K = (1, 2, 3)
MIN_COUNTS = (20, 50)
MIN_MEANS = (0.0, .00025)
SCORE_METHODS = ("mean", "mean_x_sqrt_n")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def variant_id(item: dict) -> str:
    return (f"h{item['lookback_hours']}_{item['style']}_"
            f"z{item['z_threshold']}_m{item['minimum_move_pct']}_"
            f"t{item['target_pct']}_s{item['stop_pct']}_"
            f"b{item['hold_bars_5m']}")


def build_library(candles: dict[str, dict], items: list[dict]
                  ) -> dict[str, list[dict]]:
    features = {}
    labeled = {}
    output = {}
    for item in items:
        feature_key = (int(item["lookback_hours"]), str(item["style"]))
        if feature_key not in features:
            features[feature_key] = base.feature_rows(
                candles, feature_key[0], feature_key[1])
        geometry = (float(item["target_pct"]), float(item["stop_pct"]),
                    int(item["hold_bars_5m"]))
        label_key = (*feature_key, geometry)
        if label_key not in labeled:
            labeled[label_key] = base.label(
                features[feature_key], candles, geometry)
        chosen = base.entries(
            labeled[label_key], float(item["z_threshold"]),
            float(item["minimum_move_pct"]) / 100)
        _, rows = portfolio.evaluate(
            chosen, float("-inf"), int(item["loss_pause"]),
            int(item["hourly_cap"]), .03)
        name = variant_id(item)
        for row in rows:
            row["variant"] = name
            row["module"] = name
        output[name] = rows
    return output


def trailing_scores(library: dict[str, list[dict]], cutoff: pd.Timestamp,
                    lookback_months: int, min_count: int, min_mean: float,
                    method: str) -> list[tuple[float, int, str]]:
    """Score only trades whose exits are strictly before the decision time."""
    if method not in SCORE_METHODS:
        raise ValueError(f"unknown score method: {method}")
    start = cutoff - pd.DateOffset(months=lookback_months)
    left = start.timestamp()
    right = cutoff.timestamp()
    scores = []
    for name, rows in library.items():
        values = np.asarray([
            float(row["signed_price_return"]) for row in rows
            if left <= float(row["exit_time"]) < right
        ], dtype=float)
        if len(values) < min_count:
            continue
        mean = float(values.mean())
        if mean < min_mean:
            continue
        score = mean if method == "mean" else mean * np.sqrt(len(values))
        scores.append((float(score), len(values), name))
    return sorted(scores, reverse=True)


def route(library: dict[str, list[dict]], start: str, end: str, *,
          lookback_months: int, top_k: int, min_count: int,
          min_mean: float, method: str, loss_pause: int,
          hourly_cap: int) -> tuple[list[dict], list[dict]]:
    candidate_rows = []
    decisions = []
    for month in pd.period_range(start, end, freq="M"):
        cutoff = month.start_time.tz_localize("UTC")
        finish = (month + 1).start_time.tz_localize("UTC")
        scores = trailing_scores(
            library, cutoff, lookback_months, min_count, min_mean, method)
        selected = scores[:top_k]
        decisions.append({
            "month": str(month),
            "selected": [{"variant": name, "score": score, "n": count}
                         for score, count, name in selected],
        })
        for priority, (score, _, name) in enumerate(selected):
            for original in library[name]:
                entered = float(original["entry_time"])
                if cutoff.timestamp() <= entered < finish.timestamp():
                    row = dict(original)
                    row["model_probability"] = float(score)
                    row["module_priority"] = priority
                    candidate_rows.append(row)
    candidate_rows.sort(key=lambda row: (
        float(row["entry_time"]), -float(row["model_probability"]),
        str(row["symbol"]), str(row["variant"])))
    unique = {}
    for row in candidate_rows:
        unique.setdefault((str(row["symbol"]), int(row["entry_time"])), row)
    accepted = portfolio.rails.ranked_runtime_gate(
        list(unique.values()), loss_pause=loss_pause, hourly_cap=hourly_cap)
    return accepted, decisions


def rank(value: dict) -> tuple:
    return (
        value["dd_pct"] <= 15 and value["trades_per_calendar_day"] >= 3,
        -int(value["negative_months"]), base.month_floor(value),
        float(value["return_pct"]), -float(value["dd_pct"]),
    )


def main() -> None:
    source = json.loads(BASE_REPORT.read_text(encoding="utf-8"))
    items = source["development_diverse_library"]
    candles, candle_sources = base.load()
    library = build_library(candles, items)
    policies = []
    grid = itertools.product(
        LOOKBACK_MONTHS, TOP_K, MIN_COUNTS, MIN_MEANS, SCORE_METHODS,
        base.LOSS_PAUSES, base.HOURLY_CAPS)
    for index, (lookback, top_k, min_count, min_mean, method,
                loss_pause, hourly_cap) in enumerate(grid, 1):
        rows, decisions = route(
            library, "2023-01", "2024-12", lookback_months=lookback,
            top_k=top_k, min_count=min_count, min_mean=min_mean,
            method=method, loss_pause=loss_pause, hourly_cap=hourly_cap)
        value = portfolio.money(rows, .03)
        policies.append({
            "lookback_months": lookback, "top_k": top_k,
            "minimum_completed_trades": min_count,
            "minimum_mean_price_return_pct": 100 * min_mean,
            "score_method": method, "loss_pause": loss_pause,
            "hourly_cap": hourly_cap,
            "confirmation_2023_2024_3pct": value,
            "_decisions": decisions,
        })
        if index % 48 == 0:
            print(json.dumps({"phase": "policy_selection", "done": index}),
                  flush=True)
    policies.sort(
        key=lambda item: rank(item["confirmation_2023_2024_3pct"]),
        reverse=True)
    selected = policies[0]
    hidden = {}
    hidden_rows = []
    hidden_decisions = None
    for margin in (.03, .05):
        rows, decisions = route(
            library, "2025-01", "2026-09",
            lookback_months=int(selected["lookback_months"]),
            top_k=int(selected["top_k"]),
            min_count=int(selected["minimum_completed_trades"]),
            min_mean=float(selected["minimum_mean_price_return_pct"]) / 100,
            method=str(selected["score_method"]),
            loss_pause=int(selected["loss_pause"]),
            hourly_cap=int(selected["hourly_cap"]))
        hidden[f"margin_{round(100 * margin)}pct"] = portfolio.money(rows, margin)
        if margin == .03:
            hidden_rows = rows
            hidden_decisions = decisions
    pd.DataFrame(hidden_rows).to_pickle(ROWS_OUT)
    three = hidden["margin_3pct"]
    objective = bool(
        three["negative_months"] == 0 and base.month_floor(three) >= 30
        and three["dd_pct"] <= 10
        and three["trades_per_calendar_day"] >= 3)
    public_selected = {key: value for key, value in selected.items()
                       if not key.startswith("_")}
    report = {
        "status": ("MONTHLY_STYLE_ROUTER_SHADOW_CANDIDATE"
                   if objective else "REJECTED_OBJECTIVE"),
        "ready_for_live_orders": False,
        "protocol": {
            "library_selection": "2020-2022 only",
            "policy_selection": "2023-2024 only",
            "hidden_reveal": "2025-01 through 2026-09",
            "decision_time": "month start",
            "health_source": "only trades exited before month start",
            "orders": "next-bar market entry only",
            "money": "client 3% or 5% current-equity margin; fixed 10x",
        },
        "library_size": len(library),
        "policies_evaluated": len(policies),
        "selected_before_hidden_reveal": public_selected,
        "policy_frontier": [
            {key: value for key, value in item.items()
             if not key.startswith("_")} for item in policies[:20]],
        "selection_decisions": selected["_decisions"],
        "hidden_decisions": hidden_decisions,
        "hidden_2025_2026": hidden,
        "objective_pass": objective,
        "sources": {
            **candle_sources, str(BASE_REPORT): digest(BASE_REPORT),
            str(Path(__file__).resolve()): digest(Path(__file__).resolve()),
        },
        "limitations": [
            "The candidate library is price-only and highly correlated.",
            "Signals use Binance spot history and require separate X-Perp transfer testing.",
            "The 2026 hidden period ends with the available September archive.",
            "Gross returns exclude fees at user request.",
            "No backtest guarantees a future monthly return.",
        ],
    }
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True),
                   encoding="utf-8")
    print(json.dumps({
        "status": report["status"], "library_size": len(library),
        "policies_evaluated": len(policies),
        "selected": public_selected, "hidden": hidden,
        "objective_pass": objective,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
