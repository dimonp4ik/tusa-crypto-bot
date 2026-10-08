"""Target-specific material-move audit using derivatives positioning.

This is a bounded follow-up to the rejected 0.5% positioning model.  It tests
only literal 1.5% and 2.0% price moves, enters at the next 5-minute market
open, and keeps 2025/2026 sealed unless 2024 selection and validation pass.
Metric lookbacks and trade horizons that cross an archive gap are excluded.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numba as nb
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

import crypto_5m_regime_router_literal_oct6_audit as price
import crypto_futures_funding_regime_oct6_audit as futures_source
import crypto_futures_positioning_regime_oct7_audit as positioning
from src.research_profit_gate import economic_gate, minimum_binary_hit_rate_pct


ROOT = Path("reports/audit_2026_09_08")
OUT = ROOT / "crypto_futures_positioning_material_oct9.json"
TARGETS = {
    1.5: {"stop_pct": 1.0, "hold_bars": 288},
    2.0: {"stop_pct": 1.0, "hold_bars": 432},
}
MODEL = {"max_leaf_nodes": 15, "min_samples_leaf": 400,
         "l2_regularization": 6.0}
QUANTILES = (.90, .925, .95, .97, .98, .99)
PERIODS = {
    "fit_2023": (pd.Timestamp("2023-01-01", tz="UTC"),
                 pd.Timestamp("2024-01-01", tz="UTC")),
    "selection_2024_h1": (pd.Timestamp("2024-01-01", tz="UTC"),
                          pd.Timestamp("2024-07-01", tz="UTC")),
    "validation_2024_h2": (pd.Timestamp("2024-07-01", tz="UTC"),
                           pd.Timestamp("2025-01-01", tz="UTC")),
    "confirmation_2025": (pd.Timestamp("2025-01-01", tz="UTC"),
                          pd.Timestamp("2026-01-01", tz="UTC")),
    "untouched_2026": (pd.Timestamp("2026-01-01", tz="UTC"),
                       pd.Timestamp("2027-01-01", tz="UTC")),
}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def strict_metric_times(data: dict, lookback: int = 288) -> set[int]:
    times = np.asarray(data["time"], dtype=np.int64)
    order = np.argsort(times, kind="stable")
    times = times[order]
    times = times[np.r_[True, times[1:] != times[:-1]]]
    valid = np.zeros(len(times), dtype=bool)
    valid[lookback:] = (
        times[lookback:] - times[:-lookback] == lookback * 300)
    return set(times[valid].tolist())


def attach_metrics_strict(frame: pd.DataFrame, metrics: dict) -> pd.DataFrame:
    pieces = []
    for symbol, part in frame.groupby("symbol", sort=False):
        right = positioning.metric_frame(metrics[symbol])
        valid = strict_metric_times(metrics[symbol])
        right = right[right.metric_time.isin(valid)]
        left = part.sort_values("entry_time").copy()
        pieces.append(left.merge(
            right, left_on="entry_time", right_on="metric_time",
            how="inner", validate="one_to_one"))
    result = pd.concat(pieces, ignore_index=True)
    grouped = result.groupby("entry_time", sort=False)
    for source, target in (("oi_ret12", "oi_cross_z12"),
                           ("taker_mean3", "taker_cross_z"),
                           ("global_log", "global_cross_z")):
        mean = grouped[source].transform("mean")
        std = grouped[source].transform("std").replace(0, np.nan)
        result[target] = (result[source] - mean) / std
    return result.replace([np.inf, -np.inf], np.nan).dropna(
        subset=list(positioning.METRIC_FEATURES))


@nb.njit(cache=True)
def resolve(open_, high, low, close, times, indices, directions,
            target_pct, stop_pct, max_bars):
    count = len(indices)
    signed = np.empty(count, dtype=np.float64)
    exits = np.empty(count, dtype=np.int64)
    hits = np.zeros(count, dtype=np.int8)
    reasons = np.empty(count, dtype=np.int8)
    target_move, stop_move = target_pct / 100.0, stop_pct / 100.0
    for n in range(count):
        first, direction = indices[n], directions[n]
        entry = open_[first]
        target = entry * (1.0 + direction * target_move)
        stop = entry * (1.0 - direction * stop_move)
        result, reason, last = 0.0, 2, first
        resolved = False
        for j in range(first, first + max_bars):
            last = j
            stop_hit = low[j] <= stop if direction > 0 else high[j] >= stop
            target_hit = high[j] >= target if direction > 0 else low[j] <= target
            if stop_hit:
                fill = min(open_[j], stop) if direction > 0 else max(open_[j], stop)
                result = direction * (fill - entry) / entry
                reason, resolved = 0, True
                break
            if target_hit:
                result, hits[n], reason, resolved = target_move, 1, 1, True
                break
        if not resolved:
            result = direction * (close[last] - entry) / entry
            result = max(-stop_move, min(target_move, result))
        signed[n], exits[n], reasons[n] = result, times[last] + 300, reason
    return signed, exits, hits, reasons


def add_outcomes(frame: pd.DataFrame, candles: dict, target_pct: float,
                 stop_pct: float, max_bars: int) -> tuple[pd.DataFrame, int]:
    pieces, excluded = [], 0
    for symbol, part in frame.groupby("symbol", sort=False):
        candle = candles[symbol]
        times = np.asarray(candle["time"], dtype=np.int64)
        indices = part.entry_index.to_numpy(np.int64)
        eligible = indices + max_bars <= len(times)
        end = np.minimum(indices + max_bars - 1, len(times) - 1)
        eligible &= times[end] - times[indices] == (max_bars - 1) * 300
        excluded += int((~eligible).sum())
        found = part.loc[eligible].copy()
        if found.empty:
            continue
        indices = found.entry_index.to_numpy(np.int64)
        arrays = [np.asarray(candle[name], dtype=float)
                  for name in ("open", "high", "low", "close")]
        values, exits, hits, reasons = resolve(
            *arrays, times, indices, found.direction.to_numpy(np.int8),
            target_pct, stop_pct, max_bars)
        found["signed_return"], found["exit_time"] = values, exits
        found["target_hit"], found["exit_reason"] = hits.astype(bool), reasons
        pieces.append(found)
    return pd.concat(pieces, ignore_index=True), excluded


def period_mask(frame: pd.DataFrame, name: str) -> np.ndarray:
    start, end = PERIODS[name]
    stamp = pd.to_datetime(frame.entry_time, unit="s", utc=True)
    return ((stamp >= start) & (stamp < end)).to_numpy()


def evaluate(frame: pd.DataFrame, name: str, threshold: float) -> dict:
    subset = frame.loc[period_mask(frame, name)].copy()
    years = tuple(sorted(subset.year.unique().astype(int))) or (1900,)
    result = price.evaluate(subset, years, threshold)
    rows = price.portfolio_rows(subset, years, threshold)
    gate = economic_gate(
        [float(row["signed_price_return"]) for row in rows],
        float(result.get("trades_per_calendar_day", 0.0) or 0.0)) if rows else {
            "n": 0, "observed_trades_per_day": 0.0,
            "projected_monthly_return_pct": 0.0,
            "frequency_pass": False, "return_pass": False, "passes": False,
        }
    result["economic_gate"] = gate
    return result


def selection_ok(value: dict) -> bool:
    return bool(value["n"] >= 300 and value["return_pct"] > 0
                and value["literal_target_hit_rate"] >= 60
                and value["dd_pct"] <= 8
                and value["economic_gate"]["observed_trades_per_day"] >= 2.5
                and value["economic_gate"]["return_pass"])


def validation_ok(value: dict) -> bool:
    return bool(value["n"] >= 250 and value["return_pct"] > 0
                and value["literal_target_hit_rate"] >= 60
                and value["dd_pct"] <= 8
                and value["economic_gate"]["observed_trades_per_day"] >= 2.3
                and value["economic_gate"]["return_pass"])


def public(item: dict) -> dict:
    return {key: value for key, value in item.items()
            if key not in {"_frame", "_model"}}


def main() -> None:
    candles, _, price_sources = futures_source.load()
    metrics, metric_sources = positioning.load_metrics()
    raw = price.build_features(candles)
    enriched = attach_metrics_strict(raw, metrics)
    proposals = positioning.proposals(enriched)
    routes, frontier, gap_exclusions = [], [], {}
    for target_pct, contract in TARGETS.items():
        frame, excluded = add_outcomes(
            proposals, candles, target_pct,
            contract["stop_pct"], contract["hold_bars"])
        gap_exclusions[str(target_pct)] = excluded
        fit = period_mask(frame, "fit_2023")
        selection = period_mask(frame, "selection_2024_h1")
        schema = price.columns(frame.loc[fit])
        x_price, x_full = positioning.matrices(frame, schema)
        y = frame.target_hit.astype(int).to_numpy()
        for model_kind, matrix in (("price_only_control", x_price),
                                   ("positioning_plus_price", x_full)):
            model = HistGradientBoostingClassifier(
                max_iter=180, learning_rate=.06,
                max_leaf_nodes=MODEL["max_leaf_nodes"],
                min_samples_leaf=MODEL["min_samples_leaf"],
                l2_regularization=MODEL["l2_regularization"],
                random_state=20261009)
            model.fit(matrix[fit], y[fit])
            scored = frame.copy()
            scored["quality"] = model.predict_proba(matrix)[:, 1]
            for quantile in QUANTILES:
                threshold = float(scored.loc[selection, "quality"].quantile(quantile))
                selected = evaluate(scored, "selection_2024_h1", threshold)
                item = {
                    "target_pct": target_pct,
                    "stop_pct": contract["stop_pct"],
                    "hold_bars": contract["hold_bars"],
                    "model_kind": model_kind, "quantile": quantile,
                    "threshold": threshold,
                    "selection_2024_h1": selected,
                }
                frontier.append(item)
                if not selection_ok(selected):
                    continue
                validation = evaluate(scored, "validation_2024_h2", threshold)
                item["validation_2024_h2"] = validation
                if not validation_ok(validation):
                    continue
                item["score"] = round(
                    selected["return_pct"] + 2 * validation["return_pct"]
                    - selected["dd_pct"] - 2 * validation["dd_pct"], 8)
                item["_frame"], item["_model"] = scored, model
                routes.append(item)
    routes.sort(key=lambda item: item["score"], reverse=True)
    survivors = [item for item in routes
                 if item["model_kind"] == "positioning_plus_price"]
    frozen = survivors[0] if survivors else None
    confirmation = reveal = None
    if frozen:
        confirmation = evaluate(
            frozen["_frame"], "confirmation_2025", frozen["threshold"])
        if validation_ok(confirmation) and confirmation["economic_gate"]["passes"]:
            reveal = evaluate(
                frozen["_frame"], "untouched_2026", frozen["threshold"])
    frontier.sort(key=lambda item: (
        item["selection_2024_h1"]["economic_gate"]["projected_monthly_return_pct"],
        item["selection_2024_h1"]["literal_target_hit_rate"]), reverse=True)
    status = ("RESEARCH_CANDIDATE_AWAITING_DIRECT_FORWARD" if reveal and
              reveal["economic_gate"]["passes"] else "REJECTED")
    report = {
        "status": status,
        "protocol": {
            "fit": "2023", "selection": "2024-H1",
            "validation": "2024-H2", "confirmation": "2025",
            "untouched_reveal": "2026 only after all earlier gates",
            "entry": "next 5m perpetual market open",
            "targets": TARGETS,
            "metrics_availability": "archive timestamp plus 5 minutes",
            "metric_continuity": "288 consecutive 5m observations required",
            "outcome_continuity": "entire trade horizon must be gap-free",
            "money": "3% current equity margin, fixed 10x leverage",
            "fees": "excluded by user instruction",
        },
        "rows": {"raw_price": len(raw), "strict_enriched": len(enriched),
                 "proposals": len(proposals),
                 "outcome_gap_exclusions": gap_exclusions},
        "minimum_ideal_hit_rate_for_30pct_month": {
            str(target): round(minimum_binary_hit_rate_pct(
                target, contract["stop_pct"]), 6)
            for target, contract in TARGETS.items()
        },
        "qualified_positioning_routes_before_2025": len(survivors),
        "frozen_before_2025": public(frozen) if frozen else None,
        "confirmation_2025": confirmation,
        "untouched_2026": reveal,
        "selection_frontier": [public(item) for item in frontier[:24]],
        "sources": {**price_sources, **metric_sources,
                    str(Path(__file__)): digest(Path(__file__))},
        "limitations": [
            "The derivatives metrics archive does not provide the requested 2020-2021 history.",
            "The bounded model and threshold grid still creates selection risk.",
            "A surviving backtest would still require direct venue forward evidence.",
        ],
    }
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({
        "status": status, "rows": report["rows"],
        "qualified_positioning_routes_before_2025": len(survivors),
        "frozen_before_2025": report["frozen_before_2025"],
        "confirmation_2025": confirmation, "untouched_2026": reveal,
        "selection_frontier": report["selection_frontier"][:6],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
