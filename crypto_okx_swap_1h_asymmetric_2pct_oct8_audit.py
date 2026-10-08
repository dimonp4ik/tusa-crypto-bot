"""Chronological 2% target / 0.5% stop OKX one-hour audit.

This is a separate high-payoff route rather than an extension of the frozen
0.5% cascade.  Entries are semantic one-hour proposals, labels use a strict
target-before-stop test, and the probability threshold is chosen on 2022 only.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numba as nb
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

import crypto_5m_regime_router_literal_oct6_audit as portfolio
import crypto_okx_swap_1h_regime_oct7_audit as base
from src.research_profit_gate import economic_gate, minimum_binary_hit_rate_pct


OUT = Path("reports/audit_2026_09_08/crypto_okx_swap_1h_asymmetric_2pct_oct8.json")
TARGET_PCT = 2.0
STOP_PCT = .5
HOLD_BARS = 48
FIT_YEARS = (2020, 2021)
SELECTION_YEAR = 2022
VALIDATION_YEAR = 2023
HIDDEN_YEARS = (2024, 2025, 2026)
CONFIGS = (
    {"leaves": 15, "leaf": 500, "l2": 5.0},
    {"leaves": 31, "leaf": 1000, "l2": 10.0},
)
QUANTILES = (.50, .60, .70, .75, .80, .85, .875, .90, .925, .94,
             .95, .96, .97, .975, .98, .985, .99, .995)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@nb.njit(cache=True)
def resolve(open_, high, low, close, times, entries, directions):
    n = len(entries)
    signed = np.empty(n, dtype=np.float64)
    exits = np.empty(n, dtype=np.int64)
    hits = np.zeros(n, dtype=np.int8)
    reasons = np.empty(n, dtype=np.int8)
    target_move = TARGET_PCT / 100
    stop_move = STOP_PCT / 100
    for row in range(n):
        first = entries[row]
        direction = directions[row]
        entry = open_[first]
        target = entry * (1 + direction * target_move)
        stop = entry * (1 - direction * stop_move)
        last = first
        value = 0.0
        reason = 2
        resolved = False
        end = min(len(times), first + HOLD_BARS)
        for index in range(first, end):
            if index > first and times[index] - times[index - 1] != 3600:
                break
            last = index
            stop_hit = low[index] <= stop if direction > 0 else high[index] >= stop
            target_hit = high[index] >= target if direction > 0 else low[index] <= target
            # Stop first on an ambiguous hourly candle.
            if stop_hit:
                fill = min(open_[index], stop) if direction > 0 else max(open_[index], stop)
                value = direction * (fill / entry - 1)
                reason = 0
                resolved = True
                break
            if target_hit:
                value = target_move
                hits[row] = 1
                reason = 1
                resolved = True
                break
        if not resolved:
            value = direction * (close[last] / entry - 1)
            value = max(-stop_move, min(target_move, value))
        signed[row] = value
        exits[row] = times[last] + 3600
        reasons[row] = reason
    return signed, exits, hits, reasons


def add_outcomes(frame: pd.DataFrame, candles: dict) -> pd.DataFrame:
    parts = []
    for symbol, part in frame.groupby("symbol", sort=False):
        candle = candles[symbol]
        arrays = [np.asarray(candle[key], dtype=float)
                  for key in ("open", "high", "low", "close")]
        values, exits, hits, reasons = resolve(
            *arrays, np.asarray(candle["time"], dtype=np.int64),
            part.entry_index.to_numpy(np.int64),
            part.direction.to_numpy(np.int8))
        found = part.copy()
        found["signed_return"] = values
        found["exit_time"] = exits
        found["target_hit"] = hits.astype(bool)
        found["exit_reason"] = reasons
        found["selected_target_pct"] = TARGET_PCT
        parts.append(found)
    return pd.concat(parts, ignore_index=True)


def evaluate(frame: pd.DataFrame, years: tuple[int, ...], threshold: float) -> dict:
    metrics = portfolio.evaluate(frame, years, threshold)
    rows = portfolio.portfolio_rows(frame, years, threshold)
    gate = economic_gate(
        [float(row["signed_price_return"]) for row in rows],
        metrics["trades_per_calendar_day"], market_days=30)
    return {"portfolio": metrics, "economic_gate": gate}


def development_pass(value: dict) -> bool:
    metrics, gate = value["portfolio"], value["economic_gate"]
    return bool(metrics["n"] >= 800
                and metrics["literal_target_hit_rate"] >= 60
                and metrics["dd_pct"] <= 8
                and gate["passes"])


def rank(value: dict) -> tuple[float, float, float]:
    metrics, gate = value["portfolio"], value["economic_gate"]
    return (gate["projected_monthly_return_pct"],
            metrics["literal_target_hit_rate"], -metrics["dd_pct"])


def main() -> None:
    candles, sources = base.load()
    raw = base.build_features(candles)
    frame = add_outcomes(base.proposals(raw), candles)
    fit = frame.year.isin(FIT_YEARS).to_numpy()
    columns = base.schema(frame.loc[fit])
    x = base.matrix(frame, columns)
    labels = frame.target_hit.astype(int).to_numpy()
    model_summaries = []
    candidates = []
    for config in CONFIGS:
        model = HistGradientBoostingClassifier(
            max_iter=180, learning_rate=.06,
            max_leaf_nodes=config["leaves"],
            min_samples_leaf=config["leaf"],
            l2_regularization=config["l2"], random_state=20261008)
        model.fit(x[fit], labels[fit])
        scored = frame.copy()
        scored["quality"] = model.predict_proba(x)[:, 1]
        selection_scores = scored.loc[
            scored.year.eq(SELECTION_YEAR), "quality"]
        frontier = []
        for quantile in QUANTILES:
            threshold = float(selection_scores.quantile(quantile))
            value = evaluate(scored, (SELECTION_YEAR,), threshold)
            frontier.append({
                "quantile": quantile, "threshold": threshold,
                "selection_2022": value,
            })
        frontier.sort(key=lambda item: rank(item["selection_2022"]), reverse=True)
        for item in frontier[:10]:
            item["validation_2023"] = evaluate(
                scored, (VALIDATION_YEAR,), item["threshold"])
            item["development_pass"] = (
                development_pass(item["selection_2022"])
                and development_pass(item["validation_2023"]))
            candidates.append({"model": config, "_frame": scored, **item})
        model_summaries.append({
            "model": config,
            "fit_rows": int(fit.sum()),
            "fit_target_prevalence_pct": round(100 * float(labels[fit].mean()), 4),
            "frontier_top5": frontier[:5],
        })
        print(json.dumps({"model": config, "best": frontier[0]}, default=float),
              flush=True)
    eligible = [item for item in candidates if item["development_pass"]]
    eligible.sort(key=lambda item: (
        min(rank(item["selection_2022"])[0],
            rank(item["validation_2023"])[0]),
        min(rank(item["selection_2022"])[1],
            rank(item["validation_2023"])[1])), reverse=True)
    frozen = eligible[0] if eligible else None
    hidden = None
    if frozen is not None:
        hidden = {str(year): evaluate(
            frozen["_frame"], (year,), frozen["threshold"])
            for year in HIDDEN_YEARS}
    public_candidates = []
    for item in sorted(candidates, key=lambda candidate: (
            min(rank(candidate["selection_2022"])[0],
                rank(candidate["validation_2023"])[0])), reverse=True)[:20]:
        public_candidates.append({key: value for key, value in item.items()
                                  if key != "_frame"})
    status = "REJECTED_DEVELOPMENT"
    if frozen is not None:
        status = ("RESEARCH_CANDIDATE" if all(
            development_pass(value) for value in hidden.values())
                  else "REJECTED_HIDDEN")
    report = {
        "status": status,
        "ready_for_live_orders": False,
        "protocol": {
            "venue": "OKX global USDT swap", "bar": "1H",
            "entry": "next hourly market open", "target_pct": TARGET_PCT,
            "stop_pct": STOP_PCT, "hold_bars": HOLD_BARS,
            "fit": list(FIT_YEARS), "selection": SELECTION_YEAR,
            "validation": VALIDATION_YEAR,
            "hidden_only_after_development_pass": list(HIDDEN_YEARS),
            "money": "3% current equity margin, fixed 10x gross",
            "fees_and_slippage": "excluded per client research scope",
            "ambiguous_bar": "stop first",
        },
        "idealized_minimum_hit_rate_pct": minimum_binary_hit_rate_pct(
            TARGET_PCT, STOP_PCT, market_days=30),
        "models": model_summaries,
        "development_top20": public_candidates,
        "frozen_before_hidden": ({key: value for key, value in frozen.items()
                                  if key != "_frame"} if frozen else None),
        "hidden": hidden,
        "decision": ("No runtime change unless the route passes every stage, "
                     "including the economic gate."),
        "sources": {**sources, str(Path(__file__)): digest(Path(__file__))},
    }
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({
        "status": status,
        "idealized_minimum_hit_rate_pct": report["idealized_minimum_hit_rate_pct"],
        "best_development": public_candidates[:3],
        "hidden": hidden,
    }, indent=2, default=float), flush=True)


if __name__ == "__main__":
    main()
