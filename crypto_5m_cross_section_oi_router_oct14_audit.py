"""Cross-sectional continuation/reversion router with causal OI and flow state."""
from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

import crypto_5m_sweep_oi_flush_oct14_audit as oi_source
import crypto_5m_sweep_taker_flow_oct14_audit as flow_source
from src import research_crypto_replay as replay
from src.research_profit_gate import economic_gate


ROOT = Path("reports/audit_2026_09_08")
OUT = ROOT / "crypto_5m_cross_section_oi_router_oct14.json"
TARGETS = (.5, 1.0, 1.5, 2.0)
STOPS = (.5, .75, 1.0)
HOLDS = (24, 48, 72)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def features(candles: dict, metrics: dict) -> pd.DataFrame:
    pieces = []
    for symbol, candle in candles.items():
        close = pd.Series(np.asarray(candle["close"], dtype=float))
        times = np.asarray(candle["time"], dtype=np.int64)
        quote = np.asarray(candle["quote_volume"], dtype=float)
        taker_buy = np.asarray(candle["taker_buy_quote_volume"], dtype=float)
        flow_delta = np.divide(2 * taker_buy, quote, out=np.zeros_like(quote),
                               where=quote > 0) - 1
        signal = np.flatnonzero((times % 1800) == 1500)
        signal = signal[
            (signal + 1 < len(times))
            & (times[np.minimum(signal + 1, len(times) - 1)]
               - times[signal] == 300)]
        raw_metric = metrics[symbol]
        metric = pd.DataFrame({
            key: np.asarray(value, dtype=float)
            for key, value in raw_metric.items()}).set_index("time")
        oi = metric.sum_open_interest.replace(0, np.nan)
        oi_change_3 = oi.pct_change(3, fill_method=None).reindex(
            times[signal]).to_numpy()
        stamp = pd.to_datetime(times[signal + 1], unit="s", utc=True)
        pieces.append(pd.DataFrame({
            "symbol": symbol, "signal_time": times[signal],
            "entry_time": times[signal + 1], "entry_index": signal + 1,
            "year": stamp.year.astype(int),
            "month": (stamp.year * 100 + stamp.month).astype(int),
            "ret12": (close / close.shift(12) - 1).iloc[
                signal].to_numpy(float),
            "ret48": (close / close.shift(48) - 1).iloc[
                signal].to_numpy(float),
            "flow_delta": flow_delta[signal], "oi_change_3": oi_change_3,
        }))
    frame = pd.concat(pieces, ignore_index=True).replace(
        [np.inf, -np.inf], np.nan).dropna()
    grouped = frame.groupby("signal_time", sort=False)
    frame["available_symbols"] = grouped.symbol.transform("size")
    frame = frame[frame.available_symbols.ge(6)].copy()
    grouped = frame.groupby("signal_time", sort=False)
    frame["rank48"] = grouped.ret48.rank(pct=True)
    frame["dispersion48"] = grouped.ret48.transform("std")
    return frame[
        frame.rank48.le(.15) | frame.rank48.ge(.85)
    ].reset_index(drop=True)


def directional_frames(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rank_direction = np.where(frame.rank48.ge(.85), 1, -1)
    continuation = frame.copy()
    continuation["direction"] = rank_direction
    reversion = frame.copy()
    reversion["direction"] = -rank_direction
    for found in (continuation, reversion):
        found["flow_aligned"] = (found.direction * found.flow_delta)
        found["_row_id"] = np.arange(len(found))
    return continuation, reversion


def masks(continuation: pd.DataFrame,
          reversion: pd.DataFrame) -> dict[str, tuple[str, np.ndarray]]:
    return {
        "continuation_all": (
            "continuation", np.ones(len(continuation), dtype=bool)),
        "continuation_oi_ge_001_flow_ge_0": (
            "continuation", (continuation.oi_change_3.ge(.001)
                             & continuation.flow_aligned.ge(0)).to_numpy()),
        "continuation_oi_ge_0025_flow_ge_01": (
            "continuation", (continuation.oi_change_3.ge(.0025)
                             & continuation.flow_aligned.ge(.1)).to_numpy()),
        "reversion_all": (
            "reversion", np.ones(len(reversion), dtype=bool)),
        "reversion_oi_le_m001_flow_ge_0": (
            "reversion", (reversion.oi_change_3.le(-.001)
                          & reversion.flow_aligned.ge(0)).to_numpy()),
        "reversion_oi_le_m0025_flow_ge_01": (
            "reversion", (reversion.oi_change_3.le(-.0025)
                          & reversion.flow_aligned.ge(.1)).to_numpy()),
    }


def annual(values: np.ndarray, years: np.ndarray) -> dict:
    output = {}
    for year in range(2022, 2027):
        found = values[years == year]
        output[str(year)] = {
            "n": len(found),
            "win_rate_pct": round(100 * float(np.mean(found > 0)), 4)
                if len(found) else 0.0,
            "mean_signed_price_return_pct": round(
                100 * float(found.mean()), 6) if len(found) else 0.0,
        }
    return output


def normalize(frame: pd.DataFrame) -> list[dict]:
    rows = []
    for row in frame.to_dict("records"):
        reason = int(row["outcome_code"])
        won = float(row["signed_return"]) > 0
        rows.append({
            **row,
            "direction": "LONG" if int(row["direction"]) > 0 else "SHORT",
            "source": "cross_section_oi", "source_priority": 0,
            "module": "cross_section_oi", "module_priority": 0,
            "outcome": "TP" if won else "SL",
            "signed_price_return": float(row["signed_return"]),
            "target_hit": reason == 2,
        })
    return rows


def exact_year(frame: pd.DataFrame, year: int) -> dict:
    rows = [row for row in normalize(frame) if int(row["year"]) == year]
    accepted = replay.runtime_gate(rows)
    metrics = replay.money(accepted, .03)
    signed = [float(row["signed_price_return"]) for row in accepted]
    hits = sum(bool(row["target_hit"]) for row in accepted)
    metrics.update({
        "literal_target_hits": hits,
        "literal_target_hit_rate": round(100 * hits / len(accepted), 4)
            if accepted else 0.0,
        "mean_signed_price_return_pct": round(
            100 * float(np.mean(signed)), 6) if signed else 0.0,
        "economic_gate": economic_gate(
            signed, float(metrics.get("trades_per_calendar_day", 0)),
            market_days=30),
    })
    return metrics


def objective(value: dict) -> bool:
    return bool(value["n"] >= 100 and 75 <= value["wr"] <= 95
                and value["trades_per_calendar_day"] >= 3
                and value["negative_months"] == 0
                and value["dd_pct"] <= 5
                and value["economic_gate"]["passes"])


def component(value: dict) -> bool:
    return bool(value["n"] >= 100 and value["return_pct"] > 0
                and value["mean_signed_price_return_pct"] > 0
                and value["dd_pct"] <= 8)


def main() -> None:
    candles, sources = flow_source.load()
    metrics, metric_sources = oi_source.load_metrics()
    frame = features(candles, metrics)
    continuation, reversion = directional_frames(frame)
    definitions = masks(continuation, reversion)
    bases = {"continuation": continuation, "reversion": reversion}
    frontier = []
    outcome_cache = {}
    for target, stop, hold in itertools.product(TARGETS, STOPS, HOLDS):
        for family, base_frame in bases.items():
            resolved = replay.add_outcomes(
                base_frame.copy(), candles, target, stop, hold
            ).sort_values("_row_id")
            outcome_cache[(family, target, stop, hold)] = resolved
        for name, (family, mask) in definitions.items():
            resolved = outcome_cache[(family, target, stop, hold)]
            values = resolved.signed_return.to_numpy(float)[mask]
            years = resolved.year.to_numpy(int)[mask]
            yearly = annual(values, years)
            qualified = all(
                yearly[str(year)]["n"] >= 500
                and yearly[str(year)]["mean_signed_price_return_pct"] > 0
                for year in (2022, 2023, 2024))
            frontier.append({
                "variant": name, "family": family,
                "target_pct": target, "stop_pct": stop,
                "hold_bars": hold, "annual_raw": yearly,
                "qualified_before_2025": qualified,
                "pretest_minimum_mean_pct": min(
                    yearly[str(year)]["mean_signed_price_return_pct"]
                    for year in (2022, 2023, 2024)),
                "pretest_minimum_win_rate_pct": min(
                    yearly[str(year)]["win_rate_pct"]
                    for year in (2022, 2023, 2024)),
            })
    qualified = sorted(
        (item for item in frontier if item["qualified_before_2025"]),
        key=lambda item: (item["pretest_minimum_mean_pct"],
                          item["pretest_minimum_win_rate_pct"]), reverse=True)
    exact_frontier = []
    for item in qualified:
        mask = definitions[item["variant"]][1]
        resolved = outcome_cache[(
            item["family"], item["target_pct"],
            item["stop_pct"], item["hold_bars"])]
        selected = resolved.loc[mask].copy()
        pretest = {str(year): exact_year(selected, year)
                   for year in range(2022, 2025)}
        pretest_component = all(component(value) for value in pretest.values())
        exact_frontier.append({
            **item,
            "exact_pretest": pretest,
            "exact_component_before_2025": pretest_component,
            "exact_minimum_return_pct": min(
                value["return_pct"] for value in pretest.values()),
            "exact_maximum_drawdown_pct": max(
                value["dd_pct"] for value in pretest.values()),
            "exact_total_return_pct": sum(
                value["return_pct"] for value in pretest.values()),
        })
    exact_frontier.sort(key=lambda item: (
        item["exact_component_before_2025"],
        item["exact_minimum_return_pct"],
        item["exact_total_return_pct"],
        -item["exact_maximum_drawdown_pct"]), reverse=True)
    frozen = exact_frontier[0] if exact_frontier else None
    exact = checks = component_checks = None
    if frozen:
        mask = definitions[frozen["variant"]][1]
        resolved = outcome_cache[(
            frozen["family"], frozen["target_pct"],
            frozen["stop_pct"], frozen["hold_bars"])]
        selected = resolved.loc[mask].copy()
        exact = dict(frozen["exact_pretest"])
        exact.update({str(year): exact_year(selected, year)
                      for year in range(2025, 2027)})
        checks = {year: objective(value) for year, value in exact.items()}
        component_checks = {year: component(value)
                            for year, value in exact.items()}
    status = ("OBJECTIVE_PASS" if checks and all(checks.values())
              else "STABLE_COMPONENT" if component_checks
              and all(component_checks.values())
              else "REJECTED_PORTFOLIO" if frozen else "REJECTED_PRETEST")
    frontier.sort(key=lambda item: (
        item["pretest_minimum_mean_pct"],
        item["pretest_minimum_win_rate_pct"]), reverse=True)
    report = {
        "status": status, "ready_for_live_orders": False,
        "protocol": {
            "venue": "Binance USD-M futures", "decision_interval": "30m",
            "universe": sorted(candles),
            "event": "top and bottom 4h cross-sectional return rank",
            "entry": "next 5m market-open proxy",
            "routes": list(definitions), "targets_pct": list(TARGETS),
            "stops_pct": list(STOPS), "hold_bars": list(HOLDS),
            "selection": [2022, 2023], "validation": 2024,
            "test": 2025, "observed": 2026,
            "pretest_gate": (
                ">=500 rows and positive raw expectancy in every 2022-2024 year"),
            "portfolio": "frozen exposure, cooldown, cluster and daily loss rails",
            "money": "3% current-equity margin per accepted leg, fixed 10x gross",
            "orders": "market proxy only",
        },
        "feature_rows": len(frame), "tested_combinations": len(frontier),
        "qualified_before_2025": len(qualified),
        "exact_components_before_2025": sum(
            item["exact_component_before_2025"] for item in exact_frontier),
        "frozen_before_2025": frozen,
        "exact_frozen_portfolio": exact,
        "objective_checks": checks, "component_checks": component_checks,
        "top_frontier": frontier[:20],
        "top_exact_frontier": exact_frontier[:10],
        "decision": (
            "Runtime remains unchanged. A route may be retained for research "
            "only if exact portfolio results stay positive in every revealed year."),
        "sources": {
            **sources, **metric_sources,
            str(Path(flow_source.__file__)): digest(Path(flow_source.__file__)),
            str(Path(oi_source.__file__)): digest(Path(oi_source.__file__)),
            str(Path(replay.__file__)): digest(Path(replay.__file__)),
            str(Path(__file__)): digest(Path(__file__)),
        },
        "limitations": [
            "Both directional legs are evaluated as separate client-sized trades.",
            "Five-minute OI and taker flow are aggregates rather than order-book depth.",
            "2026 is observed stress evidence rather than a pristine holdout.",
        ],
    }
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({
        "status": status, "feature_rows": len(frame),
        "tested": len(frontier), "qualified": len(qualified),
        "frozen": frozen, "exact": exact,
        "objective_checks": checks, "component_checks": component_checks,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
