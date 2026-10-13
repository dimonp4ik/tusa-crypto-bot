"""Sequential audit of open-interest flushes around futures 5m sweeps."""
from __future__ import annotations

import hashlib
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

import crypto_5m_sweep_taker_flow_oct14_audit as flow
from src import research_crypto_replay as sweep
from src import research_crypto_replay as portfolio
from src.research_profit_gate import economic_gate


ROOT = Path("reports/audit_2026_09_08")
METRICS_MANIFEST = ROOT / "crypto_futures_metrics_history" / "manifest.json"
OUT = ROOT / "crypto_5m_sweep_oi_flush_oct14.json"
ENTRY_SPEC = (24, .10, .50, 1.25, .25)
TARGET_PCT = 1.0
STOP_PCT = .5
MAX_BARS = 24


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_metrics() -> tuple[dict, dict]:
    manifest = json.loads(METRICS_MANIFEST.read_text(encoding="utf-8"))
    metrics, sources = { }, {str(METRICS_MANIFEST): digest(METRICS_MANIFEST)}
    for symbol, item in sorted(manifest["symbols"].items()):
        path = Path(item["path"])
        raw = path.read_bytes()
        found = hashlib.sha256(raw).hexdigest()
        if found != item["sha256"]:
            raise ValueError(f"metrics source hash mismatch: {symbol}")
        metrics[symbol] = pickle.loads(raw)
        sources[str(path)] = found
    return metrics, sources


def enrich(entries: pd.DataFrame, candles: dict, metrics: dict) -> pd.DataFrame:
    pieces = []
    for symbol, part in entries.groupby("symbol", sort=False):
        candle = candles[symbol]
        indices = part.entry_index.to_numpy(np.int64)
        quote = np.asarray(candle["quote_volume"], dtype=float)
        taker_buy = np.asarray(candle["taker_buy_quote_volume"], dtype=float)
        delta = np.divide(2 * taker_buy, quote, out=np.zeros_like(quote),
                          where=quote > 0) - 1
        found = part.copy()
        found["flow_absorption"] = (
            -found.direction.to_numpy(np.int8) * delta[indices - 1])

        raw = metrics[symbol]
        metric = pd.DataFrame({key: np.asarray(value, dtype=float)
                               for key, value in raw.items()}).set_index("time")
        oi = metric.sum_open_interest.replace(0, np.nan)
        feature = pd.DataFrame(index=metric.index)
        for lag in (1, 3, 12):
            feature[f"oi_change_{lag}"] = oi.pct_change(
                lag, fill_method=None)
        ratio = metric.sum_taker_long_short_vol_ratio.replace(0, np.nan)
        signal_times = found.entry_time.to_numpy(np.int64) - 300
        aligned = feature.reindex(signal_times)
        found[["oi_change_1", "oi_change_3", "oi_change_12"]] = \
            aligned.to_numpy()
        metric_ratio = ratio.reindex(signal_times).to_numpy(float)
        found["metric_flow_absorption"] = (
            -found.direction.to_numpy(float) * np.log(metric_ratio))
        pieces.append(found)
    return pd.concat(pieces, ignore_index=True).replace(
        [np.inf, -np.inf], np.nan).dropna(subset=[
            "oi_change_1", "oi_change_3", "oi_change_12",
            "metric_flow_absorption"])


def variants(frame: pd.DataFrame) -> dict[str, pd.Series]:
    value = {}
    for threshold in (0.0, -.001, -.0025, -.005):
        value[f"oi1_le_{threshold}"] = frame.oi_change_1.le(threshold)
    for threshold in (0.0, -.0025, -.005, -.01):
        value[f"oi3_le_{threshold}"] = frame.oi_change_3.le(threshold)
    for threshold in (0.0, -.005, -.01, -.02):
        value[f"oi12_le_{threshold}"] = frame.oi_change_12.le(threshold)
    value["oi1_le_m001_flow_absorption_ge_0"] = (
        frame.oi_change_1.le(-.001) & frame.flow_absorption.ge(0))
    value["oi3_le_m0025_flow_absorption_ge_0"] = (
        frame.oi_change_3.le(-.0025) & frame.flow_absorption.ge(0))
    value["oi3_le_m005_metric_absorption_ge_01"] = (
        frame.oi_change_3.le(-.005)
        & frame.metric_flow_absorption.ge(.1))
    return value


def raw_year(frame: pd.DataFrame, year: int) -> dict:
    values = frame.loc[frame.year.eq(year), "signed_return"].to_numpy(float)
    return {
        "n": len(values),
        "win_rate_pct": round(100 * float(np.mean(values > 0)), 4)
            if len(values) else 0.0,
        "mean_signed_price_return_pct": round(
            100 * float(values.mean()), 6) if len(values) else 0.0,
    }


def normalize(frame: pd.DataFrame) -> list[dict]:
    rows = []
    for row in frame.to_dict("records"):
        reason = int(row["outcome_code"])
        won = float(row["signed_return"]) > 0
        rows.append({
            **row,
            "direction": "LONG" if int(row["direction"]) > 0 else "SHORT",
            "source": "oi_flush_sweep", "source_priority": 0,
            "module": "oi_flush_sweep", "module_priority": 0,
            # The shared resolver codes literal TP as 2 and profitable timeout
            # as 1.  Daily loss rails must react to every non-positive exit.
            "outcome": "TP" if won else "SL",
            "signed_price_return": float(row["signed_return"]),
            "target_hit": reason == 2,
        })
    return rows


def exact_year(frame: pd.DataFrame, year: int) -> dict:
    rows = [row for row in normalize(frame) if int(row["year"]) == year]
    accepted = portfolio.runtime_gate(rows)
    metrics = portfolio.money(accepted, .03)
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
    return bool(value["n"] >= 100
                and 75 <= value["wr"] <= 95
                and value["trades_per_calendar_day"] >= 3
                and value["negative_months"] == 0
                and value["dd_pct"] <= 5
                and value["economic_gate"]["passes"])


def main() -> None:
    candles, sources = flow.load()
    metrics, metric_sources = load_metrics()
    raw = pd.concat([
        sweep.features(symbol, candle, ENTRY_SPEC[0])
        for symbol, candle in candles.items()
    ], ignore_index=True)
    entries = enrich(sweep.apply_spec(raw, ENTRY_SPEC), candles, metrics)
    results = []
    resolved_by_name = {}
    for name, mask in variants(entries).items():
        resolved = sweep.add_outcomes(
            entries.loc[mask].copy(), candles,
            TARGET_PCT, STOP_PCT, MAX_BARS)
        resolved_by_name[name] = resolved
        annual = {str(year): raw_year(resolved, year)
                  for year in range(2022, 2027)}
        qualified = all(annual[str(year)]["n"] >= 200
                        and annual[str(year)][
                            "mean_signed_price_return_pct"] > 0
                        for year in (2022, 2023, 2024))
        score = min(annual[str(year)]["mean_signed_price_return_pct"]
                    for year in (2022, 2023, 2024))
        results.append({"name": name, "annual_raw": annual,
                        "qualified_before_2025": qualified,
                        "pretest_score": score})
    qualified = sorted(
        (item for item in results if item["qualified_before_2025"]),
        key=lambda item: item["pretest_score"], reverse=True)
    frozen = qualified[0] if qualified else None
    exact = None
    checks = None
    if frozen:
        resolved = resolved_by_name[frozen["name"]]
        exact = {str(year): exact_year(resolved, year)
                 for year in range(2022, 2027)}
        checks = {year: objective(value) for year, value in exact.items()}
    pretest_exact_pass = bool(checks and all(
        checks[str(year)] for year in (2022, 2023, 2024)))
    status = ("OBJECTIVE_PASS" if checks and all(checks.values())
              else "REJECTED_2025" if pretest_exact_pass
              else "REJECTED_PORTFOLIO_VALIDATION" if frozen
              else "REJECTED_DEVELOPMENT")
    report = {
        "status": status,
        "ready_for_live_orders": False,
        "protocol": {
            "venue": "Binance USD-M futures",
            "entry": "next 5m market-open proxy after closed sweep candle",
            "base_entry": ENTRY_SPEC, "target_pct": TARGET_PCT,
            "stop_pct": STOP_PCT, "maximum_hold_bars": MAX_BARS,
            "feature_selection": [2022, 2023, 2024],
            "test": 2025, "observed": 2026,
            "pretest_gate": "positive raw expectancy and >=200 rows in each 2022-2024 year",
            "portfolio": "frozen exposure, cooldown, cluster and daily loss rails",
            "money": "3% current-equity margin, fixed 10x gross",
        },
        "feature_rows": len(entries), "variants": results,
        "qualified_before_2025": len(qualified),
        "frozen_after_raw_pretest": frozen,
        "exact_frozen_portfolio": exact,
        "objective_checks": checks,
        "decision": (
            "No runtime change. The OI-flush route is rejected if its frozen "
            "pre-2025 rule fails the 2025 test or any complete objective gate."),
        "sources": {
            **sources, **metric_sources,
            str(Path(flow.__file__)): digest(Path(flow.__file__)),
            str(Path(sweep.__file__)): digest(Path(sweep.__file__)),
            str(Path(portfolio.__file__)): digest(Path(portfolio.__file__)),
            str(Path(__file__)): digest(Path(__file__)),
        },
        "limitations": [
            "Open-interest snapshots are five-minute aggregates and have recorded gaps.",
            "Liquidation is inferred from falling OI; individual forced orders are unavailable.",
            "2026 is observed stress evidence rather than a pristine holdout.",
        ],
    }
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({
        "status": report["status"],
        "qualified_before_2025": len(qualified),
        "frozen": frozen, "exact": exact, "checks": checks,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
