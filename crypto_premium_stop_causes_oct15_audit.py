"""Causal stop attribution for the frozen premium-dislocation route.

All explanatory flags use candles that had closed before the next-bar market
entry.  The script audits fixed market hypotheses; it does not fit a new model
to the confirmation years.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

import crypto_5m_regime_router_literal_oct6_audit as price
import crypto_futures_funding_regime_oct6_audit as funding
import crypto_premium_dislocation_oct15_audit as premium
import crypto_premium_exit_geometry_oct15_audit as geometry


ROOT = Path("reports/audit_2026_09_08")
PARENT = ROOT / "crypto_premium_dislocation_oct15.json"
GEOMETRY = ROOT / "crypto_premium_exit_geometry_oct15.json"
OUT = ROOT / "crypto_premium_stop_causes_oct15.json"
YEARS = (2020, 2021, 2022, 2023, 2024, 2025, 2026)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def causal_structure(frame: pd.DataFrame, candles: dict[str, dict]) -> pd.DataFrame:
    """Attach only information known at the close immediately before entry."""
    pieces = []
    for symbol, part in frame.groupby("symbol", sort=False):
        candle = candles[str(symbol)]
        open_ = np.asarray(candle["open"], dtype=float)
        high = np.asarray(candle["high"], dtype=float)
        low = np.asarray(candle["low"], dtype=float)
        close = np.asarray(candle["close"], dtype=float)
        volume = np.asarray(candle["volume"], dtype=float)
        found = part.copy()
        values = {name: [] for name in (
            "supportive_sweep", "adverse_rejection", "target_blocked",
            "directed_range_position", "low_volume", "shock_bar")}
        for row in found.itertuples(index=False):
            entry_i = int(row.entry_index)
            signal_i = entry_i-1
            start = max(0, signal_i-24)
            # Excludes the signal candle, so the level existed before the setup.
            prior_high = float(np.max(high[start:signal_i]))
            prior_low = float(np.min(low[start:signal_i]))
            entry = float(open_[entry_i])
            direction = int(row.direction)
            supportive = ((low[signal_i] < prior_low and close[signal_i] > prior_low)
                          if direction > 0 else
                          (high[signal_i] > prior_high and close[signal_i] < prior_high))
            adverse = ((high[signal_i] > prior_high and close[signal_i] < prior_high)
                       if direction > 0 else
                       (low[signal_i] < prior_low and close[signal_i] > prior_low))
            obstacle = ((prior_high/entry-1) if direction > 0
                        else (entry/prior_low-1))
            span = max(prior_high-prior_low, entry*1e-9)
            raw_position = (entry-prior_low)/span
            directed_position = raw_position if direction > 0 else 1-raw_position
            prior_volume = np.median(volume[max(0, signal_i-72):signal_i])
            signal_range = high[signal_i]-low[signal_i]
            prior_range = np.median(
                high[max(0, signal_i-72):signal_i]
                - low[max(0, signal_i-72):signal_i])
            values["supportive_sweep"].append(bool(supportive))
            values["adverse_rejection"].append(bool(adverse))
            values["target_blocked"].append(bool(0 <= obstacle <= .0075))
            values["directed_range_position"].append(float(directed_position))
            values["low_volume"].append(bool(
                prior_volume > 0 and volume[signal_i] < .55*prior_volume))
            values["shock_bar"].append(bool(
                prior_range > 0 and signal_range > 2.5*prior_range))
        for name, column in values.items():
            found[name] = column
        pieces.append(found)
    result = pd.concat(pieces, ignore_index=True)
    directed_premium = result.premium_delta4_z*result.direction
    result["premium_accelerating_against"] = directed_premium <= -1.0
    result["premium_extreme"] = result.premium_z672.abs() >= 2.5
    result["price_chasing_edge"] = result.directed_range_position >= .88
    result["against_btc_6h"] = result.btc_ret72*result.direction < 0
    same_time = result.groupby("entry_time").symbol.transform("size")
    result["crowded_timestamp"] = same_time >= 4
    return result


FLAGS = (
    "adverse_rejection", "target_blocked", "low_volume", "shock_bar",
    "premium_accelerating_against", "premium_extreme", "price_chasing_edge",
    "against_btc_6h", "crowded_timestamp", "supportive_sweep",
)


RULES = {
    "drop_adverse_rejection": ("adverse_rejection",),
    "drop_target_blocked": ("target_blocked",),
    "drop_low_volume": ("low_volume",),
    "drop_shock_bar": ("shock_bar",),
    "drop_premium_against": ("premium_accelerating_against",),
    "drop_extreme_premium": ("premium_extreme",),
    "drop_price_chase": ("price_chasing_edge",),
    "drop_against_btc": ("against_btc_6h",),
    "drop_crowded": ("crowded_timestamp",),
    "drop_level_or_rejection": ("target_blocked", "adverse_rejection"),
    "drop_chase_or_premium_against": (
        "price_chasing_edge", "premium_accelerating_against"),
}


def attribution(frame: pd.DataFrame, threshold: float) -> dict:
    result = {}
    for year in YEARS:
        rows = pd.DataFrame(price.portfolio_rows(frame, (year,), threshold))
        if rows.empty:
            result[str(year)] = {}
            continue
        baseline_stop = 100*rows.exit_reason.eq(0).mean()
        year_result = {}
        for flag in FLAGS:
            marked = rows[flag].astype(bool)
            flagged = rows[marked]
            clear = rows[~marked]
            stop_rate = 100*flagged.exit_reason.eq(0).mean() if len(flagged) else 0
            clear_stop = 100*clear.exit_reason.eq(0).mean() if len(clear) else 0
            year_result[flag] = {
                "n": int(len(flagged)),
                "share_pct": round(100*len(flagged)/len(rows), 4),
                "stop_rate_pct": round(float(stop_rate), 4),
                "clear_stop_rate_pct": round(float(clear_stop), 4),
                "stop_rate_lift_pp": round(float(stop_rate-clear_stop), 4),
                "target_hit_rate_pct": round(
                    100*float(flagged.target_hit.mean()), 4) if len(flagged) else 0,
                "mean_price_return_pct": round(
                    100*float(flagged.signed_return.mean()), 5) if len(flagged) else 0,
            }
        result[str(year)] = {
            "n": int(len(rows)), "baseline_stop_rate_pct": round(baseline_stop, 4),
            "flags": year_result,
        }
    return result


def rule_audit(frame: pd.DataFrame, threshold: float) -> dict:
    baseline = {str(year): price.evaluate(frame, (year,), threshold)
                for year in YEARS}
    results = {}
    for name, flags in RULES.items():
        blocked = np.logical_or.reduce(
            [frame[flag].to_numpy(bool) for flag in flags])
        candidate = frame.copy()
        candidate.loc[blocked, "quality"] = -np.inf
        yearly = {str(year): price.evaluate(candidate, (year,), threshold)
                  for year in YEARS}
        stable_confirmation = all(
            yearly[str(year)]["return_pct"] >= baseline[str(year)]["return_pct"]
            and yearly[str(year)]["stops"] <= baseline[str(year)]["stops"]
            and yearly[str(year)]["trades_per_calendar_day"] >= 2.0
            for year in (2023, 2024, 2025, 2026))
        results[name] = {
            "flags": flags, "stable_confirmation_improvement": stable_confirmation,
            "yearly": yearly,
        }
    return {"baseline": baseline, "rules": results}


def main() -> None:
    parent_raw = PARENT.read_bytes()
    geometry_raw = GEOMETRY.read_bytes()
    parent = json.loads(parent_raw)
    chosen_geometry = json.loads(geometry_raw)["frozen_geometry"]["config"]
    frozen = parent["frozen_before_test"]
    candles, _, candle_sources = funding.load()
    premium_data, premium_sources = premium.load_premium()
    raw = price.build_features(candles)
    enriched = premium.attach(raw, premium_data)
    proposals = premium.proposals(enriched)
    baseline = price.add_outcomes(proposals, candles)
    fit = baseline.year.isin((2020, 2021)).to_numpy()
    schema = price.columns(baseline.loc[fit])
    _, matrix = premium.matrices(baseline, schema)
    config = frozen["model"]
    model = HistGradientBoostingClassifier(
        max_iter=180, learning_rate=.06,
        max_leaf_nodes=config["max_leaf_nodes"],
        min_samples_leaf=config["min_samples_leaf"],
        l2_regularization=config["l2"], random_state=20261015)
    model.fit(matrix[fit], baseline.target_hit.astype(int).to_numpy()[fit])
    proposals = proposals.copy()
    proposals["quality"] = model.predict_proba(
        premium.matrices(proposals, schema)[1])[:, 1]
    frame = geometry.add_outcomes(
        proposals, candles, chosen_geometry["target_pct"],
        chosen_geometry["stop_pct"], chosen_geometry["max_bars"])
    frame = causal_structure(frame, candles)
    threshold = float(frozen["threshold"])
    factors = attribution(frame, threshold)
    filters = rule_audit(frame, threshold)
    stable = [name for name, value in filters["rules"].items()
              if value["stable_confirmation_improvement"]]
    report = {
        "status": "DIAGNOSTIC_ONLY_NO_RUNTIME_CHANGE",
        "ready_for_live_orders": False,
        "decision": "No fixed causal stop filter improved every confirmation year.",
        "protocol": {
            "signal_model_fit": [2020, 2021], "selection": 2022,
            "validation": [2023, 2024], "confirmation": [2025, 2026],
            "entry": "next 5m market open",
            "geometry": chosen_geometry,
            "causality": "all flags use bars closed before entry",
            "order_block_note": (
                "This premium route does not use order blocks; false OB cannot "
                "directly cause its signals. Prior-range obstacles and rejection "
                "sweeps are audited as causal structure proxies."),
        },
        "selected_signal_threshold": threshold,
        "factor_attribution": factors,
        "filter_ablation": filters,
        "stable_confirmation_rules": stable,
        "sources": {
            str(PARENT): hashlib.sha256(parent_raw).hexdigest(),
            str(GEOMETRY): hashlib.sha256(geometry_raw).hexdigest(),
            **candle_sources, **premium_sources,
            str(Path(__file__)): digest(Path(__file__)),
        },
        "limitations": [
            "OHLCV and premium-index history contain no historical order-book depth.",
            "Low candle volume is only a liquidity proxy, not executable depth.",
            "Binance futures remain a proxy for OKX X-Perp execution.",
        ],
    }
    OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(json.dumps({
        "status": report["status"], "stable_rules": stable,
        "baseline": filters["baseline"],
    }, indent=2))


if __name__ == "__main__":
    main()
