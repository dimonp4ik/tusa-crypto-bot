"""Cross-era audit of causal rolling-VWAP crypto entries.

Two mechanisms are evaluated without sharing thresholds:

* range reversion after a large 8h/24h VWAP displacement and a closed reversal
  candle;
* trend continuation after price reclaims the 8h VWAP in the direction of the
  8h/24h VWAP trend.

Signals use only the completed 5m bar and enter at the next market open.  The
bounded grid uses literal 0.5%-2% price targets and pessimistic stop-first
ordering.  2025-2026 are not evaluated unless a rule first passes 2020-2022
development and separate 2023/2024 validation.
"""
from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

import crypto_5m_liquidation_sweep_actual_oct5_audit as resolver
import crypto_5m_tree_filter_audit as base
from crypto_runtime_shadow_portfolio_oct3_audit import money, runtime_gate


ROOT = Path("reports/audit_2026_09_08")
OUT = ROOT / "crypto_5m_vwap_regime_oct21.json"
ROWS_OUT = ROOT / "crypto_5m_vwap_regime_oct21.pkl"
VWAP_WINDOWS = (96, 288)
RANGE_DEVIATIONS = (1.0, 1.5, 2.0)
RANGE_EFFICIENCY_MAX = (.20, .35, .50)
TREND_EFFICIENCY_MIN = (.20, .35, .50)
TREND_SLOPE_MIN = (0.0, .25)
VOLUME_MINS = (.75, 1.25)
TARGETS = (.5, 1.0, 1.5, 2.0)
STOPS = (.5, 1.0)
MAX_BARS = (24, 48)
DEVELOPMENT_YEARS = (2020, 2021, 2022)
VALIDATION_YEARS = (2023, 2024)
HIDDEN_YEARS = (2025, 2026)
BASELINE_MARGIN = .03
LEVERAGE = 10.0
ECONOMIC_DAYS = 30
ECONOMIC_MINIMUM_DAILY = 3.0
ECONOMIC_TARGET_RETURN = .30


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def safe_ratio(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    result = np.full(len(numerator), np.nan, dtype=float)
    valid = (np.isfinite(numerator) & np.isfinite(denominator)
             & (denominator != 0))
    np.divide(numerator, denominator, out=result, where=valid)
    return result


def feature_frame(symbol: str, candle: dict) -> pd.DataFrame:
    """Create closed-bar features; every rolling window ends at this bar."""
    frame = pd.DataFrame({
        key: np.asarray(candle[key], dtype=float)
        for key in ("open", "high", "low", "close", "volume")
    })
    frame["time"] = np.asarray(candle["time"], dtype=np.int64)
    previous = frame.close.shift(1)
    tr = pd.concat(((frame.high - frame.low),
                    (frame.high - previous).abs(),
                    (frame.low - previous).abs()), axis=1).max(axis=1)
    # Exclude the signal candle from scale and volume baselines.  The signal's
    # own price/volume remains known at its close and is safe to compare.
    frame["atr"] = tr.shift(1).rolling(14).mean()
    typical = (frame.high + frame.low + frame.close) / 3.0
    for window in VWAP_WINDOWS:
        numerator = (typical * frame.volume).rolling(window).sum()
        denominator = frame.volume.rolling(window).sum().replace(0, np.nan)
        frame[f"vwap_{window}"] = numerator / denominator
    movement = frame.close.diff().abs().rolling(96).sum()
    frame["efficiency_96"] = (
        (frame.close - frame.close.shift(96)).abs()
        / movement.replace(0, np.nan)
    )
    frame["volume_ratio"] = (
        frame.volume / frame.volume.shift(1).rolling(20).median()
    )
    frame["symbol"] = symbol
    return frame


def rule_specs() -> list[dict]:
    rules = []
    for window, deviation, efficiency, volume in itertools.product(
            VWAP_WINDOWS, RANGE_DEVIATIONS, RANGE_EFFICIENCY_MAX,
            VOLUME_MINS):
        rules.append({
            "family": "range_vwap_reversion", "vwap_window": window,
            "deviation_atr_min": deviation,
            "efficiency_96_max": efficiency,
            "volume_ratio_min": volume,
        })
    for efficiency, slope, volume in itertools.product(
            TREND_EFFICIENCY_MIN, TREND_SLOPE_MIN, VOLUME_MINS):
        rules.append({
            "family": "trend_vwap_reclaim", "vwap_window": 96,
            "efficiency_96_min": efficiency,
            "trend_slope_atr_min": slope,
            "volume_ratio_min": volume,
        })
    return rules


def entries_for_rule(features: dict[str, pd.DataFrame], rule: dict) -> pd.DataFrame:
    pieces = []
    for symbol, frame in features.items():
        close = frame.close.to_numpy(float)
        open_ = frame.open.to_numpy(float)
        times = frame.time.to_numpy(np.int64)
        atr = frame.atr.to_numpy(float)
        efficiency = frame.efficiency_96.to_numpy(float)
        volume_ratio = frame.volume_ratio.to_numpy(float)
        fast = frame.vwap_96.to_numpy(float)
        slow = frame.vwap_288.to_numpy(float)
        direction = np.zeros(len(frame), dtype=np.int8)
        if rule["family"] == "range_vwap_reversion":
            vwap = frame[f"vwap_{rule['vwap_window']}"] .to_numpy(float)
            deviation = safe_ratio(close - vwap, atr)
            direction[deviation <= -rule["deviation_atr_min"]] = 1
            direction[deviation >= rule["deviation_atr_min"]] = -1
            toward = direction * (close - np.roll(close, 1)) > 0
            candle = direction * (close - open_) > 0
            mask = (
                (direction != 0) & toward & candle
                & (efficiency <= rule["efficiency_96_max"])
                & (volume_ratio >= rule["volume_ratio_min"])
            )
            score = np.abs(deviation) * volume_ratio
            displacement = deviation
            slope = np.full(len(frame), np.nan)
        else:
            trend = np.zeros(len(frame), dtype=np.int8)
            trend[fast > slow] = 1
            trend[fast < slow] = -1
            fast_previous = np.roll(fast, 1)
            close_previous = np.roll(close, 1)
            crossed = (
                (trend * (close_previous - fast_previous) <= 0)
                & (trend * (close - fast) > 0)
            )
            candle = trend * (close - open_) > 0
            slope = safe_ratio(
                trend * (fast - np.roll(fast, 12)), atr)
            mask = (
                (trend != 0) & crossed & candle
                & (efficiency >= rule["efficiency_96_min"])
                & (slope >= rule["trend_slope_atr_min"])
                & (volume_ratio >= rule["volume_ratio_min"])
            )
            direction = trend
            score = (1.0 + slope) * (1.0 + efficiency) * volume_ratio
            displacement = safe_ratio(close - fast, atr)
        entry_index = np.arange(len(frame), dtype=np.int64) + 1
        valid = (
            mask & np.isfinite(score) & np.isfinite(atr)
            & (entry_index < len(frame))
        )
        valid[-1] = False
        safe_entry = np.minimum(entry_index, len(frame) - 1)
        valid &= times[safe_entry] - times == 300
        signal = np.flatnonzero(valid)
        entry = signal + 1
        stamp = pd.to_datetime(times[entry], unit="s", utc=True)
        pieces.append(pd.DataFrame({
            "symbol": symbol,
            "signal_index": signal,
            "entry_index": entry,
            "entry_time": times[entry],
            "direction": direction[signal],
            "year": stamp.year.astype(int),
            "month": (stamp.year * 100 + stamp.month).astype(int),
            "score": score[signal],
            "vwap_displacement_atr": displacement[signal],
            "efficiency_96": efficiency[signal],
            "volume_ratio": volume_ratio[signal],
            "trend_slope_atr": slope[signal],
            "family": rule["family"],
        }))
    return pd.concat(pieces, ignore_index=True)


def raw_bundle(frame: pd.DataFrame, years: tuple[int, ...]) -> dict:
    result = {}
    for label, wanted in [("combined", years)] + [
            (str(year), (year,)) for year in years]:
        part = frame[frame.year.isin(wanted)]
        values = part.signed_return.to_numpy(float)
        if not len(part):
            result[label] = {
                "n": 0, "mean_signed_move_pct": 0.0,
                "sum_signed_move_pct": 0.0, "target_rate_pct": 0.0,
                "trades_per_calendar_day": 0.0,
            }
            continue
        first = pd.Timestamp(int(part.entry_time.min()), unit="s", tz="UTC")
        last = pd.Timestamp(int(part.entry_time.max()), unit="s", tz="UTC")
        days = max(1, (last.date() - first.date()).days + 1)
        result[label] = {
            "n": int(len(part)),
            "mean_signed_move_pct": round(100 * float(values.mean()), 6),
            "sum_signed_move_pct": round(100 * float(values.sum()), 4),
            "target_rate_pct": round(
                100 * float(np.mean(part.outcome_code.to_numpy(int) == 2)), 4),
            "trades_per_calendar_day": round(len(part) / days, 4),
        }
    return result


def raw_pass(bundle: dict) -> bool:
    return bool(
        bundle["combined"]["n"] >= 900
        and bundle["combined"]["trades_per_calendar_day"] >= 1
        and all(bundle[str(year)]["n"] >= 200
                and bundle[str(year)]["mean_signed_move_pct"] > 0
                for year in DEVELOPMENT_YEARS)
    )


def raw_rank(bundle: dict) -> tuple:
    annual = [bundle[str(year)] for year in DEVELOPMENT_YEARS]
    return (
        raw_pass(bundle),
        min(value["mean_signed_move_pct"] for value in annual),
        min(value["sum_signed_move_pct"] for value in annual),
        bundle["combined"]["mean_signed_move_pct"],
        bundle["combined"]["trades_per_calendar_day"],
    )


def row_records(frame: pd.DataFrame) -> list[dict]:
    ranked = frame.copy()
    ranked["source_priority"] = (
        ranked.groupby("entry_time").score.rank(
            method="first", ascending=False).astype(int) - 1
    )
    output = []
    for row in ranked.itertuples(index=False):
        value = float(row.signed_return)
        output.append({
            "symbol": str(row.symbol),
            "entry_time": int(row.entry_time),
            "exit_time": int(row.exit_time),
            "year": int(row.year),
            "direction": "LONG" if int(row.direction) > 0 else "SHORT",
            "source": "rolling_vwap",
            "source_priority": int(row.source_priority),
            "module": str(row.family),
            "module_priority": 0,
            "outcome": "SL" if value < 0 else "TP",
            "signed_price_return": value,
            "outcome_code": int(row.outcome_code),
        })
    return output


def add_economic_fields(result: dict, margin: float) -> dict:
    mean_move = float(result.get("mean_signed_move_pct", 0.0)) / 100.0
    frequency = float(result.get("trades_per_calendar_day", 0.0))
    periods = frequency * ECONOMIC_DAYS
    trade_factor = 1.0 + margin * LEVERAGE * mean_move
    projected = ((trade_factor ** periods - 1.0) * 100.0
                 if periods > 0 and trade_factor > 0 else 0.0)
    required = (
        ((1.0 + ECONOMIC_TARGET_RETURN) ** (1.0 / periods) - 1.0)
        / (margin * LEVERAGE) * 100.0
        if periods > 0 else None
    )
    mandatory = (
        ((1.0 + ECONOMIC_TARGET_RETURN) **
         (1.0 / (ECONOMIC_DAYS * ECONOMIC_MINIMUM_DAILY)) - 1.0)
        / (margin * LEVERAGE) * 100.0
    )
    result["projected_30d_return_pct"] = round(projected, 6)
    result["required_mean_signed_move_at_actual_frequency_pct"] = (
        round(required, 6) if required is not None else None)
    result["mandatory_three_daily_mean_signed_move_pct"] = round(
        mandatory, 6)
    result["economic_gap_multiple"] = (
        round(required / (mean_move * 100.0), 4)
        if required is not None and mean_move > 0 else None)
    result["economic_30pct_gate_pass"] = bool(projected >= 30.0)
    return result


def metrics(accepted: list[dict], margin: float) -> dict:
    result = money(accepted, margin)
    monthly = list(result.get("monthly_return_pct", {}).values())
    result["months"] = len(monthly)
    result["months_ge_30pct"] = sum(value >= 30 for value in monthly)
    result["months_positive"] = sum(value > 0 for value in monthly)
    result["minimum_month_pct"] = round(min(monthly), 4) if monthly else 0.0
    result["median_month_pct"] = round(
        float(np.median(monthly)), 4) if monthly else 0.0
    result["literal_target_rate_pct"] = round(
        100 * sum(int(row["outcome_code"]) == 2 for row in accepted)
        / len(accepted), 4) if accepted else 0.0
    result["mean_signed_move_pct"] = round(
        100 * float(np.mean([
            row["signed_price_return"] for row in accepted
        ])), 6) if accepted else 0.0
    return add_economic_fields(result, margin)


def exact_bundle(records: list[dict], years: tuple[int, ...],
                 margin: float = .03) -> dict:
    wanted = set(years)
    accepted = runtime_gate(
        [row for row in records if int(row["year"]) in wanted]
    )
    result = {"combined": metrics(accepted, margin)}
    for year in years:
        result[str(year)] = metrics(
            [row for row in accepted if int(row["year"]) == year], margin)
    return result


def development_ok(bundle: dict) -> bool:
    return bool(
        bundle["combined"]["n"] >= 900
        and bundle["combined"]["trades_per_calendar_day"] >= 1
        and bundle["combined"]["mean_signed_move_pct"] > 0
        and all(bundle[str(year)]["return_pct"] > 0
                and bundle[str(year)]["dd_pct"] <= 12
                for year in DEVELOPMENT_YEARS)
    )


def validation_ok(bundle: dict) -> bool:
    return all(
        bundle[str(year)]["n"] >= 150
        and bundle[str(year)]["return_pct"] > 0
        and bundle[str(year)]["dd_pct"] <= 8
        and bundle[str(year)]["trades_per_calendar_day"] >= .75
        for year in VALIDATION_YEARS
    )


def objective_ok(value: dict) -> bool:
    return bool(
        value["n"] >= 100 and value["return_pct"] > 0
        and value["dd_pct"] <= 5 and value["negative_months"] == 0
        and value["trades_per_calendar_day"] >= 3
        and value["months"] > 0
        and value["months_ge_30pct"] == value["months"]
    )


def exact_rank(bundle: dict) -> tuple:
    annual = [bundle[str(year)] for year in DEVELOPMENT_YEARS]
    monthly = [value for year in DEVELOPMENT_YEARS
               for value in bundle[str(year)]["monthly_return_pct"].values()]
    return (
        development_ok(bundle),
        min(monthly, default=-999.0),
        -sum(value["negative_months"] for value in annual),
        min(value["return_pct"] for value in annual),
        bundle["combined"]["mean_signed_move_pct"],
        -max(value["dd_pct"] for value in annual),
    )


def candidate_outcomes(candidate: dict, entries: dict[int, pd.DataFrame],
                       candles: dict[str, dict]) -> pd.DataFrame:
    return resolver.add_outcomes(
        entries[candidate["rule_index"]], candles,
        target=candidate["target_pct"], stop=candidate["stop_pct"],
        max_bars=candidate["max_bars"])


def public(candidate: dict | None) -> dict | None:
    if candidate is None:
        return None
    return {key: value for key, value in candidate.items()
            if key not in {"_records", "_resolved"}}


def main() -> None:
    candles, sources = base.load()
    features = {
        symbol: feature_frame(symbol, candle)
        for symbol, candle in candles.items()
    }
    rules = rule_specs()
    entry_counts = {}
    raw_candidates = []
    for rule_index, rule in enumerate(rules):
        frame = entries_for_rule(features, rule)
        entry_counts[str(rule_index)] = len(frame)
        if len(frame) < 900:
            continue
        for target, stop, max_bars in itertools.product(
                TARGETS, STOPS, MAX_BARS):
            resolved = resolver.add_outcomes(
                frame, candles, target=target, stop=stop,
                max_bars=max_bars)
            development = raw_bundle(resolved, DEVELOPMENT_YEARS)
            raw_candidates.append({
                "rule_index": rule_index, "rule": rules[rule_index],
                "target_pct": target, "stop_pct": stop,
                "max_bars": max_bars, "raw_development": development,
            })
    raw_candidates.sort(
        key=lambda item: raw_rank(item["raw_development"]), reverse=True)
    for raw_rank_index, candidate in enumerate(raw_candidates):
        candidate["raw_rank_index"] = raw_rank_index

    needed_rules = {
        candidate["rule_index"] for candidate in raw_candidates[:80]
    }
    entries = {
        rule_index: entries_for_rule(features, rules[rule_index])
        for rule_index in sorted(needed_rules)
    }

    exact_candidates = []
    for candidate in raw_candidates[:80]:
        resolved = candidate_outcomes(candidate, entries, candles)
        records = row_records(resolved)
        development = exact_bundle(records, DEVELOPMENT_YEARS)
        exact_candidates.append({
            **candidate, "development": development,
        })
    exact_candidates.sort(
        key=lambda item: exact_rank(item["development"]), reverse=True)

    validation_diagnostics = []
    frozen = None
    for candidate in exact_candidates[:30]:
        resolved = candidate_outcomes(candidate, entries, candles)
        records = row_records(resolved)
        validation = exact_bundle(records, VALIDATION_YEARS)
        diagnostic = {
            **candidate, "validation": validation,
            "validation_pass": validation_ok(validation),
        }
        validation_diagnostics.append(diagnostic)
        if development_ok(candidate["development"]) and validation_ok(validation):
            frozen = diagnostic
            frozen["_resolved"] = resolved
            frozen["_records"] = records
            break

    hidden = None
    frozen_margins = None
    objective_pass = False
    if frozen is not None:
        hidden = exact_bundle(frozen["_records"], HIDDEN_YEARS)
        frozen_margins = {
            f"margin_{int(100 * margin)}pct": {
                "development": exact_bundle(
                    frozen["_records"], DEVELOPMENT_YEARS, margin),
                "validation": exact_bundle(
                    frozen["_records"], VALIDATION_YEARS, margin),
                "hidden": exact_bundle(
                    frozen["_records"], HIDDEN_YEARS, margin),
            } for margin in (.03, .05)
        }
        objective_pass = all(
            objective_ok(hidden[str(year)]) for year in HIDDEN_YEARS
        )
        frozen["_resolved"].to_pickle(ROWS_OUT, compression="gzip")

    report = {
        "status": ("VWAP_REGIME_AWAITING_DIRECT_VENUE_FORWARD"
                   if objective_pass else "REJECTED_OBJECTIVE"),
        "ready_for_live_orders": False,
        "protocol": {
            "development": list(DEVELOPMENT_YEARS),
            "validation": list(VALIDATION_YEARS),
            "hidden_reveal": list(HIDDEN_YEARS),
            "families": ["range_vwap_reversion", "trend_vwap_reclaim"],
            "entry": "next 5m market open after closed signal candle",
            "exit": "literal 0.5%-2% target, 0.5%-1% stop, stop first",
            "selection": "raw development screen, exact development rank, validation gate",
            "money": "client 3% or 5% current-equity margin; fixed 10x",
            "economic_gate": (
                "+30% over 30 calendar days; 0.973137% constant signed "
                "price move per trade at 3% margin, 10x and 3 trades/day"
            ),
        },
        "rule_count": len(rules),
        "entry_counts": entry_counts,
        "candidate_count": len(raw_candidates),
        "raw_development_pass_count": sum(
            raw_pass(item["raw_development"]) for item in raw_candidates),
        "exact_development_pass_count": sum(
            development_ok(item["development"]) for item in exact_candidates),
        "frozen_before_hidden": public(frozen),
        "top20_exact_development": [
            public(item) for item in exact_candidates[:20]
        ],
        "validation_diagnostics_before_hidden": [
            public(item) for item in validation_diagnostics
        ],
        "hidden_3pct": hidden,
        "frozen_all_margins": frozen_margins,
        "objective_pass": objective_pass,
        "sources": {
            **sources,
            str(Path(resolver.__file__).resolve()):
                digest(Path(resolver.__file__).resolve()),
            str(Path(__file__).resolve()): digest(Path(__file__).resolve()),
        },
        "limitations": [
            "The bounded rule and exit grid creates selection risk.",
            "Binance spot history still requires direct X-Perp transfer.",
            "Gross results exclude fees at user request.",
        ],
    }
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True),
                   encoding="utf-8")
    print(json.dumps({
        "status": report["status"],
        "candidate_count": report["candidate_count"],
        "raw_development_pass_count": report["raw_development_pass_count"],
        "exact_development_pass_count": report["exact_development_pass_count"],
        "frozen": public(frozen),
        "hidden_3pct": hidden,
        "objective_pass": objective_pass,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
