"""Delayed-confirmation audit for genuine 5m sweep/reclaim reversals.

The earlier ``false_breakout_fade`` experiment reversed candles that closed
outside the broken level.  This audit starts from a genuine sweep: price must
trade beyond a prior 1-4 hour extreme and close back inside it.  One complete
5m candle must then confirm the reversal, and the trade enters at the following
5m market open.  All exits use literal price distances with pessimistic
stop-first ordering.
"""
from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

import crypto_5m_liquidation_sweep_actual_oct5_audit as sweep
import crypto_5m_tree_filter_audit as base
from crypto_runtime_shadow_portfolio_oct3_audit import money, runtime_gate


ROOT = Path("reports/audit_2026_09_08")
OUT = ROOT / "crypto_5m_sweep_delayed_confirmation_oct21.json"
ROWS_OUT = ROOT / "crypto_5m_sweep_delayed_confirmation_oct21.pkl"
ENTRY_SPECS = sweep.ENTRY_SPECS
CONFIRMATION_MODES = (
    "body_hold",
    "progress_hold",
    "body_progress_hold",
    "body_progress_hold_no_resweep",
)
TARGETS = (.5, 1.0, 1.5)
STOPS = (.5, .75, 1.0)
MAX_BARS = (24, 48)
DEVELOPMENT_YEARS = (2020, 2021, 2022)
VALIDATION_YEARS = (2023, 2024)
HIDDEN_YEARS = (2025, 2026)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def confirmation_entries(entries: pd.DataFrame, candles: dict[str, dict],
                         mode: str) -> pd.DataFrame:
    """Require one closed confirmation bar and move entry one bar later."""
    if mode not in CONFIRMATION_MODES:
        raise ValueError(f"unknown confirmation mode: {mode}")
    pieces = []
    for symbol, part in entries.groupby("symbol", sort=False):
        candle = candles[symbol]
        open_ = np.asarray(candle["open"], dtype=float)
        high = np.asarray(candle["high"], dtype=float)
        low = np.asarray(candle["low"], dtype=float)
        close = np.asarray(candle["close"], dtype=float)
        times = np.asarray(candle["time"], dtype=np.int64)
        signal = part.entry_index.to_numpy(np.int64) - 1
        confirmation = signal + 1
        actual_entry = signal + 2
        valid = actual_entry < len(times)
        safe_confirmation = np.minimum(confirmation, len(times) - 1)
        safe_entry = np.minimum(actual_entry, len(times) - 1)
        valid &= times[safe_confirmation] - times[signal] == 300
        valid &= times[safe_entry] - times[safe_confirmation] == 300
        if not valid.any():
            continue
        found = part.loc[valid].copy()
        signal = signal[valid]
        confirmation = confirmation[valid]
        actual_entry = actual_entry[valid]
        direction = found.direction.to_numpy(np.int8)
        lookback = int(found.lookback.iloc[0])

        previous = pd.Series(close).shift(1)
        prior_high = pd.Series(high).shift(1).rolling(lookback).max().to_numpy()
        prior_low = pd.Series(low).shift(1).rolling(lookback).min().to_numpy()
        level = np.where(direction > 0, prior_low[signal], prior_high[signal])
        body = direction * (close[confirmation] - open_[confirmation]) > 0
        progress = direction * (close[confirmation] - close[signal]) > 0
        hold = np.where(direction > 0, close[confirmation] > level,
                        close[confirmation] < level)
        no_resweep = np.where(direction > 0,
                              low[confirmation] >= low[signal],
                              high[confirmation] <= high[signal])
        if mode == "body_hold":
            accepted = body & hold
        elif mode == "progress_hold":
            accepted = progress & hold
        elif mode == "body_progress_hold":
            accepted = body & progress & hold
        else:
            accepted = body & progress & hold & no_resweep
        if not accepted.any():
            continue
        found = found.loc[accepted].copy()
        signal = signal[accepted]
        confirmation = confirmation[accepted]
        actual_entry = actual_entry[accepted]
        direction = direction[accepted]
        found["signal_index"] = signal
        found["confirmation_index"] = confirmation
        found["entry_index"] = actual_entry
        found["entry_time"] = times[actual_entry]
        found["confirmation_body_pct"] = 100 * direction * (
            close[confirmation] - open_[confirmation]) / open_[confirmation]
        found["confirmation_progress_pct"] = 100 * direction * (
            close[confirmation] - close[signal]) / close[signal]
        found["confirmation_mode"] = mode
        stamp = pd.to_datetime(times[actual_entry], unit="s", utc=True)
        found["year"] = stamp.year.astype(int)
        found["month"] = (stamp.year * 100 + stamp.month).astype(int)
        pieces.append(found)
    if not pieces:
        return entries.iloc[:0].copy()
    result = pd.concat(pieces, ignore_index=True)
    result["score"] = (
        result.sweep_depth_atr * result.wick_fraction * result.volume_ratio
        * (1.0 + result.confirmation_progress_pct.clip(lower=0.0))
    )
    return result


def row_records(frame: pd.DataFrame) -> list[dict]:
    ranked = frame.copy()
    ranked["source_priority"] = (
        ranked.groupby("entry_time").score.rank(
            method="first", ascending=False).astype(int) - 1
    )
    output = []
    for row in ranked.to_dict("records"):
        value = float(row["signed_return"])
        output.append({
            **row,
            "direction": "LONG" if float(row["direction"]) > 0 else "SHORT",
            "source": "sweep_delayed_confirmation",
            "module": "sweep_delayed_confirmation",
            "module_priority": 0,
            "outcome": "SL" if value < 0 else "TP",
            "signed_price_return": value,
        })
    return output


def evaluate(records: list[dict], years: tuple[int, ...], margin: float) -> dict:
    wanted = set(years)
    accepted = runtime_gate(
        [row for row in records if int(row["year"]) in wanted]
    )
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
        ])), 5) if accepted else 0.0
    return result


def period_bundle(records: list[dict], years: tuple[int, ...],
                  margin: float = .03) -> dict:
    return {
        "combined": evaluate(records, years, margin),
        **{str(year): evaluate(records, (year,), margin) for year in years},
    }


def development_ok(bundle: dict) -> bool:
    combined = bundle["combined"]
    annual = [bundle[str(year)] for year in DEVELOPMENT_YEARS]
    return bool(
        combined["n"] >= 900
        and combined["trades_per_calendar_day"] >= 1.0
        and combined["mean_signed_move_pct"] > 0
        and all(value["return_pct"] > 0 and value["dd_pct"] <= 12
                for value in annual)
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


def development_rank(bundle: dict) -> tuple:
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


def resolve_candidate(entry_frames: dict[tuple[int, str], pd.DataFrame],
                      candles: dict[str, dict], candidate: dict) -> tuple[pd.DataFrame, list[dict]]:
    key = (candidate["entry_spec_index"], candidate["confirmation_mode"])
    resolved = sweep.add_outcomes(
        entry_frames[key], candles, target=candidate["target_pct"],
        stop=candidate["stop_pct"], max_bars=candidate["max_bars"])
    return resolved, row_records(resolved)


def public(candidate: dict | None) -> dict | None:
    if candidate is None:
        return None
    return {key: value for key, value in candidate.items()
            if key not in {"_records", "_resolved"}}


def main() -> None:
    candles, sources = base.load()
    lookbacks = sorted({spec[0] for spec in ENTRY_SPECS})
    raw = pd.concat([
        sweep.features(symbol, candle, lookback)
        for symbol, candle in candles.items()
        for lookback in lookbacks
    ], ignore_index=True)
    entry_frames = {}
    entry_counts = {}
    for spec_index, spec in enumerate(ENTRY_SPECS):
        base_entries = sweep.apply_spec(raw, spec)
        for mode in CONFIRMATION_MODES:
            confirmed = confirmation_entries(base_entries, candles, mode)
            entry_frames[(spec_index, mode)] = confirmed
            entry_counts[f"{spec_index}:{mode}"] = len(confirmed)

    candidates = []
    for (spec_index, mode), entries in entry_frames.items():
        if len(entries) < 900:
            continue
        for target, stop, max_bars in itertools.product(
                TARGETS, STOPS, MAX_BARS):
            resolved = sweep.add_outcomes(
                entries, candles, target=target, stop=stop,
                max_bars=max_bars)
            records = row_records(resolved)
            development = period_bundle(records, DEVELOPMENT_YEARS)
            candidates.append({
                "entry_spec_index": spec_index,
                "entry": {
                    "lookback": ENTRY_SPECS[spec_index][0],
                    "min_sweep_atr": ENTRY_SPECS[spec_index][1],
                    "min_wick_fraction": ENTRY_SPECS[spec_index][2],
                    "min_volume_ratio": ENTRY_SPECS[spec_index][3],
                    "min_prior_move_atr": ENTRY_SPECS[spec_index][4],
                },
                "confirmation_mode": mode,
                "target_pct": target,
                "stop_pct": stop,
                "max_bars": max_bars,
                "development": development,
            })
    candidates.sort(key=lambda item: development_rank(item["development"]),
                    reverse=True)

    validation_diagnostics = []
    frozen = None
    for candidate in candidates[:50]:
        resolved, records = resolve_candidate(entry_frames, candles, candidate)
        validation = period_bundle(records, VALIDATION_YEARS)
        diagnostic = {**candidate, "validation": validation,
                      "validation_pass": validation_ok(validation)}
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
        hidden = period_bundle(frozen["_records"], HIDDEN_YEARS)
        frozen_margins = {
            f"margin_{int(100 * margin)}pct": {
                "development": period_bundle(
                    frozen["_records"], DEVELOPMENT_YEARS, margin),
                "validation": period_bundle(
                    frozen["_records"], VALIDATION_YEARS, margin),
                "hidden": period_bundle(
                    frozen["_records"], HIDDEN_YEARS, margin),
            } for margin in (.03, .05)
        }
        objective_pass = bool(
            development_ok(frozen["development"])
            and validation_ok(frozen["validation"])
            and all(objective_ok(hidden[str(year)])
                    for year in HIDDEN_YEARS)
        )
        frozen["_resolved"].to_pickle(ROWS_OUT, compression="gzip")

    report = {
        "status": ("SWEEP_CONFIRMATION_AWAITING_DIRECT_VENUE_FORWARD"
                   if objective_pass else "REJECTED_OBJECTIVE"),
        "ready_for_live_orders": False,
        "protocol": {
            "development": list(DEVELOPMENT_YEARS),
            "validation": list(VALIDATION_YEARS),
            "hidden_reveal": list(HIDDEN_YEARS),
            "signal": "closed 5m sweep beyond prior extreme and reclaim",
            "confirmation": "one additional closed 5m candle",
            "entry": "market open after confirmation candle",
            "exit": "literal 0.5%-1.5% target, 0.5%-1% stop, stop first",
            "selection": "development-ranked; validation used as pass/fail only",
            "money": "client 3% or 5% current-equity margin; fixed 10x",
        },
        "raw_events": len(raw),
        "entry_counts": entry_counts,
        "candidate_count": len(candidates),
        "development_pass_count": sum(
            development_ok(item["development"]) for item in candidates),
        "frozen_before_hidden": public(frozen),
        "validation_diagnostics_before_hidden": [
            public(item) for item in validation_diagnostics[:20]
        ],
        "top20_development_before_validation": [
            public(item) for item in candidates[:20]
        ],
        "hidden_3pct": hidden,
        "frozen_all_margins": frozen_margins,
        "objective_pass": objective_pass,
        "sources": {
            **sources,
            str(Path(sweep.__file__).resolve()):
                digest(Path(sweep.__file__).resolve()),
            str(Path(__file__).resolve()): digest(Path(__file__).resolve()),
        },
        "limitations": [
            "The bounded entry, confirmation and exit grid creates selection risk.",
            "Binance spot history still requires direct X-Perp transfer.",
            "Gross results exclude fees at user request.",
        ],
    }
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True),
                   encoding="utf-8")
    print(json.dumps({
        "status": report["status"],
        "candidate_count": report["candidate_count"],
        "development_pass_count": report["development_pass_count"],
        "frozen": public(frozen),
        "hidden_3pct": hidden,
        "objective_pass": objective_pass,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
