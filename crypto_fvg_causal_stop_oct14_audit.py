"""Causal FVG stop diagnosis with chronological freeze and exact portfolio replay."""
from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src import research_crypto_replay as replay
from src.research_profit_gate import economic_gate


ROOT = Path("reports/audit_2026_09_08")
SOURCE = ROOT / "multiyear_taker_raw.csv"
OUT = ROOT / "crypto_fvg_causal_stop_oct14.json"
DEVELOPMENT_YEAR = 2023
VALIDATION_YEAR = 2024
HOLDOUT_YEAR = 2025
OBSERVED_YEAR = 2026


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load() -> tuple[pd.DataFrame, dict[int, int]]:
    frame = pd.read_csv(SOURCE)
    frame["year"] = pd.to_datetime(
        frame.entry_time, unit="s", utc=True).dt.year.astype(int)
    frame["risk_pct"] = (frame.entry-frame.sl).abs()/frame.entry
    frame["target_move_pct"] = (frame.tp1-frame.entry).abs()/frame.entry*100
    frame["signed_price_return"] = frame.gross_r*frame.risk_pct
    direction = np.where(frame.direction.eq("LONG"), 1, -1)
    frame["aligned_1h"] = (
        (direction > 0) & frame.trend_1h.eq("bullish")
        | (direction < 0) & frame.trend_1h.eq("bearish"))
    frame["aligned_4h"] = (
        (direction > 0) & frame.trend_4h.eq("bullish")
        | (direction < 0) & frame.trend_4h.eq("bearish"))
    signals = frame.signals.fillna("").astype(str)
    tags = frame.score_tags.fillna("").astype(str)
    frame["has_liquidity_sweep"] = signals.str.contains("LiqSweep", regex=False)
    frame["has_engulfing"] = signals.str.contains("Engulfing", regex=False)
    frame["has_wick"] = signals.str.contains("BullWick|BearWick", regex=True)
    frame["has_macd_divergence"] = signals.str.contains("MACD_Div", regex=False)
    frame["has_nested_ob"] = tags.str.contains("NestedOB_1h", regex=False)
    frame["has_body_strength"] = tags.str.contains("BodyStrong", regex=False)
    frame["btc_strong_alignment"] = tags.str.contains("BTC+2", regex=False)
    frame["closed_confirmation_count"] = frame[[
        "has_liquidity_sweep", "has_engulfing", "has_wick",
        "has_macd_divergence", "has_body_strength",
    ]].sum(axis=1)
    frame = frame[
        frame.entry_source.eq("FVG")
        & frame.target_move_pct.between(.5, 2.0)
        & frame.risk_pct.gt(0)
        & np.isfinite(frame.signed_price_return)
        & frame.year.isin((2023, 2024, 2025, 2026))
    ].copy()
    # Multiple research configurations can emit the same causal setup. Freeze
    # one without consulting its outcome, using only the existing quality rank.
    frame = frame.sort_values(
        ["symbol", "entry_time", "direction", "quality_score", "mtf_score"],
        ascending=[True, True, True, False, False], kind="stable",
    ).drop_duplicates(["symbol", "entry_time", "direction"], keep="first")
    frame = frame.sort_values(["entry_time", "symbol"], kind="stable").reset_index(
        drop=True)
    coverage = {}
    for year, part in frame.groupby("year"):
        first = pd.Timestamp(float(part.entry_time.min()), unit="s", tz="UTC")
        last = pd.Timestamp(float(part.entry_time.max()), unit="s", tz="UTC")
        coverage[int(year)] = max(1, (last.date()-first.date()).days+1)
    return frame, coverage


def normalize(frame: pd.DataFrame) -> list[dict]:
    rows = []
    for row in frame.to_dict("records"):
        won = float(row["gross_r"]) > 0
        rows.append({
            **row,
            "source": "historical_fvg", "source_priority": 0,
            "module": "historical_fvg", "module_priority": 0,
            "outcome": "TP" if won else "SL",
        })
    return rows


def exact_year(frame: pd.DataFrame, year: int, coverage: dict[int, int]) -> dict:
    rows = normalize(frame[frame.year.eq(year)])
    accepted = replay.runtime_gate(rows)
    result = replay.money(accepted, .03)
    signed = [float(row["signed_price_return"]) for row in accepted]
    wins = [value for value in signed if value > 0]
    result["coverage_days"] = coverage.get(year, 0)
    result["trades_per_calendar_day"] = round(
        len(accepted)/coverage[year], 3) if coverage.get(year) else 0.0
    result["mean_signed_price_return_pct"] = round(
        100*float(np.mean(signed)), 6) if signed else 0.0
    result["mean_winning_price_move_pct"] = round(
        100*float(np.mean(wins)), 6) if wins else 0.0
    result["economic_gate"] = economic_gate(
        signed, result["trades_per_calendar_day"], market_days=30)
    return result


def atom_masks(frame: pd.DataFrame) -> dict[str, pd.Series]:
    masks: dict[str, pd.Series] = {"base": pd.Series(True, index=frame.index)}
    numeric = {
        "volume_ratio_ge": ("volume_ratio", (1.5, 2.0, 2.5, 3.0)),
        "extension_atr_ge": ("extension_atr", (.75, 1.0, 1.25, 1.5)),
        "entry_range_atr_ge": ("entry_range_atr", (.75, 1.0, 1.25, 1.5)),
        "quality_score_ge": ("quality_score", (80.0, 82.5, 85.0, 87.5)),
        "mtf_score_ge": ("mtf_score", (14.0, 15.0, 16.0, 17.0)),
        "zone_age_le": ("zone_age_bars", (1.0, 2.0, 3.0)),
        "bos_age_le": ("bos_candles_ago", (2.0, 3.0, 4.0)),
        "confirmation_count_ge": ("closed_confirmation_count", (1.0, 2.0, 3.0)),
    }
    for prefix, (column, thresholds) in numeric.items():
        for threshold in thresholds:
            masks[f"{prefix}_{threshold}"] = frame[column].ge(threshold)
    for name in (
        "aligned_1h", "aligned_4h", "has_liquidity_sweep", "has_engulfing",
        "has_wick", "has_macd_divergence", "has_nested_ob",
        "has_body_strength", "btc_strong_alignment",
    ):
        masks[name] = frame[name].fillna(False)
    masks["aligned_1h_4h"] = frame.aligned_1h & frame.aligned_4h
    masks["reversal_candle"] = frame.has_engulfing | frame.has_wick
    masks["reversal_or_sweep"] = (
        frame.has_engulfing | frame.has_wick | frame.has_liquidity_sweep)
    masks["volume2_structure"] = (
        frame.volume_ratio.ge(2) & frame.entry_range_atr.ge(1.0))
    masks["volume2_reversal"] = (
        frame.volume_ratio.ge(2) & (frame.has_engulfing | frame.has_wick))
    masks["aligned_reversal"] = (
        frame.aligned_4h & (frame.has_engulfing | frame.has_wick))
    for value in ("LONDON", "NEW_YORK", "OVERLAP", "OFF_HOURS", "DEAD_ZONE"):
        masks[f"session_{value.lower()}"] = frame.session.eq(value)
    for value in ("LONG", "SHORT"):
        masks[f"direction_{value.lower()}"] = frame.direction.eq(value)
    return masks


def score(value: dict) -> tuple:
    return (
        value["return_pct"] > 0,
        -value["negative_months"],
        value["return_pct"],
        value["wr"],
        -value["dd_pct"],
    )


def viable(value: dict, *, minimum_trades: int = 250) -> bool:
    return bool(value["n"] >= minimum_trades
                and value["trades_per_calendar_day"] >= 1.25
                and value["return_pct"] > 0
                and value["mean_signed_price_return_pct"] > 0)


def evaluate_rule(frame: pd.DataFrame, mask: pd.Series, coverage: dict,
                  years: tuple[int, ...]) -> dict[str, dict]:
    selected = frame[mask.fillna(False)]
    return {str(year): exact_year(selected, year, coverage) for year in years}


def main() -> None:
    frame, coverage = load()
    atoms = atom_masks(frame)
    development = []
    seen = set()
    for name, mask in atoms.items():
        key = tuple(frame.index[mask.fillna(False) & frame.year.eq(2023)])
        if key in seen:
            continue
        seen.add(key)
        result = exact_year(frame[mask.fillna(False)], 2023, coverage)
        if result["n"] >= 250:
            development.append({"variant": name, "atoms": [name],
                                "development": result})
    development.sort(key=lambda item: score(item["development"]), reverse=True)
    leaders = [item["variant"] for item in development[:16]]
    candidates = list(development)
    for first, second in itertools.combinations(leaders, 2):
        mask = atoms[first] & atoms[second]
        key = tuple(frame.index[mask.fillna(False) & frame.year.eq(2023)])
        if key in seen:
            continue
        seen.add(key)
        result = exact_year(frame[mask.fillna(False)], 2023, coverage)
        if result["n"] >= 250:
            candidates.append({
                "variant": f"{first}__and__{second}",
                "atoms": [first, second], "development": result,
            })
    candidates.sort(key=lambda item: score(item["development"]), reverse=True)
    shortlist = candidates[:30]
    for item in shortlist:
        mask = pd.Series(True, index=frame.index)
        for name in item["atoms"]:
            mask &= atoms[name]
        item["validation"] = exact_year(frame[mask], 2024, coverage)
        item["qualified_before_2025"] = (
            viable(item["development"]) and viable(item["validation"])
        )
        item["minimum_pretest_return_pct"] = min(
            item["development"]["return_pct"],
            item["validation"]["return_pct"])
        item["minimum_pretest_wr"] = min(
            item["development"]["wr"], item["validation"]["wr"])
    qualified = [item for item in shortlist if item["qualified_before_2025"]]
    qualified.sort(key=lambda item: (
        item["minimum_pretest_return_pct"], item["minimum_pretest_wr"],
        -max(item["development"]["dd_pct"], item["validation"]["dd_pct"])),
        reverse=True)
    frozen = qualified[0] if qualified else None
    exact = objective_checks = None
    if frozen:
        mask = pd.Series(True, index=frame.index)
        for name in frozen["atoms"]:
            mask &= atoms[name]
        exact = {
            "2023": frozen["development"], "2024": frozen["validation"],
            "2025": exact_year(frame[mask], 2025, coverage),
            "2026": exact_year(frame[mask], 2026, coverage),
        }
        objective_checks = {
            year: bool(value["n"] >= 100 and 75 <= value["wr"] <= 95
                       and value["trades_per_calendar_day"] >= 3
                       and value["negative_months"] == 0
                       and value["dd_pct"] <= 5
                       and value["economic_gate"]["passes"])
            for year, value in exact.items()
        }
    status = ("OBJECTIVE_PASS" if objective_checks
              and all(objective_checks.values())
              else "REJECTED_FORWARD" if frozen else "REJECTED_PRETEST")
    baseline = {str(year): exact_year(frame, year, coverage)
                for year in range(2023, 2027)}
    report = {
        "status": status, "ready_for_live_orders": False,
        "protocol": {
            "source": str(SOURCE), "setup": "FVG only",
            "planned_target_move_pct": "0.5 to 2.0",
            "orders": "next available market-entry records only",
            "position": "3% current-equity margin, fixed 10x gross",
            "development": 2023, "validation": 2024,
            "holdout": 2025, "observed": 2026,
            "candidate_generation": (
                "single causal atoms, then pairs among the 16 best 2023 atoms; "
                "30 variants reach 2024; only one frozen route reaches 2025"),
            "excluded_future_fields": [
                "post_sl_tp1", "post_sl_max_r", "mae_r", "mfe_tp1",
                "exit_bar", "exit_time", "outcome", "gross_r", "net_r"],
        },
        "rows_after_causal_dedup": len(frame),
        "coverage_days": coverage,
        "development_candidates": len(candidates),
        "validation_shortlist": len(shortlist),
        "qualified_before_2025": len(qualified),
        "baseline": baseline,
        "frozen_before_2025": frozen,
        "exact_frozen_portfolio": exact,
        "objective_checks": objective_checks,
        "top_pretest": qualified[:15],
        "decision": (
            "Do not alter runtime unless the frozen FVG rule passes both the "
            "untouched 2025 holdout and observed 2026 regime."),
        "sources": {
            str(SOURCE): digest(SOURCE),
            str(Path(replay.__file__)): digest(Path(replay.__file__)),
            str(Path(__file__)): digest(Path(__file__)),
        },
        "limitations": [
            "Each year is a historical multi-month slice rather than a full calendar year.",
            "The source setup generator was developed before this audit.",
            "Gross candle outcomes do not prove live fills or future profitability.",
        ],
    }
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({
        "status": status, "rows": len(frame),
        "development_candidates": len(candidates),
        "qualified": len(qualified), "baseline": baseline,
        "frozen": frozen, "exact": exact,
        "objective_checks": objective_checks,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
