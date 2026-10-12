"""Test whether causal market regimes separate barriers from liquidity magnets."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

import crypto_stop_cause_oct11_audit as cause
import crypto_valid_liquidity_level_oct13_audit as levels


OUT = cause.ROOT / "crypto_barrier_regime_oct13.json"
DEVELOPMENT = (2020, 2021, 2022)
VALIDATION = (2023,)
TEST = (2024,)
CONFIRMATION = (2025, 2026)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def derived(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    direction = out.direction.astype(float)
    out["btc_dir24"] = direction * out.btc_ret24
    out["btc_dir168"] = direction * out.btc_ret168
    out["stock_dir24"] = direction * out.ret24
    out["breadth_alignment"] = out.breadth24.where(direction > 0, 1 - out.breadth24)
    out["barrier_distance_ratio"] = (
        out.nearest_barrier_distance_pct / out.selected_target_pct).clip(0, 2)
    return out


def candidate_masks(frame: pd.DataFrame) -> list[tuple[str, pd.Series]]:
    conditions = {
        "btc24_against": frame.btc_dir24 <= -.005,
        "btc24_neutral": frame.btc_dir24.abs() < .005,
        "btc24_aligned": frame.btc_dir24 >= .005,
        "btc168_against": frame.btc_dir168 <= 0,
        "btc168_aligned": frame.btc_dir168 > 0,
        "stock24_against": frame.stock_dir24 <= 0,
        "stock24_aligned": frame.stock_dir24 > 0,
        "breadth_against": frame.breadth_alignment < .375,
        "breadth_aligned": frame.breadth_alignment >= .625,
        "atr_high": frame.atr14_pct >= .015,
        "atr_low": frame.atr14_pct < .015,
        "volume_abnormal": frame.abnormal_quote_volume.astype(bool),
        "efficiency_low": frame.eff24 < .25,
        "efficiency_high": frame.eff24 >= .25,
        "barrier_near": frame.barrier_distance_ratio <= .5,
        "barrier_far": frame.barrier_distance_ratio > .5,
        "reaction_strong": frame.barrier_reaction_pct >= 1.0,
    }
    barriers = {
        "fresh_swing": frame.fresh_swing_barrier.astype(bool),
        "repeated_pool": frame.repeated_liquidity_pool.astype(bool),
        "rejection_barrier": frame.confirmed_rejection_barrier.astype(bool),
    }
    output = []
    for barrier_name, barrier in barriers.items():
        for condition_name, condition in conditions.items():
            output.append((f"{barrier_name} AND {condition_name}", barrier & condition))
    return output


def accepted_flag_stats(frame: pd.DataFrame, mask: pd.Series, year: int) -> dict:
    tagged = frame.copy()
    tagged["candidate_block"] = mask.reindex(tagged.index, fill_value=False).astype(bool)
    accepted = cause.accepted_frame(tagged, (year,))
    flag = accepted.candidate_block.astype(bool)
    stop = accepted.exit_reason.eq(0)
    baseline_rate = float(stop.mean()) if len(accepted) else 0.0
    rate = float(stop[flag].mean()) if flag.any() else 0.0
    return {
        "accepted_n": len(accepted), "flagged_n": int(flag.sum()),
        "flagged_stop_rate_pct": round(100 * rate, 4),
        "baseline_stop_rate_pct": round(100 * baseline_rate, 4),
        "stop_rate_lift": round(rate / baseline_rate, 4) if baseline_rate else None,
    }


def stable_in_development(stats: dict[str, dict]) -> bool:
    pooled_n = sum(item["flagged_n"] for item in stats.values())
    informative = [item for item in stats.values() if item["flagged_n"] >= 10]
    return bool(
        pooled_n >= 40
        and len(informative) >= 2
        and sum(item["stop_rate_lift"] >= 1.0 for item in informative) >= 2
        and min(item["stop_rate_lift"] for item in informative) >= .80
    )


def dominates(candidate: dict, baseline: dict, minimum_n: int) -> bool:
    return bool(
        candidate["n"] >= minimum_n
        and candidate["trades_per_calendar_day"] >= 2.0
        and candidate["return_pct"] >= baseline["return_pct"]
        and candidate["literal_target_hit_rate"] >= baseline["literal_target_hit_rate"]
        and candidate["dd_pct"] <= baseline["dd_pct"]
        and candidate["negative_months"] <= baseline["negative_months"]
        and candidate["minimum_month_pct"] >= baseline["minimum_month_pct"]
        and candidate["max_loss_streak"] <= baseline["max_loss_streak"]
    )


def main() -> None:
    candles, sources = cause.load_candles()
    raw = pd.read_pickle(cause.ROWS)
    frame = derived(levels.level_features(cause.enrich(raw, candles), candles))
    baseline = {
        "development": cause.metrics(frame, DEVELOPMENT),
        "development_by_year": {str(year): cause.metrics(frame, (year,))
                                for year in DEVELOPMENT},
        "validation": cause.metrics(frame, VALIDATION),
        "test": cause.metrics(frame, TEST),
        "confirmation": {str(year): cause.metrics(frame, (year,))
                         for year in CONFIRMATION},
    }
    frontier = []
    eligible = []
    for name, mask in candidate_masks(frame):
        development_stats = {str(year): accepted_flag_stats(frame, mask, year)
                             for year in DEVELOPMENT}
        stable = stable_in_development(development_stats)
        candidate = frame.loc[~mask].copy()
        score = cause.metrics(candidate, DEVELOPMENT)
        removed = baseline["development"]["n"] - score["n"]
        passed = (stable and removed >= 40
                  and dominates(score, baseline["development"], 1800))
        item = {"rule": name, "development_flag_stats": development_stats,
                "stable_stop_lift": stable, "accepted_trades_removed": removed,
                "development": score, "development_pass": passed}
        frontier.append(item)
        if passed:
            rank = (baseline["development"]["negative_months"] - score["negative_months"],
                    score["minimum_month_pct"], score["return_pct"],
                    score["literal_target_hit_rate"], -score["dd_pct"])
            eligible.append((rank, name, candidate, item))

    frontier.sort(key=lambda item: (
        item["development_pass"], item["stable_stop_lift"],
        item["development"]["return_pct"], -item["development"]["dd_pct"]),
        reverse=True)
    frozen = validation = test = confirmation = None
    status = "REJECTED_DEVELOPMENT"
    if eligible:
        _, name, candidate, selected = max(eligible, key=lambda value: value[0])
        frozen = {"rule": name, "development": selected["development"],
                  "development_flag_stats": selected["development_flag_stats"],
                  "development_baseline": baseline["development"],
                  "eligible_rules": len(eligible)}
        validation = cause.metrics(candidate, VALIDATION)
        status = "REJECTED_VALIDATION"
        if dominates(validation, baseline["validation"], 500):
            test = cause.metrics(candidate, TEST)
            status = "REJECTED_TEST"
            if dominates(test, baseline["test"], 500):
                confirmation = {str(year): cause.metrics(candidate, (year,))
                                for year in CONFIRMATION}
                status = "BARRIER_REGIME_RESEARCH_CANDIDATE"

    report = {
        "status": status, "ready_for_live_orders": False,
        "protocol": {
            "input": "exact frozen OKX hourly target router plus causal valid levels",
            "development": list(DEVELOPMENT), "validation": list(VALIDATION),
            "test": list(TEST), "confirmation_only": list(CONFIRMATION),
            "candidate_count": len(frontier),
            "stability": "at least 40 flagged development trades, >=10 in at least two years, no informative year below 0.80 stop-rate lift",
            "market_entry": "next hourly open", "stop_pct": cause.STOP_PCT,
            "targets_pct": [.5, 1.0, 1.5, 2.0],
            "money": "3% current-equity margin, fixed 10x gross",
        },
        "baseline": baseline, "frontier": frontier,
        "frozen_before_validation": frozen, "validation": validation,
        "test": test, "confirmation": confirmation,
        "sources": {**sources, str(cause.ROWS): digest(cause.ROWS),
                    str(Path(cause.__file__)): digest(Path(cause.__file__)),
                    str(Path(levels.__file__)): digest(Path(levels.__file__)),
                    str(Path(__file__)): digest(Path(__file__))},
        "limitations": [
            "Regimes use closed hourly OHLCV features, not historical order-book depth.",
            "The candidate list is deliberately small and interpretable to limit multiple testing.",
            "A historical pass would still require fresh shadow outcomes before runtime activation.",
        ],
    }
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"status": status, "eligible": len(eligible),
                      "frozen": frozen, "validation": validation,
                      "test": test, "confirmation": confirmation,
                      "top_frontier": frontier[:10]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
