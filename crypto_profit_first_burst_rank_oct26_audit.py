"""Causal ranking and same-timestamp caps for the frozen Profit-First route.

Policy choices use 2022 through 2023 only.  The selected policy is then shown
on 2024, 2025-2026 and the untouched Oct-16-to-Oct-26 X-Perp forward.  The
fresh interval never participates in selection.
"""
from __future__ import annotations

import collections
import hashlib
import heapq
import json
import math
from pathlib import Path

import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

import crypto_05_2r_precision_barbell_oct5_audit as legacy
import crypto_high_target_tree_oct5_audit as base
import crypto_profit_first_material_barbell_oct18_audit as joiner
import crypto_profit_first_material_target_oct18_audit as frozen
import crypto_profit_first_material_target_oct26_forward as forward
from crypto_runtime_shadow_portfolio_oct3_audit import money


ROOT = Path("reports/audit_2026_09_08")
OUT = ROOT / "crypto_profit_first_burst_rank_oct26.json"
RANKERS = ("alphabetical", "probability_desc", "risk_asc", "risk_desc",
           "abs_z_desc", "efficiency_desc")
TIMESTAMP_DIRECTION_CAPS = (1, 2)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rank_value(row: dict, ranker: str) -> float:
    if ranker == "probability_desc":
        return -float(row["model_probability"])
    if ranker == "risk_asc":
        return float(row["risk_pct"])
    if ranker == "risk_desc":
        return -float(row["risk_pct"])
    if ranker == "abs_z_desc":
        return -float(row["abs_z"])
    if ranker == "efficiency_desc":
        return -float(row["eff_ratio"])
    return 0.0


def gate(rows: list[dict], ranker: str,
         timestamp_direction_cap: int) -> list[dict]:
    pending: list[tuple[float, int, dict]] = []
    opened: dict[str, dict] = {}
    last: dict[tuple[str, str], float] = {}
    accepted_times: collections.deque[float] = collections.deque()
    output: list[dict] = []
    day = None
    streak = 0
    paused = False
    stamp = None
    stamp_directions: collections.Counter[str] = collections.Counter()
    ordered = sorted(rows, key=lambda row: (
        float(row["entry_time"]), int(row["source_priority"]),
        rank_value(row, ranker), str(row["symbol"]),
        int(row.get("module_priority", 999))))
    for row in ordered:
        now = float(row["entry_time"])
        new_day = int(now // 86400)
        if new_day != day:
            day, streak, paused = new_day, 0, False
        while pending and pending[0][0] < now:
            _, _, done = heapq.heappop(pending)
            opened.pop(str(done["symbol"]), None)
            if int(float(done["exit_time"]) // 86400) == day:
                streak = streak + 1 if done["outcome"] == "SL" else 0
                if streak >= 3:
                    paused = True
        while accepted_times and accepted_times[0] <= now - 3600:
            accepted_times.popleft()
        if stamp != now:
            stamp = now
            stamp_directions.clear()
        symbol, direction = str(row["symbol"]), str(row["direction"])
        if (paused or symbol in opened or len(accepted_times) >= 2
                or stamp_directions[direction] >= timestamp_direction_cap):
            continue
        if now - last.get((symbol, direction), -1e20) < 10800:
            continue
        if sum(str(item["direction"]) == direction
               for item in opened.values()) >= 3:
            continue
        opened[symbol] = row
        last[(symbol, direction)] = now
        accepted_times.append(now)
        stamp_directions[direction] += 1
        output.append(row)
        heapq.heappush(pending, (float(row["exit_time"]), len(output), row))
    return output


def candidates(frame: pd.DataFrame, periods: tuple[str, ...], threshold: float,
               fraction: float) -> list[dict]:
    chosen = base.stage(frame, periods)
    chosen = chosen[chosen.model_probability >= threshold].copy()
    chosen = frozen.barbell_frame(chosen, fraction, 2.0)
    return base.standardized(chosen, "profit_first_material_2r")


def detailed(rows: list[dict], periods: tuple[str, ...], ranker: str,
             cap: int) -> dict:
    def one(labels: tuple[str, ...]) -> dict:
        subset = [row for row in rows if str(row["period"]) in labels]
        return money(gate(subset, ranker, cap), .03)
    return {"all": one(periods),
            "periods": {period: one((period,)) for period in periods}}


def monthly_equivalent(value: dict) -> float:
    months = max(1, len(value["all"].get("monthly_return_pct", {})))
    growth = 1 + float(value["all"]["return_pct"]) / 100
    return 100 * math.expm1(math.log(growth) / months) if growth > 0 else -100


def score(selection: dict, confirmation: dict) -> float:
    monthly = min(monthly_equivalent(selection),
                  monthly_equivalent(confirmation))
    drawdown = max(selection["all"]["dd_pct"], confirmation["all"]["dd_pct"])
    negative = (selection["all"]["negative_months"]
                + confirmation["all"]["negative_months"])
    return round(monthly - .20 * drawdown - .20 * negative, 8)


def development_ok(selection: dict, confirmation: dict) -> bool:
    return bool(
        selection["all"]["return_pct"] > 0
        and confirmation["all"]["return_pct"] > 0
        and selection["all"]["dd_pct"] <= 10
        and confirmation["all"]["dd_pct"] <= 10
        and selection["all"]["trades_per_calendar_day"] >= 3
        and confirmation["all"]["trades_per_calendar_day"] >= 1.5
        and all(item["return_pct"] > 0
                for item in confirmation["periods"].values()))


def compact(value: dict) -> dict:
    all_ = value["all"]
    return {key: all_[key] for key in (
        "n", "wr", "return_pct", "dd_pct", "max_loss_streak",
        "negative_months", "trades_per_calendar_day", "monthly_return_pct")}


def identity(rows: list[dict]) -> set[tuple]:
    return {(str(row["symbol"]), int(float(row["entry_time"])),
             str(row["direction"]), str(row["family"])) for row in rows}


def main() -> None:
    source = json.loads(forward.SOURCE_REPORT.read_text(encoding="utf-8"))
    policy = source["frozen_before_2024"]
    threshold, fraction = float(policy["threshold"]), float(policy["near_fraction"])
    historical = frozen.load_joined(2.0)
    fit = base.stage(historical, frozen.FIT)
    schema = base.design(fit)
    model = HistGradientBoostingClassifier(
        **frozen.MODEL, max_iter=150, l2_regularization=1.0,
        early_stopping=False, random_state=20260908)
    model.fit(base.matrix(fit, schema), fit.outcome.eq("TP2").astype(int))
    scored = legacy.scored(historical, model, schema)
    historical_rows = candidates(
        scored, (*frozen.SELECT, *frozen.CONFIRM, *frozen.VALIDATION,
                 *frozen.LATER), threshold, fraction)

    choices = []
    for ranker in RANKERS:
        for cap in TIMESTAMP_DIRECTION_CAPS:
            selection = detailed(historical_rows, frozen.SELECT, ranker, cap)
            confirmation = detailed(historical_rows, frozen.CONFIRM, ranker, cap)
            if development_ok(selection, confirmation):
                choices.append({
                    "ranker": ranker, "timestamp_direction_cap": cap,
                    "selection_2022h1": selection,
                    "confirmation_2022h2_2023": confirmation,
                    "score": score(selection, confirmation),
                })
    choices.sort(key=lambda item: item["score"], reverse=True)
    selected = choices[0]
    baseline = next(item for item in choices
                    if item["ranker"] == "alphabetical"
                    and item["timestamp_direction_cap"] == 2)

    direct = joiner.join_direct_outcomes(forward.FAR, forward.NEAR)
    eligible = direct[
        direct.symbol.isin(base.ROBUST_SYMBOLS)
        & direct.risk_pct.between(legacy.RISK_PCT_MIN, legacy.RISK_PCT_MAX,
                                  inclusive="both")].copy()
    direct_scored = legacy.scored(eligible, model, schema)
    direct_scored["period"] = "direct"
    direct_rows = candidates(direct_scored, ("direct",), threshold, fraction)
    baseline_direct = gate(direct_rows, "alphabetical", 2)
    saved = pd.read_pickle(forward.ROWS_OUT).to_dict("records")
    baseline_parity = identity(baseline_direct) == identity(saved)
    if not baseline_parity:
        raise ValueError("variant gate does not reproduce frozen baseline")

    ranker, cap = selected["ranker"], selected["timestamp_direction_cap"]
    selected_direct = gate(direct_rows, ranker, cap)
    original_cutoff = int(pd.Timestamp(
        source["fresh_direct_coverage"]["last_entry"]).timestamp())
    fresh_baseline = [row for row in baseline_direct
                      if float(row["entry_time"]) > original_cutoff]
    fresh_selected = [row for row in selected_direct
                      if float(row["entry_time"]) > original_cutoff]
    validation = detailed(historical_rows, frozen.VALIDATION, ranker, cap)
    later = detailed(historical_rows, frozen.LATER, ranker, cap)
    baseline_validation = detailed(
        historical_rows, frozen.VALIDATION, "alphabetical", 2)
    baseline_later = detailed(historical_rows, frozen.LATER, "alphabetical", 2)
    fresh = {
        "baseline": money(fresh_baseline, .03),
        "selected": money(fresh_selected, .03),
    }
    robust_improvement = bool(
        selected["score"] > baseline["score"]
        and validation["all"]["return_pct"] >= baseline_validation["all"]["return_pct"]
        and later["all"]["return_pct"] >= baseline_later["all"]["return_pct"]
        and fresh["selected"]["return_pct"] >= fresh["baseline"]["return_pct"]
        and fresh["selected"]["trades_per_calendar_day"] >= 3)
    report = {
        "status": ("SHADOW_CANDIDATE_REQUIRES_NEW_FORWARD"
                   if robust_improvement else "REJECTED_NO_ROBUST_IMPROVEMENT"),
        "ready_for_live_orders": False,
        "protocol": {
            "selection": "2022-2023 only",
            "validation": "2024 then 2025-2026",
            "final_diagnostic": "X-Perp after frozen Oct-16 cutoff",
            "rankers": list(RANKERS),
            "timestamp_direction_caps": list(TIMESTAMP_DIRECTION_CAPS),
            "hourly_cap": 2, "loss_pause": 3, "retune_on_forward": False,
        },
        "baseline_replay_parity": baseline_parity,
        "baseline": {
            "policy": {"ranker": "alphabetical", "timestamp_direction_cap": 2},
            "score": baseline["score"],
            "selection": compact(baseline["selection_2022h1"]),
            "confirmation": compact(baseline["confirmation_2022h2_2023"]),
            "validation_2024": compact(baseline_validation),
            "later_2025_2026": compact(baseline_later),
        },
        "selected_before_2024": {
            "policy": {"ranker": ranker, "timestamp_direction_cap": cap},
            "score": selected["score"],
            "selection": compact(selected["selection_2022h1"]),
            "confirmation": compact(selected["confirmation_2022h2_2023"]),
        },
        "validation_2024": compact(validation),
        "later_2025_2026": compact(later),
        "unseen_after_oct16_margin_3pct": fresh,
        "robust_improvement": robust_improvement,
        "development_frontier": [{
            "ranker": item["ranker"],
            "timestamp_direction_cap": item["timestamp_direction_cap"],
            "score": item["score"],
            "selection": compact(item["selection_2022h1"]),
            "confirmation": compact(item["confirmation_2022h2_2023"]),
        } for item in choices],
        "sources": {str(path): digest(path) for path in (
            forward.SOURCE_REPORT, forward.FAR, forward.NEAR, Path(__file__))},
    }
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True),
                   encoding="utf-8")
    print(json.dumps({
        "status": report["status"],
        "baseline": report["baseline"],
        "selected_before_2024": report["selected_before_2024"],
        "validation_2024": report["validation_2024"],
        "later_2025_2026": report["later_2025_2026"],
        "unseen_after_oct16_margin_3pct": fresh,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
