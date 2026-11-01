"""Causal adaptive exact-exit router with sealed post-2023 holdouts."""
from __future__ import annotations

import collections
from datetime import datetime, timezone
import hashlib
import heapq
import json
from pathlib import Path

import pandas as pd

import crypto_high_target_tree_oct5_audit as base
import crypto_profit_first_exact_health_router_oct30_audit as health
import crypto_profit_first_material_target_oct18_audit as frozen
import crypto_profit_first_runner_protection_oct30_audit as exact
from crypto_runtime_shadow_portfolio_oct3_audit import money, runtime_gate


ROOT = Path("reports/audit_2026_09_08")
ROWS = ROOT / "_tmp_exact_historical_rows.pkl"
OUT = ROOT / "crypto_profit_first_adaptive_exit_router_nov1.json"
BASE_POLICY = "a0p5_l0"
SCOPES = (
    ("btc_regime",), ("utc_session",), ("family",), ("direction",),
    ("btc_regime", "utc_session"),
)
LOOKBACKS = (20, 40, 80)
MINIMUMS = (5, 10, 20)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def policies(frame: pd.DataFrame) -> tuple[str, ...]:
    result = []
    for activation in exact.ACTIVATIONS:
        for lock in exact.LOCKS:
            key = exact.policy_key(activation, lock)
            if f"{key}_runner_r" in frame:
                result.append(key)
    return tuple(result)


def policy_r(row: dict, policy: str) -> float:
    runner = float(row[f"{policy}_runner_r"])
    return (.05 * .5 + .95 * runner
            if bool(row[f"{policy}_banked"]) else runner)


def route(frame: pd.DataFrame, *, scope: tuple[str, ...], lookback: int,
          minimum: int) -> tuple[pd.DataFrame, int]:
    """Choose using only hypothetical exits strictly before each entry."""
    options = policies(frame)
    histories = {}
    pending = []
    sequence = 0
    records = []
    causal_violations = 0
    ordered = frame.sort_values(
        ["entry_time", "symbol", "direction"], kind="mergesort")
    for row in ordered.to_dict("records"):
        now = float(row["entry_time"])
        while pending and pending[0][0] < now:
            exit_time, _, group, policy, value = heapq.heappop(pending)
            causal_violations += int(exit_time >= now)
            histories.setdefault(
                (group, policy), collections.deque(maxlen=lookback)).append(value)
        group = tuple(str(row[field]) for field in scope)
        scores = {}
        for policy in options:
            history = histories.get((group, policy), ())
            if len(history) >= minimum:
                scores[policy] = sum(history) / len(history)
        chosen = (max(scores, key=lambda key: (scores[key], key))
                  if scores else BASE_POLICY)
        result = dict(row)
        result.update({
            "chosen_exit_policy": chosen,
            "net_r": policy_r(row, chosen),
            "exit_time": float(row[f"{chosen}_exit_time"]),
            "runner_outcome": str(row[f"{chosen}_label"]),
        })
        result["signed_price_return"] = (
            result["net_r"] * float(result["risk_pct"]))
        records.append(result)
        for policy in options:
            sequence += 1
            heapq.heappush(pending, (
                float(row[f"{policy}_exit_time"]), sequence, group, policy,
                policy_r(row, policy)))
    standardized = base.standardized(pd.DataFrame(records), "adaptive_exact_exit")
    return pd.DataFrame(runtime_gate(standardized)), causal_violations


def metrics(frame: pd.DataFrame, periods: tuple[str, ...]) -> dict:
    rows = frame[frame.period.astype(str).isin(periods)].to_dict("records")
    return money(rows, .03)


def compact(value: dict) -> dict:
    return {key: value[key] for key in (
        "n", "wr", "return_pct", "dd_pct", "max_loss_streak",
        "negative_months", "trades_per_calendar_day", "monthly_return_pct")}


def development_pass(candidate: dict, baseline: dict) -> bool:
    return bool(candidate["return_pct"] >= baseline["return_pct"]
                and candidate["dd_pct"] <= 1.10 * baseline["dd_pct"]
                and candidate["negative_months"] <= baseline["negative_months"])


def main() -> None:
    raw = pd.read_pickle(ROWS)
    baseline_raw = health.current_outcomes(raw)
    baseline = pd.DataFrame(runtime_gate(
        base.standardized(baseline_raw, "adaptive_exit_baseline")))
    periods = {
        "selection_2022h1": frozen.SELECT,
        "confirmation_2022h2_2023": frozen.CONFIRM,
        "validation_2024": frozen.VALIDATION,
        "later_2025_2026": frozen.LATER,
    }
    baseline_metrics = {
        name: metrics(baseline, values) for name, values in periods.items()
    }
    frontier = []
    routed = {}
    for scope in SCOPES:
        for lookback in LOOKBACKS:
            for minimum in MINIMUMS:
                if minimum > lookback:
                    continue
                candidate, violations = route(
                    raw, scope=scope, lookback=lookback, minimum=minimum)
                selection = metrics(candidate, frozen.SELECT)
                confirmation = metrics(candidate, frozen.CONFIRM)
                passed = (
                    development_pass(
                        selection, baseline_metrics["selection_2022h1"])
                    and development_pass(
                        confirmation,
                        baseline_metrics["confirmation_2022h2_2023"]))
                minimum_gain = min(
                    selection["return_pct"]
                    - baseline_metrics["selection_2022h1"]["return_pct"],
                    confirmation["return_pct"]
                    - baseline_metrics["confirmation_2022h2_2023"]["return_pct"])
                config = {
                    "scope": list(scope), "lookback": lookback,
                    "minimum_completed_per_policy": minimum,
                }
                item = {
                    "config": config,
                    "development_pass": passed,
                    "development_min_return_gain": round(minimum_gain, 6),
                    "strictly_prior_exit_violations": violations,
                    "selection_2022h1": compact(selection),
                    "confirmation_2022h2_2023": compact(confirmation),
                }
                frontier.append(item)
                routed[json.dumps(config, sort_keys=True)] = candidate
    eligible = [item for item in frontier if item["development_pass"]]
    eligible.sort(key=lambda item: (
        -item["development_min_return_gain"],
        json.dumps(item["config"], sort_keys=True)))
    selected = eligible[0] if eligible else None
    holdouts = None
    passed = False
    if selected:
        candidate = routed[json.dumps(selected["config"], sort_keys=True)]
        holdouts = {
            "validation_2024": metrics(candidate, frozen.VALIDATION),
            "later_2025_2026": metrics(candidate, frozen.LATER),
        }
        passed = all(
            holdouts[name]["return_pct"] >= baseline_metrics[name]["return_pct"]
            and holdouts[name]["dd_pct"] <= baseline_metrics[name]["dd_pct"]
            and holdouts[name]["negative_months"]
                <= baseline_metrics[name]["negative_months"]
            for name in ("validation_2024", "later_2025_2026"))
    report = {
        "status": ("ADAPTIVE_EXIT_HISTORICAL_HOLDOUT_PASSED"
                   if passed else "REJECTED_ADAPTIVE_EXIT_LATER_HOLDOUT"
                   if selected else "REJECTED_ADAPTIVE_EXIT_DEVELOPMENT"),
        "ready_for_live_orders": False,
        "orders_possible": False,
        "runtime_change_allowed": False,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "protocol": {
            "choice": "highest mean exact R in the scoped rolling history",
            "causality": "only policy exits strictly before the new entry",
            "scopes": [list(value) for value in SCOPES],
            "lookbacks": list(LOOKBACKS), "minimums": list(MINIMUMS),
            "selection": [*frozen.SELECT, *frozen.CONFIRM],
            "sealed_holdouts": [*frozen.VALIDATION, *frozen.LATER],
        },
        "candidate_count": len(frontier),
        "development_pass_count": len(eligible),
        "baseline": {key: compact(value)
                     for key, value in baseline_metrics.items()},
        "selected_before_holdouts": selected,
        "holdouts": ({key: compact(value) for key, value in holdouts.items()}
                     if holdouts else None),
        "historical_holdout_pass": passed,
        "frontier": sorted(frontier, key=lambda item: (
            item["development_pass"], item["development_min_return_gain"]),
            reverse=True),
        "sources": {str(ROWS): digest(ROWS),
                    str(Path(__file__).resolve()): digest(Path(__file__).resolve())},
        "limitations": [
            "The best pre-2024 configuration loses its edge in 2025-2026.",
            "Hypothetical policies require continuous shadow tracking.",
            "Gross returns exclude fees at user request.",
        ],
    }
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({
        "status": report["status"], "selected": selected,
        "baseline_holdouts": {
            key: compact(baseline_metrics[key])
            for key in ("validation_2024", "later_2025_2026")},
        "holdouts": report["holdouts"],
    }, indent=2))


if __name__ == "__main__":
    main()
