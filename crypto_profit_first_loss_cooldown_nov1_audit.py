"""Test short causal cooldowns after closed losses on the exact route.

The existing daily three-loss rail remains part of the baseline.  This audit
adds a rolling 1/3/6/12-hour pause after 1/2/3 consecutive losses.  Only
accepted candidate trades can later update the streak, so rejected outcomes
cannot leak into a live decision.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import heapq
import json
from pathlib import Path

import pandas as pd

import crypto_profit_first_exact_health_router_oct30_audit as health
import crypto_profit_first_exact_oct31_1700_forward as latest
from crypto_runtime_shadow_portfolio_oct3_audit import money


ROOT = Path("reports/audit_2026_09_08")
HISTORICAL = ROOT / "_tmp_exact_historical_rows.pkl"
DIRECT = latest.ROWS_OUT
OUT = ROOT / "crypto_profit_first_loss_cooldown_nov1.json"
ROWS_OUT = ROOT / "crypto_profit_first_loss_cooldown_nov1.pkl"
DEVELOPMENT = ("2022h1", "2022h2", "2023")
HIDDEN = ("2024", "2025", "2026")
POLICIES = tuple((losses, hours) for losses in (1, 2, 3)
                 for hours in (1, 3, 6, 12))


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def apply_cooldown(rows: list[dict], consecutive_losses: int,
                   cooldown_hours: int) -> list[dict]:
    pending: list[tuple[float, int, dict]] = []
    output: list[dict] = []
    streak = 0
    pause_until = float("-inf")
    sequence = 0
    for row in sorted(rows, key=lambda item: (
            float(item["entry_time"]), str(item["symbol"]))):
        now = float(row["entry_time"])
        # Match the runtime convention: an exit at the same timestamp is not
        # known before that timestamp's entries are considered.
        while pending and pending[0][0] < now:
            exit_time, _, done = heapq.heappop(pending)
            if float(done["net_r"]) <= 0:
                streak += 1
            else:
                streak = 0
            if streak >= consecutive_losses:
                pause_until = max(
                    pause_until, exit_time + cooldown_hours * 3600)
        if now < pause_until:
            continue
        output.append(row)
        sequence += 1
        heapq.heappush(pending, (float(row["exit_time"]), sequence, row))
    return output


def baseline_rows(frame: pd.DataFrame, period: str) -> list[dict]:
    return health.accepted(frame[frame.period.eq(period)], "loss_cooldown")


def compact(value: dict) -> dict:
    return {key: value[key] for key in (
        "n", "wr", "return_pct", "dd_pct", "max_loss_streak",
        "negative_months", "trades_per_calendar_day", "monthly_return_pct")}


def development_ok(candidate: dict, baseline: dict) -> bool:
    return bool(
        candidate["return_pct"] >= baseline["return_pct"]
        and candidate["dd_pct"] <= baseline["dd_pct"]
        and candidate["negative_months"] <= baseline["negative_months"]
        and candidate["max_loss_streak"] <= baseline["max_loss_streak"]
        and candidate["n"] >= .80 * baseline["n"])


def hidden_ok(candidate: dict, baseline: dict) -> bool:
    return bool(
        candidate["return_pct"] >= baseline["return_pct"]
        and candidate["dd_pct"] <= baseline["dd_pct"] + .5
        and candidate["negative_months"] <= baseline["negative_months"]
        and candidate["max_loss_streak"] <= baseline["max_loss_streak"] + 1)


def direct_metric(frame: pd.DataFrame, policy: tuple[int, int] | None) -> dict:
    rows = health.accepted(frame, "loss_cooldown_direct")
    if policy:
        rows = apply_cooldown(rows, *policy)
    full = pd.DataFrame(rows)
    october = full[full.entry_time.ge(
        int(pd.Timestamp("2026-10-01", tz="UTC").timestamp()))]
    return {"all_direct": latest.metrics(full, .03),
            "october": latest.metrics(october, .03)}


def main() -> None:
    historical = health.current_outcomes(pd.read_pickle(HISTORICAL))
    direct = pd.read_pickle(DIRECT)
    baseline_rows_by_period = {
        period: baseline_rows(historical, period)
        for period in DEVELOPMENT + HIDDEN}
    baseline = {period: money(rows, .03)
                for period, rows in baseline_rows_by_period.items()}
    baseline_direct = direct_metric(direct, None)
    candidates, eligible = [], []
    for losses, hours in POLICIES:
        metrics = {period: money(apply_cooldown(
            baseline_rows_by_period[period], losses, hours), .03)
            for period in DEVELOPMENT}
        passed = all(development_ok(metrics[period], baseline[period])
                     for period in DEVELOPMENT)
        item = {
            "consecutive_losses": losses, "cooldown_hours": hours,
            "development": {period: compact(metrics[period])
                            for period in DEVELOPMENT},
            "development_pass": passed,
        }
        candidates.append(item)
        if passed:
            eligible.append(item)
    eligible.sort(key=lambda item: (
        min(item["development"][period]["return_pct"]
            - baseline[period]["return_pct"] for period in DEVELOPMENT),
        -max(item["development"][period]["dd_pct"]
             for period in DEVELOPMENT)), reverse=True)
    selected = eligible[0] if eligible else None
    hidden = candidate_direct = None
    robust = False
    selected_direct = None
    if selected:
        policy = (int(selected["consecutive_losses"]),
                  int(selected["cooldown_hours"]))
        hidden = {period: money(apply_cooldown(
            baseline_rows_by_period[period], *policy), .03)
            for period in HIDDEN}
        candidate_direct = direct_metric(direct, policy)
        robust = bool(
            all(hidden_ok(hidden[period], baseline[period]) for period in HIDDEN)
            and hidden_ok(candidate_direct["all_direct"],
                          baseline_direct["all_direct"])
            and hidden_ok(candidate_direct["october"],
                          baseline_direct["october"]))
        if robust:
            selected_direct = apply_cooldown(
                health.accepted(direct, "loss_cooldown_direct"), *policy)
    if selected_direct is not None:
        pd.DataFrame(selected_direct).to_pickle(ROWS_OUT)

    report = {
        "status": ("LOSS_COOLDOWN_SHADOW_CHALLENGER" if robust
                   else "REJECTED_LOSS_COOLDOWN"),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "ready_for_live_orders": False,
        "orders_possible": False,
        "runtime_change_allowed": False,
        "protocol": {
            "base_rail": "existing causal daily pause after 3 closed losses",
            "candidate": "additional rolling cooldown after consecutive accepted losses",
            "development": list(DEVELOPMENT),
            "sealed_holdouts": list(HIDDEN) + ["direct X-Perp", "October 2026"],
            "policies": [{"losses": a, "hours": b} for a, b in POLICIES],
            "money": "3% current-equity margin, fixed 10x gross",
        },
        "baseline": {period: compact(value) for period, value in baseline.items()},
        "baseline_direct": baseline_direct,
        "candidate_count": len(candidates),
        "development_pass_count": len(eligible),
        "selected_before_holdouts": selected,
        "hidden": ({period: compact(value) for period, value in hidden.items()}
                   if hidden else None),
        "direct_xperp": candidate_direct,
        "robust_improvement": robust,
        "target_30pct_each_month_proven": False,
        "candidates": candidates,
        "sources": {str(path): digest(path) for path in (
            HISTORICAL, DIRECT, Path(__file__).resolve())},
        "limitations": [
            "Filtering an already gated stream does not refill slots freed by a cooldown.",
            "Gross returns exclude fees at user request.",
        ],
    }
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({
        "status": report["status"],
        "development_pass_count": len(eligible),
        "selected": selected,
        "hidden": report["hidden"],
        "direct": candidate_direct,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
