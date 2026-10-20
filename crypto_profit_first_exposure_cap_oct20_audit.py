"""Causal concurrent-exposure caps for the frozen material crypto route."""
from __future__ import annotations

import collections
import hashlib
import heapq
import json
import math
from pathlib import Path

import pandas as pd

import crypto_05_2r_precision_barbell_oct5_audit as legacy
import crypto_high_target_tree_oct5_audit as base
import crypto_profit_first_material_barbell_oct18_audit as prior
import crypto_profit_first_material_regime_family_oct18_audit as family
import crypto_profit_first_material_target_oct18_audit as target
import crypto_profit_first_material_xperp_oct20_audit as refresh
import crypto_profit_first_route_health_oct20_audit as route_source
from crypto_runtime_shadow_portfolio_oct3_audit import money


ROOT = Path("reports/audit_2026_09_08")
OUT = ROOT / "crypto_profit_first_exposure_cap_oct20.json"
ROWS_OUT = ROOT / "crypto_profit_first_exposure_cap_xperp_oct20.pkl"
FRESH_OUT = ROOT / "crypto_profit_first_exposure_cap_xperp_oct20_fresh.pkl"
DIRECTION_CAPS = (1, 2, 3)
TOTAL_CAPS = (2, 3, 4, 6)
FIT, SELECT, CONFIRM = prior.FIT, prior.SELECT, prior.CONFIRM
VALIDATION, LATER = prior.VALIDATION, prior.LATER


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def exposure_gate(rows: list[dict], direction_cap: int,
                  total_cap: int) -> list[dict]:
    """Replay existing rails with explicit total and direction exposure caps."""
    pending: list[tuple[float, int, dict]] = []
    opened: dict[str, dict] = {}
    last: dict[tuple[str, str], float] = {}
    accepted_times: collections.deque[float] = collections.deque()
    output: list[dict] = []
    day = None
    streak = 0
    paused = False
    ordered = sorted(rows, key=lambda row: (
        float(row["entry_time"]), int(row["source_priority"]),
        str(row["symbol"]), int(row.get("module_priority", 999))))
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
        symbol, direction = str(row["symbol"]), str(row["direction"])
        if paused or symbol in opened or len(accepted_times) >= 2:
            continue
        if now - last.get((symbol, direction), -1e20) < 10800:
            continue
        if len(opened) >= total_cap:
            continue
        if sum(str(item["direction"]) == direction
               for item in opened.values()) >= direction_cap:
            continue
        opened[symbol] = row
        last[(symbol, direction)] = now
        accepted_times.append(now)
        output.append(row)
        heapq.heappush(pending, (float(row["exit_time"]), len(output), row))
    return output


def accepted(frame: pd.DataFrame, periods: tuple[str, ...], direction_cap: int,
             total_cap: int) -> pd.DataFrame:
    rows = frame[frame.period.isin(periods)].to_dict("records")
    return pd.DataFrame(exposure_gate(rows, direction_cap, total_cap))


def detailed(frame: pd.DataFrame, periods: tuple[str, ...], direction_cap: int,
             total_cap: int, margin: float) -> dict:
    def evaluate(labels: tuple[str, ...]) -> dict:
        rows = accepted(frame, labels, direction_cap, total_cap)
        return money(rows.to_dict("records"), margin)
    return {"all": evaluate(periods),
            "periods": {period: evaluate((period,)) for period in periods}}


def monthly_equivalent(value: dict) -> float:
    months = max(1, len(value["all"].get("monthly_return_pct", {})))
    factor = 1 + value["all"]["return_pct"] / 100
    return 100 * math.expm1(math.log(factor) / months) if factor > 0 else -100


def development_ok(selection: dict, confirmation: dict) -> bool:
    a, b = selection["all"], confirmation["all"]
    return bool(a["return_pct"] > 0 and b["return_pct"] > 0
                and a["dd_pct"] <= 15 and b["dd_pct"] <= 15
                and a["trades_per_calendar_day"] >= 3
                and b["trades_per_calendar_day"] >= 2
                and all(item["return_pct"] > 0
                        for item in confirmation["periods"].values()))


def score(selection: dict, confirmation: dict) -> float:
    monthly = min(monthly_equivalent(selection), monthly_equivalent(confirmation))
    drawdown = max(selection["all"]["dd_pct"], confirmation["all"]["dd_pct"])
    negative = selection["all"]["negative_months"] + confirmation["all"]["negative_months"]
    return round(monthly - .35 * drawdown - .25 * negative, 8)


def main() -> None:
    model, schema, routes, _ = refresh.train_frozen()
    history = target.load_joined(family.TARGET_R)
    historical_scored = legacy.scored(history, model, schema)
    historical_candidates = route_source.candidates(historical_scored, routes)
    choices = []
    for direction_cap in DIRECTION_CAPS:
        for total_cap in TOTAL_CAPS:
            if total_cap < direction_cap:
                continue
            selection = detailed(historical_candidates, SELECT, direction_cap,
                                 total_cap, .05)
            confirmation = detailed(historical_candidates, CONFIRM,
                                    direction_cap, total_cap, .05)
            if development_ok(selection, confirmation):
                choices.append({
                    "direction_cap": direction_cap, "total_cap": total_cap,
                    "selection_2022h1": selection,
                    "confirmation_2022h2_2023": confirmation,
                    "score": score(selection, confirmation),
                })
    choices.sort(key=lambda item: item["score"], reverse=True)
    frozen = choices[0] if choices else None
    if frozen is None:
        raise ValueError("no stable exposure-cap policy")
    direction_cap, total_cap = frozen["direction_cap"], frozen["total_cap"]
    validation = detailed(historical_candidates, VALIDATION, direction_cap,
                          total_cap, .05)
    later = detailed(historical_candidates, LATER, direction_cap, total_cap, .05)

    old = prior.join_direct_outcomes(refresh.OLD_FAR, refresh.OLD_NEAR)
    new = prior.join_direct_outcomes(refresh.NEW_FAR, refresh.NEW_NEAR)
    cutoff = float(old.entry_time.max())
    eligible = new[
        new.symbol.isin(base.ROBUST_SYMBOLS)
        & new.risk_pct.between(legacy.RISK_PCT_MIN, legacy.RISK_PCT_MAX,
                               inclusive="both")].copy()
    direct_scored = legacy.scored(eligible, model, schema)
    direct_scored["period"] = "direct"
    direct_candidates = route_source.candidates(direct_scored, routes)
    all_rows = accepted(direct_candidates, ("direct",), direction_cap, total_cap)
    fresh_rows = all_rows[all_rows.entry_time > cutoff].copy()
    all_rows.to_pickle(ROWS_OUT)
    fresh_rows.to_pickle(FRESH_OUT)
    results = {
        "all_refreshed_xperp": {
            f"margin_{int(100*m)}pct": refresh.diagnostics(all_rows, m)
            for m in refresh.MARGINS},
        "untouched_after_oct17_raw_cutoff": {
            f"margin_{int(100*m)}pct": refresh.diagnostics(fresh_rows, m)
            for m in refresh.MARGINS},
    }
    all5 = results["all_refreshed_xperp"]["margin_5pct"]
    fresh5 = results["untouched_after_oct17_raw_cutoff"]["margin_5pct"]
    pass_gate = bool(
        len(fresh_rows) >= 30 and fresh5["return_pct"] > 0
        and all5["dd_pct"] <= 10 and all5["negative_months"] == 0
        and all5["economic_30pct_month_gate"])
    report = {
        "status": ("EXPOSURE_CAP_SHADOW_CANDIDATE" if pass_gate
                   else "EXPOSURE_CAP_REJECTED_OBJECTIVE"),
        "ready_for_live_orders": False,
        "protocol": {
            "signal_policy": "frozen Oct-18 union regime/family material route",
            "selection": "2022-2023 only at fixed 5% margin",
            "direction_caps": list(DIRECTION_CAPS), "total_caps": list(TOTAL_CAPS),
            "existing_rails": "one symbol, 3h same-direction cooldown, two entries/hour, pause after three stops",
            "orders": "market only", "leverage": 10,
        },
        "development_frontier": choices,
        "frozen_before_2024": {key: frozen[key] for key in (
            "direction_cap", "total_cap", "score")},
        "validation_2024": validation, "later_2025_2026": later,
        "results": results, "economic_pass": pass_gate,
        "sources": {str(path): digest(path) for path in (
            refresh.FROZEN_REPORT, refresh.NEW_NEAR, refresh.NEW_FAR,
            Path(__file__))},
        "limitations": [
            "The untouched refresh is only a few calendar days.",
            "Exposure caps reject opportunities and can also remove recovery trades.",
            "Gross returns exclude fees at user request.",
        ],
    }
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"status": report["status"],
                      "frozen": report["frozen_before_2024"],
                      "validation": validation["all"],
                      "later": later["all"], "results": results,
                      "economic_pass": pass_gate}, indent=2), flush=True)


if __name__ == "__main__":
    main()
