"""Causal rolling route-health gate for the frozen material crypto policy.

Each BTC-regime/signal-family route is observed in shadow.  An entry can use
only outcomes whose exits are strictly earlier than that entry.  Lookback and
minimum expected move are selected on 2022-2023, then frozen before later
years and the refreshed X-Perp replay.
"""
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
from crypto_runtime_shadow_portfolio_oct3_audit import money, runtime_gate


ROOT = Path("reports/audit_2026_09_08")
OUT = ROOT / "crypto_profit_first_route_health_oct20.json"
ROWS_OUT = ROOT / "crypto_profit_first_route_health_xperp_oct20.pkl"
FRESH_OUT = ROOT / "crypto_profit_first_route_health_xperp_oct20_fresh.pkl"
LOOKBACKS = (10, 20, 40, 80)
MINIMUM_MEANS = (-0.001, 0.0, 0.001)
FIT, SELECT, CONFIRM = prior.FIT, prior.SELECT, prior.CONFIRM
VALIDATION, LATER = prior.VALIDATION, prior.LATER


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def candidates(frame: pd.DataFrame, routes: dict) -> pd.DataFrame:
    parts = []
    for route in routes.values():
        part = frame[
            frame.btc_regime.eq(route["btc_regime"])
            & frame.family.eq(route["family"])
            & (frame.model_probability >= float(route["threshold"]))].copy()
        if len(part):
            parts.append(target.barbell_frame(
                part, float(route["near_fraction"]), family.TARGET_R))
    if not parts:
        return frame.iloc[0:0].copy()
    rows = pd.DataFrame(base.standardized(
        pd.concat(parts, ignore_index=True),
        "profit_first_material_route_health"))
    rows["health_key"] = rows.btc_regime.astype(str) + "|" + rows.family.astype(str)
    return rows.sort_values(["entry_time", "symbol", "health_key"]).reset_index(drop=True)


def health_gate(frame: pd.DataFrame, lookback: int,
                minimum_mean: float) -> pd.DataFrame:
    """Use only route outcomes with exit_time strictly before each entry."""
    kept = []
    for _, group in frame.groupby("health_key", sort=False):
        pending: list[tuple[float, int, float]] = []
        completed: collections.deque[float] = collections.deque(maxlen=lookback)
        ordered = group.sort_values(["entry_time", "symbol"]).to_dict("records")
        for sequence, row in enumerate(ordered):
            now = float(row["entry_time"])
            while pending and pending[0][0] < now:
                _, _, value = heapq.heappop(pending)
                completed.append(value)
            minimum_observations = max(5, lookback // 2)
            health_mean = (sum(completed) / len(completed)) if completed else None
            row["route_health_observations"] = len(completed)
            row["route_health_mean"] = health_mean
            row["route_health_pass"] = bool(
                len(completed) < minimum_observations
                or health_mean >= minimum_mean)
            if row["route_health_pass"]:
                kept.append(row)
            heapq.heappush(pending, (float(row["exit_time"]), sequence,
                                    float(row["signed_price_return"])))
    if not kept:
        return frame.iloc[0:0].copy()
    return pd.DataFrame(kept).sort_values(
        ["entry_time", "symbol", "health_key"]).reset_index(drop=True)


def accepted(frame: pd.DataFrame, periods: tuple[str, ...], lookback: int,
             minimum_mean: float) -> pd.DataFrame:
    selected = frame[frame.period.isin(periods)].copy()
    return pd.DataFrame(runtime_gate(
        health_gate(selected, lookback, minimum_mean).to_dict("records"),
        loss_pause=3, hourly_cap=2))


def detailed(frame: pd.DataFrame, periods: tuple[str, ...], lookback: int,
             minimum_mean: float, margin: float) -> dict:
    def evaluate(labels: tuple[str, ...]) -> dict:
        return money(accepted(frame, labels, lookback, minimum_mean).to_dict("records"),
                     margin)
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
    return round(monthly - .20 * drawdown - .25 * negative, 8)


def main() -> None:
    model, schema, routes, _ = refresh.train_frozen()
    history = target.load_joined(family.TARGET_R)
    historical_scored = legacy.scored(history, model, schema)
    historical_candidates = candidates(historical_scored, routes)
    choices = []
    for lookback in LOOKBACKS:
        for minimum_mean in MINIMUM_MEANS:
            selection = detailed(historical_candidates, SELECT, lookback,
                                 minimum_mean, .05)
            confirmation = detailed(historical_candidates, CONFIRM, lookback,
                                    minimum_mean, .05)
            if development_ok(selection, confirmation):
                choices.append({
                    "lookback": lookback, "minimum_mean": minimum_mean,
                    "selection_2022h1": selection,
                    "confirmation_2022h2_2023": confirmation,
                    "score": score(selection, confirmation),
                })
    choices.sort(key=lambda item: item["score"], reverse=True)
    frozen = choices[0] if choices else None
    if frozen is None:
        raise ValueError("no development-stable route-health policy")
    lookback, minimum_mean = frozen["lookback"], frozen["minimum_mean"]
    validation = detailed(historical_candidates, VALIDATION, lookback,
                          minimum_mean, .05)
    later = detailed(historical_candidates, LATER, lookback, minimum_mean, .05)

    old = prior.join_direct_outcomes(refresh.OLD_FAR, refresh.OLD_NEAR)
    new = prior.join_direct_outcomes(refresh.NEW_FAR, refresh.NEW_NEAR)
    cutoff = float(old.entry_time.max())
    eligible = new[
        new.symbol.isin(base.ROBUST_SYMBOLS)
        & new.risk_pct.between(legacy.RISK_PCT_MIN, legacy.RISK_PCT_MAX,
                               inclusive="both")].copy()
    direct_scored = legacy.scored(eligible, model, schema)
    direct_scored["period"] = "direct"
    direct_candidates = candidates(direct_scored, routes)
    all_rows = accepted(direct_candidates, ("direct",), lookback, minimum_mean)
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
        "status": ("ROUTE_HEALTH_SHADOW_CANDIDATE" if pass_gate
                   else "ROUTE_HEALTH_REJECTED_OBJECTIVE"),
        "ready_for_live_orders": False,
        "protocol": {
            "signal_policy": "frozen Oct-18 union regime/family material route",
            "health_key": "BTC regime plus signal family",
            "causality": "only outcomes with exit_time strictly before entry",
            "selection": "2022-2023 only at fixed 5% margin",
            "lookbacks": list(LOOKBACKS), "minimum_means": list(MINIMUM_MEANS),
            "orders": "market only", "leverage": 10,
        },
        "development_frontier": choices,
        "frozen_before_2024": {key: frozen[key] for key in (
            "lookback", "minimum_mean", "score")},
        "validation_2024": validation, "later_2025_2026": later,
        "results": results, "economic_pass": pass_gate,
        "sources": {str(path): digest(path) for path in (
            refresh.FROZEN_REPORT, refresh.NEW_NEAR, refresh.NEW_FAR,
            Path(__file__))},
        "limitations": [
            "The untouched refresh is only a few calendar days.",
            "Filtered candidates continue as a virtual shadow stream for health updates.",
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
