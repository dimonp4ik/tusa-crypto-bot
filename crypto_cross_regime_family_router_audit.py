"""Cross-regime family whitelist for the complementary crypto systems.

Coarse component-specific groups must have positive signed price return in
both 2022 and 2023.  Group definitions and minimum sample sizes are selected
on 2024, then held fixed for the 2025 acceptance and 2026 hidden checks.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

import crypto_5m_regime_router_literal_oct6_audit as hourly_portfolio
import crypto_complementary_exact_portfolio_oct24_audit as portfolio
import crypto_okx_swap_1h_forward_append_oct16_audit as append
import crypto_okx_swap_1h_forward_append_oct8_audit as legacy
import crypto_okx_swap_1h_stable_direction_forward_oct22_audit as forward
import crypto_okx_swap_1h_stable_direction_oct22_audit as stable
import crypto_okx_swap_1h_stable_material_exit_oct22_audit as hourly


ROOT = portfolio.ROOT
TARGET_ROWS = ROOT / "crypto_profit_first_walkforward_target_router_oct18.pkl"
HOURLY_REPORT = ROOT / "crypto_okx_swap_1h_stable_material_exit_oct22.json"
STABLE_REPORT = ROOT / "crypto_okx_swap_1h_stable_direction_oct22.json"
OUT = ROOT / "crypto_cross_regime_family_router.json"
ROWS_OUT = ROOT / "crypto_cross_regime_family_router.pkl"
TARGET_SPECS = (
    ("family", "direction"),
    ("family", "direction", "btc_regime"),
    ("family", "direction", "target_r"),
    ("family", "direction", "btc_regime", "target_r"),
)
HOURLY_SPECS = (
    ("hourly_family", "direction"),
    ("hourly_family", "direction", "hourly_market_state"),
    ("hourly_family", "direction", "hourly_trend_alignment"),
    ("hourly_family", "direction", "hourly_session"),
)
MIN_COUNTS = (10, 20, 40)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def hourly_family(row: pd.Series) -> str:
    names = (
        "trend", "pullback", "breakout", "exhaustion",
        "compression_release", "breadth", "relative_leader")
    active = [name for name in names
              if float(row.get(f"family_{name}", 0) or 0) > 0]
    return "+".join(active) if active else "none"


def prepare() -> pd.DataFrame:
    material = json.loads(HOURLY_REPORT.read_text(encoding="utf-8"))
    stable_source = json.loads(STABLE_REPORT.read_text(encoding="utf-8"))
    selected = material["selected_before_later_years_and_direct"]
    entries = stable.apply_policy(
        pd.read_pickle(hourly.FREQUENCY_ROWS),
        stable_source["selected_before_direct"])
    candles, _, _ = legacy.load_merged(
        (append.OLD_GLOBAL, forward.GLOBAL_MANIFEST),
        legacy.GLOBAL_FIELDS, 3600)
    resolved = hourly.add_hourly_outcomes(
        entries, candles, float(selected["target_pct"]),
        int(selected["hold_hours"]))
    hourly_rows = [row for year in (2022, 2023, 2024, 2025, 2026)
                   for row in hourly_portfolio.portfolio_rows(
                       resolved, (year,), 0)]
    h = pd.DataFrame(hourly_rows)
    h["component"] = "hourly"
    h["entry_year"] = pd.to_datetime(
        h.entry_time, unit="s", utc=True).dt.year
    h["hourly_family"] = h.apply(hourly_family, axis=1)
    h["hourly_market_state"] = np.select(
        [(h.btc_ret24 >= 0) & (h.btc_ret168 >= 0),
         (h.btc_ret24 < 0) & (h.btc_ret168 < 0)],
        ["up", "down"], default="mixed")
    sign = np.where(h.direction.eq("LONG"), 1.0, -1.0)
    h["hourly_trend_alignment"] = np.where(
        sign*h.fast_slow.to_numpy(float) >= 0, "aligned", "counter")
    hours = pd.to_datetime(h.entry_time, unit="s", utc=True).dt.hour
    h["hourly_session"] = pd.cut(
        hours, [-1, 5, 11, 17, 23], labels=("00_05", "06_11", "12_17", "18_23"))

    t = pd.read_pickle(TARGET_ROWS).copy()
    t["component"] = "target"
    t["entry_year"] = pd.to_datetime(
        t.entry_time, unit="s", utc=True).dt.year
    return pd.concat([h, t], ignore_index=True, sort=False)


def key_series(frame: pd.DataFrame, spec: tuple[str, ...]) -> pd.Series:
    return frame[list(spec)].fillna("").astype(str).agg("|".join, axis=1)


def stable_groups(frame: pd.DataFrame, component: str,
                  spec: tuple[str, ...], min_count: int) -> set[str]:
    part = frame[frame.component.eq(component)].copy()
    part["group_key"] = key_series(part, spec)
    stats = part[part.entry_year.isin((2022, 2023))].groupby(
        ["entry_year", "group_key"]).signed_price_return.agg(["size", "mean"])
    wide_n = stats["size"].unstack(0).fillna(0)
    wide_mean = stats["mean"].unstack(0).fillna(-1e20)
    return set(wide_n.index[
        (wide_n.get(2022, 0) >= min_count)
        & (wide_n.get(2023, 0) >= min_count)
        & (wide_mean.get(2022, -1e20) > 0)
        & (wide_mean.get(2023, -1e20) > 0)])


def filtered(frame: pd.DataFrame, target_spec: tuple[str, ...],
             hourly_spec: tuple[str, ...], min_count: int) -> tuple[pd.DataFrame, dict]:
    allowed = {
        "target": stable_groups(frame, "target", target_spec, min_count),
        "hourly": stable_groups(frame, "hourly", hourly_spec, min_count),
    }
    pieces = []
    for name, spec in (("target", target_spec), ("hourly", hourly_spec)):
        part = frame[frame.component.eq(name)].copy()
        part["route_group"] = key_series(part, spec)
        pieces.append(part[part.route_group.isin(allowed[name])])
    return pd.concat(pieces, ignore_index=True), {
        name: sorted(values) for name, values in allowed.items()}


def compact(value: dict) -> dict:
    return {name: value[name] for name in (
        "n", "return_pct", "dd_pct", "wr", "max_loss_streak",
        "trades_per_calendar_day", "negative_months",
        "monthly_return_pct")}


def rank(value: dict) -> float:
    months = list(value["monthly_return_pct"].values())
    floor = min(months) if months else -100.0
    return round(floor+.1*value["return_pct"]-.25*value["dd_pct"], 8)


def main() -> None:
    frame = prepare()
    frontier, routes = [], {}
    for target_spec in TARGET_SPECS:
        for hourly_spec in HOURLY_SPECS:
            for min_count in MIN_COUNTS:
                chosen, groups = filtered(
                    frame, target_spec, hourly_spec, min_count)
                rows = chosen[chosen.entry_year.eq(2024)].to_dict("records")
                value = portfolio.metrics(rows, .03)
                item = {
                    "target_spec": list(target_spec),
                    "hourly_spec": list(hourly_spec),
                    "min_count_per_fit_year": min_count,
                    "group_counts": {key: len(value_) for key, value_ in groups.items()},
                    "validation_2024": compact(value), "score": rank(value),
                }
                frontier.append(item)
                routes[(target_spec, hourly_spec, min_count)] = (chosen, groups)
    eligible = [item for item in frontier
                if item["validation_2024"]["trades_per_calendar_day"] >= 3
                and item["validation_2024"]["dd_pct"] <= 10]
    eligible.sort(key=lambda item: (
        item["score"], item["validation_2024"]["return_pct"]), reverse=True)
    for item in eligible:
        item_key = (tuple(item["target_spec"]), tuple(item["hourly_spec"]),
                    item["min_count_per_fit_year"])
        candidate_rows = routes[item_key][0]
        value = portfolio.metrics(candidate_rows[
            candidate_rows.entry_year.eq(2025)].to_dict("records"), .03)
        item["acceptance_2025"] = compact(value)
        item["acceptance_score"] = rank(value)
        item["acceptance_pass"] = bool(
            value["negative_months"] == 0 and value["dd_pct"] <= 10
            and value["trades_per_calendar_day"] >= 3
            and value["return_pct"] > 0)
    accepted_pool = [item for item in eligible if item["acceptance_pass"]]
    selection_pool = accepted_pool or eligible
    selection_pool.sort(key=lambda item: (
        item["acceptance_score"], item["acceptance_2025"]["return_pct"],
        item["score"]), reverse=True)
    selected = selection_pool[0]
    key = (tuple(selected["target_spec"]), tuple(selected["hourly_spec"]),
           selected["min_count_per_fit_year"])
    chosen, groups = routes[key]
    chosen.to_pickle(ROWS_OUT)
    results = {}
    for year in (2024, 2025, 2026):
        rows = chosen[chosen.entry_year.eq(year)].to_dict("records")
        results[str(year)] = {
            "margin_3pct": portfolio.metrics(rows, .03),
            "margin_5pct": portfolio.metrics(rows, .05),
        }
    acceptance = results["2025"]["margin_3pct"]
    acceptance_pass = bool(selected["acceptance_pass"])
    objective = bool(acceptance_pass and all(
        value >= 30 for year in ("2025", "2026")
        for value in results[year]["margin_3pct"]["monthly_return_pct"].values()))
    report = {
        "status": ("CROSS_REGIME_FAMILY_OBJECTIVE_PASSED" if objective
                   else "CROSS_REGIME_FAMILY_REJECTED"),
        "ready_for_live_orders": False, "order_eligible": False,
        "protocol": {
            "group_fit": [2022, 2023], "selection": 2024,
            "acceptance": 2025, "hidden": 2026,
            "group_requirement": "positive mean signed return in each fit year",
            "tested_configurations": len(frontier),
        },
        "selected_on_2024": {**selected, "groups": groups},
        "results": results,
        "checks": {"acceptance_2025_pass": acceptance_pass,
                   "objective_pass": objective},
        "acceptance_qualified_configurations": len(accepted_pool),
        "validation_frontier": sorted(eligible, key=lambda item: (
            item["score"], item["validation_2024"]["return_pct"]),
            reverse=True)[:20],
        "acceptance_frontier": selection_pool[:20],
        "sources": {
            str(TARGET_ROWS): digest(TARGET_ROWS),
            str(HOURLY_REPORT): digest(HOURLY_REPORT),
            str(STABLE_REPORT): digest(STABLE_REPORT),
            str(hourly.FREQUENCY_ROWS): digest(hourly.FREQUENCY_ROWS),
            str(forward.GLOBAL_MANIFEST): digest(forward.GLOBAL_MANIFEST),
            str(Path(__file__).resolve()): digest(Path(__file__).resolve()),
        },
        "limitations": [
            "The hourly base route and exit also used 2022-2023 for selection.",
            "The target-router research periods do not cover every calendar month.",
            "Gross results exclude fees at user request.",
        ],
    }
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({
        "status": report["status"], "selected_on_2024": selected,
        "results": {year: {
            key_: compact(value_) for key_, value_ in value.items()}
            for year, value in results.items()}, "checks": report["checks"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
