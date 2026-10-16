"""Profit-first regime exits for frozen causal OKX one-hour entries.

Each pre-entry market family selects one of six fixed payoff profiles on 2022.
The choices are frozen before 2023 validation and the 2024 test.  Win rate is
reported but never used as an acceptance condition.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

import crypto_5m_regime_router_literal_oct6_audit as portfolio
import crypto_okx_swap_1h_conditional_exit_oct7_audit as conditional
import crypto_okx_swap_1h_regime_oct7_audit as base
import crypto_okx_swap_1h_runner_oct7_audit as runner
from crypto_profit_first_runner_oct16_audit import enrich


ROOT = Path("reports/audit_2026_09_08")
SOURCE_REPORT = ROOT / "crypto_okx_swap_1h_regime_oct7.json"
SOURCE_ROWS = ROOT / "crypto_okx_swap_1h_regime_oct7.pkl"
OUT = ROOT / "crypto_profit_first_regime_exit_oct16.json"
ROWS_OUT = ROOT / "crypto_profit_first_regime_exit_oct16.pkl"
GROUPS = conditional.GROUPS
PROFILES = (
    {"name": "fixed_05_h12", "partial": 1.0, "runner": .5, "hold": 12},
    {"name": "p25_r20_h48", "partial": .25, "runner": 2.0, "hold": 48},
    {"name": "p25_r30_h96", "partial": .25, "runner": 3.0, "hold": 96},
    {"name": "p10_r40_h96", "partial": .10, "runner": 4.0, "hold": 96},
    {"name": "p10_r60_h168", "partial": .10, "runner": 6.0, "hold": 168},
    {"name": "p00_r80_h168", "partial": 0.0, "runner": 8.0, "hold": 168},
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def evaluate(frame: pd.DataFrame, years: tuple[int, ...]) -> dict:
    return enrich(portfolio.evaluate(frame, years, 0))


def profile_score(value: dict) -> float:
    return round(
        value["equivalent_monthly_return_pct"]
        - .10 * value["dd_pct"]
        - .20 * value["negative_months"], 8)


def selection_ok(value: dict) -> bool:
    return bool(value["n"] >= 80 and value["return_pct"] > 0
                and value["dd_pct"] <= 15)


def validation_ok(value: dict) -> bool:
    return bool(value["n"] >= 80 and value["return_pct"] > 0
                and value["dd_pct"] <= 15)


def portfolio_ok(value: dict) -> bool:
    return bool(value["n"] >= 500 and value["return_pct"] > 0
                and value["dd_pct"] <= 15
                and value["trades_per_calendar_day"] >= 2)


def objective_ok(value: dict) -> bool:
    return bool(value["months"] > 0 and value["dd_pct"] <= 10
                and value["months_ge_30pct"] == value["months"])


def resolve_profile(entries: pd.DataFrame, candles: dict,
                    profile: dict) -> pd.DataFrame:
    return runner.add_outcomes(
        entries, candles, profile["partial"], profile["runner"],
        profile["hold"])


def main() -> None:
    source = json.loads(SOURCE_REPORT.read_text(encoding="utf-8"))
    if source.get("status") != "RESEARCH_CANDIDATE_AWAITING_DIRECT_FORWARD":
        raise ValueError("source OKX one-hour entry candidate is not frozen")
    candles, sources = base.load()
    entries = pd.read_pickle(SOURCE_ROWS).reset_index(drop=True)
    entries["row_id"] = entries.index.astype(int)
    entries["route_group"] = conditional.assign_group(entries)
    cache = {}
    for profile in PROFILES:
        rows = resolve_profile(entries, candles, profile)
        rows["exit_profile"] = profile["name"]
        cache[profile["name"]] = rows

    frozen = {}
    diagnostics = {}
    for group in GROUPS:
        candidates = []
        diagnostics[group] = []
        for profile in PROFILES:
            rows = cache[profile["name"]]
            rows = rows[rows.route_group.eq(group)]
            selection = evaluate(rows, (2022,))
            item = {"profile": profile, "selection_2022": selection,
                    "selection_pass": selection_ok(selection),
                    "score": profile_score(selection)}
            diagnostics[group].append(item)
            if item["selection_pass"]:
                candidates.append(item)
        candidates.sort(key=lambda item: item["score"], reverse=True)
        if not candidates:
            frozen = {}
            break
        chosen = candidates[0]
        rows = cache[chosen["profile"]["name"]]
        rows = rows[rows.route_group.eq(group)]
        chosen["validation_2023"] = evaluate(rows, (2023,))
        chosen["validation_pass"] = validation_ok(chosen["validation_2023"])
        frozen[group] = chosen

    combined = None
    if frozen:
        parts = []
        for group, item in frozen.items():
            rows = cache[item["profile"]["name"]]
            parts.append(rows[rows.route_group.eq(group)].copy())
        combined = pd.concat(parts, ignore_index=True).sort_values("row_id")

    development = evaluate(combined, (2022,)) if combined is not None else None
    validation = evaluate(combined, (2023,)) if combined is not None else None
    pass_before_test = bool(
        combined is not None
        and all(item["validation_pass"] for item in frozen.values())
        and portfolio_ok(development) and portfolio_ok(validation))
    test = confirmation = objective_checks = None
    status = "REJECTED_BEFORE_TEST"
    if pass_before_test:
        test = evaluate(combined, (2024,))
        status = "REJECTED_TEST"
        if portfolio_ok(test):
            confirmation = {
                str(year): evaluate(combined, (year,))
                for year in (2025, 2026)}
            objective_checks = {
                "2022": objective_ok(development),
                "2023": objective_ok(validation),
                "2024": objective_ok(test),
                **{str(year): objective_ok(confirmation[str(year)])
                   for year in (2025, 2026)},
            }
            status = ("PROFIT_FIRST_REGIME_RESEARCH_CANDIDATE"
                      if all(objective_checks.values()) else "REJECTED_OBJECTIVE")
        combined.to_pickle(ROWS_OUT)

    report = {
        "status": status,
        "ready_for_live_orders": False,
        "protocol": {
            "priority": "account return first; win rate diagnostic only",
            "parent_entry_model_fit": [2020, 2021],
            "profile_selection": 2022,
            "profile_validation": 2023,
            "untouched_test": 2024,
            "confirmation_only": [2025, 2026],
            "groups": list(GROUPS),
            "profiles": list(PROFILES),
            "entry": "frozen causal signal; next 1h market open",
            "orders": "market only",
            "money": "3% current-equity margin, fixed 10x gross",
            "objective": ">=30% account return in every month; DD <=10%",
        },
        "group_counts": entries.route_group.value_counts().to_dict(),
        "group_diagnostics": diagnostics,
        "frozen_before_validation": frozen,
        "development_2022": development,
        "validation_2023": validation,
        "pass_before_test": pass_before_test,
        "test_2024": test,
        "confirmation_2025_2026": confirmation,
        "objective_checks": objective_checks,
        "sources": {
            **sources,
            str(SOURCE_REPORT): digest(SOURCE_REPORT),
            str(SOURCE_ROWS): digest(SOURCE_ROWS),
            str(Path(runner.__file__)): digest(Path(runner.__file__)),
            str(Path(conditional.__file__)): digest(Path(conditional.__file__)),
            str(Path(__file__)): digest(Path(__file__)),
        },
        "limitations": [
            "Six fixed profiles across three semantic groups still create selection risk.",
            "Historical OKX global results require EU X-Perp forward transfer.",
        ],
    }
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({key: report[key] for key in (
        "status", "frozen_before_validation", "development_2022",
        "validation_2023", "pass_before_test", "test_2024",
        "confirmation_2025_2026", "objective_checks")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
