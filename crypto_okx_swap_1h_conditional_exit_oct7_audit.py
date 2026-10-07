"""Causal family-conditional exits for frozen OKX one-hour entries."""
from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path

import pandas as pd

import crypto_5m_regime_router_literal_oct6_audit as portfolio
import crypto_okx_swap_1h_regime_oct7_audit as base
import crypto_okx_swap_1h_runner_oct7_audit as runner


ROOT = Path("reports/audit_2026_09_08")
SOURCE_REPORT = ROOT / "crypto_okx_swap_1h_regime_oct7.json"
SOURCE_ROWS = ROOT / "crypto_okx_swap_1h_regime_oct7.pkl"
OUT = ROOT / "crypto_okx_swap_1h_conditional_exit_oct7.json"
ROWS_OUT = ROOT / "crypto_okx_swap_1h_conditional_exit_oct7.pkl"
PROFILES = (
    {"name": "fixed_05", "partial": 1.0, "runner": .5, "hold": 12},
    *tuple({"name": f"p{int(100*p)}_r{int(10*r):02d}_h{h}",
            "partial": p, "runner": r, "hold": h}
           for p, r, h in itertools.product(
               runner.PARTIALS, runner.RUNNER_TARGETS, runner.HOLDS)),
)
GROUPS = ("continuation", "pullback", "transition")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def assign_group(frame: pd.DataFrame) -> pd.Series:
    output = pd.Series("continuation", index=frame.index, dtype="object")
    output.loc[frame.family_compression_release.eq(1)] = "transition"
    output.loc[frame.family_pullback.eq(1)] = "pullback"
    output.loc[frame.family_exhaustion.eq(1)] = "transition"
    return output


def resolve_profile(entries: pd.DataFrame, candles: dict,
                    profile: dict) -> pd.DataFrame:
    if profile["partial"] == 1.0:
        # A 100% first partial is exactly the fixed 0.5% exit.  Reuse the
        # runner resolver so collision and gap handling stay identical.
        return runner.add_outcomes(entries, candles, 1.0, .5, profile["hold"])
    return runner.add_outcomes(entries, candles, profile["partial"],
                               profile["runner"], profile["hold"])


def group_ok(value22: dict, value23: dict) -> bool:
    return bool(value22["n"] >= 80 and value23["n"] >= 80
                and value22["return_pct"] > 0 and value23["return_pct"] > 0
                and value22["dd_pct"] <= 8 and value23["dd_pct"] <= 8
                and value22["literal_target_hit_rate"] >= 60
                and value23["literal_target_hit_rate"] >= 60)


def portfolio_ok(value: dict) -> bool:
    return bool(value["n"] >= 500 and value["return_pct"] > 0
                and value["literal_target_hit_rate"] >= 65
                and value["dd_pct"] <= 7
                and value["trades_per_calendar_day"] >= 2.0)


def main() -> None:
    source = json.loads(SOURCE_REPORT.read_text(encoding="utf-8"))
    if source.get("status") != "RESEARCH_CANDIDATE_AWAITING_DIRECT_FORWARD":
        raise ValueError("source OKX one-hour entry candidate is not frozen")
    candles, sources = base.load()
    entries = pd.read_pickle(SOURCE_ROWS).reset_index(drop=True)
    entries["row_id"] = entries.index.astype(int)
    entries["route_group"] = assign_group(entries)
    cache = {}
    for profile in PROFILES:
        resolved = resolve_profile(entries, candles, profile)
        resolved["exit_profile"] = profile["name"]
        cache[profile["name"]] = resolved
    frozen = {}
    diagnostics = {}
    for group in GROUPS:
        candidates = []
        diagnostics[group] = []
        for profile in PROFILES:
            rows = cache[profile["name"]]
            rows = rows[rows.route_group.eq(group)]
            value22 = portfolio.evaluate(rows, (2022,), 0)
            value23 = portfolio.evaluate(rows, (2023,), 0)
            item = {"profile": profile, "selection_2022": value22,
                    "validation_2023": value23}
            diagnostics[group].append(item)
            if group_ok(value22, value23):
                item["score"] = round(
                    min(value22["return_pct"], value23["return_pct"])
                    - 2 * max(value22["dd_pct"], value23["dd_pct"]), 8)
                candidates.append(item)
        candidates.sort(key=lambda item: item["score"], reverse=True)
        if not candidates:
            frozen = {}
            break
        frozen[group] = candidates[0]
    combined = None
    if frozen:
        parts = []
        for group, item in frozen.items():
            rows = cache[item["profile"]["name"]]
            parts.append(rows[rows.route_group.eq(group)].copy())
        combined = pd.concat(parts, ignore_index=True).sort_values("row_id")
    development = ({str(year): portfolio.evaluate(combined, (year,), 0)
                    for year in (2022, 2023)} if combined is not None else {})
    pass_development = bool(combined is not None and all(
        portfolio_ok(value) for value in development.values()))
    test_2024 = later = checks = None
    if pass_development:
        test_2024 = portfolio.evaluate(combined, (2024,), 0)
        if portfolio_ok(test_2024):
            later = {str(year): portfolio.evaluate(combined, (year,), 0)
                     for year in (2025, 2026)}
            checks = {year: portfolio_ok(value) for year, value in later.items()}
        combined.to_pickle(ROWS_OUT)
    report = {
        "status": ("RESEARCH_CANDIDATE_AWAITING_DIRECT_FORWARD"
                   if checks and all(checks.values()) else "REJECTED"),
        "protocol": {
            "entry_filter": "frozen direct OKX 1h regime candidate",
            "exit_groups": list(GROUPS), "exit_profiles": list(PROFILES),
            "profile_selection": 2022, "profile_validation_and_ranking": 2023,
            "untouched_test": 2024, "hidden_if_test_passes": [2025, 2026],
            "entry": "next 1h market open",
            "money": "3% equity margin, fixed 10x gross",
        },
        "group_counts": entries.route_group.value_counts().to_dict(),
        "group_diagnostics": diagnostics,
        "frozen_before_2024": frozen,
        "development_portfolio": development,
        "pass_before_2024": pass_development,
        "test_2024": test_2024, "later": later, "later_checks": checks,
        "sources": {**sources, str(SOURCE_REPORT): digest(SOURCE_REPORT),
                    str(SOURCE_ROWS): digest(SOURCE_ROWS),
                    str(Path(__file__)): digest(Path(__file__))},
        "limitations": [
            "Four economic groups and 28 finite exit profiles create bounded selection risk.",
            "Any historical pass still requires direct EU X-Perp transfer and new forward shadow outcomes.",
        ],
    }
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({key: report[key] for key in (
        "status", "group_counts", "frozen_before_2024",
        "development_portfolio", "test_2024", "later", "later_checks")},
        indent=2), flush=True)


if __name__ == "__main__":
    main()
