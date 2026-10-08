"""Frozen fast-volatility regime experts for the expanded 2% hourly route.

Two semantic modules were chosen using only 2022-2023 diagnostics: trend and
relative-leader proposals while BTC volatility is accelerating.  They are
evaluated as a supplement, not claimed to meet the complete account objective.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

import crypto_5m_regime_router_literal_oct6_audit as portfolio
import crypto_okx_swap_1h_expanded_health_gate_oct8_audit as source
from src.research_profit_gate import economic_gate


ROOT = Path("reports/audit_2026_09_08")
OUT = ROOT / "crypto_okx_swap_1h_regime_experts_oct9.json"
EXPERTS = {
    "fast_vol_trend": "family_trend",
    "fast_vol_relative_leader": "family_relative_leader",
}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def subset(frame: pd.DataFrame, family_column: str) -> pd.DataFrame:
    found = frame[
        frame.quality.ge(source.BASE_THRESHOLD)
        & frame.fast_slow.gt(1.2)
        & frame[family_column].eq(1)
    ].copy()
    found["quality"] = 1.0
    return found


def evaluate(frame: pd.DataFrame, years: tuple[int, ...]) -> dict:
    metrics = portfolio.evaluate(frame, years, .5)
    rows = portfolio.portfolio_rows(frame, years, .5)
    gate = economic_gate(
        [float(row["signed_price_return"]) for row in rows],
        metrics["trades_per_calendar_day"], market_days=30)
    return {"portfolio": metrics, "economic_gate": gate}


def supplement_ok(value: dict, minimum: int) -> bool:
    metrics = value["portfolio"]
    return bool(metrics["n"] >= minimum
                and metrics["return_pct"] > 0
                and metrics["literal_target_hit_rate"] >= 35
                and metrics["dd_pct"] <= 8)


def main() -> None:
    frame, sources = source.build_scored()
    expert_frames = {name: subset(frame, column)
                     for name, column in EXPERTS.items()}
    union = pd.concat(expert_frames.values(), ignore_index=False).sort_values(
        ["entry_time", "symbol", "direction"]).drop_duplicates(
            ["entry_time", "symbol", "direction"]).copy()
    development = {
        name: {"2022": evaluate(found, (2022,)),
               "2023": evaluate(found, (2023,))}
        for name, found in expert_frames.items()
    }
    development["union"] = {
        "2022": evaluate(union, (2022,)),
        "2023": evaluate(union, (2023,)),
    }
    development_pass = all(
        supplement_ok(value, 50 if name != "union" else 100)
        for name, periods in development.items()
        for value in periods.values())
    test_2024 = evaluate(union, (2024,)) if development_pass else None
    test_pass = bool(test_2024 and supplement_ok(test_2024, 100))
    later = ({str(year): evaluate(union, (year,)) for year in (2025, 2026)}
             if test_pass else None)
    later_pass = bool(later and all(
        supplement_ok(value, 80) for value in later.values()))
    objective_pass = bool(later_pass and all(
        value["economic_gate"]["passes"]
        for periods in (*development.values(),)
        for value in periods.values())
        and test_2024["economic_gate"]["passes"]
        and all(value["economic_gate"]["passes"] for value in later.values()))
    status = ("OBJECTIVE_PASS" if objective_pass
              else "STABLE_SUPPLEMENT_ONLY" if later_pass
              else "REJECTED_LATER" if later is not None
              else "REJECTED_2024" if test_2024 is not None
              else "REJECTED_DEVELOPMENT")
    report = {
        "status": status, "ready_for_live_orders": False,
        "protocol": {
            "source": "expanded cross-symbol TP2%/SL1% HGB",
            "source_threshold": source.BASE_THRESHOLD,
            "experts": {
                "fast_vol_trend": "fast_slow > 1.2 and family_trend",
                "fast_vol_relative_leader": (
                    "fast_slow > 1.2 and family_relative_leader"),
            },
            "selection_and_validation": [2022, 2023],
            "test_revealed_after_freeze": 2024,
            "later_revealed_only_if_test_passes": [2025, 2026],
            "target_pct": 2.0, "stop_pct": 1.0, "hold_hours": 48,
            "money": "3% current equity margin, fixed 10x gross",
            "orders": "next-hour market proxy; no live orders",
        },
        "rows": {name: len(found) for name, found in expert_frames.items()}
        | {"union": len(union)},
        "development": development,
        "development_pass": development_pass,
        "test_2024": test_2024, "test_pass": test_pass,
        "later": later, "later_pass": later_pass,
        "objective_pass": objective_pass,
        "decision": (
            "A stable supplement may be retained for portfolio research, but "
            "runtime stays unchanged until the complete objective passes."),
        "sources": {**sources, str(Path(__file__)): digest(Path(__file__))},
    }
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({key: report[key] for key in (
        "status", "rows", "development", "test_2024", "test_pass",
        "later", "later_pass", "objective_pass")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
