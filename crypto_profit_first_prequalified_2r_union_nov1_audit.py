"""Sealed audit of the union of every prequalified 2R supplement rule.

The rules come only from the pre-reveal frontier frozen before 2025-2026.
No rule or threshold is chosen from the late historical years or the current
X-Perp replay.  The audit checks whether their union improves the currently
frozen Profit-First route after normal duplicate and cluster admission.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import pandas as pd

from crypto_cluster_brake_audit import causal_gate
import crypto_high_payoff_complement_forward_audit as high_payoff
import crypto_profit_first_exact_health_router_oct30_audit as health
from crypto_runtime_shadow_portfolio_oct3_audit import money


ROOT = Path("reports/audit_2026_09_08")
FROZEN_FRONTIER = ROOT / "crypto_high_payoff_complement_forward_oct2.json"
HISTORICAL_BASE = ROOT / "_tmp_exact_historical_rows.pkl"
DIRECT_BASE = ROOT / "crypto_profit_first_exact_oct31_1700.pkl"
DIRECT_2R = (
    ROOT / "crypto_xperp_raw_router_audit_2_0_oct31_1700_material_raw_families.csv"
)
OUT = ROOT / "crypto_profit_first_prequalified_2r_union_nov1.json"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compact(value: dict) -> dict:
    return {key: value[key] for key in (
        "n", "wr", "return_pct", "dd_pct", "max_loss_streak",
        "negative_months", "trades_per_calendar_day", "monthly_return_pct",
    )}


def frozen_rules() -> list[dict]:
    report = json.loads(FROZEN_FRONTIER.read_text(encoding="utf-8"))
    return [rule for rule in report["top_before_reveal"]
            if float(rule["target_r"]) == 2.0]


def additions(frame: pd.DataFrame, rules: list[dict], period: str | None = None) -> pd.DataFrame:
    rows: list[dict] = []
    for rule in rules:
        rows.extend(high_payoff.addition(frame, rule, period))
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    result = result.sort_values(["entry_time", "symbol", "row_id"])
    result = result.drop_duplicates(["symbol", "entry_time"], keep="first")
    result["signed_price_return"] = result.net_r * result.risk_pct
    result["module"] = "prequalified_2r_union"
    result["source"] = "prequalified_2r_union"
    result["source_priority"] = 10
    result["module_priority"] = 10
    return result


def admitted(frame: pd.DataFrame) -> pd.DataFrame:
    rows = causal_gate(frame.to_dict("records"), 3, None)
    return pd.DataFrame(rows)


def combine(base: pd.DataFrame, extra: pd.DataFrame) -> pd.DataFrame:
    if extra.empty:
        return admitted(base)
    base = base.copy()
    base["module_priority"] = pd.to_numeric(
        base.get("module_priority", 0), errors="coerce").fillna(0)
    joined = pd.concat([base, extra], ignore_index=True, sort=False)
    joined = joined.sort_values(["entry_time", "module_priority", "symbol"])
    joined = joined.drop_duplicates(["symbol", "entry_time"], keep="first")
    return admitted(joined)


def metric(frame: pd.DataFrame, periods: tuple[str, ...] | None = None,
           margin: float = .03) -> dict:
    if periods is not None:
        frame = frame[frame.period.isin(periods)]
    return money(frame.to_dict("records"), margin)


def main() -> None:
    rules = frozen_rules()
    historical = health.current_outcomes(pd.read_pickle(HISTORICAL_BASE))
    historical = pd.DataFrame(health.accepted(historical, "2r_union_baseline"))
    historical_extra = additions(high_payoff.load_target(2.0), rules)
    historical_combined = combine(historical, historical_extra)

    periods = {
        "development_2022_2023": ("2022h1", "2022h2", "2023"),
        "validation_2024": ("2024",),
        "late_2025_2026": ("2025", "2026"),
    }
    historical_results = {}
    for name, selected_periods in periods.items():
        historical_results[name] = {
            "baseline": compact(metric(historical, selected_periods)),
            "combined": compact(metric(historical_combined, selected_periods)),
        }

    direct = pd.read_pickle(DIRECT_BASE).copy()
    direct_extra = additions(pd.read_csv(DIRECT_2R), rules, "direct")
    direct_combined = combine(direct, direct_extra)
    direct_results = {}
    for margin in (.03, .05):
        direct_results[f"margin_{int(margin * 100)}pct"] = {
            "baseline": compact(metric(direct, margin=margin)),
            "supplement_only": compact(metric(direct_extra, margin=margin)),
            "combined": compact(metric(direct_combined, margin=margin)),
        }

    validation = historical_results["validation_2024"]
    late = historical_results["late_2025_2026"]
    current = direct_results["margin_3pct"]
    direct_months_not_worse = all(
        current["combined"]["monthly_return_pct"].get(month, -100.0) >= value
        for month, value in current["baseline"]["monthly_return_pct"].items()
    )
    robust = bool(
        validation["combined"]["return_pct"] >= validation["baseline"]["return_pct"]
        and validation["combined"]["dd_pct"] <= validation["baseline"]["dd_pct"]
        and late["combined"]["return_pct"] >= late["baseline"]["return_pct"]
        and late["combined"]["dd_pct"] <= late["baseline"]["dd_pct"]
        and current["combined"]["return_pct"] >= current["baseline"]["return_pct"]
        and current["combined"]["dd_pct"] <= current["baseline"]["dd_pct"]
        and current["combined"]["max_loss_streak"]
            <= current["baseline"]["max_loss_streak"]
    )
    report = {
        "status": ("PREQUALIFIED_2R_UNION_SHADOW_CHALLENGER"
                   if robust else "REJECTED_PREQUALIFIED_2R_UNION"),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "ready_for_live_orders": False,
        "runtime_change_allowed": False,
        "robust_improvement": robust,
        "target_30pct_each_month_proven": False,
        "protocol": {
            "rule_source": "all 2R rules in the frontier frozen before 2025-2026",
            "rule_count": len(rules),
            "late_holdouts": ["2024", "2025-2026", "X-Perp through 2026-10-31"],
            "entry": "next closed-bar market open",
            "admission": "existing duplicate removal and causal cluster gate",
            "position_fraction": "client setting; measured at 3% and 5% margin, 10x",
        },
        "rules": [{"route": rule["route"], "conditions": rule["conditions"]}
                  for rule in rules],
        "historical": historical_results,
        "direct_xperp": direct_results,
        "checks": {
            "robust_improvement": robust,
            "late_return_not_worse": (
                late["combined"]["return_pct"] >= late["baseline"]["return_pct"]),
            "direct_drawdown_not_worse": (
                current["combined"]["dd_pct"] <= current["baseline"]["dd_pct"]),
            "direct_loss_streak_not_worse": (
                current["combined"]["max_loss_streak"]
                <= current["baseline"]["max_loss_streak"]),
            "every_direct_month_not_worse": direct_months_not_worse,
            "latest_october_not_worse": (
                current["combined"]["monthly_return_pct"].get("2026-10", -100.0)
                >= current["baseline"]["monthly_return_pct"].get("2026-10", -100.0)),
        },
        "sources": {str(path): digest(path) for path in (
            FROZEN_FRONTIER, HISTORICAL_BASE, DIRECT_BASE, DIRECT_2R,
            Path(__file__).resolve(),
        )},
        "limitations": [
            "The union improves aggregate holdouts but does not improve every direct month.",
            "The supplement remains shadow-only and is not imported by runtime code.",
            "Gross results exclude fees at the user's request.",
        ],
    }
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({
        "status": report["status"],
        "historical": historical_results,
        "direct_xperp": direct_results,
        "checks": report["checks"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
