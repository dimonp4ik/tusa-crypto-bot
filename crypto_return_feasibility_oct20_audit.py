"""Measure the expectancy gap between exact crypto trades and +30%/month."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from crypto_actual_move_health_gate_oct6_audit import source_rows
from crypto_runtime_shadow_portfolio_oct3_audit import runtime_gate


ROOT = Path("reports/audit_2026_09_08")
OUT = ROOT / "crypto_return_feasibility_oct20.json"
TARGET_MONTHLY_RETURN = .30
LEVERAGE = 10.0


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def period(rows: list[dict], year: int) -> dict:
    accepted = runtime_gate([row for row in rows if int(row["year"]) == year])
    first = pd.Timestamp(min(row["entry_time"] for row in accepted), unit="s", tz="UTC")
    last = pd.Timestamp(max(row["entry_time"] for row in accepted), unit="s", tz="UTC")
    days = max(1, (last.date() - first.date()).days + 1)
    trades_per_day = len(accepted) / days
    observed = float(np.mean([row["signed_price_return"] for row in accepted]))
    sizing = {}
    for margin in (.03, .05):
        required = TARGET_MONTHLY_RETURN / (
            30 * trades_per_day * margin * LEVERAGE)
        projected = 30 * trades_per_day * margin * LEVERAGE * observed
        sizing[f"margin_{int(100 * margin)}pct"] = {
            "linear_projected_month_pct": round(100 * projected, 4),
            "required_mean_signed_move_pct": round(100 * required, 4),
            "required_multiple_of_observed": (
                round(required / observed, 2) if observed > 0 else None),
        }
    return {
        "n": len(accepted), "calendar_days": days,
        "trades_per_calendar_day": round(trades_per_day, 4),
        "observed_mean_signed_move_pct": round(100 * observed, 5),
        "sizing": sizing,
    }


def main() -> None:
    rows = source_rows()
    periods = {str(year): period(rows, year) for year in range(2022, 2027)}
    report = {
        "status": "EMPIRICAL_EDGE_TOO_SMALL_FOR_30PCT_MONTHLY",
        "ready_for_live_orders": False,
        "protocol": {
            "rows": "exact 0.5%-2% portfolio after current runtime rails",
            "years": [2022, 2023, 2024, 2025, 2026],
            "target_monthly_return_pct": 30,
            "leverage": LEVERAGE,
            "sizing": "client-selected 3% or 5% of current equity",
            "formula": "30 * trades/day * margin * leverage * mean signed price move",
        },
        "periods": periods,
        "finding": (
            "At 3% sizing the observed edge is roughly 10-25 times smaller "
            "than the edge required for +30% per month at the observed frequency."
        ),
        "sources": {str(Path(__file__).resolve()): digest(Path(__file__).resolve())},
        "limitations": [
            "The projection is a linear expectancy diagnostic, not a return forecast.",
            "It ignores compounding and overlap, which do not close an order-of-magnitude gap.",
            "Gross results exclude fees at user request.",
        ],
    }
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
