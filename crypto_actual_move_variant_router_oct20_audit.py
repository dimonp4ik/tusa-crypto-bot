"""Causal monthly router across three literal-move portfolio variants."""
from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

import crypto_5m_impulse_regime_oct5_audit as impulse_context
import crypto_5m_tree_filter_audit as candle_source
import crypto_actual_move_portfolio_oct6_audit as portfolio
from crypto_runtime_shadow_portfolio_oct3_audit import money, runtime_gate


ROOT = Path("reports/audit_2026_09_08")
OUT = ROOT / "crypto_actual_move_variant_router_oct20.json"
ROWS_OUT = ROOT / "crypto_actual_move_variant_router_oct20.pkl"
WINDOW_DAYS = (30, 60, 90, 180)
MINIMUMS = (20, 50, 100)
MIN_MEANS = (-.00025, 0.0, .00025)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_variants() -> dict[str, list[dict]]:
    candles, _ = candle_source.load()
    sweep = pd.read_pickle(portfolio.SWEEP_SOURCE, compression="gzip")
    impulse = pd.read_pickle(portfolio.IMPULSE_SOURCE, compression="gzip")
    impulse = impulse_context.add_context(impulse, candles)
    broad = impulse[impulse_context.atom_mask(
        impulse, portfolio.IMPULSE_BROAD)].copy()
    stable = impulse[impulse_context.atom_mask(
        impulse, portfolio.IMPULSE_STABLE)].copy()

    def rows(parts: tuple[tuple[str, pd.DataFrame], ...]) -> list[dict]:
        output = []
        for priority, (name, frame) in enumerate(parts):
            output.extend(portfolio.normalize(frame, name, priority))
        return output

    return {
        "sweep_only": rows((("sweep_actual_ladder", sweep),)),
        "broad_first_plus_sweep": rows((
            ("impulse_broad_05", broad), ("sweep_actual_ladder", sweep))),
        "stable_first_plus_sweep": rows((
            ("impulse_stable_05", stable), ("sweep_actual_ladder", sweep))),
    }


def observable_history(variants: dict[str, list[dict]]) -> dict[str, pd.DataFrame]:
    """Hypothetical variant ledgers with only normal runtime rails applied."""
    return {
        name: pd.DataFrame(runtime_gate(rows)).sort_values("exit_time")
        for name, rows in variants.items()
    }


def scores_at(ledgers: dict[str, pd.DataFrame], cutoff: int, *,
              window_days: int, minimum: int,
              minimum_mean: float) -> list[dict]:
    lower = cutoff - window_days * 86400
    output = []
    for name, frame in ledgers.items():
        part = frame[
            frame.exit_time.lt(cutoff) & frame.exit_time.ge(lower)]
        values = part.signed_price_return.to_numpy(float)
        mean = float(values.mean()) if len(values) else -1.0
        if len(values) >= minimum and mean >= minimum_mean:
            output.append({"variant": name, "n": len(values), "mean": mean,
                           "win_rate": float(np.mean(values > 0))})
    return sorted(output, key=lambda item: (
        item["mean"], item["n"], item["variant"]), reverse=True)


def monthly_route(variants: dict[str, list[dict]],
                  ledgers: dict[str, pd.DataFrame], start: str, end: str, *,
                  window_days: int, minimum: int,
                  minimum_mean: float) -> tuple[list[dict], list[dict]]:
    pieces, decisions = [], []
    for month in pd.period_range(start, end, freq="M"):
        first = int(month.start_time.tz_localize("UTC").timestamp())
        last = int((month + 1).start_time.tz_localize("UTC").timestamp())
        ranking = scores_at(
            ledgers, first, window_days=window_days,
            minimum=minimum, minimum_mean=minimum_mean)
        chosen = ranking[0]["variant"] if ranking else "sweep_only"
        decisions.append({"month": str(month), "selected": chosen,
                          "ranking": ranking})
        pieces.extend(row for row in variants[chosen]
                      if first <= int(row["entry_time"]) < last)
    return runtime_gate(pieces), decisions


def evaluate(rows: list[dict], years: tuple[int, ...], margin: float) -> dict:
    wanted = set(years)
    selected = [row for row in rows if int(row["year"]) in wanted]
    value = money(selected, margin)
    months = list(value.get("monthly_return_pct", {}).values())
    value.update({
        "minimum_month_pct": round(min(months), 4) if months else 0.0,
        "median_month_pct": round(float(np.median(months)), 4) if months else 0.0,
    })
    return value


def rank(item: dict) -> tuple:
    y23, y24 = item["selection_2023"], item["selection_2024"]
    combined = item["selection_2023_2024"]
    return (
        min(y23["return_pct"], y24["return_pct"]),
        -max(y23["dd_pct"], y24["dd_pct"]),
        -int(combined["negative_months"]),
        min(y23["minimum_month_pct"], y24["minimum_month_pct"]),
        combined["return_pct"],
    )


def public(item: dict) -> dict:
    return {key: value for key, value in item.items()
            if not key.startswith("_")}


def main() -> None:
    variants = build_variants()
    ledgers = observable_history(variants)
    policies = []
    for window, minimum, minimum_mean in itertools.product(
            WINDOW_DAYS, MINIMUMS, MIN_MEANS):
        rows, decisions = monthly_route(
            variants, ledgers, "2023-01", "2024-12",
            window_days=window, minimum=minimum,
            minimum_mean=minimum_mean)
        policies.append({
            "window_days": window, "minimum_completed": minimum,
            "minimum_mean_price_return_pct": 100 * minimum_mean,
            "selection_2023": evaluate(rows, (2023,), .03),
            "selection_2024": evaluate(rows, (2024,), .03),
            "selection_2023_2024": evaluate(rows, (2023, 2024), .03),
            "_decisions": decisions,
        })
    policies.sort(key=rank, reverse=True)
    selected = policies[0]
    rows, decisions = monthly_route(
        variants, ledgers, "2025-01", "2026-09",
        window_days=int(selected["window_days"]),
        minimum=int(selected["minimum_completed"]),
        minimum_mean=float(selected["minimum_mean_price_return_pct"]) / 100)
    hidden = {
        "margin_3pct": {
            "2025": evaluate(rows, (2025,), .03),
            "2026": evaluate(rows, (2026,), .03),
            "all": evaluate(rows, (2025, 2026), .03),
        },
        "margin_5pct": {
            "2025": evaluate(rows, (2025,), .05),
            "2026": evaluate(rows, (2026,), .05),
            "all": evaluate(rows, (2025, 2026), .05),
        },
    }
    pd.DataFrame(rows).to_pickle(ROWS_OUT)
    three = hidden["margin_3pct"]
    objective = bool(all(
        value["negative_months"] == 0
        and value["minimum_month_pct"] >= 30
        and value["dd_pct"] <= 10
        and value["trades_per_calendar_day"] >= 3
        for value in (three["2025"], three["2026"])))
    report = {
        "status": ("ACTUAL_MOVE_VARIANT_SHADOW_CANDIDATE"
                   if objective else "REJECTED_OBJECTIVE"),
        "ready_for_live_orders": False,
        "protocol": {
            "component_selection": "pre-existing frozen literal-move components",
            "policy_selection": "2023-2024 only",
            "decision": "UTC month start using exits strictly before cutoff",
            "hidden_reveal": "2025-01 through available 2026 archive",
            "variants": list(variants),
            "entry": "next five-minute market open", "orders": "market only",
            "exit": "literal 0.5%-2% ladder or literal 0.5% impulse route",
            "money": "client 3% or 5% current-equity margin; fixed 10x",
        },
        "variant_rows": {name: len(rows_) for name, rows_ in variants.items()},
        "policies_evaluated": len(policies),
        "selected_before_hidden_reveal": public(selected),
        "selection_decisions": selected["_decisions"],
        "policy_frontier": [public(item) for item in policies[:20]],
        "hidden_decisions": decisions,
        "hidden_2025_2026": hidden, "objective_pass": objective,
        "sources": {
            str(portfolio.SWEEP_SOURCE): digest(portfolio.SWEEP_SOURCE),
            str(portfolio.IMPULSE_SOURCE): digest(portfolio.IMPULSE_SOURCE),
            str(portfolio.OUT): digest(portfolio.OUT),
            str(Path(__file__).resolve()): digest(Path(__file__).resolve()),
        },
        "limitations": [
            "The component filters came from earlier finite searches.",
            "Hypothetical variant ledgers assume all alternatives are shadow-recorded.",
            "The 2026 archive ends in September.",
            "Binance spot history still requires direct X-Perp transfer evidence.",
            "Gross returns exclude fees at user request.",
            "No backtest guarantees a future monthly return.",
        ],
    }
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({
        "status": report["status"], "selected": public(selected),
        "hidden": hidden, "objective_pass": objective,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
