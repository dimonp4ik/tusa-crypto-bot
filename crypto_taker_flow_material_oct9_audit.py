"""Causal multi-session crypto moves from Binance Futures taker flow.

The older flow experiment used a 0.5% target and a two-hour horizon.  This
audit keeps the source and semantic mechanisms, but predeclares materially
larger 1%-2% moves and 12-48 hour holding windows.  Later calendar periods are
never evaluated unless every earlier objective gate passes.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

import crypto_5m_regime_router_literal_oct6_audit as price
import crypto_deribit_dvol_regime_oct9_audit as outcome_source
import crypto_futures_trade_flow_regime_oct7_audit as flow_source
from crypto_runtime_shadow_portfolio_oct3_audit import money, runtime_gate
from src.research_profit_gate import economic_gate, minimum_binary_hit_rate_pct


ROOT = Path("reports/audit_2026_09_08")
OUT = ROOT / "crypto_taker_flow_material_oct9.json"
FAMILIES = flow_source.FAMILIES
PROFILES = ("broad", "strict")
CONTRACTS = (
    {"target_pct": 1.0, "stop_pct": .75, "hold_bars": 144},
    {"target_pct": 1.5, "stop_pct": .75, "hold_bars": 288},
    {"target_pct": 2.0, "stop_pct": 1.0, "hold_bars": 576},
)
PERIODS = {
    "development_2020_2021": (pd.Timestamp("2020-04-01", tz="UTC"),
                              pd.Timestamp("2022-01-01", tz="UTC")),
    "validation_2022": (pd.Timestamp("2022-01-01", tz="UTC"),
                        pd.Timestamp("2023-01-01", tz="UTC")),
    "confirmation_2023": (pd.Timestamp("2023-01-01", tz="UTC"),
                          pd.Timestamp("2024-01-01", tz="UTC")),
    "test_2024": (pd.Timestamp("2024-01-01", tz="UTC"),
                  pd.Timestamp("2025-01-01", tz="UTC")),
    "test_2025": (pd.Timestamp("2025-01-01", tz="UTC"),
                  pd.Timestamp("2026-01-01", tz="UTC")),
    "untouched_2026": (pd.Timestamp("2026-01-01", tz="UTC"),
                       pd.Timestamp("2027-01-01", tz="UTC")),
}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def strict_mask(frame: pd.DataFrame, family: str) -> pd.Series:
    direction = frame.direction.astype(float)
    if family == "flow_continuation":
        return (frame.imbalance_mean3.abs().ge(.12)
                & (frame.cvd12 * direction).ge(.04)
                & frame.quote_volume_ratio.ge(1.20))
    if family == "flow_absorption":
        return (frame.imbalance.abs().ge(.30)
                & frame.quote_volume_ratio.ge(1.50)
                & frame.range_ratio.ge(1.20))
    if family == "large_trade_breakout":
        return (frame.avg_trade_ratio.ge(1.50)
                & frame.quote_volume_ratio.ge(1.50)
                & (frame.imbalance_mean3 * direction).ge(.07))
    if family == "cvd_divergence":
        return (frame.cvd12.abs().ge(.08)
                & (frame.ret12 * direction).le(-.002)
                & frame.quote_volume_ratio.ge(1.0))
    if family == "flow_exhaustion":
        move = -direction
        return (frame.ret12.abs().ge(.007)
                & (frame.imbalance_z288 * move).ge(1.5)
                & frame.range_ratio.ge(1.10))
    if family == "cross_flow_leader":
        return (frame.flow_cross_z.abs().ge(1.50)
                & (frame.cvd12 * direction).gt(0)
                & frame.quote_volume_ratio.ge(1.10))
    raise ValueError(f"unknown flow family: {family}")


def semantic_events() -> tuple[pd.DataFrame, dict[str, str], int, int]:
    candles, sources = flow_source.load()
    raw = price.build_features(candles)
    enriched = flow_source.attach_flow(raw, candles)
    events = flow_source.proposals(enriched)
    events["semantic_score"] = (
        events.imbalance_mean3.abs()
        + events.cvd12.abs()
        + .10 * events.imbalance_z288.abs().clip(upper=10)
        + .05 * events.flow_cross_z.abs().clip(upper=10)
        + .05 * np.log1p(events.quote_volume_ratio.clip(lower=0))
        + .05 * np.log1p(events.avg_trade_ratio.clip(lower=0))
    )
    return events, sources, len(raw), len(enriched)


def route(frame: pd.DataFrame, family: str, profile: str) -> pd.DataFrame:
    if family == "all_regimes":
        chosen = frame.copy()
        if profile == "strict":
            masks = [
                frame[f"flow_family_{name}"].eq(1) & strict_mask(frame, name)
                for name in FAMILIES
            ]
            chosen = frame[np.logical_or.reduce(masks)].copy()
    else:
        chosen = frame[frame[f"flow_family_{family}"].eq(1)].copy()
        if profile == "strict":
            chosen = chosen[strict_mask(chosen, family)].copy()
    chosen["family"] = family
    chosen["profile"] = profile
    return chosen.sort_values("semantic_score", ascending=False).drop_duplicates(
        ["symbol", "entry_time", "direction"])


def period_rows(frame: pd.DataFrame, period: str) -> pd.DataFrame:
    start, end = PERIODS[period]
    stamp = pd.to_datetime(frame.entry_time, unit="s", utc=True)
    return frame[(stamp >= start) & (stamp < end)]


def evaluate(frame: pd.DataFrame, period: str) -> dict:
    selected = period_rows(frame, period).sort_values(
        ["entry_time", "semantic_score", "symbol"],
        ascending=[True, False, True]).copy()
    selected["source_priority"] = selected.groupby("entry_time").cumcount()
    rows = []
    for row in selected.to_dict("records"):
        rows.append({
            **row,
            "direction": "LONG" if row["direction"] > 0 else "SHORT",
            "source": f"taker_flow_{row['family']}",
            "module": "taker_flow_material_router", "module_priority": 0,
            "outcome": "TP" if row["signed_return"] > 0 else "SL",
            "signed_price_return": row["signed_return"],
        })
    accepted = runtime_gate(rows)
    result = money(accepted, .03)
    n = len(accepted)
    months = list(result.get("monthly_return_pct", {}).values())
    result.update({
        "literal_target_hits": sum(bool(row["target_hit"]) for row in accepted),
        "literal_target_hit_rate": round(
            100 * sum(bool(row["target_hit"]) for row in accepted) / n, 4)
            if n else 0.0,
        "stops": sum(int(row["exit_reason"]) == 0 for row in accepted),
        "timeouts": sum(int(row["exit_reason"]) == 2 for row in accepted),
        "months": len(months),
        "months_ge_30pct": sum(value >= 30 for value in months),
        "minimum_month_pct": round(min(months), 4) if months else 0.0,
        "median_month_pct": round(float(np.median(months)), 4) if months else 0.0,
    })
    observed = float(result.get("trades_per_calendar_day", 0.0) or 0.0)
    result["economic_gate"] = (
        economic_gate(
            [float(row["signed_price_return"]) for row in accepted], observed)
        if accepted else {
            "n": 0, "observed_trades_per_day": 0.0,
            "projected_monthly_return_pct": 0.0,
            "frequency_pass": False, "return_pass": False, "passes": False,
        }
    )
    return result


def gate(value: dict, minimum_n: int) -> bool:
    return bool(value["n"] >= minimum_n and value["return_pct"] > 0
                and value["dd_pct"] <= 8 and value["negative_months"] == 0
                and value["economic_gate"]["passes"])


def main() -> None:
    events, sources, raw_count, enriched_count = semantic_events()
    candles, _ = flow_source.load()
    frontier, survivors, exclusions = [], [], []
    for contract in CONTRACTS:
        outcomes, excluded = outcome_source.add_outcomes(events, candles, contract)
        exclusions.append({**contract, "excluded_crossing_gap_or_end": excluded,
                           "outcome_rows": len(outcomes)})
        for family in (*FAMILIES, "all_regimes"):
            for profile in PROFILES:
                selected = route(outcomes, family, profile)
                development = evaluate(selected, "development_2020_2021")
                item = {**contract, "family": family, "profile": profile,
                        "development_2020_2021": development}
                frontier.append(item)
                if not gate(development, 800):
                    continue
                validation = evaluate(selected, "validation_2022")
                item["validation_2022"] = validation
                if not gate(validation, 350):
                    continue
                item["score"] = round(
                    development["return_pct"] + 2 * validation["return_pct"]
                    - development["dd_pct"] - 2 * validation["dd_pct"], 8)
                item["_frame"] = selected
                survivors.append(item)
    survivors.sort(key=lambda item: item["score"], reverse=True)
    frozen = survivors[0] if survivors else None
    confirmation = test_2024 = test_2025 = reveal = None
    if frozen:
        confirmation = evaluate(frozen["_frame"], "confirmation_2023")
        if gate(confirmation, 300):
            test_2024 = evaluate(frozen["_frame"], "test_2024")
        if test_2024 and gate(test_2024, 300):
            test_2025 = evaluate(frozen["_frame"], "test_2025")
        if test_2025 and gate(test_2025, 300):
            reveal = evaluate(frozen["_frame"], "untouched_2026")
    frontier.sort(key=lambda item: (
        item["development_2020_2021"]["economic_gate"][
            "projected_monthly_return_pct"],
        item["development_2020_2021"]["literal_target_hit_rate"],
        item["development_2020_2021"]["return_pct"]), reverse=True)
    public_frozen = ({key: value for key, value in frozen.items()
                      if key != "_frame"} if frozen else None)
    status = ("RESEARCH_CANDIDATE_AWAITING_DIRECT_FORWARD"
              if reveal and gate(reveal, 200) else "REJECTED")
    report = {
        "status": status,
        "ready_for_live_orders": False,
        "protocol": {
            "source": "Binance Futures 5m klines with taker buy volume",
            "entry": "next 5m market open after closed causal bars",
            "families": list(FAMILIES), "profiles": list(PROFILES),
            "contracts": list(CONTRACTS),
            "development": "2020-04 through 2021",
            "validation": "2022 only after development gate",
            "confirmation": "2023 only after validation gate",
            "later": "2024, 2025, 2026 revealed sequentially",
            "money": "3% current-equity margin, fixed 10x leverage",
            "fees": "excluded by user instruction",
        },
        "rows": {"raw": raw_count, "flow_enriched": enriched_count,
                 "semantic_events": len(events)},
        "minimum_ideal_hit_rate_for_30pct_month_at_3_trades_day": {
            f"tp{item['target_pct']}_sl{item['stop_pct']}": round(
                minimum_binary_hit_rate_pct(
                    item["target_pct"], item["stop_pct"]), 6)
            for item in CONTRACTS
        },
        "exclusions": exclusions,
        "qualified_before_2023": len(survivors),
        "frozen_before_2023": public_frozen,
        "confirmation_2023": confirmation,
        "test_2024": test_2024, "test_2025": test_2025,
        "untouched_2026": reveal,
        "development_frontier": frontier[:24],
        "sources": {**sources, str(Path(__file__)): digest(Path(__file__))},
        "limitations": [
            "Kline taker flow is aggregated and cannot reconstruct order-book depth.",
            "The source is Binance USDT perpetuals while the runtime venue is OKX EU.",
            "Any survivor still requires untouched direct-venue forward evidence.",
        ],
    }
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({
        "status": status, "rows": report["rows"],
        "qualified_before_2023": len(survivors),
        "frozen_before_2023": public_frozen,
        "confirmation_2023": confirmation, "test_2024": test_2024,
        "test_2025": test_2025, "untouched_2026": reveal,
        "best_development": frontier[:6],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
