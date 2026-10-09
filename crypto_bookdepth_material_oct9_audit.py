"""Causal material crypto moves from historical Binance Futures book depth."""
from __future__ import annotations

import hashlib
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

import crypto_deribit_dvol_regime_oct9_audit as outcome_source
import crypto_futures_trade_flow_regime_oct7_audit as price_source
from crypto_runtime_shadow_portfolio_oct3_audit import money, runtime_gate
from src.research_profit_gate import economic_gate, minimum_binary_hit_rate_pct


ROOT = Path("reports/audit_2026_09_08")
DEPTH_ROOT = ROOT / "crypto_binance_bookdepth_hourly_oct9"
OUT = ROOT / "crypto_bookdepth_material_oct9.json"
FAMILIES = ("depth_continuation", "depth_recovery", "wall_shift",
            "inner_outer_divergence", "liquidity_vacuum")
PROFILES = ("broad", "strict")
CONTRACTS = (
    {"target_pct": 1.0, "stop_pct": .75, "hold_bars": 144},
    {"target_pct": 1.5, "stop_pct": .75, "hold_bars": 288},
    {"target_pct": 2.0, "stop_pct": 1.0, "hold_bars": 576},
)
PERIODS = {
    "development_2023": (pd.Timestamp("2023-01-01", tz="UTC"),
                         pd.Timestamp("2024-01-01", tz="UTC")),
    "validation_2024": (pd.Timestamp("2024-01-01", tz="UTC"),
                        pd.Timestamp("2025-01-01", tz="UTC")),
    "test_2025": (pd.Timestamp("2025-01-01", tz="UTC"),
                  pd.Timestamp("2026-01-01", tz="UTC")),
}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_depth() -> tuple[pd.DataFrame, dict, dict[str, str]]:
    manifest_path = DEPTH_ROOT / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    data_path = Path(manifest["data_path"])
    raw = data_path.read_bytes()
    found = hashlib.sha256(raw).hexdigest()
    if found != manifest["data_sha256"]:
        raise ValueError("bookDepth aggregate hash mismatch")
    values = pickle.loads(raw)
    frame = pd.DataFrame(values).sort_values("available_time").drop_duplicates(
        "available_time")
    times = frame.available_time.to_numpy(np.int64)
    if np.any(np.diff(times) <= 0) or np.any(times % 3600 != 0):
        raise ValueError("bookDepth availability times are not ordered UTC hours")
    return frame, manifest, {
        str(manifest_path): digest(manifest_path), str(data_path): found}


def depth_features(frame: pd.DataFrame) -> pd.DataFrame:
    value = frame.copy()
    times = value.available_time.to_numpy(np.int64)
    continuous = np.ones(len(value), dtype=bool)
    for window in (6, 24, 168):
        valid = np.zeros(len(value), dtype=bool)
        valid[window:] = times[window:] - times[:-window] == window*3600
        continuous &= valid
    value["imb1"] = value.imbalance_1_0
    value["imb2"] = value.imbalance_2_0
    value["imb5"] = value.imbalance_5_0
    for window in (6, 24):
        value[f"imb1_mean{window}"] = value.imb1.rolling(window).mean()
    value["imb1_delta6"] = value.imb1 - value.imb1.shift(6)
    value["inner_outer"] = value.imb1-value.imb5
    value["notional1_log"] = np.log1p(value.notional_1_0.clip(lower=0))
    mean = value.notional1_log.rolling(168).mean()
    std = value.notional1_log.rolling(168).std().replace(0, np.nan)
    value["notional1_z168"] = (value.notional1_log-mean)/std
    value["notional1_delta6"] = (
        value.notional1_log-value.notional1_log.shift(6))
    value["imbalance_volatility"] = value.imbalance_1_0_std
    needed = ["imb1", "imb2", "imb5", "imb1_mean6", "imb1_mean24",
              "imb1_delta6", "inner_outer", "notional1_z168",
              "notional1_delta6", "imbalance_volatility"]
    return value.loc[continuous].replace([np.inf, -np.inf], np.nan).dropna(
        subset=needed)


def attach_depth(hourly: pd.DataFrame, depth: pd.DataFrame) -> pd.DataFrame:
    return hourly.merge(depth, left_on="entry_time", right_on="available_time",
                        how="inner", validate="many_to_one")


def build_events(frame: pd.DataFrame) -> pd.DataFrame:
    pieces = []

    def add(family: str, profile: str, direction: pd.Series,
            condition: pd.Series, strength: pd.Series) -> None:
        selected = condition & direction.ne(0)
        part = frame.loc[selected].copy()
        part["direction"] = direction.loc[selected].astype(np.int8)
        part["family"], part["profile"] = family, profile
        part["semantic_score"] = strength.loc[selected].astype(float)
        pieces.append(part)

    for profile in PROFILES:
        strict = profile == "strict"

        direction = np.sign(frame.imb1_mean6)
        breadth = frame.breadth72.where(direction > 0, 1-frame.breadth72)
        condition = (
            frame.imb1_mean6.abs().ge(.16 if strict else .09)
            & (frame.ret12*direction).ge(.002 if strict else 0)
            & (frame.body1*direction).gt(0)
            & breadth.ge(.625 if strict else .50)
        )
        add("depth_continuation", profile, direction, condition,
            frame.imb1_mean6.abs()+50*(frame.ret12*direction)+breadth)

        direction = np.sign(frame.imb1_mean24)
        rejection = frame.lower_wick.where(direction > 0, frame.upper_wick)
        condition = (
            frame.imb1_mean24.abs().ge(.12 if strict else .07)
            & (frame.ret12*direction).le(-.006 if strict else -.003)
            & (frame.body1*direction).gt(0)
            & rejection.ge(.30 if strict else .18)
        )
        add("depth_recovery", profile, direction, condition,
            frame.imb1_mean24.abs()-50*(frame.ret12*direction)+rejection)

        direction = np.sign(frame.imb1_delta6)
        condition = (
            frame.imb1_delta6.abs().ge(.20 if strict else .10)
            & (frame.imb1*direction).gt(0)
            & (frame.ret3*direction).gt(0)
            & frame.imbalance_volatility.ge(.03 if strict else .015)
        )
        add("wall_shift", profile, direction, condition,
            frame.imb1_delta6.abs()+frame.imbalance_volatility
            +100*(frame.ret3*direction))

        direction = np.sign(frame.imb1)
        condition = (
            frame.imb1.abs().ge(.15 if strict else .08)
            & (frame.inner_outer*direction).ge(.16 if strict else .08)
            & (frame.ret12*direction).between(-.004, .008)
            & (frame.body1*direction).gt(0)
        )
        add("inner_outer_divergence", profile, direction, condition,
            frame.imb1.abs()+frame.inner_outer*direction
            +100*(frame.body1*direction))

        direction = np.sign(frame.ret12)
        breadth = frame.breadth72.where(direction > 0, 1-frame.breadth72)
        condition = (
            frame.notional1_z168.le(-1.25 if strict else -.75)
            & frame.ret12.abs().ge(.006 if strict else .003)
            & (frame.ret3*direction).gt(0)
            & breadth.ge(.625 if strict else .50)
        )
        add("liquidity_vacuum", profile, direction, condition,
            -frame.notional1_z168+100*frame.ret12.abs()+breadth)

    return pd.concat(pieces, ignore_index=True).sort_values(
        ["entry_time", "semantic_score", "symbol"],
        ascending=[True, False, True]).reset_index(drop=True)


def route(frame: pd.DataFrame, family: str, profile: str) -> pd.DataFrame:
    selected = frame[frame.profile.eq(profile)]
    if family != "all_regimes":
        selected = selected[selected.family.eq(family)]
    return selected.sort_values("semantic_score", ascending=False).drop_duplicates(
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
    rows = [{
        **row,
        "direction": "LONG" if row["direction"] > 0 else "SHORT",
        "source": f"bookdepth_{row['family']}",
        "module": "bookdepth_material_router", "module_priority": 0,
        "outcome": "TP" if row["signed_return"] > 0 else "SL",
        "signed_price_return": row["signed_return"],
    } for row in selected.to_dict("records")]
    accepted = runtime_gate(rows)
    result = money(accepted, .03)
    n = len(accepted)
    months = list(result.get("monthly_return_pct", {}).values())
    result.update({
        "literal_target_hits": sum(bool(row["target_hit"]) for row in accepted),
        "literal_target_hit_rate": round(
            100*sum(bool(row["target_hit"]) for row in accepted)/n, 4)
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
        economic_gate([float(row["signed_price_return"]) for row in accepted],
                      observed) if accepted else {
            "n": 0, "observed_trades_per_day": 0.0,
            "projected_monthly_return_pct": 0.0,
            "frequency_pass": False, "return_pass": False, "passes": False})
    return result


def gate(value: dict, minimum_n: int) -> bool:
    return bool(value["n"] >= minimum_n and value["return_pct"] > 0
                and value["dd_pct"] <= 8 and value["negative_months"] == 0
                and value["economic_gate"]["passes"])


def main() -> None:
    candles, price_sources = price_source.load()
    hourly = outcome_source.strict_hourly_price(candles)
    raw_depth, manifest, depth_sources = load_depth()
    depth = depth_features(raw_depth)
    enriched = attach_depth(hourly, depth)
    events = build_events(enriched)
    frontier, survivors, exclusions = [], [], []
    for contract in CONTRACTS:
        outcomes, excluded = outcome_source.add_outcomes(events, candles, contract)
        exclusions.append({**contract, "excluded_crossing_gap_or_end": excluded,
                           "outcome_rows": len(outcomes)})
        for family in (*FAMILIES, "all_regimes"):
            for profile in PROFILES:
                selected = route(outcomes, family, profile)
                development = evaluate(selected, "development_2023")
                item = {**contract, "family": family, "profile": profile,
                        "development_2023": development}
                frontier.append(item)
                if not gate(development, 300):
                    continue
                validation = evaluate(selected, "validation_2024")
                item["validation_2024"] = validation
                if not gate(validation, 300):
                    continue
                item["score"] = round(
                    development["return_pct"]+2*validation["return_pct"]
                    - development["dd_pct"]-2*validation["dd_pct"], 8)
                item["_frame"] = selected
                survivors.append(item)
    survivors.sort(key=lambda item: item["score"], reverse=True)
    frozen = survivors[0] if survivors else None
    test = evaluate(frozen["_frame"], "test_2025") if frozen else None
    frontier.sort(key=lambda item: (
        item["development_2023"]["economic_gate"][
            "projected_monthly_return_pct"],
        item["development_2023"]["literal_target_hit_rate"],
        item["development_2023"]["return_pct"]), reverse=True)
    public_frozen = ({key: value for key, value in frozen.items()
                      if key != "_frame"} if frozen else None)
    status = ("RESEARCH_CANDIDATE_AWAITING_NEW_FORWARD"
              if test and gate(test, 250) else "REJECTED")
    report = {
        "status": status, "ready_for_live_orders": False,
        "protocol": {
            "source": "Binance Futures public bookDepth daily archives",
            "causal_availability": manifest["causal_availability"],
            "families": list(FAMILIES), "profiles": list(PROFILES),
            "contracts": list(CONTRACTS),
            "development": "2023", "validation": "2024 after development gate",
            "test": "2025 after validation gate",
            "money": "3% current-equity margin, fixed 10x leverage",
            "fees": "excluded by user instruction",
        },
        "source_quality": {key: manifest["quality"][key] for key in (
            "requested_days", "available_days", "missing_days", "error_days",
            "days_with_less_than_20_hours", "qualified_hours")},
        "rows": {"strict_hourly_price": len(hourly),
                 "continuous_depth": len(depth), "enriched": len(enriched),
                 "semantic_events": len(events)},
        "minimum_ideal_hit_rate_for_30pct_month_at_3_trades_day": {
            f"tp{item['target_pct']}_sl{item['stop_pct']}": round(
                minimum_binary_hit_rate_pct(
                    item["target_pct"], item["stop_pct"]), 6)
            for item in CONTRACTS},
        "exclusions": exclusions,
        "qualified_before_2025": len(survivors),
        "frozen_before_2025": public_frozen,
        "test_2025": test,
        "development_frontier": frontier[:24],
        "sources": {**price_sources, **depth_sources,
                    str(Path(__file__)): digest(Path(__file__))},
        "limitations": [
            "Depth is cumulative notional by percentage band, not a replayable L2 book.",
            "The source is Binance USDT perpetuals while runtime is OKX EU.",
            "Public history starts in 2023; any survivor needs new direct forward data.",
        ],
    }
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({
        "status": status, "source_quality": report["source_quality"],
        "rows": report["rows"], "qualified_before_2025": len(survivors),
        "frozen_before_2025": public_frozen, "test_2025": test,
        "best_development": frontier[:6],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
