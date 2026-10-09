"""Causal material-move audit of BTC/ETH network activity regimes."""
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
CHAIN_ROOT = ROOT / "crypto_coinmetrics_onchain_daily_oct9"
OUT = ROOT / "crypto_onchain_material_oct9.json"
FAMILIES = (
    "network_demand_continuation", "activity_price_divergence",
    "fee_capitulation", "valuation_recovery", "eth_network_rotation",
)
PROFILES = ("broad", "strict")
CONTRACTS = (
    {"target_pct": 1.0, "stop_pct": .75, "hold_bars": 144},
    {"target_pct": 1.5, "stop_pct": .75, "hold_bars": 288},
    {"target_pct": 2.0, "stop_pct": 1.0, "hold_bars": 576},
)
PERIODS = {
    "development_2021": (pd.Timestamp("2021-03-01", tz="UTC"),
                         pd.Timestamp("2022-01-01", tz="UTC")),
    "validation_2022": (pd.Timestamp("2022-01-01", tz="UTC"),
                        pd.Timestamp("2023-01-01", tz="UTC")),
    "confirmation_2023": (pd.Timestamp("2023-01-01", tz="UTC"),
                          pd.Timestamp("2024-01-01", tz="UTC")),
    "test_2024_2025": (pd.Timestamp("2024-01-01", tz="UTC"),
                       pd.Timestamp("2026-01-01", tz="UTC")),
}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_chain() -> tuple[pd.DataFrame, dict, dict[str, str]]:
    manifest_path = CHAIN_ROOT / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    raw_path = Path(manifest["raw_path"])
    data_path = Path(manifest["data_path"])
    if digest(raw_path) != manifest["raw_sha256"]:
        raise ValueError("Coin Metrics raw snapshot hash mismatch")
    if digest(data_path) != manifest["data_sha256"]:
        raise ValueError("Coin Metrics normalized snapshot hash mismatch")
    values = pickle.loads(data_path.read_bytes())
    frame = pd.DataFrame(values).sort_values(
        ["asset", "source_epoch"]).reset_index(drop=True)
    if frame.duplicated(["asset", "source_epoch"]).any():
        raise ValueError("duplicate Coin Metrics asset/day rows")
    for _, value in frame.groupby("asset"):
        if np.any(np.diff(value.source_epoch.to_numpy(np.int64)) != 86400):
            raise ValueError("non-continuous Coin Metrics source days")
        if np.any(np.diff(value.available_time.to_numpy(np.int64)) < 0):
            raise ValueError("non-monotonic Coin Metrics availability")
    return frame, manifest, {
        str(manifest_path): digest(manifest_path),
        str(raw_path): digest(raw_path), str(data_path): digest(data_path),
    }


def _asset_features(value: pd.DataFrame, prefix: str) -> pd.DataFrame:
    value = value.sort_values("source_epoch").copy()
    components = []
    for raw, label in (("AdrActCnt", "activity"), ("TxTfrCnt", "transfer"),
                       ("FeeTotNtv", "fee")):
        logged = np.log1p(value[raw].clip(lower=0))
        mean = logged.rolling(90, min_periods=60).mean()
        std = logged.rolling(90, min_periods=60).std().replace(0, np.nan)
        value[f"{label}_z90"] = (logged-mean)/std
        value[f"{label}_delta7"] = logged-logged.shift(7)
        value[f"{label}_delta30"] = logged-logged.shift(30)
        components.append(value[f"{label}_z90"])
    mvrv_log = np.log(value.CapMVRVCur.clip(lower=1e-12))
    mean = mvrv_log.rolling(180, min_periods=120).mean()
    std = mvrv_log.rolling(180, min_periods=120).std().replace(0, np.nan)
    value["mvrv_z180"] = (mvrv_log-mean)/std
    value["mvrv_delta30"] = mvrv_log-mvrv_log.shift(30)
    value["network_score"] = sum(components)/len(components)
    value["network_delta7"] = (
        value.activity_delta7+value.transfer_delta7+value.fee_delta7)/3
    keep = [
        "source_epoch", "available_time", "causal_eligible", "network_score",
        "network_delta7", "fee_z90", "mvrv_z180", "mvrv_delta30",
    ]
    return value[keep].rename(columns={
        name: f"{prefix}_{name}" for name in keep
        if name not in ("source_epoch",)
    })


def chain_features(frame: pd.DataFrame) -> pd.DataFrame:
    btc = _asset_features(frame[frame.asset.eq("btc")], "btc")
    eth = _asset_features(frame[frame.asset.eq("eth")], "eth")
    value = btc.merge(eth, on="source_epoch", how="inner", validate="one_to_one")
    value["available_time"] = value[[
        "btc_available_time", "eth_available_time"]].max(axis=1).astype(np.int64)
    value["causal_eligible"] = (
        value.btc_causal_eligible & value.eth_causal_eligible)
    value["network_demand_z"] = (
        value.btc_network_score+value.eth_network_score)/2
    value["network_delta7"] = (
        value.btc_network_delta7+value.eth_network_delta7)/2
    value["fee_shock_z"] = (value.btc_fee_z90+value.eth_fee_z90)/2
    value["mvrv_z"] = (value.btc_mvrv_z180+value.eth_mvrv_z180)/2
    value["mvrv_delta30"] = (
        value.btc_mvrv_delta30+value.eth_mvrv_delta30)/2
    value["eth_rotation"] = value.eth_network_score-value.btc_network_score
    value["eth_rotation_delta7"] = (
        value.eth_network_delta7-value.btc_network_delta7)
    needed = [
        "network_demand_z", "network_delta7", "fee_shock_z", "mvrv_z",
        "mvrv_delta30", "eth_rotation", "eth_rotation_delta7",
    ]
    value = value[value.causal_eligible].replace(
        [np.inf, -np.inf], np.nan).dropna(subset=needed)
    return value[["source_epoch", "available_time", *needed]].sort_values(
        "available_time").drop_duplicates("available_time")


def attach_chain(hourly: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    value = pd.merge_asof(
        hourly.sort_values("entry_time"), features.sort_values("available_time"),
        left_on="entry_time", right_on="available_time", direction="backward",
        tolerance=72*3600,
    ).dropna(subset=["available_time"])
    if (value.entry_time < value.available_time).any():
        raise ValueError("on-chain feature used before recorded availability")
    return value


def build_events(frame: pd.DataFrame) -> pd.DataFrame:
    pieces = []

    def add(family: str, profile: str, direction: pd.Series,
            condition: pd.Series, strength: pd.Series,
            symbols: pd.Series | None = None) -> None:
        selected = condition & direction.ne(0)
        if symbols is not None:
            selected &= symbols
        part = frame.loc[selected].copy()
        part["direction"] = direction.loc[selected].astype(np.int8)
        part["family"], part["profile"] = family, profile
        part["semantic_score"] = strength.loc[selected].astype(float)
        pieces.append(part)

    for profile in PROFILES:
        strict = profile == "strict"

        direction = np.sign(frame.ret12)
        breadth = frame.breadth72.where(direction > 0, 1-frame.breadth72)
        condition = (
            frame.network_demand_z.ge(1.0 if strict else .5)
            & frame.network_delta7.ge(.05 if strict else 0)
            & (frame.ret12*direction).ge(.006 if strict else .003)
            & (frame.body1*direction).gt(0)
            & breadth.ge(.625 if strict else .50)
        )
        add("network_demand_continuation", profile, direction, condition,
            frame.network_demand_z+5*frame.network_delta7
            +50*(frame.ret12*direction)+breadth)

        direction = np.sign(frame.network_delta7)
        condition = (
            frame.network_delta7.abs().ge(.20 if strict else .10)
            & (frame.btc_ret72*direction).le(-.02 if strict else -.01)
            & (frame.body1*direction).gt(0)
            & (frame.ret3*direction).gt(0)
        )
        add("activity_price_divergence", profile, direction, condition,
            5*frame.network_delta7.abs()-20*(frame.btc_ret72*direction)
            +50*(frame.body1*direction))

        direction = -np.sign(frame.btc_ret72)
        rejection = frame.lower_wick.where(direction > 0, frame.upper_wick)
        condition = (
            frame.fee_shock_z.ge(1.5 if strict else 1.0)
            & frame.btc_ret72.abs().ge(.05 if strict else .03)
            & (frame.body1*direction).gt(0)
            & rejection.ge(.30 if strict else .18)
        )
        add("fee_capitulation", profile, direction, condition,
            frame.fee_shock_z+20*frame.btc_ret72.abs()+rejection)

        direction = np.sign(frame.mvrv_delta30)
        condition = (
            frame.mvrv_delta30.abs().ge(.10 if strict else .05)
            & (frame.ret12*direction).ge(.004 if strict else .002)
            & (frame.body1*direction).gt(0)
            & frame.network_demand_z.ge(-.5 if strict else -1.0)
        )
        add("valuation_recovery", profile, direction, condition,
            5*frame.mvrv_delta30.abs()+frame.mvrv_z.abs()
            +50*(frame.ret12*direction))

        direction = np.sign(frame.eth_rotation_delta7)
        alt = ~frame.symbol.isin(("BTCUSDT", "ETHUSDT"))
        condition = (
            frame.eth_rotation_delta7.abs().ge(.25 if strict else .15)
            & (frame.ret12*direction).ge(.004 if strict else .002)
            & (frame.cross_z72*direction).gt(0)
            & (frame.body1*direction).gt(0)
        )
        add("eth_network_rotation", profile, direction, condition,
            4*frame.eth_rotation_delta7.abs()+frame.eth_rotation.abs()
            +50*(frame.ret12*direction), alt)

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
        **row, "direction": "LONG" if row["direction"] > 0 else "SHORT",
        "source": f"onchain_{row['family']}",
        "module": "onchain_material_router", "module_priority": 0,
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


def public(item: dict | None) -> dict | None:
    return ({key: value for key, value in item.items() if key != "_frame"}
            if item else None)


def main() -> None:
    candles, price_sources = price_source.load()
    hourly = outcome_source.strict_hourly_price(candles)
    raw_chain, manifest, chain_sources = load_chain()
    features = chain_features(raw_chain)
    enriched = attach_chain(hourly, features)
    events = build_events(enriched)
    frontier, survivors, exclusions = [], [], []
    for contract in CONTRACTS:
        outcomes, excluded = outcome_source.add_outcomes(events, candles, contract)
        exclusions.append({**contract, "excluded_crossing_gap_or_end": excluded,
                           "outcome_rows": len(outcomes)})
        for family in (*FAMILIES, "all_regimes"):
            for profile in PROFILES:
                selected = route(outcomes, family, profile)
                development = evaluate(selected, "development_2021")
                item = {**contract, "family": family, "profile": profile,
                        "development_2021": development}
                frontier.append(item)
                if not gate(development, 250):
                    continue
                validation = evaluate(selected, "validation_2022")
                item["validation_2022"] = validation
                if not gate(validation, 300):
                    continue
                confirmation = evaluate(selected, "confirmation_2023")
                item["confirmation_2023"] = confirmation
                if not gate(confirmation, 300):
                    continue
                item["score"] = round(
                    development["return_pct"]+validation["return_pct"]
                    +2*confirmation["return_pct"]-development["dd_pct"]
                    -validation["dd_pct"]-2*confirmation["dd_pct"], 8)
                item["_frame"] = selected
                survivors.append(item)
    survivors.sort(key=lambda item: item["score"], reverse=True)
    frozen = survivors[0] if survivors else None
    test = evaluate(frozen["_frame"], "test_2024_2025") if frozen else None
    frontier.sort(key=lambda item: (
        item["development_2021"]["economic_gate"][
            "projected_monthly_return_pct"],
        item["development_2021"]["literal_target_hit_rate"],
        item["development_2021"]["return_pct"]), reverse=True)
    status = ("RESEARCH_CANDIDATE_AWAITING_NEW_FORWARD"
              if test and gate(test, 500) else "REJECTED")
    report = {
        "status": status, "ready_for_live_orders": False,
        "protocol": {
            "source": "Coin Metrics Community BTC/ETH daily network metrics",
            "causal_availability": manifest["causal_availability"],
            "excluded_metrics": manifest["reviewable_entity_metrics_excluded"],
            "families": list(FAMILIES), "profiles": list(PROFILES),
            "contracts": list(CONTRACTS),
            "development": "2021-03 through 2021",
            "validation": "2022 after development gate",
            "confirmation": "2023 after validation gate",
            "test": "2024-2025 after confirmation gate",
            "money": "3% current-equity margin, fixed 10x leverage",
            "orders": "market entries only", "fees": "excluded by user instruction",
        },
        "source_quality": manifest["quality"],
        "rows": {"strict_hourly_price": len(hourly),
                 "causal_feature_days": len(features),
                 "enriched": len(enriched), "semantic_events": len(events)},
        "minimum_ideal_hit_rate_for_30pct_month_at_3_trades_day": {
            f"tp{item['target_pct']}_sl{item['stop_pct']}": round(
                minimum_binary_hit_rate_pct(
                    item["target_pct"], item["stop_pct"]), 6)
            for item in CONTRACTS},
        "exclusions": exclusions,
        "qualified_before_2024": len(survivors),
        "frozen_before_2024": public(frozen), "test_2024_2025": test,
        "development_frontier": frontier[:24],
        "sources": {**price_sources, **chain_sources,
                    str(Path(__file__)): digest(Path(__file__))},
        "limitations": [
            "Daily network metrics can gate intraday entries but cannot time them alone.",
            "Current API snapshots can be revised; the exact raw response is hashed.",
            "The audit excludes entity-labelled exchange flows because their history is backfilled.",
            "Signals transfer from Binance history to OKX EU and need direct forward evidence.",
        ],
    }
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({
        "status": status, "rows": report["rows"],
        "qualified_before_2024": len(survivors),
        "frozen_before_2024": public(frozen), "test_2024_2025": test,
        "best_development": frontier[:6],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
