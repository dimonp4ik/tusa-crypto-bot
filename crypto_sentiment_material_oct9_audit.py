"""Causal material-move audit of delayed crypto sentiment regimes."""
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
SENTIMENT_ROOT = ROOT / "crypto_alternative_fng_daily_oct9"
OUT = ROOT / "crypto_sentiment_material_oct9.json"
FAMILIES = (
    "extreme_sentiment_reversal", "sentiment_recovery",
    "sentiment_momentum", "price_sentiment_divergence", "neutral_trend",
)
PROFILES = ("broad", "strict")
CONTRACTS = (
    {"target_pct": 1.0, "stop_pct": .75, "hold_bars": 144},
    {"target_pct": 1.5, "stop_pct": .75, "hold_bars": 288},
    {"target_pct": 2.0, "stop_pct": 1.0, "hold_bars": 576},
)
PERIODS = {
    "development_2020_2021": (pd.Timestamp("2020-01-01", tz="UTC"),
                              pd.Timestamp("2022-01-01", tz="UTC")),
    "validation_2022": (pd.Timestamp("2022-01-01", tz="UTC"),
                        pd.Timestamp("2023-01-01", tz="UTC")),
    "confirmation_2023": (pd.Timestamp("2023-01-01", tz="UTC"),
                          pd.Timestamp("2024-01-01", tz="UTC")),
    "test_2024_2026": (pd.Timestamp("2024-01-01", tz="UTC"),
                       pd.Timestamp("2026-10-01", tz="UTC")),
}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_sentiment() -> tuple[pd.DataFrame, dict, dict[str, str]]:
    manifest_path = SENTIMENT_ROOT / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    raw_path, data_path = Path(manifest["raw_path"]), Path(manifest["data_path"])
    if digest(raw_path) != manifest["raw_sha256"]:
        raise ValueError("Alternative.me raw snapshot hash mismatch")
    if digest(data_path) != manifest["data_sha256"]:
        raise ValueError("Alternative.me normalized snapshot hash mismatch")
    frame = pd.DataFrame(pickle.loads(data_path.read_bytes())).sort_values(
        "source_epoch").drop_duplicates("source_epoch")
    times = frame.source_epoch.to_numpy(np.int64)
    if np.any(np.diff(times) < 86400) or np.any(times % 86400 != 0):
        raise ValueError("invalid sentiment source chronology")
    if not np.array_equal(frame.available_time.to_numpy(np.int64), times+86400):
        raise ValueError("sentiment availability is not delayed by 24 hours")
    return frame, manifest, {
        str(manifest_path): digest(manifest_path), str(raw_path): digest(raw_path),
        str(data_path): digest(data_path),
    }


def sentiment_features(frame: pd.DataFrame) -> pd.DataFrame:
    value = frame.sort_values("source_epoch").copy()
    score = value.value.astype(float)
    for window in (1, 3, 7, 30):
        value[f"sentiment_delta{window}"] = score-score.shift(window)
    value["sentiment_mean7"] = score.rolling(7, min_periods=7).mean()
    value["sentiment_mean30"] = score.rolling(30, min_periods=25).mean()
    mean = score.rolling(90, min_periods=60).mean()
    std = score.rolling(90, min_periods=60).std().replace(0, np.nan)
    value["sentiment_z90"] = (score-mean)/std
    value["sentiment_vol7"] = score.diff().abs().rolling(7, min_periods=5).mean()
    needed = [
        "sentiment_delta1", "sentiment_delta3", "sentiment_delta7",
        "sentiment_delta30", "sentiment_mean7", "sentiment_mean30",
        "sentiment_z90", "sentiment_vol7",
    ]
    return value.replace([np.inf, -np.inf], np.nan).dropna(subset=needed)[
        ["source_epoch", "available_time", "value", *needed]]


def attach_sentiment(hourly: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    value = pd.merge_asof(
        hourly.sort_values("entry_time"), features.sort_values("available_time"),
        left_on="entry_time", right_on="available_time", direction="backward",
        tolerance=48*3600,
    ).dropna(subset=["available_time"])
    if (value.entry_time < value.available_time).any():
        raise ValueError("sentiment value used before conservative availability")
    return value


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

        low, high = ((20, 80) if strict else (25, 75))
        direction = pd.Series(np.where(
            frame.value.le(low), 1, np.where(frame.value.ge(high), -1, 0)),
            index=frame.index)
        rejection = frame.lower_wick.where(direction > 0, frame.upper_wick)
        condition = (
            (frame.btc_ret72*direction).le(-.04 if strict else -.02)
            & (frame.body1*direction).gt(0)
            & rejection.ge(.30 if strict else .18)
        )
        add("extreme_sentiment_reversal", profile, direction, condition,
            (frame.value-50).abs()/25-20*(frame.btc_ret72*direction)+rejection)

        direction = np.sign(frame.sentiment_delta3)
        turning = ((frame.value.le(30 if strict else 35) & direction.gt(0))
                   | (frame.value.ge(70 if strict else 65) & direction.lt(0)))
        condition = (
            turning
            & frame.sentiment_delta3.abs().ge(8 if strict else 5)
            & (frame.ret3*direction).gt(0)
            & (frame.body1*direction).gt(0)
        )
        add("sentiment_recovery", profile, direction, condition,
            frame.sentiment_delta3.abs()/10+50*(frame.ret3*direction)
            +(frame.value-50).abs()/50)

        direction = np.sign(frame.sentiment_delta7)
        breadth = frame.breadth72.where(direction > 0, 1-frame.breadth72)
        condition = (
            frame.sentiment_delta7.abs().ge(20 if strict else 12)
            & (frame.ret12*direction).ge(.006 if strict else .003)
            & (frame.body1*direction).gt(0)
            & breadth.ge(.625 if strict else .50)
        )
        add("sentiment_momentum", profile, direction, condition,
            frame.sentiment_delta7.abs()/10+50*(frame.ret12*direction)+breadth)

        direction = np.sign(frame.sentiment_delta7)
        condition = (
            frame.sentiment_delta7.abs().ge(18 if strict else 10)
            & (frame.btc_ret72*direction).le(-.02 if strict else -.01)
            & (frame.ret3*direction).gt(0)
            & (frame.body1*direction).gt(0)
        )
        add("price_sentiment_divergence", profile, direction, condition,
            frame.sentiment_delta7.abs()/10-20*(frame.btc_ret72*direction)
            +50*(frame.ret3*direction))

        direction = np.sign(frame.ret72)
        breadth = frame.breadth72.where(direction > 0, 1-frame.breadth72)
        neutral = frame.value.between(45, 55) if strict else frame.value.between(40, 60)
        condition = (
            neutral & frame.sentiment_delta7.abs().le(5 if strict else 10)
            & (frame.ret72*direction).ge(.02 if strict else .01)
            & (frame.ret12*direction).gt(0)
            & (frame.body1*direction).gt(0)
            & breadth.ge(.625 if strict else .50)
        )
        add("neutral_trend", profile, direction, condition,
            20*(frame.ret72*direction)+50*(frame.ret12*direction)+breadth)

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
        "source": f"sentiment_{row['family']}",
        "module": "sentiment_material_router", "module_priority": 0,
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
    raw_sentiment, manifest, sentiment_sources = load_sentiment()
    features = sentiment_features(raw_sentiment)
    enriched = attach_sentiment(hourly, features)
    events = build_events(enriched)
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
                if not gate(development, 500):
                    continue
                validation = evaluate(selected, "validation_2022")
                item["validation_2022"] = validation
                if not gate(validation, 250):
                    continue
                confirmation = evaluate(selected, "confirmation_2023")
                item["confirmation_2023"] = confirmation
                if not gate(confirmation, 250):
                    continue
                item["score"] = round(
                    development["return_pct"]+validation["return_pct"]
                    +2*confirmation["return_pct"]-development["dd_pct"]
                    -validation["dd_pct"]-2*confirmation["dd_pct"], 8)
                item["_frame"] = selected
                survivors.append(item)
    survivors.sort(key=lambda item: item["score"], reverse=True)
    frozen = survivors[0] if survivors else None
    test = evaluate(frozen["_frame"], "test_2024_2026") if frozen else None
    frontier.sort(key=lambda item: (
        item["development_2020_2021"]["economic_gate"][
            "projected_monthly_return_pct"],
        item["development_2020_2021"]["literal_target_hit_rate"],
        item["development_2020_2021"]["return_pct"]), reverse=True)
    status = ("RESEARCH_CANDIDATE_AWAITING_NEW_FORWARD"
              if test and gate(test, 500) else "REJECTED")
    report = {
        "status": status, "ready_for_live_orders": False,
        "protocol": {
            "source": "Alternative.me Crypto Fear and Greed Index",
            "causal_availability": manifest["causal_availability"],
            "families": list(FAMILIES), "profiles": list(PROFILES),
            "contracts": list(CONTRACTS),
            "development": "2020-2021",
            "validation": "2022 after development gate",
            "confirmation": "2023 after validation gate",
            "test": "2024 through September 2026 after confirmation gate",
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
        "frozen_before_2024": public(frozen), "test_2024_2026": test,
        "development_frontier": frontier[:24],
        "sources": {**price_sources, **sentiment_sources,
                    str(Path(__file__)): digest(Path(__file__))},
        "limitations": [
            "The composite index includes price-derived inputs and is not an independent execution feed.",
            "Historical publication times are absent; the audit delays every value by 24 hours.",
            "The exact API response is hashed because the provider may revise history.",
            "Signals transfer from Binance history to OKX EU and need direct forward evidence.",
        ],
    }
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({
        "status": status, "rows": report["rows"],
        "qualified_before_2024": len(survivors),
        "frozen_before_2024": public(frozen), "test_2024_2026": test,
        "best_development": frontier[:6],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
