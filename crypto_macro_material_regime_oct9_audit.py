"""Material crypto moves conditioned on causal daily cross-asset regimes."""
from __future__ import annotations

import hashlib
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

import crypto_deribit_dvol_regime_oct9_audit as outcome_source
import crypto_futures_trade_flow_regime_oct7_audit as flow_source
from crypto_runtime_shadow_portfolio_oct3_audit import money, runtime_gate
from src.research_profit_gate import economic_gate, minimum_binary_hit_rate_pct


ROOT = Path("reports/audit_2026_09_08")
MACRO_ROOT = ROOT / "crypto_macro_daily_history_oct9"
OUT = ROOT / "crypto_macro_material_regime_oct9.json"
ASSETS = ("qqq", "vix", "dxy", "gold", "us10y")
FAMILIES = ("risk_continuation", "crypto_catchup", "shock_reversal",
            "liquidity_trend", "nasdaq_btc_divergence")
PROFILES = ("broad", "strict")
CONFIGS = (
    {"max_leaf_nodes": 15, "min_samples_leaf": 250, "l2": 6.0},
    {"max_leaf_nodes": 31, "min_samples_leaf": 450, "l2": 10.0},
)
QUANTILES = (.70, .75, .80, .85, .90, .925, .95, .97, .98, .99)
PERIODS = {
    "fit_2020_2021": (pd.Timestamp("2020-04-01", tz="UTC"),
                      pd.Timestamp("2022-01-01", tz="UTC")),
    "selection_2022": (pd.Timestamp("2022-01-01", tz="UTC"),
                       pd.Timestamp("2023-01-01", tz="UTC")),
    "validation_2023": (pd.Timestamp("2023-01-01", tz="UTC"),
                        pd.Timestamp("2024-01-01", tz="UTC")),
    "confirmation_2024": (pd.Timestamp("2024-01-01", tz="UTC"),
                          pd.Timestamp("2025-01-01", tz="UTC")),
    "test_2025": (pd.Timestamp("2025-01-01", tz="UTC"),
                  pd.Timestamp("2026-01-01", tz="UTC")),
    "untouched_2026": (pd.Timestamp("2026-01-01", tz="UTC"),
                       pd.Timestamp("2027-01-01", tz="UTC")),
}
PRICE_DIRECTIONAL = (
    "ret3", "ret6", "ret12", "ret36", "ret72", "ret144", "ret288",
    "body1", "ema12_dist", "ema48_dist", "cross_z12", "cross_z72",
    "btc_ret12", "btc_ret72", "btc_ret288",
)
PRICE_ABSOLUTE = ("range1", "atr14_pct", "volume_ratio", "range_ratio",
                  "eff72", "upper_wick", "lower_wick")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_macro() -> tuple[dict[str, dict], dict[str, str], dict]:
    manifest_path = MACRO_ROOT / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    data, sources = {}, {str(manifest_path): digest(manifest_path)}
    for asset in ASSETS:
        item = manifest["assets"][asset]
        path = Path(item["path"])
        raw = path.read_bytes()
        found = hashlib.sha256(raw).hexdigest()
        if found != item["sha256"]:
            raise ValueError(f"macro hash mismatch: {asset}")
        data[asset] = pickle.loads(raw)
        sources[str(path)] = found
    return data, sources, manifest


def asset_features(asset: str, raw: dict, timezone_name: str) -> pd.DataFrame:
    frame = pd.DataFrame(raw).sort_values("time").drop_duplicates("time")
    local = pd.to_datetime(frame.time, unit="s", utc=True).dt.tz_convert(timezone_name)
    session_date = local.dt.date
    # A bar dated D is exposed only at 00:00 UTC on D+1.  Every source in this
    # archive has completed its regular settlement by then.
    available = pd.to_datetime(session_date, utc=True) + pd.Timedelta(days=1)
    close = frame.adjusted_close.astype(float)
    # Pandas may preserve second rather than nanosecond resolution here, so
    # convert explicitly instead of assuming the integer unit.
    output = pd.DataFrame({
        "available_time": available.map(lambda value: int(value.timestamp()))
    })
    for days in (1, 3, 5, 10, 20):
        output[f"{asset}_ret{days}"] = close / close.shift(days) - 1
    mean = close.rolling(60, min_periods=60).mean()
    std = close.rolling(60, min_periods=60).std().replace(0, np.nan)
    output[f"{asset}_z60"] = (close - mean) / std
    output[f"{asset}_vol20"] = close.pct_change().rolling(20, min_periods=20).std()
    output[f"{asset}_range"] = (frame.high - frame.low) / frame.close
    return output.replace([np.inf, -np.inf], np.nan).dropna()


def attach_macro(hourly: pd.DataFrame, data: dict[str, dict],
                 manifest: dict) -> pd.DataFrame:
    result = hourly.sort_values("entry_time").copy()
    for asset in ASSETS:
        timezone_name = manifest["assets"][asset]["exchange_timezone"]
        right = asset_features(asset, data[asset], timezone_name)
        result = pd.merge_asof(
            result.sort_values("entry_time"), right.sort_values("available_time"),
            left_on="entry_time", right_on="available_time", direction="backward",
            tolerance=7 * 86400)
        result = result.drop(columns="available_time")
    needed = [column for column in result if any(
        column.startswith(f"{asset}_") for asset in ASSETS)]
    return result.dropna(subset=needed).reset_index(drop=True)


def build_events(frame: pd.DataFrame) -> pd.DataFrame:
    pieces = []
    # Fixed scaling converts unlike daily assets into a bounded directional
    # vote without fitting on later returns.
    risk = (frame.qqq_ret3 / .02 - frame.vix_ret3 / .15
            - frame.dxy_ret3 / .01 - frame.us10y_ret3 / .08) / 4
    liquidity = -(frame.dxy_ret5 / .015 + frame.us10y_ret5 / .10) / 2

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
        direction = np.sign(risk)
        breadth = frame.breadth72.where(direction > 0, 1 - frame.breadth72)
        condition = (
            risk.abs().ge(1.0 if strict else .55)
            & (frame.btc_ret72 * direction).ge(.006 if strict else .002)
            & (frame.ret12 * direction).ge(.001 if strict else 0)
            & breadth.ge(.70 if strict else .60)
        )
        add("risk_continuation", profile, direction, condition,
            risk.abs() + 50 * (frame.btc_ret72 * direction) + breadth)

        direction = np.sign(risk)
        condition = (
            risk.abs().ge(1.1 if strict else .65)
            & (frame.btc_ret72 * direction).le(-.008 if strict else -.003)
            & (frame.body1 * direction).gt(0)
            & (frame.ret3 * direction).gt(0)
        )
        add("crypto_catchup", profile, direction, condition,
            risk.abs() - 50 * (frame.btc_ret72 * direction)
            + 100 * (frame.body1 * direction))

        macro_move = np.sign(frame.qqq_ret1)
        direction = -macro_move
        rejection = frame.lower_wick.where(direction > 0, frame.upper_wick)
        shock = (frame.vix_ret1.abs().ge(.15 if strict else .08)
                 & frame.qqq_ret1.abs().ge(.015 if strict else .008)
                 & (frame.vix_ret1 * frame.qqq_ret1).lt(0))
        condition = (
            shock & (frame.ret72 * macro_move).ge(.008 if strict else .003)
            & rejection.ge(.35 if strict else .22)
            & (frame.body1 * direction).gt(0)
        )
        add("shock_reversal", profile, direction, condition,
            5 * frame.vix_ret1.abs() + 20 * frame.qqq_ret1.abs() + rejection)

        direction = np.sign(liquidity)
        condition = (
            liquidity.abs().ge(1.0 if strict else .55)
            & (frame.qqq_ret5 * direction).ge(.015 if strict else .004)
            & (frame.btc_ret288 * direction).ge(.012 if strict else .004)
            & (frame.ret12 * direction).gt(0)
        )
        add("liquidity_trend", profile, direction, condition,
            liquidity.abs() + 30 * (frame.qqq_ret5 * direction)
            + 20 * (frame.btc_ret288 * direction))

        direction = np.sign(frame.qqq_ret5)
        gap = frame.qqq_ret5 - frame.btc_ret288
        directional_gap = gap * direction
        condition = (
            frame.qqq_ret5.abs().ge(.02 if strict else .01)
            & directional_gap.ge(.025 if strict else .012)
            & (frame.ret12 * direction).gt(0)
            & (frame.cross_z72 * direction).le(.5 if strict else 1.0)
        )
        add("nasdaq_btc_divergence", profile, direction, condition,
            20 * directional_gap + 20 * frame.qqq_ret5.abs()
            - frame.cross_z72 * direction)

    return pd.concat(pieces, ignore_index=True).sort_values(
        ["entry_time", "semantic_score", "symbol"],
        ascending=[True, False, True]).reset_index(drop=True)


def candidate_union(outcomes: pd.DataFrame) -> pd.DataFrame:
    broad = outcomes[outcomes.profile.eq("broad")].copy()
    keys = ["symbol", "entry_time", "direction"]
    flags = broad.assign(value=1).pivot_table(
        index=keys, columns="family", values="value", aggfunc="max",
        fill_value=0).reset_index()
    flags.columns.name = None
    flags = flags.rename(columns={name: f"family_{name}" for name in FAMILIES})
    base = broad.sort_values("semantic_score", ascending=False).drop_duplicates(keys)
    base = base.drop(columns=["family", "profile"])
    result = base.merge(flags, on=keys, how="left", validate="one_to_one")
    strict = outcomes[outcomes.profile.eq("strict")][keys].drop_duplicates()
    strict["strict_profile_match"] = 1
    result = result.merge(strict, on=keys, how="left", validate="one_to_one")
    result["strict_profile_match"] = result.strict_profile_match.fillna(0)
    for family in FAMILIES:
        if f"family_{family}" not in result:
            result[f"family_{family}"] = 0
    return result


def matrix(frame: pd.DataFrame, include_macro: bool) -> np.ndarray:
    value = pd.DataFrame(index=frame.index)
    direction = frame.direction.to_numpy(float)
    for name in PRICE_DIRECTIONAL:
        value[f"{name}_dir"] = frame[name] * direction
        value[f"{name}_abs"] = frame[name].abs()
    for name in PRICE_ABSOLUTE:
        value[name] = frame[name]
    value["breadth12_dir"] = np.where(
        direction > 0, frame.breadth12, 1 - frame.breadth12)
    value["breadth72_dir"] = np.where(
        direction > 0, frame.breadth72, 1 - frame.breadth72)
    value["breakout_dir"] = np.where(
        direction > 0, frame.breakout_up, frame.breakout_down)
    stamp = pd.to_datetime(frame.entry_time, unit="s", utc=True)
    value["hour_sin"] = np.sin(2 * np.pi * stamp.dt.hour / 24)
    value["hour_cos"] = np.cos(2 * np.pi * stamp.dt.hour / 24)
    for family in FAMILIES:
        value[f"family_{family}"] = frame[f"family_{family}"].astype(float)
    value["strict_profile_match"] = frame.strict_profile_match.astype(float)
    for symbol in sorted(frame.symbol.unique()):
        value[f"symbol_{symbol}"] = (frame.symbol == symbol).astype(float)
    if include_macro:
        for asset in ASSETS:
            for suffix in ("ret1", "ret3", "ret5", "ret10", "ret20",
                           "z60", "vol20", "range"):
                column = f"{asset}_{suffix}"
                value[column] = frame[column]
        for asset in ASSETS:
            value[f"{asset}_ret3_dir"] = frame[f"{asset}_ret3"] * direction
            value[f"{asset}_ret5_dir"] = frame[f"{asset}_ret5"] * direction
    return value.replace([np.inf, -np.inf], np.nan).fillna(0).to_numpy(np.float32)


def period_mask(frame: pd.DataFrame, period: str) -> np.ndarray:
    start, end = PERIODS[period]
    stamp = pd.to_datetime(frame.entry_time, unit="s", utc=True)
    return ((stamp >= start) & (stamp < end)).to_numpy()


def evaluate(frame: pd.DataFrame, period: str, threshold: float) -> dict:
    selected = frame.loc[
        period_mask(frame, period) & frame.quality.ge(threshold)].copy()
    selected = selected.sort_values(
        ["entry_time", "quality", "semantic_score", "symbol"],
        ascending=[True, False, False, True])
    selected["source_priority"] = selected.groupby("entry_time").cumcount()
    rows = []
    for row in selected.to_dict("records"):
        family = next((name for name in FAMILIES
                       if row.get(f"family_{name}", 0)), "regime")
        rows.append({
            **row,
            "direction": "LONG" if row["direction"] > 0 else "SHORT",
            "source": f"macro_{family}", "module": "macro_material_router",
            "module_priority": 0,
            "outcome": "TP" if row["signed_return"] > 0 else "SL",
            "signed_price_return": row["signed_return"],
        })
    accepted = runtime_gate(rows)
    result = money(accepted, .03)
    n = len(accepted)
    result.update({
        "literal_target_hits": sum(bool(row["target_hit"]) for row in accepted),
        "literal_target_hit_rate": round(
            100 * sum(bool(row["target_hit"]) for row in accepted) / n, 4)
            if n else 0.0,
        "stops": sum(int(row["exit_reason"]) == 0 for row in accepted),
        "timeouts": sum(int(row["exit_reason"]) == 2 for row in accepted),
        "months": len(result.get("monthly_return_pct", {})),
        "months_ge_30pct": sum(
            value >= 30 for value in result.get("monthly_return_pct", {}).values()),
    })
    observed = float(result.get("trades_per_calendar_day", 0.0) or 0.0)
    result["economic_gate"] = economic_gate(
        [float(row["signed_price_return"]) for row in accepted], observed) \
        if accepted else {"n": 0, "observed_trades_per_day": 0.0,
                          "projected_monthly_return_pct": 0.0,
                          "frequency_pass": False, "return_pass": False,
                          "passes": False}
    return result


def pass_gate(value: dict, minimum_n: int) -> bool:
    return bool(value["n"] >= minimum_n and value["return_pct"] > 0
                and value["dd_pct"] <= 8 and value["negative_months"] == 0
                and value["economic_gate"]["passes"])


def public(item: dict) -> dict:
    return {key: value for key, value in item.items()
            if key not in {"_frame", "_model"}}


def main() -> None:
    candles, price_sources = flow_source.load()
    macro, macro_sources, manifest = load_macro()
    hourly = outcome_source.strict_hourly_price(candles)
    enriched = attach_macro(hourly, macro, manifest)
    events = build_events(enriched)
    frontier, survivors, exclusions = [], [], []
    for contract in outcome_source.CONTRACTS:
        outcomes, excluded = outcome_source.add_outcomes(events, candles, contract)
        candidates = candidate_union(outcomes)
        exclusions.append({**contract, "excluded_crossing_gap_or_end": excluded,
                           "candidate_rows": len(candidates)})
        fit = (period_mask(candidates, "fit_2020_2021").copy()
               & candidates.exit_time.lt(
                   PERIODS["fit_2020_2021"][1].timestamp()).to_numpy())
        selection = period_mask(candidates, "selection_2022")
        y = candidates.target_hit.astype(int).to_numpy()
        matrices = {
            "semantic_price_ranker": matrix(candidates, False),
            "semantic_price_macro_ranker": matrix(candidates, True),
        }
        for kind, values in matrices.items():
            for config in CONFIGS:
                model = HistGradientBoostingClassifier(
                    max_iter=180, learning_rate=.06,
                    max_leaf_nodes=config["max_leaf_nodes"],
                    min_samples_leaf=config["min_samples_leaf"],
                    l2_regularization=config["l2"], random_state=20261009)
                model.fit(values[fit], y[fit])
                scored = candidates.copy()
                scored["quality"] = model.predict_proba(values)[:, 1]
                for quantile in QUANTILES:
                    threshold = float(scored.loc[selection, "quality"].quantile(quantile))
                    selected = evaluate(scored, "selection_2022", threshold)
                    item = {**contract, "model_kind": kind, "model": config,
                            "quantile": quantile, "threshold": threshold,
                            "selection_2022": selected}
                    frontier.append(item)
                    if not pass_gate(selected, 350):
                        continue
                    validation = evaluate(scored, "validation_2023", threshold)
                    item["validation_2023"] = validation
                    if not pass_gate(validation, 300):
                        continue
                    item["score"] = round(
                        selected["return_pct"] + 2 * validation["return_pct"]
                        - selected["dd_pct"] - 2 * validation["dd_pct"], 8)
                    item["_frame"], item["_model"] = scored, model
                    survivors.append(item)
    survivors.sort(key=lambda item: item["score"], reverse=True)
    full = [item for item in survivors
            if item["model_kind"] == "semantic_price_macro_ranker"]
    frozen = full[0] if full else None
    confirmation = test = reveal = None
    if frozen:
        confirmation = evaluate(frozen["_frame"], "confirmation_2024",
                                frozen["threshold"])
        if pass_gate(confirmation, 300):
            test = evaluate(frozen["_frame"], "test_2025", frozen["threshold"])
        if test and pass_gate(test, 300):
            reveal = evaluate(frozen["_frame"], "untouched_2026",
                              frozen["threshold"])
    frontier.sort(key=lambda item: (
        item["selection_2022"]["economic_gate"]["projected_monthly_return_pct"],
        item["selection_2022"]["return_pct"]), reverse=True)
    status = ("RESEARCH_CANDIDATE_AWAITING_DIRECT_FORWARD"
              if reveal and pass_gate(reveal, 200) else "REJECTED")
    report = {
        "status": status,
        "protocol": {
            "fit": "2020-04 through 2021; labels must exit before 2022",
            "selection": "2022", "validation": "2023 after selection gate",
            "confirmation": "2024 after validation gate",
            "test": "2025 after confirmation gate",
            "untouched_reveal": "2026 after test gate",
            "daily_availability": "session D becomes usable at 00:00 UTC on D+1",
            "families": list(FAMILIES), "profiles": list(PROFILES),
            "model_grid": list(CONFIGS), "quantiles": list(QUANTILES),
            "contracts": list(outcome_source.CONTRACTS),
            "money": "3% current-equity margin, fixed 10x leverage",
            "entry": "next hourly 5m perpetual market open", "fees": "excluded",
        },
        "rows": {"strict_hourly": len(hourly), "macro_enriched": len(enriched),
                 "events": len(events), "contracts": exclusions},
        "minimum_ideal_hit_rate_for_30pct_month": {
            f"tp{item['target_pct']}_sl{item['stop_pct']}": round(
                minimum_binary_hit_rate_pct(item["target_pct"], item["stop_pct"]), 6)
            for item in outcome_source.CONTRACTS
        },
        "qualified_full_routes_before_2024": len(full),
        "frozen_before_2024": public(frozen) if frozen else None,
        "confirmation_2024": confirmation, "test_2025": test,
        "untouched_2026": reveal,
        "selection_frontier": [public(item) for item in frontier[:30]],
        "sources": {**price_sources, **macro_sources,
                    str(Path(__file__)): digest(Path(__file__))},
        "limitations": [
            "Daily macro bars cannot describe intraday US-session transitions.",
            "Public adjusted histories may be revised by the provider.",
            "Any historical survivor would still require shadow forward evidence.",
        ],
    }
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({
        "status": status, "rows": report["rows"],
        "qualified_full_routes_before_2024": len(full),
        "frozen_before_2024": report["frozen_before_2024"],
        "confirmation_2024": confirmation, "test_2025": test,
        "untouched_2026": reveal,
        "selection_frontier": report["selection_frontier"][:8],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
