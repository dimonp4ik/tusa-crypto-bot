"""Bounded target-specific ranking inside the causal DVOL regimes.

The semantic regimes are fixed by crypto_deribit_dvol_regime_oct9_audit.py.
This follow-up checks whether a small target-specific ranker can distinguish
the material moves that the raw regime rules could not.  It fits on 2021,
selects a fixed threshold on 2022, and keeps every later year sealed unless
the preceding period passes the full economic gate.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

import crypto_deribit_dvol_regime_oct9_audit as dvol
import crypto_futures_trade_flow_regime_oct7_audit as flow_source
from crypto_runtime_shadow_portfolio_oct3_audit import money, runtime_gate
from src.research_profit_gate import economic_gate, minimum_binary_hit_rate_pct


ROOT = Path("reports/audit_2026_09_08")
OUT = ROOT / "crypto_deribit_dvol_material_model_oct9.json"
PERIODS = {
    "fit_2021": (pd.Timestamp("2021-05-01", tz="UTC"),
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
CONFIGS = (
    {"max_leaf_nodes": 15, "min_samples_leaf": 200, "l2": 5.0},
    {"max_leaf_nodes": 31, "min_samples_leaf": 350, "l2": 8.0},
)
QUANTILES = (.70, .75, .80, .85, .90, .925, .95, .97, .98, .99)
PRICE_DIRECTIONAL = (
    "ret3", "ret6", "ret12", "ret36", "ret72", "ret144", "ret288",
    "body1", "ema12_dist", "ema48_dist", "cross_z12", "cross_z72",
    "btc_ret12", "btc_ret72", "btc_ret288",
)
PRICE_ABSOLUTE = (
    "range1", "atr14_pct", "volume_ratio", "range_ratio", "eff72",
    "upper_wick", "lower_wick", "breadth12", "breadth72",
)
DVOL_FEATURES = (
    "btc_dvol", "btc_dvol_z720", "btc_dvol_range", "btc_dvol_ret1",
    "btc_dvol_ret6", "btc_dvol_ret24", "btc_dvol_ret72",
    "eth_dvol", "eth_dvol_z720", "eth_dvol_range", "eth_dvol_ret1",
    "eth_dvol_ret6", "eth_dvol_ret24", "eth_dvol_ret72",
    "dvol_spread", "dvol_spread_z720", "dvol_spread_delta6",
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def candidate_union(outcomes: pd.DataFrame) -> pd.DataFrame:
    broad = outcomes[outcomes.profile.eq("broad")].copy()
    keys = ["symbol", "entry_time", "direction"]
    flags = broad.assign(value=1).pivot_table(
        index=keys, columns="family", values="value", aggfunc="max",
        fill_value=0).reset_index()
    flags.columns.name = None
    flags = flags.rename(columns={name: f"family_{name}"
                                  for name in dvol.FAMILIES})
    base = broad.sort_values("semantic_score", ascending=False).drop_duplicates(keys)
    base = base.drop(columns=["family", "profile"])
    result = base.merge(flags, on=keys, how="left", validate="one_to_one")
    strict_keys = outcomes[outcomes.profile.eq("strict")][keys].drop_duplicates()
    strict_keys["strict_profile_match"] = 1
    result = result.merge(strict_keys, on=keys, how="left", validate="one_to_one")
    result["strict_profile_match"] = result.strict_profile_match.fillna(0)
    for family in dvol.FAMILIES:
        column = f"family_{family}"
        if column not in result:
            result[column] = 0
    return result


def matrix(frame: pd.DataFrame, *, include_dvol: bool) -> np.ndarray:
    value = pd.DataFrame(index=frame.index)
    direction = frame.direction.to_numpy(float)
    for name in PRICE_DIRECTIONAL:
        value[f"{name}_dir"] = frame[name] * direction
        value[f"{name}_abs"] = frame[name].abs()
    for name in PRICE_ABSOLUTE:
        if name.startswith("breadth"):
            value[f"{name}_dir"] = np.where(
                direction > 0, frame[name], 1 - frame[name])
        else:
            value[name] = frame[name]
    value["breakout_dir"] = np.where(
        direction > 0, frame.breakout_up, frame.breakout_down)
    value["opposite_breakout"] = np.where(
        direction > 0, frame.breakout_down, frame.breakout_up)
    stamp = pd.to_datetime(frame.entry_time, unit="s", utc=True)
    value["hour_sin"] = np.sin(2 * np.pi * stamp.dt.hour / 24)
    value["hour_cos"] = np.cos(2 * np.pi * stamp.dt.hour / 24)
    value["weekday_sin"] = np.sin(2 * np.pi * stamp.dt.dayofweek / 7)
    value["weekday_cos"] = np.cos(2 * np.pi * stamp.dt.dayofweek / 7)
    for family in dvol.FAMILIES:
        value[f"family_{family}"] = frame[f"family_{family}"].astype(float)
    value["strict_profile_match"] = frame.strict_profile_match.astype(float)
    for symbol in sorted(frame.symbol.unique()):
        value[f"symbol_{symbol}"] = (frame.symbol == symbol).astype(float)
    if include_dvol:
        for name in DVOL_FEATURES:
            value[name] = frame[name]
        value["spread_delta_directional"] = frame.dvol_spread_delta6 * direction
        value["btc_dvol_price_alignment6"] = frame.btc_dvol_ret6 * frame.ret72 * direction
    return value.replace([np.inf, -np.inf], np.nan).fillna(0).to_numpy(np.float32)


def mask(frame: pd.DataFrame, period: str) -> np.ndarray:
    start, end = PERIODS[period]
    stamp = pd.to_datetime(frame.entry_time, unit="s", utc=True)
    return ((stamp >= start) & (stamp < end)).to_numpy()


def evaluate(frame: pd.DataFrame, period: str, threshold: float) -> dict:
    selected = frame.loc[mask(frame, period) & frame.quality.ge(threshold)].copy()
    selected = selected.sort_values(
        ["entry_time", "quality", "semantic_score", "symbol"],
        ascending=[True, False, False, True])
    selected["source_priority"] = selected.groupby("entry_time").cumcount()
    rows = []
    for row in selected.to_dict("records"):
        families = [name for name in dvol.FAMILIES
                    if row.get(f"family_{name}", 0)]
        rows.append({
            **row,
            "direction": "LONG" if row["direction"] > 0 else "SHORT",
            "source": "dvol_ranked_" + (families[0] if families else "regime"),
            "module": "dvol_ranked_material_router", "module_priority": 0,
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
        if accepted else {
            "n": 0, "observed_trades_per_day": 0.0,
            "projected_monthly_return_pct": 0.0,
            "frequency_pass": False, "return_pass": False, "passes": False,
        }
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
    dvol_data, dvol_sources = dvol.load_dvol()
    hourly = dvol.strict_hourly_price(candles)
    enriched = hourly.merge(dvol.dvol_frame(dvol_data), on="entry_time",
                            how="inner", validate="many_to_one")
    events = dvol.build_events(enriched)
    frontier, survivors, exclusions = [], [], []

    for contract in dvol.CONTRACTS:
        outcomes, excluded = dvol.add_outcomes(events, candles, contract)
        candidates = candidate_union(outcomes)
        exclusions.append({**contract, "excluded_crossing_gap_or_end": excluded,
                           "candidate_rows": len(candidates)})
        fit, selection = mask(candidates, "fit_2021"), mask(candidates, "selection_2022")
        y = candidates.target_hit.astype(int).to_numpy()
        matrices = {
            "semantic_price_ranker": matrix(candidates, include_dvol=False),
            "semantic_price_dvol_ranker": matrix(candidates, include_dvol=True),
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
            if item["model_kind"] == "semantic_price_dvol_ranker"]
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
            "fit": "2021-05 through 2021-12",
            "selection": "2022",
            "validation": "2023 only after selection economic gate",
            "confirmation": "2024 only after validation gate",
            "test": "2025 only after confirmation gate",
            "untouched_reveal": "2026 only after test gate",
            "model_grid": list(CONFIGS), "threshold_quantiles": list(QUANTILES),
            "targets": list(dvol.CONTRACTS),
            "money": "3% current-equity margin, fixed 10x leverage",
            "entry": "market open after closed hourly DVOL and 5m price bars",
            "fees": "excluded by user instruction",
        },
        "rows": {"hourly": len(hourly), "enriched": len(enriched),
                 "events": len(events), "contracts": exclusions},
        "minimum_ideal_hit_rate_for_30pct_month": {
            f"tp{item['target_pct']}_sl{item['stop_pct']}": round(
                minimum_binary_hit_rate_pct(item["target_pct"], item["stop_pct"]), 6)
            for item in dvol.CONTRACTS
        },
        "qualified_full_routes_before_2024": len(full),
        "frozen_before_2024": public(frozen) if frozen else None,
        "confirmation_2024": confirmation, "test_2025": test,
        "untouched_2026": reveal,
        "selection_frontier": [public(item) for item in frontier[:30]],
        "sources": {**price_sources, **dvol_sources,
                    str(Path(dvol.__file__)): digest(Path(dvol.__file__)),
                    str(Path(__file__)): digest(Path(__file__))},
        "limitations": [
            "The bounded model grid still creates selection risk.",
            "DVOL public history cannot provide a 2020 training period.",
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
