"""Append-only honest Profit-First refresh through Oct-30 20:30 UTC.

Unlike the retired independent-target report, this replay terminates a trade
at the first live event: original stop, post-TP1 breakeven, or 2R target.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

import crypto_05_2r_precision_barbell_oct5_audit as legacy
import crypto_high_target_tree_oct5_audit as base
import crypto_profit_first_material_barbell_oct18_audit as joiner
import crypto_profit_first_material_target_oct18_audit as frozen
import crypto_profit_first_material_target_oct30_1000_forward as old_forward
import crypto_profit_first_runner_protection_oct30_audit as exact
import crypto_profit_first_sequential_runner_oct30_audit as sequential
from crypto_profit_first_material_target_oct24_forward import add_projection
from crypto_runtime_shadow_portfolio_oct3_audit import money, runtime_gate


ROOT = Path("reports/audit_2026_09_08")
SOURCE_REPORT = ROOT / "crypto_profit_first_material_target_oct18.json"
PREVIOUS_EXACT_REPORT = ROOT / "crypto_profit_first_runner_protection_oct30.json"
OLD_MANIFEST = ROOT / "xperp_15m_oct30_1000_material" / "manifest.json"
NEW_MANIFEST = ROOT / "xperp_15m_oct30_2030_material" / "manifest.json"
OLD_FAR = old_forward.FAR
OLD_NEAR = old_forward.NEAR
NEW_FAR = ROOT / "crypto_xperp_raw_router_audit_2_0_oct30_2030_material_raw_raw_families.csv"
NEW_NEAR = ROOT / "crypto_xperp_raw_router_audit_0_5_oct30_2030_material_raw_raw_families.csv"
OUT = ROOT / "crypto_profit_first_exact_oct30_2030_forward.json"
ROWS_OUT = ROOT / "crypto_profit_first_exact_oct30_2030.pkl"
NEW_ROWS_OUT = ROOT / "crypto_profit_first_exact_oct30_2030_new.pkl"
FRACTION = .05


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def paths(folder: Path) -> dict[tuple[str, str], Path]:
    return {("direct", path.name.split("_15min_")[0]): path
            for path in folder.glob("*_15min_current.pkl")}


def candle_overlap() -> dict:
    old = json.loads(OLD_MANIFEST.read_text(encoding="utf-8"))
    new = json.loads(NEW_MANIFEST.read_text(encoding="utf-8"))
    per_symbol = {}
    total = mismatches = 0
    for symbol in sorted(old["symbols"]):
        left = pickle.loads(Path(old["symbols"][symbol]["path"]).read_bytes())
        right = pickle.loads(Path(new["symbols"][symbol]["path"]).read_bytes())
        left_rows = {int(t): tuple(float(left[key][i]) for key in
                     ("open", "high", "low", "close", "volume"))
                     for i, t in enumerate(left["time"])}
        right_rows = {int(t): tuple(float(right[key][i]) for key in
                      ("open", "high", "low", "close", "volume"))
                      for i, t in enumerate(right["time"])}
        common = left_rows.keys() & right_rows.keys()
        changed = sum(left_rows[t] != right_rows[t] for t in common)
        total += len(common)
        mismatches += changed
        per_symbol[symbol] = {
            "overlap_bars": len(common), "value_mismatches": changed,
            "new_bars_after_old_end": sum(t > max(left_rows) for t in right_rows),
        }
    return {
        "symbols": len(per_symbol), "overlap_bars": total,
        "value_mismatches": mismatches, "per_symbol": per_symbol,
        "passes": mismatches == 0 and len(per_symbol) == len(new["symbols"]),
    }


def fit_model():
    historical = frozen.load_joined(2.0)
    fit = base.stage(historical, frozen.FIT)
    schema = base.design(fit)
    model = HistGradientBoostingClassifier(
        **frozen.MODEL, max_iter=150, l2_regularization=1.0,
        early_stopping=False, random_state=20260908)
    model.fit(base.matrix(fit, schema), fit.outcome.eq("TP2").astype(int))
    return model, schema


def replay(far: Path, near: Path, cache: Path, model, schema: dict,
           threshold: float) -> pd.DataFrame:
    joined = joiner.join_direct_outcomes(far, near)
    eligible = joined[
        joined.symbol.isin(base.ROBUST_SYMBOLS)
        & joined.risk_pct.between(legacy.RISK_PCT_MIN, legacy.RISK_PCT_MAX,
                                  inclusive="both")].copy()
    scored = legacy.scored(eligible, model, schema)
    selected = scored[scored.model_probability.ge(threshold)].copy()
    selected["period"] = "direct"
    literal = sequential.attach_current(selected, paths(cache))
    literal = sequential.apply_fraction(literal, FRACTION)
    rows = runtime_gate(base.standardized(literal, "profit_first_exact_2r"))
    return pd.DataFrame(rows).sort_values(
        ["entry_time", "symbol", "direction"]).reset_index(drop=True)


def identity(frame: pd.DataFrame) -> set[tuple]:
    return set(zip(frame.symbol.astype(str), frame.entry_time.astype("int64"),
                   frame.direction.astype(str), frame.family.astype(str)))


def parity(old: pd.DataFrame, new: pd.DataFrame) -> dict:
    cutoff = int(old.entry_time.max())
    prefix = new[new.entry_time.le(cutoff)].copy()
    keys = ["symbol", "entry_time", "direction", "family"]
    left = old.set_index(keys)
    right = prefix.set_index(keys)
    common = left.index.intersection(right.index)
    net_delta = max((abs(float(left.loc[key, "net_r"])
                         - float(right.loc[key, "net_r"]))
                     for key in common), default=0.0)
    exit_delta = max((abs(float(left.loc[key, "exit_time"])
                          - float(right.loc[key, "exit_time"]))
                      for key in common), default=0.0)
    passed = bool(identity(old) == identity(prefix)
                  and net_delta <= np.finfo(float).eps * 8 and exit_delta == 0)
    if not passed:
        raise ValueError({"old": len(old), "prefix": len(prefix),
                          "matched": len(common), "net_delta": net_delta,
                          "exit_delta": exit_delta})
    return {
        "through_entry_utc": pd.Timestamp(cutoff, unit="s", tz="UTC").isoformat(),
        "old_rows": len(old), "new_prefix_rows": len(prefix),
        "matched_rows": len(common), "maximum_net_r_delta": net_delta,
        "maximum_exit_time_delta_seconds": exit_delta, "passes": passed,
    }


def metrics(frame: pd.DataFrame, margin: float) -> dict:
    value = add_projection(money(frame.to_dict("records"), margin), frame)
    labels = frame.runner_outcome.astype(str) if len(frame) else pd.Series(dtype=str)
    counts = labels.value_counts().sort_index().to_dict()
    value.update({
        "outcome_counts": counts,
        "literal_2r_before_be_rate_pct": round(
            100 * float(labels.eq("TP_RUNNER").mean()), 4) if len(labels) else 0.0,
        "tp1_or_better_rate_pct": round(
            100 * float((~labels.isin(["SL", "EXPIRED"])).mean()), 4)
            if len(labels) else 0.0,
        "mean_signed_price_return_pct": round(
            100 * float(frame.signed_price_return.mean()), 6) if len(frame) else 0.0,
        "largest_symbol_share_pct": round(
            100 * float(frame.symbol.value_counts().iloc[0]) / len(frame), 4)
            if len(frame) else 0.0,
    })
    projected = float(value["projected_monthly_return_pct"])
    frequency = float(value["trades_per_calendar_day"])
    value["economic_gate"] = {
        "margin_fraction": margin, "leverage": 10.0,
        "required_trades_per_calendar_day": 3.0,
        "required_monthly_return_pct": 30.0,
        "frequency_pass": frequency >= 3,
        "return_pass": projected >= 30,
        "no_negative_month_pass": value["negative_months"] == 0,
        "passes": bool(frequency >= 3 and projected >= 30
                       and value["negative_months"] == 0),
    }
    return value


def slices(frame: pd.DataFrame, original_cutoff: int) -> dict:
    october = frame[frame.entry_time.ge(pd.Timestamp(
        "2026-10-01", tz="UTC").timestamp())]
    unseen = frame[frame.entry_time.gt(original_cutoff)]
    return {
        "october": {f"margin_{int(m*100)}pct": metrics(october, m)
                    for m in (.03, .05)},
        "unseen_after_oct16": {f"margin_{int(m*100)}pct": metrics(unseen, m)
                               for m in (.03, .05)},
    }


def main() -> None:
    source = json.loads(SOURCE_REPORT.read_text(encoding="utf-8"))
    previous = json.loads(PREVIOUS_EXACT_REPORT.read_text(encoding="utf-8"))
    threshold = float(source["frozen_before_2024"]["threshold"])
    overlap = candle_overlap()
    if not overlap["passes"]:
        raise ValueError(f"candle overlap failed: {overlap}")
    model, schema = fit_model()
    old = replay(OLD_FAR, OLD_NEAR, OLD_MANIFEST.parent, model, schema, threshold)
    new = replay(NEW_FAR, NEW_NEAR, NEW_MANIFEST.parent, model, schema, threshold)
    prefix = parity(old, new)
    old_ids = identity(old)
    incremental = new[[key not in old_ids for key in zip(
        new.symbol.astype(str), new.entry_time.astype("int64"),
        new.direction.astype(str), new.family.astype(str))]].copy()
    new.to_pickle(ROWS_OUT)
    incremental.to_pickle(NEW_ROWS_OUT)
    original_cutoff = int(pd.Timestamp(
        source["fresh_direct_coverage"]["last_entry"]).timestamp())
    result = slices(new, original_cutoff)
    incremental_metrics = {f"margin_{int(m*100)}pct": metrics(incremental, m)
                           for m in (.03, .05)}
    report = {
        "status": "EXACT_FORWARD_GATE_PASSED" if (
            result["unseen_after_oct16"]["margin_3pct"]["economic_gate"]["passes"]
        ) else "EXACT_FORWARD_GATE_NOT_PASSED",
        "ready_for_live_orders": False, "orders_possible": False,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "policy": {
            "first_bank": "5% at 0.5R", "runner": "95% to 2R",
            "breakeven": "active from the bar after TP1", "stop_r": 1.0,
            "entry": "next 15-minute market open", "orders": "market only",
            "position_fraction": "client setting", "leverage": 10,
        },
        "integrity": {
            "raw_candle_overlap": overlap, "exact_accepted_prefix": prefix,
            "previous_exact_status": previous["status"],
        },
        "new_accepted_rows": len(incremental),
        "newly_accepted_after_morning_snapshot": incremental_metrics,
        "current": result,
        "sources": {
            str(SOURCE_REPORT): digest(SOURCE_REPORT),
            str(PREVIOUS_EXACT_REPORT): digest(PREVIOUS_EXACT_REPORT),
            str(OLD_MANIFEST): digest(OLD_MANIFEST),
            str(NEW_MANIFEST): digest(NEW_MANIFEST),
            str(OLD_FAR): digest(OLD_FAR), str(OLD_NEAR): digest(OLD_NEAR),
            str(NEW_FAR): digest(NEW_FAR), str(NEW_NEAR): digest(NEW_NEAR),
            str(Path(__file__).resolve()): digest(Path(__file__).resolve()),
        },
        "limitations": [
            "The incremental interval is less than one complete 48-bar horizon.",
            "Public OHLC cannot reveal intrabar path, so ambiguous bars are stop-first.",
            "Gross results exclude fees at user request.",
        ],
    }
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({
        "status": report["status"], "integrity": report["integrity"],
        "new_accepted_rows": len(incremental),
        "incremental": incremental_metrics, "current": result,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
