"""Append-only Oct-27 forward check for the frozen Profit-First 2R route.

The model, probability threshold, partial close, exits and portfolio rails were
all frozen before this snapshot.  The script verifies the accepted Oct-26
prefix exactly, then reports both the newly revealed Oct-26-to-Oct-27 slice
and the cumulative unseen interval after the original Oct-16 cutoff.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

import crypto_high_target_tree_oct5_audit as model_base
import crypto_profit_first_material_barbell_oct18_audit as joiner
import crypto_profit_first_material_target_oct18_audit as frozen
from crypto_profit_first_material_target_oct24_forward import add_projection
from crypto_runtime_shadow_portfolio_oct3_audit import money


ROOT = Path("reports/audit_2026_09_08")
SOURCE_REPORT = ROOT / "crypto_profit_first_material_target_oct18.json"
PREVIOUS_REPORT = ROOT / "crypto_profit_first_material_target_oct26_forward.json"
OLD_ROWS = ROOT / "crypto_profit_first_material_target_xperp_oct26.pkl"
FAR = ROOT / "crypto_xperp_raw_router_audit_2_0_oct27_material_raw_raw_families.csv"
NEAR = ROOT / "crypto_xperp_raw_router_audit_0_5_oct27_material_raw_raw_families.csv"
MANIFEST = ROOT / "xperp_15m_oct27_material" / "manifest.json"
OUT = ROOT / "crypto_profit_first_material_target_oct27_forward.json"
ROWS_OUT = ROOT / "crypto_profit_first_material_target_xperp_oct27.pkl"
NEW_ROWS_OUT = ROOT / "crypto_profit_first_material_target_xperp_oct27_new.pkl"
CUMULATIVE_ROWS_OUT = ROOT / "crypto_profit_first_material_target_xperp_oct27_cumulative_unseen.pkl"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def identity(frame: pd.DataFrame) -> set[tuple]:
    return set(zip(frame.symbol.astype(str),
                   pd.to_numeric(frame.entry_time).astype("int64"),
                   frame.direction.astype(str), frame.family.astype(str)))


def metrics(frame: pd.DataFrame, margin: float) -> dict:
    value = add_projection(money(frame.to_dict("records"), margin), frame)
    if frame.empty:
        value.update({
            "literal_2r_runner_hit_rate_pct": 0.0,
            "near_target_hit_rate_pct": 0.0,
            "mean_signed_price_return_pct": 0.0,
            "largest_symbol_share_pct": 0.0,
        })
    else:
        counts = frame.symbol.astype(str).value_counts()
        value.update({
            "literal_2r_runner_hit_rate_pct": round(
                100*float(frame.barbell_outcome.eq("TP_RUNNER").mean()), 4),
            "near_target_hit_rate_pct": round(
                100*float(frame.near_outcome.eq("TP2").mean()), 4),
            "mean_signed_price_return_pct": round(
                100*float(frame.signed_price_return.mean()), 6),
            "largest_symbol_share_pct": round(
                100*float(counts.iloc[0])/len(frame), 4),
        })
    projected = float(value["projected_monthly_return_pct"])
    frequency = float(value["trades_per_calendar_day"])
    value["economic_gate"] = {
        "margin_fraction": margin, "leverage": 10.0,
        "required_trades_per_calendar_day": 3.0,
        "required_constant_signed_price_return_pct": .973137,
        "frequency_pass": frequency >= 3,
        "return_pass": projected >= 30,
        "passes": bool(frequency >= 3 and projected >= 30
                       and value["negative_months"] == 0),
    }
    return value


def main() -> None:
    source = json.loads(SOURCE_REPORT.read_text(encoding="utf-8"))
    previous = json.loads(PREVIOUS_REPORT.read_text(encoding="utf-8"))
    policy = source["frozen_before_2024"]
    if float(policy["runner_target_r"]) != 2.0:
        raise ValueError("frozen source is no longer the 2R policy")
    threshold = float(policy["threshold"])
    fraction = float(policy["near_fraction"])

    historical = frozen.load_joined(2.0)
    fit = model_base.stage(historical, frozen.FIT)
    schema = model_base.design(fit)
    model = HistGradientBoostingClassifier(
        **frozen.MODEL, max_iter=150, l2_regularization=1.0,
        early_stopping=False, random_state=20260908)
    model.fit(model_base.matrix(fit, schema), fit.outcome.eq("TP2").astype(int))

    direct = joiner.join_direct_outcomes(FAR, NEAR)
    refresh_metrics, coverage = frozen.score_direct(
        direct, model, schema, threshold, fraction, 2.0, ROWS_OUT)
    refresh_rows = pd.read_pickle(ROWS_OUT).sort_values("entry_time")
    old_rows = pd.read_pickle(OLD_ROWS).sort_values("entry_time")
    previous_cutoff = int(old_rows.entry_time.max())
    prefix = refresh_rows[refresh_rows.entry_time.le(previous_cutoff)].copy()

    old_ids = identity(old_rows)
    prefix_ids = identity(prefix)
    common = old_ids & prefix_ids
    keys = ["symbol", "entry_time", "direction", "family"]
    old_returns = old_rows.set_index(keys).signed_price_return.astype(float)
    prefix_returns = prefix.set_index(keys).signed_price_return.astype(float)
    max_return_delta = max(
        (abs(float(old_returns.loc[key]) - float(prefix_returns.loc[key]))
         for key in common), default=0.0)
    parity = {
        "through_entry_utc": pd.Timestamp(
            previous_cutoff, unit="s", tz="UTC").isoformat(),
        "old_rows": len(old_rows), "refresh_prefix_rows": len(prefix),
        "matched_rows": len(common),
        "missing_from_refresh": len(old_ids - prefix_ids),
        "unexpected_in_refresh_prefix": len(prefix_ids - old_ids),
        "maximum_signed_return_delta": float(max_return_delta),
        "passes": bool(old_ids == prefix_ids
                       and max_return_delta <= np.finfo(float).eps * 8),
    }
    if not parity["passes"]:
        raise ValueError(f"frozen replay parity failed: {parity}")

    incremental = refresh_rows[
        refresh_rows.entry_time.gt(previous_cutoff)].copy()
    incremental.to_pickle(NEW_ROWS_OUT)
    original_cutoff = int(pd.Timestamp(
        source["fresh_direct_coverage"]["last_entry"]).timestamp())
    cumulative = refresh_rows[
        refresh_rows.entry_time.gt(original_cutoff)].copy()
    cumulative.to_pickle(CUMULATIVE_ROWS_OUT)

    october_start = int(pd.Timestamp("2026-10-01", tz="UTC").timestamp())
    october = refresh_rows[refresh_rows.entry_time.ge(october_start)].copy()
    current_october = {f"margin_{int(m * 100)}pct": metrics(october, m)
                       for m in (.03, .05)}
    newly_revealed = {f"margin_{int(m * 100)}pct": metrics(incremental, m)
                      for m in (.03, .05)}
    cumulative_unseen = {f"margin_{int(m * 100)}pct": metrics(cumulative, m)
                         for m in (.03, .05)}

    three = cumulative_unseen["margin_3pct"]
    forward_gate = bool(
        three["n"] >= 20 and three["return_pct"] > 0
        and three["dd_pct"] <= 10
        and three["trades_per_calendar_day"] >= 3
        and three["projected_monthly_return_pct"] >= 30)
    enough_time = int(three["observed_calendar_days"]) >= 14
    enough_trades = int(three["n"]) >= 60
    activation_met = bool(forward_gate and enough_time and enough_trades)
    report = {
        "status": ("ACTIVATION_GATE_PASSED"
                   if activation_met else
                   "FORWARD_GATE_PASSED_AWAIT_MORE_TIME"
                   if forward_gate else "FORWARD_GATE_FAILED"),
        "ready_for_live_orders": False,
        "policy": {
            "runner_target_r": 2.0, "near_fraction": fraction,
            "threshold": threshold,
            "entry": "next 15-minute market open", "orders": "market only",
            "money": "client-selected 3% or 5% current-equity margin; fixed 10x",
            "retuned_on_october": False,
        },
        "coverage": coverage,
        "integrity": {
            "previous_snapshot_prefix": parity,
            "previous_report_october_margin_3pct":
                previous["current_october"]["margin_3pct"],
        },
        "current_october": current_october,
        "newly_revealed_after_oct26_snapshot": newly_revealed,
        "cumulative_unseen_after_original_oct16_cutoff": cumulative_unseen,
        "frozen_full_refresh_margin_3pct": refresh_metrics,
        "forward_gate_pass": forward_gate,
        "activation_gate": {
            "required_new_completed_trades": 60,
            "observed_new_completed_trades": len(cumulative),
            "required_calendar_days": 14,
            "observed_calendar_days": int(three["observed_calendar_days"]),
            "met": activation_met,
        },
        "sources": {
            str(SOURCE_REPORT): digest(SOURCE_REPORT),
            str(PREVIOUS_REPORT): digest(PREVIOUS_REPORT),
            str(OLD_ROWS): digest(OLD_ROWS), str(FAR): digest(FAR),
            str(NEAR): digest(NEAR), str(MANIFEST): digest(MANIFEST),
            str(Path(__file__).resolve()): digest(Path(__file__).resolve()),
        },
        "limitations": [
            "The cumulative unseen interval is still shorter than two weeks.",
            "The venue-native history begins in April 2026.",
            "Public candles cannot reproduce order-book slippage or latency.",
            "Gross results exclude fees at user request.",
        ],
    }
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True),
                   encoding="utf-8")
    print(json.dumps({
        "status": report["status"], "integrity": report["integrity"],
        "newly_revealed_after_oct26_snapshot": newly_revealed,
        "cumulative_unseen_after_original_oct16_cutoff": cumulative_unseen,
        "current_october": current_october,
        "activation_gate": report["activation_gate"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
