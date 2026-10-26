"""Refresh the frozen profit-first 2R candidate on Oct 24 X-Perp data.

The model, threshold, 5% near fraction and 2R runner were all frozen by the
Oct 18 audit.  This script refits that deterministic frozen model on the same
historical fit rows, checks replay parity through Oct 16, and reveals only the
new completed venue-native rows after that cutoff.
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
from crypto_runtime_shadow_portfolio_oct3_audit import money


ROOT = Path("reports/audit_2026_09_08")
SOURCE_REPORT = ROOT / "crypto_profit_first_material_target_oct18.json"
OLD_ROWS = ROOT / "crypto_profit_first_material_target_xperp_oct17_fresh.pkl"
FAR = ROOT / "crypto_xperp_raw_router_audit_2_0_oct24_material_raw_raw_families.csv"
NEAR = ROOT / "crypto_xperp_raw_router_audit_0_5_oct24_material_raw_raw_families.csv"
OUT = ROOT / "crypto_profit_first_material_target_oct24_forward.json"
ROWS_OUT = ROOT / "crypto_profit_first_material_target_xperp_oct24.pkl"
NEW_ROWS_OUT = ROOT / "crypto_profit_first_material_target_xperp_oct24_new.pkl"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def identity(frame: pd.DataFrame) -> set[tuple]:
    return set(zip(frame.symbol.astype(str),
                   pd.to_numeric(frame.entry_time).astype("int64"),
                   frame.direction.astype(str), frame.family.astype(str)))


def add_projection(metrics: dict, rows: pd.DataFrame) -> dict:
    result = dict(metrics)
    if rows.empty:
        result["observed_calendar_days"] = 0
        result["projected_monthly_return_pct"] = 0.0
        return result
    stamps = pd.to_datetime(rows.entry_time, unit="s", utc=True)
    days = max(1, (stamps.max().date() - stamps.min().date()).days + 1)
    growth = max(0.0, 1 + float(result["return_pct"]) / 100)
    result["observed_calendar_days"] = days
    result["projected_monthly_return_pct"] = round(
        100 * (growth ** (30 / days) - 1), 4)
    return result


def main() -> None:
    source = json.loads(SOURCE_REPORT.read_text(encoding="utf-8"))
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
    cutoff = int(pd.Timestamp(
        source["fresh_direct_coverage"]["last_entry"]).timestamp())
    # The refreshed cache starts earlier, so its rolling indicators have a
    # longer warm-up.  Exclude the first two common days from parity; all three
    # changed identities are inside that warm-up edge.  The remainder must be
    # bit-for-bit identical through the old cutoff.
    parity_start = int((pd.Timestamp(
        source["fresh_direct_coverage"]["first_entry"])
        + pd.Timedelta(days=2)).timestamp())
    prefix = refresh_rows[
        refresh_rows.entry_time.between(parity_start, cutoff)].copy()
    old_parity_rows = old_rows[
        old_rows.entry_time.between(parity_start, cutoff)].copy()
    new_rows = refresh_rows[refresh_rows.entry_time.gt(cutoff)].copy()
    new_rows.to_pickle(NEW_ROWS_OUT)

    old_ids = identity(old_parity_rows)
    prefix_ids = identity(prefix)
    common = old_ids & prefix_ids
    old_return = {
        key: value for key, value in zip(
            zip(old_parity_rows.symbol.astype(str),
                old_parity_rows.entry_time.astype("int64"),
                old_parity_rows.direction.astype(str),
                old_parity_rows.family.astype(str)),
            old_parity_rows.signed_price_return.astype(float))}
    prefix_return = {
        key: value for key, value in zip(
            zip(prefix.symbol.astype(str), prefix.entry_time.astype("int64"),
                prefix.direction.astype(str), prefix.family.astype(str)),
            prefix.signed_price_return.astype(float))}
    max_return_delta = max(
        (abs(old_return[key] - prefix_return[key]) for key in common),
        default=0.0)
    parity = {
        "parity_start_utc": pd.Timestamp(
            parity_start, unit="s", tz="UTC").isoformat(),
        "old_rows": len(old_parity_rows),
        "old_warmup_rows_excluded": len(old_rows) - len(old_parity_rows),
        "refresh_prefix_rows": len(prefix),
        "missing_from_refresh": len(old_ids - prefix_ids),
        "unexpected_in_refresh_prefix": len(prefix_ids - old_ids),
        "maximum_signed_return_delta": float(max_return_delta),
        "passes": bool(old_ids == prefix_ids
                       and max_return_delta <= np.finfo(float).eps * 8),
    }
    if not parity["passes"]:
        raise ValueError(f"frozen replay parity failed: {parity}")

    new_results = {
        "margin_3pct": add_projection(
            money(new_rows.to_dict("records"), .03), new_rows),
        "margin_5pct": add_projection(
            money(new_rows.to_dict("records"), .05), new_rows),
    }
    october_start = int(pd.Timestamp("2026-10-01", tz="UTC").timestamp())
    october_rows = refresh_rows[refresh_rows.entry_time.ge(october_start)].copy()
    current_october = {
        "margin_3pct": add_projection(
            money(october_rows.to_dict("records"), .03), october_rows),
        "margin_5pct": add_projection(
            money(october_rows.to_dict("records"), .05), october_rows),
    }
    three = new_results["margin_3pct"]
    forward_gate = bool(
        three["n"] >= 20
        and three["return_pct"] > 0
        and three["dd_pct"] <= 10
        and three["trades_per_calendar_day"] >= 3
        and three["projected_monthly_return_pct"] >= 30)
    report = {
        "status": ("FORWARD_GATE_PASSED_AWAIT_MORE_TIME"
                   if forward_gate else "FORWARD_GATE_FAILED"),
        "ready_for_live_orders": False,
        "policy": {
            "runner_target_r": 2.0,
            "near_fraction": fraction,
            "threshold": threshold,
            "entry": "next 15-minute market open",
            "orders": "market only",
            "money": "client-selected 3% or 5% current-equity margin; fixed 10x",
        },
        "coverage": coverage,
        "prior_cutoff_utc": pd.Timestamp(cutoff, unit="s", tz="UTC").isoformat(),
        "replay_parity": parity,
        "refreshed_all_rows_margin_3pct": refresh_metrics,
        "current_october": current_october,
        "new_unseen_forward": new_results,
        "forward_gate_pass": forward_gate,
        "activation_gate": {
            "required_new_completed_trades": 60,
            "observed_new_completed_trades": len(new_rows),
            "required_calendar_days": 14,
            "observed_calendar_days": three["active_days"],
            "met": bool(forward_gate and len(new_rows) >= 60
                        and three["active_days"] >= 14),
        },
        "sources": {
            str(SOURCE_REPORT): digest(SOURCE_REPORT),
            str(OLD_ROWS): digest(OLD_ROWS),
            str(FAR): digest(FAR),
            str(NEAR): digest(NEAR),
            str(Path(__file__).resolve()): digest(Path(__file__).resolve()),
        },
        "limitations": [
            "The newly revealed interval is less than two weeks.",
            "The venue-native history begins in April 2026.",
            "Public candles cannot reproduce order-book slippage or latency.",
            "Gross results exclude fees at user request.",
        ],
    }
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True),
                   encoding="utf-8")
    print(json.dumps({
        "status": report["status"],
        "replay_parity": parity,
        "new_unseen_forward": new_results,
        "current_october": current_october,
        "activation_gate": report["activation_gate"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
