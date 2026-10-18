"""Native X-Perp geometry selection with August-October holdout.

Models fit partial April.  May selects threshold and rails, while June and
July must confirm the same policy.  Only then are August, September and the
partial October window revealed.  All entries and exits remain market-only.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

import crypto_raw_geometry_walkforward_oct20_audit as geometry_audit
import crypto_raw_hourly_direction_oct19_audit as parent


OUT = parent.ROOT / "crypto_xperp_native_geometry_oct20.json"
ROWS_OUT = parent.ROOT / "crypto_xperp_native_geometry_oct20.pkl"
FIT_MONTH = "2026-04"
SELECT_MONTH = "2026-05"
CONFIRM_MONTHS = ("2026-06", "2026-07")
HOLDOUT_MONTHS = ("2026-08", "2026-09", "2026-10")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def month(frame: pd.DataFrame, value: str) -> pd.DataFrame:
    return frame[frame.month.eq(value)].copy()


def period_ok(value: dict) -> bool:
    return bool(value["n"] >= 80 and value["return_pct"] >= 0
                and value["dd_pct"] <= 10
                and value["trades_per_calendar_day"] >= 3
                and value["negative_months"] == 0)


def rank_key(item: dict) -> tuple:
    values = [item["selection"], *item["confirmation"].values()]
    return (
        bool(item["admissible"]),
        min(float(value["return_pct"]) for value in values),
        -max(float(value["dd_pct"]) for value in values),
        sum(float(value["return_pct"]) for value in values),
    )


def main() -> None:
    direct_parts = []
    for symbol in parent.CORE_SYMBOLS:
        path = parent.rails.DIRECT_CACHE / f"{symbol}_15min_current.pkl"
        direct_parts.append(parent.build_symbol_period(
            "xperp", symbol, parent.load_candles(path)))
    base_direct = pd.concat(direct_parts, ignore_index=True)
    base_direct["month"] = pd.to_datetime(
        base_direct.entry_time, unit="s", utc=True).dt.strftime("%Y-%m")

    candidates = []
    for geometry in geometry_audit.GEOMETRIES:
        labeled = geometry_audit.relabel(
            base_direct, geometry, direct=True)
        fit = month(labeled, FIT_MONTH)
        schema = parent.design(fit)
        for target_name in geometry_audit.MODEL_CONFIG:
            model = geometry_audit.fit_model(fit, schema, target_name)
            scored = geometry_audit.score(
                labeled, model, schema, target_name)
            selection_frame = month(scored, SELECT_MONTH)
            for quantile in parent.QUANTILES:
                threshold = float(
                    selection_frame.model_quality.quantile(quantile))
                for loss_pause in parent.LOSS_PAUSES:
                    for hourly_cap in parent.HOURLY_CAPS:
                        selection, _ = parent.evaluate(
                            selection_frame, threshold, loss_pause,
                            hourly_cap, .03)
                        confirmation = {
                            value: parent.evaluate(
                                month(scored, value), threshold, loss_pause,
                                hourly_cap, .03)[0]
                            for value in CONFIRM_MONTHS
                        }
                        values = [selection, *confirmation.values()]
                        candidates.append({
                            "target_return_pct": 100 * geometry[0],
                            "stop_return_pct": 100 * geometry[1],
                            "hold_bars": geometry[2],
                            "model_target": target_name,
                            "model_config": list(
                                geometry_audit.MODEL_CONFIG[target_name]),
                            "selection_quantile": quantile,
                            "threshold": threshold,
                            "loss_pause": loss_pause,
                            "hourly_cap": hourly_cap,
                            "selection": selection,
                            "confirmation": confirmation,
                            "admissible": all(period_ok(item)
                                              for item in values),
                        })
    candidates.sort(key=rank_key, reverse=True)
    selected = candidates[0]
    geometry = geometry_audit.geometry_tuple(selected)
    labeled = geometry_audit.relabel(base_direct, geometry, direct=True)
    schema = parent.design(month(labeled, FIT_MONTH))
    model = geometry_audit.fit_model(
        month(labeled, FIT_MONTH), schema, str(selected["model_target"]))
    scored = geometry_audit.score(
        labeled, model, schema, str(selected["model_target"]))
    threshold = float(selected["threshold"])
    loss_pause = int(selected["loss_pause"])
    hourly_cap = int(selected["hourly_cap"])
    holdouts = {}
    output_rows = []
    for margin in (.03, .05):
        name = f"margin_{round(100 * margin)}pct"
        holdouts[name] = {}
        for value in HOLDOUT_MONTHS:
            metrics, rows = parent.evaluate(
                month(scored, value), threshold,
                loss_pause, hourly_cap, margin)
            holdouts[name][value] = metrics
            if margin == .03:
                output_rows.extend(rows)
    pd.DataFrame(output_rows).to_pickle(ROWS_OUT)

    five = holdouts["margin_5pct"]
    complete = ("2026-08", "2026-09")
    objective = bool(
        selected["admissible"]
        and all(five[value]["return_pct"] >= 30 for value in complete)
        and all(five[value]["negative_months"] == 0
                and five[value]["dd_pct"] <= 10
                and five[value]["trades_per_calendar_day"] >= 3
                for value in complete)
        and five["2026-10"]["return_pct"] > 0)
    report = {
        "status": ("XPERP_NATIVE_GEOMETRY_SHADOW_CANDIDATE"
                   if objective else "REJECTED_OBJECTIVE"),
        "ready_for_live_orders": False,
        "protocol": {
            "geometries": [{"target_pct": 100 * item[0],
                             "stop_pct": 100 * item[1],
                             "hold_bars": item[2]}
                            for item in geometry_audit.GEOMETRIES],
            "fit": FIT_MONTH, "selection": SELECT_MONTH,
            "confirmation": list(CONFIRM_MONTHS),
            "holdouts": list(HOLDOUT_MONTHS),
            "entry": "market at next 15-minute open; hourly decisions",
            "orders": "market only",
            "money": "client 3% or 5% current-equity margin; fixed 10x",
            "priority": "worst development month; no win-rate gate",
        },
        "candidate_count": len(candidates),
        "admissible_count": sum(item["admissible"] for item in candidates),
        "selected_before_holdouts": selected,
        "top20_before_holdouts": candidates[:20],
        "holdouts": holdouts,
        "objective_pass": objective,
        "sources": {
            **geometry_audit.SOURCE_HASHES,
            str(Path(__file__).resolve()): digest(Path(__file__).resolve()),
        },
        "limitations": [
            "April is a partial fit month and native history is short.",
            "October is incomplete and remains diagnostic.",
            "Only five symbols have continuous long-history parity.",
            "Gross returns exclude fees at user request.",
            "No backtest guarantees a future monthly return.",
        ],
    }
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({
        "status": report["status"],
        "candidate_count": len(candidates),
        "admissible_count": report["admissible_count"],
        "selected": selected, "holdouts": holdouts,
        "objective_pass": objective,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
