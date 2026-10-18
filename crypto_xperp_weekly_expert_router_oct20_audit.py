"""Weekly causal router between frequent and precision X-Perp experts.

Each weekly test uses models fit only on trades closed before a trailing
21-day calibration window.  That window selects expert, threshold and rails.
If neither expert is profitable in calibration, the next week stays flat.
"""
from __future__ import annotations

import collections
import hashlib
import json
from pathlib import Path

import pandas as pd

import crypto_raw_geometry_walkforward_oct20_audit as geometry_audit
import crypto_raw_hourly_direction_oct19_audit as parent
import crypto_xperp_monthly_adaptive_oct20_audit as monthly
from crypto_runtime_shadow_portfolio_oct3_audit import money


OUT = parent.ROOT / "crypto_xperp_weekly_expert_router_oct20.json"
ROWS_OUT = parent.ROOT / "crypto_xperp_weekly_expert_router_oct20.pkl"
EXPERTS = (
    {"name": "frequent", "geometry": (.01, .0075, 48),
     "model_target": "signed_return"},
    {"name": "precision", "geometry": (.01, .005, 48),
     "model_target": "profitable_exit"},
)
CALIBRATION_DAYS = 21
TEST_DAYS = 7
FIRST_TEST = pd.Timestamp("2026-06-01", tz="UTC")
END_TEST = pd.Timestamp("2026-10-17", tz="UTC")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def causal_parts(frame: pd.DataFrame, test_start: pd.Timestamp,
                 test_end: pd.Timestamp) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    start = test_start.timestamp()
    end = test_end.timestamp()
    calibration_start = (
        test_start - pd.Timedelta(days=CALIBRATION_DAYS)).timestamp()
    train = frame[frame.exit_time < calibration_start].copy()
    calibration = frame[
        frame.entry_time.ge(calibration_start)
        & frame.entry_time.lt(start)
        & frame.exit_time.lt(start)
    ].copy()
    test = frame[frame.entry_time.ge(start) & frame.entry_time.lt(end)].copy()
    return train, calibration, test


def expert_key(item: dict) -> tuple:
    value = item["policy"]["metrics"]
    return (
        value["trades_per_calendar_day"] >= 3,
        value["dd_pct"] <= 5,
        value["negative_months"] == 0,
        value["return_pct"],
        -value["dd_pct"],
    )


def should_trade(item: dict) -> bool:
    value = item["policy"]["metrics"]
    return bool(value["return_pct"] > 0
                and value["negative_months"] == 0
                and value["dd_pct"] <= 5
                and value["trades_per_calendar_day"] >= 3)


def main() -> None:
    direct_parts = []
    for symbol in parent.CORE_SYMBOLS:
        path = parent.rails.DIRECT_CACHE / f"{symbol}_15min_current.pkl"
        direct_parts.append(parent.build_symbol_period(
            "xperp", symbol, parent.load_candles(path)))
    base_direct = pd.concat(direct_parts, ignore_index=True)
    expert_frames = {
        expert["name"]: geometry_audit.relabel(
            base_direct, expert["geometry"], direct=True)
        for expert in EXPERTS
    }

    steps = []
    accepted_rows = []
    test_start = FIRST_TEST
    while test_start < END_TEST:
        test_end = min(test_start + pd.Timedelta(days=TEST_DAYS), END_TEST)
        choices = []
        for expert in EXPERTS:
            name = str(expert["name"])
            train, calibration, test = causal_parts(
                expert_frames[name], test_start, test_end)
            if len(train) < 1000 or len(calibration) < 100:
                continue
            schema = parent.design(train)
            model = geometry_audit.fit_model(
                train, schema, str(expert["model_target"]))
            scored = geometry_audit.score(
                pd.concat([calibration, test], ignore_index=True),
                model, schema, str(expert["model_target"]))
            calibration_scored = scored[
                scored.entry_time < test_start.timestamp()].copy()
            test_scored = scored[
                scored.entry_time >= test_start.timestamp()].copy()
            policy, frontier = monthly.choose_policy(calibration_scored)
            choices.append({
                "expert": name,
                "geometry": {
                    "target_pct": 100 * expert["geometry"][0],
                    "stop_pct": 100 * expert["geometry"][1],
                    "hold_bars": expert["geometry"][2],
                    "model_target": expert["model_target"],
                },
                "train_rows": len(train),
                "calibration_rows": len(calibration),
                "policy": policy,
                "calibration_frontier": frontier,
                "_test": test_scored,
            })
        choices.sort(key=expert_key, reverse=True)
        selected = choices[0] if choices else None
        traded = bool(selected and should_trade(selected))
        test_metrics = None
        if traded:
            policy = selected["policy"]
            test_metrics, rows = parent.evaluate(
                selected["_test"], float(policy["threshold"]),
                int(policy["loss_pause"]), int(policy["hourly_cap"]), .03)
            accepted_rows.extend(rows)
        steps.append({
            "test_start": test_start.isoformat(),
            "test_end": test_end.isoformat(),
            "traded": traded,
            "selected_expert": selected["expert"] if selected else None,
            "selected_policy": ({key: selected["policy"][key]
                                 for key in ("quantile", "threshold",
                                             "loss_pause", "hourly_cap")}
                                if selected else None),
            "selected_calibration": (
                selected["policy"]["metrics"] if selected else None),
            "expert_calibrations": [{
                "expert": item["expert"],
                "geometry": item["geometry"],
                "train_rows": item["train_rows"],
                "calibration_rows": item["calibration_rows"],
                "policy": {key: item["policy"][key]
                           for key in ("quantile", "threshold",
                                       "loss_pause", "hourly_cap")},
                "metrics": item["policy"]["metrics"],
            } for item in choices],
            "test_metrics_3pct": test_metrics,
        })
        test_start = test_end

    combined_rows = parent.rails.ranked_runtime_gate(
        accepted_rows, loss_pause=999, hourly_cap=2)
    pd.DataFrame(combined_rows).to_pickle(ROWS_OUT)
    combined = {
        "margin_3pct": money(combined_rows, .03),
        "margin_5pct": money(combined_rows, .05),
    }
    complete_months = ("2026-06", "2026-07", "2026-08", "2026-09")
    five = combined["margin_5pct"]
    values = five.get("monthly_return_pct", {})
    objective = bool(
        set(complete_months).issubset(values)
        and min(float(values[month]) for month in complete_months) >= 30
        and five["negative_months"] == 0
        and five["dd_pct"] <= 10
        and five["trades_per_calendar_day"] >= 3)
    report = {
        "status": ("XPERP_WEEKLY_EXPERT_SHADOW_CANDIDATE"
                   if objective else "REJECTED_OBJECTIVE"),
        "ready_for_live_orders": False,
        "protocol": {
            "experts": list(EXPERTS),
            "calibration_days": CALIBRATION_DAYS,
            "test_days": TEST_DAYS,
            "train": "direct rows closed before calibration window",
            "flat_rule": "skip next week if selected calibration policy fails profit/frequency/DD rails",
            "entry": "market at next 15-minute open; hourly decisions",
            "orders": "market only",
            "money": "client 3% or 5% current-equity margin; fixed 10x",
        },
        "walkforward": steps,
        "combined": combined,
        "objective_complete_months": list(complete_months),
        "objective_pass": objective,
        "sources": {
            **geometry_audit.SOURCE_HASHES,
            str(Path(__file__).resolve()): digest(Path(__file__).resolve()),
        },
        "limitations": [
            "The two experts were motivated by earlier research on the same direct history.",
            "Twenty-one calibration days remain a short regime sample.",
            "October is incomplete and remains diagnostic.",
            "Gross returns exclude fees at user request.",
            "No backtest guarantees a future monthly return.",
        ],
    }
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({
        "status": report["status"],
        "weeks": len(steps),
        "traded_weeks": sum(step["traded"] for step in steps),
        "expert_counts": dict(collections.Counter(
            step["selected_expert"] for step in steps if step["traded"])),
        "combined": combined,
        "objective_pass": objective,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
