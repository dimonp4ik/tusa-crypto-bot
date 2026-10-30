"""Causal profit-bank fraction controller for the frozen crypto route.

The entry model, 0.5R first target, 2R runner, stop and portfolio rails remain
unchanged.  Only the fraction banked at 0.5R may increase when the strictly
completed recent runner-hit rate falls below a predeclared threshold.  The
decision for an entry never sees an outcome whose exit time is equal to or
later than that entry.  Selection ends in 2023; 2024, 2025-2026 and current
X-Perp are holdouts.
"""
from __future__ import annotations

from collections import deque
from datetime import datetime, timezone
import hashlib
import heapq
import itertools
import json
from pathlib import Path

import pandas as pd

import crypto_high_target_tree_oct5_audit as base
import crypto_profit_first_material_target_oct30_forward as forward


ROOT = Path("reports/audit_2026_09_08")
HISTORICAL_ROWS = ROOT / "crypto_profit_first_material_target_oct18.pkl"
DIRECT_ROWS = ROOT / "crypto_profit_first_material_target_xperp_oct30.pkl"
SOURCE_REPORT = ROOT / "crypto_profit_first_material_target_oct30_forward.json"
ORIGINAL_REPORT = ROOT / "crypto_profit_first_material_target_oct18.json"
OUT = ROOT / "crypto_profit_first_causal_fraction_oct30.json"
ROWS_OUT = ROOT / "crypto_profit_first_causal_fraction_oct30.pkl"
BASE_FRACTION = .05
WINDOWS = (20, 40, 60)
THRESHOLDS = (.20, .25, .30)
DEFENSIVE_FRACTIONS = (.25, .50, 1.0)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def causal_fraction(frame: pd.DataFrame, *, window: int, threshold: float,
                    defensive_fraction: float) -> pd.DataFrame:
    """Reweight frozen accepted rows using only strictly completed outcomes."""
    if not 0 <= BASE_FRACTION <= defensive_fraction <= 1:
        raise ValueError("invalid bank fractions")
    pending: list[tuple[float, int, dict]] = []
    history: deque[int] = deque(maxlen=window)
    output = []
    maximum_causal_violation = 0
    ordered = frame.sort_values(["entry_time", "symbol", "direction"]).to_dict("records")
    for sequence, row in enumerate(ordered):
        now = float(row["entry_time"])
        while pending and pending[0][0] < now:
            exit_time, _, completed = heapq.heappop(pending)
            maximum_causal_violation = max(
                maximum_causal_violation, int(exit_time >= now))
            outcome = str(completed["barbell_outcome"])
            if outcome in ("TP_RUNNER", "TP_BE"):
                history.append(int(outcome == "TP_RUNNER"))
        rate = (sum(history) / len(history)) if history else None
        defensive = bool(len(history) >= window and rate < threshold)
        fraction = defensive_fraction if defensive else BASE_FRACTION
        result = dict(row)
        outcome = str(row["barbell_outcome"])
        if outcome == "SL":
            net_r = float(row["net_r"])
        elif outcome == "TP_RUNNER":
            net_r = fraction * .5 + (1-fraction) * 2.0
        elif outcome == "TP_BE":
            net_r = fraction * .5
        else:
            raise ValueError(f"unexpected frozen outcome: {outcome}")
        result.update({
            "net_r": float(net_r),
            "signed_price_return": float(net_r) * float(row["risk_pct"]),
            "causal_near_fraction": float(fraction),
            "runner_history_n": len(history),
            "runner_history_hit_rate": rate,
            "defensive_fraction_active": defensive,
        })
        output.append(result)
        heapq.heappush(pending, (
            float(row["exit_time"]), sequence, dict(row)))
    result = pd.DataFrame(output)
    result.attrs["strictly_prior_exit_violations"] = maximum_causal_violation
    return result


def fixed_fraction(frame: pd.DataFrame) -> pd.DataFrame:
    return causal_fraction(frame, window=20, threshold=-1,
                           defensive_fraction=BASE_FRACTION)


def stage(frame: pd.DataFrame, periods: tuple[str, ...]) -> pd.DataFrame:
    return frame[frame.period.astype(str).isin(periods)].copy()


def metric(frame: pd.DataFrame) -> dict:
    return base.evaluate(frame.to_dict("records"))


def compact(value: dict) -> dict:
    return {key: value[key] for key in (
        "n", "return_pct", "dd_pct", "wr", "negative_months",
        "max_loss_streak", "trades_per_calendar_day")}


def historical_metrics(frame: pd.DataFrame) -> dict:
    return {
        "selection_2022h1": metric(stage(frame, ("2022h1",))),
        "confirmation_2022h2_2023": metric(stage(frame, ("2022h2", "2023"))),
        "validation_2024": metric(stage(frame, ("2024",))),
        "later_2025_2026": metric(stage(frame, ("2025", "2026"))),
    }


def dominates(value: dict, reference: dict) -> bool:
    return bool(value["return_pct"] > reference["return_pct"]
                and value["dd_pct"] <= reference["dd_pct"] + .25
                and value["negative_months"] <= reference["negative_months"]
                and value["trades_per_calendar_day"]
                >= reference["trades_per_calendar_day"])


def direct_metrics(frame: pd.DataFrame) -> dict:
    original = json.loads(ORIGINAL_REPORT.read_text(encoding="utf-8"))
    cutoff = int(pd.Timestamp(
        original["fresh_direct_coverage"]["last_entry"]
    ).timestamp())
    cumulative = frame[frame.entry_time.gt(cutoff)].copy()
    october = frame[frame.entry_time.ge(
        int(pd.Timestamp("2026-10-01", tz="UTC").timestamp()))].copy()
    return {
        "all": forward.metrics(frame, .03),
        "october": forward.metrics(october, .03),
        "after_oct16": forward.metrics(cumulative, .03),
    }


def main() -> None:
    historical = pd.read_pickle(HISTORICAL_ROWS)
    direct = pd.read_pickle(DIRECT_ROWS)
    baseline_historical_rows = fixed_fraction(historical)
    baseline = historical_metrics(baseline_historical_rows)
    candidates = []
    rows_by_config = {}
    for window, threshold, fraction in itertools.product(
            WINDOWS, THRESHOLDS, DEFENSIVE_FRACTIONS):
        rows = causal_fraction(
            historical, window=window, threshold=threshold,
            defensive_fraction=fraction)
        values = historical_metrics(rows)
        development_pass = all(dominates(values[name], baseline[name])
                               for name in ("selection_2022h1",
                                            "confirmation_2022h2_2023"))
        config = {"window": window, "runner_hit_threshold": threshold,
                  "defensive_near_fraction": fraction}
        key = json.dumps(config, sort_keys=True)
        rows_by_config[key] = rows
        candidates.append({
            "config": config, "development_pass": development_pass,
            "development_min_return_gain": min(
                values[name]["return_pct"]-baseline[name]["return_pct"]
                for name in ("selection_2022h1", "confirmation_2022h2_2023")),
            "periods": {name: compact(value) for name, value in values.items()},
            "strictly_prior_exit_violations":
                rows.attrs["strictly_prior_exit_violations"],
        })
    qualified = [item for item in candidates if item["development_pass"]]
    qualified.sort(key=lambda item: (
        -item["development_min_return_gain"],
        json.dumps(item["config"], sort_keys=True)))
    selected = qualified[0] if qualified else None
    historical_holdout_pass = False
    selected_direct = None
    direct_pass = False
    if selected:
        config = selected["config"]
        historical_holdout_pass = all(
            selected["periods"][name]["return_pct"] >= baseline[name]["return_pct"]
            and selected["periods"][name]["dd_pct"] <= baseline[name]["dd_pct"] + .25
            and selected["periods"][name]["negative_months"] <= baseline[name]["negative_months"]
            for name in ("validation_2024", "later_2025_2026"))
        selected_direct_rows = causal_fraction(
            direct, window=int(config["window"]),
            threshold=float(config["runner_hit_threshold"]),
            defensive_fraction=float(config["defensive_near_fraction"]))
        selected_direct_rows.to_pickle(ROWS_OUT)
        selected_direct = direct_metrics(selected_direct_rows)
        base_direct = direct_metrics(fixed_fraction(direct))
        direct_pass = all(
            selected_direct[scope]["return_pct"] >= base_direct[scope]["return_pct"]
            and selected_direct[scope]["dd_pct"] <= base_direct[scope]["dd_pct"] + .25
            and selected_direct[scope]["negative_months"] <= base_direct[scope]["negative_months"]
            for scope in ("all", "october", "after_oct16"))
    else:
        base_direct = direct_metrics(fixed_fraction(direct))
    accepted = bool(historical_holdout_pass and direct_pass)
    report = {
        "status": ("SHADOW_CANDIDATE_REQUIRES_GENUINELY_NEW_FORWARD"
                   if accepted else "REJECTED_HOLDOUT" if selected
                   else "REJECTED_DEVELOPMENT"),
        "ready_for_live_orders": False, "orders_possible": False,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "protocol": {
            "base_exit": "bank 5% at 0.5R; protected remainder to 2R",
            "candidate": "raise bank fraction after weak strictly completed runner history",
            "windows": list(WINDOWS), "runner_hit_thresholds": list(THRESHOLDS),
            "defensive_near_fractions": list(DEFENSIVE_FRACTIONS),
            "selection": ["2022h1", "2022h2-2023"],
            "holdouts": ["2024", "2025-2026", "direct X-Perp through Oct30"],
            "entry": "unchanged next-15m market entry", "orders": "market only",
        },
        "baseline_historical": {name: compact(value)
                                for name, value in baseline.items()},
        "candidate_count": len(candidates),
        "selected_before_holdouts": selected,
        "historical_holdout_pass": historical_holdout_pass,
        "direct_xperp": {"baseline": base_direct,
                         "candidate": selected_direct, "pass": direct_pass},
        "top_development": sorted(
            candidates, key=lambda item: item["development_min_return_gain"],
            reverse=True)[:10],
        "sources": {
            str(HISTORICAL_ROWS): digest(HISTORICAL_ROWS),
            str(DIRECT_ROWS): digest(DIRECT_ROWS),
            str(SOURCE_REPORT): digest(SOURCE_REPORT),
            str(ORIGINAL_REPORT): digest(ORIGINAL_REPORT),
            str(Path(__file__).resolve()): digest(Path(__file__).resolve()),
        },
        "limitations": [
            "The controller uses frozen-route outcomes and leaves portfolio identities unchanged.",
            "Direct X-Perp history begins in 2026.",
            "Gross returns exclude fees at user request.",
        ],
    }
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({
        "status": report["status"], "selected": selected,
        "historical_holdout_pass": historical_holdout_pass,
        "direct_xperp": report["direct_xperp"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
