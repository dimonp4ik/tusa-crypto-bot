"""Causal group-health router for exact 0.5%-2% crypto moves.

Each review period admits only groups whose completed trades in the trailing
window retained positive expectancy and target-hit quality.  Policy selection
ends in 2024; 2025-2026 are evaluated only after the policy is frozen.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

import crypto_actual_move_health_gate_oct6_audit as base
from crypto_runtime_shadow_portfolio_oct3_audit import money, runtime_gate


ROOT = Path("reports/audit_2026_09_08")
OUT = ROOT / "crypto_actual_move_group_health_oct20.json"
ROWS_OUT = ROOT / "crypto_actual_move_group_health_oct20.pkl"
GROUP_FIELDS = (
    ("module",),
    ("module", "direction"),
    ("module", "symbol"),
    ("module", "symbol", "direction"),
)
REVIEW_DAYS = (1, 3, 7, 14)
LOOKBACK_DAYS = (14, 30, 60, 90, 180)
MINIMUMS = (3, 5, 10, 20)
MEAN_MINS = (-.0005, -.00025, 0.0, .00025, .0005)
HIT_MINS = (.50, .55, .60, .65)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def group_health_features(rows: list[dict], group_fields: tuple[str, ...],
                          review_days: int,
                          lookback_days: int) -> pd.DataFrame:
    """Attach statistics available strictly before each review boundary."""
    frame = pd.DataFrame(rows).sort_values("entry_time").reset_index(drop=True)
    review_seconds = int(review_days * 86400)
    lookback_seconds = int(lookback_days * 86400)
    frame["review_start"] = (
        frame.entry_time.astype(np.int64) // review_seconds) * review_seconds
    n_values = np.zeros(len(frame), dtype=np.int64)
    mean_values = np.full(len(frame), -1.0, dtype=float)
    hit_values = np.zeros(len(frame), dtype=float)
    grouper = group_fields[0] if len(group_fields) == 1 else list(group_fields)
    for _, part in frame.groupby(grouper, sort=False, observed=True):
        history = part.sort_values("exit_time")
        exits = history.exit_time.to_numpy(np.int64)
        signed = history.signed_price_return.to_numpy(float)
        hits = history.literal_target_hit.to_numpy(float)
        signed_sum = np.concatenate(([0.0], np.cumsum(signed)))
        hit_sum = np.concatenate(([0.0], np.cumsum(hits)))
        for start in part.review_start.unique():
            lower = np.searchsorted(exits, start - lookback_seconds, side="left")
            upper = np.searchsorted(exits, start, side="left")
            count = int(upper - lower)
            indices = part.index[part.review_start.eq(start)].to_numpy()
            n_values[indices] = count
            if count:
                mean_values[indices] = (
                    signed_sum[upper] - signed_sum[lower]) / count
                hit_values[indices] = (hit_sum[upper] - hit_sum[lower]) / count
    frame["health_n"] = n_values
    frame["health_mean"] = mean_values
    frame["health_hit"] = hit_values
    return frame


def apply_spec(frame: pd.DataFrame, spec: dict) -> list[dict]:
    chosen = frame[
        frame.health_n.ge(spec["minimum_completed"])
        & frame.health_mean.ge(spec["mean_min"])
        & frame.health_hit.ge(spec["hit_min"])]
    return chosen.to_dict("records")


def year_rows(rows: list[dict], selected: tuple[int, ...]) -> list[dict]:
    wanted = set(selected)
    return [row for row in rows if int(row["year"]) in wanted]


def evaluate(rows: list[dict], selected: tuple[int, ...], margin: float,
             loss_pause: int = 3, hourly_cap: int = 2) -> dict:
    accepted = runtime_gate(year_rows(rows, selected), loss_pause=loss_pause,
                            hourly_cap=hourly_cap)
    result = money(accepted, margin)
    result["literal_target_hit_rate"] = round(
        100 * sum(bool(row["literal_target_hit"]) for row in accepted)
        / len(accepted), 4) if accepted else 0.0
    return result


def floor_month(value: dict) -> float:
    months = value.get("monthly_return_pct", {}).values()
    return min(map(float, months), default=-999.0)


def development_ok(periods: dict[str, dict]) -> bool:
    return all(
        value["n"] >= 500 and value["return_pct"] > 0
        and value["dd_pct"] <= 8
        and value["trades_per_calendar_day"] >= 2.5
        and value["literal_target_hit_rate"] >= 58
        for value in periods.values())


def row_key(row: dict) -> tuple[str, str, int]:
    return (str(row["module"]), str(row["symbol"]),
            int(float(row["entry_time"])))


def main() -> None:
    rows = base.source_rows()
    print(f"exact source rows={len(rows)}", flush=True)
    feature_cache = {}
    rough = []
    for fields, review, lookback in itertools.product(
            GROUP_FIELDS, REVIEW_DAYS, LOOKBACK_DAYS):
        if lookback < review:
            continue
        features = group_health_features(rows, fields, review, lookback)
        feature_cache[(fields, review, lookback)] = features
        development = features[features.year.isin((2022, 2023, 2024))]
        for minimum, mean_min, hit_min in itertools.product(
                MINIMUMS, MEAN_MINS, HIT_MINS):
            chosen = development[
                development.health_n.ge(minimum)
                & development.health_mean.ge(mean_min)
                & development.health_hit.ge(hit_min)]
            grouped = chosen.groupby("year").agg(
                n=("literal_target_hit", "size"),
                hit=("literal_target_hit", "mean"),
                signed=("signed_price_return", "sum"))
            if (set(grouped.index) != {2022, 2023, 2024}
                    or grouped.n.min() < 500
                    or grouped.hit.min() < .55
                    or grouped.signed.min() <= 0):
                continue
            rough.append({
                "group_fields": list(fields), "review_days": review,
                "lookback_days": lookback, "minimum_completed": minimum,
                "mean_min": mean_min, "hit_min": hit_min,
                "rough_floor_signed": float(grouped.signed.min()),
                "rough_floor_hit": float(grouped.hit.min()),
            })
        print(f"features fields={fields} review={review} lookback={lookback}",
              flush=True)
    rough.sort(key=lambda item: (
        item["rough_floor_signed"], item["rough_floor_hit"]), reverse=True)
    print(f"rough candidates={len(rough)}", flush=True)

    candidates = []
    for spec in rough[:160]:
        fields = tuple(spec["group_fields"])
        features = feature_cache[(fields, spec["review_days"],
                                  spec["lookback_days"])]
        gated = apply_spec(features, spec)
        periods = {
            str(year): evaluate(gated, (year,), .03)
            for year in (2022, 2023, 2024)
        }
        candidates.append({
            "policy_kind": "single_health_gate",
            "spec": spec, "development": periods,
            "development_floor_month_pct": round(min(
                floor_month(value) for value in periods.values()), 4),
            "development_worst_return_pct": min(
                value["return_pct"] for value in periods.values()),
            "development_max_dd_pct": max(
                value["dd_pct"] for value in periods.values()),
            "admissible": development_ok(periods),
            "_rows": gated,
        })
    candidates.sort(key=lambda item: (
        item["admissible"], item["development_floor_month_pct"],
        item["development_worst_return_pct"],
        -item["development_max_dd_pct"]), reverse=True)
    single_candidate_count = len(candidates)

    # Freeze a consensus library using development results only.  A trade is
    # admitted when the required fraction of the best admissible causal gates
    # agrees that its group is healthy.
    admissible_library = [item for item in candidates if item["admissible"]]
    base_by_key = {row_key(row): row for row in rows}
    ensemble_candidates = []
    for library_size, agreement in itertools.product(
            (3, 5, 10, 20, 40), (.50, .67, .80, 1.0)):
        library = admissible_library[:library_size]
        if len(library) < library_size:
            continue
        votes = Counter(row_key(row) for item in library for row in item["_rows"])
        required = math.ceil(library_size * agreement)
        agreed = [base_by_key[key] for key, count in votes.items()
                  if count >= required]
        periods = {
            str(year): evaluate(agreed, (year,), .03)
            for year in (2022, 2023, 2024)
        }
        ensemble_candidates.append({
            "policy_kind": "development_consensus",
            "spec": {"library_size": library_size,
                     "agreement_fraction": agreement,
                     "required_votes": required,
                     "library_specs": [item["spec"] for item in library]},
            "development": periods,
            "development_floor_month_pct": round(min(
                floor_month(value) for value in periods.values()), 4),
            "development_worst_return_pct": min(
                value["return_pct"] for value in periods.values()),
            "development_max_dd_pct": max(
                value["dd_pct"] for value in periods.values()),
            "admissible": development_ok(periods),
            "_rows": agreed,
        })
    candidates.extend(ensemble_candidates)
    candidates.sort(key=lambda item: (
        item["admissible"], item["development_floor_month_pct"],
        item["development_worst_return_pct"],
        -item["development_max_dd_pct"]), reverse=True)
    frozen = candidates[0] if candidates else None
    holdouts = None
    selected_rows = []
    if frozen:
        selected_rows = frozen["_rows"]
        holdouts = {
            f"margin_{int(margin * 100)}pct": {
                str(year): evaluate(selected_rows, (year,), margin)
                for year in (2025, 2026)
            } for margin in (.03, .05)
        }
        pd.DataFrame(selected_rows).to_pickle(ROWS_OUT)
    passed = bool(
        frozen and frozen["admissible"] and holdouts
        and all(value["return_pct"] > 0 and value["dd_pct"] <= 10
                and value["trades_per_calendar_day"] >= 3
                and value["negative_months"] == 0
                for value in holdouts["margin_3pct"].values()))

    def public(item):
        return ({key: value for key, value in item.items() if key != "_rows"}
                if item else None)

    report = {
        "status": ("GROUP_HEALTH_AWAITING_DIRECT_VENUE_FORWARD" if passed
                   else "REJECTED_OBJECTIVE"),
        "ready_for_live_orders": False,
        "protocol": {
            "component_source": "exact 0.5%-2% move portfolio",
            "policy_selection": [2022, 2023, 2024],
            "holdouts": [2025, 2026],
            "causality": "only exits strictly before each review boundary",
            "entry": "next 5m market open", "orders": "market only",
            "money": "client 3% or 5% current-equity margin; fixed 10x",
        },
        "source_rows": len(rows), "rough_candidate_count": len(rough),
        "single_candidate_count": single_candidate_count,
        "ensemble_candidate_count": len(ensemble_candidates),
        "exact_candidate_count": len(candidates),
        "admissible_count": sum(item["admissible"] for item in candidates),
        "selected_before_holdouts": public(frozen),
        "top20_before_holdouts": [public(item) for item in candidates[:20]],
        "holdouts": holdouts, "objective_pass": passed,
        "sources": {
            str(Path(base.__file__).resolve()): digest(Path(base.__file__).resolve()),
            str(Path(__file__).resolve()): digest(Path(__file__).resolve()),
        },
        "limitations": [
            "Component definitions came from earlier finite searches.",
            "The health grid creates selection risk despite causal windows.",
            "Binance spot history still requires direct X-Perp transfer.",
            "Gross results exclude fees at user request.",
        ],
    }
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({
        "status": report["status"],
        "rough_candidate_count": len(rough),
        "exact_candidate_count": len(candidates),
        "admissible_count": report["admissible_count"],
        "selected": public(frozen), "holdouts": holdouts,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
