"""Causal stability filter for the exact X-Perp target router.

The October target policy is kept frozen.  Coarse route groups are learned
from two disjoint pre-September X-Perp windows, the filter form is selected
on September, and October is opened once as the untouched test month.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

import crypto_profit_first_walkforward_target_router_oct18_audit as frozen
import crypto_walkforward_target_router_oct24_forward as forward
from crypto_runtime_shadow_portfolio_oct3_audit import money
from src.research_profit_gate import economic_gate


ROOT = Path("reports/audit_2026_09_08")
SOURCE = ROOT / "crypto_walkforward_target_router_oct24_forward.json"
OUT = ROOT / "crypto_stable_route_group_oct24_audit.json"
ROWS_OUT = ROOT / "crypto_stable_route_group_oct24_audit.pkl"
SELECTION_ROWS_OUT = ROOT / "crypto_stable_route_group_oct24_selection.pkl"

GROUP_SPECS = (
    ("family", "direction"),
    ("family", "direction", "btc_regime"),
    ("family", "direction", "target_r"),
    ("direction", "btc_regime", "target_r"),
    ("family", "direction", "btc_regime", "target_r"),
    ("family", "direction", "utc_session"),
    ("symbol", "direction"),
    ("symbol", "family", "direction"),
)
MIN_COUNTS = (3, 5, 8, 12)
MEAN_FLOORS = (0.0, 0.03, 0.06)
LOSS_PAUSES = (3, 4, 5, 6, 7, 8)
HOURLY_CAPS = (1, 2, 3, 4)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def as_month(frame: pd.DataFrame) -> pd.Series:
    return pd.to_datetime(frame.entry_time, unit="s", utc=True).dt.strftime(
        "%Y-%m")


def route_key(row: dict, spec: tuple[str, ...]) -> tuple:
    return tuple(row[name] for name in spec)


def group_stats(rows: list[dict], spec: tuple[str, ...]) -> dict[tuple, dict]:
    frame = pd.DataFrame(rows)
    if frame.empty:
        return {}
    result = {}
    for key, group in frame.groupby(list(spec), dropna=False, sort=True):
        if not isinstance(key, tuple):
            key = (key,)
        result[key] = {
            "n": int(len(group)),
            "mean_r": float(group.net_r.mean()),
            "sum_r": float(group.net_r.sum()),
        }
    return result


def learned_groups(early: list[dict], late: list[dict],
                   spec: tuple[str, ...], min_count: int,
                   mean_floor: float, mode: str) -> set[tuple]:
    a = group_stats(early, spec)
    b = group_stats(late, spec)
    common = set(a) & set(b)
    if mode == "allow_stable_positive":
        return {key for key in common
                if a[key]["n"] >= min_count and b[key]["n"] >= min_count
                and a[key]["mean_r"] > mean_floor
                and b[key]["mean_r"] > mean_floor}
    if mode == "deny_stable_negative":
        return {key for key in common
                if a[key]["n"] >= min_count and b[key]["n"] >= min_count
                and a[key]["mean_r"] < -mean_floor
                and b[key]["mean_r"] < -mean_floor}
    raise ValueError(mode)


def apply_filter(rows: list[dict], spec: tuple[str, ...], groups: set[tuple],
                 mode: str) -> list[dict]:
    if mode == "allow_stable_positive":
        return [row for row in rows if route_key(row, spec) in groups]
    return [row for row in rows if route_key(row, spec) not in groups]


def gated(rows: list[dict], policy: dict | None = None, *,
          loss_pause: int | None = None,
          hourly_cap: int | None = None) -> list[dict]:
    if policy is not None:
        loss_pause = int(policy["loss_pause"])
        hourly_cap = int(policy["hourly_cap"])
    if loss_pause is None or hourly_cap is None:
        raise ValueError("runtime rails are required")
    return frozen.be.ranked_runtime_gate(
        rows, loss_pause=loss_pause, hourly_cap=hourly_cap)


def metrics(rows: list[dict], margin: float = .03) -> dict:
    value = money(rows, margin)
    signed = [float(row["signed_price_return"]) for row in rows]
    value["economic_gate"] = economic_gate(
        signed, float(value["trades_per_calendar_day"]),
        margin_fraction=margin) if signed else {"passes": False}
    return value


def compact(value: dict) -> dict:
    return {name: value[name] for name in (
        "n", "return_pct", "dd_pct", "wr", "max_loss_streak",
        "trades_per_calendar_day", "negative_months")}


def main() -> None:
    source = json.loads(SOURCE.read_text(encoding="utf-8"))
    policy = source["protocol"]["policy"]
    historical = frozen.historical_wide()
    historical_fit = historical[~historical.period.eq("2026")].copy()
    old_direct = forward.direct_wide(
        frozen.prior.FRESH_NEAR, frozen.prior.FRESH_FAR,
        frozen.be.DIRECT_CACHE)
    direct = forward.direct_wide(
        forward.NEW_NEAR, forward.NEW_FAR, forward.NEW_CACHE)
    old_direct["month"] = as_month(old_direct)
    direct["month"] = as_month(direct)

    # Reuse the exact pre-September snapshot from the frozen October model.
    # Later downloads are allowed only on the scoring side.
    older_direct = old_direct[old_direct.month < "2026-09"].copy()
    train = pd.concat([historical_fit, older_direct], ignore_index=True)
    scoring = direct[direct.month <= "2026-10"].copy().reset_index(drop=True)
    routed = frozen.route(
        scoring, frozen.fit_models_on_rows(train, scoring))
    routed["month"] = as_month(routed)

    by_month = {}
    for month in sorted(routed.month.unique()):
        frame = routed[routed.month.eq(month)].copy()
        by_month[month] = frozen.candidate_rows(
            frame, float(policy["threshold"]))

    early = [row for month in ("2026-02", "2026-03", "2026-04", "2026-05")
             for row in by_month.get(month, [])]
    late = [row for month in ("2026-06", "2026-07", "2026-08")
            for row in by_month.get(month, [])]
    september = by_month["2026-09"]
    october = by_month["2026-10"]

    baseline_sep_rows = gated(september, policy)
    baseline_oct_rows = gated(october, policy)
    baseline_sep = metrics(baseline_sep_rows)
    baseline_oct = metrics(baseline_oct_rows)
    expected_oct = source["october_to_date"]["margin_3pct"]
    for field in ("n", "return_pct", "dd_pct", "wr",
                  "trades_per_calendar_day", "negative_months"):
        if baseline_oct[field] != expected_oct[field]:
            raise RuntimeError(
                f"frozen October mismatch {field}: "
                f"{baseline_oct[field]} != {expected_oct[field]}")

    frontier = []
    learned = {}
    for spec in GROUP_SPECS:
        for mode in ("allow_stable_positive", "deny_stable_negative"):
            for min_count in MIN_COUNTS:
                for mean_floor in MEAN_FLOORS:
                    groups = learned_groups(
                        early, late, spec, min_count, mean_floor, mode)
                    if not groups:
                        continue
                    sep_rows = gated(apply_filter(
                        september, spec, groups, mode), policy)
                    if not sep_rows:
                        continue
                    value = metrics(sep_rows)
                    candidate = {
                        "spec": list(spec), "mode": mode,
                        "min_count_per_development_block": min_count,
                        "mean_r_floor": mean_floor,
                        "learned_group_count": len(groups),
                        "selection_metrics": compact(value),
                        "selection_score": round(
                            value["return_pct"] - .25 * value["dd_pct"], 8),
                    }
                    frontier.append(candidate)
                    learned[(spec, mode, min_count, mean_floor)] = groups

    eligible = [item for item in frontier
                if item["selection_metrics"]["trades_per_calendar_day"] >= 3
                and item["selection_metrics"]["dd_pct"] <= 10
                and item["selection_metrics"]["n"] >= 90]
    pool = eligible or frontier
    pool.sort(key=lambda item: (
        item["selection_score"],
        item["selection_metrics"]["return_pct"],
        -item["selection_metrics"]["dd_pct"]), reverse=True)
    selected = pool[0]
    spec = tuple(selected["spec"])
    lookup = (spec, selected["mode"],
              selected["min_count_per_development_block"],
              selected["mean_r_floor"])
    groups = learned[lookup]
    selected_oct_rows = gated(apply_filter(
        october, spec, groups, selected["mode"]), policy)
    selected_sep_rows = gated(apply_filter(
        september, spec, groups, selected["mode"]), policy)
    oct3 = metrics(selected_oct_rows, .03)
    oct5 = metrics(selected_oct_rows, .05)
    pd.DataFrame(selected_oct_rows).to_pickle(ROWS_OUT)
    pd.DataFrame(selected_sep_rows).to_pickle(SELECTION_ROWS_OUT)

    improved = bool(
        oct3["return_pct"] > baseline_oct["return_pct"]
        and oct3["dd_pct"] <= baseline_oct["dd_pct"]
        and oct3["trades_per_calendar_day"] >= 3)
    objective = bool(
        oct3["economic_gate"].get("passes", False)
        and oct3["negative_months"] == 0 and oct3["dd_pct"] <= 10)
    report = {
        "status": ("STABLE_GROUP_FILTER_IMPROVED_OCTOBER"
                   if improved else "REJECTED_STABLE_GROUP_FILTER"),
        "ready_for_live_orders": False,
        "order_eligible": False,
        "protocol": {
            "model_and_target_policy": "frozen before October",
            "development_early": "2026-02 through 2026-05",
            "development_late": "2026-06 through 2026-08",
            "filter_selection": "2026-09",
            "untouched_test": "2026-10 through Oct 24",
            "candidate_filter_count": len(frontier),
            "october_used_for_selection": False,
        },
        "baseline": {
            "september": compact(baseline_sep),
            "october": compact(baseline_oct),
        },
        "selected_filter": {
            **selected,
            "learned_groups": [list(key) for key in sorted(groups, key=str)],
        },
        "october_test": {
            "margin_3pct": oct3,
            "margin_5pct": oct5,
        },
        "improves_baseline_with_frequency_rail": improved,
        "objective_pass": objective,
        "selection_frontier": sorted(frontier, key=lambda item: (
            item["selection_score"],
            item["selection_metrics"]["return_pct"]), reverse=True)[:20],
        "sources": {
            str(SOURCE): digest(SOURCE),
            str(forward.NEW_NEAR): digest(forward.NEW_NEAR),
            str(forward.NEW_FAR): digest(forward.NEW_FAR),
            str(forward.NEW_MANIFEST): digest(forward.NEW_MANIFEST),
            str(Path(__file__).resolve()): digest(Path(__file__).resolve()),
        },
        "limitations": [
            "One partial October month is available as untouched validation.",
            "The group search is bounded but September selection can still overfit.",
            "Gross results exclude fees at user request.",
        ],
    }
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True),
                   encoding="utf-8")
    print(json.dumps({
        "status": report["status"], "baseline": report["baseline"],
        "selected_filter": report["selected_filter"],
        "october_test": {
            "margin_3pct": compact(oct3), "margin_5pct": compact(oct5)},
        "objective_pass": objective,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
