"""Audit causal CPI, payroll and FOMC entry blackouts on Profit-First.

The event calendar is normalized from official BLS and Federal Reserve
calendars.  Candidate windows are selected on 2022H1, confirmed separately on
2022H2 and 2023, and only then allowed to see 2024-2026/direct holdouts.
Existing positions are never force-closed; a policy only suppresses new entries
inside its predeclared event window.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

import crypto_profit_first_exact_health_router_oct30_audit as health
import crypto_profit_first_exact_oct31_1700_forward as latest
from crypto_runtime_shadow_portfolio_oct3_audit import money


ROOT = Path("reports/audit_2026_09_08")
EVENT_ROOT = ROOT / "official_macro_events_nov1"
EVENTS = EVENT_ROOT / "events.csv"
MANIFEST = EVENT_ROOT / "manifest.json"
HISTORICAL = ROOT / "_tmp_exact_historical_rows.pkl"
DIRECT = latest.ROWS_OUT
OUT = ROOT / "crypto_profit_first_official_macro_news_nov1.json"
ROWS_OUT = ROOT / "crypto_profit_first_official_macro_news_nov1.pkl"

EVENT_GROUPS = {
    "cpi": ("CPI",),
    "payrolls": ("EMPLOYMENT_SITUATION",),
    "fomc": ("FOMC",),
    "bls_major": ("CPI", "EMPLOYMENT_SITUATION"),
    "all_major": ("CPI", "EMPLOYMENT_SITUATION", "FOMC"),
}
# Minutes before/after the scheduled release.  These were fixed before any
# candidate outcome was inspected.
WINDOWS = ((30, 90), (60, 180), (15, 240), (0, 180))
DEVELOPMENT = ("2022h1", "2022h2", "2023")
HIDDEN = ("2024", "2025", "2026")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_events() -> pd.DataFrame:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if digest(EVENTS) != manifest["events_sha256"]:
        raise ValueError("official event calendar hash mismatch")
    events = pd.read_csv(EVENTS)
    required = {"event_type", "event_time_utc", "source_url"}
    if not required.issubset(events.columns):
        raise ValueError(f"missing event columns: {sorted(required - set(events))}")
    events["event_time"] = pd.to_datetime(
        events.event_time_utc, utc=True, errors="raise").map(
            lambda value: int(value.timestamp()))
    if events.event_time.duplicated().any():
        # CPI and payroll do not share a release time in this snapshot, and a
        # duplicate would make event attribution ambiguous.
        raise ValueError("duplicate official event timestamp")
    return events.sort_values("event_time").reset_index(drop=True)


def event_mask(frame: pd.DataFrame, events: pd.DataFrame,
               event_types: tuple[str, ...], pre_minutes: int,
               post_minutes: int) -> pd.Series:
    """Return entries in any inclusive scheduled-event blackout window."""
    event_times = events.loc[
        events.event_type.isin(event_types), "event_time"].to_numpy(dtype=float)
    entries = frame.entry_time.to_numpy(dtype=float)
    if not len(event_times) or not len(entries):
        return pd.Series(False, index=frame.index)
    event_times.sort()
    previous = np.searchsorted(event_times, entries, side="right") - 1
    following = previous + 1
    hit = np.zeros(len(entries), dtype=bool)
    valid_previous = previous >= 0
    if valid_previous.any():
        delta = entries[valid_previous] - event_times[previous[valid_previous]]
        hit[valid_previous] |= ((delta >= -pre_minutes * 60)
                                & (delta <= post_minutes * 60))
    valid_following = following < len(event_times)
    if valid_following.any():
        delta = entries[valid_following] - event_times[following[valid_following]]
        hit[valid_following] |= ((delta >= -pre_minutes * 60)
                                 & (delta <= post_minutes * 60))
    return pd.Series(hit, index=frame.index)


def apply_blackout(frame: pd.DataFrame, events: pd.DataFrame,
                   event_types: tuple[str, ...], pre_minutes: int,
                   post_minutes: int) -> pd.DataFrame:
    return frame.loc[~event_mask(
        frame, events, event_types, pre_minutes, post_minutes)].copy()


def metric(frame: pd.DataFrame, period: str, module: str) -> dict:
    rows = health.accepted(frame[frame.period.eq(period)], module)
    return money(rows, .03)


def compact(value: dict) -> dict:
    return {key: value[key] for key in (
        "n", "wr", "return_pct", "dd_pct", "max_loss_streak",
        "negative_months", "trades_per_calendar_day", "monthly_return_pct")}


def development_ok(candidate: dict, baseline: dict) -> bool:
    return bool(
        candidate["return_pct"] >= baseline["return_pct"]
        and candidate["dd_pct"] <= baseline["dd_pct"]
        and candidate["negative_months"] <= baseline["negative_months"]
        and candidate["max_loss_streak"] <= baseline["max_loss_streak"]
        and candidate["n"] >= .80 * baseline["n"])


def hidden_ok(candidate: dict, baseline: dict) -> bool:
    return bool(
        candidate["return_pct"] >= baseline["return_pct"]
        and candidate["dd_pct"] <= baseline["dd_pct"] + .5
        and candidate["negative_months"] <= baseline["negative_months"]
        and candidate["max_loss_streak"] <= baseline["max_loss_streak"] + 1)


def direct_metrics(frame: pd.DataFrame) -> dict:
    accepted = pd.DataFrame(health.accepted(frame, "macro_news_direct"))
    october_cutoff = int(pd.Timestamp("2026-10-01", tz="UTC").timestamp())
    october = accepted[accepted.entry_time.ge(october_cutoff)]
    return {
        "all_direct": latest.metrics(accepted, .03),
        "october": latest.metrics(october, .03),
    }


def proximity_attribution(frame: pd.DataFrame, events: pd.DataFrame) -> dict:
    """Describe baseline accepted trades around each event class."""
    accepted = pd.DataFrame(health.accepted(frame, "macro_news_attribution"))
    result = {}
    for group, event_types in EVENT_GROUPS.items():
        for pre, post in WINDOWS:
            nearby = accepted[event_mask(
                accepted, events, event_types, pre, post)]
            result[f"{group}_{pre}m_pre_{post}m_post"] = {
                "n": int(len(nearby)),
                "share_pct": round(100 * len(nearby) / max(len(accepted), 1), 4),
                "mean_r": round(float(nearby.net_r.mean()), 6)
                    if len(nearby) else 0.0,
                "stop_rate_pct": round(100 * float(
                    nearby.runner_outcome.astype(str).str.startswith("SL").mean()), 4)
                    if len(nearby) else 0.0,
                "net_r_sum": round(float(nearby.net_r.sum()), 6),
            }
    return result


def main() -> None:
    events = load_events()
    historical = health.current_outcomes(pd.read_pickle(HISTORICAL))
    direct = pd.read_pickle(DIRECT)
    baseline = {period: metric(historical, period, "macro_news_baseline")
                for period in DEVELOPMENT + HIDDEN}
    baseline_direct = direct_metrics(direct)

    candidates = []
    eligible = []
    for group, event_types in EVENT_GROUPS.items():
        for pre, post in WINDOWS:
            filtered = apply_blackout(
                historical, events, event_types, pre, post)
            development = {
                period: metric(filtered, period, "macro_news_candidate")
                for period in DEVELOPMENT}
            passes = all(development_ok(
                development[period], baseline[period])
                for period in DEVELOPMENT)
            item = {
                "event_group": group,
                "event_types": list(event_types),
                "pre_minutes": pre,
                "post_minutes": post,
                "development": {period: compact(development[period])
                                for period in DEVELOPMENT},
                "development_pass": passes,
            }
            candidates.append(item)
            if passes:
                eligible.append(item)

    def score(item: dict) -> tuple[float, float, float]:
        # Absolute return deltas remain well-defined when a baseline period is
        # negative (2023 is negative in this frozen dataset).
        deltas = [item["development"][period]["return_pct"]
                  - baseline[period]["return_pct"]
                  for period in DEVELOPMENT]
        drawdowns = [item["development"][period]["dd_pct"]
                     for period in DEVELOPMENT]
        return min(deltas), sum(deltas), -max(drawdowns)

    eligible.sort(key=score, reverse=True)
    selected = eligible[0] if eligible else None
    hidden = candidate_direct = None
    robust = False
    selected_direct = None
    if selected:
        event_types = tuple(selected["event_types"])
        selected_historical = apply_blackout(
            historical, events, event_types,
            int(selected["pre_minutes"]), int(selected["post_minutes"]))
        hidden = {period: metric(
            selected_historical, period, "macro_news_hidden")
            for period in HIDDEN}
        selected_direct = apply_blackout(
            direct, events, event_types,
            int(selected["pre_minutes"]), int(selected["post_minutes"]))
        candidate_direct = direct_metrics(selected_direct)
        robust = bool(
            all(hidden_ok(hidden[period], baseline[period]) for period in HIDDEN)
            and hidden_ok(candidate_direct["all_direct"],
                          baseline_direct["all_direct"])
            and hidden_ok(candidate_direct["october"],
                          baseline_direct["october"]))
    if robust and selected_direct is not None:
        selected_direct.to_pickle(ROWS_OUT)

    report = {
        "status": ("OFFICIAL_MACRO_NEWS_SHADOW_CHALLENGER" if robust
                   else "REJECTED_OFFICIAL_MACRO_NEWS_BLACKOUT"),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "ready_for_live_orders": False,
        "orders_possible": False,
        "runtime_change_allowed": False,
        "protocol": {
            "action": "suppress new entries only; never close an open trade",
            "event_timezone": "America/New_York converted to UTC with DST",
            "development": list(DEVELOPMENT),
            "sealed_holdouts": list(HIDDEN) + ["direct X-Perp", "October 2026"],
            "event_groups": {key: list(value)
                             for key, value in EVENT_GROUPS.items()},
            "windows_minutes": [list(value) for value in WINDOWS],
            "money": "3% current-equity margin, fixed 10x gross",
            "orders": "market only",
        },
        "event_count": int(len(events)),
        "event_counts": events.event_type.value_counts().sort_index().to_dict(),
        "baseline": {period: compact(value)
                     for period, value in baseline.items()},
        "baseline_direct": baseline_direct,
        "historical_event_attribution": proximity_attribution(
            historical, events),
        "direct_event_attribution": proximity_attribution(direct, events),
        "candidate_count": len(candidates),
        "development_pass_count": len(eligible),
        "selected_before_holdouts": selected,
        "hidden": ({period: compact(value) for period, value in hidden.items()}
                   if hidden else None),
        "direct_xperp": candidate_direct,
        "robust_improvement": robust,
        "target_30pct_each_month_proven": False,
        "candidates": candidates,
        "sources": {str(path): digest(path) for path in (
            EVENTS, MANIFEST, HISTORICAL, DIRECT, Path(__file__).resolve())},
        "limitations": [
            "The direct replay is already portfolio-gated; removed entries do not refill freed slots.",
            "Scheduled times measure release risk, not the numerical surprise in the release.",
            "Gross returns exclude fees at user request.",
        ],
    }
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({
        "status": report["status"],
        "event_count": report["event_count"],
        "development_pass_count": len(eligible),
        "selected": selected,
        "hidden": report["hidden"],
        "direct": candidate_direct,
        "closest_event_attribution": sorted(
            report["historical_event_attribution"].items(),
            key=lambda item: item[1]["mean_r"])[:5],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
