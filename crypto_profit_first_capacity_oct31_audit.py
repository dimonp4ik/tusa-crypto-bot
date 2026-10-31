"""Chronologically test whether wider portfolio rails improve Profit-First.

The entry model, probability threshold and 5%/95% 0.5R/2R exit are the frozen
October policy.  Only capacity rails vary.  2022H1 is development, 2022H2-2023
is validation, and 2024/2025/2026 are revealed one at a time after freezing.
"""
from __future__ import annotations

import collections
import hashlib
import heapq
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from src.crypto_venue_router import ROBUST_SYMBOLS
from src.research_profit_gate import economic_gate


ROOT = Path("reports/audit_2026_09_08")
OUT = ROOT / "crypto_profit_first_capacity_oct31.json"
ROWS_OUT = ROOT / "crypto_profit_first_capacity_oct31.pkl"
PERIODS = ("2020h1", "2020h2", "2021h1", "2021h2", "2022h1",
           "2022h2", "2023", "2024", "2025", "2026")
FIT = PERIODS[:4]
DEVELOPMENT = ("2022h1",)
VALIDATION = ("2022h2", "2023")
HIDDEN = (("2024",), ("2025",), ("2026",))
JOIN_KEYS = ("period", "symbol", "entry_time", "direction", "family")
NUMERIC = (
    "risk_pct", "eff_ratio", "dir_z", "abs_z", "hour_sin", "hour_cos",
    "btc_fast_slow_atr", "btc_eff20", "btc_return20_atr", "btc_z20",
    "btc_atr_pct", "btc_slow_long_atr", "btc_return96_atr", "btc_eff96",
    "btc_trend_dir_atr", "btc_return_dir_atr",
)
CATEGORICAL = ("symbol", "direction", "family", "utc_session", "btc_regime")
MODEL = {"max_leaf_nodes": 7, "min_samples_leaf": 300,
         "learning_rate": .05, "max_iter": 150,
         "l2_regularization": 1.0, "early_stopping": False,
         "random_state": 20260908}
THRESHOLD = 0.3273125824651171
NEAR_FRACTION = .05
POLICIES = tuple(
    {"direction_cap": direction, "rolling_hour_cap": hourly}
    for direction in (3, 4, 5, 6)
    for hourly in (2, 3, 4)
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def cache_path(target: float, period: str) -> Path:
    suffix = str(target).replace(".", "_")
    return ROOT / "raw_router_cache" / f"crypto_raw_{suffix}_{period}.pkl"


def load_target(target: float) -> pd.DataFrame:
    parts = []
    for period in PERIODS:
        part = pd.read_pickle(cache_path(target, period)).copy()
        part["period"] = period
        parts.append(part)
    frame = pd.concat(parts, ignore_index=True)
    return frame[frame.symbol.isin(ROBUST_SYMBOLS)].reset_index(drop=True)


def load_joined() -> pd.DataFrame:
    far, near = load_target(2.0), load_target(.5)
    keys = list(JOIN_KEYS)
    if far.duplicated(keys).any() or near.duplicated(keys).any():
        raise ValueError("target replay contains duplicate signal identities")
    near = near[keys+["outcome", "exit_time"]].rename(columns={
        "outcome": "near_outcome", "exit_time": "near_exit_time"})
    joined = far.merge(near, on=keys, how="inner", validate="one_to_one")
    unmatched = len(far)+len(near)-2*len(joined)
    if unmatched > max(5, int(.0001*max(len(far), len(near)))):
        raise ValueError("0.5R and 2R replays differ materially")
    return joined[joined.risk_pct.between(.01, .02, inclusive="both")].copy()


def design(train: pd.DataFrame) -> dict:
    return {
        "medians": {field: float(pd.to_numeric(
            train[field], errors="coerce").median()) for field in NUMERIC},
        "categories": {field: sorted(
            train[field].fillna("").astype(str).unique().tolist())
            for field in CATEGORICAL},
    }


def matrix(frame: pd.DataFrame, schema: dict) -> np.ndarray:
    columns = []
    for field in NUMERIC:
        columns.append(pd.to_numeric(frame[field], errors="coerce").fillna(
            schema["medians"][field]).to_numpy(np.float32))
    for field in CATEGORICAL:
        values = frame[field].fillna("").astype(str).to_numpy()
        for category in schema["categories"][field]:
            columns.append((values == category).astype(np.float32))
    return np.column_stack(columns)


def frozen_rows() -> list[dict]:
    frame = load_joined()
    fit = frame[frame.period.isin(FIT)]
    schema = design(fit)
    model = HistGradientBoostingClassifier(**MODEL)
    model.fit(matrix(fit, schema), fit.outcome.eq("TP2").astype(int))
    frame["model_probability"] = model.predict_proba(matrix(frame, schema))[:, 1]
    chosen = frame[frame.model_probability >= THRESHOLD].copy()
    near_hit, far_hit = chosen.near_outcome.eq("TP2"), chosen.outcome.eq("TP2")
    chosen["net_r"] = np.where(
        ~near_hit, -1.0,
        np.where(far_hit, NEAR_FRACTION*.5+(1-NEAR_FRACTION)*2.0,
                 NEAR_FRACTION*.5))
    chosen["barbell_outcome"] = np.where(
        ~near_hit, "SL", np.where(far_hit, "TP_RUNNER", "TP_BE"))
    rows = []
    for row in chosen.to_dict("records"):
        rows.append({
            **row, "module": "profit_first_material_2r",
            "source": "high_target_tree", "source_priority": 3,
            "module_priority": 999,
            "signed_price_return": float(row["risk_pct"])*float(row["net_r"]),
            "outcome": "SL" if float(row["net_r"]) <= 0 else "TP",
        })
    return rows


def capacity_gate(rows: list[dict], *, direction_cap: int,
                  rolling_hour_cap: int, loss_pause: int = 3) -> list[dict]:
    pending: list[tuple[float, int, dict]] = []
    opened: dict[str, dict] = {}
    last: dict[tuple[str, str], float] = {}
    recent: collections.deque[float] = collections.deque()
    output = []
    day = None
    streak = 0
    paused = False
    for row in sorted(rows, key=lambda item: (
            float(item["entry_time"]), int(item["source_priority"]),
            str(item["symbol"]), int(item["module_priority"]))):
        now = float(row["entry_time"])
        new_day = int(now//86400)
        if new_day != day:
            day, streak, paused = new_day, 0, False
        while pending and pending[0][0] < now:
            _, _, done = heapq.heappop(pending)
            opened.pop(str(done["symbol"]), None)
            if int(float(done["exit_time"])//86400) == day:
                streak = streak+1 if done["outcome"] == "SL" else 0
                if streak >= loss_pause:
                    paused = True
        while recent and recent[0] <= now-3600:
            recent.popleft()
        symbol, direction = str(row["symbol"]), str(row["direction"])
        if (paused or symbol in opened or len(recent) >= rolling_hour_cap
                or now-last.get((symbol, direction), -1e20) < 10800
                or sum(str(item["direction"]) == direction
                       for item in opened.values()) >= direction_cap):
            continue
        opened[symbol] = row
        last[(symbol, direction)] = now
        recent.append(now)
        output.append(row)
        heapq.heappush(pending, (float(row["exit_time"]), len(output), row))
    return output


def money(rows: list[dict]) -> dict:
    if not rows:
        return {"n": 0, "wr": 0.0, "return_pct": 0.0, "dd_pct": 0.0,
                "negative_months": 0, "max_loss_streak": 0,
                "trades_per_calendar_day": 0.0, "monthly_return_pct": {}}
    equity = peak = 1.0
    max_dd = 0.0
    pending, completed = [], []
    sequence = 0

    def realize(until):
        nonlocal equity, peak, max_dd
        while pending and (until is None or pending[0][0] < until):
            exit_time, _, pnl, row = heapq.heappop(pending)
            equity += pnl
            peak = max(peak, equity)
            max_dd = max(max_dd, (peak-equity)/peak)
            completed.append((exit_time, pnl, row))

    for row in sorted(rows, key=lambda item: (item["entry_time"], item["symbol"])):
        realize(float(row["entry_time"]))
        pnl = equity*.03*10*float(row["signed_price_return"])
        sequence += 1
        heapq.heappush(pending, (float(row["exit_time"]), sequence, pnl, row))
    realize(None)
    completed.sort(key=lambda item: (item[0], item[2]["entry_time"], item[2]["symbol"]))
    ledger = 1.0
    starts, ends = {}, {}
    wins = streak = max_streak = 0
    for exit_time, pnl, row in completed:
        wins += int(pnl > 0)
        streak = 0 if pnl > 0 else streak+1
        max_streak = max(max_streak, streak)
        month = pd.Timestamp(exit_time, unit="s", tz="UTC").strftime("%Y-%m")
        starts.setdefault(month, ledger)
        ledger += pnl
        ends[month] = ledger
    monthly = {key: 100*(ends[key]/starts[key]-1) for key in sorted(starts)}
    first = pd.Timestamp(min(row["entry_time"] for row in rows), unit="s", tz="UTC")
    last = pd.Timestamp(max(row["entry_time"] for row in rows), unit="s", tz="UTC")
    days = max(1, (last.date()-first.date()).days+1)
    tpd = len(rows)/days
    result = {
        "n": len(rows), "wr": round(100*wins/len(rows), 2),
        "return_pct": round(100*(equity-1), 4),
        "dd_pct": round(100*max_dd, 4), "max_loss_streak": max_streak,
        "negative_months": sum(value < -1e-12 for value in monthly.values()),
        "monthly_return_pct": {key: round(value, 4) for key, value in monthly.items()},
        "minimum_month_pct": round(min(monthly.values()), 4) if monthly else 0.0,
        "trades_per_calendar_day": round(tpd, 3),
    }
    result["economic_gate"] = economic_gate(
        [float(row["signed_price_return"]) for row in rows], tpd,
        market_days=30)
    return result


def stage(rows: list[dict], periods: tuple[str, ...], policy: dict) -> tuple[dict, list[dict]]:
    selected = [row for row in rows if row["period"] in periods]
    accepted = capacity_gate(selected, **policy)
    value = money(accepted)
    value["peak_open_positions"] = peak_open(accepted)[0]
    value["peak_same_direction"] = peak_open(accepted)[1]
    value["peak_margin_pct_at_3pct"] = 3*value["peak_open_positions"]
    return value, accepted


def peak_open(rows: list[dict]) -> tuple[int, int]:
    events = []
    for row in rows:
        events.append((float(row["entry_time"]), 1, str(row["direction"])))
        events.append((float(row["exit_time"]), -1, str(row["direction"])))
    counts = collections.Counter()
    total = peak_total = peak_direction = 0
    for _, delta, direction in sorted(events, key=lambda item: (item[0], item[1])):
        total += delta
        counts[direction] += delta
        peak_total = max(peak_total, total)
        peak_direction = max(peak_direction, counts[direction])
    return peak_total, peak_direction


def relative_ok(candidate: dict, baseline: dict) -> bool:
    return bool(candidate["n"] >= baseline["n"]
                and candidate["return_pct"] >= 1.05*baseline["return_pct"]
                and candidate["dd_pct"] <= baseline["dd_pct"]+1
                and candidate["negative_months"] <= baseline["negative_months"]+1
                and candidate["minimum_month_pct"]
                    >= baseline["minimum_month_pct"]-1
                and candidate["max_loss_streak"] <= baseline["max_loss_streak"]+1)


def hidden_ok(candidate: dict, baseline: dict) -> bool:
    return bool(relative_ok(candidate, baseline)
                and candidate["economic_gate"]["return_pass"]
                and candidate["peak_margin_pct_at_3pct"] <= 90)


def public(item: dict | None) -> dict | None:
    return ({key: value for key, value in item.items() if key != "_rows"}
            if item else None)


def main() -> None:
    rows = frozen_rows()
    baseline_policy = {"direction_cap": 3, "rolling_hour_cap": 2}
    baseline_dev, _ = stage(rows, DEVELOPMENT, baseline_policy)
    baseline_validation, _ = stage(rows, VALIDATION, baseline_policy)
    frontier, survivors = [], []
    for policy in POLICIES:
        development, _ = stage(rows, DEVELOPMENT, policy)
        item = {"policy": policy, "development_2022h1": development,
                "development_baseline": baseline_dev}
        frontier.append(item)
        if policy == baseline_policy:
            continue
        if not relative_ok(development, baseline_dev):
            continue
        validation, _ = stage(rows, VALIDATION, policy)
        item["validation_2022h2_2023"] = validation
        item["validation_baseline"] = baseline_validation
        if not relative_ok(validation, baseline_validation):
            continue
        item["score"] = round(validation["return_pct"]-2*validation["dd_pct"], 8)
        item["_rows"] = rows
        survivors.append(item)
    survivors.sort(key=lambda item: item["score"], reverse=True)
    frozen = survivors[0] if survivors else None
    results = checks = None
    if frozen:
        pd.DataFrame(rows).to_pickle(ROWS_OUT)
        results, checks = {}, {}
        for periods in HIDDEN:
            name = periods[0]
            candidate, _ = stage(rows, periods, frozen["policy"])
            baseline, _ = stage(rows, periods, baseline_policy)
            results[name] = {"candidate": candidate, "baseline": baseline}
            checks[name] = hidden_ok(candidate, baseline)
            if not checks[name]:
                for sealed in ("2024", "2025", "2026"):
                    results.setdefault(sealed, None)
                    checks.setdefault(sealed, None)
                break
    passed = bool(checks and all(value is True for value in checks.values()))
    report = {
        "status": "CAPACITY_CHALLENGER_AWAITING_FORWARD" if passed else "REJECTED",
        "ready_for_runtime": False,
        "protocol": {
            "fixed_entry_model": MODEL, "fixed_probability_threshold": THRESHOLD,
            "fixed_exit": "5% at 0.5R; 95% at 2R or breakeven",
            "development": list(DEVELOPMENT), "validation": list(VALIDATION),
            "sealed_order": [item[0] for item in HIDDEN],
            "policies": list(POLICIES),
            "money": "3% current-equity margin, fixed 10x gross, 90% account cap",
        },
        "raw_selected_rows": len(rows), "survivors_before_hidden": len(survivors),
        "frozen_before_hidden": public(frozen),
        "frontier": [public(item) for item in frontier],
        "results": results, "checks": checks,
        "sources": {
            **{str(cache_path(target, period)): sha256(cache_path(target, period))
               for target in (.5, 2.0) for period in PERIODS},
            str(Path(__file__)): sha256(Path(__file__)),
        },
        "limitations": [
            "Capacity cannot create entry edge; it only admits more correlated setups.",
            "Binance history is a structural proxy and direct X-Perp forward remains required.",
            "Projected monthly return is a diagnostic and not a forecast.",
        ],
    }
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"status": report["status"],
                      "survivors_before_hidden": len(survivors),
                      "frozen": public(frozen), "results": results,
                      "checks": checks}, indent=2), flush=True)


if __name__ == "__main__":
    main()
