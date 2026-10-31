"""Audit CME Bitcoin CFTC positioning over frozen Profit-First trades."""
from __future__ import annotations

import hashlib
import heapq
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.research_profit_gate import economic_gate


ROOT = Path("reports/audit_2026_09_08")
CFTC = ROOT / "crypto_cftc_tff_history_oct31.json"
TRADES = ROOT / "crypto_profit_first_material_target_oct18.pkl"
OUT = ROOT / "crypto_cftc_tff_material_oct31.json"
ROWS_OUT = ROOT / "crypto_cftc_tff_material_oct31.pkl"
RULES = (
    {"name": "asset_manager_confirms", "am_min": 0.0},
    {"name": "asset_manager_strong", "am_min": 0.5},
    {"name": "leveraged_not_crowded", "lev_max": 1.0},
    {"name": "leveraged_contrarian", "lev_max": 0.0},
    {"name": "dealer_hedge_confirms", "dealer_max": 0.0},
    {"name": "other_reportable_confirms", "other_min": 0.0},
    {"name": "institutional_spread_confirms", "spread_min": 0.0},
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_cftc() -> pd.DataFrame:
    payload = json.loads(CFTC.read_text(encoding="utf-8"))
    if payload.get("status") != "IMMUTABLE_OFFICIAL_CFTC_RESEARCH_INPUT":
        raise ValueError("unexpected CFTC input status")
    frame = pd.DataFrame([
        row for row in payload["records"] if row["contract"] == "bitcoin_cme"])
    frame = frame.sort_values("available_at_utc").reset_index(drop=True)
    frame["available_time"] = (
        pd.to_datetime(frame.available_at_utc, utc=True).astype("int64")//10**9)
    oi = frame.open_interest.astype(float)
    definitions = (
        ("am", "asset_manager_long", "asset_manager_short"),
        ("lev", "leveraged_money_long", "leveraged_money_short"),
        ("dealer", "dealer_long", "dealer_short"),
        ("other", "other_reportable_long", "other_reportable_short"),
    )
    for prefix, long_col, short_col in definitions:
        frame[f"{prefix}_net"] = (
            frame[long_col].astype(float)-frame[short_col].astype(float))/oi
        mean = frame[f"{prefix}_net"].shift(1).rolling(52, min_periods=26).mean()
        std = frame[f"{prefix}_net"].shift(1).rolling(52, min_periods=26).std()
        frame[f"{prefix}_z"] = (
            frame[f"{prefix}_net"]-mean)/std.replace(0, np.nan)
    frame["spread_z"] = frame.am_z-frame.lev_z
    columns = ["available_time", "report_date", "am_z", "lev_z",
               "dealer_z", "other_z", "spread_z"]
    return frame[columns].dropna().sort_values("available_time")


def attach(trades: pd.DataFrame, cftc: pd.DataFrame) -> pd.DataFrame:
    output = pd.merge_asof(
        trades.sort_values("entry_time"), cftc,
        left_on="entry_time", right_on="available_time",
        direction="backward", allow_exact_matches=True)
    return output.dropna(subset=["am_z", "lev_z", "dealer_z", "other_z",
                                 "spread_z"])


def apply_rule(frame: pd.DataFrame, rule: dict) -> pd.DataFrame:
    direction = np.where(frame.direction.astype(str).str.upper().eq("LONG"), 1.0, -1.0)
    mask = pd.Series(True, index=frame.index)
    for key, column, sense in (
            ("am_min", "am_z", "min"), ("lev_max", "lev_z", "max"),
            ("dealer_max", "dealer_z", "max"),
            ("other_min", "other_z", "min"),
            ("spread_min", "spread_z", "min")):
        if key not in rule:
            continue
        signed = pd.Series(direction, index=frame.index)*frame[column]
        mask &= signed.ge(rule[key]) if sense == "min" else signed.le(rule[key])
    output = frame.loc[mask].copy()
    output["cftc_rule"] = rule["name"]
    return output


def account_metrics(frame: pd.DataFrame, margin: float = .03) -> dict:
    rows = frame.sort_values(["entry_time", "symbol"]).to_dict("records")
    if not rows:
        return {"n": 0, "wr": 0.0, "return_pct": 0.0, "dd_pct": 0.0,
                "negative_months": 0, "max_loss_streak": 0,
                "trades_per_calendar_day": 0.0, "monthly_return_pct": {}}
    equity = peak = 1.0
    maximum_dd = 0.0
    pending: list[tuple[float, int, float, dict]] = []
    completed: list[tuple[float, float, dict]] = []
    sequence = 0

    def realize(until: float | None) -> None:
        nonlocal equity, peak, maximum_dd
        while pending and (until is None or pending[0][0] < until):
            exit_time, _, pnl, row = heapq.heappop(pending)
            equity += pnl
            peak = max(peak, equity)
            maximum_dd = max(maximum_dd, (peak-equity)/peak)
            completed.append((exit_time, pnl, row))

    for row in rows:
        realize(float(row["entry_time"]))
        pnl = equity*margin*10*float(row["signed_price_return"])
        sequence += 1
        heapq.heappush(pending, (float(row["exit_time"]), sequence, pnl, row))
    realize(None)
    completed.sort(key=lambda item: (item[0], item[2]["entry_time"], item[2]["symbol"]))
    ledger = 1.0
    starts: dict[str, float] = {}
    ends: dict[str, float] = {}
    wins = streak = maximum_streak = targets = 0
    for exit_time, pnl, row in completed:
        won = pnl > 0
        wins += int(won)
        streak = 0 if won else streak+1
        maximum_streak = max(maximum_streak, streak)
        targets += int(str(row.get("barbell_outcome")) == "TP_RUNNER")
        month = pd.Timestamp(exit_time, unit="s", tz="UTC").strftime("%Y-%m")
        starts.setdefault(month, ledger)
        ledger += pnl
        ends[month] = ledger
    monthly = {month: 100*(ends[month]/starts[month]-1) for month in sorted(starts)}
    first = pd.Timestamp(min(float(row["entry_time"]) for row in rows), unit="s", tz="UTC")
    last = pd.Timestamp(max(float(row["entry_time"]) for row in rows), unit="s", tz="UTC")
    days = max(1, (last.date()-first.date()).days+1)
    tpd = len(rows)/days
    value = {
        "n": len(rows), "wr": round(100*wins/len(rows), 2),
        "return_pct": round(100*(equity-1), 4),
        "dd_pct": round(100*maximum_dd, 4),
        "negative_months": sum(value < -1e-12 for value in monthly.values()),
        "months": len(monthly),
        "monthly_return_pct": {key: round(value, 4) for key, value in monthly.items()},
        "minimum_month_pct": round(min(monthly.values()), 4) if monthly else 0.0,
        "max_loss_streak": maximum_streak,
        "trades_per_calendar_day": round(tpd, 3),
        "literal_runner_hits": targets,
        "literal_runner_hit_rate": round(100*targets/len(rows), 4),
    }
    value["economic_gate"] = economic_gate(
        [float(row["signed_price_return"]) for row in rows], tpd,
        market_days=30)
    return value


def stage(frame: pd.DataFrame, years: tuple[int, ...]) -> dict:
    return account_metrics(frame[frame.year.astype(int).isin(years)])


def development_ok(candidate: dict, baseline: dict) -> bool:
    return bool(candidate["n"] >= 150 and candidate["return_pct"] > 0
                and candidate["dd_pct"] <= baseline["dd_pct"]+1e-9
                and candidate["negative_months"] <= baseline["negative_months"]
                and candidate["economic_gate"]["projected_monthly_return_pct"]
                    >= baseline["economic_gate"]["projected_monthly_return_pct"])


def validation_ok(candidate: dict, baseline: dict) -> bool:
    return bool(candidate["n"] >= 100 and candidate["return_pct"] > 0
                and candidate["dd_pct"] <= baseline["dd_pct"]+1e-9
                and candidate["negative_months"] <= baseline["negative_months"]
                and candidate["economic_gate"]["return_pass"])


def hidden_ok(value: dict) -> bool:
    return bool(value["n"] >= 100 and value["return_pct"] > 0
                and value["dd_pct"] <= 5 and value["max_loss_streak"] <= 3
                and value["economic_gate"]["return_pass"])


def public(item: dict | None) -> dict | None:
    return ({key: value for key, value in item.items() if key != "_rows"}
            if item else None)


def main() -> None:
    source = pd.read_pickle(TRADES)
    enriched = attach(source, load_cftc())
    baseline_dev = stage(enriched, (2022,))
    baseline_validation = stage(enriched, (2023,))
    frontier, survivors = [], []
    for rule in RULES:
        rows = apply_rule(enriched, rule)
        dev = stage(rows, (2022,))
        item = {"rule": rule, "development_2022": dev,
                "development_baseline": baseline_dev}
        frontier.append(item)
        if not development_ok(dev, baseline_dev):
            continue
        validation = stage(rows, (2023,))
        item["validation_2023"] = validation
        item["validation_baseline"] = baseline_validation
        if not validation_ok(validation, baseline_validation):
            continue
        item["score"] = round(
            validation["economic_gate"]["projected_monthly_return_pct"]
            -2*validation["dd_pct"]-validation["negative_months"], 8)
        item["_rows"] = rows
        survivors.append(item)
    survivors.sort(key=lambda item: item["score"], reverse=True)
    frontier.sort(key=lambda item: (
        item["development_2022"]["economic_gate"]["projected_monthly_return_pct"],
        -item["development_2022"]["dd_pct"]), reverse=True)
    frozen = survivors[0] if survivors else None
    results = checks = None
    if frozen:
        rows = frozen["_rows"]
        rows.to_pickle(ROWS_OUT)
        y2024 = stage(rows, (2024,))
        results = {"test_2024": y2024, "test_2025": None, "test_2026": None}
        checks = {"test_2024": hidden_ok(y2024), "test_2025": None, "test_2026": None}
        if checks["test_2024"]:
            y2025 = stage(rows, (2025,))
            results["test_2025"] = y2025
            checks["test_2025"] = hidden_ok(y2025)
            if checks["test_2025"]:
                y2026 = stage(rows, (2026,))
                results["test_2026"] = y2026
                checks["test_2026"] = hidden_ok(y2026)
    passed = bool(checks and all(value is True for value in checks.values()))
    report = {
        "status": "RESEARCH_CANDIDATE_AWAITING_FORWARD" if passed else "REJECTED",
        "ready_for_runtime": False,
        "protocol": {
            "source": "official CFTC TFF CME Bitcoin futures-only positions",
            "contract_code": "133741",
            "availability": "Tuesday report becomes usable seven full days later",
            "base": "frozen Profit-First accepted historical trades",
            "development": 2022, "validation": 2023,
            "sealed_order": [2024, 2025, 2026],
            "fixed_rules": list(RULES),
            "money": "3% current-equity margin, fixed 10x gross",
        },
        "source_trade_rows": len(source), "causally_enriched_rows": len(enriched),
        "survivors_before_hidden": len(survivors),
        "frozen_before_hidden": public(frozen),
        "frontier": [public(item) for item in frontier],
        "results": results, "checks": checks,
        "sources": {str(CFTC): sha256(CFTC), str(TRADES): sha256(TRADES),
                    str(Path(__file__)): sha256(Path(__file__))},
        "limitations": [
            "The overlay can remove accepted trades but does not restore trades that prior runtime rails blocked.",
            "CFTC data are weekly and cannot time intraday entries.",
            "CME institutional positioning is a regime proxy for Binance/X-Perp prices.",
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
