"""Replay the latest frozen crypto and stock routes on one shared account."""
from __future__ import annotations

import collections
import hashlib
import heapq
import json
import math
import os
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
STOCK_ROOT = Path(os.getenv("STOCK_BOT_REPO", ROOT.parent / "tusa stocks"))
CRYPTO_ROWS = ROOT / "reports/audit_2026_09_08/crypto_profit_first_exact_oct31_1700.pkl"
STOCK_ROWS = STOCK_ROOT / "reports/audit_2026_09_08/stock_xperp_regime_target_router_nov1_forward.csv"
REPORT = ROOT / "reports/audit_2026_09_08/shared_account_dual_bot_nov1.json"
LEVERAGE = 10.0
MAX_MARGIN = .90
SCENARIOS = ((.03, .03), (.03, .05), (.05, .03), (.05, .05))


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_rows() -> list[dict]:
    crypto = pd.read_pickle(CRYPTO_ROWS).copy()
    stocks = pd.read_csv(STOCK_ROWS).copy()
    crypto = crypto.assign(
        bot="crypto", signed_return=crypto.signed_price_return.astype(float),
        quality=crypto.model_probability.astype(float))
    stocks = stocks.assign(
        bot="stocks", signed_return=stocks.signed_return.astype(float),
        quality=stocks.cross_z.abs().astype(float))
    keep = ["bot", "symbol", "direction", "entry_time", "exit_time",
            "signed_return", "quality", "module"]
    both = pd.concat([crypto[keep], stocks[keep]], ignore_index=True)
    for key in ("entry_time", "exit_time", "signed_return", "quality"):
        both[key] = pd.to_numeric(both[key], errors="coerce")
    both = both.replace([math.inf, -math.inf], math.nan).dropna(
        subset=["entry_time", "exit_time", "signed_return"])
    both = both[both.exit_time.ge(both.entry_time)].copy()
    first = max(float(crypto.entry_time.min()), float(stocks.entry_time.min()))
    last = min(float(crypto.entry_time.max()), float(stocks.entry_time.max()))
    return both[both.entry_time.between(first, last)].to_dict("records")


def margin_for(row: dict, crypto_margin: float, stock_margin: float) -> float:
    return crypto_margin if row["bot"] == "crypto" else stock_margin


def shared_margin_gate(rows: list[dict], crypto_margin: float,
                       stock_margin: float) -> tuple[list[dict], dict]:
    grouped = collections.defaultdict(list)
    for row in rows:
        grouped[float(row["entry_time"])].append(row)
    accepted = []
    rejected = collections.Counter()
    active: list[tuple[float, int, dict, float]] = []
    sequence = 0
    peak_margin = 0.0
    peak_positions = 0
    peak_by_bot = collections.Counter()
    for now in sorted(grouped):
        while active and active[0][0] <= now:
            heapq.heappop(active)
        counts = collections.Counter(item[2]["bot"] for item in active)
        batch = sorted(grouped[now], key=lambda row: (
            counts[row["bot"]], -float(row.get("quality") or 0),
            str(row["bot"]), str(row["symbol"])))
        for row in batch:
            requested = margin_for(row, crypto_margin, stock_margin)
            occupied = sum(item[3] for item in active)
            if occupied + requested > MAX_MARGIN + 1e-12:
                rejected[row["bot"]] += 1
                continue
            accepted.append(row)
            sequence += 1
            heapq.heappush(active, (
                float(row["exit_time"]), sequence, row, requested))
            counts[row["bot"]] += 1
            occupied += requested
            peak_margin = max(peak_margin, occupied)
            peak_positions = max(peak_positions, len(active))
            peak_by_bot[row["bot"]] = max(
                peak_by_bot[row["bot"]], counts[row["bot"]])
    return accepted, {
        "candidate_count": len(rows),
        "accepted_count": len(accepted),
        "rejected_by_bot": dict(rejected),
        "peak_open_positions": peak_positions,
        "peak_margin_occupied_pct": round(100 * peak_margin, 4),
        "peak_open_by_bot": dict(peak_by_bot),
    }


def account_metrics(rows: list[dict], crypto_margin: float,
                    stock_margin: float) -> dict:
    ordered = sorted(rows, key=lambda row: (
        float(row["entry_time"]), str(row["bot"]), str(row["symbol"])))
    equity = peak = 1.0
    max_drawdown = 0.0
    sequence = 0
    exits: list[tuple[float, int, float, dict]] = []
    completed = []

    def realize(until: float | None) -> None:
        nonlocal equity, peak, max_drawdown
        while exits and (until is None or exits[0][0] <= until):
            stamp, _, pnl, row = heapq.heappop(exits)
            equity += pnl
            peak = max(peak, equity)
            max_drawdown = max(max_drawdown, (peak - equity) / peak)
            completed.append((stamp, pnl, row))

    for row in ordered:
        realize(float(row["entry_time"]))
        margin = margin_for(row, crypto_margin, stock_margin)
        pnl = equity * margin * LEVERAGE * float(row["signed_return"])
        sequence += 1
        heapq.heappush(exits, (
            float(row["exit_time"]), sequence, pnl, row))
    realize(None)
    completed.sort(key=lambda item: (
        item[0], item[2]["entry_time"], item[2]["symbol"]))

    ledger = 1.0
    month_start = {}
    month_end = {}
    month_bot_pnl = collections.defaultdict(lambda: collections.Counter())
    bot_n = collections.Counter()
    bot_wins = collections.Counter()
    bot_pnl = collections.Counter()
    for exit_time, pnl, row in completed:
        month = pd.Timestamp(exit_time, unit="s", tz="UTC").strftime("%Y-%m")
        month_start.setdefault(month, ledger)
        ledger += pnl
        month_end[month] = ledger
        bot = row["bot"]
        month_bot_pnl[month][bot] += pnl
        bot_pnl[bot] += pnl
        bot_n[bot] += 1
        bot_wins[bot] += int(pnl > 0)
    monthly = {}
    for month in sorted(month_start):
        start = month_start[month]
        crypto = 100 * month_bot_pnl[month]["crypto"] / start
        stocks = 100 * month_bot_pnl[month]["stocks"] / start
        monthly[month] = {
            "combined_return_pct": round(100 * (month_end[month] / start - 1), 4),
            "crypto_contribution_pct": round(crypto, 4),
            "stocks_contribution_pct": round(stocks, 4),
            "both_bots_ge_30pct": bool(crypto >= 30 and stocks >= 30),
            "combined_ge_60pct": bool(crypto + stocks >= 60),
        }
    return {
        "n": len(rows),
        "return_pct": round(100 * (equity - 1), 4),
        "max_drawdown_pct": round(100 * max_drawdown, 4),
        "by_bot": {
            bot: {
                "n": bot_n[bot],
                "profitable_exit_rate_pct": round(
                    100 * bot_wins[bot] / bot_n[bot], 4),
                "account_contribution_pct": round(100 * bot_pnl[bot], 4),
            } for bot in sorted(bot_n)
        },
        "monthly": monthly,
    }


def scenario(rows: list[dict], crypto_margin: float,
             stock_margin: float) -> dict:
    accepted, gate = shared_margin_gate(rows, crypto_margin, stock_margin)
    metrics = account_metrics(accepted, crypto_margin, stock_margin)
    complete = {month: values for month, values in metrics["monthly"].items()
                if month in {"2026-07", "2026-08", "2026-09", "2026-10"}}
    return {
        "crypto_margin_pct": 100 * crypto_margin,
        "stock_margin_pct": 100 * stock_margin,
        "gate": gate,
        "metrics": metrics,
        "complete_months_both_bots_ge_30": sum(
            value["both_bots_ge_30pct"] for value in complete.values()),
        "complete_months_combined_ge_60": sum(
            value["combined_ge_60pct"] for value in complete.values()),
    }


def main() -> None:
    rows = load_rows()
    scenarios = [scenario(rows, crypto, stocks)
                 for crypto, stocks in SCENARIOS]
    five = next(item for item in scenarios
                if item["crypto_margin_pct"] == 5
                and item["stock_margin_pct"] == 5)
    report = {
        "status": "SHARED_ACCOUNT_OBJECTIVE_NOT_PROVEN",
        "ready_for_live_orders": False,
        "orders_possible": False,
        "scope": {
            "first_entry_utc": pd.Timestamp(
                min(row["entry_time"] for row in rows), unit="s", tz="UTC").isoformat(),
            "last_entry_utc": pd.Timestamp(
                max(row["entry_time"] for row in rows), unit="s", tz="UTC").isoformat(),
            "leverage": LEVERAGE,
            "shared_margin_cap_pct": 100 * MAX_MARGIN,
            "candidate_counts": dict(collections.Counter(
                row["bot"] for row in rows)),
        },
        "scenarios": scenarios,
        "five_pct_each": five,
        "finding": (
            "Five percent per trade stays below the authorized shared margin "
            "ceiling in this overlap, but the crypto route does not contribute "
            "30% in every complete month."),
        "target": {
            "per_bot_monthly_pct": 30,
            "combined_monthly_pct": 60,
            "proven": False,
        },
        "sources": {
            str(CRYPTO_ROWS): digest(CRYPTO_ROWS),
            str(STOCK_ROWS): digest(STOCK_ROWS),
            str(Path(__file__).resolve()): digest(Path(__file__).resolve()),
        },
        "limitations": [
            "The common stock X-Perp history begins in late June 2026.",
            "July-August were used while selecting the stock route.",
            "Public OHLC cannot reproduce latency or order-book slippage.",
            "Gross returns exclude fees at user request.",
        ],
    }
    REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
