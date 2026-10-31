"""Audit both shadow strategies as one 3%-margin, 10x account.

The Railway services have separate databases, so their local direction and
open-risk limits do not form an account-wide limit.  This replay adds only a
shared admission gate; it does not change either strategy or place orders.
"""
from __future__ import annotations

import collections
import heapq
import json
import math
import os
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
STOCK_ROOT = Path(os.getenv("STOCK_BOT_REPO", ROOT.parent / "tusa stocks"))
CRYPTO_ROWS = ROOT / "reports/audit_2026_09_08/crypto_profit_first_exact_oct30_2030.pkl"
STOCK_ROWS = STOCK_ROOT / "reports/audit_2026_09_08/stock_xperp_regime_target_router_oct30_2000_forward.csv"
REPORT = ROOT / "reports/audit_2026_09_08/shared_account_dual_bot_oct31.json"
MARGIN_FRACTION = 0.03
LEVERAGE = 10.0


def load_rows() -> list[dict]:
    crypto = pd.read_pickle(CRYPTO_ROWS).copy()
    stocks = pd.read_csv(STOCK_ROWS).copy()
    crypto = crypto.assign(
        bot="crypto", signed_return=crypto["signed_price_return"].astype(float),
        quality=crypto["model_probability"].astype(float),
    )
    # Stock candidates are already admitted by their frozen router.  cross_z is
    # used only to order simultaneous candidates from the same bot.
    stocks = stocks.assign(
        bot="stocks", signed_return=stocks["signed_return"].astype(float),
        quality=stocks["cross_z"].abs().astype(float),
    )
    keep = ["bot", "symbol", "direction", "entry_time", "exit_time",
            "signed_return", "quality", "module"]
    both = pd.concat([crypto[keep], stocks[keep]], ignore_index=True)
    for col in ("entry_time", "exit_time", "signed_return", "quality"):
        both[col] = pd.to_numeric(both[col], errors="coerce")
    both = both.replace([math.inf, -math.inf], math.nan).dropna(
        subset=["entry_time", "exit_time", "signed_return"])
    both = both[both.exit_time.ge(both.entry_time)].copy()

    # Compare only dates for which both frozen streams exist.
    first = max(crypto.entry_time.min(), stocks.entry_time.min())
    last = min(crypto.entry_time.max(), stocks.entry_time.max())
    both = both[both.entry_time.between(first, last, inclusive="both")]
    return both.to_dict("records")


def _ordered(rows: list[dict], priority: str) -> list[dict]:
    def key(row: dict) -> tuple:
        if priority == "crypto_first":
            bot_rank = 0 if row["bot"] == "crypto" else 1
        elif priority == "stocks_first":
            bot_rank = 0 if row["bot"] == "stocks" else 1
        else:
            bot_rank = 0
        return (float(row["entry_time"]), bot_rank,
                -float(row.get("quality") or 0), str(row["symbol"]))
    return sorted(rows, key=key)


def shared_gate(rows: list[dict], max_total: int | None,
                max_direction: int | None, max_hour: int | None,
                priority: str = "balanced") -> tuple[list[dict], dict]:
    """Admit rows against positions from both services.

    ``balanced`` gives a simultaneous slot to the bot with fewer currently
    open positions.  The two fixed priority variants expose ordering risk.
    """
    accepted: list[dict] = []
    rejected = collections.Counter()
    open_heap: list[tuple[float, int, dict]] = []
    recent: collections.deque[float] = collections.deque()
    sequence = 0
    peak_total = peak_same_direction = 0
    grouped = collections.defaultdict(list)
    for row in rows:
        grouped[float(row["entry_time"])].append(row)

    for now in sorted(grouped):
        while open_heap and open_heap[0][0] < now:
            heapq.heappop(open_heap)
        while recent and recent[0] <= now - 3600:
            recent.popleft()
        batch = _ordered(grouped[now], priority)
        if priority == "balanced":
            counts = collections.Counter(item[2]["bot"] for item in open_heap)
            batch.sort(key=lambda row: (counts[row["bot"]],
                                        -float(row.get("quality") or 0),
                                        str(row["bot"]), str(row["symbol"])))
        for row in batch:
            directions = collections.Counter(item[2]["direction"] for item in open_heap)
            if max_total is not None and len(open_heap) >= max_total:
                rejected["total_cap"] += 1
                continue
            if max_direction is not None and directions[row["direction"]] >= max_direction:
                rejected["direction_cap"] += 1
                continue
            if max_hour is not None and len(recent) >= max_hour:
                rejected["rolling_hour_cap"] += 1
                continue
            accepted.append(row)
            recent.append(now)
            sequence += 1
            heapq.heappush(open_heap, (float(row["exit_time"]), sequence, row))
            directions[row["direction"]] += 1
            peak_total = max(peak_total, len(open_heap))
            peak_same_direction = max(peak_same_direction, max(directions.values()))
    return accepted, {
        "candidate_count": len(rows), "accepted_count": len(accepted),
        "rejected": dict(sorted(rejected.items())),
        "peak_open_positions": peak_total,
        "peak_margin_occupied_pct_at_3pct_each": round(peak_total * 3.0, 4),
        "peak_same_direction": peak_same_direction,
    }


def account_metrics(rows: list[dict]) -> dict:
    equity = peak = 1.0
    max_dd = 0.0
    open_exits: list[tuple[float, int, float, dict]] = []
    completed: list[tuple[float, float, dict]] = []
    sequence = 0

    def realize(until: float | None) -> None:
        nonlocal equity, peak, max_dd
        while open_exits and (until is None or open_exits[0][0] < until):
            exit_time, _, pnl, row = heapq.heappop(open_exits)
            equity += pnl
            peak = max(peak, equity)
            max_dd = max(max_dd, (peak - equity) / peak)
            completed.append((exit_time, pnl, row))

    for row in _ordered(rows, "balanced"):
        realize(float(row["entry_time"]))
        pnl = equity * MARGIN_FRACTION * LEVERAGE * float(row["signed_return"])
        sequence += 1
        heapq.heappush(open_exits, (float(row["exit_time"]), sequence, pnl, row))
    realize(None)
    completed.sort(key=lambda item: (item[0], item[2]["entry_time"], item[2]["symbol"]))

    month_start: dict[str, float] = {}
    month_end: dict[str, float] = {}
    month_bot_pnl = collections.defaultdict(lambda: collections.Counter())
    bot_pnl = collections.Counter()
    bot_n = collections.Counter()
    bot_wins = collections.Counter()
    ledger = 1.0
    loss_streak = max_loss_streak = 0
    for exit_time, pnl, row in completed:
        month = pd.Timestamp(exit_time, unit="s", tz="UTC").strftime("%Y-%m")
        month_start.setdefault(month, ledger)
        ledger += pnl
        month_end[month] = ledger
        bot = row["bot"]
        bot_pnl[bot] += pnl
        month_bot_pnl[month][bot] += pnl
        bot_n[bot] += 1
        bot_wins[bot] += int(pnl > 0)
        loss_streak = 0 if pnl > 0 else loss_streak + 1
        max_loss_streak = max(max_loss_streak, loss_streak)

    monthly = {}
    for month in sorted(month_start):
        start = month_start[month]
        monthly[month] = {
            "combined_return_pct": round(100 * (month_end[month] / start - 1), 4),
            "crypto_contribution_pct": round(100 * month_bot_pnl[month]["crypto"] / start, 4),
            "stocks_contribution_pct": round(100 * month_bot_pnl[month]["stocks"] / start, 4),
        }
    return {
        "n": len(rows), "return_pct": round(100 * (equity - 1), 4),
        "max_drawdown_pct": round(100 * max_dd, 4),
        "max_loss_streak": max_loss_streak,
        "by_bot": {
            bot: {"n": bot_n[bot],
                  "profitable_exit_rate_pct": round(100 * bot_wins[bot] / bot_n[bot], 3),
                  "account_contribution_pct": round(100 * bot_pnl[bot], 4)}
            for bot in sorted(bot_n)
        },
        "monthly": monthly,
    }


def evaluate(rows: list[dict], max_total: int | None,
             max_direction: int | None, max_hour: int | None,
             priority: str) -> dict:
    accepted, gate = shared_gate(rows, max_total, max_direction, max_hour, priority)
    return {"limits": {"max_total": max_total, "max_direction": max_direction,
                       "max_rolling_hour": max_hour, "priority": priority},
            "gate": gate, "metrics": account_metrics(accepted)}


def main() -> None:
    rows = load_rows()
    variants = [evaluate(rows, None, None, None, "balanced")]
    for total in (3, 4, 5, 6):
        for direction in (2, 3, 4):
            if direction > total:
                continue
            for hourly in (2, 3, 4, 6):
                for priority in ("balanced", "crypto_first", "stocks_first"):
                    variants.append(evaluate(rows, total, direction, hourly, priority))
    policy_groups = collections.defaultdict(list)
    for item in variants[1:]:
        limits = item["limits"]
        key = (limits["max_total"], limits["max_direction"],
               limits["max_rolling_hour"])
        policy_groups[key].append(item)
    robustness = []
    for (total, direction, hourly), items in policy_groups.items():
        october = [item["metrics"]["monthly"]["2026-10"] for item in items]
        robustness.append({
            "limits": {"max_total": total, "max_direction": direction,
                       "max_rolling_hour": hourly},
            "worst_total_return_pct": min(item["metrics"]["return_pct"] for item in items),
            "worst_october_combined_pct": min(item["combined_return_pct"] for item in october),
            "worst_october_crypto_contribution_pct": min(item["crypto_contribution_pct"] for item in october),
            "worst_october_stocks_contribution_pct": min(item["stocks_contribution_pct"] for item in october),
            "worst_drawdown_pct": max(item["metrics"]["max_drawdown_pct"] for item in items),
        })
    robustness.sort(key=lambda item: (-item["worst_total_return_pct"],
                                      item["worst_drawdown_pct"]))
    frontier = sorted(variants[1:], key=lambda item: (
        item["metrics"]["max_drawdown_pct"], -item["metrics"]["return_pct"]))[:20]
    by_return = sorted(variants[1:], key=lambda item: (
        -item["metrics"]["return_pct"], item["metrics"]["max_drawdown_pct"]))[:20]
    baseline = variants[0]
    authorized_margin_policy = evaluate(rows, 30, None, None, "balanced")
    report = {
        "scope": {
            "first_entry_utc": pd.Timestamp(min(r["entry_time"] for r in rows), unit="s", tz="UTC").isoformat(),
            "last_entry_utc": pd.Timestamp(max(r["entry_time"] for r in rows), unit="s", tz="UTC").isoformat(),
            "margin_fraction": MARGIN_FRACTION, "leverage": LEVERAGE,
            "authorized_total_margin_pct": 90.0,
            "authorized_max_simultaneous_3pct_positions": 30,
            "candidate_counts": dict(collections.Counter(r["bot"] for r in rows)),
            "fees_and_funding_included": False,
            "period_interpretation": {
                "2026-06": "partial overlap beginning June 28; not a full month",
                "2026-07": "stock router selection period",
                "2026-08": "stock router selection period",
                "2026-09": "stock router held-out test period",
                "2026-10": "confirmation plus short append-only forward period",
            },
        },
        "finding": "Separate Railway databases cannot enforce a shared account limit.",
        "ungated_baseline": baseline,
        "authorized_90pct_margin_policy": authorized_margin_policy,
        "authorized_margin_cap_binding_in_sample": bool(
            authorized_margin_policy["gate"]["rejected"]),
        "lowest_drawdown_frontier": frontier,
        "highest_return_frontier": by_return,
        "priority_robust_policies": robustness[:20],
        "target": {
            "per_bot_monthly_pct": 30.0, "combined_monthly_pct": 60.0,
            "proven": False,
            "october_grid_policies_meeting_each_bot_target": sum(
                item["worst_october_crypto_contribution_pct"] >= 30
                and item["worst_october_stocks_contribution_pct"] >= 30
                for item in robustness),
            "october_grid_policies_meeting_combined_target": sum(
                item["worst_october_combined_pct"] >= 60 for item in robustness),
            "reason": "No tested common-cap policy reaches the requested target, and overlapping selection plus short forward data cannot prove a monthly guarantee.",
        },
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"report": str(REPORT), "baseline": baseline,
                      "best_return": by_return[0], "lowest_dd": frontier[0]},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
