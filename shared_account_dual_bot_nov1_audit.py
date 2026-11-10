"""Replay the latest frozen crypto and stock routes on one shared account."""
from __future__ import annotations

from bisect import bisect_left
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
MAX_POSITIONS = 12
STOCK_CRYPTO_CLUSTER = {"COINUSDT", "MSTRUSDT", "CRCLUSDT"}
LOCAL_SLOTS = {"crypto": 4, "stocks": 8}
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
    stocks = stocks.assign(sequential_banked=False, near_exit_time=math.nan)
    keep = ["bot", "symbol", "direction", "entry_time", "exit_time",
            "signed_return", "quality", "module", "entry", "risk_pct",
            "sequential_banked", "near_exit_time"]
    both = pd.concat([crypto[keep], stocks[keep]], ignore_index=True)
    for key in ("entry_time", "exit_time", "signed_return", "quality"):
        both[key] = pd.to_numeric(both[key], errors="coerce")
    both = both.replace([math.inf, -math.inf], math.nan).dropna(
        subset=["entry_time", "exit_time", "signed_return"])
    both = both[both.exit_time.ge(both.entry_time)].copy()
    first = max(float(crypto.entry_time.min()), float(stocks.entry_time.min()))
    last = min(float(crypto.entry_time.max()), float(stocks.entry_time.max()))
    return both[both.entry_time.between(first, last)].to_dict("records")


def filter_crypto_shorts(rows: list[dict]) -> list[dict]:
    """Use completed shadow shorts, including those previously filtered out."""
    shorts = [row for row in rows
              if row["bot"] == "crypto" and row["direction"] == "SHORT"]
    outcomes = sorted((row["exit_time"], row["signed_return"]) for row in shorts)
    times = [stamp for stamp, _ in outcomes]
    blocked = set()
    for row in shorts:
        entry = row["entry_time"]
        start = bisect_left(times, entry - 14 * 86400)
        end = bisect_left(times, entry)
        if end - start >= 15 and sum(value for _, value in outcomes[start:end]) < 0:
            blocked.add(id(row))
    return [row for row in rows if id(row) not in blocked]


def margin_for(row: dict, crypto_margin: float, stock_margin: float) -> float:
    return crypto_margin if row["bot"] == "crypto" else stock_margin


def shared_margin_gate(rows: list[dict], crypto_margin: float,
                       stock_margin: float, *, max_shorts: int | None = None,
                       max_per_direction: int | None = None,
                       btc_closes: dict | None = None,
                       model_live_slots: bool = False,
                       limit_stock_crypto_cluster: bool = False,
                       position_writers: int = 1, margin_writers: int = 1,
                       slot_release_delay: float = 0.0
                       ) -> tuple[list[dict], dict]:
    if max_shorts is not None and max_shorts < 0:
        raise ValueError('max_shorts must be nonnegative')
    if max_per_direction is not None and max_per_direction < 1:
        raise ValueError('max_per_direction must be positive')
    if position_writers < 1:
        raise ValueError('position_writers must be positive')
    if margin_writers < 1:
        raise ValueError('margin_writers must be positive')
    if not 0 <= slot_release_delay < float("inf"):
        raise ValueError("slot_release_delay must be finite and nonnegative")
    grouped = collections.defaultdict(list)
    for row in rows:
        grouped[float(row["entry_time"])].append(row)
    accepted = []
    rejected = collections.Counter()
    rejected_positions = collections.Counter()
    rejected_positions_with_other_bot = collections.Counter()
    rejected_margin = collections.Counter()
    rejected_instrument = collections.Counter()
    rejected_shorts = collections.Counter()
    rejected_direction = collections.Counter()
    rejected_local = collections.Counter()
    rejected_cluster = 0
    missing_btc_context = collections.Counter()
    active: list[tuple[float, int, dict, float]] = []
    exits: list[tuple[float, int, float]] = []
    sequence = 0
    equity = 1.0
    peak_margin = 0.0
    peak_positions = 0
    peak_by_bot = collections.Counter()
    for now in sorted(grouped):
        # Equal timestamps do not prove that the exit preceded the entry.
        while active and active[0][0] < now:
            heapq.heappop(active)
        while exits and exits[0][0] < now:
            _, _, pnl = heapq.heappop(exits)
            equity += pnl
        counts = collections.Counter(item[2]["bot"] for item in active)
        batch = sorted(grouped[now], key=lambda row: (
            counts[row["bot"]], -float(row.get("quality") or 0),
            str(row["bot"]), str(row["symbol"])))
        for row in batch:
            if model_live_slots and counts[row["bot"]] >= LOCAL_SLOTS[row["bot"]]:
                rejected[row["bot"]] += 1
                rejected_local[row["bot"]] += 1
                continue
            apply_direction_cap = True
            if btc_closes is not None:
                stamp = (math.ceil(now / 900) - 1) * 900
                current = btc_closes.get(stamp)
                previous = btc_closes.get(stamp - 86400)
                if current is None or previous is None:
                    missing_btc_context[row["bot"]] += 1
                    apply_direction_cap = False
                else:
                    sign = 1 if row["direction"] == "LONG" else -1
                    apply_direction_cap = sign * (current / previous - 1) < -.01
            requested = equity * margin_for(row, crypto_margin, stock_margin)
            occupied = sum(item[3] for item in active)
            instrument = row.get("inst_id") or row["symbol"]
            if any((item[2].get("inst_id") or item[2]["symbol"]) == instrument
                   for item in active):
                rejected[row["bot"]] += 1
                rejected_instrument[row["bot"]] += 1
                continue
            if len(active) + position_writers > MAX_POSITIONS:
                rejected[row["bot"]] += 1
                rejected_positions[row["bot"]] += 1
                if counts[row["bot"]] + position_writers <= MAX_POSITIONS:
                    rejected_positions_with_other_bot[row["bot"]] += 1
                continue
            if (max_shorts is not None and row["direction"] == "SHORT"
                    and sum(item[2]["direction"] == "SHORT"
                            for item in active) >= max_shorts):
                rejected[row["bot"]] += 1
                rejected_shorts[row["bot"]] += 1
                continue
            if (max_per_direction is not None and apply_direction_cap
                    and sum(item[2]["direction"] == row["direction"]
                            for item in active) >= max_per_direction):
                rejected[row["bot"]] += 1
                rejected_direction[row["bot"]] += 1
                continue
            if equity <= 0 or occupied + requested * margin_writers > equity * MAX_MARGIN + 1e-12:
                rejected[row["bot"]] += 1
                rejected_margin[row["bot"]] += 1
                continue
            if (limit_stock_crypto_cluster and row["bot"] == "stocks"
                    and row["direction"] == "SHORT" and row["symbol"] in STOCK_CRYPTO_CLUSTER
                    and any(other["bot"] == "stocks" and other["direction"] == "SHORT"
                            and other["symbol"] in STOCK_CRYPTO_CLUSTER
                            for _, _, other, _ in active)):
                rejected["stocks"] += 1
                rejected_cluster += 1
                continue
            accepted.append(row)
            sequence += 1
            exit_time = float(row["exit_time"])
            heapq.heappush(active, (
                exit_time + slot_release_delay, sequence, row, requested))
            heapq.heappush(exits, (
                exit_time, sequence, requested * LEVERAGE * float(row["signed_return"])))
            counts[row["bot"]] += 1
            occupied += requested
            peak_margin = max(peak_margin, occupied / equity)
            peak_positions = max(peak_positions, len(active))
            peak_by_bot[row["bot"]] = max(
                peak_by_bot[row["bot"]], counts[row["bot"]])
    return accepted, {
        "candidate_count": len(rows),
        "position_writers": position_writers,
        "margin_writers": margin_writers,
        "slot_release_delay": slot_release_delay,
        "max_per_direction": max_per_direction,
        "accepted_count": len(accepted),
        "candidate_count_by_bot": dict(collections.Counter(row["bot"] for row in rows)),
        "accepted_count_by_bot": dict(collections.Counter(row["bot"] for row in accepted)),
        "rejected_by_bot": dict(rejected),
        "rejected_by_margin_cap": dict(rejected_margin),
        "rejected_by_open_instrument": dict(rejected_instrument),
        "missing_btc_context": dict(missing_btc_context),
        "rejected_by_stock_crypto_cluster": rejected_cluster,
        "rejected_by_local_slots": dict(rejected_local),
        "rejected_by_direction_cap": dict(rejected_direction),
        "rejected_by_short_cap": dict(rejected_shorts),
        "rejected_by_position_cap": dict(rejected_positions),
        "position_rejections_with_other_bot_open": dict(
            rejected_positions_with_other_bot),
        "peak_open_positions": peak_positions,
        "peak_margin_occupied_pct": round(100 * peak_margin, 4),
        "peak_open_by_bot": dict(peak_by_bot),
    }


def candle_close_drawdown(rows: list[dict], close_prices: dict,
                          crypto_margin: float, stock_margin: float,
                          *, slippage: float = 0.0, adverse_prices: dict | None = None) -> float:
    """Estimate drawdown using completed 15m candles and modeled trade exits."""
    equity = 1.0
    exits = []
    changes = collections.defaultdict(float)
    stress_adjustments = collections.defaultdict(float)
    for sequence, row in enumerate(sorted(rows, key=lambda item: item["entry_time"])):
        entry_time = row["entry_time"]
        exit_time = row["exit_time"]
        while exits and exits[0][0] < entry_time:
            _, _, pnl = heapq.heappop(exits)
            equity += pnl
        notional = equity * margin_for(row, crypto_margin, stock_margin) * LEVERAGE
        pnl = notional * row["signed_return"]
        heapq.heappush(exits, (exit_time, sequence, pnl))
        prices = close_prices[row["bot"], row["symbol"]]
        sign = 1 if row["direction"] == "LONG" else -1
        previous_mark = 0.0
        for stamp in range((int(entry_time) // 900 + 1) * 900, int(exit_time), 900):
            move = sign * (prices[stamp] / row["entry"] - 1)
            remaining = 1.0
            if (row["bot"] == "crypto" and row["sequential_banked"]
                    and row["near_exit_time"] + 900 <= stamp):
                remaining = .95
                move = remaining * move + .025 * row["risk_pct"]
            if adverse_prices is not None:
                adverse = adverse_prices[row["bot"], row["symbol"], row["direction"]][stamp]
                stress_adjustments[stamp] += min(
                    0.0, notional * remaining * sign * (adverse - prices[stamp]) / row["entry"])
            mark = notional * (move - slippage)
            changes[stamp] += mark - previous_mark
            previous_mark = mark
        changes[exit_time] += pnl - previous_mark
    equity = peak = 1.0
    drawdown = 0.0
    for stamp in sorted(changes):
        equity += changes[stamp]
        stressed_equity = equity + stress_adjustments[stamp]
        drawdown = max(drawdown, (peak - stressed_equity) / peak)
        peak = max(peak, equity)
    return 100 * drawdown


def load_candle_marks(rows: list[dict]) -> tuple[dict, dict]:
    folders = {
        "crypto": CRYPTO_ROWS.parent / "xperp_15m_oct31_1700_material",
        "stocks": STOCK_ROWS.parent / "stock_xperp_forward_nov1",
    }
    closes = {}
    adverse = {}
    for bot, symbol in sorted({(row["bot"], row["symbol"]) for row in rows}):
        candles = pd.read_pickle(folders[bot] / f"{symbol}_15min_current.pkl")
        close_times = [int(stamp) + 900 for stamp in candles["time"]]
        closes[bot, symbol] = dict(zip(close_times, candles["close"], strict=True))
        for direction, field in (("LONG", "low"), ("SHORT", "high")):
            adverse[bot, symbol, direction] = dict(
                zip(close_times, candles[field], strict=True))
    return closes, adverse


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
        while exits and (until is None or exits[0][0] < until):
            stamp = exits[0][0]
            # Exit timestamps do not establish an order within the batch.
            while exits and exits[0][0] == stamp:
                _, _, pnl, row = heapq.heappop(exits)
                equity += pnl
                completed.append((stamp, pnl, row))
            peak = max(peak, equity)
            max_drawdown = max(max_drawdown, (peak - equity) / peak)

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
        "drawdown_basis": "realized_equity_after_exit_batches",
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
             stock_margin: float, *, max_per_direction: int | None = None,
             slot_release_delay: float = 0.0,
             candle_marks: tuple[dict, dict] | None = None,
             btc_closes: dict | None = None, model_live_slots: bool = False,
             limit_stock_crypto_cluster: bool = False) -> dict:
    accepted, gate = shared_margin_gate(
        rows, crypto_margin, stock_margin, max_per_direction=max_per_direction,
        slot_release_delay=slot_release_delay, btc_closes=btc_closes,
        model_live_slots=model_live_slots,
        limit_stock_crypto_cluster=limit_stock_crypto_cluster,
        position_writers=2 if model_live_slots else 1,
        margin_writers=2 if model_live_slots else 1,
    )
    metrics = account_metrics(accepted, crypto_margin, stock_margin)
    if candle_marks is not None:
        closes, adverse = candle_marks
        metrics["candle_close_drawdown_pct"] = round(candle_close_drawdown(
            accepted, closes, crypto_margin, stock_margin), 4)
        metrics["candle_adverse_stress_drawdown_pct"] = round(candle_close_drawdown(
            accepted, closes, crypto_margin, stock_margin, adverse_prices=adverse), 4)
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
    candle_marks = load_candle_marks(rows)
    short_filtered = filter_crypto_shorts(rows)
    scenarios = [scenario(rows, crypto, stocks, candle_marks=candle_marks)
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
            "shared_position_cap": MAX_POSITIONS,
            "candidate_counts": dict(collections.Counter(
                row["bot"] for row in rows)),
        },
        "scenarios": scenarios,
        "five_pct_each": five,
        "crypto_short_health": {
            "filtered_signals": len(rows) - len(short_filtered),
            "scenarios": [
                scenario(short_filtered, margin, margin, slot_release_delay=900,
                         model_live_slots=True, candle_marks=candle_marks,
                         max_per_direction=4 if use_btc else None,
                         btc_closes=candle_marks[0]["crypto", "BTCUSDT"] if use_btc else None)
                for margin in (.03, .05) for use_btc in (False, True)
            ],
        },
        "stock_crypto_cluster": [
            scenario(data, margin, margin, slot_release_delay=900,
                     candle_marks=candle_marks, model_live_slots=True,
                     limit_stock_crypto_cluster=True)
            for margin in (.03, .05) for data in (rows, short_filtered)
        ],
        "live_slot_sensitivity": [
            scenario(rows, margin, margin, slot_release_delay=delay,
                     candle_marks=candle_marks, model_live_slots=True)
            for margin in (.03, .05) for delay in (0, 900, 3600)
        ],
        "btc_countertrend_cap": [
            scenario(rows, margin, margin, max_per_direction=4,
                     slot_release_delay=delay, candle_marks=candle_marks,
                     btc_closes=candle_marks[0]["crypto", "BTCUSDT"])
            for margin in (.03, .05) for delay in (0, 900, 3600)
        ],
        "direction_cap_sensitivity": [
            scenario(rows, margin, margin, max_per_direction=cap,
                     slot_release_delay=delay, candle_marks=candle_marks)
            for margin in (.03, .05) for delay in (0, 900, 3600)
            for cap in (None, 4, 5)
        ],
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
