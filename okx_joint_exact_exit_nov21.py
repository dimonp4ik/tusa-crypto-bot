"""Compare archived executable exits with exit-price stress on a common account."""

from collections import Counter
from datetime import datetime, timezone
import json
import pickle
from statistics import median

from okx_cross_bot_overlap_nov21 import DATA, START, crypto_cadence, joint_inputs, overlap_view
from okx_xperp_cash_tick_nov19 import simulate
from okx_xperp_multiday_capture_nov20 import HOLD, file_hash
from okx_xperp_multiday_replay_nov20 import exit_books
from stock_liquid_quote_routes_nov21 import ROOT, limit_bursts, write_json


AVAILABLE_BOOK_END = int(datetime(2026, 11, 20, tzinfo=timezone.utc).timestamp())
ENTRY_END = AVAILABLE_BOOK_END - max(HOLD.values())
CAPS = (4, 6, 12)


def choose_rows(crypto, stocks):
    case = [r for r in crypto + stocks if START <= r["epoch"] < ENTRY_END]
    if {r["bot"] for r in case} != {"stocks", "crypto"}:
        raise ValueError("joint quote window lacks one route")
    if any(r["epoch"] + HOLD[r["bot"]] + 1 > AVAILABLE_BOOK_END for r in case):
        raise ValueError("candidate life extends outside the archive window")
    return case


def matched_exit_cost(book, stress, directions):
    if "closed" not in book or "closed" not in stress:
        return {}
    by_key = {(r["symbol"], r["entry_ms"], r["exit_ms"]): r for r in stress["closed"]}
    costs = {"crypto": [], "stocks": []}
    for trade in book["closed"]:
        key = trade["symbol"], trade["entry_ms"], trade["exit_ms"]
        other = by_key.get(key)
        if other is None or trade["runner_px"] is None or other["runner_px"] is None:
            continue
        sign = 1 if directions[trade["symbol"], trade["entry_ms"]] == "LONG" else -1
        last_trade = other["runner_px"] / (1 - sign * .0015)
        costs[trade["bot"]].append(10_000 * sign * (1 - trade["runner_px"] / last_trade))
    return {bot: dict(n=len(values), median_bps=median(values), worst_bps=max(values),
                      above_15bps=sum(v > 15 for v in values))
            for bot, values in costs.items() if values}


def score(rows, specs, ticks, books, exit_proxy):
    crypto = list(crypto_cadence(r for r in rows if r["bot"] == "crypto"))
    stocks = [r for r in rows if r["bot"] == "stocks"]
    patterns = {
        "crypto_only": crypto,
        "both_base_cadence": crypto + list(limit_bursts(stocks, max_same_side=2)),
        "both_one_per_side_hour": crypto + list(limit_bursts(stocks, max_same_side=1)),
    }
    directions = {(r["symbol"], r["entry_ms"]): r["direction"] for r in rows}
    cases = {}
    for label, proposals in patterns.items():
        for cap in CAPS if label.startswith("both") else (12,):
            for setting in ("requested5", "half_minimum"):
                for lag in (200, 1000):
                    exact = simulate(proposals, specs, ticks, books, clock=60, lag=lag, setting=setting,
                                     max_positions=cap)
                    stress = simulate(proposals, specs, ticks, books, clock=60, lag=lag, setting=setting,
                                      max_positions=cap, exit_price=exit_proxy(15))
                    for result in (exact, stress):
                        if "closed" in result:
                            result.update(overlap_view(result["closed"]))
                    if "closed" in stress:
                        stress["status"] = "SCORED_PUBLIC_TRADE_EXIT_STRESS"
                    cases[f"{label}|cap{cap}|{setting}|{lag}ms"] = dict(
                        book=exact, stress15bps=stress, matched_runner_cost=matched_exit_cost(exact, stress, directions))
    return cases


def main():
    (crypto, stocks, specs, ticks, books, hashes, entry_checks, tapes, exit_proxy, unknown) = joint_inputs()
    rows = choose_rows(crypto, stocks)
    if any(START <= r["epoch"] < ENTRY_END for r in unknown):
        raise ValueError("missing entry quote within common window")
    crypto_books = DATA / "crypto_books.json"
    stock_report = ROOT / "stock_liquid_quote_routes_nov21.json"
    stock_files = json.loads(stock_report.read_bytes())["close_book_files"]
    available = json.loads(crypto_books.read_bytes())["books"] | stock_files
    sources = dict(hashes, **{str(stock_report): file_hash(stock_report)})
    cache = DATA / "joint_exact_exit_samples.pkl"
    saved = pickle.loads(cache.read_bytes()) if cache.exists() else None
    if saved and saved["sources"] == sources and saved["entry_end"] == ENTRY_END:
        exit_snapshots, extra, checks = saved["result"]
        for symbol, samples in exit_snapshots.items():
            books[symbol].update(samples)
    else:
        original = {s: set(samples) for s, samples in books.items()}
        extra, checks = exit_books(rows, {"books": available}, specs, books)
        exit_snapshots = {s: {at: v for at, v in samples.items() if at not in original[s]}
                          for s, samples in books.items()}
        cache.write_bytes(pickle.dumps(dict(sources=sources, entry_end=ENTRY_END,
                                           result=(exit_snapshots, extra, checks)), protocol=5))
    cases = score(rows, specs, ticks, books, exit_proxy)
    report = dict(start=START, entry_end=ENTRY_END, available_book_end=AVAILABLE_BOOK_END,
                  source_hashes=sources, proposals=Counter(r["bot"] + ":" + r["entry_status"] for r in rows),
                  tape_checks=tapes, entry_book_checks=entry_checks, exit_book_checks=checks,
                  additional_book_files=extra, scenarios=cases, live_activation_supported=False,
                  limitations=[
                      "Common dated entry window chosen by maximum 16h holding period inside the L2 archive.",
                      "Exit attempts use an observed book before the trigger plus the declared delay.",
                      "Only realized cash and contract rounding. Stock eligibility assumed for this simulation.",
                      "An unavailable or stale exit book makes the affected case unscorable.",
                  ])
    write_json(ROOT / "okx_joint_exact_exit_nov21.json", report)
    for label, fits in cases.items():
        print(label, {name: {key: out[key] for key in ("status", "trades", "return_pct", "drawdown_pct",
                                                      "reason", "symbol") if key in out}
                      for name, out in fits.items() if name != "matched_runner_cost"}, flush=True)


if __name__ == "__main__":
    main()
