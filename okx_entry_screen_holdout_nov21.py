"""Score the fixed entry screens on a disjoint four-day X-Perp archive."""

import argparse
import json
import pickle
from collections import Counter
from statistics import median

from account_cash_guard_nov17 import stamp
from okx_entry_screen_cash_nov21 import POLICIES, entry_features, screened_rows, summarize
from okx_xperp_cash_tick_nov19 import simulate
from okx_xperp_multiday_capture_nov20 import book_files, file_hash, freeze, trade_files
from okx_xperp_multiday_replay_nov20 import FOLDER, ROOT, cached_entries, entries, exit_books, resolve_all, utc_day


HOLDOUT = FOLDER.with_name("nov5_8_entry_screen")
START = int(stamp("2026-11-05"))
ENTRY_END = int(stamp("2026-11-09"))
FROZEN_POLICY_COMMIT = "f6c752352d54cde8ecb97763fe52ff3cd4ac9df8"


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")


def capture_inputs():
    HOLDOUT.mkdir(parents=True, exist_ok=True)
    plan = freeze(start=START, entry_end=ENTRY_END)
    plan_file = HOLDOUT / "plan.json"
    if plan_file.exists() and json.loads(plan_file.read_bytes()) != plan:
        raise ValueError("holdout plan changed")
    if not plan_file.exists():
        write_json(plan_file, plan)
    if not plan["rows"]:
        raise ValueError("empty holdout")
    print("frozen", Counter((utc_day(r["epoch"]), r["bot"]) for r in plan["rows"]), flush=True)
    books = book_files(plan)
    trades = trade_files(plan)
    complete = all("path" in item for item in books.values()) and all(item["complete"] for item in trades.values())
    write_json(HOLDOUT / "capture.json",
               dict(plan_sha256=file_hash(plan_file), books=books, trades=trades, complete=complete))
    print("captured", complete, len(books), sum(len(t["days"]) for t in trades.values()), flush=True)


def score_policies(rows, specs, ticks, books):
    if any(r["entry_status"] == "unknown_quote" for r in rows):
        return {}
    eligible = [r for r in rows if r["bot"] == "crypto" and r["entry_status"] == "eligible"]
    features = {(r["symbol"], r["entry_ms"]): entry_features(r, books[r["symbol"]][r["entry_ms"]])
                for r in eligible}
    results = {}
    for name, policy in POLICIES.items():
        selected = screened_rows(rows, features, **policy)
        result = simulate(selected, specs, ticks, books, clock=60, lag=200, setting="requested5")
        results[name] = summarize(result)
    return results


def combine_windows(rows, specs, books, reference_path, universe_path, tick_by_id):
    source = ROOT / "okx_xperp_multiday_cash_nov20.json"
    later = json.loads(source.read_bytes())["paths"]
    if any(r["epoch"] < ENTRY_END for r in later):
        raise ValueError("discovery and holdout contain overlapping entry dates")
    for row in later:
        if "path" in row:
            row["path"] = {int(key): value for key, value in row["path"].items()}

    plan_path, capture_path = FOLDER / "plan.json", FOLDER / "capture.json"
    plan = json.loads(plan_path.read_bytes())
    capture = json.loads(capture_path.read_bytes())
    if any(s in specs and specs[s] != spec for s, spec in plan["specs"].items()):
        raise ValueError("instrument specifications differ between windows")
    shared_specs = specs | plan["specs"]
    ticks = {symbol: tick_by_id[spec["instId"]] for symbol, spec in shared_specs.items()}

    cache_path = FOLDER / "entry_exit_samples_nov21.pkl"
    digest = file_hash(source)
    saved_digest, later_books = pickle.loads(cache_path.read_bytes()) if cache_path.exists() else (None, None)
    if saved_digest != digest:
        inputs = plan_path, capture_path, reference_path, universe_path
        _, later_books, _ = cached_entries(plan, capture, inputs)
        exit_books(later, capture, plan["specs"], later_books)
        cache_path.write_bytes(pickle.dumps((digest, later_books), protocol=5))
    for symbol, snapshots in later_books.items():
        if any(books[symbol][key] != snapshots[key] for key in books[symbol].keys() & snapshots.keys()):
            raise ValueError("overlapping book snapshots disagree")
        books[symbol].update(snapshots)
    return dict(discovery_digest=digest, results=score_policies(rows + later, shared_specs, ticks, books))


def score():
    plan_path, capture_path = HOLDOUT / "plan.json", HOLDOUT / "capture.json"
    reference_path = ROOT / "paired_contract_volume_nov19/manifest.json"
    universe_path = ROOT / "native_universe_nov19_snapshot.json"
    inputs = plan_path, capture_path, reference_path, universe_path
    plan, capture, reference, universe = (json.loads(path.read_bytes()) for path in inputs)
    if capture["plan_sha256"] != file_hash(plan_path) or plan["start"] != START:
        raise ValueError("holdout digest or cutoff changed")
    if any(not START <= r["epoch"] < ENTRY_END for r in plan["rows"]):
        raise ValueError("overlapping entry dates")
    specs = plan["specs"]
    tick_by_id = {r["instId"]: float(r["tickSz"]) for r in universe["instruments"] if r.get("tickSz")}
    ticks = {symbol: tick_by_id[spec["instId"]] for symbol, spec in specs.items()}
    local_cache = HOLDOUT / "verified_public_paths.pkl"
    signature = [file_hash(path) for path in inputs]
    cached = pickle.loads(local_cache.read_bytes()) if local_cache.exists() else None
    if cached and cached["signature"] == signature:
        rows, books, checks = cached["data"]
    else:
        rows, books, entry_checks = entries(plan, capture)
        tape_checks = resolve_all(rows, {"files": capture["trades"]}, reference, ticks)
        extra, exit_checks = exit_books(rows, capture, specs, books)
        checks = dict(entry=entry_checks, tape=tape_checks, exit=exit_checks, additional_book_files=extra)
        local_cache.write_bytes(pickle.dumps(dict(signature=signature, data=(rows, books, checks)), protocol=5))
    unknown = [(r["symbol"], r["entry_ms"]) for r in rows if r["entry_status"] == "unknown_quote"]
    stocks = [r for r in rows if r["bot"] == "stocks"]
    stock_gate = {}
    for state in sorted({r["entry_status"] for r in stocks}):
        cohort = [r for r in stocks if r["entry_status"] == state]
        spreads = [min(quotes) for r in cohort
                   if (quotes := [t["spread_bps"] for t in r["tries"] if "spread_bps" in t])]
        stock_gate[state] = {
            "signals": len(cohort),
            "paper_readiness_passed": sum(r["stock_reference_ready"] for r in cohort),
            "median_best_observed_spread_bps": median(spreads) if spreads else None,
            "paper_outcomes": Counter(r["paper_path"].get("code", "unknown") for r in cohort),
        }
    results = score_policies(rows, specs, ticks, books)
    report = {
        "fixed_policy_commit": FROZEN_POLICY_COMMIT,
        "holdout_entry_start": START,
        "holdout_entry_end": ENTRY_END,
        "input_hashes": {str(path): digest for path, digest in zip(inputs, signature)},
        "proposals": Counter(r["bot"] for r in rows),
        "entry_status": {bot: Counter(r["entry_status"] for r in rows if r["bot"] == bot)
                         for bot in ("crypto", "stocks")},
        "stock_gate_diagnostics": stock_gate,
        "unknown_quotes": unknown,
        "cash_results": results,
        "continuous_cash_results": combine_windows(rows, specs, books, reference_path, universe_path, tick_by_id),
        "tape_checks": checks["tape"],
        "archive_complete": capture["complete"],
        "live_activation_supported": False,
        "limitations": [
            "Four earlier days evaluated with historical public market data.",
            "Simultaneous proposals are processed in symbol order.",
            "Three entry attempts and a modeled 200ms exit delay at fixed historical latency.",
            "Common 3% collateral and 10x, realized cash sizing, no fees or floating-equity pause.",
        ],
    }
    write_json(ROOT / "okx_entry_screen_holdout_nov21.json", report)
    print(json.dumps({k: {q: v for q, v in result.items()
                         if q in ("status", "trades", "return_pct", "drawdown_pct", "reason", "symbol")}
                      for k, result in results.items()}, indent=2))
    if unknown:
        raise ValueError(f"{len(unknown)} proposals lack an entry quote")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("capture", "score"))
    args = parser.parse_args()
    if args.stage == "capture":
        capture_inputs()
    else:
        score()
