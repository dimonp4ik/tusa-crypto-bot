"""Replay the stock bot's native 1% stop and the observer's structural stop side by side."""

import argparse
from bisect import bisect_left, bisect_right
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
from pathlib import Path
import pickle

from okx_recent_trade_repair_nov21 import load_repaired_tape
from okx_tp1_cash_sensitivity_nov22 import detailed
from okx_unbankable_tp1_nov21 import read_json
from okx_xperp_cash_tick_nov19 import CAPITAL, simulate
from okx_xperp_l2_entry_replay_nov19 import cash_phase
from okx_xperp_multiday_capture_nov20 import file_hash
from okx_xperp_tick_path_nov19 import reconcile, resolve_ticks
from stock_liquid_quote_routes_nov21 import (
    ARCHIVES, HOLD_SECONDS, NATIVE_PATH, ROOT, capture, limit_bursts, native_module, repair_trades, write_json,
)
from stock_quote_exit_attribution_nov19 import book_variant
from stock_quote_index_overlay_nov21 import index_marks


STEM = "stock_native_geometry_match_nov22"
DATA = ROOT / STEM
MATURE_BY = int(datetime(2026, 11, 22, 9, 30, tzinfo=timezone.utc).timestamp())
MAX_PRINT_AGE = 60_000


def freeze():
    DATA.mkdir(parents=True, exist_ok=True)
    path = DATA / "plan.json"
    if path.exists():
        return read_json(path)
    folder, profile = ARCHIVES["contract_followup"]
    native_sha = hashlib.sha256(NATIVE_PATH.read_text(encoding="utf-8").encode()).hexdigest()
    rows, archives, index_sources = [], {}, {}
    for source in sorted(folder.glob("bucket_*.json")):
        bucket = int(source.stem.removeprefix("bucket_"))
        if bucket + HOLD_SECONDS > MATURE_BY:
            continue
        scan = read_json(source)
        if scan["bucket"] != bucket or scan["profile"] != profile:
            raise ValueError("invalid stock archive identity")
        summary = scan["summary"]
        if not scan["completed_at"] or not summary.get("history_complete") or not summary.get("protocol_ok"):
            continue
        if summary["native_source_sha256"] != native_sha:
            raise ValueError("observer's native scan and local production implementation differ")
        archives[source.name] = file_hash(source)
        proposals = [x for x in scan["observations"] if x["payload"]["native_candidate"]]
        if len(proposals) != summary["raw_setups"]:
            raise ValueError("native candidate manifest is incomplete")
        if proposals:
            index_sources[str(bucket)] = index_marks(bucket)
        for item in proposals:
            payload = item["payload"]
            if item["closed_frame"]["time"][-1] != bucket - 900:
                raise ValueError("future or missing stock signal close")
            receipt = float(item["received_at"])
            rows.append(dict(symbol=item["symbol"], instrument=item["instrument"], epoch=bucket,
                             bot="stocks", route="native_observer", direction=payload["native_candidate"]["direction"],
                             native_candidate=payload["native_candidate"], structural=payload["geometry"],
                             entry_ms=round(receipt * 1000), receipt=receipt, phase=cash_phase(receipt),
                             spec=payload["spec"], public_book=payload["book"], valid_book=bool(item["quote_ok"]),
                             preceding_close=float(item["closed_frame"]["close"][-1])))
    if not rows or any(not row["spec"] for row in rows):
        raise ValueError("stock plan has missing native candidates or instrument specifications")
    plan = dict(cases=dict(contract_observer=rows), counts=Counter(r["phase"] for r in rows),
                archive_sha256=archives, maturity_cutoff=MATURE_BY, native_sha256=native_sha, indexes=index_sources)
    write_json(path, plan)
    return plan


def price_before(tapes, symbol, at, *, inclusive=False):
    times, tape = tapes[symbol]
    position = (bisect_right if inclusive else bisect_left)(times, at) - 1
    if position < 0 or at - times[position] > MAX_PRINT_AGE:
        return None
    return tape[position].price


def ticker_at_receipt(row, times, tape):
    index = bisect_left(times, row["entry_ms"]) - 1
    if index < 0 or times[index] < row["epoch"] * 1000:
        # Validated contract volume rules out unseen prints within this bucket.
        # The preceding observer close is the last known price, even after a quiet bar.
        return row["preceding_close"], "carried_closed_price", None
    age = row["entry_ms"] - times[index]
    return tape[index].price, "current_bucket_trade", age


def stop_recovery(path, tape, times, row):
    if not path["known"] or path["code"] != "SL":
        return None
    first = bisect_right(times, path["end_ms"])
    last = bisect_left(times, (row["epoch"] + HOLD_SECONDS) * 1000)
    if first >= last:
        return None
    sign = 1 if row["direction"] == "LONG" else -1
    prices = [sign * t.price for t in tape[first:last]]
    return dict(target_reached_after_stop=max(prices) >= sign * row["target"],
                entry_reclaimed=max(prices) >= sign * row["entry"],
                another_stop_distance_adverse=min(prices) <= sign * (2 * row["stop"] - row["entry"]))


def geometry(anchor, row, *, stop=None):
    sign = 1 if row["direction"] == "LONG" else -1
    return dict(side=row["direction"], signal_entry=anchor,
                target=round(anchor + sign * anchor * .01 * row["native_candidate"]["target_r"], 8),
                stop=round(anchor - sign * anchor * .01, 8) if stop is None else stop)


def cash_view(rows, specs, ticks, books, tapes, penalty):
    if not rows:
        return dict(status="NO_QUALIFIED_ENTRIES", trades=0)

    def close(symbol, at, direction, _):
        price = price_before(tapes, symbol, at, inclusive=True)
        if price is None:
            raise ValueError("no recent trade at the evaluated market exit")
        return price * (1 - (1 if direction == "LONG" else -1) * penalty / 10_000)

    result = detailed(simulate(list(limit_bursts(rows, max_same_side=2)), specs, ticks, books,
                               clock=60, lag=200, setting="requested5", exit_price=close))
    if "cash_pnl_usd" in result:
        result["status"] = "SCORED_OBSERVED_ENTRY_AND_EXIT_STRESS"
    return result


def score(plan):
    capture_path, repair_path = DATA / "capture.json", DATA / "trade_repair.json"
    captured, repair = map(read_json, (capture_path, repair_path))
    if (captured["plan_sha256"] != file_hash(DATA / "plan.json")
            or repair["capture_sha256"] != file_hash(capture_path)):
        raise ValueError("stock inputs are inconsistent")
    native = native_module()
    rows = plan["cases"]["contract_observer"]
    specs = captured["specs"]
    ticks = {s: float(spec["tickSz"]) for s, spec in specs.items()}
    tape_checks, tapes, books = {}, {}, defaultdict(dict)
    for symbol in sorted(specs):
        item = captured["trades"][symbol]
        tape = load_repaired_tape(dict(item, repaired=repair["intervals"].get(symbol, {})))
        frame_item = captured["reference"][symbol]
        source = Path(frame_item["path"])
        if file_hash(source) != frame_item["sha256"]:
            raise ValueError("stock trade reconciliation reference changed")
        frame = pickle.loads(source.read_bytes())
        own = [r for r in rows if r["symbol"] == symbol]
        mismatches = reconcile(tape, frame, start=min(r["epoch"] for r in own),
                               end=max(r["epoch"] for r in own) + HOLD_SECONDS)
        relevant = {t for r in own for t in range(r["epoch"], r["epoch"] + HOLD_SECONDS, 900)}
        tape_checks[symbol] = dict(bars_checked=len(relevant), mismatches={t: m for t, m in mismatches.items()
                                                                        if t in relevant})
        tapes[symbol] = ([t.ms for t in tape], tape)
    diagnostics = []
    cases = dict(production_1pct=[], structural_matched=[], production_matched=[])
    for row in rows:
        symbol, at = row["symbol"], row["entry_ms"]
        sign = 1 if row["direction"] == "LONG" else -1
        damaged = any(row["epoch"] <= t < row["epoch"] + HOLD_SECONDS for t in tape_checks[symbol]["mismatches"])
        indexed = plan["indexes"].get(str(row["epoch"]))
        trends = ({s: sign * value for s, value in indexed["trends"].items()}
                  if indexed and indexed["complete_at"] * 1000 < at else None)
        notes = dict(symbol=symbol, epoch=row["epoch"], entry_ms=at, direction=row["direction"], phase=row["phase"],
                     index_returns_signed=trends)
        diagnostics.append(notes)
        if damaged or not 0 <= row["receipt"] - row["epoch"] <= 180:
            notes["status"] = "unknown_evidence" if damaged else "late_quote"
            continue
        times, tape = tapes[symbol]
        last, anchor_kind, age = ticker_at_receipt(row, times, tape)
        notes.update(ticker_reference=last, ticker_anchor=anchor_kind, current_bucket_trade_age_ms=age)
        fixed = geometry(last, row)
        ticker = dict(row, entry=last, stop=fixed["stop"], target=fixed["target"])

        def path(candidate, entry):
            return resolve_ticks(tape, times, signal=candidate, entered_at=at, entry_px=entry,
                                 tick_size=ticks[symbol], amend_seconds=60)

        has_backing = native.backing_level_allows_entry(row["native_candidate"], last)
        paper_path = path(ticker, last) if has_backing else None
        notes.update(paper_path=paper_path, ticker_backing=has_backing, fixed_stop=fixed["stop"],
                     fixed_target=fixed["target"],
                     paper_stop_recovery=stop_recovery(paper_path, tape, times, ticker) if paper_path else None)
        quote = book_variant(dict(row, bucket=row["epoch"], book=row["public_book"], geometry=fixed))
        notes["quote"] = quote
        if not has_backing:
            notes["status"] = "ticker_broke_backing"
            continue
        notes["status"] = quote["status"]
        if quote["status"] != "qualified_5bp":
            continue
        if not native.backing_level_allows_entry(row["native_candidate"], quote["average"]):
            notes["status"] = "book_broke_backing"
            continue
        book = row["public_book"]
        books[symbol][at] = dict(ts=book["exchange_at"]*1000, bids=book["bids"], asks=book["asks"])
        fixed_candidate = dict(ticker, entry_status="eligible", stock_reference_ready=True,
                               path={60: path(ticker, quote["average"])})
        cases["production_1pct"].append(fixed_candidate)
        if row["structural"] is not None:
            structure = geometry(last, row, stop=row["structural"]["stop"])
            width = sign * (quote["average"] - structure["stop"]) / quote["average"]
            other = book_variant(dict(row, bucket=row["epoch"], book=book, geometry=structure))
            if .003 <= width <= .02 and other["status"] == "qualified_5bp":
                moved = dict(fixed_candidate, stop=structure["stop"])
                moved["path"] = {60: path(moved, quote["average"])}
                cases["production_matched"].append(fixed_candidate)
                cases["structural_matched"].append(moved)
                notes["structural_width"] = width
    results = {name: {str(bps): cash_view(cohort, specs, ticks, books, tapes, bps) for bps in (5, 15, 30)}
               for name, cohort in cases.items()}
    paper_stats = Counter(r["paper_path"]["code"] for r in diagnostics if (r.get("paper_path") or {}).get("known"))
    reports = dict(maturity_cutoff=plan["maturity_cutoff"], source_hashes={p.name: file_hash(p) for p in
                   (DATA/"plan.json", capture_path, repair_path)}, starting_capital_usd=CAPITAL,
                   examined=len(diagnostics), status_counts=Counter(r["status"] for r in diagnostics),
                   ticker_paper_path_counts=paper_stats, matched_cases={k: len(v) for k, v in cases.items()},
                   diagnostics=diagnostics, results=results, tape_checks=tape_checks, live_activation_supported=False,
                   limitations=["The quote observer is independent of the Telegram signal journal and portfolios.",
                     "The last current-bucket contract trade, or the preceding confirmed close if no trade occurred, proxies the ticker.",
                     "Native fixed stop is 1%; the structural observer stores a different known-close stop.",
                     "The matched structural comparison keeps the native target and the same executable entry.",
                     "Unverified tape windows do not produce an outcome. Cash uses adverse exit assumptions."])
    write_json(ROOT / f"{STEM}.json", reports)
    print(reports["status_counts"], paper_stats, reports["matched_cases"], flush=True)
    print({k: {p: (s["status"], s.get("trades"), s.get("cash_pnl_usd")) for p, s in v.items()}
           for k, v in results.items()}, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--score-only", action="store_true")
    args = parser.parse_args()
    frozen = freeze()
    print("Frozen stock proposals", frozen["counts"], flush=True)
    if not args.score_only:
        if not (DATA / "capture.json").exists():
            capture(data=DATA)
        if not (DATA / "trade_repair.json").exists():
            repair_trades(data=DATA)
    score(frozen)
