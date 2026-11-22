"""Test stock setups at the original public book receipt, with causal closed bars."""

import argparse
import hashlib
import importlib.util
import json
import math
import pickle
from bisect import bisect_left, bisect_right
from collections import Counter, defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

from okx_recent_trade_repair_nov21 import load_repaired_tape, repair_recent_days
from okx_xperp_cash_tick_nov19 import simulate
from okx_xperp_l2_entry_replay_nov19 import cash_phase
from okx_xperp_multiday_capture_nov20 import file_hash, trade_files
from okx_xperp_multiday_replay_nov20 import exit_books, resolve_all
from public_candle_window import fetch_window
from src.okx_trader import round_to_tick
from stock_quote_exit_attribution_nov19 import HOLD_SECONDS, ROOT, book_variant, minute_exit


STOCK_ROOT = Path("../tusa stocks")
ARCHIVES = {
    "legacy_discovery": (
        STOCK_ROOT / "reports/structure_observer_prospective_nov18",
        "stock_swing2_original_target_quote_observer_v1",
    ),
    "contract_followup": (
        STOCK_ROOT / "reports/structure_observer_contracts_nov19",
        "stock_swing2_original_target_quote_observer_v2",
    ),
}
MATURE_AT = 1795221000
DATA = ROOT / "stock_liquid_quote_routes_nov21"
ROUTES = ("native", "reversal_z1_25", "aligned_continuation")
NATIVE_PATH = STOCK_ROOT / "src/stock_xperp_native_shadow.py"


def native_module():
    spec = importlib.util.spec_from_file_location("research_stock_native", NATIVE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def hypothetical_setups(native, frames, decision, diagnostics):
    # Only change cross-sectional selectivity. Keep the native market filters.
    with patch.object(native, "MINIMUM_ABS_Z", 1.25):
        relaxed = native.native_shadow_setups(frames, now=decision, diagnostics=diagnostics)
    if diagnostics.get("result") not in {"candidates", "no_candidate_passed"}:
        return relaxed, []
    aligned = []
    found = native._latest_rows(frames)
    if found is None:
        return relaxed, aligned
    for row in found[1]:
        frame = row["candles"]
        i = row["signal_index"]
        move = row["signal_return"]
        side = "LONG" if move > 0 else "SHORT"
        sign = 1 if side == "LONG" else -1
        trend = diagnostics["basket_trend_16_pct"] / 100
        close = frame["close"][i]
        bar_range = frame["high"][i] - frame["low"][i]
        atr = native.normalized_atr14(frame, i)
        if atr is None or atr > .01 or abs(move) < .004 or sign * trend < .005 or bar_range <= 0:
            continue
        close_strength = (close - frame["low"][i]) / bar_range
        if (close_strength if sign > 0 else 1 - close_strength) < .70:
            continue
        aligned.append(dict(symbol=row["symbol"], direction=side, entry=close, target_r=2.,
                            atr14_fraction=atr, backing_level=None))
    return relaxed, aligned


def geometry(setup, frame):
    sign = 1 if setup["direction"] == "LONG" else -1
    e = setup["entry"]
    pad = .25 * setup["atr14_fraction"] * e
    extreme = min(frame["low"][-2:]) - pad if sign > 0 else max(frame["high"][-2:]) + pad
    risk = max(.003, sign * (e - extreme) / e)
    if risk > .02:
        return None
    return dict(side=setup["direction"], signal_entry=e, stop=e * (1 - sign * risk),
                target=e * (1 + sign * setup["target_r"] * .01))


def freeze(*, archives=ARCHIVES, first_bucket=0, mature_at=MATURE_AT):
    native = native_module()
    cases, counts, hashes = {}, {}, {}
    for cohort, (folder, profile) in archives.items():
        cases[cohort] = []
        hashes[cohort] = {}
        counts[cohort] = {route: Counter() for route in ROUTES}
        for path in sorted(folder.glob("bucket_*.json")):
            at = int(path.stem.removeprefix("bucket_"))
            if at < first_bucket or at + HOLD_SECONDS > mature_at:
                continue
            raw = path.read_bytes()
            scan = json.loads(raw)
            bucket = scan["bucket"]
            if path.stem != f"bucket_{bucket}" or scan["profile"] != profile:
                raise ValueError("archive profile or timestamp mismatch")
            summary = scan["summary"]
            if not scan["completed_at"] or not summary.get("protocol_ok") or not summary.get("history_complete"):
                continue
            by_symbol = {row["symbol"]: row for row in scan["observations"]}
            if len(by_symbol) != len(scan["observations"]) or set(by_symbol) != native.AUDITED_SYMBOLS:
                raise ValueError("incomplete archived universe")
            frames = {symbol: row["closed_frame"] for symbol, row in by_symbol.items()}
            if any(frame["time"][-1] != bucket - 900 for frame in frames.values()):
                raise ValueError("unconfirmed decision-bar close")
            hashes[cohort][path.name] = hashlib.sha256(raw).hexdigest()
            relaxed, continuation = hypothetical_setups(native, frames, summary["decision_at"], {})
            setups_by_route = {
                "native": [row["payload"]["native_candidate"] for row in by_symbol.values() if row["candidate"]],
                "reversal_z1_25": relaxed,
                "aligned_continuation": continuation,
            }
            for route, setups in setups_by_route.items():
                for setup in setups:
                    symbol = setup["symbol"]
                    row = by_symbol[symbol]
                    payload = row["payload"]
                    levels = payload["geometry"] if route == "native" else geometry(setup, frames[symbol])
                    if not levels:
                        counts[cohort][route]["no_stop_geometry"] += 1
                        continue
                    receipt = float(row["received_at"])
                    book = payload["book"]
                    spec = payload.get("spec")
                    ev = book_variant(dict(valid_book=row["quote_ok"], spec=spec, receipt=receipt,
                                           bucket=bucket, book=book, geometry=levels))
                    if ev["status"] != "qualified_5bp":
                        counts[cohort][route][ev["status"]] += 1
                        continue
                    if (route != "aligned_continuation"
                            and not native.backing_level_allows_entry(setup, ev["average"])):
                        counts[cohort][route]["structure_invalidated"] += 1
                        continue
                    counts[cohort][route]["qualified"] += 1
                    cases[cohort].append(dict(
                        route=route, symbol=symbol, epoch=bucket, entry_ms=round(receipt * 1000),
                        bot="stocks", direction=levels["side"], entry=levels["signal_entry"],
                        stop=levels["stop"], target=levels["target"], stock_reference_ready=True,
                        entry_status="eligible", reference_entry=ev["average"], phase=cash_phase(receipt),
                        instrument=row["instrument"], spec=spec,
                        book=dict(ts=round(book["exchange_at"] * 1000), bids=book["bids"], asks=book["asks"]),
                    ))
    return dict(cases=cases, counts=counts, archive_sha256=hashes, native_sha256=file_hash(NATIVE_PATH),
                mature_at=mature_at)


def write_json(path, payload):
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")


def load_plan(data=DATA):
    data.mkdir(parents=True, exist_ok=True)
    path = data / "plan.json"
    if path.exists():
        return json.loads(path.read_bytes())
    plan = freeze()
    write_json(path, plan)
    return plan


def capture(*, data=DATA):
    plan = load_plan(data)
    rows = [r for cases in plan["cases"].values() for r in cases]
    print("proposals", plan["counts"], flush=True)
    print("qualified markets", dict(Counter((r["symbol"], r["route"]) for r in rows)), flush=True)
    if not rows:
        return
    # Canonical settlement, lot, and tick metadata must agree for one instrument.
    specs = {}
    for row in rows:
        spec = row["spec"]
        if spec["instId"] != row["instrument"] or specs.setdefault(row["symbol"], spec) != spec:
            raise ValueError("instrument specification mismatch")
    start = min(r["epoch"] for r in rows)
    end = max(r["epoch"] + HOLD_SECONDS for r in rows)
    trades = trade_files(dict(rows=rows, specs=specs, start=start, coverage_end=end))
    references = {}

    def candles(symbol, subset):
        begin = min(r["epoch"] for r in subset)
        finish = max(r["epoch"] + HOLD_SECONDS for r in subset)
        frame = defaultdict(list)
        for at in range(begin, finish, 86400):
            source, _ = fetch_window(specs[symbol]["instId"], at, min(at + 86400, finish), 900, data / "candles")
            for field, values in source.items():
                frame[field].extend(values)
        frame["volume_contracts"] = frame["volume"]
        path = data / f"{symbol}_reference.pkl"
        path.write_bytes(pickle.dumps(dict(frame), protocol=5))
        return symbol, dict(path=str(path), sha256=file_hash(path))

    grouped = defaultdict(list)
    for row in rows:
        grouped[row["symbol"]].append(row)
    with ThreadPoolExecutor(max_workers=3) as pool:
        references.update(pool.map(lambda item: candles(*item), sorted(grouped.items())))
    write_json(data / "capture.json", dict(plan_sha256=file_hash(data / "plan.json"),
                                           trades=trades, reference=references, specs=specs, books={}))
    print("captured", len(trades), "instruments", sum(len(v["days"]) for v in trades.values()), "files", flush=True)


def score_minutes(*, data=DATA, stem="stock_liquid_quote_routes_nov21"):
    plan = load_plan(data)
    grouped = defaultdict(list)
    for cohort, rows in plan["cases"].items():
        for row in rows:
            grouped[row["instrument"]].append((cohort, row))

    def review(item):
        instrument, candidates = item
        start = min(r["epoch"] for _, r in candidates)
        finish = max(r["epoch"] + HOLD_SECONDS for _, r in candidates)
        frame = defaultdict(list)
        hashes = []
        for begin in range(start, finish, 86400):
            source, digest = fetch_window(instrument, begin, min(begin + 86400, finish), 60, data / "minutes")
            hashes.append(digest)
            for field, values in source.items():
                frame[field].extend(values)
        output = []
        for cohort, row in candidates:
            left = bisect_left(frame["time"], row["entry_ms"] // 60_000 * 60)
            end = bisect_left(frame["time"], row["epoch"] + HOLD_SECONDS)
            stamp = row["entry_ms"] / 1000
            tick = float(row["spec"]["tickSz"])
            levels = dict(side=row["direction"], stop=round_to_tick(row["stop"], tick),
                          target=round_to_tick(row["target"], tick))
            path = minute_exit({field: values[left:end] for field, values in frame.items()},
                               {"geometry": levels, "receipt": stamp})
            basis = row["reference_entry"]
            sign = 1 if row["direction"] == "LONG" else -1
            price_return = sign * (path["fill_reference"] / basis - 1) if path["known"] else None
            output.append(dict(cohort=cohort, route=row["route"], symbol=row["symbol"],
                               phase=row["phase"], direction=row["direction"], entry_ms=row["entry_ms"],
                               path=path, price_return=price_return))
        return instrument, hashes, output

    results = []
    sources = {}
    with ThreadPoolExecutor(max_workers=2) as pool:
        for instrument, hashes, rows in pool.map(review, sorted(grouped.items())):
            sources[instrument] = hashes
            results.extend(rows)
            print("minute data checked", instrument, len(rows), flush=True)
    aggregates = {}
    for cohort in plan["cases"]:
        aggregates[cohort] = {}
        for route in ROUTES:
            proposals = [r for r in results if r["cohort"] == cohort and r["route"] == route]
            for label, subset in (("all", proposals), ("regular", [r for r in proposals if r["phase"] == "regular"])):
                values = [r["price_return"] for r in subset if r["path"]["known"]]
                aggregates[cohort][f"{route}|{label}"] = dict(
                    qualified=len(subset), scored=len(values), wins=sum(v > 0 for v in values),
                    mean_price_pct=100 * math.fsum(values) / len(values) if values else None,
                    exits=Counter(r["path"].get("outcome", r["path"].get("reason")) for r in subset),
                )
    write_json(ROOT / f"{stem}_minutes.json",
               dict(plan_sha256=file_hash(data / "plan.json"), minute_sources=sources, aggregates=aggregates,
                    results=results, live_activation_supported=False,
                    limitations=["Entry on observed top-five book at 3% margin and 10x.",
                                 "Each setup is independent. Exit price is a one-minute trade-based reference.",
                                 "Ambiguous entry minute and simultaneous stop/target touches stay unknown.",
                                 "Average refers to the underlying price move. Shared account drawdown requires the L2 exit replay."]))
    print(json.dumps(aggregates, indent=2))


def repair_trades(*, data=DATA):
    plan = load_plan(data)
    capture_path = data / "capture.json"
    captured = json.loads(capture_path.read_bytes())
    if captured["plan_sha256"] != file_hash(data / "plan.json"):
        raise ValueError("trade repair plan changed")
    rows = [r for cohort in plan["cases"].values() for r in cohort]
    evidence = repair_recent_days(captured["trades"], rows, hold_seconds=HOLD_SECONDS,
                                  folder=data / "repaired_prints")
    write_json(data / "trade_repair.json", dict(capture_sha256=file_hash(capture_path), intervals=evidence))


def limit_bursts(proposals, *, max_same_side):
    """Admit proposals in quote receipt order, including visible per-scan limits."""
    last_scan = Counter()
    recent = deque()
    for row in sorted(proposals, key=lambda r: (r["entry_ms"], r["symbol"])):
        at = row["entry_ms"]
        while recent and recent[0]["entry_ms"] <= at - 3_600_000:
            recent.popleft()
        side_count = sum(r["direction"] == row["direction"] for r in recent)
        if last_scan[row["epoch"]] >= 2 or len(recent) >= 4 or side_count >= max_same_side:
            yield dict(row, entry_status="publication_rate_limit")
        else:
            last_scan[row["epoch"]] += 1
            recent.append(row)
            yield row


def load_replay(data, *, keep_tapes=False):
    plan = load_plan(data)
    capture_path = data / "capture.json"
    captured = json.loads(capture_path.read_bytes())
    if captured["plan_sha256"] != file_hash(data / "plan.json"):
        raise ValueError("frozen proposals changed")
    rows = [r for cases in plan["cases"].values() for r in cases]
    specs = captured["specs"]
    ticks = {s: float(value["tickSz"]) for s, value in specs.items()}
    books = defaultdict(dict)
    for row in rows:
        books[row["symbol"]][row["entry_ms"]] = row["book"]
    repair_path = data / "trade_repair.json"
    repair = json.loads(repair_path.read_bytes()) if repair_path.exists() else None
    if repair:
        if repair["capture_sha256"] != file_hash(capture_path):
            raise ValueError("trade repair and archive capture disagree")
        for symbol, intervals in repair["intervals"].items():
            captured["trades"][symbol]["repaired"] = intervals
    stored_tapes = {}

    def load_tape(item):
        tape = load_repaired_tape(item)
        if keep_tapes:
            stored_tapes[item["instrument"]] = ([t.ms for t in tape], tape)
        return tape

    tape = resolve_all(rows, {"files": captured["trades"]}, {"symbols": captured["reference"]},
                       ticks, trade_loader=load_tape)
    return plan, captured, rows, specs, ticks, books, tape, stored_tapes, repair


def stressed_exit_penalty(stored_tapes, specs, bps):
    def price(symbol, at, side, _):
        times, entries = stored_tapes[specs[symbol]["instId"]]
        index = bisect_right(times, at) - 1
        if index < 0 or at - times[index] > 60_000:
            raise ValueError("no recent trade for exit stress proxy")
        sign = 1 if side == "LONG" else -1
        return entries[index].price * (1 - sign * bps / 10_000)
    return price


def score(*, stress=False, data=DATA, stem="stock_liquid_quote_routes_nov21", extra_variants=None):
    plan, captured, rows, specs, ticks, books, tape, stored_tapes, repair = load_replay(data, keep_tapes=stress)
    capture_path, repair_path = data / "capture.json", data / "trade_repair.json"
    extra, exits = ({}, {}) if stress else exit_books(rows, captured, specs, books)

    results = {}
    for cohort, cases in plan["cases"].items():
        results[cohort] = {}
        for route in ROUTES:
            proposals = [r for r in cases if r["route"] == route]
            subsets = {"all": proposals, "regular": [r for r in proposals if r["phase"] == "regular"]}
            if route == "aligned_continuation":
                subsets.update(stock_cadence=list(limit_bursts(proposals, max_same_side=2)),
                               one_per_side_hour=list(limit_bursts(proposals, max_same_side=1)))
                if extra_variants:
                    subsets.update({name: list(apply(proposals)) for name, apply in extra_variants.items()})
            for label, subset in subsets.items():
                unknown = [(r["symbol"], r["entry_ms"], r["path"][60].get("reason"))
                           for r in subset if r["entry_status"] == "eligible" and not r["path"][60]["known"]]
                pricers = ({f"|stress_{bps}bps": stressed_exit_penalty(stored_tapes, specs, bps)
                            for bps in (5, 15, 30, 50)}
                           if stress else {"": None})
                for suffix, exit_price in pricers.items():
                    case = simulate(subset, specs, ticks, books, clock=60, lag=200, setting="requested5",
                                    exit_price=exit_price)
                    results[cohort][f"{route}|{label}{suffix}"] = dict(case, qualified=len(subset), unknown_paths=unknown,
                        profitable_closes=sum(t["bank_pnl"] + t["runner_pnl"] > 0 for t in case.get("closed", []))
                        if "closed" in case else None)
    report = dict(plan_sha256=captured["plan_sha256"], capture_sha256=file_hash(capture_path),
                  trade_repair_sha256=file_hash(repair_path) if repair else None,
                  count=plan["counts"], tape=tape, close_book_checks=exits, close_book_files=extra,
                  results=results, live_activation_supported=False,
                  limitations=[
                      "Separate stock-only account scenarios at 3% margin and 10x.",
                      "Readiness assumed for the funding experiment. Actual order readiness remains gated.",
                      "Public entry depth and public last-trade exit triggers followed by an archived market book at +200ms.",
                      "Legacy sample uses base volume and later sample uses contract volume.",
                      "Cadence consumes the slot at quote eligibility before the separate account admission checks.",
                      "Only mature proposals from the defined archives. Costs outside visible execution and floating equity excluded.",
                  ])
    if stress:
        report["execution_model"] = "LAST_PUBLIC_TRADE_EXIT_PENALTY"
        report["limitations"][2] = ("Exit uses the last public trade observed by trigger plus 200ms, "
                                    "with an adverse price penalty. Actual exit depth is unobserved.")
        for cohort in report["results"].values():
            for result in cohort.values():
                if result.get("status") == "SCORED_PUBLIC_L2_SIMULATION":
                    result["status"] = "SCORED_PUBLIC_TRADE_EXIT_STRESS"
    suffix = "_exit_stress" if stress else ""
    write_json(ROOT / f"{stem}{suffix}.json", report)
    print(json.dumps({cohort: {key: {k: v for k, v in case.items() if k != "closed"}
                              for key, case in cases.items()} for cohort, cases in results.items()}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("freeze", "capture", "repair", "score", "stress", "minute"))
    args = parser.parse_args()
    if args.stage == "freeze":
        plan = load_plan()
        print(json.dumps(plan["counts"], indent=2))
    elif args.stage == "capture":
        capture()
    elif args.stage == "repair":
        repair_trades()
    elif args.stage in ("score", "stress"):
        score(stress=args.stage == "stress")
    else:
        score_minutes()
