"""Replay liquidity-first pullbacks and range reversals on timestamped stock books."""

import argparse
from bisect import bisect_right
from collections import Counter, defaultdict, deque
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from statistics import median

import okx_cross_bot_overlap_nov21 as joint
from okx_joint_exact_exit_nov21 import ENTRY_END as COMMON_L2_END, choose_rows
from okx_recent_trade_repair_nov21 import load_repaired_tape
from okx_tp1_cash_sensitivity_nov22 import detailed
from okx_unbankable_tp1_nov21 import read_json
from okx_xperp_cash_tick_nov19 import CAPITAL, FRACTION, LEVERAGE, simulate
from okx_xperp_l2_entry_replay_nov19 import attempt, cash_phase
from okx_xperp_multiday_replay_nov20 import file_hash, resolve_all
from stock_liquid_quote_routes_nov21 import ARCHIVES, HOLD_SECONDS, ROOT, capture, limit_bursts, repair_trades, write_json
from stock_quote_index_overlay_nov21 import INDEXES, MARKS


STEM = "stock_liquidity_first_phases_nov22"
DATA = ROOT / STEM
MATURE_BY = int(datetime(2026, 11, 22, 9, 30, tzinfo=timezone.utc).timestamp())


@dataclass(frozen=True)
class Rules:
    liquidity_lookback: int = 4
    minimum_passing_quotes: int = 3
    minimum_liquid_universe: int = 5
    market_trend: float = .0035
    trend_member_move: float = .003
    trend_pullback: float = .0018
    relative_stretch: float = .0045
    reversal_close_move: float = .0008
    max_entry_drift: float = .001
    stop_fraction: float = .01
    pullback_target: float = .02
    range_target: float = .015


RULES = Rules()


def quote_at(bucket, item):
    payload = item["payload"]
    book, spec = payload["book"], payload["spec"]
    receipt = item["received_at"]
    if not item["quote_ok"] or not spec or not 0 <= receipt-bucket <= 180:
        return None
    return dict(bids=book["bids"], asks=book["asks"], ts=round(book["exchange_at"]*1000))


def stable_symbols(history, bucket):
    expected = list(range(bucket - RULES.liquidity_lookback*900, bucket, 900))
    if [t for t, _ in history] != expected:
        return set()
    counts = Counter(s for _, passing in history for s in passing)
    return {s for s, count in counts.items() if count >= RULES.minimum_passing_quotes}


def price_features(item, bucket):
    c = item["closed_frame"]
    if c["time"][-17:] != list(range(bucket-17*900, bucket, 900)):
        raise ValueError("unconfirmed or gapped history in stock route")
    close = c["close"]
    return dict(known_close=close[-1], four_hour=close[-1]/close[-17]-1,
                impulse=close[-1]/close[-2]-1, pullback=close[-2]/close[-4]-1,
                recent_trade=c["volume"][-1]>0 and c["volume"][-2]>0)


def previous_index_context(bucket, known_at):
    path = MARKS / f"bucket_{bucket-900}.json"
    if not path.exists():
        return None
    snapshot = read_json(path)
    if snapshot["bucket"] != bucket-900 or snapshot["profile"] != "stock_liquid_mark_context_v1":
        raise ValueError("invalid archived stock index context")
    if not snapshot["completed_at"] or snapshot["completed_at"] >= known_at:
        return None
    series = {}
    for item in snapshot["observations"]:
        if item["symbol"] not in INDEXES or item["payload"]["series_kind"] != "mark":
            continue
        frame = item.get("closed_frame")
        if frame is None:
            continue
        if frame["time"][-17:] != list(range(bucket-18*900, bucket-900, 900)):
            raise ValueError("stock index context is late or incomplete")
        close = frame["close"]
        series[item["symbol"]] = close[-1]/close[-17]-1, close[-1]/close[-3]-1
    if set(series) != set(INDEXES):
        return None
    return dict(trend=median(v[0] for v in series.values()), pullback=median(v[1] for v in series.values()),
                sha256=file_hash(path))


def choose_signals(observations, stable, bucket, *, reference=None):
    features = {s: price_features(observations[s], bucket) for s in sorted(stable)
                if s in observations}
    if not features or reference is None and len(features) < RULES.minimum_liquid_universe:
        return [], None
    broad = reference["trend"] if reference else median(x["four_hour"] for x in features.values())
    prior_relative = reference["pullback"] if reference else median(x["pullback"] for x in features.values())
    found = []
    for symbol, values in features.items():
        if not values["recent_trade"]:
            continue
        impulse = values["impulse"]
        relative = values["pullback"]-prior_relative
        if abs(broad) >= RULES.market_trend:
            sign = 1 if broad > 0 else -1
            if (sign*values["four_hour"] >= RULES.trend_member_move
                    and sign*relative <= -RULES.trend_pullback and sign*impulse >= RULES.reversal_close_move):
                found.append((symbol, sign, "trend_pullback", sign*impulse, values))
        elif abs(relative) >= RULES.relative_stretch:
            sign = -1 if relative > 0 else 1
            if sign*impulse >= RULES.reversal_close_move:
                found.append((symbol, sign, "range_reversal", abs(relative), values))
    selected = {}
    for item in sorted(found, key=lambda r: (-r[3], r[0])):
        selected.setdefault(item[1], item)
    return list(selected.values()), broad


def freeze(data, *, use_index):
    data.mkdir(parents=True, exist_ok=True)
    path = data / "plan.json"
    if path.exists():
        frozen = read_json(path)
        if (frozen["rules"] != asdict(RULES) or frozen["mature_by"] != MATURE_BY
                or frozen.get("uses_independent_indexes", False) != use_index):
            raise ValueError("frozen stock policy or endpoint changed")
        return frozen
    history = deque(maxlen=RULES.liquidity_lookback)
    archive_sha, index_sha, cases, diagnostics = {}, {}, {}, {}
    for cohort, (folder, profile) in ARCHIVES.items():
        cases[cohort], archive_sha[cohort], diagnostics[cohort] = [], {}, Counter()
        notes = diagnostics[cohort]
        for source in sorted(folder.glob("bucket_*.json")):
            bucket = int(source.stem.removeprefix("bucket_"))
            if bucket + HOLD_SECONDS > MATURE_BY:
                continue
            scan = read_json(source)
            summary = scan["summary"]
            if scan["bucket"] != bucket or scan["profile"] != profile:
                raise ValueError("incorrect observer window profile or bucket")
            if not (scan["completed_at"] and summary.get("protocol_ok") and summary.get("history_complete")):
                continue
            archive_sha[cohort][source.name] = file_hash(source)
            observations = {r["symbol"]: r for r in scan["observations"]}
            if len(observations) != summary["expected"] or any(r["received_at"] >= bucket+900 for r in observations.values()):
                raise ValueError("observer universe or quote time is inconsistent")
            notes["windows"] += 1
            stable = stable_symbols(history, bucket)
            reference = previous_index_context(bucket, summary["decision_at"]) if use_index else None
            if use_index and reference is None:
                notes["windows_without_prior_index"] += 1
                selected = []
            else:
                if reference:
                    index_sha[str(bucket-900)] = reference["sha256"]
                selected, broad = choose_signals(observations, stable, bucket, reference=reference)
                if broad is None:
                    notes["windows_without_liquid_universe"] += 1
            passing = set()
            quotes = {}
            for symbol, item in observations.items():
                raw = quote_at(bucket, item)
                if raw is None:
                    continue
                quotes[symbol] = raw
                if all(attempt(raw, item["received_at"]*1000, side, item["payload"]["spec"])["status"] == "eligible"
                       for side in ("LONG", "SHORT")):
                    passing.add(symbol)
            history.append((bucket, passing))
            if summary.get("native_result") == "premarket_liquidity_regime_trigger":
                notes["thin_market_veto"] += len(selected)
                continue
            for symbol, sign, route, _, values in selected:
                notes[f"{route}:proposed"] += 1
                item = observations[symbol]
                spec = item["payload"]["spec"]
                at = round(item["received_at"]*1000)
                if item["received_at"] < summary["decision_at"] or symbol not in quotes:
                    notes[f"{route}:missing_eligible_time"] += 1
                    continue
                side = "LONG" if sign == 1 else "SHORT"
                observed = attempt(quotes[symbol], at, side, spec)
                notes[f"{route}:{observed['status']}"] += 1
                if observed["status"] != "eligible":
                    continue
                entry = observed["average"]
                if sign*(entry/values["known_close"]-1) > RULES.max_entry_drift:
                    notes[f"{route}:drift_veto"] += 1
                    continue
                target = RULES.pullback_target if route == "trend_pullback" else RULES.range_target
                cases[cohort].append(dict(symbol=symbol, route=route, instrument=item["instrument"], bot="stocks",
                    epoch=bucket, entry_ms=at, phase=cash_phase(item["received_at"]), direction=side, entry=entry,
                    stop=entry*(1-sign*RULES.stop_fraction), target=entry*(1+sign*target), stock_reference_ready=True,
                    entry_status="eligible", reference_entry=entry, book=quotes[symbol], spec=spec,
                    market_four_hour=broad, liquid_symbol_count=len(stable)))
    if not any(cases.values()):
        raise ValueError("the predeclared routes produced no eligible market-book candidates")
    plan = dict(rules=asdict(RULES), mature_by=MATURE_BY, archive_sha256=archive_sha,
                uses_independent_indexes=use_index, index_sha256=index_sha,
                cases=cases, counts=diagnostics)
    write_json(path, plan)
    return plan


def stressed_stock_exit(tapes, specs, symbol, at, direction, penalty):
    times, tape = tapes[specs[symbol]["instId"]]
    position = bisect_right(times, at)-1
    if position < 0 or at-times[position] > 60_000:
        raise ValueError("no recent trade at simulated exit")
    sign = 1 if direction == "LONG" else -1
    return tape[position].price*(1-sign*penalty/10_000)


def shared_account(new_stocks, specs, sizes, books, tapes):
    crypto, previous, common_specs, common_sizes, common_books, sources, _, _, common_exit, unknown = joint.joint_inputs()
    relevant = choose_rows(crypto, previous)
    if any(joint.START <= r["epoch"] < COMMON_L2_END for r in unknown):
        raise ValueError("unobserved control entry quote in shared window")
    new_stocks = [r for r in new_stocks if joint.START <= r["epoch"] < COMMON_L2_END]
    original = list(joint.crypto_cadence(r for r in relevant if r["bot"] == "crypto"))
    prior_stock = [r for r in relevant if r["bot"] == "stocks"]
    crypto_symbols = {r["symbol"] for r in original}
    for row in new_stocks:
        symbol, at = row["symbol"], row["entry_ms"]
        if symbol in crypto_symbols:
            raise ValueError("the two routes disagree over the market identity")
        if at in common_books[symbol] and common_books[symbol][at] != books[symbol][at]:
            raise ValueError("stock quote differs from shared source")
        common_books[symbol][at] = books[symbol][at]

    def stock_cases(proposals):
        return original + list(limit_bursts(proposals, max_same_side=2))

    cohorts = dict(crypto_only=original, prior_stock=stock_cases(prior_stock),
                   new_stock=stock_cases(new_stocks))
    new_symbols = {r["symbol"] for r in new_stocks}
    out = {}
    source = ROOT / "okx_joint_exact_exit_nov21.json"
    baseline = read_json(source)["scenarios"]["both_base_cadence|cap6|requested5|200ms"]["stress15bps"]
    for penalty in (15, 30):
        fallback = common_exit(penalty)
        group = {}
        for label, cohort in cohorts.items():
            def price(symbol, at, direction, quantity):
                if label == "new_stock" and symbol in new_symbols:
                    return stressed_stock_exit(tapes, specs, symbol, at, direction, penalty)
                return fallback(symbol, at, direction, quantity)

            result = simulate(cohort, common_specs | specs, common_sizes | sizes, common_books, clock=60,
                              lag=200, setting="requested5", max_positions=6, exit_price=price)
            if label == "prior_stock" and penalty == 15 and result.get("closed") != baseline.get("closed"):
                raise ValueError("shared comparison no longer reproduces the prior stock cash ledger")
            data = detailed(result)
            if "cash_pnl_usd" in data:
                data["status"] = "SCORED_REAL_ENTRY_BOOK_EXIT_STRESS"
            group[label] = data
        out[str(penalty)] = group
    return dict(start=joint.START, end=COMMON_L2_END, total_slots=6, prior_stock_quotes=len(prior_stock),
                new_stock_quotes=len(new_stocks), input_sha256=sources | {source.name: file_hash(source)}, results=out)


def score(plan, data):
    capture_path, repair_path = data / "capture.json", data / "trade_repair.json"
    captured, repair = map(read_json, (capture_path, repair_path))
    if captured["plan_sha256"] != file_hash(data/"plan.json") or repair["capture_sha256"] != file_hash(capture_path):
        raise ValueError("frozen stock inputs changed")
    specs = captured["specs"]
    tick_sizes = {s: float(v["tickSz"]) for s, v in specs.items()}
    rows = [r for group in plan["cases"].values() for r in group]
    books, tapes = defaultdict(dict), {}
    for row in rows:
        books[row["symbol"]][row["entry_ms"]] = row["book"]
    for symbol, intervals in repair["intervals"].items():
        captured["trades"][symbol]["repaired"] = intervals

    def load(item):
        tape = load_repaired_tape(item)
        tapes[item["instrument"]] = ([t.ms for t in tape], tape)
        return tape

    checks = resolve_all(rows, {"files": captured["trades"]}, {"symbols": captured["reference"]},
                         tick_sizes, trade_loader=load)

    def evaluate(rows, penalty):
        if not rows:
            return dict(status="NO_QUALIFIED_ENTRIES", trades=0)

        def exit_price(symbol, at, direction, _):
            return stressed_stock_exit(tapes, specs, symbol, at, direction, penalty)

        subset = list(limit_bursts(rows, max_same_side=2))
        result = detailed(simulate(subset, specs, tick_sizes, books, clock=60, lag=200, setting="requested5",
                                   exit_price=exit_price, max_positions=6))
        if "cash_pnl_usd" in result:
            result["status"] = "SCORED_REAL_ENTRY_BOOK_EXIT_STRESS"
        return result

    results = {}
    for cohort, window in plan["cases"].items():
        variants = dict(all=window, regular=[r for r in window if r["phase"]=="regular"])
        for route in ("trend_pullback", "range_reversal"):
            variants[route] = [r for r in window if r["route"]==route]
        results[cohort] = {name: {str(bps): evaluate(subset, bps) for bps in (15, 30)}
                           for name, subset in variants.items()}
    report = dict(capital_usd=CAPITAL, margin_fraction=FRACTION, leverage=LEVERAGE, rules=plan["rules"],
                  source_hashes={p.name: file_hash(p) for p in (data/"plan.json", capture_path, repair_path)},
                  uses_independent_indexes=plan.get("uses_independent_indexes", False),
                  eligible={name: Counter(r["route"]+":"+r["phase"] for r in rows)
                            for name, rows in plan["cases"].items()}, diagnostics=plan["counts"],
                  results=results, tape_checks=checks, live_activation_supported=False,
                  limitations=["Stocks-only on a modeled six-position account. The live route is different.",
                    "Universe preselection uses four strictly earlier quote windows, each at original receipt time.",
                    "Eligibility and quantity use actual public entry books; exits are trade-print stress marked.",
                    "Missing exit quotes and unverified trade paths are not counted as flat.",
                    "Both date groups are short, previously researched market windows, with separate $130 starts."])
    if report["uses_independent_indexes"]:
        report["limitations"].append("QQQ/SPY context is from the earlier completed mark observer window.")
        report["shared_common_window"] = shared_account(rows, specs, tick_sizes, books, tapes)
    write_json(ROOT / f"{data.name}.json", report)
    for cohort, values in results.items():
        print(cohort, {k: {p: (x["status"], x.get("trades"), x.get("cash_pnl_usd"), x.get("reason"))
                           for p, x in levels.items()} for k, levels in values.items()}, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--declare-only", action="store_true")
    parser.add_argument("--index-context", action="store_true")
    args = parser.parse_args()
    directory = DATA.with_name(f"{STEM}_indexes") if args.index_context else DATA
    plan = freeze(directory, use_index=args.index_context)
    print(plan["counts"], flush=True)
    if not args.declare_only:
        if not (directory/"capture.json").exists():
            capture(data=directory)
        if not (directory/"trade_repair.json").exists():
            repair_trades(data=directory)
        score(plan, directory)
