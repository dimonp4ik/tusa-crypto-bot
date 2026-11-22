"""Test how choosing among simultaneous crypto proposals changes account cash."""

from collections import Counter, defaultdict, deque
from itertools import groupby
from pathlib import Path
import pickle
from unittest.mock import patch

import okx_cross_bot_overlap_nov21 as joint
from account_cash_guard_nov17_audit import runtime_gate
from crypto_active_entry_runner_nov19_audit import activity_gate
from crypto_minimum_bank_runner_nov19_audit import CLIENTS, cash_result, checked_row
from crypto_narrow_reversal_nov22 import bitcoin_returns
from crypto_native_price_pool_nov19 import MANIFEST, load_pool
from crypto_nearer_target_long_history_nov22 import EXIT_STRESS
from okx_joint_exact_exit_nov21 import ENTRY_END, choose_rows
from okx_tp1_cash_sensitivity_nov22 import detailed
from okx_unbankable_tp1_nov21 import read_json
from okx_xperp_cash_tick_nov19 import simulate
from okx_xperp_multiday_replay_nov20 import ROOT, file_hash
from src.crypto_profit_first_material_shadow_router import (
    ARTIFACT_PATH, REQUIRED_BTC_HISTORY_BARS, REQUIRED_SYMBOL_HISTORY_BARS, model_probability, shadow_setups,
)
from src.crypto_venue_router import btc_context, family_signals
from stock_liquid_quote_routes_nov21 import limit_bursts, write_json


POLICIES = ("alphabetical", "model_quality", "market_aligned_then_quality", "prefer_hedge_in_countertrend")


def order_scan(rows, policy):
    if policy == "alphabetical":
        return sorted(rows, key=lambda r: r["symbol"])
    if policy not in POLICIES:
        raise ValueError("unknown publication priority")
    ordered = sorted(rows, key=lambda r: (-r["model_quality"], r["symbol"]))
    if policy == "market_aligned_then_quality":
        return sorted(ordered, key=lambda r: r["market_alignment"] != "aligned")
    if policy == "prefer_hedge_in_countertrend" and ordered and ordered[0]["market_alignment"] == "opposed":
        first, other = ordered[0], ordered[1:]
        return [first] + sorted(other, key=lambda r: r["direction"] == first["direction"])
    return ordered


def alignment(frame, at, side):
    values = bitcoin_returns(frame, at, side)
    return "aligned" if min(values) > 0 else "opposed" if max(values) < 0 else "mixed"


def book_cadence(rows, policy):
    admitted = deque()
    for at, group in groupby(sorted(rows, key=lambda r: r["epoch"]), key=lambda r: r["epoch"]):
        while admitted and admitted[0] <= at - 3600:
            admitted.popleft()
        for row in order_scan(list(group), policy):
            if row["entry_status"] != "eligible":
                yield row
            elif len(admitted) >= 2:
                yield dict(row, entry_status="publication_rate_limit")
            else:
                admitted.append(at)
                yield row


def attribute(closed, rows):
    indexed = {(r["symbol"], r["entry_ms"]): r for r in rows if r["bot"] == "crypto"}
    first_per_scan = set()
    recent = defaultdict(deque)
    same_scan, rolling_hour = defaultdict(Counter), defaultdict(Counter)
    for trade in closed:
        if trade["bot"] != "crypto":
            continue
        row = indexed[trade["symbol"], trade["entry_ms"]]
        scan = row["epoch"], row["direction"]
        hour = recent[row["direction"]]
        while hour and hour[0] <= row["entry_ms"]-3_600_000:
            hour.popleft()
        for stats, repeated in ((same_scan, scan in first_per_scan), (rolling_hour, bool(hour))):
            name = f"{row['market_alignment']}|{'additional' if repeated else 'first'}"
            stats[name].update(trades=1, stop=int(trade["code"] == "SL"),
                               usd=trade["bank_pnl"]+trade["runner_pnl"])
        first_per_scan.add(scan)
        hour.append(row["entry_ms"])
    return dict(same_scan=dict(same_scan), same_direction_hour=dict(rolling_hour))


def reconstruct_rank(captured):
    btc_item = captured["references"]["BTCUSDT"]
    path = Path(btc_item["path"])
    if file_hash(path) != btc_item["sha256"]:
        raise ValueError("Bitcoin context changed")
    btc = pickle.loads(path.read_bytes())
    annotations = {}

    def observe(candles, btc_candles, **args):
        setups = shadow_setups(candles, btc_candles, **args)
        for row in setups:
            at, side = args["entry_time"], row["direction"]
            annotations[args["symbol"], at] = dict(model_quality=row["model_quality"],
                 market_alignment=alignment(btc, at, side))
        return setups

    with patch.object(joint, "shadow_setups", observe):
        rows, tie_checks = joint.reconstruct(captured)
    return rows, annotations, tie_checks


def with_book_evidence():
    capture_path = joint.DATA / "crypto_capture.json"
    plan_path = joint.DATA / "crypto_plan.json"
    captured, reference = map(read_json, (capture_path, plan_path))
    rows, annotations, tie_checks = reconstruct_rank(captured)
    if rows != reference["rows"] or tie_checks != reference["tape_mismatches"]:
        raise ValueError("frozen market projection changed")
    crypto, stocks, specs, ticks, books, hashes, _, _, proxy, unknown = joint.joint_inputs()
    if any(joint.START <= row["epoch"] < ENTRY_END for row in unknown):
        raise ValueError("required baseline entry is missing")
    candidates = choose_rows(crypto, stocks)
    crypto = [dict(row, **annotations[row["symbol"], row["epoch"]]) for row in candidates if row["bot"] == "crypto"]
    stocks = list(limit_bursts((row for row in candidates if row["bot"] == "stocks"), max_same_side=2))
    snapshot = pickle.loads((joint.DATA/"joint_exact_exit_samples.pkl").read_bytes())
    stock_report = ROOT/"stock_liquid_quote_routes_nov21.json"
    if snapshot["sources"] != hashes | {str(stock_report): file_hash(stock_report)} or snapshot["entry_end"] != ENTRY_END:
        raise ValueError("exact exit data changed")
    for symbol, quotes in snapshot["result"][0].items():
        books[symbol].update(quotes)
    original_cash = read_json(ROOT/"okx_joint_exact_exit_nov21.json")["scenarios"]["both_base_cadence|cap6|requested5|200ms"]
    scenarios = {}
    for policy in POLICIES:
        admitted = list(book_cadence(crypto, policy))
        scores = {}
        for label, cap, other in (("crypto", 12, []), ("shared", 6, stocks), ("shared_cap12", 12, stocks)):
            for cost in (None, 15):
                selected = admitted + other
                result = simulate(selected, specs, ticks, books, clock=60, lag=200, setting="requested5",
                                  max_positions=cap, exit_price=proxy(cost) if cost is not None else None)
                if policy == "alphabetical" and label == "shared":
                    if result.get("closed") != original_cash["stress15bps" if cost else "book"].get("closed"):
                        raise ValueError("shared symbol-order control no longer reproduces the archived baseline")
                scores[f"{label}|{cost or 'archived_books'}"] = dict(detailed(result),
                    crypto_phase_attribution=attribute(result.get("closed", []), admitted),
                    settled_trades=[dict(symbol=t["symbol"], bot=t["bot"], entered=t["entry_ms"], exited=t["exit_ms"],
                                         outcome=t["code"], usd=t["bank_pnl"]+t["runner_pnl"])
                                    for t in result.get("closed", [])])
        scenarios[policy] = scores
    return dict(scenarios=scenarios, proposals=len(crypto), decision_start=joint.START, decision_end=ENTRY_END,
                source_hashes=hashes | {str(plan_path): file_hash(plan_path)},
                baseline_clock="Signal bucket and symbol, matching rank order before book retries.")


def historical_inputs():
    pool, metadata = load_pool()
    manifest = read_json(MANIFEST)
    candles = {}
    for symbol, item in manifest["symbols"].items():
        path = Path(item["path"])
        if file_hash(path) != item["sha256"]:
            raise ValueError("native history source changed")
        frame = pickle.loads(path.read_bytes())
        candles[symbol] = frame, {int(t): i for i, t in enumerate(frame["time"])}
    btc = candles["BTCUSDT"][0]
    rows = []
    btc_features = {}
    for r in pool[pool.is_frozen_candidate].to_dict("records"):
        symbol, at = r["symbol"], r["entry_time"]
        frame, lookup = candles[symbol]
        i = lookup[int(at)]
        if not activity_gate(frame, i):
            continue
        window = joint.closed(frame, at, REQUIRED_SYMBOL_HISTORY_BARS)
        options = [s for s in family_signals(window) if s["family"] == r["family"] and s["direction"] == r["direction"]]
        if len(options) != 1:
            raise ValueError("frozen family is absent or ambiguous")
        if at not in btc_features:
            btc_features[at] = btc_context(joint.closed(btc, at, REQUIRED_BTC_HISTORY_BARS))
        context = btc_features[at]
        if context is None:
            raise ValueError("incomplete archived Bitcoin phase")
        r["market_alignment"] = alignment(btc, at, r["direction"])
        r["model_quality"] = model_probability(symbol=symbol, signal=options[0], context=context,
                                               entry_time=at, risk_pct=r["risk_pct"])
        r = checked_row(r, frame, i, "be_tp1", EXIT_STRESS)
        rows.append(dict(r, module="risk_guard_research", module_priority=999,
                         signed_price_return=r["risk_pct"]*r["net_r"],
                         outcome="SL" if r["sequential_outcome"]=="SL" else "TP"))
    return rows, metadata["sha256"]


def longer_history():
    rows, pool_hash = historical_inputs()
    spec_path = ROOT / "shared_xperp_specs_nov14.json"
    control = ROOT / "crypto_active_entry_runner_nov19.json"
    expected = read_json(control)["results"]["be_tp1"]["cash"]
    specs = read_json(spec_path)
    ordering = sorted(rows, key=lambda r: r["entry_time"])
    result = {}
    for policy in POLICIES:
        ranked = [dict(row, source_priority=rank) for _, cohort in groupby(ordering, key=lambda r: r["entry_time"])
                  for rank, row in enumerate(order_scan(list(cohort), policy))]
        selected = runtime_gate(ranked, latch_loss_pause=False)
        unknown = [(r["symbol"], r["entry_time"], r.get("unknown_reason")) for r in selected if not r["known"]]
        values = {c: cash_result(selected, specs, c) for c in CLIENTS} if not unknown else {}
        if policy == "alphabetical" and values != expected:
            raise ValueError("original cash path does not reproduce the published test")
        result[policy] = dict(cash=values, admitted=len(selected), unknown=unknown)
        print("Candle history", policy, {c: (v["summary"]["return_pct"], v["summary"]["cash_drawdown_pct"])
                                         for c, v in values.items()}, flush=True)
    return dict(results=result, candle_pool_sha256=pool_hash, control_parity=True, proposals=len(rows),
                source_hashes={p.name: file_hash(p) for p in (spec_path, control, MANIFEST, ARTIFACT_PATH)})


def main():
    long = longer_history()
    recent = with_book_evidence()
    write_json(ROOT/"crypto_scan_priority_nov22.json", dict(candle_history=long, public_book_window=recent,
        live_activation_supported=False, limitations=[
            "November market replay has all original reconstructed candidates, including those outside the hourly cap.",
            "Market-book replay models no deliberate delay between ranked proposals, routing or exchange acknowledgments.",
            "Candidate alpha ties use symbol; phase is from the two closed BTC windows before entry.",
            "The older replay uses closed market anchors and a candle exit proxy, with legacy activity screening.",
            "Market-book daily loss latch, per-user account data and overlapping intraday mark-to-market are outside this study.",
        ]))
    for label, candidate in recent["scenarios"].items():
        print("Books", label, {k: (v.get("trades"), v.get("cash_pnl_usd"), v.get("reason"))
                               for k, v in candidate.items()}, flush=True)


if __name__ == "__main__":
    main()
