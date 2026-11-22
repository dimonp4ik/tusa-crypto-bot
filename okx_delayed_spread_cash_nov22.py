"""Value wide-spread retries on AAVE and AVAX with complete post-trigger order books."""

from bisect import bisect_left
from collections import Counter, defaultdict
import pickle

import okx_cross_bot_overlap_nov21 as joint
from crypto_scan_priority_holdout_nov22 import DATA as NOV20_DATA, load_cohort
from okx_avax_aave_retry_nov21 import SYMBOLS
from okx_crypto_risk_range_nov22 import attach_entry_quotes, merge_books
from okx_entry_screen_holdout_nov21 import ENTRY_END as NOV09
from okx_joint_exact_exit_nov21 import ENTRY_END as COMMON_END, choose_rows
from okx_tp1_cash_sensitivity_nov22 import detailed, verify_control
from okx_unbankable_tp1_nov21 import old_inputs, read_json
from okx_xperp_cash_tick_nov19 import CAPITAL, CLOSE_LAGS, FRACTION, LEVERAGE, simulate
from okx_xperp_l2_entry_replay_nov19 import attempt
from okx_xperp_multiday_replay_nov20 import ROOT, exit_books, file_hash, key_for, pull_samples, resolve_all
from okx_xperp_tick_path_nov19 import load_trade_tape
from src.okx_trader import round_to_tick
from stock_liquid_quote_routes_nov21 import limit_bursts, write_json


STEM = "okx_delayed_spread_cash_nov22"
MAX_DELAYS = (30_000, 60_000, 120_000)
START_LAG = 10_000
POLICIES = dict(short_cautious=(60_000, True), full_cautious=(120_000, True),
                full_instantaneous=(120_000, False))


def delayed_samples(rows, archives):
    requests = defaultdict(set)
    for r in rows:
        for delay in MAX_DELAYS:
            at = r["epoch"]*1000 + delay
            requests[key_for(r["symbol"], at)].add(at)
    jobs = []
    for key, stamps in sorted(requests.items()):
        day, symbol = key.split(":", 1)
        record = archives.get(key, {})
        if "path" not in record:
            raise ValueError(f"no observed delay archive for {key}")
        jobs.append((symbol, int(day), record, sorted(stamps)))
    return pull_samples(jobs, stage="delayed entry books")


def earlier_level_touched(row, events, times, end, near, stop):
    start = (row["epoch"]*1000 + START_LAG)
    sign = 1 if row["direction"] == "LONG" else -1
    left, right = bisect_left(times, start), bisect_left(times, end)
    return any(not sign*stop < sign*events[i].price < sign*near for i in range(left, right))


def build_variant(row, spec, tick, quotes, tapes, *, max_delay, require_unspent):
    attempts = []
    symbol = row["symbol"]
    near = round_to_tick(row["entry"] + .5*(row["entry"]-row["stop"]), tick)
    stop = round_to_tick(row["stop"], tick)
    sign = 1 if row["direction"] == "LONG" else -1
    times, events = tapes[symbol]
    final_status = "wide"
    for delay in MAX_DELAYS:
        if delay > max_delay:
            break
        at = row["epoch"]*1000 + delay
        if require_unspent and earlier_level_touched(row, events, times, at, near, stop):
            return dict(row, entry_status="original_bracket_already_touched", delayed_attempts=attempts)
        value = attempt(quotes.get(at), at, row["direction"], spec, stop, near)
        attempts.append(dict(ms=at, **value))
        final_status = value["status"]
        if final_status == "wide":
            continue
        if final_status != "eligible":
            break
        price = value["worst"]
        rr = sign*(row["target"]-price)/(sign*(price-stop))
        if rr < 1.5:
            final_status = "below_delayed_reward_risk"
            break
        return dict(row, entry_status="eligible", entry_ms=at, reference_entry=value["average"],
                    delayed_attempts=attempts, delay_ms=delay, delayed_rr=rr)
    return dict(row, entry_status=final_status, delayed_attempts=attempts)


def ordered(rows, cap):
    if cap != 6:
        # The old cohort contains proposals already filtered by its original cadence.
        return rows
    crypto = joint.crypto_cadence(r for r in rows if r["bot"] == "crypto")
    stock = limit_bursts((r for r in rows if r["bot"] == "stocks"), max_same_side=2)
    return [*crypto, *stock]


def score(name, rows, specs, ticks, books, archives, trades, frames, *, cap):
    retired = [r for r in rows if r["bot"] == "crypto" and r["symbol"] in SYMBOLS and r["entry_status"] == "wide"]
    if any(r["entry_status"] == "unknown_quote" for r in rows):
        raise ValueError(f"{name} has an unknown original book")
    new, entry_evidence = delayed_samples(retired, archives)
    merge_books(books, new)
    tapes = {}
    for symbol in sorted({r["symbol"] for r in retired}):
        events = load_trade_tape(trades[symbol])
        tapes[symbol] = [t.ms for t in events], events
    by_id = {specs[s]["instId"]: s for s in tapes}
    sets = {label: [build_variant(r, specs[r["symbol"]], ticks[r["symbol"]], new[r["symbol"]], tapes,
                                   max_delay=maximum, require_unspent=untouched) for r in retired]
            for label, (maximum, untouched) in POLICIES.items()}
    checked = [r for variants in sets.values() for r in variants if r["entry_status"] == "eligible"]
    # Reconcile complete trade volume over the signal horizon before relying on intrabar touches.
    evidence = resolve_all(checked + [dict(r) for r in retired], {"files": trades}, {"symbols": frames}, ticks,
                           trade_loader=lambda item: tapes[by_id[item["instrument"]]][1])
    if any("error" in v or v["mismatches"] for v in evidence.values()):
        raise ValueError(f"{name} has irreconcilable pre-entry or outcome trades")
    more, exit_evidence = exit_books(checked, {"books": archives}, specs, books)
    print(name, "wide proposals", len(retired), "delayed candidates", {k: sum(r["entry_status"]=="eligible" for r in v)
                                   for k, v in sets.items()}, flush=True)
    results = {}
    variants = {"current": rows}
    for key, overrides in sets.items():
        replacements = {(r["symbol"], r["epoch"]): r for r in overrides}
        variants[key] = [replacements.get((r["symbol"], r["epoch"]), r) if r["bot"]=="crypto" else r for r in rows]

    for key, material in variants.items():
        candidates = ordered(material, cap)
        prospective = {(r["symbol"], r["entry_ms"]) for r in candidates if r.get("delay_ms")}
        values = {}
        for lag in CLOSE_LAGS:
            if any(r["entry_status"] == "unknown_quote" for r in candidates):
                result = dict(status="UNSCORABLE_UNKNOWN_DELAYED_ENTRY")
            else:
                result = simulate(candidates, specs, ticks, books, clock=60, lag=lag, setting="requested5",
                                  max_positions=cap)
            delayed = [{"symbol": t["symbol"], "entered": t["entry_ms"], "outcome": t["code"],
                        "usd": t["bank_pnl"]+t["runner_pnl"]} for t in result.get("closed", [])
                       if (t["symbol"], t["entry_ms"]) in prospective]
            values[str(lag)] = dict(detailed(result), entered_after_retry=delayed)
        results[key] = values
    return dict(cases=results, delayed_candidates=sets, archive_entry=entry_evidence, archive_exit=exit_evidence,
                archive_exit_extra={key: v["sha256"] for key, v in more.items()}, tape=evidence, wide=len(retired))


def older():
    rows, specs, ticks, books, archives, trades, frames = old_inputs()
    verify_control(rows, specs, ticks, books, 12, ROOT/"okx_entry_screen_holdout_nov21.json", joint=False)
    return score("nov09_14", [r for r in rows if r["epoch"] >= NOV09], specs, ticks, books, archives, trades, frames,
                 cap=12)


def joint_rows():
    crypto, stocks, specs, ticks, books, hashes, _, _, _, unknown = joint.joint_inputs()
    if any(joint.START <= r["epoch"] < COMMON_END for r in unknown):
        raise ValueError("unknown quote within shared overlap")
    rows = choose_rows(crypto, stocks)
    saved = pickle.loads((joint.DATA/"joint_exact_exit_samples.pkl").read_bytes())
    stocks_report = ROOT/"stock_liquid_quote_routes_nov21.json"
    if saved["sources"] != hashes | {str(stocks_report): file_hash(stocks_report)} or saved["entry_end"] != COMMON_END:
        raise ValueError("shared exact archive changed")
    snapshots, more, _ = saved["result"]
    merge_books(books, snapshots)
    verify_control(ordered(rows, 6), specs, ticks, books, 6, ROOT/"okx_joint_exact_exit_nov21.json", joint=True)
    available = read_json(joint.DATA/"crypto_books.json")["books"] | more
    captured = read_json(joint.DATA/"crypto_capture.json")
    return score("nov17_19_joint", rows, specs, ticks, books, available, captured["trades"],
                 captured["references"], cap=6)


def latest():
    captured, rows, _ = load_cohort(NOV20_DATA/"crypto_capture.json")
    rows = [r for r in rows if r["symbol"] in SYMBOLS]
    specs = captured["specs"]
    ticks = {s: float(spec["tickSz"]) for s, spec in specs.items()}
    plan = dict(rows=rows, specs=specs, parameters=dict(crypto_offset_ms=START_LAG, retry_offsets_ms=[0, 200, 800]))
    available = {}
    actual, books, baseline_quotes = attach_entry_quotes(rows, plan, available, ticks)
    trades, frames = captured["trades"], captured["references"]
    baseline_tape = resolve_all(actual, {"files": trades}, {"symbols": frames}, ticks)
    extra, exit_checks = exit_books(actual, {"books": available}, specs, books)
    value = score("nov20_targeted", actual, specs, ticks, books, available | extra, trades, frames, cap=6)
    return dict(value, baseline_quotes=baseline_quotes, baseline_tape=baseline_tape, baseline_exits=exit_checks)


def main():
    results = dict(nov09_14=older(), nov17_19_joint=joint_rows(), nov20_targeted=latest())
    write_json(ROOT/f"{STEM}.json", dict(capital_usd=CAPITAL, margin_fraction=FRACTION, leverage=LEVERAGE,
        start_lag_ms=START_LAG, additional_offsets_ms=MAX_DELAYS, live_activation_supported=False,
        policies=POLICIES, windows=results, limitations=[
            "Only already-wide AAVE and AVAX get extra checks. A tight book still needs reward/risk >= 1.5 and room to TP1.",
            "Cautious variants reject either original stop or TP1 touched during the wait, using verified ordered trades.",
            "New positions are replayed at observed delayed bids/asks. Exits use public books and two delays.",
            "The fresh November 20 sample contains the two targeted instruments, not the full account.",
        ]))
    for name, group in results.items():
        print(name, {case: (lags["200"].get("cash_pnl_usd"), lags["200"]["entered_after_retry"])
                     for case, lags in group["cases"].items()}, flush=True)


if __name__ == "__main__":
    main()
