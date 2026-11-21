"""Test a fixed narrow-stop reversal rule on later verified OKX prints."""

from bisect import bisect_left, bisect_right
from collections import Counter, defaultdict
from pathlib import Path
import pickle

import okx_cross_bot_overlap_nov21 as joint
from crypto_narrow_reversal_nov22 import reversal
from okx_crypto_risk_range_nov22 import candidate_plan
from okx_joint_exact_exit_nov21 import ENTRY_END as DISCOVERY_STOP
from okx_tp1_cash_sensitivity_nov22 import detailed
from okx_unbankable_tp1_nov21 import read_json
from okx_xperp_cash_tick_nov19 import CAPITAL, FRACTION, LEVERAGE, simulate
from okx_xperp_multiday_replay_nov20 import ROOT, file_hash, resolve_all
from okx_xperp_tick_path_nov19 import load_trade_tape
from src.okx_trader import calc_contracts, round_to_tick
from stock_liquid_quote_routes_nov21 import write_json


STRESSES_BPS = ((5, 15), (15, 30))
ENTRY_LAG = joint.ENTRY_OFFSET_MS
EXIT_LAG = 1000
MAX_PRINT_AGE = 60_000


def recent_print(tape, at, *, end_inclusive):
    times, prints = tape
    select = bisect_right if end_inclusive else bisect_left
    index = select(times, at) - 1
    if index < 0 or at - times[index] > MAX_PRINT_AGE:
        return None
    return prints[index]


def stress_entry(candidates, loader, capture, bps):
    books = defaultdict(dict)
    scenarios = []
    for row in candidates:
        symbol = row["symbol"]
        spec = capture["specs"][symbol]
        at = row["epoch"] * 1000 + ENTRY_LAG
        last = recent_print(loader(symbol), at, end_inclusive=False)
        status = "unknown_quote"
        price = None
        if last is not None:
            sign = 1 if row["direction"] == "LONG" else -1
            price = last.price * (1 + sign * bps / 10_000)
            tp1 = round_to_tick(row["entry"] + .5 * (row["entry"] - row["stop"]), float(spec["tickSz"]))
            if not sign * row["stop"] < sign * price < sign * tp1:
                status = "outside_tp1_bracket"
            elif not calc_contracts(CAPITAL * FRACTION, LEVERAGE, price, spec):
                status = "below_minimum_size"
            else:
                status = "eligible"
                # Compatibility with the common cash ledger. Supply one flat level at the stressed fill.
                # These levels and timestamp are synthetic, with no claim about visible spread or depth.
                bid = price if sign < 0 else price * .9999
                ask = price if sign > 0 else price * 1.0001
                books[symbol][at] = dict(ts=at, bids=[(bid, 1e12)], asks=[(ask, 1e12)])
        scenarios.append(dict(row, entry_status=status, entry_ms=at, reference_entry=price,
                              stressed_entry_bps=bps))
    return scenarios, books


def score(captured, base, alt, btc, tapes, entry_bps, exit_bps):
    def load(symbol):
        if symbol not in tapes:
            tape = load_trade_tape(captured["trades"][symbol])
            tapes[symbol] = [t.ms for t in tape], tape
        return tapes[symbol]

    specs = captured["specs"]
    ticks = {symbol: float(spec["tickSz"]) for symbol, spec in specs.items()}
    symbols = {spec["instId"]: symbol for symbol, spec in specs.items()}
    candidates, books = stress_entry(base + alt, load, captured, entry_bps)
    checks = resolve_all(candidates, {"files": captured["trades"]}, {"symbols": captured["references"]}, ticks,
                         trade_loader=lambda one: load(symbols[one["instrument"]])[1])

    def close_price(symbol, at, direction, _):
        anchor = recent_print(load(symbol), at, end_inclusive=True)
        if anchor is None:
            raise ValueError("no recent public print at the stressed exit")
        sign = 1 if direction == "LONG" else -1
        return anchor.price * (1 - sign * exit_bps / 10_000)

    original = [r for r in candidates if "band" not in r]
    narrow = [r for r in candidates if "band" in r]
    chosen = [r for r in narrow if reversal(btc, r["epoch"], r["direction"])]
    selections = dict(control=original, all_narrow=original+narrow, narrow_reversal=original+chosen)
    cases = {}
    for name, cohort in selections.items():
        admissions = list(joint.crypto_cadence(cohort))
        unknown = [r for r in admissions if r["entry_status"] == "unknown_quote"]
        if unknown:
            outcome = dict(status="UNSCORABLE_UNKNOWN_ENTRY",
                           missing=[(r["symbol"], r["epoch"]) for r in unknown])
        else:
            outcome = detailed(simulate(admissions, specs, ticks, books, clock=60, lag=EXIT_LAG,
                                        setting="requested5", exit_price=close_price))
            if "cash_pnl_usd" in outcome:
                outcome["status"] = "SCORED_PRINT_STRESS_PROXY"
        cases[name] = dict(outcome, proposals=Counter(r["entry_status"] for r in admissions))
    return dict(cases=cases, tape_checks=checks, candidate_count=len(candidates), reversal_count=len(chosen))


def main():
    capture_path = joint.DATA / "crypto_capture.json"
    frozen_path = joint.DATA / "crypto_plan.json"
    captured, frozen = map(read_json, (capture_path, frozen_path))
    alternates, _ = candidate_plan(captured, frozen)
    base = [r for r in frozen["rows"] if DISCOVERY_STOP <= r["epoch"] < joint.ENTRY_END]
    narrow = [r for r in alternates["below_1pct"] if DISCOVERY_STOP <= r["epoch"] < joint.ENTRY_END]
    ref = captured["references"]["BTCUSDT"]
    btc_path = Path(ref["path"])
    if file_hash(btc_path) != ref["sha256"]:
        raise ValueError("Bitcoin window hash does not match the frozen data")
    btc = pickle.loads(btc_path.read_bytes())
    print("Later distinct entries", len(base), "control", len(narrow), "additional", flush=True)
    tapes = {}
    results = {f"{a}|{b}": score(captured, base, narrow, btc, tapes, a, b) for a, b in STRESSES_BPS}
    output = dict(start=DISCOVERY_STOP, stop=joint.ENTRY_END, stake_usd=CAPITAL, margin_fraction=FRACTION,
                  leverage=LEVERAGE, candidate_count=len(base)+len(narrow), results=results,
                  source_hashes={p.name: file_hash(p) for p in (capture_path, frozen_path)},
                  live_activation_supported=False, limitations=[
                      "Entries are strictly separated by time from the narrower-stop L2 discovery window.",
                      "Public BTC, crypto candle and trade inputs were frozen before this exit study.",
                      "Public trade order and 15-minute contract volume reconcile. Fill prices use adverse stresses.",
                      "No proof of spread or auction depth. No stock route or execution under shared concurrent writers.",
                      "The same rule also receives a separate earlier four-month candle falsification check.",
                  ])
    write_json(ROOT / "okx_crypto_narrow_reversal_holdout_nov22.json", output)
    for key, evaluated in results.items():
        print(key, {name: (x["status"], x.get("trades"), x.get("cash_pnl_usd"), x.get("missing", x.get("reason")))
                    for name, x in evaluated["cases"].items()}, flush=True)


if __name__ == "__main__":
    main()
