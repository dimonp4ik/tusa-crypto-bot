"""Replay runner protection when a client's chosen TP1 size rounds to zero."""

from collections import Counter
from decimal import Decimal
from functools import partial
import json
import pickle

from okx_calendar_exposure_nov21 import read_case
from okx_cross_bot_overlap_nov21 import DATA as JOINT_DATA, crypto_cadence, joint_inputs
from okx_entry_screen_cash_nov21 import summarize
from okx_joint_exact_exit_nov21 import choose_rows
from okx_xperp_cash_tick_nov19 import TP1_SETTINGS, simulate
from okx_xperp_multiday_replay_nov20 import FOLDER, ROOT, exit_books, file_hash, resolve_all
from okx_xperp_tick_path_nov19 import load_trade_tape, resolve_ticks
from src.tp1_plan import plan_tp1_close
from stock_liquid_quote_routes_nov21 import limit_bursts, write_json


TRIALS = ("be_1r", "half_stop_then_1r")


def read_json(path):
    return json.loads(path.read_bytes())


def merge_trades(a, b):
    combined = dict(a)
    for symbol, other in b.items():
        if symbol not in combined:
            combined[symbol] = other
            continue
        current = combined[symbol]
        if current["instrument"] != other["instrument"]:
            raise ValueError("old crypto instrument changed between windows")
        shared = current["days"].keys() & other["days"].keys()
        if any(current["days"][d]["sha256"] != other["days"][d]["sha256"] for d in shared):
            raise ValueError("overlapping trade archive changed")
        combined[symbol] = dict(current, days=current["days"] | other["days"],
                                complete=current["complete"] and other["complete"])
    return combined


def old_inputs():
    before = FOLDER.with_name("nov5_8_entry_screen")
    history = ROOT / "okx_xperp_multiday_cash_nov20.json"
    a_specs, a_rows, books = read_case(before, "verified_public_paths.pkl")
    b_specs, b_rows, b_books = read_case(FOLDER, "entry_exit_samples_nov21.pkl", file_hash(history))
    for symbol, other in b_books.items():
        if any(books[symbol][stamp] != other[stamp] for stamp in books[symbol].keys() & other.keys()):
            raise ValueError("earlier exit quotes differ for the same timestamp")
        books[symbol].update(other)
    specs = a_specs | b_specs
    universe = read_json(ROOT / "native_universe_nov19_snapshot.json")["instruments"]
    tick_by_id = {r["instId"]: float(r["tickSz"]) for r in universe if r.get("tickSz")}
    ticks = {s: tick_by_id[v["instId"]] for s, v in specs.items()}
    a_capture, b_capture = read_json(before / "capture.json"), read_json(FOLDER / "capture.json")
    archive = a_capture["books"] | b_capture["books"]
    trades = merge_trades(a_capture["trades"], b_capture["trades"])
    reference = read_json(ROOT / "paired_contract_volume_nov19/manifest.json")["symbols"]
    return a_rows + b_rows, specs, ticks, books, archive, trades, reference


def recent_inputs():
    crypto, stocks, specs, ticks, books, hashes, *_ = joint_inputs()
    rows = choose_rows(crypto, stocks)
    stocks = [r for r in rows if r["bot"] == "stocks"]
    rows = list(crypto_cadence(r for r in rows if r["bot"] == "crypto")) + list(
        limit_bursts(stocks, max_same_side=2))
    saved = pickle.loads((JOINT_DATA / "joint_exact_exit_samples.pkl").read_bytes())
    snapshots, extra, _ = saved["result"]
    stock_report = ROOT / "stock_liquid_quote_routes_nov21.json"
    if saved["sources"] != dict(hashes, **{str(stock_report): file_hash(stock_report)}):
        raise ValueError("shared exit quote cache does not match the frozen inputs")
    for symbol, book in snapshots.items():
        books[symbol].update(book)
    archive = read_json(JOINT_DATA / "crypto_books.json")["books"] | extra | read_json(stock_report)["close_book_files"]
    captured = read_json(JOINT_DATA / "crypto_capture.json")
    return rows, specs, ticks, books, archive, captured["trades"], captured["references"]


def new_paths(rows, specs, ticks, books, archive, trades, reference):
    active = [r for r in rows if r["bot"] == "crypto" and r["entry_status"] == "eligible"]
    tapes = {}

    def tape_for(item):
        key = item["instrument"]
        if key not in tapes:
            tapes[key] = load_trade_tape(item)
        return tapes[key]

    updated = []
    evidence = {}
    for name in TRIALS:
        proposals = [dict(r) for r in active]
        evidence[name] = resolve_all(
            proposals, {"files": trades}, {"symbols": reference}, ticks, trade_loader=tape_for,
            path_resolver=partial(resolve_ticks, runner_policy=name))
        for source, trial in zip(active, proposals):
            source.setdefault("runner_paths", {})[name] = trial["path"]
            if source["path"][60] != trial["path"][60]:
                updated.append(trial)
    downloaded, quote_checks = exit_books(updated, {"books": archive}, specs, books)
    return dict(tape=evidence, changed_candidate_profiles=len(updated),
                new_exit_archives=downloaded, exit_quote_checks=quote_checks)


def bank_diagnostics(rows, closed, specs, setting):
    by_key = {(r["symbol"], r["entry_ms"]): r for r in rows}
    fraction, minimum = TP1_SETTINGS[setting]
    unbanked = Counter()
    targets = Counter()
    possible = {"crypto": Counter(), "stocks": Counter()}
    for trade in closed:
        symbol, bot = trade["symbol"], trade["bot"]
        spec = specs[symbol]
        r = by_key[symbol, trade["entry_ms"]]
        if bot == "crypto" and r["path"][60]["bank"] and not plan_tp1_close(
                trade["qty"], fraction, spec, ensure_minimum=minimum)["size"]:
            unbanked[trade["code"]] += 1
        if trade["code"] != "TARGET" or trade["runner_px"] is None:
            continue
        targets[bot] += 1
        runner = Decimal(str(trade["qty"])) - Decimal(str(trade["bank_qty"]))
        min_close = max(Decimal(str(spec["minSz"])), Decimal(str(spec["lotSz"])))
        for pct in (10, 20, 30):
            tail = Decimal(str(plan_tp1_close(float(runner), pct, spec)["size"]))
            if tail > 0 and runner - tail >= min_close:
                possible[bot][pct] += 1
    return dict(baseline_tp1_hit_below_min_lot=unbanked, baseline_target_closes=targets,
                feasible_tail_division=possible)


def analyze(cohort, material, reference):
    rows, specs, ticks, books, archive, trades, frames = material
    cap = 12 if cohort == "nov5_14" else 6
    controls = dict(clock=60, lag=200, max_positions=cap)
    checked = new_paths(rows, specs, ticks, books, archive, trades, frames)
    results = {}
    for setting in ("requested5", "half_minimum"):
        control = simulate(rows, specs, ticks, books, setting=setting, **controls)
        previous = reference.get(setting)
        if previous and (summarize(control) if cohort == "nov5_14" else control["closed"]) != previous:
            raise ValueError(f"cash control changed for {cohort} {setting}")
        results[setting] = {"baseline": summarize(control)}
        for variant in TRIALS:
            trial = simulate(rows, specs, ticks, books, setting=setting, unbankable_runner=variant, **controls)
            results[setting][variant] = summarize(trial)
        results[setting].update(bank_diagnostics(rows, control.get("closed", []), specs, setting))
    return dict(evidence=checked, cases=results)


def main():
    prior_path = ROOT / "okx_entry_screen_holdout_nov21.json"
    recent_path = ROOT / "okx_joint_exact_exit_nov21.json"
    prior = {"requested5": read_json(prior_path)["continuous_cash_results"]["results"]["baseline"]}
    joint = {s: read_json(recent_path)["scenarios"][f"both_base_cadence|cap6|{s}|200ms"]["book"]["closed"]
             for s in ("requested5", "half_minimum")}
    results = dict(nov5_14=analyze("nov5_14", old_inputs(), prior),
                   nov17_19_joint=analyze("nov17_19_joint", recent_inputs(), joint))
    report = dict(source_hashes={str(p): file_hash(p) for p in (prior_path, recent_path)},
                  results=results, control_parity=True, live_activation_supported=False,
                  limitations=[
                      "Only entries whose requested partial rounds to zero use the alternative protection.",
                      "Bankability uses the account's simulated entry size at that instant.",
                      "An amendment applies from the next minute boundary. Targets and entry sizing are fixed.",
                      "Market exits use observed public books after the declared 200ms delay.",
                      "Third-target counts measure contract-size feasibility at TP2.",
                  ])
    write_json(ROOT / "okx_unbankable_tp1_nov21.json", report)
    for name, cohort in results.items():
        print(name, json.dumps(cohort["cases"], indent=2), flush=True)


if __name__ == "__main__":
    main()
