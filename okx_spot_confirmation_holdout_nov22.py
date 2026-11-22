"""Price the predeclared spot checks in the disjoint November 20 L2 window."""

from collections import Counter
from pathlib import Path
import pickle

from crypto_scan_priority_holdout_nov22 import DATA as ARCHIVE, ENTRY_STOP, START, load_cohort
from crypto_scan_priority_nov22 import book_cadence
from okx_crypto_risk_range_nov22 import attach_entry_quotes
from okx_spot_confirmation_nov22 import (
    DATA as SPOT, POLICIES, capture_spot, features, load_native, load_spot, recent_family, veto,
)
from okx_tp1_cash_sensitivity_nov22 import detailed
from okx_unbankable_tp1_nov21 import read_json
from okx_xperp_cash_tick_nov19 import CAPITAL, simulate
from okx_xperp_multiday_capture_nov20 import HOLD
from okx_xperp_multiday_replay_nov20 import ROOT, exit_books, file_hash, resolve_all
from stock_liquid_quote_routes_nov21 import write_json


STEM = "okx_spot_confirmation_holdout_nov22"
ENTRY_LAG = 10_000


def execution_samples(captured, candidates, capture_hash):
    # Cache locally reconstructed public quotes. Reuse them only when the
    # frozen inputs and the archive readers match.
    source_files = tuple(Path(path) for path in (
        "okx_crypto_risk_range_nov22.py", "okx_xperp_multiday_replay_nov20.py",
        "okx_xperp_l2_entry_replay_nov19.py", "okx_xperp_tick_path_nov19.py", "src/execution_guard.py",
    ))
    signature = dict(capture=capture_hash, source=[file_hash(p) for p in source_files])
    location = ARCHIVE / "verified_public_books_nov22.pkl"
    if location.exists():
        saved = pickle.loads(location.read_bytes())
        if saved["signature"] == signature:
            return saved["samples"]
    specs = captured["specs"]
    ticks = {s: float(spec["tickSz"]) for s,spec in specs.items()}
    plan = dict(rows=candidates, specs=specs, start=START, coverage_end=ENTRY_STOP+HOLD["crypto"],
                parameters=dict(crypto_offset_ms=ENTRY_LAG, retry_offsets_ms=[0, 200, 800]))
    archives = {}
    rows, books, entry_checks = attach_entry_quotes(candidates, plan, archives, ticks)
    tape = resolve_all(rows, {"files": captured["trades"]}, {"symbols": captured["references"]}, ticks)
    extra, exit_checks = exit_books(rows, {"books": archives}, specs, books)
    if any("error" in result or result["mismatches"] for result in tape.values()):
        raise ValueError("spot holdout contains an unreconciled trade archive")
    evidence = dict(entry=entry_checks, exit=exit_checks, tape=tape,
                    archives={day: entry["sha256"] for day,entry in (archives|extra).items() if "sha256" in entry})
    samples = (rows, books, evidence)
    temporary = location.with_suffix(".tmp")
    temporary.write_bytes(pickle.dumps(dict(signature=signature, samples=samples), protocol=5))
    temporary.replace(location)
    return samples


def main():
    source = ARCHIVE/"crypto_capture.json"
    captured, proposals, _ = load_cohort(source)
    print("Frozen spot holdout", len(proposals), "native proposals", flush=True)
    rows, books, evidence = execution_samples(captured, proposals, file_hash(source))
    specs = captured["specs"]
    ticks = {s: float(spec["tickSz"]) for s,spec in specs.items()}
    spots = load_spot(capture_spot())
    natives = load_native(captured["references"])
    rows = [dict(r, family=recent_family(r, natives)) for r in rows]
    contexts = {(r["symbol"], r["entry_ms"]): features(spots, natives, r, stamp=r["epoch"]) for r in rows}
    expected_path = ROOT/"crypto_scan_priority_holdout_nov22.json"
    prior = read_json(expected_path)["results"]["alphabetical"]
    results = {}
    for name in POLICIES:
        shortlisted = [dict(r, entry_status="spot_veto")
                       if r["entry_status"]=="eligible" and veto(contexts[r["symbol"],r["entry_ms"]], name) else r for r in rows]
        selected = list(book_cadence(shortlisted, "alphabetical"))
        values = {}
        for lag in (200, 1000):
            no_quote = [(r["symbol"], r["entry_ms"]) for r in selected if r["entry_status"]=="unknown_quote"]
            outcome = (dict(status="UNSCORABLE_UNKNOWN_ENTRY", missing=no_quote) if no_quote else
                       detailed(simulate(selected, specs, ticks, books, clock=60, lag=lag, setting="requested5")))
            if name == "control" and any(outcome.get(key) != prior[str(lag)].get(key)
                    for key in ("cash_pnl_usd", "trades", "return_pct", "drawdown_pct", "by_bot")):
                raise ValueError("cash control differs from the independently measured holdout")
            values[str(lag)] = outcome
        results[name] = dict(cases=values, counts=Counter(r["entry_status"] for r in selected))
        print(name, [(lag, value.get("trades"), value.get("cash_pnl_usd"), value.get("reason"))
                     for lag,value in values.items()], flush=True)
    write_json(ROOT/f"{STEM}.json", dict(start=START, end=ENTRY_STOP, starting_capital=CAPITAL,
        source_hashes={p.name:file_hash(p) for p in (source, expected_path, SPOT/"manifest.json")},
        candidate_count=len(rows), spot_coverage=sum(v is not None for v in contexts.values()),
        execution_evidence=evidence, control_parity=True, results=results, live_activation_supported=False,
        limitations=["Four-hour disjoint crypto decision window, stock route not included.",
                     "The complete 12-hour outcome horizon uses recorded exchange trade and book archives.",
                     "These rules were fixed using earlier windows. Chronological position limits are reapplied.",
                     "Cash from these closed trades does not extrapolate to monthly return."]))


if __name__ == "__main__":
    main()
