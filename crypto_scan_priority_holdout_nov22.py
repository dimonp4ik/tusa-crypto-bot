"""Check fixed publication priorities on the newly available November 20 book archive."""

from collections import Counter
from datetime import datetime, timezone
from unittest.mock import patch

import okx_cross_bot_overlap_nov21 as joint
from crypto_scan_priority_nov22 import POLICIES, attribute, book_cadence, reconstruct_rank
from okx_crypto_risk_range_nov22 import attach_entry_quotes
from okx_tp1_cash_sensitivity_nov22 import detailed
from okx_unbankable_tp1_nov21 import read_json
from okx_xperp_cash_tick_nov19 import simulate
from okx_xperp_multiday_capture_nov20 import HOLD, file_hash
from okx_xperp_multiday_replay_nov20 import ROOT, exit_books, resolve_all
from stock_liquid_quote_routes_nov21 import write_json


STEM = "crypto_scan_priority_holdout_nov22"
DATA = ROOT / STEM
START = int(datetime(2026, 11, 20, tzinfo=timezone.utc).timestamp())
ENTRY_STOP = START + 4*3600


def load_cohort(capture_path):
    with patch.multiple(joint, START=START, ENTRY_END=ENTRY_STOP, DATA=DATA):
        if not capture_path.exists():
            joint.capture_crypto()
        captured = read_json(capture_path)
        if not all(item["complete"] for item in captured["trades"].values()):
            raise ValueError("the holdout's 12-hour exit horizon has missing tape")
        rows, annotations, tape_checks = reconstruct_rank(captured)
    return captured, [dict(r, **annotations[r["symbol"], r["epoch"]]) for r in rows], tape_checks


def main():
    capture_path = DATA / "crypto_capture.json"
    captured, rows, tape = load_cohort(capture_path)
    print("Independent archived proposals", len(rows), "across", len({r['epoch'] for r in rows}), "scans", flush=True)
    specs = captured["specs"]
    ticks = {s: float(v["tickSz"]) for s, v in specs.items()}
    plan = dict(rows=rows, specs=specs, start=START, coverage_end=ENTRY_STOP+HOLD["crypto"],
                parameters=dict(crypto_offset_ms=joint.ENTRY_OFFSET_MS, retry_offsets_ms=[0, 200, 800]))
    available = {}
    proposals, books, entry_checks = attach_entry_quotes(rows, plan, available, ticks)
    tape_checks = resolve_all(proposals, {"files": captured["trades"]}, {"symbols": captured["references"]}, ticks)
    additional, exit_checks = exit_books(proposals, {"books": available}, specs, books)
    results = {}
    for policy in POLICIES:
        picked = list(book_cadence(proposals, policy))
        unknown = [(r["symbol"], r["entry_ms"]) for r in picked if r["entry_status"] == "unknown_quote"]
        each = {}
        for lag in (200, 1000):
            result = (dict(status="UNSCORABLE_UNKNOWN_ENTRY", missing=unknown) if unknown else
                      simulate(picked, specs, ticks, books, clock=60, lag=lag, setting="requested5"))
            each[lag] = dict(detailed(result), phase=attribute(result.get("closed", []), picked),
                             admitted=[(r["symbol"], r["entry_ms"]) for r in picked if r["entry_status"]=="eligible"])
        results[policy] = each
    write_json(ROOT/f"{STEM}.json", dict(start=START, stop=ENTRY_STOP, source_sha256=file_hash(capture_path),
        fixed_variants=POLICIES, proposals=len(rows), path_checks=tape, trade_checks=tape_checks,
        entry_books=entry_checks, exit_books=exit_checks, eligible=Counter(r["entry_status"] for r in proposals),
        exit_extra_sha256={k: v["sha256"] for k, v in additional.items()}, results=results,
        live_activation_supported=False, limitations=[
            "Only a four-hour decision window. All 12-hour outcome horizons fit inside both archived sources.",
            "New entries strictly follow the November 17–19 decision sample used for the prior result.",
            "Only crypto, with original brackets and public trade/book replay. No new live strategy selected.",
            "Four experimental rank rules were fixed before inspecting this holdout's candidate paths.",
        ]))
    for name, lags in results.items():
        print(name, {lag: (r["status"], r.get("trades"), r.get("cash_pnl_usd"), r.get("reason"))
                      for lag, r in lags.items()}, flush=True)


if __name__ == "__main__":
    main()
