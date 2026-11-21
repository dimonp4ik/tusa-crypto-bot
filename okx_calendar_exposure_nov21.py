"""Measure exposure to scheduled US macro releases on frozen crypto proposals."""

import json
import math
import pickle
from datetime import datetime
from zoneinfo import ZoneInfo

from okx_entry_screen_cash_nov21 import summarize
from okx_xperp_cash_tick_nov19 import simulate
from okx_xperp_multiday_replay_nov20 import FOLDER, ROOT, file_hash


NEW_YORK = ZoneInfo("America/New_York")
CALENDAR = "https://www.bls.gov/schedule/2026/11_sched.htm"
RELEASES = (
    ("US productivity and costs", "2026-11-05 08:30"),
    ("US employment situation", "2026-11-06 08:30"),
    ("US CPI", "2026-11-10 08:30"),
    ("US PPI", "2026-11-13 08:30"),
)
MINUTE_MS = 60_000


def read_case(folder, cache, expected_digest=None):
    plan = json.loads((folder / "plan.json").read_bytes())
    result = pickle.loads((folder / cache).read_bytes())
    if expected_digest is not None:
        digest, books = result
        if digest != expected_digest:
            raise ValueError("stock/crypto source digest mismatch")
        report = json.loads((ROOT / "okx_xperp_multiday_cash_nov20.json").read_bytes())
        rows = report["paths"]
        for row in rows:
            if "path" in row:
                row["path"] = {int(key): path for key, path in row["path"].items()}
        return plan["specs"], rows, books
    inputs = (folder / "plan.json", folder / "capture.json",
              ROOT / "paired_contract_volume_nov19/manifest.json", ROOT / "native_universe_nov19_snapshot.json")
    if result["signature"] != [file_hash(path) for path in inputs]:
        raise ValueError("early quote cache changed")
    rows, books, _ = result["data"]
    return plan["specs"], rows, books


def entry_blackout(proposals, releases, before, after):
    for row in proposals:
        if row["bot"] != "crypto" or row["entry_status"] != "eligible":
            yield row
            continue
        at = row["entry_ms"]
        blocked = any(event - before * MINUTE_MS <= at < event + after * MINUTE_MS for event in releases)
        yield dict(row, entry_status="scheduled_release") if blocked else row


def main():
    first = FOLDER.with_name("nov5_8_entry_screen")
    source = ROOT / "okx_xperp_multiday_cash_nov20.json"
    early_specs, early, early_books = read_case(first, "verified_public_paths.pkl")
    late_specs, late, late_books = read_case(FOLDER, "entry_exit_samples_nov21.pkl", file_hash(source))
    for symbol, values in late_books.items():
        relevant = early_books[symbol]
        if any(relevant[key] != values[key] for key in relevant.keys() & values.keys()):
            raise ValueError("overlapping book evidence differs")
        relevant.update(values)
    specs = early_specs | late_specs
    universe = json.loads((ROOT / "native_universe_nov19_snapshot.json").read_bytes())
    by_id = {r["instId"]: float(r["tickSz"]) for r in universe["instruments"] if r.get("tickSz")}
    ticks = {symbol: by_id[spec["instId"]] for symbol, spec in specs.items()}
    events = {name: int(datetime.fromisoformat(when).replace(tzinfo=NEW_YORK).timestamp() * 1000)
              for name, when in RELEASES}
    rows = early + late
    controls = dict(clock=60, lag=200, setting="requested5")
    control = simulate(rows, specs, ticks, early_books, **controls)
    previous = json.loads((ROOT / "okx_entry_screen_holdout_nov21.json").read_bytes())
    if summarize(control) != previous["continuous_cash_results"]["results"]["baseline"]:
        raise ValueError("calendar control did not match the frozen baseline")
    exposure = {}
    for name, at in events.items():
        held = [t for t in control["closed"] if t["entry_ms"] < at < t["exit_ms"]]
        exposure[name] = dict(release_ms=at, open_at_release=len(held), gross_outcome_cash=math.fsum(
            t["bank_pnl"] + t["runner_pnl"] for t in held), symbols=sorted(t["symbol"] for t in held))
    cases = {}
    for before, after in ((15, 45), (30, 120)):
        selected = entry_blackout(rows, events.values(), before, after)
        cases[f"{before}m_before_{after}m_after"] = summarize(
            simulate(selected, specs, ticks, early_books, **controls))
    report = dict(status="FROZEN_HISTORICAL_ENTRY_COMPARISON", calendar=CALENDAR, sample="2026-11-05 to 2026-11-14",
                  baseline_parity=True, exposure=exposure, baseline=summarize(control), cases=cases,
                  limitations=["Four scheduled US releases. Open exposure shows whole-trade outcomes.",
                               "Historical public-market replay with the same account assumptions.",
                               "The simulation isolates new-entry blackout. It preserves existing positions."])
    target = ROOT / "okx_calendar_exposure_nov21.json"
    target.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(dict(exposure=exposure, cases=cases), indent=2))


if __name__ == "__main__":
    main()
