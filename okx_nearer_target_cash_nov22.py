"""Reprice nearer crypto targets against the same timestamped entry and exit books."""

from collections import Counter

from okx_entry_screen_holdout_nov21 import ENTRY_END as SPLIT
from okx_tp1_cash_sensitivity_nov22 import detailed, verify_control
from okx_unbankable_tp1_nov21 import old_inputs, recent_inputs
from okx_xperp_cash_tick_nov19 import CAPITAL, CLOSE_LAGS, FRACTION, LEVERAGE, simulate
from okx_xperp_multiday_replay_nov20 import ROOT, exit_books, file_hash, resolve_all
from src.okx_trader import round_to_tick
from stock_liquid_quote_routes_nov21 import write_json


TARGETS = (0.75, 1.0, 1.25, 1.5)


def build_paths(rows, ticks, trades, reference):
    by_target = {}
    for target_r in TARGETS:
        candidates = []
        for r in rows:
            if r["bot"] != "crypto" or r["entry_status"] != "eligible":
                continue
            moved = round_to_tick(r["entry"] + target_r * (r["entry"] - r["stop"]), ticks[r["symbol"]])
            candidates.append(dict(r, target=moved))
        by_target[target_r] = candidates
    flat = [candidate for candidates in by_target.values() for candidate in candidates]
    control_candidates = [dict(r) for r in rows if r["bot"] == "crypto" and r["entry_status"] == "eligible"]
    tape_check = resolve_all(flat + control_candidates, {"files": trades}, {"symbols": reference}, ticks)
    original = [r for r in rows if r["bot"] == "crypto" and r["entry_status"] == "eligible"]
    if any(r["path"] != verified["path"] for r, verified in zip(original, control_candidates, strict=True)):
        raise ValueError("trade tape replay no longer reproduces the original paths")
    return by_target, flat, tape_check


def score(rows, specs, ticks, books, variants, cap):
    configured = {"control_2r": rows}
    for target_r, candidates in variants.items():
        variants_by_key = {(r["symbol"], r["entry_ms"]): r for r in candidates}
        configured[f"{target_r:g}r"] = [variants_by_key.get((r["symbol"], r["entry_ms"]), r)
                                        if r["bot"] == "crypto" else r for r in rows]
    return {str(lag): {name: detailed(simulate(candidates, specs, ticks, books, clock=60, lag=lag,
                                               setting="requested5", max_positions=cap))
                       for name, candidates in configured.items()} for lag in CLOSE_LAGS}


def evaluate(material, control_path, *, joint):
    rows, specs, ticks, books, archive, trades, reference = material
    cap = 6 if joint else 12
    verify_control(rows, specs, ticks, books, cap, control_path, joint=joint)
    variants, flat, tape_check = build_paths(rows, ticks, trades, reference)
    print("Reconciled trade paths", "joint" if joint else "earlier", len(flat), flush=True)
    downloads, book_checks = exit_books(flat, {"books": archive}, specs, books)
    if joint:
        cohorts = {"nov17_19_joint": rows}
    else:
        cohorts = {"nov05_08": [r for r in rows if r["epoch"] < SPLIT],
                   "nov09_14": [r for r in rows if r["epoch"] >= SPLIT]}
    results = {name: score(cohort, specs, ticks, books, variants, cap) for name, cohort in cohorts.items()}
    evidence = dict(tape=tape_check, books=book_checks,
                    new_archive_sha256={key: item["sha256"] for key, item in downloads.items()},
                    changed_paths={str(r): Counter(row["path"][60].get("code", "unknown") for row in candidates)
                                   for r, candidates in variants.items()})
    return results, evidence


def main():
    earlier = ROOT / "okx_entry_screen_holdout_nov21.json"
    later = ROOT / "okx_joint_exact_exit_nov21.json"
    cases, early_evidence = evaluate(old_inputs(), earlier, joint=False)
    latest, latest_evidence = evaluate(recent_inputs(), later, joint=True)
    cases.update(latest)
    write_json(ROOT / "okx_nearer_target_cash_nov22.json", dict(
        capital_usd=CAPITAL, margin_fraction=FRACTION, leverage=LEVERAGE, targets_r=TARGETS,
        control_sha256={p.name: file_hash(p) for p in (earlier, later)},
        source_checks=dict(earlier=early_evidence, joint=latest_evidence), cases=cases,
        live_activation_supported=False, limitations=[
            "Original TP1 0.5R, client TP1 5% and next-minute break-even are unchanged.",
            "New target closes the entire remainder and may release a shared slot earlier.",
            "No fill if an essential book snapshot is absent; such a cohort case is unscorable.",
            "Investment results cannot be inferred out of sample from these previously studied dates.",
        ]))
    for name, lags in cases.items():
        for lag, values in lags.items():
            print(name, lag, {target: round(result["cash_pnl_usd"], 3) if "cash_pnl_usd" in result
                              else (result["symbol"], result["reason"])
                              for target, result in values.items()}, flush=True)


if __name__ == "__main__":
    main()
