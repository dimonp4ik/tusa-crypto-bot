"""Replay previously excluded crypto stop widths against archived OKX books."""

from collections import Counter
import pickle
from unittest.mock import patch

import okx_cross_bot_overlap_nov21 as joint
from okx_joint_exact_exit_nov21 import ENTRY_END, choose_rows
from okx_tp1_cash_sensitivity_nov22 import detailed
from okx_unbankable_tp1_nov21 import read_json
from okx_xperp_cash_tick_nov19 import CAPITAL, CLOSE_LAGS, FRACTION, LEVERAGE, simulate
from okx_xperp_multiday_capture_nov20 import book_files, file_hash
from okx_xperp_multiday_replay_nov20 import entries, exit_books, key_for, resolve_all
from src.crypto_profit_first_material_shadow_router import MODEL, shadow_setups
from src.okx_trader import round_to_tick
from stock_liquid_quote_routes_nov21 import ROOT, limit_bursts, write_json


LOW = float(MODEL["risk_pct_min"])
HIGH = float(MODEL["risk_pct_max"])
EXTENSIONS = {"below_1pct": (.008, LOW), "above_2pct": (HIGH, .024)}
RISK_REJECTS = {"model_risk_outside_1_to_2pct", "execution_risk_outside_1_to_2pct"}


def candidate_plan(captured, frozen):
    candidates = {band: [] for band in EXTENSIONS}
    filters = Counter()

    def replay(candles, btc, *, entry_time, market_price, model_price, symbol):
        arguments = dict(entry_time=entry_time, market_price=market_price,
                         model_price=model_price, symbol=symbol)
        audit = {}
        control = shadow_setups(candles, btc, **arguments, diagnostics=audit)
        if control or RISK_REJECTS.isdisjoint(audit["rejected"]):
            return control
        for name, (lower, upper) in EXTENSIONS.items():
            with patch.dict(MODEL, risk_pct_min=lower, risk_pct_max=upper):
                alternates = shadow_setups(candles, btc, **arguments, diagnostics=audit)
            for reason, count in audit["rejected"].items():
                filters[f"{name}:{reason}"] += count
            for row in alternates:
                model_risk, execution_risk = row["model_risk_pct"], row["risk_pct"]
                if name == "below_1pct" and max(model_risk, execution_risk) >= LOW:
                    continue
                if name == "above_2pct" and min(model_risk, execution_risk) <= HIGH:
                    continue
                entry, stop = row["entry"], row["sl"]
                candidates[name].append(dict(epoch=entry_time, bot="crypto", symbol=symbol,
                    direction=row["direction"], entry=entry, stop=stop, band=name,
                    target=entry + row["runner_target_r"] * (entry - stop),
                    model_quality=row["model_quality"], model_risk_pct=model_risk,
                    execution_risk_pct=execution_risk, paper_reference_ready=False))
        return control

    with patch.object(joint, "shadow_setups", replay):
        base, checks = joint.reconstruct(captured)
    if base != frozen["rows"] or checks != frozen["tape_mismatches"]:
        raise ValueError("reconstructed 1–2% control changed")
    for group in candidates.values():
        group.sort(key=lambda r: (r["epoch"], r["symbol"]))
    return candidates, filters


def attach_entry_quotes(group, frozen, available, ticks):
    plan = dict(frozen, rows=group)
    required = {key_for(r["symbol"], r["epoch"] * 1000 + joint.ENTRY_OFFSET_MS) for r in group}
    missing = {key for key in required if "path" not in available.get(key, {})}
    if missing:
        request = [r for r in group if key_for(r["symbol"], r["epoch"] * 1000) in missing]
        available.update(book_files(dict(plan, rows=request)))
    entry_archives = {}
    for key in required:
        raw_day, symbol = key.split(":", 1)
        entry_archives[key] = dict(available[key], day=int(raw_day), symbol=symbol)
    rows, books, checks = entries(plan, {"books": entry_archives})
    for row in rows:
        if row["entry_status"] != "eligible":
            continue
        tp1 = round_to_tick(row["entry"] + .5 * (row["entry"] - row["stop"]), ticks[row["symbol"]])
        sign = 1 if row["direction"] == "LONG" else -1
        if not sign * row["stop"] < sign * row["tries"][-1]["worst"] < sign * tp1:
            row["entry_status"] = "outside_tp1_bracket"
    return rows, books, checks


def merge_books(books, snapshots):
    for symbol, samples in snapshots.items():
        if any(books[symbol][t] != samples[t] for t in books[symbol].keys() & samples.keys()):
            raise ValueError("archive samples at the same timestamp disagree")
        books[symbol].update(samples)


def replay_cash(crypto, stocks, specs, ticks, books, lag, cap):
    rows = list(joint.crypto_cadence(crypto)) + list(limit_bursts(stocks, max_same_side=2))
    if any(r["entry_status"] == "unknown_quote" for r in rows):
        return dict(status="UNSCORABLE_UNKNOWN_ENTRY", missing=[(r["symbol"], r["entry_ms"]) for r in rows
                                                               if r["entry_status"] == "unknown_quote"])
    result = simulate(rows, specs, ticks, books, clock=60, lag=lag, setting="requested5", max_positions=cap)
    added = {(r["symbol"], r["entry_ms"]) for r in rows if "band" in r}
    return dict(detailed(result), added_positions=[(t["symbol"], t["entry_ms"], t["code"],
        t["bank_pnl"] + t["runner_pnl"]) for t in result.get("closed", []) if (t["symbol"], t["entry_ms"]) in added])


def main():
    capture_path = joint.DATA / "crypto_capture.json"
    plan_path = joint.DATA / "crypto_plan.json"
    book_path = joint.DATA / "crypto_books.json"
    reference_path = ROOT / "okx_joint_exact_exit_nov21.json"
    captured, frozen, existing, reference = map(read_json, (capture_path, plan_path, book_path, reference_path))
    groups, filter_counts = candidate_plan(captured, frozen)
    additions = [r for group in groups.values() for r in group if r["epoch"] < ENTRY_END]
    print("Baseline reconstructed", len(frozen["rows"]), "range extensions",
          {band: (len(rows), sum(r["epoch"] < ENTRY_END for r in rows)) for band, rows in groups.items()}, flush=True)
    crypto, stocks, specs, ticks, books, hashes, _, _, _, _ = joint.joint_inputs()
    selected = choose_rows(crypto, stocks)
    crypto = [r for r in selected if r["bot"] == "crypto"]
    stocks = [r for r in selected if r["bot"] == "stocks"]
    stock_report = ROOT / "stock_liquid_quote_routes_nov21.json"
    saved = pickle.loads((joint.DATA / "joint_exact_exit_samples.pkl").read_bytes())
    if saved["entry_end"] != ENTRY_END or saved["sources"] != dict(hashes, **{str(stock_report): file_hash(stock_report)}):
        raise ValueError("archived control exit observations changed")
    snapshots, previous_exits, _ = saved["result"]
    merge_books(books, snapshots)
    control = replay_cash(crypto, stocks, specs, ticks, books, 200, 6)
    expected = reference["scenarios"]["both_base_cadence|cap6|requested5|200ms"]["book"]
    if any(control[k] != expected[k] for k in ("status", "trades", "return_pct", "drawdown_pct", "skipped")):
        raise ValueError("published executable 1–2% cash control changed")
    available = existing["books"] | previous_exits
    alt, alt_books, entry_checks = attach_entry_quotes(additions, frozen, available, ticks)
    merge_books(books, alt_books)
    tape_checks = resolve_all(alt, {"files": captured["trades"]}, {"symbols": captured["references"]}, ticks)
    extra, exit_checks = exit_books(alt, {"books": available}, specs, books)
    cohorts = dict(control_1_to_2pct=crypto)
    for name in EXTENSIONS:
        cohorts[f"with_{name}"] = crypto + [r for r in alt if r["band"] == name]
    cohorts["with_both_extensions"] = crypto + alt
    cases = {name: {f"{pool}|{lag}ms": replay_cash(rows, other, specs, ticks, books, lag, cap)
                    for pool, other, cap in (("crypto_only", [], 12), ("both_cap6", stocks, 6),
                                             ("both_cap12", stocks, 12)) for lag in CLOSE_LAGS}
             for name, rows in cohorts.items()}
    paths = [dict(r, entry_status=r["entry_status"], path=r.get("path", {}).get(60)) for r in alt]
    report = dict(start=joint.START, stop=ENTRY_END, capital_usd=CAPITAL, margin_fraction=FRACTION,
                  leverage=LEVERAGE, ranges=EXTENSIONS, control_parity=True, runner_target_r=2.,
                  source_hashes={p.name: file_hash(p) for p in (capture_path, plan_path, book_path, reference_path)},
                  candidates=groups, filter_counts=filter_counts, executable_window=paths, cases=cases,
                  entry_book_checks=entry_checks, trade_checks=tape_checks, exit_book_checks=exit_checks,
                  exit_archive_hashes={k: v["sha256"] for k, v in extra.items()}, live_activation_supported=False,
                  limitations=["The frozen classifier was calibrated with model stop widths of 1–2%.",
                    "Extensions add signals only at slots with no baseline model-qualified candidate.",
                    "Later proposals without a full 16-hour common archive window are listed but have no cash result.",
                    "Stock continuation in the portfolio comparison is hypothetical and PAPER gated in production.",
                    "Cash ledger assumes displayed public depth, 200ms or 1000ms exits and 5% requested at TP1."])
    write_json(ROOT / "okx_crypto_risk_range_nov22.json", report)
    for name, values in cases.items():
        print(name, {k: (round(v["cash_pnl_usd"], 3), v["trades"], len(v["added_positions"]))
                    if "cash_pnl_usd" in v else (v["status"], v.get("reason")) for k, v in values.items()}, flush=True)


if __name__ == "__main__":
    main()
