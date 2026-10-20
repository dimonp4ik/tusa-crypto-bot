"""Direct X-Perp transfer of frozen literal sweep/impulse components."""
from __future__ import annotations

import collections
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier

import crypto_5m_impulse_pullback_transfer_oct5_audit as impulse
import crypto_5m_impulse_regime_oct5_audit as context
import crypto_5m_liquidation_sweep_actual_oct5_audit as sweep
import crypto_5m_liquidation_sweep_model_oct6_audit as sweep_model
import crypto_5m_liquidation_sweep_model_transfer_oct6_audit as sweep_transfer
import crypto_5m_liquidation_sweep_regime_oct5_audit as sweep_regime
import crypto_5m_liquidation_sweep_transfer_oct6_audit as sweep_filter
import crypto_5m_squeeze_absolute_ladder_oct5_audit as ladder
import crypto_5m_tree_filter_audit as historical_source
import crypto_actual_move_portfolio_oct6_audit as portfolio
from crypto_runtime_shadow_portfolio_oct3_audit import money, runtime_gate


ROOT = Path("reports/audit_2026_09_08")
MANIFEST = ROOT / "xperp_5m_oct20_actual_move" / "manifest.json"
OUT = ROOT / "crypto_actual_move_xperp_oct20.json"
ROWS_OUT = ROOT / "crypto_actual_move_xperp_oct20.pkl"
SYMBOLS = ("BTCUSDT", "ETHUSDT", "XRPUSDT", "XLMUSDT",
           "LINKUSDT", "SOLUSDT", "AVAXUSDT", "AAVEUSDT")
IMPULSE_SPEC = {
    "window": 12, "pullback": 3, "direction": -1,
    "max_impulse_pct": .75, "max_body_atr": .2,
    "min_retrace": .65, "target_pct": .5, "stop_pct": 1.0,
}
REQUIRED_SIGNED_MOVE_3_TRADES_DAY_PCT = .973137


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_xperp() -> tuple[dict[str, dict], dict[str, str], dict]:
    raw = MANIFEST.read_bytes()
    manifest = json.loads(raw)
    sources = {str(MANIFEST): hashlib.sha256(raw).hexdigest()}
    candles = {}
    for symbol in SYMBOLS:
        item = manifest["symbols"][symbol]
        if item.get("status") != "OK":
            raise ValueError(f"incomplete X-Perp source: {symbol}")
        path = Path(item["path"])
        payload = path.read_bytes()
        found = hashlib.sha256(payload).hexdigest()
        if found != item["sha256"]:
            raise ValueError(f"X-Perp hash mismatch: {symbol}")
        candles[symbol] = pickle.loads(payload)
        sources[str(path)] = found
    reference = np.asarray(candles[SYMBOLS[0]]["time"], dtype=np.int64)
    for symbol in SYMBOLS[1:]:
        current = np.asarray(candles[symbol]["time"], dtype=np.int64)
        if not np.array_equal(reference, current):
            raise ValueError(f"unsynchronized X-Perp candles: {symbol}")
    if len(reference) < 2 or not np.all(np.diff(reference) == 300):
        raise ValueError("X-Perp time grid contains gaps")
    return candles, sources, manifest


def frozen_sweep(candles: dict[str, dict]) -> tuple[pd.DataFrame, dict]:
    historical, _ = historical_source.load()
    raw = pd.concat([
        sweep.features(symbol, candle, sweep_regime.ENTRY_SPEC[0])
        for symbol, candle in historical.items()
    ], ignore_index=True)
    entries = sweep.apply_spec(raw, sweep_regime.ENTRY_SPEC)
    entries = context.add_context(entries, historical)
    entries = entries[sweep_regime.mask(
        entries, sweep_filter.FROZEN_CONDITIONS)].copy()
    labeled = ladder.add_outcomes(entries, historical, **sweep_model.EXIT_POLICY)
    fit = labeled[labeled.year.isin((2020, 2021))]
    columns = sweep_model.schema(fit)
    config = sweep_transfer.MODEL
    estimator = ExtraTreesClassifier(
        n_estimators=250, max_depth=config["max_depth"],
        min_samples_leaf=config["min_samples_leaf"],
        max_features=config["max_features"], class_weight="balanced",
        n_jobs=-1, random_state=20261006)
    estimator.fit(sweep_model.matrix(fit, columns),
                  (fit.signed_return > 0).astype(int).to_numpy())
    labeled["quality"] = estimator.predict_proba(
        sweep_model.matrix(labeled, columns))[:, 1]
    threshold = float(labeled[labeled.year == 2022].quality.quantile(
        config["quantile"]))

    venue_raw = pd.concat([
        sweep.features(symbol, candle, sweep_regime.ENTRY_SPEC[0])
        for symbol, candle in candles.items()
    ], ignore_index=True)
    venue = sweep.apply_spec(venue_raw, sweep_regime.ENTRY_SPEC)
    venue = context.add_context(venue, candles)
    venue = venue[sweep_regime.mask(
        venue, sweep_filter.FROZEN_CONDITIONS)].copy()
    venue = ladder.add_outcomes(venue, candles, **sweep_model.EXIT_POLICY)
    venue["quality"] = estimator.predict_proba(
        sweep_model.matrix(venue, columns))[:, 1]
    return venue[venue.quality.ge(threshold)].copy(), {
        "model": dict(config), "threshold": threshold,
        "entry_spec": list(sweep_regime.ENTRY_SPEC),
        "conditions": [list(value) for value in sweep_filter.FROZEN_CONDITIONS],
        "exit_policy": dict(sweep_model.EXIT_POLICY),
    }


def frozen_impulse(candles: dict[str, dict]) -> pd.DataFrame:
    pieces = [impulse.events(
        symbol, candle, IMPULSE_SPEC["window"], IMPULSE_SPEC["pullback"])
        for symbol, candle in candles.items()]
    frame = pd.concat(pieces, ignore_index=True)
    frame = frame[
        frame.direction.eq(IMPULSE_SPEC["direction"])
        & frame.impulse_pct.le(IMPULSE_SPEC["max_impulse_pct"])
        & frame.body_atr.le(IMPULSE_SPEC["max_body_atr"])
        & frame.retrace.ge(IMPULSE_SPEC["min_retrace"])
    ].copy()
    return impulse.add_outcomes(
        frame, candles, IMPULSE_SPEC["target_pct"],
        IMPULSE_SPEC["stop_pct"])


def variants(candles: dict[str, dict]) -> tuple[dict[str, list[dict]], dict]:
    sweep_rows, sweep_policy = frozen_sweep(candles)
    impulse_rows = frozen_impulse(candles)
    enriched = context.add_context(impulse_rows, candles)
    broad = enriched[context.atom_mask(
        enriched, portfolio.IMPULSE_BROAD)].copy()
    stable = enriched[context.atom_mask(
        enriched, portfolio.IMPULSE_STABLE)].copy()

    def normalize(parts: tuple[tuple[str, pd.DataFrame], ...]) -> list[dict]:
        output = []
        for priority, (name, frame) in enumerate(parts):
            values = portfolio.normalize(frame, name, priority)
            for row in values:
                row["literal_final_target_hit"] = int(row["outcome_code"]) == 2
            output.extend(values)
        return output

    found = {
        "sweep_only": normalize((("sweep_actual_ladder", sweep_rows),)),
        "broad_first_plus_sweep": normalize((
            ("impulse_broad_05", broad),
            ("sweep_actual_ladder", sweep_rows))),
        "stable_first_plus_sweep": normalize((
            ("impulse_stable_05", stable),
            ("sweep_actual_ladder", sweep_rows))),
    }
    detail = {
        "sweep_policy": sweep_policy,
        "impulse_policy": dict(IMPULSE_SPEC),
        "raw_rows": {"sweep": len(sweep_rows), "impulse": len(impulse_rows),
                     "impulse_broad": len(broad), "impulse_stable": len(stable)},
    }
    return found, detail


def projected_month(mean_signed: float, trades_per_day: float,
                    margin: float) -> float:
    periods = trades_per_day * 30
    factor = 1 + margin * 10 * mean_signed
    return (100 * (factor ** periods - 1)) if factor > 0 else -100.0


def evaluate(rows: list[dict], start: int, end: int,
             margin: float) -> tuple[dict, list[dict]]:
    selected = [row for row in rows
                if start <= int(row["entry_time"]) <= end]
    accepted = runtime_gate(selected)
    value = money(accepted, margin)
    mean_signed = float(np.mean([
        row["signed_price_return"] for row in accepted])) if accepted else 0.0
    literal_hits = sum(bool(row["literal_final_target_hit"])
                       for row in accepted)
    hourly = collections.Counter(
        int(row["entry_time"]) // 3600 for row in accepted)
    value.update({
        "mean_signed_price_return_pct": round(100 * mean_signed, 6),
        "literal_final_target_hit_rate_pct": round(
            100 * literal_hits / len(accepted), 4) if accepted else 0.0,
        "projected_30d_return_pct": round(projected_month(
            mean_signed, value["trades_per_calendar_day"], margin), 4),
        "hours_with_multiple_entries": sum(count > 1 for count in hourly.values()),
        "maximum_entries_same_clock_hour": max(hourly.values(), default=0),
        "required_signed_move_at_3_trades_day_pct":
            REQUIRED_SIGNED_MOVE_3_TRADES_DAY_PCT,
        "economic_30pct_month_gate": bool(projected_month(
            mean_signed, value["trades_per_calendar_day"], margin) >= 30),
    })
    return value, accepted


def main() -> None:
    candles, sources, manifest = load_xperp()
    routes, policy = variants(candles)
    history_manifest = json.loads((
        ROOT / "crypto_5m_history" / "manifest.json").read_text(encoding="utf-8"))
    cutoff = int(history_manifest["symbols"]["BTCUSDT"]["end"])
    first = min(int(values["time"][0]) for values in candles.values())
    last = max(int(values["time"][-1]) for values in candles.values())
    periods = {"all_xperp": (first, last),
               "overlap_through_spot_cutoff": (first, cutoff),
               "fresh_after_spot_cutoff": (cutoff + 1, last)}
    results, row_output = {}, []
    for name, rows in routes.items():
        results[name] = {}
        for period, bounds in periods.items():
            results[name][period] = {}
            for margin in (.03, .05):
                value, accepted = evaluate(rows, *bounds, margin)
                results[name][period][f"margin_{round(100*margin)}pct"] = value
                if name == "broad_first_plus_sweep" and period == "all_xperp" and margin == .03:
                    row_output = accepted
    pd.DataFrame(row_output).to_pickle(ROWS_OUT)
    fresh = results["broad_first_plus_sweep"]["fresh_after_spot_cutoff"]["margin_3pct"]
    pass_gate = bool(
        fresh["n"] >= 100 and fresh["negative_months"] == 0
        and fresh["dd_pct"] <= 10
        and fresh["trades_per_calendar_day"] >= 3
        and fresh["economic_30pct_month_gate"])
    report = {
        "status": ("DIRECT_XPERP_ECONOMIC_PASS_AWAIT_MORE_FORWARD"
                   if pass_gate else "DIRECT_XPERP_REJECTED_OBJECTIVE"),
        "ready_for_live_orders": False,
        "protocol": {
            "transfer": "all entry, model, threshold and exit parameters frozen before X-Perp",
            "venue": "OKX public USDT-margined X-Perp",
            "entry": "next five-minute market open", "orders": "market only",
            "collision": "stop first",
            "spot_cutoff": cutoff,
            "money": "client 3% or 5% current-equity margin; fixed 10x",
            "economic_gate": "+30% projected over 30 calendar days",
        },
        "manifest_created_at": manifest["created_at"],
        "coverage": {"start": first, "end": last,
                     "bars_each": len(candles[SYMBOLS[0]]["time"])},
        "frozen_policy": policy,
        "variant_candidate_rows": {name: len(rows) for name, rows in routes.items()},
        "results": results, "economic_pass": pass_gate,
        "sources": {
            **sources,
            str(portfolio.SWEEP_SOURCE): digest(portfolio.SWEEP_SOURCE),
            str(portfolio.IMPULSE_SOURCE): digest(portfolio.IMPULSE_SOURCE),
            str(Path(__file__).resolve()): digest(Path(__file__).resolve()),
        },
        "limitations": [
            "The direct X-Perp window is only about 69 days.",
            "October after the spot cutoff is a short forward slice.",
            "The sweep model is refit deterministically on its original 2020-2021 sample.",
            "Gross returns exclude fees at user request.",
            "Projected monthly return is a diagnostic, not a forecast or guarantee.",
        ],
    }
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({
        "status": report["status"], "coverage": report["coverage"],
        "candidate_rows": report["variant_candidate_rows"],
        "fresh": {name: values["fresh_after_spot_cutoff"]["margin_3pct"]
                  for name, values in results.items()},
        "economic_pass": pass_gate,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
