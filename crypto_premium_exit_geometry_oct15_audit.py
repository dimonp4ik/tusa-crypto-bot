"""Exit-geometry refinement for the frozen premium-dislocation router."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numba as nb
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

import crypto_5m_regime_router_literal_oct6_audit as price
import crypto_futures_funding_regime_oct6_audit as funding
import crypto_premium_dislocation_oct15_audit as premium


ROOT = Path("reports/audit_2026_09_08")
PARENT = ROOT / "crypto_premium_dislocation_oct15.json"
OUT = ROOT / "crypto_premium_exit_geometry_oct15.json"
TARGETS = (.5, .75, 1.0)
STOPS = (.5, .75, 1.0)
MAX_BARS = (24, 72)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@nb.njit(cache=True)
def resolve(open_, high, low, close, times, indices, directions,
            target_pct, stop_pct, max_bars):
    count = len(indices)
    signed = np.empty(count, dtype=np.float64)
    exits = np.empty(count, dtype=np.int64)
    target_hits = np.zeros(count, dtype=np.int8)
    reasons = np.empty(count, dtype=np.int8)
    target_move, stop_move = target_pct/100, stop_pct/100
    for n in range(count):
        first, direction = indices[n], directions[n]
        entry = open_[first]
        target = entry*(1+direction*target_move)
        stop = entry*(1-direction*stop_move)
        result, reason, last, resolved = 0.0, 2, first, False
        for j in range(first, min(len(times), first+max_bars)):
            if j > first and times[j]-times[j-1] != 300:
                break
            last = j
            stop_hit = low[j] <= stop if direction > 0 else high[j] >= stop
            target_hit = high[j] >= target if direction > 0 else low[j] <= target
            if stop_hit:
                fill = min(open_[j], stop) if direction > 0 else max(open_[j], stop)
                result, reason, resolved = direction*(fill-entry)/entry, 0, True
                break
            if target_hit:
                result, reason, resolved = target_move, 1, True
                target_hits[n] = 1
                break
        if not resolved:
            result = direction*(close[last]-entry)/entry
            result = max(-stop_move, min(target_move, result))
        signed[n], exits[n], reasons[n] = result, times[last]+300, reason
    return signed, exits, target_hits, reasons


def add_outcomes(frame, candles, target, stop, max_bars):
    pieces = []
    for symbol, part in frame.groupby("symbol", sort=False):
        candle = candles[str(symbol)]
        arrays = [np.asarray(candle[key], dtype=float)
                  for key in ("open", "high", "low", "close")]
        values, exits, hits, reasons = resolve(
            *arrays, np.asarray(candle["time"], dtype=np.int64),
            part.entry_index.to_numpy(np.int64),
            part.direction.to_numpy(np.int8), target, stop, max_bars)
        found = part.copy()
        found["signed_return"], found["exit_time"] = values, exits
        found["target_hit"] = hits.astype(bool)
        found["exit_reason"] = reasons
        found["target_pct"], found["stop_pct"] = target, stop
        found["max_bars"] = max_bars
        pieces.append(found)
    return __import__("pandas").concat(pieces, ignore_index=True)


def acceptable(value: dict, validation: bool) -> bool:
    return bool(value["n"] >= (350 if validation else 500)
                and value["return_pct"] > 0
                and value["dd_pct"] <= 8
                and value["trades_per_calendar_day"] >= 2.5
                and value["negative_months"] <= 5)


def main() -> None:
    parent_raw = PARENT.read_bytes()
    parent = json.loads(parent_raw)
    frozen = parent["frozen_before_test"]
    if not frozen or frozen["model_kind"] != "premium_plus_price":
        raise ValueError("parent premium route is not frozen")
    candles, _, candle_sources = funding.load()
    premium_data, premium_sources = premium.load_premium()
    raw = price.build_features(candles)
    enriched = premium.attach(raw, premium_data)
    proposals = premium.proposals(enriched)
    baseline = price.add_outcomes(proposals, candles)
    fit = baseline.year.isin((2020, 2021)).to_numpy()
    schema = price.columns(baseline.loc[fit])
    _, matrix = premium.matrices(baseline, schema)
    config = frozen["model"]
    model = HistGradientBoostingClassifier(
        max_iter=180, learning_rate=.06,
        max_leaf_nodes=config["max_leaf_nodes"],
        min_samples_leaf=config["min_samples_leaf"],
        l2_regularization=config["l2"], random_state=20261015)
    model.fit(matrix[fit], baseline.target_hit.astype(int).to_numpy()[fit])
    proposals = proposals.copy()
    proposals["quality"] = model.predict_proba(
        premium.matrices(proposals, schema)[1])[:, 1]
    threshold = float(frozen["threshold"])

    frontier, eligible = [], []
    for target in TARGETS:
        for stop in STOPS:
            for max_bars in MAX_BARS:
                frame = add_outcomes(proposals, candles, target, stop, max_bars)
                selection = price.evaluate(frame, (2022,), threshold)
                item = {"config": {"target_pct": target, "stop_pct": stop,
                                   "max_bars": max_bars},
                        "selection_2022": selection}
                frontier.append(item)
                if not acceptable(selection, False):
                    continue
                item["validation_2023"] = price.evaluate(
                    frame, (2023,), threshold)
                item["validation_2024"] = price.evaluate(
                    frame, (2024,), threshold)
                if not (acceptable(item["validation_2023"], True)
                        and acceptable(item["validation_2024"], True)):
                    continue
                item["_frame"] = frame
                eligible.append(item)
    eligible.sort(key=lambda item: (
        -sum(item[name]["negative_months"] for name in (
            "selection_2022", "validation_2023", "validation_2024")),
        min(item[name]["return_pct"] for name in (
            "selection_2022", "validation_2023", "validation_2024")),
        min(item[name]["literal_target_hit_rate"] for name in (
            "selection_2022", "validation_2023", "validation_2024")),
        -max(item[name]["dd_pct"] for name in (
            "selection_2022", "validation_2023", "validation_2024"))),
        reverse=True)
    chosen = eligible[0] if eligible else None
    results = checks = None
    if chosen:
        results = {str(year): price.evaluate(
            chosen["_frame"], (year,), threshold)
            for year in (2022, 2023, 2024, 2025, 2026)}
        checks = {year: price.objective_ok(value)
                  for year, value in results.items()}
    for item in eligible:
        item.pop("_frame", None)
    public_chosen = dict(chosen) if chosen else None
    frontier.sort(key=lambda item: (
        item["selection_2022"]["negative_months"] == 0,
        item["selection_2022"]["return_pct"],
        item["selection_2022"]["literal_target_hit_rate"],
        -item["selection_2022"]["dd_pct"]), reverse=True)
    status = ("OBJECTIVE_PASS" if checks and all(checks.values())
              else "REJECTED_CONFIRMATION" if chosen else "REJECTED_PRETEST")
    report = {
        "status": status, "ready_for_live_orders": False,
        "decision": "No runtime change unless all chronological objectives pass.",
        "protocol": {
            "parent": str(PARENT),
            "frozen_signal_model": frozen,
            "geometry_selection": "2022", "validation": [2023, 2024],
            "confirmation_only": [2025, 2026],
            "targets_pct": TARGETS, "stops_pct": STOPS,
            "max_bars": MAX_BARS, "entry": "next 5m market open",
            "ambiguity": "stop first", "position": "3% margin, fixed 10x",
        },
        "grid": len(frontier), "qualified_before_confirmation": len(eligible),
        "frozen_geometry": public_chosen,
        "results": results, "objective_checks": checks,
        "selection_frontier": frontier,
        "sources": {str(PARENT): hashlib.sha256(parent_raw).hexdigest(),
                    **candle_sources, **premium_sources,
                    str(Path(__file__)): digest(Path(__file__))},
        "limitations": [
            "The signal route was already inspected through 2026 before this geometry refinement.",
            "Confirmation years are robustness evidence, not pristine holdouts.",
            "Binance futures remain a proxy for OKX X-Perp execution.",
        ],
    }
    OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({key: report[key] for key in (
        "status", "grid", "qualified_before_confirmation",
        "frozen_geometry", "results", "objective_checks")}, indent=2))


if __name__ == "__main__":
    main()
