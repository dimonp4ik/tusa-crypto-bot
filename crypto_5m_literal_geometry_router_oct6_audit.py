"""Select literal +0.5% crypto entries across stop/holding geometries.

The feature model is fit only on 2020-2021.  Geometry and score threshold are
selected on 2022, checked on 2023-2024, then frozen before 2025 and 2026 are
read.  All entries use the next 5-minute market open and explicit candle
high/low target touches.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numba as nb
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

import crypto_5m_regime_router_literal_oct6_audit as base
import crypto_5m_tree_filter_audit as candles_source


ROOT = Path("reports/audit_2026_09_08")
OUT = ROOT / "crypto_5m_literal_geometry_router_oct6.json"
TARGET_PCT = .5
STOP_PCTS = (.5, .75, 1.0)
MAX_BARS_VALUES = (12, 24, 48)
QUANTILES = (.90, .925, .95, .96, .97, .975, .98, .985, .99)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@nb.njit(cache=True)
def resolve(open_, high, low, close, times, indices, directions,
            stop_pct, max_bars):
    count = len(indices)
    signed = np.empty(count, dtype=np.float64)
    exits = np.empty(count, dtype=np.int64)
    hits = np.zeros(count, dtype=np.int8)
    reasons = np.empty(count, dtype=np.int8)
    target_move = TARGET_PCT / 100.0
    stop_move = stop_pct / 100.0
    for n in range(count):
        first, direction = indices[n], directions[n]
        entry = open_[first]
        target = entry * (1.0 + direction * target_move)
        stop = entry * (1.0 - direction * stop_move)
        last = first
        result = 0.0
        reason = 2
        resolved = False
        end = min(len(times), first + max_bars)
        for j in range(first, end):
            if j > first and times[j] - times[j - 1] != 300:
                break
            last = j
            stop_hit = low[j] <= stop if direction > 0 else high[j] >= stop
            target_hit = high[j] >= target if direction > 0 else low[j] <= target
            if stop_hit:
                fill = min(open_[j], stop) if direction > 0 else max(open_[j], stop)
                result = direction * (fill - entry) / entry
                reason = 0
                resolved = True
                break
            if target_hit:
                result = target_move
                hits[n] = 1
                reason = 1
                resolved = True
                break
        if not resolved:
            result = direction * (close[last] - entry) / entry
            result = max(-stop_move, min(target_move, result))
        signed[n], exits[n], reasons[n] = result, times[last] + 300, reason
    return signed, exits, hits, reasons


def add_outcomes(frame: pd.DataFrame, candles: dict, stop: float,
                 max_bars: int) -> pd.DataFrame:
    frame = frame.copy()
    frame["_row_id"] = np.arange(len(frame), dtype=np.int64)
    output = []
    for symbol, part in frame.groupby("symbol", sort=False):
        candle = candles[symbol]
        values, exits, hits, reasons = resolve(
            *[np.asarray(candle[key], dtype=float)
              for key in ("open", "high", "low", "close")],
            np.asarray(candle["time"], dtype=np.int64),
            part.entry_index.to_numpy(np.int64),
            part.direction.to_numpy(np.int8), stop, max_bars)
        found = part.copy()
        found["signed_return"] = values
        found["exit_time"] = exits
        found["target_hit"] = hits.astype(bool)
        found["exit_reason"] = reasons
        output.append(found)
    return pd.concat(output, ignore_index=True).sort_values(
        "_row_id", kind="mergesort").drop(columns="_row_id").reset_index(
            drop=True)


def public(value: dict) -> dict:
    return {key: item for key, item in value.items()
            if key not in ("_frame", "_model")}


def selection_ok(value: dict) -> bool:
    return bool(value["n"] >= 900 and value["return_pct"] > 0
                and value["literal_target_hit_rate"] >= 55
                and value["dd_pct"] <= 8
                and value["trades_per_calendar_day"] >= 3)


def validation_ok(value: dict) -> bool:
    return bool(value["n"] >= 250 and value["return_pct"] > 0
                and value["literal_target_hit_rate"] >= 55
                and value["dd_pct"] <= 8
                and value["trades_per_calendar_day"] >= 1.5)


def main() -> None:
    candles, sources = candles_source.load()
    raw = base.build_features(candles)
    proposals = base.proposals(raw)
    schema = base.columns(proposals[proposals.year.isin((2020, 2021))])
    matrix_all = base.matrix(proposals, schema)
    fit_mask = proposals.year.isin((2020, 2021)).to_numpy()
    selection_mask = proposals.year.eq(2022).to_numpy()
    candidates = []
    diagnostics = []
    for stop in STOP_PCTS:
        for max_bars in MAX_BARS_VALUES:
            frame = add_outcomes(proposals, candles, stop, max_bars)
            model = HistGradientBoostingClassifier(
                max_iter=180, learning_rate=.06, max_leaf_nodes=15,
                min_samples_leaf=300, l2_regularization=4.0,
                random_state=20261006)
            model.fit(matrix_all[fit_mask],
                      frame.loc[fit_mask, "target_hit"].astype(int).to_numpy())
            frame["quality"] = model.predict_proba(matrix_all)[:, 1]
            selection_scores = frame.loc[selection_mask, "quality"]
            for quantile in QUANTILES:
                threshold = float(selection_scores.quantile(quantile))
                selected = base.evaluate(frame, (2022,), threshold)
                item = {"stop_pct": stop, "max_bars": max_bars,
                        "quantile": quantile, "threshold": threshold,
                        "selection_2022": selected}
                if not selection_ok(selected):
                    continue
                item["validation_2023"] = base.evaluate(
                    frame, (2023,), threshold)
                item["validation_2024"] = base.evaluate(
                    frame, (2024,), threshold)
                diagnostics.append(public(item))
                if not (validation_ok(item["validation_2023"])
                        and validation_ok(item["validation_2024"])):
                    continue
                item["score"] = round(sum(
                    item[name]["return_pct"] - item[name]["dd_pct"]
                    for name in ("selection_2022", "validation_2023",
                                 "validation_2024")), 8)
                item["_frame"] = frame
                item["_model"] = model
                candidates.append(item)
    candidates.sort(key=lambda item: item["score"], reverse=True)
    frozen = candidates[0] if candidates else None
    results = checks = None
    if frozen:
        threshold = frozen["threshold"]
        results = {
            "selection_2022": base.evaluate(frozen["_frame"], (2022,), threshold),
            "validation_2023": base.evaluate(frozen["_frame"], (2023,), threshold),
            "validation_2024": base.evaluate(frozen["_frame"], (2024,), threshold),
            "test_2025": base.evaluate(frozen["_frame"], (2025,), threshold),
            "observed_2026": base.evaluate(frozen["_frame"], (2026,), threshold),
        }
        checks = {name: validation_ok(value) for name, value in results.items()}
    status = ("RESEARCH_CANDIDATE_AWAITING_VENUE_FORWARD"
              if checks and all(checks.values()) else "REJECTED")
    report = {
        "status": status,
        "protocol": {"fit": "2020-2021", "selection": "2022",
                     "validation": ["2023", "2024"], "test": "2025",
                     "observed": "2026", "entry": "next 5m market open",
                     "target_pct": TARGET_PCT,
                     "stop_pct_grid": list(STOP_PCTS),
                     "max_bars_grid": list(MAX_BARS_VALUES),
                     "money": "3% equity margin per trade, fixed 10x gross",
                     "collision": "stop first"},
        "proposal_rows": len(proposals),
        "qualified_before_test": len(candidates),
        "frozen_before_test": public(frozen) if frozen else None,
        "results": results,
        "checks": checks,
        "top_validation_diagnostics": sorted(
            diagnostics,
            key=lambda item: item["selection_2022"]["return_pct"],
            reverse=True)[:30],
        "limitations": [
            "Binance spot is structural history rather than X-Perp execution data.",
            "Geometry and threshold search create multiple-testing risk.",
            "Any surviving rule remains shadow-only pending new venue outcomes.",
        ],
        "sources": {**sources, str(Path(__file__)): digest(Path(__file__))},
    }
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({key: report[key] for key in (
        "status", "proposal_rows", "qualified_before_test",
        "frozen_before_test", "results", "checks")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
