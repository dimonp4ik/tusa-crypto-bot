"""Verify end-to-end parity of the frozen precision shadow router.

The direct audit rows contain every raw family candidate.  This verifier
rebuilds each candidate from closed candle prefixes, compares every model
feature, and checks that the runtime chooses the same first qualifying family
at every symbol/timestamp group.
"""
from __future__ import annotations

import hashlib
import json
import math
import pickle
from pathlib import Path

import pandas as pd

import crypto_high_target_tree_oct5_audit as base
import crypto_precision_profit_first_oct18_audit as audit
from src.crypto_precision_profit_first_shadow import (
    MODEL_PATH, QUALITY_THRESHOLD, _MODEL, _feature_values, shadow_setups,
)
from src.crypto_venue_router import btc_context, family_signals
from src.portable_hgb import score_named


CACHE_DIR = audit.ROOT / "xperp_15m_oct17_material"
OUT = audit.ROOT / "crypto_precision_profit_first_runtime_parity.json"
TOLERANCE = 1e-11


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tail_before(series: dict, index: int, length: int) -> dict:
    """Return only the closed tail used by the runtime calculations."""
    start = max(0, index - length)
    return {key: values[start:index] for key, values in series.items()}


def expected_values(row: dict) -> dict[str, float]:
    categories = {
        "symbol": str(row["symbol"]),
        "direction": str(row["direction"]),
        "family": str(row["family"]),
        "utc_session": str(row["utc_session"]),
        "btc_regime": str(row["btc_regime"]),
    }
    result = {}
    for name in _MODEL["features"]:
        if "=" in name:
            field, expected = name.split("=", 1)
            result[name] = float(categories.get(field) == expected)
        else:
            result[name] = float(row[name])
    return result


def main() -> None:
    direct = audit.direct_frame()
    direct = direct[direct.symbol.isin(base.ROBUST_SYMBOLS)].copy()
    sources = {
        str(audit.FRESH_FAR): digest(audit.FRESH_FAR),
        str(audit.FRESH_NEAR): digest(audit.FRESH_NEAR),
        str(MODEL_PATH): digest(MODEL_PATH),
        str(Path(__file__)): digest(Path(__file__)),
    }
    caches = {}
    lookups = {}

    def candles(symbol: str) -> dict:
        if symbol not in caches:
            path = CACHE_DIR / f"{symbol}_15min_current.pkl"
            raw = path.read_bytes()
            sources[str(path)] = hashlib.sha256(raw).hexdigest()
            caches[symbol] = pickle.loads(raw)
            lookups[symbol] = {
                int(timestamp): index
                for index, timestamp in enumerate(caches[symbol]["time"])
            }
        return caches[symbol]

    btc = candles("BTCUSDT")
    btc_lookup = lookups["BTCUSDT"]
    context_cache = {}
    failures = []
    feature_rows = 0
    groups_checked = 0
    max_feature_error = 0.0
    max_probability_error = 0.0

    for (symbol, timestamp), group in direct.groupby(
            ["symbol", "entry_time"], sort=True):
        timestamp = int(timestamp)
        series = candles(str(symbol))
        index = lookups[str(symbol)].get(timestamp)
        btc_index = btc_lookup.get(timestamp)
        if index is None or btc_index is None:
            failures.append({"symbol": symbol, "entry_time": timestamp,
                             "reason": "entry timestamp absent from cache"})
            continue
        prefix = tail_before(series, index, 100)
        if timestamp not in context_cache:
            btc_prefix = tail_before(btc, btc_index, 385)
            context_cache[timestamp] = (btc_prefix, btc_context(btc_prefix))
        btc_prefix, context = context_cache[timestamp]
        if context is None:
            failures.append({"symbol": symbol, "entry_time": timestamp,
                             "reason": "BTC context unexpectedly absent"})
            continue
        signals = family_signals(prefix)
        by_identity = {
            (signal["family"], signal["direction"]): signal
            for signal in signals
        }
        rows = group.to_dict("records")
        expected_passes = []
        entry = float(rows[0]["entry"])
        for row in rows:
            identity = (row["family"], row["direction"])
            signal = by_identity.get(identity)
            if signal is None:
                failures.append({
                    "symbol": symbol, "entry_time": timestamp,
                    "family": row["family"], "direction": row["direction"],
                    "reason": "runtime family signal missing",
                })
                continue
            observed = _feature_values(
                str(symbol), signal, context, timestamp, float(row["entry"]))
            expected = expected_values(row)
            if observed is None:
                failures.append({"symbol": symbol, "entry_time": timestamp,
                                 "reason": "runtime features absent"})
                continue
            row_error = max(abs(observed[name] - expected[name])
                            for name in _MODEL["features"])
            max_feature_error = max(max_feature_error, row_error)
            expected_probability = score_named(_MODEL, expected)
            observed_probability = score_named(_MODEL, observed)
            probability_error = abs(expected_probability - observed_probability)
            max_probability_error = max(max_probability_error,
                                        probability_error)
            if row_error > TOLERANCE or probability_error > TOLERANCE:
                failures.append({
                    "symbol": symbol, "entry_time": timestamp,
                    "family": row["family"], "direction": row["direction"],
                    "reason": "feature or probability mismatch",
                    "max_feature_error": row_error,
                    "probability_error": probability_error,
                })
            if expected_probability >= QUALITY_THRESHOLD:
                expected_passes.append(identity)
            feature_rows += 1

        # family_signals defines the audited same-symbol priority.  Filter the
        # passing identities through that order before comparing the one setup
        # returned by the deployed shadow router.
        passing = set(expected_passes)
        ordered_passes = [
            (signal["family"], signal["direction"])
            for signal in signals
            if (signal["family"], signal["direction"]) in passing
        ]
        actual = shadow_setups(
            prefix, btc_prefix, entry_time=timestamp,
            market_price=entry, symbol=str(symbol))
        actual_identity = ((actual[0]["family"], actual[0]["direction"])
                           if actual else None)
        expected_identity = ordered_passes[0] if ordered_passes else None
        if actual_identity != expected_identity:
            failures.append({
                "symbol": symbol, "entry_time": timestamp,
                "reason": "threshold/priority decision mismatch",
                "expected": expected_identity, "actual": actual_identity,
            })
        if actual:
            setup = actual[0]
            risk = abs(float(setup["entry"]) - float(setup["sl"]))
            contract_ok = (
                setup["entry_source"] == "MARKET"
                and setup["_shadow_experiment"] is True
                and setup["_precision_profit_first_shadow"] is True
                and math.isclose(abs(float(setup["tp"]) - entry),
                                 .25 * risk, rel_tol=0, abs_tol=TOLERANCE)
                and math.isclose(float(setup["runner_target_r"]), 2.0)
                and "position_size" not in setup
                and "margin" not in setup
            )
            if not contract_ok:
                failures.append({"symbol": symbol, "entry_time": timestamp,
                                 "reason": "runtime contract mismatch"})
        groups_checked += 1

    report = {
        "status": "PASS" if not failures else "FAIL",
        "raw_candidate_rows": len(direct),
        "feature_rows_checked": feature_rows,
        "symbol_timestamp_groups_checked": groups_checked,
        "max_feature_error": max_feature_error,
        "max_probability_error": max_probability_error,
        "quality_threshold": QUALITY_THRESHOLD,
        "failures": failures[:100],
        "failure_count": len(failures),
        "invariants": [
            "closed-candle family signal parity",
            "all portable-model features match direct replay",
            "portable probability and threshold decision parity",
            "same-symbol family priority parity",
            "market entry, shadow-only, fixed 0.25R/2R bracket, no sizing",
        ],
        "sources": sources,
    }
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True),
                   encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items()
                      if key not in ("failures", "sources")}, indent=2))
    if failures:
        print(json.dumps(failures[:10], indent=2))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
