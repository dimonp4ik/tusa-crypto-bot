"""Verify the deployed Profit-First router against its exact research replay.

The verifier checks every raw family decision with closed-candle prefixes,
replays the runtime's first-passing-family contract, and quantifies how much
candidate admission changes when a live ticker drifts from the immutable entry
bar open.  It never reads credentials or places orders.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

import crypto_high_target_tree_oct5_audit as base
import crypto_profit_first_exact_oct31_1700_forward as exact
import crypto_profit_first_material_barbell_oct18_audit as joiner
import crypto_profit_first_sequential_runner_oct30_audit as sequential
from crypto_runtime_shadow_portfolio_oct3_audit import runtime_gate
from src import crypto_profit_first_material_shadow_router as router


ROOT = Path("reports/audit_2026_09_08")
FAR = ROOT / "crypto_xperp_raw_router_audit_2_0_oct31_1700_material_raw_families.csv"
NEAR = ROOT / "crypto_xperp_raw_router_audit_0_5_oct31_1700_material_raw_families.csv"
CACHE = ROOT / "xperp_15m_oct31_1700_material"
RESEARCH_ROWS = ROOT / "crypto_profit_first_exact_oct31_1700.pkl"
OUT = ROOT / "verify_crypto_profit_first_runtime_parity_nov3.json"
ROWS_OUT = ROOT / "crypto_profit_first_runtime_order_nov3.csv"
FAMILY_ORDER = {"range_reversion": 0, "trend_pullback": 1, "breakout": 2}
CONTEXT_FIELDS = (
    "btc_fast_slow_atr", "btc_eff20", "btc_return20_atr", "btc_z20",
    "btc_atr_pct", "btc_slow_long_atr", "btc_return96_atr",
    "btc_eff96", "btc_regime",
)
TOLERANCE = 1e-10


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def identity(row: dict | pd.Series) -> tuple[str, int, str, str]:
    return (str(row["symbol"]), int(float(row["entry_time"])),
            str(row["direction"]), str(row["family"]))


def portable_probability(row) -> float:
    return router.model_probability(
        symbol=str(row.symbol),
        signal={
            "direction": str(row.direction), "family": str(row.family),
            "eff_ratio": float(row.eff_ratio), "z": float(row.z),
        },
        context={field: getattr(row, field) for field in CONTEXT_FIELDS},
        entry_time=float(row.entry_time), risk_pct=float(row.risk_pct),
    )


def feature_matrix(frame: pd.DataFrame) -> np.ndarray:
    rows = []
    for row in frame.itertuples(index=False):
        values = router.feature_values(
            symbol=str(row.symbol),
            signal={
                "direction": str(row.direction), "family": str(row.family),
                "eff_ratio": float(row.eff_ratio), "z": float(row.z),
            },
            context={field: getattr(row, field) for field in CONTEXT_FIELDS},
            entry_time=float(row.entry_time), risk_pct=float(row.risk_pct),
        )
        rows.append([
            float(values.get(name, math.nan))
            for name in router.MODEL["features"]
        ])
    return np.asarray(rows, dtype=float)


def portable_probability_batch(matrix: np.ndarray) -> np.ndarray:
    """Vectorized equivalent of the deployed pure-Python tree evaluator."""
    raw = np.full(len(matrix), float(router.MODEL["baseline"]), dtype=float)
    for source_nodes in router.MODEL["trees"]:
        nodes = np.asarray(source_nodes, dtype=float)
        indices = np.zeros(len(matrix), dtype=np.int64)
        while True:
            active = ~nodes[indices, 6].astype(bool)
            if not active.any():
                break
            rows = np.flatnonzero(active)
            current = indices[rows]
            features = nodes[current, 0].astype(np.int64)
            observed = matrix[rows, features]
            missing = np.isnan(observed)
            go_left = np.where(
                missing, nodes[current, 2].astype(bool),
                observed <= nodes[current, 1])
            indices[rows] = np.where(
                go_left, nodes[current, 3], nodes[current, 4]).astype(np.int64)
        raw += nodes[indices, 5]
    output = np.empty(len(raw), dtype=float)
    positive = raw >= 0
    output[positive] = 1.0 / (1.0 + np.exp(-raw[positive]))
    exp_value = np.exp(raw[~positive])
    output[~positive] = exp_value / (1.0 + exp_value)
    return output


def selected_candidates(scored: pd.DataFrame) -> pd.DataFrame:
    selected = scored[
        scored.model_probability.ge(float(router.MODEL["quality_threshold"]))
    ].copy()
    selected["_family_order"] = selected.family.map(FAMILY_ORDER)
    return (selected.sort_values(
        ["symbol", "entry_time", "_family_order"], kind="stable")
        .drop_duplicates(["symbol", "entry_time"], keep="first")
        .drop(columns="_family_order"))


def routed(frame: pd.DataFrame) -> pd.DataFrame:
    values = frame.copy()
    values["period"] = "direct"
    literal = sequential.attach_current(values, exact.paths(CACHE))
    literal = sequential.apply_fraction(literal, exact.FRACTION)
    rows = runtime_gate(base.standardized(
        literal, "profit_first_exact_2r"))
    return pd.DataFrame(rows).sort_values(
        ["entry_time", "symbol", "direction"]).reset_index(drop=True)


def route_parity(expected: pd.DataFrame, observed: pd.DataFrame) -> dict:
    keys = ["symbol", "entry_time", "direction", "family"]
    left = expected.set_index(keys)
    right = observed.set_index(keys)
    common = left.index.intersection(right.index)
    net_delta = max((abs(float(left.loc[key, "net_r"])
                         - float(right.loc[key, "net_r"]))
                     for key in common), default=0.0)
    exit_delta = max((abs(float(left.loc[key, "exit_time"])
                          - float(right.loc[key, "exit_time"]))
                      for key in common), default=0.0)
    left_ids, right_ids = set(left.index), set(right.index)
    passed = bool(left_ids == right_ids and net_delta <= TOLERANCE
                  and exit_delta <= TOLERANCE)
    return {
        "expected_rows": len(expected), "observed_rows": len(observed),
        "matched_rows": len(common),
        "missing": [list(value) for value in sorted(left_ids - right_ids)],
        "unexpected": [list(value) for value in sorted(right_ids - left_ids)],
        "maximum_net_r_delta": net_delta,
        "maximum_exit_time_delta_seconds": exit_delta,
        "passes": passed,
    }


def compact_metrics(frame: pd.DataFrame) -> dict:
    values = exact.metrics(frame, .03)
    return {key: values[key] for key in (
        "n", "wr", "return_pct", "dd_pct", "max_loss_streak",
        "negative_months", "monthly_return_pct", "trades_per_calendar_day",
        "literal_2r_before_be_rate_pct", "tp1_or_better_rate_pct",
        "projected_monthly_return_pct", "economic_gate",
    )}


def sensitivity(frame: pd.DataFrame, matrix: np.ndarray,
                baseline: dict) -> dict:
    output = {}
    risk_index = list(router.MODEL["features"]).index("risk_pct")
    for multiplier in (.99, .995, .9975, .999, 1.001, 1.0025, 1.005, 1.01):
        rows = frame.copy()
        rows["risk_pct"] = rows.risk_pct / multiplier
        values = matrix.copy()
        values[:, risk_index] = rows.risk_pct.to_numpy(dtype=float)
        rows["model_probability"] = portable_probability_batch(values)
        rows = rows[rows.risk_pct.between(
            float(router.MODEL["risk_pct_min"]),
            float(router.MODEL["risk_pct_max"]), inclusive="both")].copy()
        chosen = selected_candidates(rows)
        ids = {
            (str(row.symbol), int(row.entry_time)):
            (str(row.direction), str(row.family))
            for row in chosen.itertuples(index=False)
        }
        common = baseline.keys() & ids.keys()
        output[f"{100 * (multiplier - 1):+.2f}pct"] = {
            "candidate_groups": len(ids),
            "removed_groups": len(baseline.keys() - ids.keys()),
            "added_groups": len(ids.keys() - baseline.keys()),
            "changed_family_or_direction": sum(
                baseline[key] != ids[key] for key in common),
        }
    return output


def main() -> None:
    joined = joiner.join_direct_outcomes(FAR, NEAR)
    raw = joined[joined.symbol.isin(base.ROBUST_SYMBOLS)].copy()
    eligible = raw[raw.risk_pct.between(
        float(router.MODEL["risk_pct_min"]),
        float(router.MODEL["risk_pct_max"]), inclusive="both")].copy()
    research_scored = eligible.copy()
    research_scored["model_probability"] = [
        portable_probability(row)
        for row in research_scored.itertuples(index=False)
    ]
    eligible_matrix = feature_matrix(research_scored)
    batch_probability_delta = float(np.max(np.abs(
        research_scored.model_probability.to_numpy(dtype=float)
        - portable_probability_batch(eligible_matrix))))
    research_candidates = research_scored[
        research_scored.model_probability.ge(
            float(router.MODEL["quality_threshold"]))].copy()
    runtime_candidates = selected_candidates(research_scored)

    cache = {}
    hashes = {
        str(FAR): digest(FAR), str(NEAR): digest(NEAR),
        str(RESEARCH_ROWS): digest(RESEARCH_ROWS),
        str(router.ARTIFACT_PATH): digest(router.ARTIFACT_PATH),
        str(Path(router.__file__).resolve()): digest(
            Path(router.__file__).resolve()),
        str(Path("main.py").resolve()): digest(Path("main.py").resolve()),
    }
    for symbol in sorted(raw.symbol.unique()):
        path = CACHE / f"{symbol}_15min_current.pkl"
        cache[symbol] = pickle.loads(path.read_bytes())
        hashes[str(path)] = digest(path)
    lookups = {
        symbol: {int(value): index for index, value in enumerate(candles["time"])}
        for symbol, candles in cache.items()
    }
    expected = {
        (str(row.symbol), int(row.entry_time)): row
        for row in runtime_candidates.itertuples(index=False)
    }
    feature_fields = (
        "eff_ratio", "z", "btc_fast_slow_atr", "btc_eff20",
        "btc_return20_atr", "btc_z20", "btc_atr_pct",
        "btc_slow_long_atr", "btc_return96_atr", "btc_eff96",
    )
    max_error = {field: 0.0 for field in (
        *feature_fields, "risk_pct", "model_probability", "entry", "sl", "tp")}
    failures = []
    groups_checked = accepted_checked = 0
    btc = cache["BTCUSDT"]
    for (symbol, stamp), group in raw.groupby(
            ["symbol", "entry_time"], sort=False):
        groups_checked += 1
        stamp = int(stamp)
        source = cache[str(symbol)]
        index = lookups[str(symbol)][stamp]
        btc_index = lookups["BTCUSDT"][stamp]
        candles = {
            key: list(values[max(0, index - 400):index])
            for key, values in source.items()
        }
        btc_candles = {
            key: list(values[max(0, btc_index - 400):btc_index])
            for key, values in btc.items()
        }
        entry = float(group.iloc[0].entry)
        observed = router.shadow_setups(
            candles, btc_candles, entry_time=stamp,
            market_price=entry, model_price=entry, symbol=str(symbol))
        wanted = expected.get((str(symbol), stamp))
        if wanted is None:
            if observed:
                failures.append({
                    "symbol": str(symbol), "entry_time": stamp,
                    "reason": "unexpected_runtime_candidate",
                    "observed": identity({**observed[0], "symbol": symbol,
                                          "entry_time": stamp}),
                })
            continue
        accepted_checked += 1
        if len(observed) != 1:
            failures.append({
                "symbol": str(symbol), "entry_time": stamp,
                "reason": "missing_runtime_candidate",
                "expected": identity(wanted._asdict()),
            })
            continue
        row = observed[0]
        if (str(row["direction"]), str(row["family"])) != (
                str(wanted.direction), str(wanted.family)):
            failures.append({
                "symbol": str(symbol), "entry_time": stamp,
                "reason": "family_or_direction_mismatch",
                "expected": [str(wanted.direction), str(wanted.family)],
                "observed": [str(row["direction"]), str(row["family"])],
            })
            continue
        for field in feature_fields:
            max_error[field] = max(
                max_error[field],
                abs(float(row[field]) - float(getattr(wanted, field))))
        risk = abs(float(wanted.entry) - float(wanted.sl))
        sign = 1.0 if str(wanted.direction) == "LONG" else -1.0
        comparisons = {
            "risk_pct": (row["model_risk_pct"], wanted.risk_pct),
            "model_probability": (
                row["model_quality"], wanted.model_probability),
            "entry": (row["entry"], wanted.entry),
            "sl": (row["sl"], wanted.sl),
            "tp": (row["tp"], float(wanted.entry) + sign * risk * .5),
        }
        for field, (actual, wanted_value) in comparisons.items():
            max_error[field] = max(
                max_error[field], abs(float(actual) - float(wanted_value)))

    frozen_route = pd.read_pickle(RESEARCH_ROWS)
    scored_lookup = {
        identity(row._asdict()): float(row.model_probability)
        for row in research_scored.itertuples(index=False)
    }
    accepted_probability_delta = max((
        abs(scored_lookup[identity(row)] - float(row["model_probability"]))
        for row in frozen_route.to_dict("records")
    ), default=0.0)
    research_route = routed(research_candidates)
    research_parity = route_parity(frozen_route, research_route)
    runtime_route = routed(runtime_candidates)
    runtime_route.to_csv(ROWS_OUT, index=False)

    frozen_ids = {identity(row) for row in frozen_route.to_dict("records")}
    runtime_ids = {identity(row) for row in runtime_route.to_dict("records")}
    route_changes = {
        "removed_from_frozen_research": [
            list(value) for value in sorted(frozen_ids - runtime_ids)],
        "added_by_runtime_family_order": [
            list(value) for value in sorted(runtime_ids - frozen_ids)],
    }

    open_gaps = []
    for row in raw[["symbol", "entry_time", "entry"]].drop_duplicates().itertuples(
            index=False):
        index = lookups[str(row.symbol)][int(row.entry_time)]
        previous_close = float(cache[str(row.symbol)]["close"][index - 1])
        open_gaps.append(float(row.entry) / previous_close - 1.0)
    abs_gaps = np.abs(np.asarray(open_gaps, dtype=float))

    baseline_ids = {
        (str(row.symbol), int(row.entry_time)):
        (str(row.direction), str(row.family))
        for row in runtime_candidates.itertuples(index=False)
    }
    raw_matrix = feature_matrix(raw)
    decision_sensitivity = sensitivity(raw, raw_matrix, baseline_ids)

    function_pass = bool(
        not failures and all(value <= TOLERANCE for value in max_error.values()))
    passed = bool(
        accepted_probability_delta <= TOLERANCE
        and batch_probability_delta <= TOLERANCE
        and research_parity["passes"] and function_pass)
    report = {
        "status": (
            "RUNTIME_REPLAY_PARITY_PASSED" if passed
            else "RUNTIME_REPLAY_PARITY_FAILED"),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "ready_for_live_orders": False,
        "orders_possible": False,
        "runtime_change_allowed": passed,
        "model_artifact_parity": {
            "rows": len(research_scored),
            "saved_accepted_rows": len(frozen_route),
            "maximum_saved_accepted_probability_delta": (
                accepted_probability_delta),
            "maximum_scalar_vs_batch_probability_delta": (
                batch_probability_delta),
            "passes": bool(
                accepted_probability_delta <= TOLERANCE
                and batch_probability_delta <= TOLERANCE),
        },
        "router_function_parity": {
            "decision_groups_checked": groups_checked,
            "accepted_groups_checked": accepted_checked,
            "failures": failures[:100],
            "failure_count": len(failures),
            "maximum_feature_error": max_error,
            "passes": function_pass,
        },
        "frozen_research_route_parity": research_parity,
        "runtime_first_family_route": {
            "model_accepted_rows_before_first_family": len(research_candidates),
            "candidate_groups_after_first_family": len(runtime_candidates),
            "portfolio_rows": len(runtime_route),
            "metrics_at_3pct_margin_10x": compact_metrics(runtime_route),
            "frozen_research_metrics_at_3pct_margin_10x": compact_metrics(
                frozen_route),
            "identity_changes": route_changes,
        },
        "entry_price_contract": {
            "model_price": "immutable forming 15-minute candle open",
            "execution_price": "current X-Perp market price",
            "execution_rule": (
                "actual 2ATR risk must also remain inside the frozen 1-2% band"
            ),
            "previous_close_to_next_open_absolute_gap_fraction": {
                "n": len(abs_gaps), "maximum": float(abs_gaps.max()),
                "p50": float(np.percentile(abs_gaps, 50)),
                "p90": float(np.percentile(abs_gaps, 90)),
                "p95": float(np.percentile(abs_gaps, 95)),
                "p99": float(np.percentile(abs_gaps, 99)),
            },
            "candidate_sensitivity_to_market_price_multiplier": (
                decision_sensitivity),
        },
        "sources": {
            **hashes,
            str(Path(__file__).resolve()): digest(Path(__file__).resolve()),
        },
        "limitations": [
            "Parity proves deterministic implementation, not future profitability.",
            "The snapshot has no historical one-minute ticker at the scheduler instant.",
            "Sensitivity counts admission changes and is not a repriced return estimate.",
            "Gross performance excludes fees at user request.",
        ],
    }
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                   encoding="utf-8")
    print(json.dumps({
        "status": report["status"],
        "model_artifact_parity": report["model_artifact_parity"],
        "router_function_parity": report["router_function_parity"],
        "frozen_research_route_parity": research_parity,
        "runtime_first_family_route": report["runtime_first_family_route"],
        "entry_price_contract": report["entry_price_contract"],
    }, indent=2), flush=True)
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
