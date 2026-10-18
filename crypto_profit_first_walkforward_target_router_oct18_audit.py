"""Walk-forward exact target router for the 15-minute crypto families.

For each evaluation period, three regressors learn realized R for 0.5R, 1R
and 2R targets using only older periods.  The immediately preceding period is
used solely to choose a probability-free expected-R cutoff and portfolio rails.
The target must imply a 0.5%-2% price move.  All target outcomes come from a
single conservative OHLC path with stop priority on ambiguous bars.
"""
from __future__ import annotations

import collections
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

import crypto_high_target_tree_oct5_audit as base
import crypto_precision_be_runner_oct18_audit as be
import crypto_precision_profit_first_oct18_audit as prior
from crypto_runtime_shadow_portfolio_oct3_audit import money


OUT = be.ROOT / "crypto_profit_first_walkforward_target_router_oct18.json"
ROWS_OUT = be.ROOT / "crypto_profit_first_walkforward_target_router_oct18.pkl"
DIRECT_ROWS_OUT = be.ROOT / "crypto_profit_first_walkforward_target_router_xperp_oct18.pkl"
TARGETS = (.5, 1.0, 2.0)
PERIODS = tuple(base.PERIODS)
TEST_PERIODS = tuple(PERIODS[4:])
QUANTILES = (.50, .60, .70, .75, .80, .85, .875, .90, .925, .95, .97)
LOSS_PAUSES = (3, 4, 5, 6, 7, 8)
HOURLY_CAPS = (1, 2, 3, 4)
MIN_TARGET_MOVE = .005
MAX_TARGET_MOVE = .02
KEYS = ("symbol", "entry_time", "direction", "family")
DIRECT_CANDLE_HASHES: dict[str, str] = {}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_target(target: float) -> pd.DataFrame:
    frame = prior.canonical_history(base.load_target(target))
    return frame.sort_values(list(KEYS)).reset_index(drop=True)


def historical_wide() -> pd.DataFrame:
    frames = {target: canonical_target(target) for target in TARGETS}
    reference = frames[1.0].copy()
    keys = reference[list(KEYS)]
    for target, frame in frames.items():
        if not keys.equals(frame[list(KEYS)]):
            raise RuntimeError(f"target {target:g} candidate rows are not aligned")
        suffix = str(target).replace(".", "_")
        reference[f"net_r_{suffix}"] = frame.net_r.to_numpy(float)
        reference[f"outcome_{suffix}"] = frame.outcome.astype(str).to_numpy()
        reference[f"exit_time_{suffix}"] = frame.exit_time.to_numpy(float)
    return reference


def replay_direct_target(frame: pd.DataFrame, target_r: float) -> pd.DataFrame:
    result = frame.copy().reset_index(drop=True)
    values = {}
    for symbol, group in result.groupby("symbol", sort=False):
        path = be.DIRECT_CACHE / f"{symbol}_15min_current.pkl"
        raw = path.read_bytes()
        DIRECT_CANDLE_HASHES[str(path)] = hashlib.sha256(raw).hexdigest()
        candles = pickle.loads(raw)
        lookup = {int(timestamp): index
                  for index, timestamp in enumerate(candles["time"])}
        for row in group.itertuples():
            index = lookup.get(int(row.entry_time))
            if index is None or index + 47 >= len(candles["time"]):
                raise RuntimeError(
                    f"direct target window missing: {symbol} {row.entry_time}")
            direction = str(row.direction)
            sign = 1.0 if direction == "LONG" else -1.0
            entry = float(row.entry)
            sl = float(row.sl)
            risk = abs(entry - sl)
            target = entry + sign * risk * target_r
            outcome = "EXPIRED"
            net_r = 0.0
            exit_time = float(candles["time"][index + 47]) + 900
            for bar in range(index, index + 48):
                opened = float(candles["open"][bar])
                high = float(candles["high"][bar])
                low = float(candles["low"][bar])
                stop_hit = low <= sl if sign > 0 else high >= sl
                if stop_hit:
                    fill = min(opened, sl) if sign > 0 else max(opened, sl)
                    outcome = "SL"
                    net_r = sign * (fill - entry) / risk
                    exit_time = float(candles["time"][bar]) + 900
                    break
                target_hit = high >= target if sign > 0 else low <= target
                if target_hit:
                    outcome = "TP2"
                    net_r = target_r
                    exit_time = float(candles["time"][bar]) + 900
                    break
            else:
                close = float(candles["close"][index + 47])
                net_r = sign * (close - entry) / risk
            values[int(row.Index)] = (net_r, outcome, exit_time)
    suffix = str(target_r).replace(".", "_")
    ordered = [values[int(index)] for index in result.index]
    result[f"net_r_{suffix}"] = [value[0] for value in ordered]
    result[f"outcome_{suffix}"] = [value[1] for value in ordered]
    result[f"exit_time_{suffix}"] = [value[2] for value in ordered]
    return result


def direct_wide() -> pd.DataFrame:
    reference = prior.direct_frame().sort_values(list(KEYS)).reset_index(drop=True)
    for target in TARGETS:
        replayed = replay_direct_target(reference, target)
        suffix = str(target).replace(".", "_")
        for field in ("net_r", "outcome", "exit_time"):
            reference[f"{field}_{suffix}"] = replayed[f"{field}_{suffix}"]
    reference["period"] = "xperp"
    return reference


def fit_models(frame: pd.DataFrame, fit_periods: tuple[str, ...],
               score: pd.DataFrame) -> np.ndarray:
    train = base.stage(frame, fit_periods)
    return fit_models_on_rows(train, score)


def fit_models_on_rows(train: pd.DataFrame,
                       score: pd.DataFrame) -> np.ndarray:
    schema = base.design(train)
    predictions = []
    for target in TARGETS:
        suffix = str(target).replace(".", "_")
        eligible = train[
            (train.risk_pct * target >= MIN_TARGET_MOVE)
            & (train.risk_pct * target <= MAX_TARGET_MOVE)
        ].copy()
        estimator = HistGradientBoostingRegressor(
            learning_rate=.05, max_iter=150, max_leaf_nodes=15,
            min_samples_leaf=300, l2_regularization=1.0,
            early_stopping=False, random_state=20260908)
        estimator.fit(base.matrix(eligible, schema),
                      eligible[f"net_r_{suffix}"])
        predictions.append(estimator.predict(base.matrix(score, schema)))
    return np.column_stack(predictions)


def route(frame: pd.DataFrame, predictions: np.ndarray) -> pd.DataFrame:
    if predictions.shape != (len(frame), len(TARGETS)):
        raise ValueError("prediction matrix does not match target router rows")
    allowed = np.column_stack([
        frame.risk_pct.to_numpy(float) * target >= MIN_TARGET_MOVE
        for target in TARGETS
    ]) & np.column_stack([
        frame.risk_pct.to_numpy(float) * target <= MAX_TARGET_MOVE
        for target in TARGETS
    ])
    scores = np.where(allowed, predictions, -np.inf)
    choice = np.argmax(scores, axis=1)
    quality = scores[np.arange(len(frame)), choice]
    keep = np.isfinite(quality)
    result = frame.loc[keep].copy().reset_index(drop=True)
    choice = choice[keep]
    result["model_quality"] = quality[keep]
    result["target_r"] = np.asarray(TARGETS)[choice]
    for output in ("net_r", "outcome", "exit_time"):
        values = np.empty(len(result), dtype=object if output == "outcome" else float)
        source = frame.loc[keep].reset_index(drop=True)
        for index, target in enumerate(TARGETS):
            mask = choice == index
            suffix = str(target).replace(".", "_")
            values[mask] = source.loc[mask, f"{output}_{suffix}"].to_numpy()
        result[output] = values
    return result


def candidate_rows(frame: pd.DataFrame, threshold: float) -> list[dict]:
    chosen = frame[frame.model_quality >= threshold].copy()
    chosen = chosen.sort_values(
        ["symbol", "entry_time", "model_quality", "family", "direction"],
        ascending=[True, True, False, False, False], kind="mergesort")
    chosen = chosen.drop_duplicates(["symbol", "entry_time"], keep="first")
    rows = []
    for row in chosen.to_dict("records"):
        raw_outcome = str(row["outcome"])
        rows.append({
            **row,
            "module": "walkforward_exact_target_router",
            "source": "walkforward_exact_target_router",
            "source_priority": 3,
            "module_priority": 999,
            "model_probability": float(row["model_quality"]),
            "signed_price_return": float(row["risk_pct"]) * float(row["net_r"]),
            "target_outcome": raw_outcome,
            "outcome": "SL" if raw_outcome == "SL" else "TP",
        })
    return rows


def evaluate(frame: pd.DataFrame, threshold: float, loss_pause: int,
             hourly_cap: int) -> tuple[dict, list[dict]]:
    accepted = be.ranked_runtime_gate(
        candidate_rows(frame, threshold), loss_pause=loss_pause,
        hourly_cap=hourly_cap)
    value = money(accepted, .03)
    counts = collections.Counter(float(row["target_r"]) for row in accepted)
    value.update({
        "target_counts": {str(target): counts[target] for target in TARGETS},
        "target_hit_rate": round(100 * sum(
            row["target_outcome"] == "TP2" for row in accepted) / len(accepted), 2),
        "non_loss_rate": round(100 * sum(
            float(row["net_r"]) >= 0 for row in accepted) / len(accepted), 2),
        "max_entries_same_timestamp": max(collections.Counter(
            int(float(row["entry_time"])) for row in accepted).values()),
    })
    return value, accepted


def policy_score(value: dict) -> float:
    months = list(value["monthly_return_pct"].values())
    floor = min(months) if months else -100.0
    return round(floor + .10 * sum(months) / max(1, len(months))
                 - .20 * value["dd_pct"]
                 - .50 * value["negative_months"], 8)


def objective_ok(value: dict) -> bool:
    months = list(value.get("monthly_return_pct", {}).values())
    return bool(months and min(months) >= 30
                and value["negative_months"] == 0
                and value["dd_pct"] <= 10
                and value["trades_per_calendar_day"] >= 3)


def choose_policy(calibration: pd.DataFrame) -> tuple[dict, list[dict]]:
    scores = calibration.model_quality.replace([np.inf, -np.inf], np.nan).dropna()
    frontier = []
    for quantile in QUANTILES:
        threshold = float(scores.quantile(quantile))
        for loss_pause in LOSS_PAUSES:
            for hourly_cap in HOURLY_CAPS:
                value, _ = evaluate(
                    calibration, threshold, loss_pause, hourly_cap)
                frontier.append({
                    "quantile": quantile, "threshold": threshold,
                    "loss_pause": loss_pause, "hourly_cap": hourly_cap,
                    "metrics": value, "score": policy_score(value),
                })
    frontier.sort(key=lambda item: (
        item["metrics"]["trades_per_calendar_day"] >= 3,
        item["metrics"]["dd_pct"] <= 10,
        item["score"], -item["hourly_cap"]), reverse=True)
    return frontier[0], frontier[:10]


def main() -> None:
    frame = historical_wide()
    steps = []
    all_test_rows = []
    for test_index in range(4, len(PERIODS)):
        calibration_period = PERIODS[test_index - 1]
        fit_periods = PERIODS[:test_index - 1]
        test_period = PERIODS[test_index]
        scoring = frame[frame.period.isin(
            (calibration_period, test_period))].copy().reset_index(drop=True)
        predictions = fit_models(frame, fit_periods, scoring)
        routed = route(scoring, predictions)
        calibration = routed[routed.period.eq(calibration_period)].copy()
        test = routed[routed.period.eq(test_period)].copy()
        policy, frontier = choose_policy(calibration)
        test_metrics, test_rows = evaluate(
            test, float(policy["threshold"]), int(policy["loss_pause"]),
            int(policy["hourly_cap"]))
        all_test_rows.extend(test_rows)
        steps.append({
            "fit_periods": list(fit_periods),
            "calibration_period": calibration_period,
            "test_period": test_period,
            "policy": policy,
            "calibration_frontier": frontier,
            "test_metrics": test_metrics,
        })

    combined = money(all_test_rows, .03)
    pd.DataFrame(all_test_rows).to_pickle(ROWS_OUT)

    # Production-like transfer: fit through 2024, select on 2025, and reveal
    # the direct 2026 X-Perp window only after the policy is fixed.
    direct = direct_wide()
    calibration_period = "2025"
    fit_periods = tuple(PERIODS[:8])  # through 2024
    historical_calibration = frame[frame.period.eq(calibration_period)].copy()
    scoring = pd.concat([historical_calibration, direct], ignore_index=True)
    routed = route(scoring, fit_models(frame, fit_periods, scoring))
    calibration = routed[routed.period.eq(calibration_period)].copy()
    direct_routed = routed[routed.period.eq("xperp")].copy()
    policy, frontier = choose_policy(calibration)
    direct_metrics, direct_rows = evaluate(
        direct_routed, float(policy["threshold"]), int(policy["loss_pause"]),
        int(policy["hourly_cap"]))
    pd.DataFrame(direct_rows).to_pickle(DIRECT_ROWS_OUT)

    # Venue-adaptive walk-forward.  A test month can use older X-Perp labels,
    # but never its own outcomes or any later month.  Historical 2026 is kept
    # out so the direct venue remains the only 2026 adaptation source.
    direct = direct.copy()
    direct["month"] = pd.to_datetime(
        direct.entry_time, unit="s", utc=True).dt.strftime("%Y-%m")
    historical_fit = frame[~frame.period.eq("2026")].copy()
    venue_steps = []
    venue_rows = []
    venue_months = ("2026-06", "2026-07", "2026-08", "2026-09", "2026-10")
    for test_month in venue_months:
        calibration_month = str(pd.Period(test_month, freq="M") - 1)
        older_direct = direct[direct.month < calibration_month].copy()
        train = pd.concat([historical_fit, older_direct], ignore_index=True)
        scoring = direct[direct.month.isin(
            (calibration_month, test_month))].copy().reset_index(drop=True)
        routed = route(scoring, fit_models_on_rows(train, scoring))
        calibration = routed[routed.month.eq(calibration_month)].copy()
        test = routed[routed.month.eq(test_month)].copy()
        month_policy, month_frontier = choose_policy(calibration)
        metrics, rows = evaluate(
            test, float(month_policy["threshold"]),
            int(month_policy["loss_pause"]), int(month_policy["hourly_cap"]))
        venue_rows.extend(rows)
        venue_steps.append({
            "fit_direct_before": calibration_month,
            "calibration_month": calibration_month,
            "test_month": test_month,
            "policy": month_policy,
            "calibration_frontier": month_frontier,
            "test_metrics": metrics,
        })
    venue_combined = money(venue_rows, .03)

    objective_checks = {
        "historical_walkforward": objective_ok(combined),
        "frozen_direct_transfer": objective_ok(direct_metrics),
        "venue_adaptive_walkforward": objective_ok(venue_combined),
    }
    report = {
        "status": ("WALKFORWARD_TARGET_ROUTER_CANDIDATE"
                   if all(objective_checks.values())
                   else "REJECTED_OBJECTIVE"),
        "ready_for_live_orders": False,
        "protocol": {
            "targets_r": list(TARGETS),
            "target_price_move_range_pct": [100 * MIN_TARGET_MOVE,
                                            100 * MAX_TARGET_MOVE],
            "fit": "all periods older than the calibration period",
            "selection": "immediately preceding period only",
            "test": "next chronological period",
            "entry": "next 15-minute market open",
            "orders": "market entry and market close; no limit entry",
            "money": "3% current-equity margin, fixed 10x gross",
        },
        "walkforward": steps,
        "combined_walkforward_2022_2026": combined,
        "direct_transfer": {
            "fit_periods": list(fit_periods),
            "calibration_period": calibration_period,
            "policy": policy,
            "calibration_frontier": frontier,
            "fresh_xperp": direct_metrics,
        },
        "venue_adaptive_walkforward": venue_steps,
        "venue_adaptive_combined": venue_combined,
        "objective_checks": objective_checks,
        "sources": {
            **{str(base.cache_path(target, period)):
               digest(base.cache_path(target, period))
               for target in TARGETS for period in PERIODS},
            **DIRECT_CANDLE_HASHES,
            str(prior.FRESH_NEAR): digest(prior.FRESH_NEAR),
            str(prior.FRESH_FAR): digest(prior.FRESH_FAR),
            str(Path(__file__).resolve()): digest(Path(__file__).resolve()),
        },
        "limitations": [
            "Gross results exclude fees at user request.",
            "Historical named periods contain the available six-month windows.",
            "Direct X-Perp is a short transfer window, not a full market cycle.",
            "No backtest can guarantee a non-negative future month.",
        ],
    }
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({
        "walkforward": [{"period": step["test_period"],
                          "policy": {key: step["policy"][key] for key in (
                              "quantile", "threshold", "loss_pause", "hourly_cap")},
                          "metrics": step["test_metrics"]} for step in steps],
        "combined": combined,
        "direct_policy": {key: policy[key] for key in (
            "quantile", "threshold", "loss_pause", "hourly_cap")},
        "fresh_xperp": direct_metrics,
        "venue_adaptive": [{
            "month": step["test_month"],
            "policy": {key: step["policy"][key] for key in (
                "quantile", "threshold", "loss_pause", "hourly_cap")},
            "metrics": step["test_metrics"],
        } for step in venue_steps],
        "venue_adaptive_combined": venue_combined,
    }, indent=2))


if __name__ == "__main__":
    main()
