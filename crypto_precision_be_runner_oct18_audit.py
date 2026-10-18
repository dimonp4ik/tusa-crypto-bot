"""Chronological audit of the full-position 2R runner.

The frozen HGB predicts a 2R completion.  At +0.25R the whole position moves
to breakeven; no partial order is required.  Only the highest-quality setup on
each closed 15-minute bar is admitted, preventing simultaneous signal bursts.
All policy choices finish on 2022-2023 before the 2024+ and direct X-Perp
reveals.
"""
from __future__ import annotations

import collections
import hashlib
import heapq
import json
import pickle
from pathlib import Path

import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

import crypto_2r_precision_barbell_oct5_audit as precision
import crypto_high_target_tree_oct5_audit as base
import crypto_precision_profit_first_oct18_audit as prior
from crypto_runtime_shadow_portfolio_oct3_audit import money


ROOT = Path("reports/audit_2026_09_08")
OUT = ROOT / "crypto_precision_be_runner_oct18.json"
ROWS_OUT = ROOT / "crypto_precision_be_runner_xperp_oct17.pkl"
QUANTILES = (
    .03, .04, .05, .06, .08, .10, .20, .30, .40, .50,
    .60, .70, .80, .85, .90, .925, .95, .97, .98,
)
LOSS_PAUSES = (3, 4, 5, 6, 7, 8)
HOURLY_CAPS = (1, 2, 3, 4)
ACTIVATION_R = .25
RUNNER_R = 2.0
SELECTION_MONTHS = (
    "2022-02", "2022-03", "2022-04", "2022-05", "2022-06")
CONFIRMATION_MONTHS = (
    "2022-08", "2022-09", "2022-10", "2022-11", "2022-12",
    "2023-02", "2023-03", "2023-04", "2023-05", "2023-06")
VALIDATION_MONTHS = (
    "2024-02", "2024-03", "2024-04", "2024-05", "2024-06")
LATER_MONTHS = (
    "2025-02", "2025-03", "2025-04", "2025-05", "2025-06",
    "2026-03", "2026-04", "2026-05", "2026-06", "2026-07")
DIRECT_MONTHS = (
    "2026-05", "2026-06", "2026-07", "2026-08", "2026-09")
HISTORICAL_CACHE = Path("backtest_cache")
DIRECT_CACHE = ROOT / "xperp_15m_oct17_material"
PERIOD_CACHE = {
    "2020h1": "18000_end1596153600000",
    "2020h2": "18000_end1612051200000",
    "2021h1": "18000_end1627689600000",
    "2021h2": "18000_end1643587200000",
    "2022h1": "18000_end1659225600000",
    "2022h2": "18000_end1675123200000",
    "2023": "18000_end1690761600000",
    "2024": "18000_end1722384000000",
    "2025": "18000_end1753920000000",
    "2026": "18000",
}
CANDLE_SOURCES: dict[str, str] = {}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_candles(path: Path) -> dict:
    raw = path.read_bytes()
    CANDLE_SOURCES[str(path)] = hashlib.sha256(raw).hexdigest()
    return pickle.loads(raw)


def _runtime_runner(candles: dict, index: int, *, direction: str,
                    entry: float, sl: float) -> tuple[str, float, float]:
    """Replay the live 0.25R-to-BE/2R contract on 48 execution bars.

    Resting stops win ambiguous bars.  On the activation bar the original stop
    is still active and breakeven starts on the next bar, matching main.py.
    """
    sign = 1.0 if direction == "LONG" else -1.0
    risk = abs(entry - sl)
    activation = entry + sign * risk * ACTIVATION_R
    target = entry + sign * risk * RUNNER_R
    active = False
    last_index = min(index + 47, len(candles["time"]) - 1)
    for bar in range(index, last_index + 1):
        opened = float(candles["open"][bar])
        high = float(candles["high"][bar])
        low = float(candles["low"][bar])
        if not active:
            stop_hit = low <= sl if sign > 0 else high >= sl
            if stop_hit:
                fill = min(opened, sl) if sign > 0 else max(opened, sl)
                return "SL", sign * (fill - entry) / risk, float(candles["time"][bar]) + 900
            target_hit = high >= target if sign > 0 else low <= target
            if target_hit:
                return "TP2", RUNNER_R, float(candles["time"][bar]) + 900
            activation_hit = high >= activation if sign > 0 else low <= activation
            if activation_hit:
                active = True
                continue
        else:
            be_hit = low <= entry if sign > 0 else high >= entry
            if be_hit:
                fill = min(opened, entry) if sign > 0 else max(opened, entry)
                return "BE", sign * (fill - entry) / risk, float(candles["time"][bar]) + 900
            target_hit = high >= target if sign > 0 else low <= target
            if target_hit:
                return "TP2", RUNNER_R, float(candles["time"][bar]) + 900
    if active:
        return "BE", 0.0, float(candles["time"][last_index]) + 900
    close = float(candles["close"][last_index])
    return "EXPIRED", sign * (close - entry) / risk, float(candles["time"][last_index]) + 900


def attach_runtime_runner(frame: pd.DataFrame, *, direct: bool) -> pd.DataFrame:
    """Attach outcomes from raw OHLC instead of combining two target labels."""
    result = frame.copy().reset_index(drop=True)
    outcomes: dict[int, tuple[str, float, float]] = {}
    group_fields = ["symbol"] if direct else ["period", "symbol"]
    for identity, group in result.groupby(group_fields, sort=False):
        if direct:
            symbol = str(identity[0] if isinstance(identity, tuple) else identity)
            path = DIRECT_CACHE / f"{symbol}_15min_current.pkl"
        else:
            period, symbol = identity
            path = HISTORICAL_CACHE / f"{symbol}_15min_{PERIOD_CACHE[str(period)]}.pkl"
        candles = load_candles(path)
        lookup = {int(timestamp): index
                  for index, timestamp in enumerate(candles["time"])}
        for row in group.itertuples():
            index = lookup.get(int(row.entry_time))
            if index is None or index + 47 >= len(candles["time"]):
                raise RuntimeError(
                    f"runner candle window missing: {row.symbol} {row.entry_time}")
            outcomes[int(row.Index)] = _runtime_runner(
                candles, index, direction=str(row.direction),
                entry=float(row.entry), sl=float(row.sl))
    if len(outcomes) != len(result):
        raise RuntimeError("runner replay did not cover every candidate")
    ordered = [outcomes[int(index)] for index in result.index]
    result["runner_outcome"] = [item[0] for item in ordered]
    result["net_r"] = [item[1] for item in ordered]
    result["exit_time"] = [item[2] for item in ordered]
    return result


def runner_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Return rows already replayed with the exact runtime exit contract."""
    required = {"runner_outcome", "net_r", "exit_time"}
    if not required.issubset(frame.columns):
        raise ValueError("attach_runtime_runner must run before evaluation")
    return frame.copy()


def rows_for(frame: pd.DataFrame, module: str) -> list[dict]:
    rows = base.standardized(runner_frame(frame), module)
    for row in rows:
        # A BE is not a loss and must reset the causal daily loss streak even
        # though it contributes zero to account return.
        row["outcome"] = "SL" if row["runner_outcome"] == "SL" else "TP"
    return rows


def ranked_runtime_gate(rows: list[dict], *, loss_pause: int,
                        hourly_cap: int) -> list[dict]:
    """Replay the intended runtime, including one ranked setup per bar."""
    pending: list[tuple[float, int, dict]] = []
    opened: dict[str, dict] = {}
    last: dict[tuple[str, str], float] = {}
    accepted_times: collections.deque[float] = collections.deque()
    output: list[dict] = []
    day = None
    streak = 0
    paused = False
    last_entry_time = None
    ordered = sorted(rows, key=lambda row: (
        float(row["entry_time"]), -float(row["model_probability"]),
        str(row["symbol"]), str(row["direction"])))
    for row in ordered:
        now = float(row["entry_time"])
        new_day = int(now // 86400)
        if new_day != day:
            day, streak, paused = new_day, 0, False
        while pending and pending[0][0] < now:
            _, _, done = heapq.heappop(pending)
            opened.pop(str(done["symbol"]), None)
            if int(float(done["exit_time"]) // 86400) == day:
                streak = streak + 1 if done["outcome"] == "SL" else 0
                if streak >= loss_pause:
                    paused = True
        while accepted_times and accepted_times[0] <= now - 3600:
            accepted_times.popleft()
        symbol = str(row["symbol"])
        direction = str(row["direction"])
        if (paused or symbol in opened or len(accepted_times) >= hourly_cap
                or now == last_entry_time):
            continue
        if now - last.get((symbol, direction), -1e20) < 10800:
            continue
        if sum(str(item["direction"]) == direction
               for item in opened.values()) >= 3:
            continue
        opened[symbol] = row
        last[(symbol, direction)] = now
        accepted_times.append(now)
        last_entry_time = now
        output.append(row)
        heapq.heappush(pending, (float(row["exit_time"]), len(output), row))
    return output


def maximum_stop_streak(rows: list[dict]) -> int:
    current = result = 0
    for row in sorted(rows, key=lambda item: (
            float(item["exit_time"]), float(item["entry_time"]),
            str(item["symbol"]))):
        current = current + 1 if row["runner_outcome"] == "SL" else 0
        result = max(result, current)
    return result


def maximum_rolling_hour(rows: list[dict]) -> int:
    times: collections.deque[float] = collections.deque()
    result = 0
    for now in sorted(float(row["entry_time"]) for row in rows):
        while times and times[0] <= now - 3600:
            times.popleft()
        times.append(now)
        result = max(result, len(times))
    return result


def evaluate(frame: pd.DataFrame, periods: tuple[str, ...], threshold: float,
             loss_pause: int, hourly_cap: int, margin: float) -> tuple[dict, list[dict]]:
    chosen = base.stage(frame, periods)
    chosen = chosen[chosen.model_probability >= threshold].copy()
    # The runtime emits one family per symbol and timestamp before account
    # cooldown/exposure rails run.  Freeze that choice here as well; otherwise
    # the replay could fall through to a lower-ranked opposite-direction family
    # after the runtime's chosen family was blocked.
    chosen = chosen.sort_values(
        ["symbol", "entry_time", "model_probability", "family", "direction"],
        ascending=[True, True, False, False, False], kind="mergesort")
    chosen = chosen.drop_duplicates(
        ["symbol", "entry_time"], keep="first")
    accepted = ranked_runtime_gate(
        rows_for(chosen, "precision_be_runner"),
        loss_pause=loss_pause, hourly_cap=hourly_cap)
    result = money(accepted, margin)
    actual_stop_streak = maximum_stop_streak(accepted)
    actual_rolling_hour = maximum_rolling_hour(accepted)
    result.update({
        "tp2_rate": round(100 * sum(row["runner_outcome"] == "TP2"
                                     for row in accepted) / len(accepted), 2),
        "non_loss_rate": round(100 * sum(float(row["net_r"]) >= 0
                                          for row in accepted) / len(accepted), 2),
        "breakeven_rate": round(100 * sum(row["runner_outcome"] == "BE"
                                           for row in accepted) / len(accepted), 2),
        # Override money()'s generic zero-PnL-as-loss diagnostic: BE resets the
        # actual runtime stop streak and is reported separately below.
        "max_loss_streak": actual_stop_streak,
        "actual_max_stop_streak": actual_stop_streak,
        "max_entries_same_timestamp": max(collections.Counter(
            int(float(row["entry_time"])) for row in accepted).values()),
        "max_entries_same_rolling_hour": actual_rolling_hour,
        "actual_max_entries_rolling_hour": actual_rolling_hour,
    })
    return result, accepted


def complete_months(result: dict, months: tuple[str, ...]) -> dict:
    values = result["monthly_return_pct"]
    if not set(months).issubset(values):
        return {}
    return {month: values[month] for month in months}


def qualifies(result: dict, months: tuple[str, ...]) -> bool:
    complete = complete_months(result, months)
    return bool(
        result["n"] >= 100 and result["negative_months"] == 0
        and result["dd_pct"] <= 10 and result["non_loss_rate"] >= 75
        and result["trades_per_calendar_day"] >= 3
        and complete and min(complete.values()) >= 30
        and result["max_entries_same_timestamp"] == 1)


def main() -> None:
    old = json.loads(prior.OLD_REPORT.read_text(encoding="utf-8"))[
        "frozen_before_2024"]
    historical_raw = precision.load_joined()
    historical = prior.canonical_history(historical_raw)
    fit = base.stage(historical, precision.FIT)
    schema = base.design(fit)
    model = HistGradientBoostingClassifier(
        learning_rate=float(old["learning_rate"]), max_iter=150,
        max_leaf_nodes=int(old["max_leaf_nodes"]),
        min_samples_leaf=int(old["min_samples_leaf"]),
        l2_regularization=1.0, early_stopping=False,
        random_state=20260908)
    model.fit(base.matrix(fit, schema), fit.outcome.eq("TP2").astype(int))
    scored = attach_runtime_runner(
        precision.scored(historical, model, schema), direct=False)
    selection_scores = base.stage(
        scored, precision.SELECT).model_probability
    candidates = []
    development_sweep = []
    for quantile in QUANTILES:
        threshold = float(selection_scores.quantile(quantile))
        for loss_pause in LOSS_PAUSES:
            for hourly_cap in HOURLY_CAPS:
                selection, _ = evaluate(
                    scored, precision.SELECT, threshold,
                    loss_pause, hourly_cap, .03)
                confirmation, _ = evaluate(
                    scored, precision.CONFIRM, threshold,
                    loss_pause, hourly_cap, .03)
                selection_complete = complete_months(
                    selection, SELECTION_MONTHS)
                confirmation_complete = complete_months(
                    confirmation, CONFIRMATION_MONTHS)
                month_values = [*selection_complete.values(),
                                *confirmation_complete.values()]
                floor = min(month_values) if month_values else -999.0
                candidate = {
                    "quantile": quantile, "threshold": threshold,
                    "loss_pause": loss_pause, "hourly_cap": hourly_cap,
                    "selection_2022h1": selection,
                    "confirmation_2022h2_2023": confirmation,
                    "complete_month_floor_pct": floor,
                    "score": round(floor - .1 * max(
                        selection["dd_pct"], confirmation["dd_pct"]), 8),
                    "qualifies": bool(
                        qualifies(selection, SELECTION_MONTHS)
                        and qualifies(confirmation, CONFIRMATION_MONTHS)),
                }
                development_sweep.append(candidate)
                if candidate["qualifies"]:
                    candidates.append(candidate)
    development_sweep.sort(key=lambda item: (
        item["score"], -item["hourly_cap"], item["threshold"]), reverse=True)
    candidates.sort(key=lambda item: (
        item["score"], -item["hourly_cap"], item["threshold"]), reverse=True)
    frozen = candidates[0] if candidates else None
    report = {
        "status": "REJECTED_DEVELOPMENT",
        "ready_for_live_orders": False,
        "protocol": {
            "model_fit": list(precision.FIT),
            "policy_selection": [*precision.SELECT, *precision.CONFIRM],
            "first_holdout": list(precision.VALIDATION),
            "later_holdout": list(precision.HIDDEN),
            "direct_reveal": "X-Perp 2026-04-16 through 2026-10-16",
            "entry": "next 15-minute market open",
            "exit": ("full position to 2R; after +0.25R move the whole stop "
                     "to breakeven; no partial order"),
            "ranking": "highest model probability; one setup per 15m timestamp",
            "money": "client 2%, 3%, or 5% current-equity margin; fixed 10x gross",
            "priority": "minimum complete-month return, then drawdown",
            "threshold_quantiles": list(QUANTILES),
            "loss_pause_candidates": list(LOSS_PAUSES),
            "hourly_cap_candidates": list(HOURLY_CAPS),
        },
        "eligible_development_candidates": len(candidates),
        "top_development_candidates": candidates[:10],
        "top_development_sweep": development_sweep[:10],
        "frozen_before_2024": frozen,
        "history_integrity": {
            "raw_rows": len(historical_raw),
            "canonical_rows": len(historical),
            "overlap_duplicates_removed": historical.attrs[
                "duplicate_rows_removed"],
        },
    }
    if frozen:
        threshold = float(frozen["threshold"])
        loss_pause = int(frozen["loss_pause"])
        hourly_cap = int(frozen["hourly_cap"])
        stage_results = {}
        accepted_by_stage = {}
        for name, periods in (
                ("validation_2024", precision.VALIDATION),
                ("later_2025_2026", precision.HIDDEN)):
            stage_results[name] = {}
            for label, margin in (("margin_2pct", .02),
                                  ("margin_3pct", .03),
                                  ("margin_5pct", .05)):
                result, accepted = evaluate(
                    scored, periods, threshold, loss_pause,
                    hourly_cap, margin)
                stage_results[name][label] = result
                if label == "margin_3pct":
                    accepted_by_stage[name] = accepted
        direct = prior.direct_frame()
        direct_scored = attach_runtime_runner(
            precision.scored(direct, model, schema), direct=True)
        fresh = {}
        direct_rows = None
        for label, margin in (("margin_2pct", .02),
                              ("margin_3pct", .03),
                              ("margin_5pct", .05)):
            result, accepted = evaluate(
                direct_scored, ("xperp",), threshold,
                loss_pause, hourly_cap, margin)
            fresh[label] = result
            if label == "margin_3pct":
                direct_rows = accepted
        direct_rows = direct_rows or []
        pd.DataFrame(direct_rows).to_pickle(ROWS_OUT)
        old_end = float(pd.read_csv(base.direct_path(2.0)).entry_time.max())
        forward_raw = direct_scored[
            direct_scored.entry_time > old_end].copy()
        forward = {}
        for label, margin in (("margin_2pct", .02),
                              ("margin_3pct", .03),
                              ("margin_5pct", .05)):
            forward[label], _ = evaluate(
                forward_raw, ("xperp",), threshold,
                loss_pause, hourly_cap, margin)
        validation_complete = complete_months(
            stage_results["validation_2024"]["margin_3pct"],
            VALIDATION_MONTHS)
        later_complete = complete_months(
            stage_results["later_2025_2026"]["margin_3pct"],
            LATER_MONTHS)
        direct_complete = complete_months(
            fresh["margin_3pct"], DIRECT_MONTHS)
        complete_by_window = {
            "validation_2024": validation_complete,
            "later_2025_2026": later_complete,
            "fresh_direct_xperp": direct_complete,
        }
        target_move = 2 * pd.to_numeric(
            pd.DataFrame(direct_rows).risk_pct, errors="coerce")
        checks = {
            "all_complete_holdout_and_direct_months_at_least_30pct_3pct_margin":
                all(bool(months) and min(months.values()) >= 30
                    for months in complete_by_window.values()),
            "no_negative_months_at_3pct": all(
                result["negative_months"] == 0 for result in (
                    stage_results["validation_2024"]["margin_3pct"],
                    stage_results["later_2025_2026"]["margin_3pct"],
                    fresh["margin_3pct"])),
            "all_reveal_drawdowns_at_most_10pct_3pct_margin": all(
                result["dd_pct"] <= 10 for result in (
                    stage_results["validation_2024"]["margin_3pct"],
                    stage_results["later_2025_2026"]["margin_3pct"],
                    fresh["margin_3pct"])),
            "direct_at_least_3_trades_per_day":
                fresh["margin_3pct"]["trades_per_calendar_day"] >= 3,
            "one_entry_per_timestamp": all(
                result["max_entries_same_timestamp"] == 1 for result in (
                    stage_results["validation_2024"]["margin_3pct"],
                    stage_results["later_2025_2026"]["margin_3pct"],
                    fresh["margin_3pct"])),
        }
        report.update({
            "status": ("PROFIT_FIRST_3PCT_SHADOW_CANDIDATE"
                       if all(checks.values())
                       else "SHADOW_CANDIDATE_OBJECTIVE_NOT_MET"),
            **stage_results,
            "fresh_direct_xperp": fresh,
            "untouched_forward_after_oct2": forward,
            "complete_holdout_and_direct_months_3pct": complete_by_window,
            "objective_checks": checks,
            "price_geometry": {
                "runner_target_price_move_quantiles_pct": {
                    str(q): round(100 * float(target_move.quantile(q)), 4)
                    for q in (0.0, .10, .25, .50, .75, .90, 1.0)
                },
                "runner_targets_at_least_0_5pct": round(
                    float((target_move >= .005).mean()), 6),
                "runner_targets_at_most_2pct": round(
                    float((target_move <= .02).mean()), 6),
            },
        })
    report["sources"] = {
        **CANDLE_SOURCES,
        str(prior.OLD_REPORT): digest(prior.OLD_REPORT),
        str(prior.FRESH_FAR): digest(prior.FRESH_FAR),
        str(prior.FRESH_NEAR): digest(prior.FRESH_NEAR),
        str(Path(__file__)): digest(Path(__file__)),
    }
    report["limitations"] = [
        "The exact raw-candle replay rejected the previously combined-label candidate.",
        "April and October are partial direct months and excluded from the complete-month gate.",
        "Breakeven exits are non-losses, not wins; TP2 hit rate and non-loss rate are reported separately.",
        "Realized-equity drawdown does not include intratrade mark-to-market excursion.",
        "The model family was previously inspected; 2024+ is transfer evidence, not pristine data.",
        "Large total returns are theoretical compounding, not a live-profit guarantee.",
        "Gross returns exclude fees at user request.",
    ]
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True),
                   encoding="utf-8")
    print(json.dumps({
        "status": report["status"], "frozen": frozen,
        "validation": report.get("validation_2024"),
        "later": report.get("later_2025_2026"),
        "fresh": report.get("fresh_direct_xperp"),
        "forward": report.get("untouched_forward_after_oct2"),
        "objective": report.get("objective_checks"),
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
