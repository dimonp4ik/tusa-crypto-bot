"""Select a material crypto route using exact 0.5R-to-BE/2R outcomes.

The model fit ends in 2021, policy selection ends in 2023, and 2024 onward
remains hidden until the policy is frozen.  Every label is replayed from one
OHLC path: the original stop remains active through the +0.5R activation bar,
then the remaining position moves to entry from the next candle.
"""
from __future__ import annotations

import collections
import copy
import hashlib
import json
from pathlib import Path

import pandas as pd

import crypto_05_2r_precision_barbell_oct5_audit as legacy
import crypto_exact_be_expected_value_oct19_audit as expected
import crypto_high_target_tree_oct5_audit as base
import crypto_precision_be_runner_oct18_audit as exact
import crypto_profit_first_material_barbell_oct18_audit as prior
import crypto_profit_first_material_xperp_oct20_audit as refresh
from crypto_runtime_shadow_portfolio_oct3_audit import money


ROOT = Path("reports/audit_2026_09_08")
OUT = ROOT / "crypto_material_exact_value_oct20.json"
ROWS_OUT = ROOT / "crypto_material_exact_value_xperp_oct20.pkl"
FRESH_OUT = ROOT / "crypto_material_exact_value_xperp_oct20_fresh.pkl"
ACTIVATION_R = 0.5
RUNNER_R = 2.0
FRACTIONS = (0.05, 0.10, 0.20, 0.30, 0.50, 0.70)
QUANTILES = (0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.875, 0.90, 0.925, 0.95)
LOSS_PAUSES = (3, 4, 5)
HOURLY_CAPS = (1, 2)
TARGETS = ("exact_r", "tp2", "non_loss")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def with_fraction(frame: pd.DataFrame, fraction: float) -> pd.DataFrame:
    result = frame.copy()
    result["runner_net_r"] = result.net_r.astype(float)
    activated = result.runner_outcome.isin(("BE", "TP2"))
    result.loc[activated, "net_r"] = (
        fraction * ACTIVATION_R
        + (1.0 - fraction) * result.loc[activated, "runner_net_r"])
    result["near_fraction"] = fraction
    return result


def gated_rows(frame: pd.DataFrame, periods: tuple[str, ...], threshold: float,
               loss_pause: int, hourly_cap: int) -> list[dict]:
    chosen = base.stage(frame, periods)
    chosen = chosen[chosen.model_probability >= threshold].copy()
    chosen = chosen.sort_values(
        ["symbol", "entry_time", "model_probability", "family", "direction"],
        ascending=[True, True, False, False, False], kind="mergesort")
    chosen = chosen.drop_duplicates(["symbol", "entry_time"], keep="first")
    # Any positive fraction gives BE rows a positive outcome.  The accepted
    # entries are therefore identical across the fraction grid; only their
    # account PnL changes.  Gate once and reprice below.
    chosen = with_fraction(chosen, FRACTIONS[0])
    return exact.ranked_runtime_gate(
        exact.rows_for(chosen, "material_exact_value"),
        loss_pause=loss_pause, hourly_cap=hourly_cap)


def reprice_rows(rows: list[dict], fraction: float) -> list[dict]:
    priced = copy.deepcopy(rows)
    for row in priced:
        runner = float(row["runner_net_r"])
        if str(row["runner_outcome"]) in ("BE", "TP2"):
            row["net_r"] = fraction * ACTIVATION_R + (1 - fraction) * runner
        else:
            row["net_r"] = runner
        row["signed_price_return"] = float(row["risk_pct"]) * float(row["net_r"])
        row["outcome"] = "SL" if float(row["net_r"]) < 0 else "TP"
        row["near_fraction"] = fraction
    return priced


def metrics_for(rows: list[dict], fraction: float, margin: float) -> tuple[dict, list[dict]]:
    rows = reprice_rows(rows, fraction)
    value = money(rows, margin)
    value.update({
        "tp2_rate_pct": round(100 * sum(
            row["runner_outcome"] == "TP2" for row in rows) / len(rows), 2)
        if rows else 0.0,
        "breakeven_rate_pct": round(100 * sum(
            row["runner_outcome"] == "BE" for row in rows) / len(rows), 2)
        if rows else 0.0,
        "non_loss_rate_pct": round(100 * sum(
            float(row["runner_net_r"]) >= 0 for row in rows) / len(rows), 2)
        if rows else 0.0,
        "actual_max_stop_streak": exact.maximum_stop_streak(rows),
        "max_entries_same_timestamp": max(collections.Counter(
            int(float(row["entry_time"])) for row in rows).values())
        if rows else 0,
    })
    return value, rows


def evaluate(frame: pd.DataFrame, periods: tuple[str, ...], threshold: float,
             fraction: float, loss_pause: int, hourly_cap: int,
             margin: float) -> tuple[dict, list[dict]]:
    return metrics_for(gated_rows(
        frame, periods, threshold, loss_pause, hourly_cap), fraction, margin)


def month_floor(value: dict, months: tuple[str, ...]) -> float:
    observed = value.get("monthly_return_pct", {})
    return (min(float(observed[month]) for month in months)
            if set(months).issubset(observed) else -999.0)


def development_ok(value: dict) -> bool:
    return bool(value["n"] >= 100 and value["dd_pct"] <= 10
                and value["trades_per_calendar_day"] >= 3
                and value["negative_months"] == 0)


def score_key(item: dict) -> tuple:
    return (
        item["admissible"], item["development_floor_pct"],
        -max(item["selection"]["dd_pct"], item["confirmation"]["dd_pct"]),
        item["selection"]["return_pct"] + item["confirmation"]["return_pct"],
    )


def main() -> None:
    historical = legacy.load_joined()
    historical_exact = exact.attach_runtime_runner(
        historical, direct=False, activation_r=ACTIVATION_R,
        runner_r=RUNNER_R)
    fit = base.stage(historical_exact, legacy.FIT)
    design = base.design(fit)
    candidates = []
    model_cache = {}
    scored_cache = {}
    for target_name in TARGETS:
        for config in expected.CONFIGS:
            cache_key = (target_name, config)
            model = expected.fit_model(fit, design, target_name, config)
            scored = expected.score(
                historical_exact, model, design, target_name)
            model_cache[cache_key] = model
            scored_cache[cache_key] = scored
            selection_scores = base.stage(
                scored, legacy.SELECT).model_probability
            for quantile in QUANTILES:
                threshold = float(selection_scores.quantile(quantile))
                for loss_pause in LOSS_PAUSES:
                    for hourly_cap in HOURLY_CAPS:
                        selection_rows = gated_rows(
                            scored, legacy.SELECT, threshold,
                            loss_pause, hourly_cap)
                        confirmation_rows = gated_rows(
                            scored, legacy.CONFIRM, threshold,
                            loss_pause, hourly_cap)
                        for fraction in FRACTIONS:
                            selection, _ = metrics_for(
                                selection_rows, fraction, 0.03)
                            confirmation, _ = metrics_for(
                                confirmation_rows, fraction, 0.03)
                            floor = min(
                                month_floor(selection, exact.SELECTION_MONTHS),
                                month_floor(confirmation, exact.CONFIRMATION_MONTHS))
                            candidates.append({
                                "target": target_name,
                                "max_leaf_nodes": config[0],
                                "min_samples_leaf": config[1],
                                "l2_regularization": config[2],
                                "quantile": quantile,
                                "threshold": threshold,
                                "near_fraction": fraction,
                                "loss_pause": loss_pause,
                                "hourly_cap": hourly_cap,
                                "selection": selection,
                                "confirmation": confirmation,
                                "development_floor_pct": round(floor, 4),
                                "admissible": (development_ok(selection)
                                               and development_ok(confirmation)),
                                "_cache_key": cache_key,
                            })
    candidates.sort(key=score_key, reverse=True)
    selected = candidates[0]
    scored = scored_cache[selected["_cache_key"]]
    threshold = float(selected["threshold"])
    fraction = float(selected["near_fraction"])
    loss_pause = int(selected["loss_pause"])
    hourly_cap = int(selected["hourly_cap"])

    holdouts = {}
    for margin in (0.03, 0.05):
        name = f"margin_{round(100 * margin)}pct"
        holdouts[name] = {
            "validation_2024": evaluate(
                scored, legacy.VALIDATION, threshold, fraction,
                loss_pause, hourly_cap, margin)[0],
            "later_2025_2026": evaluate(
                scored, legacy.HIDDEN, threshold, fraction,
                loss_pause, hourly_cap, margin)[0],
        }

    old_direct = prior.join_direct_outcomes(refresh.OLD_FAR, refresh.OLD_NEAR)
    direct = prior.join_direct_outcomes(refresh.NEW_FAR, refresh.NEW_NEAR)
    cutoff = float(old_direct.entry_time.max())
    direct = direct[
        direct.symbol.isin(base.ROBUST_SYMBOLS)
        & direct.risk_pct.between(legacy.RISK_PCT_MIN, legacy.RISK_PCT_MAX,
                                  inclusive="both")].copy()
    direct["period"] = "xperp"
    original_cache = exact.DIRECT_CACHE
    exact.DIRECT_CACHE = ROOT / "xperp_15m_oct20_material"
    try:
        direct_exact = exact.attach_runtime_runner(
            direct, direct=True, activation_r=ACTIVATION_R,
            runner_r=RUNNER_R)
    finally:
        exact.DIRECT_CACHE = original_cache
    direct_scored = expected.score(
        direct_exact, model_cache[selected["_cache_key"]], design,
        str(selected["target"]))
    direct_values = {}
    fresh_values = {}
    chosen_rows = []
    fresh_rows = []
    for margin in (0.03, 0.05):
        name = f"margin_{round(100 * margin)}pct"
        direct_values[name], rows = evaluate(
            direct_scored, ("xperp",), threshold, fraction,
            loss_pause, hourly_cap, margin)
        fresh = direct_scored[direct_scored.entry_time > cutoff].copy()
        fresh_values[name], selected_fresh = evaluate(
            fresh, ("xperp",), threshold, fraction,
            loss_pause, hourly_cap, margin)
        if margin == 0.03:
            chosen_rows, fresh_rows = rows, selected_fresh
    pd.DataFrame(chosen_rows).to_pickle(ROWS_OUT)
    pd.DataFrame(fresh_rows).to_pickle(FRESH_OUT)

    public = {key: value for key, value in selected.items()
              if not key.startswith("_")}
    direct3 = direct_values["margin_3pct"]
    objective = bool(
        selected["admissible"] and selected["development_floor_pct"] >= 30
        and all(part["negative_months"] == 0
                for margin in holdouts.values() for part in margin.values())
        and direct3["negative_months"] == 0
        and direct3["trades_per_calendar_day"] >= 3)
    report = {
        "status": ("MATERIAL_EXACT_VALUE_SHADOW_CANDIDATE" if objective
                   else "REJECTED_OBJECTIVE"),
        "ready_for_live_orders": False,
        "protocol": {
            "fit": list(legacy.FIT),
            "selection": list(legacy.SELECT),
            "confirmation": list(legacy.CONFIRM),
            "holdouts": [*legacy.VALIDATION, *legacy.HIDDEN, "direct X-Perp"],
            "entry": "next 15-minute market open",
            "exit": "partial at +0.5R; remaining stop to entry next bar; runner +2R",
            "collision": "original stop first on activation bar; breakeven stop first afterwards",
            "price_move": "+0.5R is 0.5%-1%; +2R is 2%-4%",
            "money": "client 3% or 5% current-equity margin; fixed 10x",
        },
        "candidate_count": len(candidates),
        "admissible_count": sum(item["admissible"] for item in candidates),
        "selected_before_holdouts": public,
        "top20_before_holdouts": [
            {key: value for key, value in item.items() if not key.startswith("_")}
            for item in candidates[:20]],
        "holdouts": holdouts,
        "direct_xperp": direct_values,
        "fresh_after_oct17_cutoff": fresh_values,
        "objective_pass": objective,
        "sources": {
            **exact.CANDLE_SOURCES,
            str(refresh.NEW_NEAR): digest(refresh.NEW_NEAR),
            str(refresh.NEW_FAR): digest(refresh.NEW_FAR),
            str(Path(__file__).resolve()): digest(Path(__file__).resolve()),
        },
        "limitations": [
            "Gross results exclude fees at user request.",
            "Historical named periods are available six-month windows.",
            "Direct X-Perp history is shorter than a full market cycle.",
            "No backtest guarantees a future monthly return.",
        ],
    }
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({
        "status": report["status"],
        "candidate_count": report["candidate_count"],
        "admissible_count": report["admissible_count"],
        "selected": public,
        "holdouts": holdouts,
        "direct_xperp": direct_values,
        "fresh": fresh_values,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
