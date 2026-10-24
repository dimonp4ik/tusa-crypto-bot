"""Venue residual model layered over the frozen exact target router.

The base target model and its cutoff are frozen before October.  A second
model learns which already-qualified X-Perp routes actually delivered price
movement before September.  September selects only a score cutoff and account
rails; October is evaluated once after that choice.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.ensemble import HistGradientBoostingRegressor

import crypto_high_target_tree_oct5_audit as base
import crypto_profit_first_walkforward_target_router_oct18_audit as frozen
import crypto_walkforward_target_router_oct24_forward as forward
from crypto_runtime_shadow_portfolio_oct3_audit import money
from src.research_profit_gate import economic_gate


ROOT = Path("reports/audit_2026_09_08")
SOURCE = ROOT / "crypto_walkforward_target_router_oct24_forward.json"
OUT = ROOT / "crypto_residual_meta_router_oct24_audit.json"
ROWS_OUT = ROOT / "crypto_residual_meta_router_oct24_audit.pkl"
QUANTILES = (.50, .60, .70, .75, .80, .85, .875, .90, .925, .95)
LOSS_PAUSES = (3, 4, 5, 6, 7, 8)
HOURLY_CAPS = (1, 2, 3, 4)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def as_month(frame: pd.DataFrame) -> pd.Series:
    return pd.to_datetime(frame.entry_time, unit="s", utc=True).dt.strftime(
        "%Y-%m")


def candidates(frame: pd.DataFrame, threshold: float) -> pd.DataFrame:
    return pd.DataFrame(frozen.candidate_rows(frame, threshold))


def matrix(frame: pd.DataFrame, schema: dict) -> np.ndarray:
    ordinary = base.matrix(frame, schema)
    additions = np.column_stack([
        frame.model_quality.to_numpy(float),
        frame.target_r.to_numpy(float),
        frame.risk_pct.to_numpy(float) * frame.target_r.to_numpy(float),
    ]).astype(np.float32)
    return np.column_stack([ordinary, additions])


def fit_score(train: pd.DataFrame, score: pd.DataFrame,
              variant: str) -> np.ndarray:
    schema = base.design(train)
    common = dict(
        learning_rate=.04, max_iter=150, max_leaf_nodes=7,
        min_samples_leaf=100, l2_regularization=2.0,
        early_stopping=False, random_state=20260908)
    if variant == "signed_price_regression":
        model = HistGradientBoostingRegressor(**common)
        label = train.signed_price_return.to_numpy(float)
    elif variant == "net_r_regression":
        model = HistGradientBoostingRegressor(**common)
        label = train.net_r.to_numpy(float)
    elif variant == "win_classifier":
        model = HistGradientBoostingClassifier(**common)
        label = train.net_r.gt(0).astype(int).to_numpy()
    else:
        raise ValueError(variant)
    model.fit(matrix(train, schema), label)
    if variant == "win_classifier":
        return model.predict_proba(matrix(score, schema))[:, 1]
    return model.predict(matrix(score, schema))


def gated(frame: pd.DataFrame, cutoff: float, loss_pause: int,
          hourly_cap: int) -> list[dict]:
    selected = frame[frame.meta_score >= cutoff].copy()
    selected["base_model_quality"] = selected.model_quality
    selected["model_quality"] = selected.meta_score
    selected["model_probability"] = selected.meta_score
    return frozen.be.ranked_runtime_gate(
        selected.to_dict("records"), loss_pause=loss_pause,
        hourly_cap=hourly_cap)


def metrics(rows: list[dict], margin: float = .03) -> dict:
    value = money(rows, margin)
    signed = [float(row["signed_price_return"]) for row in rows]
    value["economic_gate"] = economic_gate(
        signed, float(value["trades_per_calendar_day"]),
        margin_fraction=margin) if signed else {"passes": False}
    return value


def compact(value: dict) -> dict:
    return {name: value[name] for name in (
        "n", "return_pct", "dd_pct", "wr", "max_loss_streak",
        "trades_per_calendar_day", "negative_months")}


def main() -> None:
    source = json.loads(SOURCE.read_text(encoding="utf-8"))
    policy = source["protocol"]["policy"]
    historical = frozen.historical_wide()
    historical_fit = historical[~historical.period.eq("2026")].copy()
    old_direct = forward.direct_wide(
        frozen.prior.FRESH_NEAR, frozen.prior.FRESH_FAR,
        frozen.be.DIRECT_CACHE)
    new_direct = forward.direct_wide(
        forward.NEW_NEAR, forward.NEW_FAR, forward.NEW_CACHE)
    old_direct["month"] = as_month(old_direct)
    new_direct["month"] = as_month(new_direct)
    older = old_direct[old_direct.month < "2026-09"].copy()
    base_train = pd.concat([historical_fit, older], ignore_index=True)
    scoring = pd.concat([
        older,
        new_direct[new_direct.month.isin(("2026-09", "2026-10"))]
    ], ignore_index=True)
    routed = frozen.route(
        scoring, frozen.fit_models_on_rows(base_train, scoring))
    routed["month"] = as_month(routed)
    qualified = candidates(routed, float(policy["threshold"]))

    meta_train = qualified[qualified.month < "2026-09"].copy()
    september = qualified[qualified.month.eq("2026-09")].copy()
    october = qualified[qualified.month.eq("2026-10")].copy()
    score_both = pd.concat([september, october], ignore_index=True)

    baseline_rows = frozen.be.ranked_runtime_gate(
        october.to_dict("records"),
        loss_pause=int(policy["loss_pause"]),
        hourly_cap=int(policy["hourly_cap"]))
    baseline = metrics(baseline_rows)
    expected = source["october_to_date"]["margin_3pct"]
    for field in ("n", "return_pct", "dd_pct", "wr",
                  "trades_per_calendar_day", "negative_months"):
        if baseline[field] != expected[field]:
            raise RuntimeError(
                f"frozen October mismatch {field}: "
                f"{baseline[field]} != {expected[field]}")

    variants = {}
    frontier = []
    for variant in ("signed_price_regression", "net_r_regression",
                    "win_classifier"):
        scored = score_both.copy()
        scored["meta_score"] = fit_score(meta_train, score_both, variant)
        sep = scored[scored.month.eq("2026-09")].copy()
        variants[variant] = scored
        for quantile in QUANTILES:
            cutoff = float(sep.meta_score.quantile(quantile))
            for loss_pause in LOSS_PAUSES:
                for hourly_cap in HOURLY_CAPS:
                    rows = gated(sep, cutoff, loss_pause, hourly_cap)
                    if not rows:
                        continue
                    value = metrics(rows)
                    frontier.append({
                        "variant": variant, "quantile": quantile,
                        "cutoff": cutoff, "loss_pause": loss_pause,
                        "hourly_cap": hourly_cap,
                        "selection_metrics": compact(value),
                        "selection_score": round(
                            value["return_pct"] - .25 * value["dd_pct"], 8),
                    })

    eligible = [item for item in frontier
                if item["selection_metrics"]["n"] >= 90
                and item["selection_metrics"]["trades_per_calendar_day"] >= 3
                and item["selection_metrics"]["dd_pct"] <= 10]
    pool = eligible or frontier
    pool.sort(key=lambda item: (
        item["selection_score"], item["selection_metrics"]["return_pct"],
        -item["selection_metrics"]["dd_pct"]), reverse=True)
    selected = pool[0]
    oct_scored = variants[selected["variant"]]
    oct_scored = oct_scored[oct_scored.month.eq("2026-10")].copy()
    test_rows = gated(
        oct_scored, float(selected["cutoff"]),
        int(selected["loss_pause"]), int(selected["hourly_cap"]))
    oct3 = metrics(test_rows, .03)
    oct5 = metrics(test_rows, .05)
    pd.DataFrame(test_rows).to_pickle(ROWS_OUT)

    improved = bool(
        oct3["return_pct"] > baseline["return_pct"]
        and oct3["dd_pct"] <= baseline["dd_pct"]
        and oct3["trades_per_calendar_day"] >= 3)
    objective = bool(
        oct3["economic_gate"].get("passes", False)
        and oct3["negative_months"] == 0 and oct3["dd_pct"] <= 10)
    report = {
        "status": ("RESIDUAL_META_ROUTER_IMPROVED_OCTOBER"
                   if improved else "REJECTED_RESIDUAL_META_ROUTER"),
        "ready_for_live_orders": False,
        "order_eligible": False,
        "protocol": {
            "base_router": "frozen before October",
            "residual_fit": "X-Perp before September",
            "selection": "September 2026 only",
            "untouched_test": "October 2026 through Oct 24",
            "october_used_for_selection": False,
            "variants": 3,
            "candidate_policies": len(frontier),
        },
        "baseline_october": baseline,
        "selected_policy": selected,
        "october_test": {"margin_3pct": oct3, "margin_5pct": oct5},
        "improves_baseline_with_frequency_rail": improved,
        "objective_pass": objective,
        "selection_frontier": pool[:20],
        "sources": {
            str(SOURCE): digest(SOURCE),
            str(forward.NEW_NEAR): digest(forward.NEW_NEAR),
            str(forward.NEW_FAR): digest(forward.NEW_FAR),
            str(forward.NEW_MANIFEST): digest(forward.NEW_MANIFEST),
            str(Path(__file__).resolve()): digest(Path(__file__).resolve()),
        },
        "limitations": [
            "Only one partial October month is available for final validation.",
            "September chooses among three fixed residual objectives.",
            "Gross results exclude fees at user request.",
        ],
    }
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True),
                   encoding="utf-8")
    print(json.dumps({
        "status": report["status"],
        "baseline_october": compact(baseline),
        "selected_policy": selected,
        "october_test": {
            "margin_3pct": compact(oct3), "margin_5pct": compact(oct5)},
        "objective_pass": objective,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
