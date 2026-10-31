"""Causal stop filter trained on exact sequential trade outcomes.

The original entry score predicts whether price reaches the near target.  That
is not the same as surviving the exact TP1 -> next-bar BE -> 2R path.  This
audit trains one regularized logistic model on 2022H1 exact outcomes, selects
its admission threshold on 2022H2-2023, then opens 2024, 2025-2026 and the
current X-Perp replay only after the threshold is frozen.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import crypto_profit_first_exact_health_router_oct30_audit as health
import crypto_profit_first_exact_oct30_2030_forward as latest
import crypto_profit_first_material_target_oct18_audit as frozen
from crypto_runtime_shadow_portfolio_oct3_audit import money


ROOT = Path("reports/audit_2026_09_08")
HISTORICAL = ROOT / "_tmp_exact_historical_rows.pkl"
DIRECT = ROOT / "crypto_profit_first_exact_oct30_2030.pkl"
OUT = ROOT / "crypto_profit_first_exact_stop_classifier_oct31.json"
ROWS_OUT = ROOT / "crypto_profit_first_exact_stop_classifier_oct31.pkl"
NUMERIC = [
    "risk_pct", "eff_ratio", "z", "abs_z", "hour_sin", "hour_cos",
    "dir_z", "btc_fast_slow_atr", "btc_eff20", "btc_return20_atr",
    "btc_z20", "btc_atr_pct", "btc_slow_long_atr", "btc_return96_atr",
    "btc_eff96", "btc_trend_dir_atr", "btc_return_dir_atr",
    "model_probability",
]
CATEGORICAL = ["symbol", "direction", "family", "utc_session", "btc_regime"]
THRESHOLDS = tuple(float(value) for value in np.arange(.35, .751, .025))
DIRECT_CUTOFF = int(pd.Timestamp(json.loads(
    latest.SOURCE_REPORT.read_text(encoding="utf-8"))
    ["fresh_direct_coverage"]["last_entry"]).timestamp())


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def model_pipeline() -> Pipeline:
    preprocessing = ColumnTransformer([
        ("numeric", Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]), NUMERIC),
        ("categorical", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]), CATEGORICAL),
    ])
    return Pipeline([
        ("features", preprocessing),
        ("model", LogisticRegression(
            C=.2, max_iter=2000, random_state=20260908)),
    ])


def accepted(frame: pd.DataFrame, module: str) -> pd.DataFrame:
    return pd.DataFrame(health.accepted(frame, module))


def metric(frame: pd.DataFrame, periods: tuple[str, ...]) -> dict:
    subset = accepted(frame[frame.period.isin(periods)], "exact_stop_classifier")
    return money(subset.to_dict("records"), .03)


def direct_metric(frame: pd.DataFrame) -> dict:
    october = frame[frame.entry_time.ge(pd.Timestamp(
        "2026-10-01", tz="UTC").timestamp())]
    unseen = frame[frame.entry_time.gt(DIRECT_CUTOFF)]
    return {
        "october": latest.metrics(october, .03),
        "after_oct16": latest.metrics(unseen, .03),
        "all_direct": latest.metrics(frame, .03),
    }


def compact(value: dict) -> dict:
    keys = ("n", "wr", "return_pct", "dd_pct", "max_loss_streak",
            "negative_months", "trades_per_calendar_day",
            "monthly_return_pct")
    return {key: value[key] for key in keys}


def main() -> None:
    historical = health.current_outcomes(pd.read_pickle(HISTORICAL))
    training = accepted(
        historical[historical.period.eq("2022h1")], "exact_stop_training")
    model = model_pipeline()
    model.fit(training[NUMERIC + CATEGORICAL], training.net_r.gt(0).astype(int))
    historical = historical.copy()
    historical["exact_survival_probability"] = model.predict_proba(
        historical[NUMERIC + CATEGORICAL])[:, 1]

    baseline = {
        "selection_2022h1": metric(historical, frozen.SELECT),
        "confirmation_2022h2_2023": metric(historical, frozen.CONFIRM),
        "validation_2024": metric(historical, frozen.VALIDATION),
        "later_2025_2026": metric(historical, frozen.LATER),
    }
    frontier = []
    for threshold in THRESHOLDS:
        candidate = historical[
            historical.exact_survival_probability.ge(threshold)]
        selection = metric(candidate, frozen.SELECT)
        confirmation = metric(candidate, frozen.CONFIRM)
        eligible = bool(
            selection["return_pct"] >= baseline["selection_2022h1"]["return_pct"]
            and selection["dd_pct"] <= baseline["selection_2022h1"]["dd_pct"]
            and selection["negative_months"] <= baseline["selection_2022h1"]["negative_months"] + 1
            and confirmation["return_pct"] >= baseline["confirmation_2022h2_2023"]["return_pct"]
            and confirmation["dd_pct"] <= 1.01 * baseline["confirmation_2022h2_2023"]["dd_pct"]
            and confirmation["trades_per_calendar_day"] >= 2.5)
        frontier.append({
            "threshold": round(threshold, 3), "eligible": eligible,
            "selection_2022h1": compact(selection),
            "confirmation_2022h2_2023": compact(confirmation),
        })
    eligible = [item for item in frontier if item["eligible"]]
    eligible.sort(key=lambda item: (
        item["confirmation_2022h2_2023"]["return_pct"],
        -item["confirmation_2022h2_2023"]["dd_pct"]), reverse=True)
    selected = eligible[0] if eligible else None

    direct = pd.read_pickle(DIRECT).copy()
    direct["exact_survival_probability"] = model.predict_proba(
        direct[NUMERIC + CATEGORICAL])[:, 1]
    base_direct = direct_metric(direct)
    holdouts = candidate_direct = None
    robust = False
    selected_rows = None
    if selected:
        threshold = float(selected["threshold"])
        filtered = historical[
            historical.exact_survival_probability.ge(threshold)]
        holdouts = {
            "validation_2024": metric(filtered, frozen.VALIDATION),
            "later_2025_2026": metric(filtered, frozen.LATER),
        }
        selected_rows = direct[
            direct.exact_survival_probability.ge(threshold)].copy()
        candidate_direct = direct_metric(selected_rows)
        robust = bool(
            holdouts["validation_2024"]["return_pct"] >= baseline["validation_2024"]["return_pct"]
            and holdouts["validation_2024"]["dd_pct"] <= baseline["validation_2024"]["dd_pct"]
            and holdouts["later_2025_2026"]["return_pct"] >= baseline["later_2025_2026"]["return_pct"]
            and holdouts["later_2025_2026"]["dd_pct"] <= baseline["later_2025_2026"]["dd_pct"]
            and candidate_direct["october"]["return_pct"] >= base_direct["october"]["return_pct"]
            and candidate_direct["after_oct16"]["return_pct"] >= base_direct["after_oct16"]["return_pct"]
            and candidate_direct["all_direct"]["return_pct"] >= base_direct["all_direct"]["return_pct"]
            and candidate_direct["all_direct"]["dd_pct"] <= base_direct["all_direct"]["dd_pct"]
            and candidate_direct["all_direct"]["negative_months"] <= base_direct["all_direct"]["negative_months"]
            and candidate_direct["october"]["trades_per_calendar_day"] >= 3)
    if robust and selected_rows is not None:
        selected_rows.to_pickle(ROWS_OUT)

    report = {
        "status": ("EXACT_STOP_FILTER_SHADOW_CHALLENGER" if robust
                   else "REJECTED_EXACT_STOP_FILTER"),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "ready_for_live_orders": False, "orders_possible": False,
        "runtime_change_allowed": False,
        "protocol": {
            "training": "2022H1 exact sequential outcomes",
            "threshold_selection": "2022H2-2023 only",
            "sealed_holdouts": ["2024", "2025-2026", "direct X-Perp through Oct 30"],
            "model": "L2 logistic regression C=0.2",
            "features": NUMERIC + CATEGORICAL,
            "threshold_grid": list(THRESHOLDS),
            "position_fraction": "client setting; measured at 3% margin and 10x",
            "orders": "market only",
        },
        "baseline": {key: compact(value) for key, value in baseline.items()},
        "baseline_direct": base_direct,
        "selected_before_holdouts": selected,
        "holdouts": ({key: compact(value) for key, value in holdouts.items()}
                     if holdouts else None),
        "direct_xperp": candidate_direct,
        "robust_improvement": robust,
        "target_30pct_each_month_proven": False,
        "frontier": frontier,
        "sources": {str(path): digest(path) for path in (
            HISTORICAL, DIRECT, health.OUT, latest.OUT,
            Path(__file__).resolve())},
        "limitations": [
            "The improvement is small and must survive new append-only data before runtime use.",
            "Historical exact results still contain losing months and do not prove 30% monthly.",
            "Gross results exclude fees at user request.",
        ],
    }
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({
        "status": report["status"], "selected": selected,
        "baseline_holdouts": report["baseline"], "holdouts": report["holdouts"],
        "baseline_direct": base_direct, "direct": candidate_direct,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
