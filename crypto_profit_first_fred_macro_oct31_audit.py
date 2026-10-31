"""Test official FRED macro regimes on the frozen exact Profit-First route.

The model sees each daily observation only from 00:00 UTC three calendar days
after its observation date.  It is trained on 2022H1, its cutoff is selected
on 2022H2, confirmed on 2023, and only then evaluated on 2024-2026 and the
direct X-Perp replay.
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
import crypto_profit_first_exact_oct31_1700_forward as latest
import crypto_profit_first_exact_stop_classifier_oct31_audit as stop_model
from crypto_runtime_shadow_portfolio_oct3_audit import money


ROOT = Path("reports/audit_2026_09_08")
MACRO_ROOT = ROOT / "fred_macro_oct31"
MANIFEST = MACRO_ROOT / "manifest.json"
HISTORICAL = ROOT / "_tmp_exact_historical_rows.pkl"
DIRECT = latest.ROWS_OUT
OUT = ROOT / "crypto_profit_first_fred_macro_oct31.json"
ROWS_OUT = ROOT / "crypto_profit_first_fred_macro_oct31.pkl"
THRESHOLDS = tuple(float(value) for value in np.arange(.35, .751, .025))
MACRO_FEATURES = [
    "vix_level", "vix_change_5", "vix_change_20", "vix_z_60",
    "dollar_return_5", "dollar_return_20", "dollar_z_60",
    "yield_change_5", "yield_change_20", "yield_z_60",
    "macro_risk_on_score",
]
CATEGORICAL = ["symbol", "direction", "family", "utc_session", "btc_regime"]
MODELS = {
    "price_only": stop_model.NUMERIC,
    "macro_only": MACRO_FEATURES,
    "price_plus_macro": stop_model.NUMERIC + MACRO_FEATURES,
}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def macro_series(series_id: str, prefix: str, *, percent_change: bool) -> pd.DataFrame:
    path = MACRO_ROOT / f"{series_id}.csv"
    raw = pd.read_csv(path)
    dates = pd.to_datetime(raw.iloc[:, 0], errors="raise", utc=True)
    values = pd.to_numeric(raw.iloc[:, 1], errors="coerce")
    frame = pd.DataFrame({"observation_date": dates, "value": values}).dropna()
    if percent_change:
        frame[f"{prefix}_change_5"] = frame.value.pct_change(5)
        frame[f"{prefix}_change_20"] = frame.value.pct_change(20)
    else:
        frame[f"{prefix}_change_5"] = frame.value.diff(5)
        frame[f"{prefix}_change_20"] = frame.value.diff(20)
    rolling = frame.value.rolling(60, min_periods=40)
    frame[f"{prefix}_z_60"] = (
        (frame.value - rolling.mean()) / rolling.std(ddof=0))
    frame[f"{prefix}_level"] = frame.value
    # Conservative publication availability.  This also keeps Friday's broad
    # dollar observation out of the crypto stream until Monday UTC.
    # Do not divide the datetime integer representation: pandas may use
    # microseconds or nanoseconds depending on its build.  Explicit timestamps
    # keep this causal boundary invariant across supported versions.
    frame["available_at"] = (
        frame.observation_date + pd.Timedelta(days=3)
    ).map(lambda value: int(value.timestamp()))
    columns = ["available_at"] + [
        column for column in frame.columns if column.startswith(prefix + "_")]
    return frame[columns].sort_values("available_at")


def attach_macro(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.sort_values("entry_time").copy()
    sources = [
        macro_series("VIXCLS", "vix", percent_change=True).rename(columns={
            "vix_change_5": "vix_change_5",
            "vix_change_20": "vix_change_20",
        }),
        macro_series("DTWEXBGS", "dollar", percent_change=True),
        macro_series("DGS10", "yield", percent_change=False),
    ]
    # Rename percentage-return fields for clearer model documentation.
    sources[1] = sources[1].rename(columns={
        "dollar_change_5": "dollar_return_5",
        "dollar_change_20": "dollar_return_20",
    })
    for source in sources:
        result = pd.merge_asof(
            result.sort_values("entry_time"), source,
            left_on="entry_time", right_on="available_at",
            direction="backward", allow_exact_matches=True,
            suffixes=("", "_macro"))
        result = result.drop(columns=["available_at"])
    # Falling volatility, a weaker dollar and falling yields are the simple
    # risk-on votes.  The continuous model receives both votes and raw changes.
    result["macro_risk_on_score"] = (
        result.vix_change_20.lt(0).astype(int)
        + result.dollar_return_20.lt(0).astype(int)
        + result.yield_change_20.lt(0).astype(int)
    )
    return result


def pipeline(numeric: list[str]) -> Pipeline:
    transform = ColumnTransformer([
        ("numeric", Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]), numeric),
        ("categorical", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]), CATEGORICAL),
    ])
    return Pipeline([
        ("features", transform),
        ("model", LogisticRegression(
            C=.2, max_iter=2000, random_state=20260908)),
    ])


def accepted_metric(frame: pd.DataFrame, periods: tuple[str, ...],
                    module: str) -> dict:
    subset = frame[frame.period.isin(periods)]
    rows = health.accepted(subset, module)
    return money(rows, .03)


def compact(value: dict) -> dict:
    return {key: value[key] for key in (
        "n", "wr", "return_pct", "dd_pct", "max_loss_streak",
        "negative_months", "trades_per_calendar_day", "monthly_return_pct")}


def development_ok(candidate: dict, baseline: dict) -> bool:
    return bool(
        candidate["return_pct"] >= baseline["return_pct"]
        and candidate["dd_pct"] <= baseline["dd_pct"]
        and candidate["negative_months"] <= baseline["negative_months"]
        and candidate["max_loss_streak"] <= baseline["max_loss_streak"] + 1
        and candidate["trades_per_calendar_day"] >= 2.5)


def hidden_ok(candidate: dict, baseline: dict) -> bool:
    return bool(
        candidate["return_pct"] >= baseline["return_pct"]
        and candidate["dd_pct"] <= baseline["dd_pct"] + .5
        and candidate["negative_months"] <= baseline["negative_months"]
        and candidate["max_loss_streak"] <= baseline["max_loss_streak"] + 1)


def direct_metrics(frame: pd.DataFrame) -> dict:
    october = frame[frame.entry_time.ge(pd.Timestamp(
        "2026-10-01", tz="UTC").timestamp())]
    return {
        "all_direct": latest.metrics(frame, .03),
        "october": latest.metrics(october, .03),
    }


def main() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    for series in manifest["series"].values():
        path = Path(series["path"])
        if digest(path) != series["sha256"]:
            raise ValueError(f"macro source hash mismatch: {path}")

    historical = attach_macro(health.current_outcomes(pd.read_pickle(HISTORICAL)))
    direct = attach_macro(pd.read_pickle(DIRECT))
    missing = {feature: int(historical[feature].isna().sum())
               for feature in MACRO_FEATURES}
    training = pd.DataFrame(health.accepted(
        historical[historical.period.eq("2022h1")], "fred_macro_training"))
    baseline = {
        "training_2022h1": accepted_metric(
            historical, ("2022h1",), "fred_macro_baseline"),
        "selection_2022h2": accepted_metric(
            historical, ("2022h2",), "fred_macro_baseline"),
        "confirmation_2023": accepted_metric(
            historical, ("2023",), "fred_macro_baseline"),
        "hidden_2024": accepted_metric(
            historical, ("2024",), "fred_macro_baseline"),
        "hidden_2025": accepted_metric(
            historical, ("2025",), "fred_macro_baseline"),
        "hidden_2026": accepted_metric(
            historical, ("2026",), "fred_macro_baseline"),
    }
    baseline_direct = direct_metrics(direct)
    frontier = []
    selected_by_model = []
    scored_frames = {}
    for name, numeric in MODELS.items():
        model = pipeline(numeric)
        model.fit(training[numeric + CATEGORICAL],
                  training.net_r.gt(0).astype(int))
        scored = historical.copy()
        scored["macro_survival_probability"] = model.predict_proba(
            scored[numeric + CATEGORICAL])[:, 1]
        direct_scored = direct.copy()
        direct_scored["macro_survival_probability"] = model.predict_proba(
            direct_scored[numeric + CATEGORICAL])[:, 1]
        scored_frames[name] = (scored, direct_scored)
        eligible = []
        for threshold in THRESHOLDS:
            candidate = scored[scored.macro_survival_probability.ge(threshold)]
            selection = accepted_metric(
                candidate, ("2022h2",), f"fred_macro_{name}")
            confirmation = accepted_metric(
                candidate, ("2023",), f"fred_macro_{name}")
            selection_pass = development_ok(
                selection, baseline["selection_2022h2"])
            confirmation_pass = development_ok(
                confirmation, baseline["confirmation_2023"])
            item = {
                "model": name, "threshold": round(threshold, 3),
                "selection_2022h2": compact(selection),
                "confirmation_2023": compact(confirmation),
                "selection_pass": selection_pass,
                "confirmation_pass": confirmation_pass,
            }
            frontier.append(item)
            if selection_pass and confirmation_pass:
                eligible.append(item)
        eligible.sort(key=lambda item: (
            min(item["selection_2022h2"]["return_pct"]
                / max(baseline["selection_2022h2"]["return_pct"], 1e-9),
                item["confirmation_2023"]["return_pct"]
                / max(baseline["confirmation_2023"]["return_pct"], 1e-9)),
            -max(item["selection_2022h2"]["dd_pct"],
                 item["confirmation_2023"]["dd_pct"])), reverse=True)
        if eligible:
            selected_by_model.append(eligible[0])

    selected_by_model.sort(key=lambda item: (
        min(item["selection_2022h2"]["return_pct"]
            / max(baseline["selection_2022h2"]["return_pct"], 1e-9),
            item["confirmation_2023"]["return_pct"]
            / max(baseline["confirmation_2023"]["return_pct"], 1e-9)),
        -max(item["selection_2022h2"]["dd_pct"],
             item["confirmation_2023"]["dd_pct"])), reverse=True)
    selected = selected_by_model[0] if selected_by_model else None
    hidden = candidate_direct = None
    robust = False
    selected_rows = None
    if selected:
        scored, direct_scored = scored_frames[selected["model"]]
        threshold = float(selected["threshold"])
        candidate = scored[scored.macro_survival_probability.ge(threshold)]
        selected_rows = direct_scored[
            direct_scored.macro_survival_probability.ge(threshold)].copy()
        hidden = {
            period: accepted_metric(candidate, (year,), "fred_macro_hidden")
            for period, year in (("hidden_2024", "2024"),
                                 ("hidden_2025", "2025"),
                                 ("hidden_2026", "2026"))
        }
        candidate_direct = direct_metrics(selected_rows)
        robust = bool(
            all(hidden_ok(hidden[key], baseline[key]) for key in hidden)
            and hidden_ok(candidate_direct["all_direct"],
                          baseline_direct["all_direct"])
            and hidden_ok(candidate_direct["october"],
                          baseline_direct["october"]))
    if robust and selected_rows is not None:
        selected_rows.to_pickle(ROWS_OUT)

    report = {
        "status": ("FRED_MACRO_SHADOW_CHALLENGER" if robust
                   else "REJECTED_FRED_MACRO_FILTER"),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "ready_for_live_orders": False,
        "orders_possible": False,
        "runtime_change_allowed": False,
        "protocol": {
            "training": "2022H1 exact sequential outcomes",
            "threshold_selection": "2022H2 only",
            "confirmation": "2023",
            "sealed_holdouts": ["2024", "2025", "2026", "direct X-Perp"],
            "macro_availability": manifest["availability_rule"],
            "models": {key: value + CATEGORICAL for key, value in MODELS.items()},
            "threshold_grid": list(THRESHOLDS),
            "money": "3% current-equity margin, fixed 10x gross",
            "orders": "market only",
        },
        "macro_missing_rows": missing,
        "baseline": {key: compact(value) for key, value in baseline.items()},
        "baseline_direct": baseline_direct,
        "selected_before_holdouts": selected,
        "hidden": ({key: compact(value) for key, value in hidden.items()}
                   if hidden else None),
        "direct_xperp": candidate_direct,
        "robust_improvement": robust,
        "target_30pct_each_month_proven": False,
        "frontier": frontier,
        "sources": {
            str(MANIFEST): digest(MANIFEST),
            str(HISTORICAL): digest(HISTORICAL),
            str(DIRECT): digest(DIRECT),
            str(Path(__file__).resolve()): digest(Path(__file__).resolve()),
            **{item["path"]: item["sha256"]
               for item in manifest["series"].values()},
        },
        "limitations": [
            "FRED current-vintage history may include later data revisions.",
            "The three-day availability delay is deliberately conservative.",
            "Filtering the already-gated direct replay does not refill freed slots.",
            "Gross returns exclude fees at user request.",
        ],
    }
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({
        "status": report["status"],
        "selected": selected,
        "baseline_hidden": {key: compact(baseline[key]) for key in
                            ("hidden_2024", "hidden_2025", "hidden_2026")},
        "hidden": report["hidden"],
        "baseline_direct": baseline_direct,
        "direct": candidate_direct,
        "macro_missing_rows": missing,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
