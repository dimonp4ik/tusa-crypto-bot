"""Diagnose exact material-exit expectancy by causal market context."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

import crypto_05_2r_precision_barbell_oct5_audit as legacy
import crypto_high_target_tree_oct5_audit as base
import crypto_precision_be_runner_oct18_audit as exact
import crypto_profit_first_material_barbell_oct18_audit as prior
import crypto_profit_first_material_xperp_oct20_audit as refresh


ROOT = Path("reports/audit_2026_09_08")
OUT = ROOT / "crypto_material_exact_group_diagnostics_oct20.json"
FRACTION = 0.5
SPECS = (
    ("btc_regime", "family", "direction"),
    ("family", "direction"),
    ("utc_session", "direction"),
    ("symbol", "direction"),
    ("btc_regime", "utc_session", "direction"),
)
STAGES = {
    "fit_2020_2021": legacy.FIT,
    "selection_2022h1": legacy.SELECT,
    "confirmation_2022h2_2023": legacy.CONFIRM,
    "validation_2024": legacy.VALIDATION,
    "later_2025_2026": legacy.HIDDEN,
}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def priced(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    activated = result.runner_outcome.isin(("BE", "TP2"))
    result.loc[activated, "net_r"] = (
        FRACTION * 0.5 + (1 - FRACTION)
        * result.loc[activated, "net_r"].astype(float))
    return result


def metrics(frame: pd.DataFrame) -> dict:
    if frame.empty:
        return {"n": 0, "mean_r": 0.0, "sum_r": 0.0,
                "positive_rate_pct": 0.0, "tp2_rate_pct": 0.0}
    return {
        "n": len(frame),
        "mean_r": round(float(frame.net_r.mean()), 6),
        "sum_r": round(float(frame.net_r.sum()), 4),
        "positive_rate_pct": round(100 * float(frame.net_r.gt(0).mean()), 2),
        "tp2_rate_pct": round(100 * float(frame.runner_outcome.eq("TP2").mean()), 2),
    }


def mask(frame: pd.DataFrame, spec: tuple[str, ...], values: tuple) -> pd.Series:
    result = pd.Series(True, index=frame.index)
    for field, value in zip(spec, values):
        result &= frame[field].eq(value)
    return result


def diagnose(historical: pd.DataFrame, direct: pd.DataFrame) -> dict:
    output = {}
    for spec in SPECS:
        development = base.stage(historical, (*legacy.SELECT, *legacy.CONFIRM))
        groups = sorted(set(tuple(row) for row in development[list(spec)].itertuples(
            index=False, name=None)))
        rows = []
        for values in groups:
            stages = {
                name: metrics(base.stage(historical, periods)[
                    mask(base.stage(historical, periods), spec, values)])
                for name, periods in STAGES.items()
            }
            stages["direct_xperp"] = metrics(direct[mask(direct, spec, values)])
            select = stages["selection_2022h1"]
            confirm = stages["confirmation_2022h2_2023"]
            if (select["n"] < 20 or confirm["n"] < 30
                    or select["mean_r"] <= 0 or confirm["mean_r"] <= 0):
                continue
            hidden_names = ("validation_2024", "later_2025_2026", "direct_xperp")
            rows.append({
                "values": dict(zip(spec, values)),
                "development_floor_mean_r": min(select["mean_r"], confirm["mean_r"]),
                "all_hidden_positive": all(stages[name]["mean_r"] > 0
                                           for name in hidden_names),
                "stages": stages,
            })
        rows.sort(key=lambda item: (
            item["all_hidden_positive"], item["development_floor_mean_r"]),
            reverse=True)
        output["|".join(spec)] = {
            "development_positive_groups": len(rows),
            "all_hidden_positive_groups": sum(
                row["all_hidden_positive"] for row in rows),
            "top20_development_ranked": rows[:20],
        }
    return output


def main() -> None:
    historical = priced(exact.attach_runtime_runner(
        legacy.load_joined(), direct=False, activation_r=0.5, runner_r=2.0))
    direct = prior.join_direct_outcomes(refresh.NEW_FAR, refresh.NEW_NEAR)
    direct = direct[
        direct.symbol.isin(base.ROBUST_SYMBOLS)
        & direct.risk_pct.between(legacy.RISK_PCT_MIN, legacy.RISK_PCT_MAX,
                                  inclusive="both")].copy()
    original_cache = exact.DIRECT_CACHE
    exact.DIRECT_CACHE = ROOT / "xperp_15m_oct20_material"
    try:
        direct = priced(exact.attach_runtime_runner(
            direct, direct=True, activation_r=0.5, runner_r=2.0))
    finally:
        exact.DIRECT_CACHE = original_cache
    diagnostics = diagnose(historical, direct)
    report = {
        "status": "DIAGNOSTIC_ONLY",
        "ready_for_live_orders": False,
        "protocol": {
            "selection": "positive raw mean R in both 2022h1 and 2022h2-2023",
            "holdouts": ["2024", "2025-2026", "direct X-Perp"],
            "exit": "50% at +0.5R; remaining stop to entry; runner +2R",
            "purpose": "locate regime/family/session/symbol causes, not select a deployable portfolio",
        },
        "diagnostics": diagnostics,
        "sources": {
            str(refresh.NEW_NEAR): digest(refresh.NEW_NEAR),
            str(refresh.NEW_FAR): digest(refresh.NEW_FAR),
            str(Path(__file__).resolve()): digest(Path(__file__).resolve()),
        },
    }
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({key: {
        "development_positive_groups": value["development_positive_groups"],
        "all_hidden_positive_groups": value["all_hidden_positive_groups"],
        "best": value["top20_development_ranked"][:3],
    } for key, value in diagnostics.items()}, indent=2), flush=True)


if __name__ == "__main__":
    main()
