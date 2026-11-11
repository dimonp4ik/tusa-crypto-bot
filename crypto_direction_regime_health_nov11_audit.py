"""Chronological direction-aware health study; November cannot select a rule."""
import hashlib
import itertools
import json
from pathlib import Path

import pandas as pd

import crypto_profit_first_tp1_volume_next_open_nov1_audit as policy
import verify_crypto_profit_first_runtime_parity_nov3 as runtime
from shadow_regime_health import filter_rows

ROOT = Path("reports/audit_2026_09_08")
SCOPES = (("direction", "btc_regime"), ("direction", "family"),
          ("direction", "family", "btc_regime"))


def metrics(rows):
    value = policy.money(rows, .03)
    result = {k: value[k] for k in ("n", "wr", "return_pct", "dd_pct",
        "max_loss_streak", "negative_months", "trades_per_calendar_day")}
    if rows:
        result.update(first_entry_utc=pd.Timestamp(min(r["entry_time"] for r in rows), unit="s", tz="UTC").isoformat(),
            last_entry_utc=pd.Timestamp(max(r["entry_time"] for r in rows), unit="s", tz="UTC").isoformat())
    return result


def periods(rows, names):
    return [r for r in rows if r["period"] in names]


def accepted(rows):
    return runtime.runtime_gate(rows)


def safe(candidate, baseline):
    return (candidate["n"] >= 100
        and candidate["return_pct"] >= baseline["return_pct"]
        and candidate["dd_pct"] <= baseline["dd_pct"]
        and candidate["negative_months"] <= baseline["negative_months"])


def main():
    raw = runtime.base.load_target(2.0)
    model = runtime.router.MODEL
    raw = raw[raw.period.isin(("2022h1", "2022h2", "2023", "2024", "2025", "2026"))
        & raw.risk_pct.between(model["risk_pct_min"], model["risk_pct_max"])].copy()
    scores = runtime.portable_probability_batch(runtime.feature_matrix(raw))
    raw["model_probability"] = scores
    raw = runtime.selected_candidates(raw)
    paths = runtime.sequential.exact.candle_paths(tuple(raw.period.unique()))
    raw = runtime.sequential.apply_fraction(runtime.sequential.attach_current(raw, paths), .05)
    preportfolio_path = ROOT / "crypto_health_preportfolio_nov11.pkl"
    raw.to_pickle(preportfolio_path)
    rows = runtime.base.standardized(raw, "direction_regime_health_research")
    baseline_rows = accepted(rows)
    select, confirm = ("2022h1",), ("2022h2", "2023")
    base_select = metrics(periods(baseline_rows, select))
    base_confirm = metrics(periods(baseline_rows, confirm))
    variants = []
    for fields, days, minimum in itertools.product(SCOPES, (30, 60, 90), (10, 20, 30)):
        filtered, counts = filter_rows(rows, fields=fields, days=days, minimum=minimum)
        routed = accepted(filtered)
        selection = metrics(periods(routed, select))
        confirmation = metrics(periods(routed, confirm))
        eligible = safe(selection, base_select) and safe(confirmation, base_confirm)
        # Prefer drawdown reduction that survives both development intervals.
        score = min(base_select["dd_pct"] - selection["dd_pct"],
                    base_confirm["dd_pct"] - confirmation["dd_pct"])
        variants.append(dict(fields=fields, days=days, minimum=minimum,
            counts=counts, selection=selection, confirmation=confirmation,
            eligible=eligible, score=score))
    eligible = [v for v in variants if v["eligible"] and v["score"] > 0]
    selected = max(eligible, key=lambda v: (v["score"], v["confirmation"]["return_pct"])) if eligible else None
    report = dict(status="RESEARCH_REJECTED" if selected is None else "RESEARCH_HOLDOUT_REQUIRED",
        orders_possible=False, margin_fraction=.03, leverage=10,
        baseline_selection=base_select, baseline_confirmation=base_confirm,
        selection_periods=select, confirmation_periods=confirm,
        historical_input="Rebuilt from raw family candidates; portfolio replacements replayed after veto",
        selection_requirements="At least 100 development trades per interval; return at least baseline, drawdown and negative months no worse. Frequency is reported separately, not treated as proof of success.",
        selected=selected, variants=variants,
        warning="Past intervals have been used in earlier research; chronological separation is not a pristine holdout. Closed-trade gross equity only; no monthly extrapolation.")
    if selected:
        params = {k: selected[k] for k in ("fields", "days", "minimum")}
        filtered, _ = filter_rows(rows, **params)
        routed = accepted(filtered)
        report["later_periods"] = {p: {"baseline": metrics(periods(baseline_rows, (p,))),
            "candidate": metrics(periods(routed, (p,)))} for p in ("2024", "2025", "2026")}
        cache = ROOT / "xperp_15m_nov11_merged"
        joined = runtime.joiner.join_direct_outcomes(
            ROOT / "crypto_xperp_raw_router_audit_2_0_nov11_merged_raw_families.csv",
            ROOT / "crypto_xperp_raw_router_audit_0_5_nov11_merged_raw_families.csv")
        model = runtime.router.MODEL
        direct = joined[joined.symbol.isin(runtime.base.ROBUST_SYMBOLS)
            & joined.risk_pct.between(model["risk_pct_min"], model["risk_pct_max"])].copy()
        direct["model_probability"] = runtime.portable_probability_batch(runtime.feature_matrix(direct))
        direct = runtime.selected_candidates(direct).assign(period="direct")
        direct = runtime.sequential.apply_fraction(runtime.sequential.attach_current(
            direct, runtime.exact.paths(cache)), .05)
        direct_rows = runtime.base.standardized(direct, "direction_regime_health_research")
        filtered, _ = filter_rows(direct_rows, **params)
        base_direct, candidate_direct = accepted(direct_rows), accepted(filtered)
        cutoff = 1794049200
        report["direct_new_after_nov7"] = {
            name: metrics([r for r in values if r["entry_time"] > cutoff])
            for name, values in (("baseline", base_direct), ("candidate", candidate_direct))}
        identity = lambda r: (r["symbol"], r["entry_time"], r["direction"], r["family"])
        old_ids, new_ids = set(map(identity, base_direct)), set(map(identity, candidate_direct))
        keep = ("symbol", "direction", "family", "btc_regime", "entry_time",
                "exit_time", "net_r", "signed_price_return")
        report["fresh_removed_trades"] = [{k: r[k] for k in keep} for r in base_direct
            if r["entry_time"] > cutoff and identity(r) not in new_ids]
        report["fresh_added_by_portfolio_replacement"] = [{k: r[k] for k in keep} for r in candidate_direct
            if r["entry_time"] > cutoff and identity(r) not in old_ids]
        later_pass = all(safe(v["candidate"], v["baseline"])
                         for v in report["later_periods"].values())
        fresh = report["direct_new_after_nov7"]
        fresh_pass = (fresh["candidate"]["return_pct"] >= fresh["baseline"]["return_pct"]
                      and fresh["candidate"]["dd_pct"] <= fresh["baseline"]["dd_pct"])
        report.update(later_periods_pass=later_pass, fresh_pass=fresh_pass,
            status="RESEARCH_REJECTED_ON_LATER_DATA" if not later_pass or not fresh_pass
            else "RESEARCH_ONLY_NOT_APPROVED_FOR_LIVE")
    report["sources"] = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (
        preportfolio_path, Path(__file__), Path("shadow_regime_health.py"),
        runtime.router.ARTIFACT_PATH)}
    out = ROOT / "crypto_direction_regime_health_nov11.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k not in ("variants", "sources")}, indent=2))


if __name__ == "__main__":
    main()
