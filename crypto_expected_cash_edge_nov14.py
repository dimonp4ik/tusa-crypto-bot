"""Prototype: predict literal price return instead of an incompatible far-TP label."""
import json
import hashlib
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
import verify_crypto_profit_first_runtime_parity_nov3 as runtime
from lot_aware_tp1_audit import simulate
from asset_entry_features_research import FIELDS,attach

ROOT = Path("reports/audit_2026_09_08")


def timestamp(value):
    return pd.Timestamp(value, tz="UTC").timestamp()


def replay(frame, scores, cutoff):
    part = frame.assign(expected_price_return=scores)
    part = part[part.expected_price_return.ge(cutoff)].copy()
    part = part.sort_values(["symbol", "entry_time", "expected_price_return", "family"],
        ascending=[True, True, False, True], kind="stable").drop_duplicates(["symbol", "entry_time"])
    return runtime.runtime_gate(runtime.base.standardized(part, "expected_cash_edge_research"))


def metrics(rows, start, end):
    rows = [r for r in rows if start <= r["entry_time"] < end and r["exit_time"] < end]
    value = runtime.exact.money(rows, .03)
    months = value.get("monthly_return_pct", {})
    return {k:value[k] for k in ("n", "wr", "return_pct", "dd_pct", "trades_per_calendar_day")} | {
        "monthly_return_pct": months,
        "worst_observed_month_pct": min(months.values(), default=-100),
        "months_ge_30": sum(v >= 30 for v in months.values())}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--asset-features',action='store_true')
    args=parser.parse_args()
    path = ROOT / f"expected_cash_edge_literal_pool_{runtime.sequential.exact.LABEL_REVISION}_nov14.pkl"
    if not path.exists():
        frame = runtime.base.load_target(2.0)
        model = runtime.router.MODEL
        frame = frame[frame.risk_pct.between(model["risk_pct_min"],model["risk_pct_max"])].copy()
        paths = runtime.sequential.exact.candle_paths(tuple(frame.period.unique()))
        frame = runtime.sequential.attach_current(frame, paths)
        frame.to_pickle(path)
    frame = pd.read_pickle(path)
    frame = frame.drop_duplicates(["symbol", "entry_time", "direction", "family"])
    frame = runtime.sequential.apply_fraction(frame, .5)
    asset_sources={}
    missing_feature_rows=0
    if args.asset_features:
        frame=attach(frame,runtime.sequential.exact.candle_paths(tuple(frame.period.unique())))
        asset_sources.update(frame.attrs['asset_sources'])
        missing_feature_rows=frame.attrs['missing_feature_rows']
    def matrix(values,schema):
        base=runtime.base.matrix(values,schema)
        return np.column_stack((base,values[list(FIELDS)].to_numpy(dtype=np.float32))) if args.asset_features else base
    train = frame[frame.entry_time.ge(timestamp("2020-01-01"))
        & frame.exit_time.lt(timestamp("2022-01-01"))].copy()
    if train.empty:
        raise ValueError("no strictly historical training rows")
    schema = runtime.base.design(train)
    xfit = matrix(train, schema)
    yfit = (train.net_r * train.risk_pct).to_numpy(float)
    xall = matrix(frame, schema)
    select_start, select_end, confirm_end = map(timestamp, ("2022-01-01", "2023-01-01", "2024-01-01"))
    sel = frame.entry_time.ge(select_start) & frame.exit_time.lt(select_end)
    variants, candidates, estimators = [], {}, {}
    for leaves in (7, 15):
        estimator = HistGradientBoostingRegressor(max_leaf_nodes=leaves,
            min_samples_leaf=300, max_iter=150, learning_rate=.05,
            l2_regularization=10, early_stopping=False, random_state=20261114)
        estimator.fit(xfit, yfit)
        estimators[leaves] = estimator
        scores = estimator.predict(xall)
        for quantile in (.5, .6, .7, .8, .9, .95, .975, .99):
            cutoff = float(np.quantile(scores[sel], quantile))
            rows = replay(frame, scores, cutoff)
            selection = metrics(rows, select_start, select_end)
            confirmation = metrics(rows, select_end, confirm_end)
            key = f"leaves_{leaves}_q_{quantile}"
            candidates[key] = rows
            variants.append(dict(key=key, leaves=leaves, quantile=quantile, cutoff=cutoff,
                selection=selection, confirmation=confirmation))
    # Freeze before inspecting 2024+; retain frequency as an explicit unmet goal.
    eligible = [v for v in variants if min(v["selection"]["n"],v["confirmation"]["n"]) >= 100]
    best = max(eligible, key=lambda v: (min(v["selection"]["worst_observed_month_pct"],
        v["confirmation"]["worst_observed_month_pct"]),
        min(v["selection"]["return_pct"],v["confirmation"]["return_pct"]))) if eligible else None
    report = dict(status="RESEARCH_ONLY_NO_DEPLOYMENT", training_rows=len(train),
        label_revision=runtime.sequential.exact.LABEL_REVISION,
        target_verified=False,
        asset_features=args.asset_features,missing_feature_rows=missing_feature_rows,
        input_feature_names=[*schema['feature_names'],*(FIELDS if args.asset_features else ())],
        training_last_exit_utc=pd.Timestamp(train.exit_time.max(),unit="s",tz="UTC").isoformat(),
        first_bank_fraction=.5, variants=variants, selected=best,
        limitations=["Archived half-year windows; not full-year proof",
            "Ideal partial fills, no lot rounding in model targets",
            "Five-symbol historical universe, not current 12-symbol live universe",
            "Existing signal rails; shared-account admission still requires separate replay",
            "All historical eras have already been examined in prior research",
            "No claim of 30 percent monthly profit"])
    if best:
        rows = candidates[best["key"]]
        report["later"] = {str(y): metrics(rows, timestamp(f"{y}-01-01"),timestamp(f"{y+1}-01-01"))
                           for y in (2024, 2025, 2026)}
        direct = runtime.joiner.join_direct_outcomes(
            ROOT/"crypto_xperp_raw_router_audit_2_0_nov14_merged_raw_families.csv",
            ROOT/"crypto_xperp_raw_router_audit_0_5_nov14_merged_raw_families.csv")
        current = runtime.router.MODEL
        direct = direct[direct.symbol.isin(runtime.base.ROBUST_SYMBOLS)
            & direct.risk_pct.between(current["risk_pct_min"],current["risk_pct_max"])].copy()
        if args.asset_features:
            direct['period']='direct'
            direct=attach(direct,runtime.exact.paths(ROOT/'xperp_15m_nov14_merged'))
            asset_sources.update(direct.attrs['asset_sources'])
        scores = estimators[best["leaves"]].predict(matrix(direct,schema))
        direct = direct.assign(expected_price_return=scores, period="direct")
        direct = direct[direct.expected_price_return.ge(best["cutoff"])].copy()
        direct = direct.sort_values(["symbol","entry_time","expected_price_return","family"],
            ascending=[True,True,False,True]).drop_duplicates(["symbol","entry_time"])
        if len(direct):
            direct = runtime.sequential.apply_fraction(runtime.sequential.attach_current(
                direct,runtime.exact.paths(ROOT/"xperp_15m_nov14_merged")),.5)
            direct_rows = runtime.runtime_gate(runtime.base.standardized(direct,"expected_cash_edge_research"))
            specs = json.loads((ROOT/"xperp_specs_nov14.json").read_text())
            old = json.loads((ROOT/"xperp_15m_nov11_merged/manifest.json").read_text())
            cutoff = max(v["end"] for v in old["symbols"].values())
            report["direct_lot_rounded"] = {name:simulate(
                [r for r in direct_rows if r["entry_time"] > start],specs,tp1_pct=50)
                for name,start in (("november",timestamp("2026-11-01")),("after_nov11",cutoff))}
    report["sources"] = {str(p):hashlib.sha256(p.read_bytes()).hexdigest()
        for p in (path,Path(__file__),Path("lot_aware_tp1_audit.py"),
            ROOT/"xperp_specs_nov14.json",runtime.router.ARTIFACT_PATH,
            ROOT/"crypto_xperp_raw_router_audit_2_0_nov14_merged_raw_families.csv",
            ROOT/"crypto_xperp_raw_router_audit_0_5_nov14_merged_raw_families.csv")}
    report['sources'].update(asset_sources)
    if args.asset_features:
        p=Path('asset_entry_features_research.py')
        report['sources'][str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
    output=f"crypto_expected_cash_edge_{'asset_features_' if args.asset_features else ''}{runtime.sequential.exact.LABEL_REVISION}_nov14.json"
    (ROOT/output).write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps({k:v for k,v in report.items() if k!='variants'},indent=2),flush=True)


if __name__ == "__main__":
    main()
