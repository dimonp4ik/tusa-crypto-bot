"""Evaluate frozen crypto rules on appended Nov-7--11 candles; no retuning."""
import json
import hashlib
from pathlib import Path

import pandas as pd

import crypto_profit_first_exact_oct31_1700_forward as exact
import crypto_profit_first_tp1_volume_next_open_nov1_audit as policy
import verify_crypto_profit_first_runtime_parity_nov3 as runtime
import shared_account_dual_bot_nov1_audit as shared

ROOT = Path("reports/audit_2026_09_08")


def compact(rows, margin):
    value = exact.metrics(rows, margin)
    return {k: value[k] for k in ("n", "wr", "return_pct", "dd_pct",
        "max_loss_streak", "outcome_counts", "mean_signed_price_return_pct")}


def routed_rows(cache, tag):
    raw = runtime.joiner.join_direct_outcomes(
        ROOT / f"crypto_xperp_raw_router_audit_2_0_{tag}_raw_families.csv",
        ROOT / f"crypto_xperp_raw_router_audit_0_5_{tag}_raw_families.csv")
    model = runtime.router.MODEL
    eligible = raw[raw.symbol.isin(runtime.base.ROBUST_SYMBOLS)
        & raw.risk_pct.between(float(model["risk_pct_min"]),
                              float(model["risk_pct_max"]))].copy()
    eligible["model_probability"] = runtime.portable_probability_batch(
        runtime.feature_matrix(eligible))
    rows = runtime.routed(runtime.selected_candidates(eligible), cache=cache)
    return policy.attach_next_open(rows, exact.paths(cache), direct=True)


def main():
    cache = ROOT / "xperp_15m_nov11_merged"
    manifest = cache / "manifest.json"
    old = ROOT / "xperp_15m_nov7_tp1/manifest.json"
    cutoff = max(v["end"] for v in json.loads(old.read_text())["symbols"].values())
    rows = routed_rows(cache, "nov11_merged")
    old_rows = routed_rows(old.parent, "nov7_tp1")
    # Both raw family exporters require a full 48-bar resolution horizon.
    # Compare only that mature prefix; tail outcomes may legitimately resolve.
    mature_end = cutoff - 48 * 900
    keys = ["symbol", "entry_time", "direction", "family"]
    fields = ["net_r", "exit_time", "model_probability", "runner_outcome"]
    left = old_rows[old_rows.entry_time.le(mature_end)].set_index(keys)[fields].sort_index()
    right = rows[rows.entry_time.le(mature_end)].set_index(keys)[fields].sort_index()
    pd.testing.assert_frame_equal(left, right, check_exact=False, atol=1e-12, rtol=0)
    frozen = json.loads(policy.OUT.read_text())["selected_before_holdouts"]
    baseline = policy.apply_policy(rows, volume_threshold=0,
                                  extra_close_fraction=0, direct=True)
    candidate = policy.apply_policy(rows, volume_threshold=frozen["volume_ratio_below"],
        extra_close_fraction=frozen["extra_close_fraction_at_next_open"], direct=True)
    report = {"status": "FROZEN_FORWARD_DIAGNOSTIC_NOT_LIVE_FILLS",
        "start_exclusive_utc": pd.Timestamp(cutoff, unit="s", tz="UTC").isoformat(),
        "snapshot_last_bar_utc": pd.Timestamp(max(v["end"] for v in
            json.loads(manifest.read_text())["symbols"].values()), unit="s", tz="UTC").isoformat(),
        "warning": "Short new sample; no monthly extrapolation. Not account PnL. dd_pct is closed-trade equity drawdown and excludes unrealized swings. Gross results exclude costs.",
        "raw_entry_horizon_bars": 48,
        "last_exported_entry_utc": pd.Timestamp(rows.entry_time.max(), unit="s", tz="UTC").isoformat(),
        "historical_prefix_parity": {"passes": True, "matched_rows": len(left),
            "through_entry_utc": pd.Timestamp(mature_end, unit="s", tz="UTC").isoformat()},
        "sources": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (
            old, manifest, policy.OUT, Path("src/frozen_crypto_profit_first_material_2r.json"),
            Path(__file__))},
        "frozen_tp1_parameters": {k: frozen[k] for k in (
            "volume_ratio_below", "extra_close_fraction_at_next_open")}}
    for name, frame in (("baseline", baseline), ("tp1_research", candidate)):
        frame.to_pickle(ROOT / f"crypto_frozen_nov11_{name}.pkl")
        fresh = frame[frame.entry_time.gt(cutoff)].copy()
        report[name] = {f"margin_{int(m*100)}pct": compact(fresh, m) for m in (.03, .05)}
        report[name]["trades"] = fresh[["symbol", "direction", "entry_time", "exit_time",
            "runner_outcome", "net_r", "signed_price_return", "model_probability",
            "tp1_extra_close_fraction"]].to_dict("records")
    shadow = baseline.assign(bot="crypto", signed_return=baseline.signed_price_return)
    health = pd.DataFrame(shared.filter_crypto_shorts(shadow.to_dict("records")))
    quality = baseline[baseline.model_probability.ge(.3356824161327173)]
    report["previously_frozen_guard_diagnostics"] = {
        "scope": "Post-signal veto only; no newly freed-slot replacements; not approval for deployment",
        "quality_threshold": .3356824161327173,
        "short_health_14d_min15": compact(health[health.entry_time.gt(cutoff)], .03),
        "quality_shadow_veto": compact(quality[quality.entry_time.gt(cutoff)], .03)}
    out = ROOT / "crypto_frozen_nov11_forward.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
