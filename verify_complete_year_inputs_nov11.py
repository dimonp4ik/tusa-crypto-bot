"""Compare expanded-year entry inputs with the old shared core intervals."""
import json

import pandas as pd

import crypto_complete_years_nov11_audit as audit
import verify_crypto_profit_first_runtime_parity_nov3 as runtime


def main():
    report = {}
    for year in audit.ENDS:
        old = pd.read_pickle(runtime.base.cache_path(2.0, str(year)))
        start, end = old.entry_time.min(), old.entry_time.max()
        model = runtime.router.MODEL
        old = old[old.symbol.isin(audit.CORE)
            & old.risk_pct.between(model["risk_pct_min"], model["risk_pct_max"])].copy()
        old["model_probability"] = runtime.portable_probability_batch(runtime.feature_matrix(old))
        old = runtime.selected_candidates(old)
        new = pd.read_pickle(audit.ROOT / f"crypto_complete_{year}_preportfolio_nov11.pkl")
        new = new[new.symbol.isin(audit.CORE) & new.entry_time.between(start, end)]
        # The old raw exporter warms BTC context before its first exported row.
        # Compare its declared exported time span, including empty signal bars.
        keys = ["symbol", "entry_time", "direction", "family"]
        fields = ["entry", "sl", "risk_pct", "eff_ratio", "z", "model_probability"]
        left, right = [d.set_index(keys)[fields].sort_index() for d in (old, new)]
        pd.testing.assert_frame_equal(left, right, check_exact=False, atol=1e-12, rtol=0)
        report[str(year)] = dict(passes=True, matched=len(left),
            first_entry_utc=pd.Timestamp(start, unit="s", tz="UTC").isoformat(),
            last_entry_utc=pd.Timestamp(end, unit="s", tz="UTC").isoformat())
    out = audit.ROOT / "verify_complete_year_inputs_nov11.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
