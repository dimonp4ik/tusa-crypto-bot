"""Fixed-cohort exit-fraction sensitivity, not a live policy selection."""
import hashlib
import json
from pathlib import Path
import pandas as pd
from crypto_complete_years_nov11_audit import compact, ROOT


def main():
    paths = {str(y): ROOT / f"crypto_complete_{y}_nov11.pkl" for y in (2023, 2024, 2025)}
    paths["xperp_after_nov7"] = ROOT / "crypto_frozen_nov11_baseline.pkl"
    results = {}
    for name, path in paths.items():
        rows = pd.read_pickle(path)
        if name == "xperp_after_nov7":
            rows = rows[rows.entry_time.gt(1794049200)].copy()
        results[name] = {}
        for fraction in (0, .05, .25, .5, .75):
            candidate = rows.copy()
            candidate["net_r"] = candidate.sequential_runner_r.where(
                ~candidate.sequential_banked.astype(bool),
                fraction*.5 + (1-fraction)*candidate.sequential_runner_r)
            candidate["signed_price_return"] = candidate.net_r * candidate.risk_pct
            value = compact(candidate)
            # The legacy summary calls non-winning streaks loss streaks;
            # do not label a run of zero-return exits as consecutive losses.
            value.pop("max_loss_streak", None)
            value["flat_exits"] = int(candidate.net_r.abs().le(1e-12).sum())
            value["negative_exits"] = int(candidate.net_r.lt(-1e-12).sum())
            results[name][str(fraction)] = value
    report = dict(results=results,
        warning="Ideal fractions, not fills. Lot rounding, partial-exit cash timing and portfolio replacement are not simulated. Same entry cohorts and runner exits. No live policy selected.",
        sources={str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                 for p in (*paths.values(), Path(__file__))})
    (ROOT / "tp1_fixed_fractions_nov14.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
