"""New X-Perp tail with client-sized, lot-rounded TP1 cash flows."""
import hashlib
import json
from pathlib import Path

import pandas as pd

from crypto_frozen_nov11_forward import ROOT, routed_rows
from lot_aware_tp1_audit import simulate


def main():
    cache = ROOT / "xperp_15m_nov14_merged"
    rows = routed_rows(cache, "nov14_merged")
    row_path = ROOT / "crypto_frozen_nov14_baseline.pkl"
    rows.to_pickle(row_path)
    old_path = ROOT / "crypto_frozen_nov11_baseline.pkl"
    old = pd.read_pickle(old_path)
    old_manifest = ROOT / "xperp_15m_nov11_merged/manifest.json"
    cutoff = min(v["end"] for v in json.loads(old_manifest.read_text())["symbols"].values())
    keys = ["symbol", "entry_time", "direction", "family"]
    fields = ["entry", "sl", "net_r", "exit_time"]
    left, right = [d[d.entry_time.le(cutoff-48*900)].set_index(keys)[fields].sort_index()
                   for d in (old, rows)]
    pd.testing.assert_frame_equal(left, right, check_exact=False, atol=1e-12, rtol=0)
    spec_path = ROOT / "xperp_specs_nov14.json"
    specs = json.loads(spec_path.read_text())
    manifest = json.loads((cache/"manifest.json").read_text())
    for symbol in rows.symbol.unique():
        if specs[symbol]["instId"] != manifest["symbols"][symbol]["instrument"]:
            raise ValueError("contract specification belongs to a different instrument")
    report = dict(status="CLIENT_LOT_ROUNDING_RESEARCH_ONLY", live_changes=False,
        starting_deposit=125.64, margin_fraction=.03, leverage=10,
        historical_prefix_parity=dict(passes=True, matched=len(left)),
        limitations=["Isolated crypto cohort, not shared-account PnL or forecast",
            "Starting balance is a scenario, not reconstructed historical account equity",
            "Uses current contract specifications throughout the short cohort",
            "OHLC trigger fills, no order-book fill reconstruction",
            "Sizing uses realized cash; open PnL excluded",
            "Partial and runner cash flows use recorded candle event times",
            "Cohort starts flat; no positions inherited from before its cutoff",
            "At most four crypto slots; no replacement candidates for post-route rejected slots",
            "Last 48 bars of entries are not exported until outcomes mature"],
        results={})
    for name, start in (("november", pd.Timestamp("2026-11-01", tz="UTC").timestamp()),
                        ("new_after_nov11", cutoff)):
        part = rows[rows.entry_time.gt(start)]
        report["results"][name] = {str(pct): simulate(part.to_dict("records"), specs, tp1_pct=pct)
                                   for pct in (5, 25, 50, 75)}
    report["sources"] = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (
        row_path, old_path, spec_path, old_manifest, cache/"manifest.json", Path(__file__),
        Path("lot_aware_tp1_audit.py"))}
    out = ROOT / "crypto_lot_aware_nov14_forward.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    for name, scenarios in report["results"].items():
        print(name, json.dumps({pct: {k:v for k,v in result.items() if k != "ledger"}
                               for pct,result in scenarios.items()}, indent=2))


if __name__ == "__main__":
    main()
