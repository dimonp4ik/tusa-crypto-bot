"""Evaluate the frozen router on complete calendar years from verified archives."""
import hashlib
import json
import pickle
from pathlib import Path

import pandas as pd

from append_candle_snapshot import merge
from crypto_walk_forward_family_model import FAMILIES, enrich, attach_btc_context
from entry_hypotheses import replay
import verify_crypto_profit_first_runtime_parity_nov3 as runtime

ROOT = Path("reports/audit_2026_09_08")
CORE = {"BTCUSDT", "ETHUSDT", "LINKUSDT", "XLMUSDT", "XRPUSDT"}
ENDS = {2023: (1675123200000, 1690761600000, 1706659200000),
        2024: (1706659200000, 1722384000000, 1738281600000),
        2025: (1738281600000, 1753920000000, 1769817600000)}


def compact(frame):
    value = runtime.exact.metrics(frame, .03)
    return {k: value[k] for k in ("n", "wr", "return_pct", "dd_pct",
        "max_loss_streak", "outcome_counts", "mean_signed_price_return_pct")}


def main():
    report = dict(status="BUILDING_COMPLETE_CALENDAR_YEAR_RESEARCH",
        margin_fraction=.03, leverage=10, model_refit=False, threshold_retuned=False,
        warning="Archived public non-X-Perp candles are a transfer stress test. Symbols without full coverage are excluded explicitly. Prior research has inspected these eras. Gross closed-trade equity; no monthly promise.",
        years={}, sources={})
    for year, anchors in ENDS.items():
        start = int(pd.Timestamp(f"{year}-01-01", tz="UTC").timestamp())
        end = int(pd.Timestamp(f"{year+1}-01-01", tz="UTC").timestamp())
        candles, exclusions, overlap = {}, {}, {}
        for symbol in sorted(runtime.base.ROBUST_SYMBOLS):
            paths = [Path("backtest_cache") / f"{symbol}_15min_18000_end{a}.pkl" for a in anchors]
            if any(not p.exists() for p in paths):
                exclusions[symbol] = "missing archive window"
                continue
            chunks = []
            for p in paths:
                raw = p.read_bytes()
                report["sources"][str(p)] = hashlib.sha256(raw).hexdigest()
                chunk = pickle.loads(raw)
                if any(b-a != 900 for a,b in zip(chunk["time"],chunk["time"][1:])):
                    raise ValueError(f"archive time gap: {p}")
                chunks.append(chunk)
            merged, first = merge(chunks[0], chunks[1])
            merged, second = merge(merged, chunks[2])
            if merged["time"][0] > start - 385*900 or merged["time"][-1] < end + 48*900:
                exclusions[symbol] = "incomplete year or indicator/outcome margins"
                continue
            indices = [i for i,t in enumerate(merged["time"]) if start - 385*900 <= t <= end + 48*900]
            candles[symbol] = {k: [v[i] for i in indices] for k,v in merged.items()}
            overlap[symbol] = first["overlap_bars"] + second["overlap_bars"]
        if "BTCUSDT" not in candles:
            raise ValueError("missing full-year BTC context")
        parts = []
        for family in FAMILIES:
            records = []
            for symbol, candle in candles.items():
                records.extend(replay(candle, symbol, family, 2.0,
                                      max_cost_fraction=None, roundtrip_cost=0.0))
            part = attach_btc_context(enrich(pd.DataFrame(records), family), candles["BTCUSDT"])
            parts.append(part)
        raw = pd.concat(parts, ignore_index=True).dropna(subset=["btc_regime"])
        model = runtime.router.MODEL
        raw = raw[raw.entry_time.ge(start) & raw.entry_time.lt(end)
            & raw.risk_pct.between(model["risk_pct_min"], model["risk_pct_max"])].copy()
        raw["model_probability"] = runtime.portable_probability_batch(runtime.feature_matrix(raw))
        selected = runtime.selected_candidates(raw).assign(period=str(year))
        labels = []
        for row in selected.to_dict("records"):
            c = candles[row["symbol"]]
            index = (int(row["entry_time"]) - c["time"][0]) // 900
            if c["time"][index] != int(row["entry_time"]):
                raise ValueError("entry timestamp does not match candle index")
            outcome, runner_r, banked, exit_time = runtime.sequential.exact.replay_runner(
                c, index, direction=row["direction"], entry=row["entry"], stop=row["sl"],
                activation_r=.5, lock_r=0)
            labels.append(dict(row, sequential_outcome=outcome,
                sequential_runner_r=runner_r, sequential_banked=banked,
                sequential_exit_time=exit_time))
        selected = runtime.sequential.apply_fraction(pd.DataFrame(labels), .05)
        selected.to_pickle(ROOT / f"crypto_complete_{year}_preportfolio_nov11.pkl")
        expanded = pd.DataFrame(runtime.runtime_gate(runtime.base.standardized(selected, "complete_year_frozen")))
        expanded.to_pickle(ROOT / f"crypto_complete_{year}_expanded_nov11.pkl")
        core = selected[selected.symbol.isin(CORE)].copy()
        if set(candles) & CORE != CORE:
            raise ValueError("missing original historical core symbol")
        rows = runtime.runtime_gate(runtime.base.standardized(core, "complete_year_frozen"))
        frame = pd.DataFrame(rows)
        frame.to_pickle(ROOT / f"crypto_complete_{year}_nov11.pkl")
        value = compact(frame)
        monthly = runtime.exact.metrics(frame, .03)["monthly_return_pct"]
        report["years"][str(year)] = dict(symbols=sorted(candles), excluded=exclusions,
            matched_overlap_bars=overlap, raw_candidates=len(raw), preportfolio=len(selected),
            metrics=value, monthly_return_pct=monthly,
            evaluated_core=sorted(CORE),
            expanded_universe_metrics=compact(expanded),
            first_entry_utc=pd.Timestamp(frame.entry_time.min(),unit="s",tz="UTC").isoformat(),
            last_entry_utc=pd.Timestamp(frame.entry_time.max(),unit="s",tz="UTC").isoformat(),
            note="Each annual cohort starts flat on January 1; no previous-year carry-in. Monthly returns are attributed by close time; January of the next year may contain carried exits. Core matches the original five historical symbols; expanded universe is reported separately.")
        out = ROOT / "crypto_complete_years_nov11.json"
        out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps({"year": year, **report["years"][str(year)]}), flush=True)
    report["status"] = "COMPLETE_CALENDAR_YEAR_RESEARCH_NOT_LIVE_FILLS"
    report["sources"][str(Path(__file__))] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    report["sources"][str(runtime.router.ARTIFACT_PATH)] = hashlib.sha256(runtime.router.ARTIFACT_PATH.read_bytes()).hexdigest()
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
