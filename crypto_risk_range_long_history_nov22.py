"""Check the same stop bands on four earlier months of native contract candles."""

from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import pickle
from unittest.mock import patch

import pandas as pd

from crypto_active_entry_runner_nov19_audit import activity_gate
from crypto_minimum_bank_runner_nov19_audit import checked_row
from crypto_native_price_pool_nov19 import MANIFEST, RAW, load_pool
from crypto_nearer_target_long_history_nov22 import EXIT_STRESS, checked_cash
from okx_crypto_risk_range_nov22 import EXTENSIONS, HIGH, LOW
from okx_unbankable_tp1_nov21 import read_json
from okx_xperp_multiday_replay_nov20 import ROOT, file_hash
from src.crypto_profit_first_material_shadow_router import (
    MODEL, REQUIRED_BTC_HISTORY_BARS, REQUIRED_SYMBOL_HISTORY_BARS, shadow_setups,
)
from stock_liquid_quote_routes_nov21 import write_json


CUTOFF = int(datetime(2026, 11, 1, tzinfo=timezone.utc).timestamp())


def prefix(frame, index, count, stamp):
    if index < count or frame["time"][index - 1] != stamp - 900:
        return None
    return {k: values[index-count:index] for k, values in frame.items()}


def build_alternates(candles, claimed):
    raw = pd.read_csv(RAW, usecols=["symbol", "entry_time", "entry", "sl"])
    btc, btc_index = candles["BTCUSDT"]
    buckets = {name: [] for name in EXTENSIONS}
    declared = Counter()
    active = Counter()
    recent_btc = {}
    universe = raw[(raw.entry_time < CUTOFF) & raw.symbol.isin(candles)]
    for (symbol, at), rows in universe.groupby(["symbol", "entry_time"], sort=False):
        if (symbol, at) in claimed:
            continue
        frame, times = candles[symbol]
        i = times[at]
        if not i:
            continue
        entry = float(frame["close"][i-1])
        widths = abs(rows.entry - rows.sl) / entry
        possible = [name for name, (lo, hi) in EXTENSIONS.items() if ((widths >= lo) & (widths <= hi)).any()]
        if not possible:
            continue
        past = prefix(frame, i, REQUIRED_SYMBOL_HISTORY_BARS, at)
        if at not in recent_btc:
            b = btc_index.get(at)
            recent_btc[at] = prefix(btc, b, REQUIRED_BTC_HISTORY_BARS, at) if b is not None else None
        context = recent_btc[at]
        if past is None or context is None:
            declared["missing_causal_prefix"] += 1
            continue
        args = dict(entry_time=at, market_price=entry, model_price=entry, symbol=symbol)
        if shadow_setups(past, context, **args):
            declared["occupied_by_control_at_known_close"] += 1
            continue
        for name in possible:
            low, high = EXTENSIONS[name]
            with patch.dict(MODEL, risk_pct_min=low, risk_pct_max=high):
                selected = shadow_setups(past, context, **args)
            for setup in selected:
                risk = setup["risk_pct"]
                if LOW <= risk <= HIGH:
                    continue
                declared[name] += 1
                if not activity_gate(frame, i):
                    continue
                active[name] += 1
                proposal = dict(setup, symbol=symbol, entry_time=int(at), exit_time=int(at)+48*900,
                                net_r=0., sequential_outcome="UNKNOWN")
                buckets[name].append(checked_row(proposal, frame, i, "be_tp1", EXIT_STRESS))
    return buckets, declared, active


def main():
    specifications = ROOT / "shared_xperp_specs_nov14.json"
    prior = ROOT / "crypto_active_entry_runner_nov19.json"
    specs = read_json(specifications)
    pool, meta = load_pool()
    candidates = pool[pool.is_frozen_candidate].to_dict("records")
    candles = {}
    for symbol, item in read_json(MANIFEST)["symbols"].items():
        path = Path(item["path"])
        if file_hash(path) != item["sha256"]:
            raise ValueError(f"historical candle digest changed for {symbol}")
        frame = pickle.loads(path.read_bytes())
        candles[symbol] = (frame, {int(t): i for i, t in enumerate(frame["time"])})
    base = []
    for row in candidates:
        frame, lookup = candles[row["symbol"]]
        i = lookup[int(row["entry_time"])]
        if activity_gate(frame, i):
            base.append(checked_row(row, frame, i, "be_tp1", EXIT_STRESS))
    if checked_cash(base, specs)["clients"] != read_json(prior)["results"]["be_tp1"]["cash"]:
        raise ValueError("published long-run baseline changed")
    claimed = {(r["symbol"], r["entry_time"]) for r in candidates}
    added, proposed, active = build_alternates(candles, claimed)
    pre_november = [r for r in base if r["entry_time"] < CUTOFF]
    cases = dict(control=pre_november, **{f"with_{k}": pre_november + v for k, v in added.items()},
                 with_both=pre_november + [r for region in added.values() for r in region])
    results = {name: checked_cash(rows, specs) for name, rows in cases.items()}
    for name, case in results.items():
        out = {k: (v["summary"]["return_pct"], v["summary"]["cash_drawdown_pct"], v["summary"]["trades"],
                   {m: round(x["return_pct"], 2) for m, x in v["calendar_months"].items()})
               for k, v in case.get("clients", {}).items()}
        print(name, case["status"], out or case["admitted_unknown"][:4], flush=True)
    phase_counts = {name: Counter(r["btc_regime"] for r in rows) for name, rows in added.items()}
    write_json(ROOT / "crypto_risk_range_long_history_nov22.json", dict(
        cutoff_exclusive=CUTOFF, control_parity=True, widths=EXTENSIONS, counts=proposed,
        activity_qualified=active, phases=phase_counts, cases=results,
        pool_hash=meta["sha256"], source_hashes={p.name: file_hash(p) for p in (RAW, MANIFEST, specifications, prior)},
        live_activation_supported=False, limitations=[
            "Native 15-minute candles and conservative exit prices, no archived order-book fills in this study.",
            "The last confirmed close is the entry proxy, even when the last candle has no trades.",
            "The previously tested positive-volume pre-entry rule and ledger are applied to both control and alternatives.",
            "The fixed classifier is outside its calibrated stop-width range for every added candidate.",
            "Unknown admitted paths make that case unscorable; paper-gated stocks contribute no active exposure here.",
        ]))


if __name__ == "__main__":
    main()
