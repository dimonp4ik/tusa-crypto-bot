"""Check a pre-entry Bitcoin reversal context for shallow crypto stops."""

from bisect import bisect_left
from collections import Counter
from pathlib import Path
import pickle

from crypto_active_entry_runner_nov19_audit import activity_gate
from crypto_minimum_bank_runner_nov19_audit import checked_row
from crypto_native_price_pool_nov19 import MANIFEST, RAW, load_pool
from crypto_nearer_target_long_history_nov22 import EXIT_STRESS, checked_cash
from crypto_risk_range_long_history_nov22 import CUTOFF, build_alternates
from okx_unbankable_tp1_nov21 import read_json
from okx_xperp_multiday_replay_nov20 import ROOT, file_hash
from stock_liquid_quote_routes_nov21 import write_json


BTC_HORIZONS = (4, 16)


def bitcoin_returns(frame, at, direction):
    i = bisect_left(frame["time"], at)
    lookback = max(BTC_HORIZONS)
    stamps = frame["time"][i-lookback-1:i]
    if list(stamps) != list(range(at - (lookback + 1) * 900, at, 900)):
        raise ValueError("Bitcoin context is incomplete at candidate entry")
    sign = {"LONG": 1, "SHORT": -1}[direction]
    close = frame["close"]
    return tuple(sign * (close[i-1] / close[i-1-n] - 1) for n in BTC_HORIZONS)


def reversal(frame, at, direction):
    return max(bitcoin_returns(frame, at, direction)) < 0


def main():
    specs_path = ROOT / "shared_xperp_specs_nov14.json"
    prior = ROOT / "crypto_risk_range_long_history_nov22.json"
    exact_path = ROOT / "okx_crypto_risk_range_nov22.json"
    baseline_reference = read_json(prior)["cases"]
    specs = read_json(specs_path)
    pool, metadata = load_pool()
    candles = {}
    for symbol, item in read_json(MANIFEST)["symbols"].items():
        path = Path(item["path"])
        if file_hash(path) != item["sha256"]:
            raise ValueError(f"native candle digest changed for {symbol}")
        frame = pickle.loads(path.read_bytes())
        candles[symbol] = (frame, {int(t): i for i, t in enumerate(frame["time"])})
    btc = candles["BTCUSDT"][0]
    candidates = pool[pool.is_frozen_candidate].to_dict("records")
    claimed = {(r["symbol"], r["entry_time"]) for r in candidates}
    base = []
    for candidate in candidates:
        at = candidate["entry_time"]
        if at >= CUTOFF:
            continue
        frame, lookup = candles[candidate["symbol"]]
        i = lookup[int(at)]
        if activity_gate(frame, i):
            base.append(checked_row(candidate, frame, i, "be_tp1", EXIT_STRESS))
    alternate, _, _ = build_alternates(candles, claimed)
    narrow = alternate["below_1pct"]
    countertrend = [r for r in narrow if reversal(btc, r["entry_time"], r["direction"])]
    groups = dict(control=base, with_below_1pct=base+narrow, with_narrow_reversal=base+countertrend)
    results = {name: checked_cash(rows, specs) for name, rows in groups.items()}
    if any(results[k] != baseline_reference[k] for k in ("control", "with_below_1pct")):
        raise ValueError("earlier cash controls do not match")
    candidates_by_phase = Counter(r["btc_regime"] for r in countertrend)
    write_json(ROOT / "crypto_narrow_reversal_nov22.json", dict(
        source_hashes={p.name: file_hash(p) for p in (specs_path, prior, exact_path, MANIFEST, RAW)},
        candle_pool_sha256=metadata["sha256"], cutoff_exclusive=CUTOFF, cases=results,
        rule="0.8–1.0% stop. Both last-closed BTC 1h and 4h returns oppose candidate direction.",
        extrapolated_family_proposals=len(narrow), reversal_proposals=len(countertrend),
        reversal_candidates_by_phase=candidates_by_phase, live_activation_supported=False,
        limitations=["The filter was motivated by two profitable hours in the newer public-book sample.",
                     "July–October entries and causal phase measurements precede and do not overlap those hours.",
                     "Uses the original candle-based delayed stop and lot-rounded cash proxies.",
                     "A favorable result would need fresh entry and exit depth for execution validation."]))
    for label, result in results.items():
        print(label, result["status"], {c: (v["summary"]["trades"], v["summary"]["return_pct"],
              v["summary"]["cash_drawdown_pct"], {m: round(x["return_pct"], 2)
              for m, x in v["calendar_months"].items()}) for c, v in result.get("clients", {}).items()}, flush=True)


if __name__ == "__main__":
    main()
