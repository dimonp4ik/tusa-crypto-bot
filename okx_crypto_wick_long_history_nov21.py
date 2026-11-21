"""Check a closed signal candle veto across the existing long-history replay."""

import json
from bisect import bisect_left
from collections import Counter

from crypto_active_entry_runner_nov19_audit import activity_gate
from crypto_minimum_bank_runner_nov19_audit import CLIENTS, admitted_records, cash_result, checked_row
from crypto_native_price_pool_nov19 import MANIFEST, META, ROOT, load_pool
from okx_crypto_stop_attribution_nov21 import btc_phase, read_verified_frames
from okx_xperp_multiday_capture_nov20 import file_hash
from stock_liquid_quote_routes_nov21 import write_json


CLOSE_THRESHOLDS = (0., .10, .15, .20)


def main():
    path = ROOT / "crypto_active_entry_runner_nov19.json"
    specs_path = ROOT / "shared_xperp_specs_nov14.json"
    prior = json.loads(path.read_bytes())["results"]["be_tp1"]
    specs = json.loads(specs_path.read_bytes())
    pool, _ = load_pool()
    frames = read_verified_frames(json.loads(MANIFEST.read_bytes())["symbols"])
    btc = frames["BTCUSDT"]
    indices = {symbol: {int(ts): i for i, ts in enumerate(frame["time"])}
               for symbol, frame in frames.items()}
    candidates = []
    for row in pool[pool.is_frozen_candidate].to_dict("records"):
        frame = frames[row["symbol"]]
        i = indices[row["symbol"]][int(row["entry_time"])]
        if not activity_gate(frame, i):
            continue
        high, low, close = (frame[field][i-1] for field in ("high", "low", "close"))
        fraction = (close - low) / (high - low) if high > low else .5
        fraction = fraction if row["direction"] == "LONG" else 1 - fraction
        if not 0 <= fraction <= 1:
            raise ValueError("invalid normalized closed signal candle")
        checked = checked_row(row, frame, i, "be_tp1", .0005)
        j = bisect_left(btc["time"], int(row["entry_time"]))
        if j < 17 or btc["time"][j-1] != int(row["entry_time"]) - 900:
            raise ValueError("missing candle in closed BTC phase benchmark")
        side = 1 if row["direction"] == "LONG" else -1
        phase = btc_phase(side * (btc["close"][j-1] / btc["close"][j-17] - 1))
        candidates.append((fraction, phase, checked))
    output = {}
    policies = [(str(th), th, None) for th in CLOSE_THRESHOLDS]
    policies.extend((phase, .15, phase) for phase in ("aligned", "countertrend", "neutral"))
    for label, threshold, having_phase in policies:
        subset = [row for fraction, phase, row in candidates
                  if fraction >= threshold or (having_phase is not None and phase != having_phase)]
        admitted, unknown = admitted_records(subset)
        if unknown:
            output[label] = dict(status="UNSCORABLE_UNKNOWN_PATHS", unknown=unknown)
            continue
        cash = {client: cash_result(admitted, specs, client) for client in CLIENTS}
        if threshold == 0. and cash != prior["cash"]:
            raise ValueError("long history control changed")
        dropped = Counter(row["direction"] for fraction, phase, row in candidates
                          if fraction < threshold and (having_phase is None or phase == having_phase))
        output[label] = dict(status="SCORED_CANDLE_BASED_CASH", excluded=dropped,
                             eligible=len(subset), admitted=len(admitted), cash=cash)
    report = dict(source_hashes={str(p): file_hash(p) for p in (path, MANIFEST, META, specs_path)},
                  control_parity=True, results=output, primary_threshold=.15, live_activation_supported=False,
                  limitations=["Closed candle signal rule with July to November public price proxy.",
                               "Original candle-based trade triggers, known-close entry and a 5bp exit stress.",
                               "The October and November observations share previously analyzed history.",
                               "November incomplete. The stock bot is in PAPER for this account comparison."])
    write_json(ROOT / "okx_crypto_wick_long_history_nov21.json", report)
    for label, item in output.items():
        print(label, item["status"], item.get("excluded"), {
            setting: {month: round(result["return_pct"], 3) for month, result in v["calendar_months"].items()}
            for setting, v in item.get("cash", {}).items()
        }, flush=True)


if __name__ == "__main__":
    main()
