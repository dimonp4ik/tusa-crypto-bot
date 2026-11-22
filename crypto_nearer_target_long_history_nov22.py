"""Check the fixed 1.5R hypothesis on the older native contract candle cohort."""

from datetime import datetime, timezone
import math
from pathlib import Path
import pickle

from crypto_active_entry_runner_nov19_audit import activity_gate
from crypto_minimum_bank_runner_nov19_audit import CLIENTS, admitted_records, cash_result, checked_row
from crypto_native_price_pool_nov19 import MANIFEST, load_pool
from okx_unbankable_tp1_nov21 import read_json
from okx_xperp_multiday_replay_nov20 import ROOT, file_hash
from stock_liquid_quote_routes_nov21 import write_json


TARGET_R = 1.5
EXIT_STRESS = .0005
STOP_CODES = {"SL", "LOCK", "TP1_STOP"}


def shorten(control, candles, i, *, target_px=None):
    if not control["known"]:
        return control
    entry = control["entry"]
    sign = 1 if control["direction"] == "LONG" else -1
    target = entry + TARGET_R * (entry - control["sl"]) if target_px is None else target_px
    for j in range(i, min(i + 48, len(candles["time"]))):
        close_time = int(candles["time"][j]) + 900
        if close_time > control["exit_time"]:
            break
        if candles["volume"][j] <= 0:
            continue
        opening_hit = j > i and sign * (candles["open"][j] - target) >= 0
        adverse_collision = (close_time == control["exit_time"]
                             and control["sequential_outcome"] in STOP_CODES)
        extreme = candles["high" if sign > 0 else "low"][j]
        if not opening_hit and (adverse_collision or sign * (extreme - target) < 0):
            continue
        bank_time = control["near_exit_time"]
        bank_time = min(bank_time, close_time) if math.isfinite(bank_time) else close_time
        runner_return = sign * (target * (1 - sign * EXIT_STRESS) / entry - 1.)
        bank_return = control["bank_price_return"]
        return dict(control, exit_time=close_time, near_exit_time=bank_time,
                    sequential_outcome="TARGET", sequential_banked=True, runner_price_return=runner_return,
                    net_r=(.05 * bank_return + .95 * runner_return) / control["risk_pct"])
    return control


def checked_cash(rows, specs):
    admitted, unknown = admitted_records(rows)
    if unknown:
        return dict(status="UNSCORABLE", admitted_unknown=unknown)
    return dict(status="SCORED_CANDLE_PROXY", clients={c: cash_result(admitted, specs, c) for c in CLIENTS})


def main():
    prior = ROOT / "crypto_active_entry_runner_nov19.json"
    specifications = ROOT / "shared_xperp_specs_nov14.json"
    reference = read_json(prior)["results"]["be_tp1"]["cash"]
    specs = read_json(specifications)
    pool, metadata = load_pool()
    candidates = pool[pool.is_frozen_candidate]
    quotes = {}
    for symbol, item in read_json(MANIFEST)["symbols"].items():
        path = Path(item["path"])
        if file_hash(path) != item["sha256"]:
            raise ValueError(f"candle hash changed for {symbol}")
        c = pickle.loads(path.read_bytes())
        quotes[symbol] = c, {int(t): index for index, t in enumerate(c["time"])}

    control, nearer = [], []
    for row in candidates.to_dict("records"):
        c, lookup = quotes[row["symbol"]]
        i = lookup[int(row["entry_time"])]
        if not activity_gate(c, i):
            continue
        case = checked_row(row, c, i, "be_tp1", EXIT_STRESS)
        control.append(case)
        nearer.append(shorten(case, c, i))
    full_control = checked_cash(control, specs)
    if full_control["clients"] != reference:
        raise ValueError("long-run candle proxy does not reproduce the published baseline")

    cutoff = datetime(2026, 11, 1, tzinfo=timezone.utc).timestamp()
    cases = {name: checked_cash([r for r in rows if r["entry_time"] < cutoff], specs)
             for name, rows in (("control_2r", control), ("trial_1_5r", nearer))}
    recent_overlap = checked_cash(nearer, specs)
    report = dict(cases=cases, control_parity=True, full_period_including_november=recent_overlap,
                  pre_november_proposals=sum(r["entry_time"] < cutoff for r in control),
                  pre_november_changed_paths=sum(a is not b for a, b in zip(control, nearer, strict=True)
                                                if a["entry_time"] < cutoff),
                  source_hashes={p.name: file_hash(p) for p in (prior, specifications, MANIFEST)},
                  candle_pool_sha256=metadata["sha256"], target_r=TARGET_R, exit_stress=EXIT_STRESS,
                  live_activation_supported=False, limitations=[
                      "Native 15-minute candles with positive trade volume, not order books.",
                      "If a candle can hit both stop and target, use the stop unless a later open already crossed the target.",
                      "Baseline unknown paths remain unknown. Both variants use the previously declared entry activity rule.",
                      "Entry dates before November are disjoint from the new L2 samples, but were used in earlier research.",
                      "The joint live account and actual order marketability are outside this candle proxy.",
                  ])
    write_json(ROOT / "crypto_nearer_target_long_history_nov22.json", report)
    for name, case in cases.items():
        print(name, case["status"], {client: (cash["summary"]["return_pct"], cash["summary"]["cash_drawdown_pct"],
              {month: round(values["return_pct"], 3) for month, values in cash["calendar_months"].items()})
              for client, cash in case.get("clients", {}).items()}, flush=True)


if __name__ == "__main__":
    main()
