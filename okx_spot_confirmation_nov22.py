"""Use independently observed spot moves to check native crypto candidates."""

import argparse
from bisect import bisect_left
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import pickle
import threading
import time

import requests

import okx_cross_bot_overlap_nov21 as joint
from crypto_active_entry_runner_nov19_audit import activity_gate
from crypto_minimum_bank_runner_nov19_audit import checked_row
from crypto_native_price_pool_nov19 import MANIFEST as NATIVE, load_pool
from crypto_nearer_target_long_history_nov22 import EXIT_STRESS, checked_cash
from crypto_scan_priority_holdout_nov22 import DATA as LATER_DATA, load_cohort
from okx_crypto_risk_range_nov22 import merge_books
from okx_entry_screen_holdout_nov21 import ENTRY_END as NOV09
from okx_joint_exact_exit_nov21 import ENTRY_END, choose_rows
from okx_tp1_cash_sensitivity_nov22 import detailed, verify_control
from okx_unbankable_tp1_nov21 import old_inputs, read_json
from okx_xperp_cash_tick_nov19 import CAPITAL, simulate
from okx_xperp_multiday_replay_nov20 import ROOT, file_hash
from public_candle_window import validate_rows
from src.crypto_profit_first_material_shadow_router import (
    REQUIRED_BTC_HISTORY_BARS, REQUIRED_SYMBOL_HISTORY_BARS, ROBUST_SYMBOLS, shadow_setups,
)
from stock_liquid_quote_routes_nov21 import limit_bursts, write_json


DATA = ROOT / "okx_spot_confirmation_nov22"
START = int(datetime(2026, 6, 30, 18, tzinfo=timezone.utc).timestamp())
END = int(datetime(2026, 11, 20, 4, tzinfo=timezone.utc).timestamp())
JULY = int(datetime(2026, 7, 1, tzinfo=timezone.utc).timestamp())
NOVEMBER = int(datetime(2026, 11, 1, tzinfo=timezone.utc).timestamp())
HOST = "https://www.okx.com"
POLICIES = ("control", "strong_spot_opposition", "local_only_impulse", "trend_only_veto",
            "veto_without_reversal_wick", "high_volume_reversal_only")


def capture_spot():
    DATA.mkdir(parents=True, exist_ok=True)
    manifest_path = DATA / "manifest.json"
    if manifest_path.exists():
        return json.loads(manifest_path.read_bytes())
    response = requests.get(HOST+"/api/v5/public/instruments", params={"instType": "SPOT"}, timeout=12)
    response.raise_for_status()
    inventory = response.json()
    if inventory.get("code") != "0":
        raise ValueError("OKX spot inventory rejected")
    available = {row["instId"] for row in inventory["data"] if row["state"] == "live"}
    wanted = {s: s[:-4]+"-USDT" for s in sorted(ROBUST_SYMBOLS)}
    request_lock, previous = threading.Lock(), 0.

    def request(session, params):
        nonlocal previous
        for attempt in range(4):
            with request_lock:
                delay = previous+.16-time.monotonic()
                if delay > 0:
                    time.sleep(delay)
                previous = time.monotonic()
            try:
                result = session.get(HOST+"/api/v5/market/history-candles", params=params, timeout=12)
                result.raise_for_status()
                payload = result.json()
                if payload.get("code") != "0":
                    raise ValueError("spot candle request was rejected")
                return payload["data"]
            except (requests.RequestException, ValueError):
                if attempt == 3:
                    raise
                time.sleep(.5*(attempt+1))

    def fetch(item):
        symbol, instrument = item
        if instrument not in available:
            return symbol, dict(instId=instrument, status="spot_pair_absent")
        path = DATA / f"{symbol}_15m.json"
        if path.exists():
            rows = json.loads(path.read_bytes())
        else:
            rows, after = [], END*1000
            with requests.Session() as session:
                for _ in range(65):
                    page = request(session, {"instId": instrument, "bar": "15m", "after": after, "limit": 300})
                    if not page:
                        raise ValueError(f"no complete spot window for {instrument}")
                    oldest = min(int(row[0]) for row in page)
                    if oldest >= after:
                        raise ValueError("spot pagination did not advance")
                    rows.extend(row for row in page if START*1000 <= int(row[0]) < END*1000)
                    if oldest <= START*1000:
                        break
                    after = oldest
                else:
                    raise ValueError("spot pagination exceeded the frozen window bound")
            validate_rows(rows, START, END, 900)
            temporary = path.with_suffix(".tmp")
            write_json(temporary, rows)
            temporary.replace(path)
        series = validate_rows(rows, START, END, 900)
        print("OKX spot", symbol, len(series["time"]), "confirmed bars", flush=True)
        return symbol, dict(path=str(path), sha256=file_hash(path), instId=instrument, status="complete")

    with ThreadPoolExecutor(max_workers=4) as pool:
        instruments = dict(pool.map(fetch, wanted.items()))
    manifest = dict(start=START, end=END, instruments=instruments, venue="OKX spot")
    write_json(manifest_path, manifest)
    return manifest


def load_spot(manifest):
    if manifest["start"] != START or manifest["end"] != END:
        raise ValueError("frozen spot window changed")
    frames = {}
    for symbol, entry in manifest["instruments"].items():
        if entry["status"] == "spot_pair_absent":
            continue
        path = Path(entry["path"])
        if file_hash(path) != entry["sha256"]:
            raise ValueError("canonical spot candles changed")
        frames[symbol] = validate_rows(read_json(path), START, END, 900)
    return frames


def load_native(items):
    frames = {}
    for symbol, item in items.items():
        path = Path(item["path"])
        if file_hash(path) != item["sha256"]:
            raise ValueError("canonical native candles changed")
        frames[symbol] = pickle.loads(path.read_bytes())
    return frames


def closed_index(frame, at, count):
    if frame is None:
        return None
    i = bisect_left(frame["time"], at)
    if i < count or list(frame["time"][i-count:i]) != list(range(at-count*900, at, 900)):
        return None
    return i


def features(spots, natives, row, *, stamp):
    symbol, side = row["symbol"], row["direction"]
    spot, native = spots.get(symbol), natives.get(symbol)
    si, ni = closed_index(spot, stamp, 20), closed_index(native, stamp, 5)
    if si is None or ni is None:
        return None
    sign = 1 if side == "LONG" else -1
    prices, volumes = spot["close"], spot["volume"]
    cash = native["close"]
    typical = math.fsum(volumes[si-20:si-4])/4
    if typical <= 0:
        return None
    low, high = spot["low"][si-1], spot["high"][si-1]
    close_location = (prices[si-1]-low)/(high-low) if high>low else .5
    return dict(spot_1h=sign*(prices[si-1]/prices[si-5]-1),
                spot_15m=sign*(prices[si-1]/prices[si-2]-1),
                spot_volume_ratio=math.fsum(volumes[si-4:si])/typical,
                native_1h=sign*(cash[ni-1]/cash[ni-5]-1),
                directional_close=close_location if sign>0 else 1-close_location, family=row.get("family"))


def veto(feature, policy):
    if policy == "control":
        return False
    if feature is None:
        return policy == "high_volume_reversal_only"
    strong = feature["spot_1h"] <= -.006 and feature["spot_15m"] < 0 and feature["spot_volume_ratio"] >= 1.5
    spread = feature["native_1h"] >= .002 and feature["spot_1h"] <= -.0015
    trend = feature["family"] in {"breakout", "trend_pullback"}
    no_reversal_wick = feature["family"] == "range_reversion" and feature["directional_close"] < .25
    return {"strong_spot_opposition": strong, "local_only_impulse": spread,
            "trend_only_veto": strong and trend,
            "veto_without_reversal_wick": strong and (trend or no_reversal_wick),
            "high_volume_reversal_only": not (strong and feature["family"] == "range_reversion")}[policy]


def candle_trials(spots, natives, pool, meta):
    by_time = {s: {int(t): i for i, t in enumerate(frame["time"])} for s, frame in natives.items()}
    baseline_report = ROOT/"crypto_active_entry_runner_nov19.json"
    specs = read_json(ROOT/"shared_xperp_specs_nov14.json")
    rows = []
    for row in pool[pool.is_frozen_candidate].to_dict("records"):
        symbol, at = row["symbol"], row["entry_time"]
        bars = natives[symbol]
        i = by_time[symbol][at]
        if activity_gate(bars, i):
            outcome = checked_row(row, bars, i, "be_tp1", EXIT_STRESS)
            outcome["spot_context"] = features(spots, natives, row, stamp=at)
            rows.append(outcome)
    tested = {}
    for name in POLICIES:
        removed = [r for r in rows if veto(r["spot_context"], name)]
        score = checked_cash([r for r in rows if not veto(r["spot_context"], name)], specs)
        tested[name] = dict(**score, vetoes=Counter(r["symbol"] for r in removed),
                            november_vetoes=sum(r["entry_time"] >= NOVEMBER for r in removed))
        print("Candle proxy", name, {p: (d["summary"]["return_pct"], d["summary"]["cash_drawdown_pct"])
                                       for p,d in score.get("clients",{}).items()}, flush=True)
    if tested["control"]["clients"] != read_json(baseline_report)["results"]["be_tp1"]["cash"]:
        raise ValueError("native cash control does not reproduce the earlier report")
    coverage = Counter(r["symbol"] for r in rows if r["entry_time"]>=JULY)
    known = Counter(r["symbol"] for r in rows if r["entry_time"]>=JULY and r["spot_context"] is not None)
    return dict(results=tested, native_pool_sha256=meta["sha256"], baseline_sha256=file_hash(baseline_report),
                proposals=len(rows), known_context=sum(r["spot_context"] is not None for r in rows),
                scored_interval_coverage={s: dict(proposals=n, known=known[s]) for s,n in sorted(coverage.items())})


def book_trials(spots, natives, rows, specs, ticks, books, *, shared):
    contexts = {(r["symbol"], r["entry_ms"]): features(spots, natives, r, stamp=r["epoch"])
                for r in rows if r["bot"]=="crypto" and r["entry_status"]=="eligible"}
    controls = dict(clock=60, setting="requested5", max_positions=6 if shared else 12)
    results = {}
    control_trades = {}
    for name in POLICIES:
        blocked = []
        kept = []
        for row in rows:
            context = contexts.get((row["symbol"], row["entry_ms"])) if row["bot"]=="crypto" else None
            if row["bot"]=="crypto" and row["entry_status"]=="eligible" and veto(context, name):
                blocked.append(dict(symbol=row["symbol"], entry=row["entry_ms"], context=context))
                kept.append(dict(row, entry_status="spot_veto"))
            else:
                kept.append(row)
        if shared:
            kept = [*joint.crypto_cadence(r for r in kept if r["bot"]=="crypto"),
                    *limit_bursts((r for r in kept if r["bot"]=="stocks"), max_same_side=2)]
        cash = {}
        affected = {}
        blocked_keys = {(r["symbol"], r["entry"]) for r in blocked}
        for lag in (200, 1000):
            value = simulate(kept, specs, ticks, books, lag=lag, **controls)
            if name == "control":
                control_trades[lag] = value.get("closed", [])
            affected[str(lag)] = [dict(symbol=r["symbol"], entry=r["entry_ms"], exit_code=r["code"],
                cash_pnl_usd=r["bank_pnl"]+r["runner_pnl"]) for r in control_trades[lag]
                if r["bot"]=="crypto" and (r["symbol"],r["entry_ms"]) in blocked_keys]
            cash[str(lag)] = detailed(value)
        results[name] = dict(blocked_proposals=blocked, control_trades_filtered=affected, cash=cash)
    return dict(cases=results, eligible_crypto=len(contexts), context_available=sum(v is not None for v in contexts.values()),
                known_family=sum(v is not None and v["family"] is not None for v in contexts.values()))


def recent_family(row, frames):
    at, symbol = row["epoch"], row["symbol"]
    known = shadow_setups(joint.closed(frames[symbol], at, REQUIRED_SYMBOL_HISTORY_BARS),
                          joint.closed(frames["BTCUSDT"], at, REQUIRED_BTC_HISTORY_BARS),
                          entry_time=at, market_price=row["market_anchor"], model_price=row["opening_anchor"], symbol=symbol)
    if (len(known) != 1 or known[0]["direction"] != row["direction"]
            or not math.isclose(known[0]["sl"], row["stop"], rel_tol=1e-12)):
        raise ValueError("original native family did not reproduce")
    return known[0]["family"]


def market_books(spots, natives, pool):
    base = ROOT/"okx_entry_screen_holdout_nov21.json"
    rows, specs, ticks, books, *_ = old_inputs()
    source = read_json(NATIVE)["symbols"]
    if any(specs[s]["instId"] != source[s]["instrument"] for s in
           {r["symbol"] for r in rows if r["bot"]=="crypto" and r["entry_status"]=="eligible"}):
        raise ValueError("historical derivatives features came from a different instrument")
    verify_control(rows, specs, ticks, books, 12, base, joint=False)
    tagged = {(r.symbol, r.entry_time, r.direction): r.family for r in pool[pool.is_frozen_candidate].itertuples()}
    rows = [dict(r, family=tagged.get((r["symbol"], r["epoch"], r["direction"])))
            if r["bot"]=="crypto" else r for r in rows]
    cases = {"nov05_08": book_trials(spots, natives, [r for r in rows if r["epoch"]<NOV09], specs, ticks, books, shared=False),
             "nov09_14": book_trials(spots, natives, [r for r in rows if r["epoch"]>=NOV09], specs, ticks, books, shared=False)}
    crypto, stocks, specs, ticks, books, hashes, _, _, _, unknown = joint.joint_inputs()
    if any(joint.START<=r["epoch"]<ENTRY_END for r in unknown):
        raise ValueError("missing exact common entry book")
    ready = choose_rows(crypto, stocks)
    saved = pickle.loads((joint.DATA/"joint_exact_exit_samples.pkl").read_bytes())
    stock_report = ROOT/"stock_liquid_quote_routes_nov21.json"
    if saved["sources"] != hashes | {str(stock_report): file_hash(stock_report)} or saved["entry_end"] != ENTRY_END:
        raise ValueError("exact common exit evidence changed")
    merge_books(books, saved["result"][0])
    baseline = ROOT/"okx_joint_exact_exit_nov21.json"
    verify_control([*joint.crypto_cadence(r for r in ready if r["bot"]=="crypto"),
                    *limit_bursts((r for r in ready if r["bot"]=="stocks"),max_same_side=2)],
                   specs, ticks, books, 6, baseline, joint=True)
    native_path = joint.DATA/"crypto_capture.json"
    fresh = load_native(read_json(native_path)["references"])
    ready = [dict(r, family=recent_family(r, fresh)) if r["bot"]=="crypto" and r["entry_status"]=="eligible" else r
             for r in ready]
    cases["nov17_19_shared"] = book_trials(spots, fresh, ready, specs, ticks, books, shared=True)
    return dict(cases=cases, source_hashes={k.name: file_hash(k) for k in (base, baseline, native_path)})


def later_opportunities(spots):
    path = LATER_DATA/"crypto_capture.json"
    capture, rows, _ = load_cohort(path)
    native = load_native(capture["references"])
    historical = ROOT/"crypto_scan_priority_holdout_nov22.json"
    market_case = read_json(historical)["results"]["alphabetical"]["200"]
    admitted = {(symbol, at//900_000*900) for symbol, at in market_case["admitted"]}
    annotations = []
    for row in rows:
        row = dict(row, family=recent_family(row, native))
        context = features(spots, native, row, stamp=row["epoch"])
        annotations.append((row, context))
    fired = {}
    for policy in POLICIES:
        blocked = {(row["symbol"], row["epoch"]) for row, context in annotations if veto(context, policy)}
        fired[policy] = dict(blocked_proposals=len(blocked), blocked_previously_admitted=len(admitted & blocked),
                             surviving_proposals=len(rows)-len(blocked))
    return dict(start=capture["start"], stop=capture["entry_end"], raw_qualified=len(rows),
                previously_admitted=len(admitted), coverage=sum(c is not None for _,c in annotations),
                screening=fired, source_hashes={p.name: file_hash(p) for p in (path,historical)},
                cash_replayed=False)


def evaluate():
    manifest = capture_spot()
    spots, natives = load_spot(manifest), load_native(read_json(NATIVE)["symbols"])
    pool, meta = load_pool()
    candle = candle_trials(spots, natives, pool, meta)
    exact = market_books(spots, natives, pool)
    later = later_opportunities(spots)
    write_json(ROOT/"okx_spot_confirmation_nov22.json", dict(spot_source_sha256=file_hash(DATA/"manifest.json"),
        native_candle_manifest_sha256=file_hash(NATIVE), capital_usd=CAPITAL, candle_history=candle, exact_books=exact,
        nov20_candidate_screen=later,
        missing_pair=[s for s,v in manifest["instruments"].items() if v["status"]=="spot_pair_absent"],
        live_activation_supported=False, limitations=[
            "Spot uses native OKX USDT pairs; known spot close and trailing volume only.",
            "Strong spot opposition is -0.6% signed 1h, signed last 15m <0, and >=1.5x typical hourly volume.",
            "Local-only impulse requires X-Perp +0.2% signed 1h while spot <=-0.15%.",
            "Family-sensitive variants retain range reversals unless their latest directional spot close is in the adverse quarter.",
            "Missing context preserves original entries for veto policies. The standalone reversal route requires complete context.",
            "The standalone reversal route is an exploratory follow-up motivated by the failure of the broad veto.",
            "The earlier order-book proposals were already shortlisted; skipped entries cannot be replaced there.",
            "The shared account comparison re-applies both cadences inside the six-position modeled limit.",
        ]))
    for name, group in exact["cases"].items():
        print("Public books", name, {p:(len(v["blocked_proposals"]), v["cash"]["200"].get("cash_pnl_usd"))
                                      for p,v in group["cases"].items()}, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluate", action="store_true")
    if parser.parse_args().evaluate:
        evaluate()
    else:
        capture_spot()
