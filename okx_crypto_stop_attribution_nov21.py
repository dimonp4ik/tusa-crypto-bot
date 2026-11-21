"""Compare causal stop-risk screens in an earlier cash replay and a joint replay."""

import json
import pickle
from bisect import bisect_left
from collections import Counter, defaultdict
from pathlib import Path

from okx_calendar_exposure_nov21 import read_case
from okx_cross_bot_overlap_nov21 import DATA, crypto_cadence, joint_inputs
from okx_entry_screen_cash_nov21 import entry_features, summarize
from okx_xperp_cash_tick_nov19 import simulate
from okx_xperp_multiday_replay_nov20 import FOLDER, ROOT, file_hash
from stock_liquid_quote_routes_nov21 import limit_bursts, write_json


SCREENS = (
    "baseline",
    "against_btc_hour",
    "btc_move_reversing",
    "unprotected_nearby_pivot",
    "adverse_last_bar",
    "adverse_last_bar_aligned",
    "adverse_last_bar_countertrend",
    "adverse_last_bar_neutral",
    "one_same_bar_tightest_spread",
    "one_same_bar_highest_reward_risk",
)


def btc_phase(aligned_return_4h):
    if aligned_return_4h >= .005:
        return "aligned"
    if aligned_return_4h <= -.005:
        return "countertrend"
    return "neutral"


def read_verified_frames(specs):
    frames = {}
    for symbol, evidence in specs.items():
        path = Path(evidence["path"])
        if file_hash(path) != evidence["sha256"]:
            raise ValueError(f"changed candle history for {symbol}")
        frames[symbol] = pickle.loads(path.read_bytes())
    return frames


def known_pivots(frame, end, *, high):
    values = frame["high" if high else "low"]
    sign = 1 if high else -1
    for i in range(end - 30, end - 2):
        price = values[i]
        if all(sign * price > sign * values[other] for other in (i - 2, i - 1, i + 1, i + 2)):
            yield price


def features(row, frames, book):
    anchor = row["epoch"]
    side = 1 if row["direction"] == "LONG" else -1
    frame, btc = frames[row["symbol"]], frames["BTCUSDT"]
    end = bisect_left(frame["time"], anchor)
    btc_end = bisect_left(btc["time"], anchor)
    if (end < 32 or btc_end < 17 or frame["time"][end - 1] != anchor - 900
            or btc["time"][btc_end - 1] != anchor - 900):
        raise ValueError("missing closed history at the entry")
    close, high, low = (frame[key][end - 1] for key in ("close", "high", "low"))
    relative_close = (close - low) / (high - low) if high > low else .5
    signed_close = relative_close if side > 0 else 1 - relative_close

    def btc_return(period):
        return side * (btc["close"][btc_end - 1] / btc["close"][btc_end - period - 1] - 1)

    actual = row["reference_entry"]
    risk = abs(actual - row["stop"])
    distances = [side * (actual - price) / risk for price in known_pivots(frame, end, high=side < 0)
                 if side * (actual - price) >= risk]
    quote = entry_features(row, book)
    return dict(btc_aligned_return_1h=btc_return(4), btc_aligned_return_4h=btc_return(16),
                btc_aligned_return_30m=btc_return(2), directional_bar_close=signed_close,
                nearest_adverse_pivot_beyond_stop_r=min(distances, default=None),
                entry_spread_bps=row["tries"][-1]["spread_bps"], **quote)


def keep_by_geometry(policy, f):
    if policy == "against_btc_hour":
        return f["btc_aligned_return_1h"] >= -.0035
    if policy == "btc_move_reversing":
        return f["btc_aligned_return_4h"] <= .015 or f["btc_aligned_return_30m"] >= -.002
    if policy == "unprotected_nearby_pivot":
        return f["nearest_adverse_pivot_beyond_stop_r"] is None or f["nearest_adverse_pivot_beyond_stop_r"] > 1.25
    if policy == "adverse_last_bar":
        return f["directional_bar_close"] >= .15
    if policy.startswith("adverse_last_bar_"):
        phase = policy.removeprefix("adverse_last_bar_")
        return btc_phase(f["btc_aligned_return_4h"]) != phase or f["directional_bar_close"] >= .15
    return True


def selected(rows, values, policy):
    prioritize = {
        "one_same_bar_tightest_spread": ("entry_spread_bps", 1.),
        "one_same_bar_highest_reward_risk": ("quote_reward_risk", -1.),
    }
    group = defaultdict(list)
    for row in rows:
        if row["entry_status"] != "eligible":
            yield row
        elif policy in prioritize:
            group[row["epoch"], row["direction"]].append(row)
        else:
            allowed = keep_by_geometry(policy, values[row["symbol"], row["entry_ms"]])
            yield row if allowed else dict(row, entry_status=policy)
    for cohort in group.values():
        key, sign = prioritize[policy]
        winner = min(cohort, key=lambda r: (r["entry_ms"], sign * values[r["symbol"], r["entry_ms"]][key], r["symbol"]))
        for row in cohort:
            yield row if row is winner else dict(row, entry_status=policy)


def feature_map(rows, frames, books):
    return {(r["symbol"], r["entry_ms"]): features(r, frames, books[r["symbol"]][r["entry_ms"]])
            for r in rows if r["entry_status"] == "eligible"}


def compare(market, rows, control, proposals, values, reprice, setting):
    if "closed" not in control:
        raise ValueError(f"missing cash baseline: {market}")
    base = {(r["symbol"], r["entry_ms"]): r for r in control["closed"] if r["bot"] == "crypto"}
    output = {}
    for screen in SCREENS:
        case = reprice(proposals(selected(rows, values, screen)), setting)
        stats = summarize(case)
        if "closed" in case:
            after = {(r["symbol"], r["entry_ms"]): r for r in case["closed"] if r["bot"] == "crypto"}
            removed = [r for key, r in base.items() if key not in after]
            stats.update(removed_baseline_exits=Counter(r["code"] for r in removed),
                         removed_baseline_profitable=sum(r["bank_pnl"] + r["runner_pnl"] > 0 for r in removed),
                         added_crypto_positions=len(after.keys() - base.keys()))
        if screen == "baseline" and case["closed"] != control["closed"]:
            raise ValueError(f"control reproduction failed: {market}")
        output[screen] = stats
    return output


def main():
    historical = ROOT / "okx_xperp_multiday_cash_nov20.json"
    old_reference = ROOT / "paired_contract_volume_nov19/manifest.json"
    new_reference = DATA / "crypto_capture.json"
    old_frames = read_verified_frames(json.loads(old_reference.read_bytes())["symbols"])
    new_frames = read_verified_frames(json.loads(new_reference.read_bytes())["references"])
    early_specs, early, old_books = read_case(FOLDER.with_name("nov5_8_entry_screen"), "verified_public_paths.pkl")
    later_specs, later, later_books = read_case(FOLDER, "entry_exit_samples_nov21.pkl", file_hash(historical))
    for s, snapshots in later_books.items():
        shared = old_books[s].keys() & snapshots.keys()
        if any(old_books[s][stamp] != snapshots[stamp] for stamp in shared):
            raise ValueError("earlier overlap changed")
        old_books[s].update(snapshots)
    old_specs = early_specs | later_specs
    snapshot = json.loads((ROOT / "native_universe_nov19_snapshot.json").read_bytes())
    by_id = {r["instId"]: float(r["tickSz"]) for r in snapshot["instruments"] if r.get("tickSz")}
    old_ticks = {s: by_id[v["instId"]] for s, v in old_specs.items()}
    old_crypto = [r for r in early + later if r["bot"] == "crypto"]
    old_stocks = [r for r in early + later if r["bot"] == "stocks"]
    old_values = feature_map(old_crypto, old_frames, old_books)

    def reprice_old(rows, setting):
        return simulate(rows, old_specs, old_ticks, old_books, clock=60, lag=200, setting=setting)

    paths = (ROOT / "okx_entry_screen_holdout_nov21.json", ROOT / "okx_cross_bot_overlap_nov21.json")
    old_control = reprice_old(old_crypto + old_stocks, "requested5")
    expected = json.loads(paths[0].read_bytes())["continuous_cash_results"]["results"]["baseline"]
    if summarize(old_control) != expected:
        raise ValueError("earlier account baseline did not match")
    older = compare("Nov5–14", old_crypto, old_control, lambda candidates: list(candidates) + old_stocks,
                    old_values, reprice_old, "requested5")

    crypto, stocks, specs, ticks, books, _, _, _, exit_proxy, unknown = joint_inputs()
    if unknown:
        raise ValueError("missing recent execution snapshot")
    values = feature_map(crypto, new_frames, books)
    published_stocks = list(limit_bursts(stocks, max_same_side=2))

    def joint(candidates):
        return list(crypto_cadence(candidates)) + published_stocks

    def reprice_joint(rows, setting):
        return simulate(rows, specs, ticks, books, clock=60, lag=200, setting=setting,
                        exit_price=exit_proxy(15), max_positions=6)

    later_control = reprice_joint(joint(crypto), "half_minimum")
    expected = json.loads(paths[1].read_bytes())["cases"]["both_stock_two_per_side|cap6|exit15bps|half_minimum"]
    if later_control["closed"] != expected["closed"]:
        raise ValueError("recent shared-account baseline did not match")
    latest = compare("Nov17–19", crypto, later_control, joint, values, reprice_joint, "half_minimum")
    ledger = []
    for label, control, v, proposals in (("Nov5–14", old_control, old_values, old_crypto),
                                        ("Nov17–19", later_control, values, crypto)):
        directions = {(r["symbol"], r["entry_ms"]): r["direction"] for r in proposals}
        for trade in control["closed"]:
            if trade["bot"] == "crypto":
                key = trade["symbol"], trade["entry_ms"]
                ledger.append(dict(window=label, direction=directions[key], **trade, **v[key]))
    report = dict(sources={str(p): file_hash(p) for p in (*paths, old_reference, new_reference)},
                  baseline_parity=True, earlier_public_l2=older, recent_joint_exit_stress=latest,
                  crypto_attribution=ledger, live_activation_supported=False,
                  limitations=[
                      "Both date windows were previously viewed. Screen thresholds are exploratory.",
                      "Earlier window uses published frozen entries. Later window rebuilds entries at the latest observed price.",
                      "Different stock readiness and exit models apply in the two windows.",
                      "Attribution uses only closed candles or the original executable quote.",
                      "Same-bar alternatives select only among quotes available at the earliest receipt.",
                  ])
    write_json(ROOT / "okx_crypto_stop_attribution_nov21.json", report)
    for alias, cases in (("earlier", older), ("recent", latest)):
        print(alias, json.dumps({name: {k: v for k, v in r.items()
                                       if k in ("return_pct", "drawdown_pct", "contribution_pct",
                                                "trades", "status", "reason", "removed_baseline_exits",
                                                "removed_baseline_profitable", "added_crypto_positions")}
                                for name, r in cases.items()}, indent=2))


if __name__ == "__main__":
    main()
