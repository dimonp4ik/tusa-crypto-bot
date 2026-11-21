"""Explore causal entry screens with the frozen November OKX cash replay."""

import json
import math
import pickle
from collections import Counter, defaultdict, deque

from okx_xperp_cash_tick_nov19 import simulate
from okx_xperp_multiday_replay_nov20 import FOLDER, ROOT, cached_entries, exit_books, file_hash, utc_day


def entry_features(row, book):
    mid = (book["bids"][0][0] + book["asks"][0][0]) / 2
    bids = math.fsum(qty for price, qty in book["bids"] if price >= mid * .999)
    asks = math.fsum(qty for price, qty in book["asks"] if price <= mid * 1.001)
    sign = 1 if row["direction"] == "LONG" else -1
    price = row["reference_entry"]
    risk = sign * (price - row["stop"])
    return {
        "aligned_imbalance_10bps": sign * (bids - asks) / (bids + asks),
        "quote_reward_risk": sign * (row["target"] - price) / risk,
    }


def screened_rows(rows, features, *, imbalance=None, reward_risk=None, same_side_per_hour=None):
    admitted = defaultdict(deque)
    for row in sorted(rows, key=lambda r: (r["entry_ms"], r["bot"], r["symbol"])):
        if row["bot"] != "crypto" or row["entry_status"] != "eligible":
            yield row
            continue
        at = row["entry_ms"]
        feature = features[row["symbol"], at]
        history = admitted[row["direction"]]
        while history and history[0] <= at - 3_600_000:
            history.popleft()
        if imbalance is not None and feature["aligned_imbalance_10bps"] < imbalance:
            yield dict(row, entry_status="adverse_book")
        elif reward_risk is not None and feature["quote_reward_risk"] < reward_risk:
            yield dict(row, entry_status="low_quote_reward_risk")
        elif same_side_per_hour is not None and len(history) >= same_side_per_hour:
            yield dict(row, entry_status="same_side_hour")
        else:
            history.append(at)
            yield row


def summarize(result):
    summary = {key: value for key, value in result.items() if key != "closed"}
    if "closed" not in result:
        return summary
    pnl_by_day = defaultdict(float)
    for trade in result["closed"]:
        pnl_by_day[utc_day(trade["entry_ms"] / 1000)] += trade["bank_pnl"] + trade["runner_pnl"]
    return dict(summary, cash_pnl_by_entry_day=pnl_by_day,
                exit_counts=Counter(trade["code"] for trade in result["closed"]))


def main():
    inputs = (FOLDER / "plan.json", FOLDER / "capture.json",
              ROOT / "paired_contract_volume_nov19/manifest.json", ROOT / "native_universe_nov19_snapshot.json")
    plan, capture, _, universe = (json.loads(path.read_bytes()) for path in inputs)
    source = ROOT / "okx_xperp_multiday_cash_nov20.json"
    prior = json.loads(source.read_bytes())
    if capture["plan_sha256"] != file_hash(inputs[0]):
        raise ValueError("frozen entry plan digest mismatch")
    rows = prior["paths"]
    for row in rows:
        if "path" in row:
            row["path"] = {int(key): value for key, value in row["path"].items()}
    specs = plan["specs"]
    tick_by_id = {r["instId"]: float(r["tickSz"]) for r in universe["instruments"] if r.get("tickSz")}
    ticks = {symbol: tick_by_id[spec["instId"]] for symbol, spec in specs.items()}
    cache = FOLDER / "entry_exit_samples_nov21.pkl"
    digest = file_hash(source)
    saved_digest, books = pickle.loads(cache.read_bytes()) if cache.exists() else (None, None)
    if saved_digest != digest:
        _, books, _ = cached_entries(plan, capture, inputs)
        exit_books(rows, capture, specs, books)
        cache.write_bytes(pickle.dumps((digest, books), protocol=5))
    features = {(r["symbol"], r["entry_ms"]): entry_features(r, books[r["symbol"]][r["entry_ms"]])
                for r in rows if r["bot"] == "crypto" and r["entry_status"] == "eligible"}
    policies = {
        "baseline": {},
        "book_imbalance_minus_30pct": {"imbalance": -.30},
        "book_imbalance_nonnegative": {"imbalance": 0.},
        "quote_reward_risk_1_8": {"reward_risk": 1.8},
        "one_same_side_per_rolling_hour": {"same_side_per_hour": 1},
        "two_same_side_per_rolling_hour": {"same_side_per_hour": 2},
    }
    cohorts = {
        "all": rows,
        "omit_13_nov": [r for r in rows if utc_day(r["entry_ms"] / 1000) != "2026-11-13"],
    }
    results = {}
    for cohort, proposals in cohorts.items():
        results[cohort] = {}
        for name, policy in policies.items():
            selected = screened_rows(proposals, features, **policy)
            result = simulate(selected, specs, ticks, books, clock=60, lag=200, setting="requested5")
            if name == "baseline":
                previous_cohort = "full" if cohort == "all" else "excluding_original_pilot"
                if result != prior[previous_cohort]["BE60s_exit200ms_requested5"]:
                    raise ValueError(f"baseline did not reproduce {cohort}")
            results[cohort][name] = summarize(result)

    report = {
        "source_sha256": digest,
        "scope": prior["scope"],
        "baseline_parity": True,
        "results": results,
        "stock_entry_gate_counts": Counter(r["entry_status"] for r in rows if r["bot"] == "stocks"),
        "features": [{"symbol": symbol, "entry_ms": stamp, **values}
                     for (symbol, stamp), values in features.items()],
        "live_activation_supported": False,
        "limitations": [
            "Six observed days with partial final coverage, exploratory only.",
            "Per-direction rate limits count eligible proposals before the common cash admission checks.",
            "Imbalance is a single visible top-50-level quote within 10 bps of midpoint, not matched trade flow.",
            "The same paper readiness and 3% collateral, 10x, fees and equity-guard limitations as the cash control apply.",
        ],
    }
    target = ROOT / "okx_entry_screen_cash_nov21.json"
    target.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
