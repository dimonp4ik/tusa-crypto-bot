"""Rebuild contemporaneous stock and crypto trials on one modeled OKX account."""

import argparse
import json
import pickle
from bisect import bisect_left, bisect_right
from collections import Counter, defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from okx_recent_trade_repair_nov21 import load_repaired_tape
from okx_xperp_cash_tick_nov19 import RELEASE_MS, simulate
from okx_xperp_multiday_capture_nov20 import HOLD, book_files, file_hash, trade_files
from okx_xperp_multiday_replay_nov20 import entries, resolve_all
from okx_xperp_tick_path_nov19 import load_trade_tape, reconcile
from public_candle_window import fetch_window
from src.crypto_profit_first_material_shadow_router import (
    ARTIFACT_PATH, REQUIRED_BTC_HISTORY_BARS, REQUIRED_SYMBOL_HISTORY_BARS,
    ROBUST_SYMBOLS, shadow_setups,
)
from src.okx_trader import round_to_tick
from stock_liquid_quote_routes_nov21 import DATA as STOCK_DATA, ROOT, limit_bursts, write_json


START = int(datetime(2026, 11, 17, 22, 30, tzinfo=timezone.utc).timestamp())
ENTRY_END = int(datetime(2026, 11, 20, tzinfo=timezone.utc).timestamp())
DATA = ROOT / "okx_cross_bot_overlap_nov21"
ARCHIVE_SPEC = ROOT / "native_universe_nov19_snapshot.json"
MODEL_MODULE = ARTIFACT_PATH.with_name("crypto_profit_first_material_shadow_router.py")
ENTRY_OFFSET_MS = 10_000


def declared_specs():
    by_symbol = {row["ctValCcy"] + "USDT": row for row in json.loads(ARCHIVE_SPEC.read_bytes())["instruments"]
                 if row.get("settleCcy") == "USD" and row.get("instId", "").split("-")[1:2] == ["USD_UM_XPERP"]
                 and row.get("state") == "live" and row.get("ctValCcy")}
    specs = {symbol: by_symbol[symbol] for symbol in sorted(ROBUST_SYMBOLS)}
    for symbol, item in specs.items():
        if not item["instId"].startswith(symbol[:-4] + "-USD_UM_XPERP-"):
            raise ValueError("instrument specs did not match the crypto universe")
    return specs


def capture_crypto():
    DATA.mkdir(parents=True, exist_ok=True)
    specs = declared_specs()
    skeleton = [dict(symbol=symbol, epoch=at, bot="crypto")
                for symbol in specs for at in range(START, ENTRY_END, 900)]
    seed = dict(rows=skeleton, specs=specs, start=START, coverage_end=ENTRY_END + HOLD["crypto"])
    trades = trade_files(seed)
    references = {}

    def candles(symbol):
        begin = START - 900 * (REQUIRED_BTC_HISTORY_BARS if symbol == "BTCUSDT" else REQUIRED_SYMBOL_HISTORY_BARS)
        end = seed["coverage_end"]
        frame = defaultdict(list)
        for start in range(begin, end, 86_400):
            values, _ = fetch_window(specs[symbol]["instId"], start, min(start + 86_400, end),
                                     900, DATA / "candles")
            for field, series in values.items():
                frame[field].extend(series)
        frame["volume_contracts"] = frame["volume"]
        ref = DATA / f"{symbol}_reference.pkl"
        ref.write_bytes(pickle.dumps(dict(frame), protocol=5))
        return symbol, dict(path=str(ref), sha256=file_hash(ref))

    with ThreadPoolExecutor(max_workers=3) as pool:
        references.update(pool.map(candles, sorted(specs)))
    manifest = dict(specs=specs, references=references, trades=trades,
                    source_digests={str(p): file_hash(p) for p in (ARCHIVE_SPEC, ARTIFACT_PATH, MODEL_MODULE)},
                    start=START, entry_end=ENTRY_END)
    write_json(DATA / "crypto_capture.json", manifest)
    print("crypto capture", len(references), "contracts",
          sum(len(v["days"]) for v in trades.values()), "trade files", flush=True)


def closed(frame, before, length):
    index = bisect_left(frame["time"], before)
    if index < length or frame["time"][index - 1] != before - 900:
        raise ValueError("crypto candle not aligned at decision")
    return {field: values[index-length:index] for field, values in frame.items()}


def reconstruct(captured):
    if captured["start"] != START or captured["entry_end"] != ENTRY_END:
        raise ValueError("crypto chronology changed")
    if captured["source_digests"] != {str(p): file_hash(p) for p in (ARCHIVE_SPEC, ARTIFACT_PATH, MODEL_MODULE)}:
        raise ValueError("frozen model or instrument provenance changed")
    frames = {}
    for symbol, evidence in captured["references"].items():
        path = Path(evidence["path"])
        if file_hash(path) != evidence["sha256"]:
            raise ValueError("crypto history digest changed")
        frames[symbol] = pickle.loads(path.read_bytes())
    proposals = []
    checks = {}
    for symbol in sorted(ROBUST_SYMBOLS):
        hist = frames[symbol]
        trades = load_trade_tape(captured["trades"][symbol])
        times = [t.ms for t in trades]
        checks[symbol] = reconcile(trades, hist, start=START, end=ENTRY_END + HOLD["crypto"] - 900)
        if checks[symbol]:
            raise ValueError(f"crypto trade and candle mismatch for {symbol}")
        for at in range(START, ENTRY_END, 900):
            candles = closed(hist, at, REQUIRED_SYMBOL_HISTORY_BARS)
            btc = closed(frames["BTCUSDT"], at, REQUIRED_BTC_HISTORY_BARS)
            start = bisect_left(times, at * 1000)
            end = bisect_left(times, at * 1000 + ENTRY_OFFSET_MS)
            anchor = trades[start].price if start < end else candles["close"][-1]
            market = trades[end - 1].price if start < end else anchor
            setups = shadow_setups(candles, btc, entry_time=at, market_price=market, model_price=anchor,
                                   symbol=symbol)
            for row in setups:
                entry = row["entry"]
                stop = row["sl"]
                proposals.append(dict(epoch=at, bot="crypto", symbol=symbol, direction=row["direction"],
                    entry=entry, stop=stop, target=entry + row["runner_target_r"] * (entry - stop),
                    market_anchor=market, opening_anchor=anchor, prints_before_decision=end-start,
                    paper_reference_ready=False))
    return sorted(proposals, key=lambda r: (r["epoch"], r["symbol"])), checks


def freeze_crypto():
    capture_path = DATA / "crypto_capture.json"
    captured = json.loads(capture_path.read_bytes())
    old_plan = DATA / "crypto_plan.json"
    rows, checks = reconstruct(captured)
    plan = dict(rows=rows, specs=captured["specs"], tape_mismatches=checks,
                input_sha256=file_hash(capture_path), start=START, coverage_end=ENTRY_END + HOLD["crypto"],
                parameters=dict(crypto_offset_ms=ENTRY_OFFSET_MS, retry_offsets_ms=[0, 200, 800]))
    if old_plan.exists() and json.loads(old_plan.read_bytes()) != plan:
        raise ValueError("the jointly observed crypto plan changed")
    if not old_plan.exists():
        write_json(old_plan, plan)
    print("crypto frozen", len(rows), "signals by symbol", dict(Counter(r["symbol"] for r in rows)), flush=True)
    files = book_files(plan)
    write_json(DATA / "crypto_books.json", dict(plan_sha256=file_hash(old_plan), books=files))
    print("crypto L2", len(files), "contracts/days", sum("path" in v for v in files.values()), "available",
          flush=True)


def crypto_cadence(rows):
    opened = deque()
    for row in sorted(rows, key=lambda r: (r["entry_ms"], r["symbol"])):
        if row["entry_status"] != "eligible":
            yield row
            continue
        stamp = row["entry_ms"]
        while opened and opened[0] <= stamp - 3_600_000:
            opened.popleft()
        if len(opened) == 2:
            yield dict(row, entry_status="publication_rate_limit")
        else:
            opened.append(stamp)
            yield row


def overlap_view(closed):
    events = defaultdict(lambda: Counter())
    for trade in closed:
        events[trade["entry_ms"]][trade["bot"]] += 1
        events[trade["exit_ms"] + RELEASE_MS][trade["bot"]] -= 1
    active = Counter()
    occupied = 0
    simultaneous_seconds = 0.
    peak_split = Counter()
    last = None
    for stamp, updates in sorted(events.items()):
        if last is not None and active["stocks"] and active["crypto"]:
            simultaneous_seconds += (stamp - last) / 1000
        active.update(updates)
        if sum(active.values()) > occupied:
            occupied = sum(active.values())
            peak_split = active.copy()
        last = stamp
    return dict(peak_reservation_split=dict(peak_split), joint_reservation_seconds=simultaneous_seconds,
                closes=Counter(t["bot"] for t in closed),
                profitable_closes=Counter(t["bot"] for t in closed if t["bank_pnl"] + t["runner_pnl"] > 0))


def score_joint():
    files = [DATA / name for name in ("crypto_plan.json", "crypto_capture.json", "crypto_books.json")]
    files += [STOCK_DATA / name for name in ("plan.json", "capture.json", "trade_repair.json")]
    plans = [json.loads(path.read_bytes()) for path in files]
    crypto_plan, crypto_capture, crypto_book_manifest, stock_plan, stock_capture, repair = plans
    hashes = {str(path): file_hash(path) for path in files}
    if (crypto_plan["input_sha256"] != hashes[str(files[1])]
            or crypto_book_manifest["plan_sha256"] != hashes[str(files[0])]
            or stock_capture["plan_sha256"] != hashes[str(files[3])]
            or repair["capture_sha256"] != hashes[str(files[4])]):
        raise ValueError("overlap inputs do not match their frozen manifests")
    stock_rows = [r for cohort in stock_plan["cases"].values() for r in cohort
                  if START <= r["epoch"] < ENTRY_END and r["route"] == "aligned_continuation"]
    if not stock_rows:
        raise ValueError("stock continuation has an empty common window")
    specs = stock_capture["specs"] | crypto_plan["specs"]
    ticks = {s: float(spec["tickSz"]) for s, spec in specs.items()}
    cache = DATA / "joint_entry_samples.pkl"
    saved = pickle.loads(cache.read_bytes()) if cache.exists() else None
    if saved and saved["hashes"] == hashes:
        crypto_rows, books, entry_checks = saved["data"]
    else:
        crypto_rows, books, entry_checks = entries(crypto_plan, crypto_book_manifest)
        cache.write_bytes(pickle.dumps(dict(hashes=hashes, data=(crypto_rows, books, entry_checks)), protocol=5))
    for row in stock_rows:
        books[row["symbol"]][row["entry_ms"]] = row["book"]
    for row in crypto_rows:
        if row["entry_status"] != "eligible":
            continue
        target1 = round_to_tick(row["entry"] + .5 * (row["entry"] - row["stop"]), ticks[row["symbol"]])
        worst = row["tries"][-1]["worst"]
        sign = 1 if row["direction"] == "LONG" else -1
        if not sign * row["stop"] < sign * worst < sign * target1:
            row["entry_status"] = "outside_tp1_bracket"
    rows = crypto_rows + stock_rows
    unknown_quotes = [r for r in rows if r["entry_status"] == "unknown_quote"]
    for symbol, intervals in repair["intervals"].items():
        stock_capture["trades"][symbol]["repaired"] = intervals
    trades = crypto_capture["trades"] | stock_capture["trades"]
    references = crypto_capture["references"] | stock_capture["reference"]
    loaded = {}

    def get_tape(item):
        tape = load_repaired_tape(item)
        loaded[item["instrument"]] = ([t.ms for t in tape], tape)
        return tape

    tape_checks = resolve_all(rows, {"files": trades}, {"symbols": references}, ticks, trade_loader=get_tape)

    def exit_proxy(penalty):
        def price(symbol, when, side, _):
            stamps, tape = loaded[specs[symbol]["instId"]]
            index = bisect_right(stamps, when) - 1
            if index < 0 or when - stamps[index] > 60_000:
                raise ValueError("exit proxy has no recent public print")
            sign = 1 if side == "LONG" else -1
            return tape[index].price * (1 - sign * penalty / 10_000)
        return price

    published = list(crypto_cadence(crypto_rows))
    scenarios = {
        "crypto_only": published,
        "stock_only": list(limit_bursts(stock_rows, max_same_side=2)),
        "both_stock_two_per_side": published + list(limit_bursts(stock_rows, max_same_side=2)),
        "both_stock_one_per_side": published + list(limit_bursts(stock_rows, max_same_side=1)),
    }
    results = {}
    for name, proposals in scenarios.items():
        for cap in (4, 6, 12) if name.startswith("both") else (12,):
            for penalty in (5, 15, 30):
                settings = ("requested5", "half_minimum") if name != "stock_only" else ("requested5",)
                for setting in settings:
                    if unknown_quotes and name != "stock_only":
                        result = dict(status="UNSCORABLE_ENTRY_ARCHIVE", unknown=len(unknown_quotes))
                    else:
                        result = simulate(proposals, specs, ticks, books, clock=60, lag=200, setting=setting,
                                          exit_price=exit_proxy(penalty), max_positions=cap)
                    if "closed" in result:
                        result.update(overlap_view(result["closed"]))
                        result["status"] = "SCORED_PUBLIC_TRADE_EXIT_STRESS"
                    results[f"{name}|cap{cap}|exit{penalty}bps|{setting}"] = result
    report = dict(start=START, entry_end=ENTRY_END, input_hashes=hashes, tape=tape_checks, crypto_books=entry_checks,
                  quotes=Counter(r["bot"] + ":" + r["entry_status"] for r in rows),
                  crypto_quotes_by_symbol={symbol: Counter(r["entry_status"] for r in crypto_rows
                                                          if r["symbol"] == symbol)
                                           for symbol in sorted({r["symbol"] for r in crypto_rows})},
                  cases=results, live_activation_supported=False,
                  limitations=["Stock continuation eligibility is assumed for this portfolio experiment.",
                               "One account with 3% initial margin per trade, 10x, local caps 4/8 and 15m slot release delay.",
                               "Crypto model opening price uses first observed trade by +10s. Market anchor uses the latest trade by +10s.",
                               "Market entry uses archived book depth. Exit price stress applies an adverse move to the last public print.",
                               "Cadence is applied before account admission. Private signal, pause and fill logs are outside this replay."])
    write_json(ROOT / "okx_cross_bot_overlap_nov21.json", report)
    print(json.dumps({label: {key: val for key, val in score.items() if key != "closed"}
                      for label, score in results.items()}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("capture", "freeze", "score"))
    args = parser.parse_args()
    if args.stage == "capture":
        capture_crypto()
    elif args.stage == "freeze":
        freeze_crypto()
    else:
        score_joint()
