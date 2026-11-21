"""Fetch bounded public prints where the UTC+8 daily catalogue has a gap."""

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import time

import requests

from okx_xperp_multiday_capture_nov20 import file_hash, local_trade_days
from okx_xperp_tick_path_nov19 import Trade, load_trade_tape


ENDPOINT = "https://www.okx.com/api/v5/market/history-trades"
PAGE_LIMIT = 500
DAY_MS = 86_400_000


def validated_print(row, instrument):
    if row["instId"] != instrument:
        raise ValueError("wrong instrument in market history")
    fill = Trade(int(row["ts"]), int(row["tradeId"]), float(row["px"]), float(row["sz"]))
    if fill.ms <= 0 or fill.identity <= 0 or not all(
            math.isfinite(v) and v > 0 for v in (fill.price, fill.contracts)):
        raise ValueError("invalid public print")
    return fill


def fetch_interval(instrument, start_ms, end_ms, folder):
    if not 0 < end_ms - start_ms <= DAY_MS:
        raise ValueError("unbounded public trade interval")
    path = folder / f"{instrument}_{start_ms}_{end_ms}.json"
    if path.exists():
        return dict(path=str(path), sha256=file_hash(path))
    fills = {}
    with requests.Session() as session:
        cursor = end_ms
        by = 2
        for _ in range(PAGE_LIMIT):
            params = dict(instId=instrument, type=by, after=cursor, limit=100)
            for attempt in range(3):
                response = session.get(ENDPOINT, params=params, timeout=12, allow_redirects=False)
                if response.status_code not in (429, 502, 503, 504):
                    break
                time.sleep(attempt + 1)
            response.raise_for_status()
            payload = response.json()
            rows = payload.get("data")
            if payload.get("code") != "0" or not isinstance(rows, list) or len(rows) > 100:
                raise ValueError("malformed market history response")
            if not rows:
                raise ValueError("public history exhausted before reaching the start")
            page = [validated_print(row, instrument) for row in rows]
            if any(a.identity <= b.identity or a.ms < b.ms for a, b in zip(page, page[1:])):
                raise ValueError("out-of-order market history page")
            if by == 1 and page[0].identity >= cursor or by == 2 and page[0].ms > cursor:
                raise ValueError("market history cursor did not advance")
            for row, fill in zip(rows, page):
                if start_ms <= fill.ms < end_ms:
                    if fill.identity in fills:
                        raise ValueError("repeated market history identity")
                    fills[fill.identity] = row
            if page[-1].ms < start_ms:
                break
            cursor = page[-1].identity
            by = 1
            time.sleep(.3)
        else:
            raise ValueError("market history exceeded fixed page limit")
    payload = dict(instrument=instrument, start_ms=start_ms, end_ms=end_ms,
                   endpoint=ENDPOINT, rows=list(fills.values()))
    path.write_text(json.dumps(payload, allow_nan=False), encoding="utf-8", newline="\n")
    return dict(path=str(path), sha256=file_hash(path))


def repair_recent_days(captured, rows, *, hold_seconds, folder):
    bounds = {}
    for row in rows:
        symbol, start = row["symbol"], row["epoch"]
        end = start + hold_seconds
        claimed = captured[symbol]["days"]
        for midnight, local_day in local_trade_days(start, end).items():
            if local_day in claimed:
                continue
            left, right = max(midnight, start * 1000), min(midnight + DAY_MS, end * 1000)
            previous = bounds.get((symbol, local_day), (left, right))
            bounds[(symbol, local_day)] = min(previous[0], left), max(previous[1], right)
    folder.mkdir(parents=True, exist_ok=True)

    def capture(job):
        (symbol, day), (start, end) = job
        market = captured[symbol]["instrument"]
        result = fetch_interval(market, start, end, folder)
        return symbol, day, dict(result, start_ms=start, end_ms=end)

    evidence = {}
    with ThreadPoolExecutor(max_workers=2) as pool:
        for symbol, day, item in pool.map(capture, sorted(bounds.items())):
            evidence.setdefault(symbol, {})[day] = item
            print("public trade interval repaired", symbol, day, flush=True)
    return evidence


def load_repaired_tape(item):
    if item["complete"]:
        return load_trade_tape(item)
    if not item.get("repaired"):
        raise ValueError("trade archive incomplete with no repaired intervals")
    archive = load_trade_tape(dict(item, complete=True)) if item["days"] else []
    merged = {t.identity: t for t in archive}
    for entry in item["repaired"].values():
        path = Path(entry["path"])
        if file_hash(path) != entry["sha256"]:
            raise ValueError("repaired trade digest mismatch")
        batch = json.loads(path.read_bytes())
        if (batch["instrument"] != item["instrument"] or batch["endpoint"] != ENDPOINT
                or any(batch[key] != entry[key] for key in ("start_ms", "end_ms"))):
            raise ValueError("repaired trade metadata mismatch")
        for row in batch["rows"]:
            fill = validated_print(row, item["instrument"])
            if (not entry["start_ms"] <= fill.ms < entry["end_ms"]
                    or merged.setdefault(fill.identity, fill) != fill):
                raise ValueError("invalid repaired trade overlap")
    tape = sorted(merged.values())
    if not tape or any(b.identity <= a.identity for a, b in zip(tape, tape[1:])):
        raise ValueError("combined tape order could not be confirmed")
    return tape
