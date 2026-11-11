"""Append verified public candles without truncating indicator warm-up history."""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from pathlib import Path

FIELDS = ("time", "open", "high", "low", "close", "volume")


def load(item: dict) -> dict:
    raw = Path(item["path"]).read_bytes()
    if hashlib.sha256(raw).hexdigest() != item["sha256"]:
        raise ValueError("source hash mismatch")
    data = pickle.loads(raw)
    times = list(data["time"])
    if not times or any(len(data[key]) != len(times) for key in FIELDS):
        raise ValueError("empty or unequal candle arrays")
    if any(b - a != 900 for a, b in zip(times, times[1:])):
        raise ValueError("non-contiguous 15-minute history")
    return data


def merge(old: dict, new: dict) -> tuple[dict, dict]:
    left = {int(t): i for i, t in enumerate(old["time"])}
    right = {int(t): i for i, t in enumerate(new["time"])}
    common = left.keys() & right.keys()
    if not common:
        raise ValueError("no overlapping candles")
    for t in common:
        if any(old[k][left[t]] != new[k][right[t]] for k in FIELDS):
            raise ValueError(f"overlap revised at {t}")
    added = [t for t in new["time"] if t > old["time"][-1]]
    if not added or added[0] != old["time"][-1] + 900:
        raise ValueError("no contiguous new tail")
    merged = {k: list(old[k]) + [new[k][right[t]] for t in added] for k in FIELDS}
    return merged, {"overlap_bars": len(common), "new_bars": len(added)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("old", type=Path)
    parser.add_argument("new", type=Path)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    old = json.loads(args.old.read_text(encoding="utf-8"))
    new = json.loads(args.new.read_text(encoding="utf-8"))
    placeholders = {s: v for s, v in old["symbols"].items()
                    if v.get("status") == "NO_INSTRUMENT" and "path" not in v
                    and s not in new["symbols"]}
    if old["symbols"].keys() - placeholders.keys() != new["symbols"].keys():
        raise ValueError("symbol universe changed")
    # Validate every source before producing an output directory.
    prepared = {}
    for symbol, item in old["symbols"].items():
        if symbol in placeholders:
            continue
        fresh = new["symbols"][symbol]
        if item["instrument"] != fresh["instrument"]:
            raise ValueError(f"instrument changed: {symbol}")
        prepared[symbol] = merge(load(item), load(fresh))
    args.out.mkdir(parents=True, exist_ok=False)
    report = dict(new, status="VERIFIED_APPEND_ONLY_PUBLIC_DATA", symbols=dict(placeholders),
                  source_manifests={str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                                    for p in (args.old, args.new)})
    for symbol, (data, stats) in prepared.items():
        path = args.out / f"{symbol}_15min_current.pkl"
        raw = pickle.dumps(data)
        path.write_bytes(raw)
        report["symbols"][symbol] = dict(new["symbols"][symbol], **stats,
            bars=len(data["time"]), start=data["time"][0], end=data["time"][-1],
            path=str(path), sha256=hashlib.sha256(raw).hexdigest())
    (args.out / "manifest.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"symbols": len(prepared), "new_bars": sum(
        s[1]["new_bars"] for s in prepared.values()), "out": str(args.out)}))


if __name__ == "__main__":
    main()
