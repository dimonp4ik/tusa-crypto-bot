"""Test prior QQQ and SPY mark context on existing stock entry quotes."""

import json
from functools import partial
from pathlib import Path

from okx_xperp_multiday_capture_nov20 import file_hash
from stock_liquid_quote_routes_nov21 import DATA as PRIOR, ROOT, limit_bursts, score, write_json
from stock_quote_continuation_holdout_nov21 import DATA as UNSEEN


MARKS = Path("../tusa stocks/reports/mark_context_prospective_nov18")
INDEXES = ("QQQUSDT", "SPYUSDT")
WINDOWS = {"observed": PRIOR, "forward": UNSEEN}
MODES = ("broad_alignment", "veto_both_opposed")


def index_marks(bucket):
    path = MARKS / f"bucket_{bucket - 900}.json"
    if not path.exists():
        return None
    snapshot = json.loads(path.read_bytes())
    if snapshot["bucket"] != bucket - 900 or snapshot["profile"] != "stock_liquid_mark_context_v1":
        raise ValueError("benchmark snapshot did not match requested prior window")
    trends = {}
    for row in snapshot["observations"]:
        if row["symbol"] not in INDEXES:
            continue
        frame = row.get("closed_frame")
        if not frame or len(frame["time"]) < 17 or row["payload"]["series_kind"] != "mark":
            continue
        if frame["time"][-17:] != list(range(bucket - 18*900, bucket - 900, 900)):
            raise ValueError("benchmark close series has a gap or future observation")
        trends[row["symbol"]] = frame["close"][-1] / frame["close"][-17] - 1
    if set(trends) != set(INDEXES) or not snapshot["completed_at"]:
        return None
    return dict(trends=trends, complete_at=snapshot["completed_at"], sha256=file_hash(path))


def candidate_context(plans):
    by_bucket = {}
    result = {}
    for plan in plans.values():
        for rows in plan["cases"].values():
            for row in rows:
                if row["route"] != "aligned_continuation":
                    continue
                bucket = row["epoch"]
                if bucket not in by_bucket:
                    by_bucket[bucket] = index_marks(bucket)
                data = by_bucket[bucket]
                key = row["symbol"], row["entry_ms"]
                if key in result:
                    raise ValueError("duplication between the observer windows")
                if data is None or data["complete_at"] * 1000 >= row["entry_ms"]:
                    result[key] = None
                else:
                    sign = 1 if row["direction"] == "LONG" else -1
                    result[key] = {s: sign * trend for s, trend in data["trends"].items()}
    return result, by_bucket


def apply_index_filter(proposals, *, mode, evidence):
    selected = []
    blocked = []
    for row in proposals:
        data = evidence[row["symbol"], row["entry_ms"]]
        reject = False
        if data is not None:
            qqq, spy = (data[s] for s in INDEXES)
            reject = ((qqq < .0025 or spy < 0.) if mode == "broad_alignment"
                      else qqq < -.0025 and spy < -.001)
        if reject:
            blocked.append(dict(row, entry_status=mode))
        else:
            selected.append(row)
    return [*limit_bursts(selected, max_same_side=2), *blocked]


def main():
    plans = {label: json.loads((where / "plan.json").read_bytes()) for label, where in WINDOWS.items()}
    contexts, snapshots = candidate_context(plans)
    comparisons = {mode: partial(apply_index_filter, mode=mode, evidence=contexts) for mode in MODES}
    comparisons["regular_stock_cadence"] = lambda rows: limit_bursts(
        (r for r in rows if r["phase"] == "regular"), max_same_side=2)
    evidence = dict(contexts=[dict(symbol=s, entry_ms=t, signed_returns=values)
                              for (s, t), values in contexts.items()],
                    prior_snapshots=snapshots,
                    plans={label: file_hash(where / "plan.json") for label, where in WINDOWS.items()},
                    context_missing=sum(values is None for values in contexts.values()),
                    availability="Previous 15m observer window must have finished before the evaluated quote.",
                    benchmark="X-Perp mark series for QQQ and SPY. Agreement uses the last 16 closed bars.",
                    context_missing_uses_baseline=True)
    write_json(ROOT / "stock_quote_index_overlay_nov21_evidence.json", evidence)
    for label, data in WINDOWS.items():
        score(stress=True, data=data, stem=f"stock_quote_index_overlay_nov21_{label}",
              extra_variants=comparisons)


if __name__ == "__main__":
    main()
