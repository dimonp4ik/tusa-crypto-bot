"""Causal shadow-outcome veto for research; not enabled in live trading."""
from collections import defaultdict, deque
import heapq
import math


def filter_rows(rows, *, fields, days, minimum, return_field="signed_price_return"):
    """Pause a group with negative completed shadow price returns.

    Rejected signals still supply later observable outcomes. Equal-time exits
    are unavailable to entries at that timestamp. No account sizing is changed.
    """
    if not fields or days <= 0 or minimum < 1:
        raise ValueError("invalid health window")
    ordered = sorted(rows, key=lambda r: (float(r["entry_time"]), str(r["symbol"])))
    histories = defaultdict(deque)
    pending = []
    output = []
    rejected = 0
    for sequence, row in enumerate(ordered):
        now, end, value = (float(row[k]) for k in ("entry_time", "exit_time", return_field))
        if not all(math.isfinite(x) for x in (now, end, value)) or end < now:
            raise ValueError("invalid shadow outcome")
        while pending and pending[0][0] < now:
            stamp, _, group, observed = heapq.heappop(pending)
            histories[group].append((stamp, observed))
        key = tuple(row[field] for field in fields)
        history = histories[key]
        while history and history[0][0] < now - days * 86400:
            history.popleft()
        if len(history) >= minimum and sum(v for _, v in history) < 0:
            rejected += 1
        else:
            output.append(row)
        heapq.heappush(pending, (end, sequence, key, value))
    return output, {"input": len(ordered), "kept": len(output), "rejected": rejected}
