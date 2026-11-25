"""Group admission using completed, priceable shadow outcomes, measured in initial risk."""

from collections import defaultdict, deque
from dataclasses import dataclass
from heapq import heappop, heappush
import math


HOUR_MS = 3_600_000


@dataclass(frozen=True)
class Policy:
    hours: int
    observations: int
    maximum_sum_r: float
    partition: str = 'side'


POLICIES = {
    'side_24h_3': Policy(24, 3, -1.5),
    'side_48h_5': Policy(48, 5, -2.),
    'side_market_48h_3': Policy(48, 3, -1.5, 'market'),
}


def filter_events(rows, outcomes, policy, *, identify, entrance, group):
    """Each shadow close is visible only after its recorded settlement time."""
    tail = defaultdict(deque)
    pending = []
    rejected = {}
    for sequence, row in enumerate(sorted(rows, key=lambda r: (entrance(r), identify(r)))):
        now = entrance(row)
        while pending and pending[0][0] < now:
            settled, _, scope, result = heappop(pending)
            tail[scope].append((settled, result))
        scope = group(row, policy.partition)
        if scope is None:
            continue
        history = tail[scope]
        while history and history[0][0] <= now - policy.hours*HOUR_MS:
            history.popleft()
        recent = tuple(r for _, r in list(history)[-policy.observations:])
        if len(recent) >= policy.observations and math.fsum(recent) <= policy.maximum_sum_r:
            rejected[identify(row)] = dict(recent_r=recent, sum_r=math.fsum(recent))
        evidence = outcomes.get(identify(row))
        if evidence is None:
            continue
        closed_ms, result_r = evidence
        if closed_ms <= now or not math.isfinite(result_r):
            raise ValueError('shadow settlement precedes its entry or is malformed')
        heappush(pending, (closed_ms, sequence, scope, result_r))
    return rejected
