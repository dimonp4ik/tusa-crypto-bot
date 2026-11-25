"""Join shadow exposure choices with confirmed OKX closures in Railway logs."""

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import re


SHADOW = re.compile(
    r'factor exposure shadow version=btc_two_phase_v1 position=(\d+) signal=(\d+) '
    r'inst=([A-Z0-9_-]+) epoch=(\d+) occupied=(\d+) matching=(\d+) unknown=(\d+) '
    r'decision=(would_allow|would_block|unknown) basis=(\w+)'
)
CLOSURE = re.compile(
    r'exit receipt verified signal=(\d+) inst=([A-Z0-9_-]+) broker_pnl=(positive|negative|flat) '
    r'signal_status=\S+ tp1_filled=(True|False) position=(\d+)'
)


def summarize(records):
    shadows = defaultdict(set)
    closures = defaultdict(set)
    for row in records:
        message = row['message']
        shadow = SHADOW.search(message)
        closed = CLOSURE.search(message)
        if shadow:
            position, *fields = shadow.groups()
            shadows[position].add(tuple(fields))
        if closed:
            signal, inst, pnl, tp1, position = closed.groups()
            closures[position].add((signal, inst, pnl, tp1))
    statuses = defaultdict(Counter)
    bases = Counter()
    conflicts = 0
    for position, observations in shadows.items():
        receipts = closures.get(position, set())
        if len(observations) != 1 or len(receipts) > 1:
            conflicts += 1
            continue
        signal, inst, _epoch, _count, _matching, _missing, decision, basis = next(iter(observations))
        item = statuses[decision]
        item['observed'] += 1
        bases[basis] += 1
        if not receipts:
            item['without_verified_close'] += 1
            continue
        matched_signal, matched_instrument, pnl, tp1 = next(iter(receipts))
        if (matched_signal, matched_instrument) != (signal, inst):
            conflicts += 1
            item['without_verified_close'] += 1
            continue
        item[pnl] += 1
        item['tp1_filled'] += tp1 == 'True'
    return dict(decisions=statuses,bases=bases,conflicts=conflicts,
                verified_closures_without_observation=len(closures.keys()-shadows.keys()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for bot in ('crypto','stocks'):
        parser.add_argument('--'+bot,type=Path,required=True)
    options = parser.parse_args()
    bots = {}
    for bot in ('crypto','stocks'):
        rows = json.loads(getattr(options,bot).read_bytes())['deploy']
        bots[bot] = summarize(rows)
    print(json.dumps(dict(bots=bots,limitations=[
        'Railway input must cover the observation and closure windows. Missing receipts are not treated as losses.',
        'Only actual baseline fills are observed. Blocked candidates represent a counterfactual, with no account PnL estimate.',
        'Each service is joined by its own position, signal and instrument. No customer or broker order identifier is exposed.',
    ]),indent=2))


if __name__=='__main__':
    main()
