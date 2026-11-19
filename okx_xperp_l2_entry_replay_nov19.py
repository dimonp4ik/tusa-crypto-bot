"""Causal 50-level entry checks reconstructed from official OKX L2 history."""
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import hashlib
import heapq
import json
import math
from pathlib import Path
import tarfile
from datetime import datetime

from okx_xperp_l2_capture_nov19 import DAY, END, ROOT_L2, START, cohort
from src.execution_guard import SpreadLimitError, execution_quote
from src.okx_trader import calc_contracts
from us_cash_session_nov19 import ET, session_bounds


CAPITAL = 130.
MARGIN_FRACTION = .03
LEVERAGE = 10
LATENCIES_MS = (10_000, 70_000)
ATTEMPTS_MS = (0, 200, 800)
DEPTH = 50


class Book:
    def __init__(self):
        self.ask = {}
        self.bid = {}
        self.ts = None

    def apply(self, action, ts, bids, asks):
        if action == 'snapshot':
            self.ask.clear()
            self.bid.clear()
        elif action != 'update' or self.ts is None:
            raise ValueError('archive update has no preceding snapshot')
        if self.ts is not None and ts < self.ts:
            raise ValueError('nonmonotonic exchange timestamps')
        for book, rows in ((self.bid, bids), (self.ask, asks)):
            for row in rows:
                price, size = float(row[0]), float(row[1])
                if not math.isfinite(price) or not math.isfinite(size) or price <= 0 or size < 0:
                    raise ValueError('invalid exchange depth level')
                if size > 0:
                    book[price] = size
                else:
                    book.pop(price, None)
            # This channel publishes the best 400 levels on each side.
            if len(book) > 400:
                order = heapq.nlargest if book is self.bid else heapq.nsmallest
                retained = set(order(400, book))
                for price in book.keys() - retained:
                    del book[price]
        self.ts = ts

    def levels(self):
        return dict(ts=self.ts,
                    bids=[(p, self.bid[p]) for p in heapq.nlargest(DEPTH, self.bid)],
                    asks=[(p, self.ask[p]) for p in heapq.nsmallest(DEPTH, self.ask)])


def sample(events, instrument, points):
    """Freeze the state strictly before requested times, even on equal ms."""
    requested = iter(sorted(set(points)))
    moment = next(requested, None)
    values = {}
    book = Book()
    counts = Counter()
    first = None
    for event in events:
        if event['instId'] != instrument:
            raise ValueError('archive instrument mismatch')
        ts = int(event['ts'])
        if not START * 1000 <= ts < END * 1000:
            raise ValueError('archive event outside declared day')
        if book.ts is not None and ts < book.ts:
            raise ValueError('nonmonotonic exchange timestamps')
        while moment is not None and moment <= ts:
            values[moment] = book.levels() if book.ts is not None else None
            moment = next(requested, None)
        action = event['action']
        book.apply(action, ts, event['bids'], event['asks'])
        counts[action] += 1
        first = ts if first is None else first
    while moment is not None:
        values[moment] = book.levels() if book.ts is not None else None
        moment = next(requested, None)
    if not counts['snapshot'] or not counts['update']:
        raise ValueError('archive unavailable or missing updates')
    return values, dict(counts, first_ts=first, last_ts=book.ts)


def archive(path, instrument, expected_digest, points):
    with path.open('rb') as stream:
        sha = hashlib.file_digest(stream, 'sha256').hexdigest()
    if sha != expected_digest:
        raise ValueError('L2 archive checksum mismatch')
    member_name = f'{instrument}-L2orderbook-400lv-{DAY}.data'
    found = None
    with tarfile.open(path, 'r|gz') as files:
        for member in files:
            if member.name != member_name:
                continue
            if found is not None or not member.isfile() or not 0 < member.size <= 8 * 1024**3:
                raise ValueError('invalid or duplicate exact-instrument member')
            with files.extractfile(member) as stream:
                found = sample((json.loads(line) for line in stream), instrument, points)
    if found is None:
        raise ValueError('exact instrument member not in daily archive')
    return found


def attempt(book, stamp, side, spec, stop=None, target=None):
    if book is None:
        return dict(status='unknown_quote')
    try:
        quote = execution_quote(book, side, None, stamp,
            size_for_price=lambda price: calc_contracts(CAPITAL * MARGIN_FRACTION, LEVERAGE, price, spec))
        if stop is not None:
            quote.validate_bracket(stop, target)
    except SpreadLimitError as exc:
        return dict(status='wide', spread_bps=exc.measured * 10_000)
    except ValueError as exc:
        return dict(status=str(exc))
    return dict(status='eligible', average=quote.average, worst=quote.worst,
                contracts=quote.contracts, spread_bps=quote.spread_fraction * 10_000,
                slippage_bps=quote.slippage_fraction * 10_000, last_event_ms=book['ts'])


def cash_phase(stamp):
    bounds = session_bounds(stamp)
    if bounds is None:
        return 'non_cash_day'
    if bounds[0] <= stamp < bounds[1]:
        return 'regular'
    local = datetime.fromtimestamp(stamp, ET)
    minutes = local.hour * 60 + local.minute
    if 240 <= minutes < 570:
        return 'premarket'
    return 'postmarket' if 960 <= minutes < 1200 else 'other'


def entry_bracket(row):
    sign = 1 if row['direction'] == 'LONG' else -1
    entry = row['entry']
    if row['bot'] == 'crypto':
        return row['sl'], entry + 2 * (entry - row['sl'])
    return entry * (1 - sign * row['risk_pct']), entry * (1 + sign * row['target_pct'] / 100)


def evaluate(job):
    symbol, item, rows, spec = job
    points = {t * 1000 + dt for t in range(START, END, 900) for dt in LATENCIES_MS}
    for row in rows:
        points.update(int(row['entry_time'] * 1000) + dt + lag
                      for dt in LATENCIES_MS for lag in ATTEMPTS_MS)
    if 'path' not in item:
        return symbol, dict(error=item['error'])
    try:
        books, coverage = archive(Path(item['path']), item['instrument'], item['sha256'], points)
    except (OSError, ValueError, KeyError, tarfile.TarError) as exc:
        return symbol, dict(error=f'{type(exc).__name__}: {exc}')
    census = {}
    for dt in LATENCIES_MS:
        by_phase = {}
        for t in range(START, END, 900):
            phase = cash_phase(t)
            clock = t * 1000 + dt
            group = by_phase.setdefault(phase, Counter())
            group['windows'] += 1
            for side in ('LONG', 'SHORT'):
                q = attempt(books[clock], clock, side, spec)
                group[q['status']] += 1
        census[str(dt)] = by_phase
    proposals = []
    for row in rows:
        stop, target = entry_bracket(row)
        variants = {}
        for dt in LATENCIES_MS:
            checks = []
            for lag in ATTEMPTS_MS:
                clock = int(row['entry_time'] * 1000) + dt + lag
                result = attempt(books[clock], clock, row['direction'], spec, stop, target)
                checks.append(dict(offset_ms=dt+lag, **result))
                if result['status'] != 'wide':
                    break
            variants[str(dt)] = checks
        proposals.append(dict(bot=row['bot'], symbol=symbol, direction=row['direction'],
                              epoch=row['entry_time'], entry=row['entry'], stop=stop,
                              target=target, variants=variants))
    return symbol, dict(events=coverage, census=census, proposals=proposals)


def main():
    manifest_path = ROOT_L2 / 'manifest.json'
    manifest = json.loads(manifest_path.read_bytes())
    rows, ready, instruments = cohort()
    if set(manifest['files']) != set(instruments):
        raise ValueError('capture manifest misses declared instruments')
    specs = json.loads((ROOT_L2.parent/'shared_xperp_specs_nov14.json').read_bytes())
    jobs = [(s, manifest['files'][s], [r for r in rows if r['symbol'] == s], specs[s]) for s in instruments]
    records = {}
    with ProcessPoolExecutor(max_workers=3) as pool:
        for symbol, value in pool.map(evaluate, jobs):
            records[symbol] = value
            print(symbol, value.get('error') or value['events'], flush=True)
    totals = {bot: {str(dt): Counter() for dt in LATENCIES_MS} for bot in ('crypto', 'stocks')}
    for value in records.values():
        for row in value.get('proposals', []):
            row['stock_reference_ready'] = (row['symbol'], row['epoch']) in ready if row['bot']=='stocks' else None
            for latency, attempt_log in row['variants'].items():
                counts = totals[row['bot']][latency]
                counts['proposals'] += 1
                counts[attempt_log[-1]['status']] += 1
                counts['rescued_by_retry'] += len(attempt_log) > 1 and attempt_log[-1]['status'] == 'eligible'
    report = dict(status='OFFICIAL_PUBLIC_L2_PILOT', day=DAY, manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
                  proposals_requested=len(rows), first_checks_ms=LATENCIES_MS, retries_ms=ATTEMPTS_MS,
                  proposals_missing={bot: sum(r['bot']==bot and 'error' in records[r['symbol']] for r in rows)
                                     for bot in totals},
                  totals=totals, instruments=records, objective_verified=False, live_activation_supported=False,
                  limitations=[
                      'Exact X-Perp files and exchange event timestamps. No private receipt/fill or network latency reconstruction.',
                      'Archives omit sequence IDs and checksums, so undisclosed gaps between updates cannot be ruled out.',
                      'Fixed past proposals, one UTC day. No publication-spread filter, account state, or exit/PnL replay.',
                      'Three illustrative checks at fixed times. The 70-second timing is a stock scan proxy, not recovered timestamps.',
                  ])
    target = ROOT_L2.parent/'okx_xperp_l2_entry_replay_nov19.json'
    target.write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps(totals), flush=True)


if __name__ == '__main__':
    main()
