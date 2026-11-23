"""Replay alphabetical versus model ordering with serial publication requests."""

from collections import Counter, defaultdict, deque
from dataclasses import asdict, dataclass
from itertools import groupby

from config import VENUE_MAX_ENTRY_DELAY_SEC
from crypto_quote_priority_nov23 import COHORTS, freeze, outcome_tape, stock_overlap
from okx_crypto_risk_range_nov22 import merge_books
from okx_post_close_cash_nov23 import trade_difference
from okx_post_stop_reclaim_nov23 import with_actual_fill_path
from okx_recent_trade_repair_nov21 import load_repaired_tape
from okx_selective_single_target_nov23 import cash_summary, key
from okx_xperp_cash_tick_nov19 import CAPITAL, CLOSE_LAGS, FRACTION, LEVERAGE
from okx_xperp_multiday_capture_nov20 import book_files
from okx_xperp_multiday_replay_nov20 import ROOT, exit_books, file_hash, key_for, pull_samples, resolve_all
from src.execution_guard import SpreadLimitError, execution_quote_with_retry
from src.okx_trader import calc_contracts, round_to_tick
from src.publication_book import publication_book
from stock_liquid_quote_routes_nov21 import write_json


STEM = 'crypto_serial_priority_nov23'
STEP_MS = 50
MAX_PUBLICATIONS = 2
RANKINGS = ('alphabetical', 'model_quality')


@dataclass(frozen=True)
class Pace:
    ready_ms: int
    request_ms: int
    send_ms: int
    account_prepare_ms: int
    account_finish_ms: int

    def envelope(self, candidates):
        public = candidates * (2 * self.request_ms + 350) + MAX_PUBLICATIONS * self.send_ms
        account = MAX_PUBLICATIONS * (self.account_prepare_ms + 3 * self.request_ms + 800 + self.account_finish_ms)
        return min(self.ready_ms + public + account, VENUE_MAX_ENTRY_DELAY_SEC * 1000)


PACES = {
    'fast': Pace(11_000, 100, 250, 500, 500),
    'middle': Pace(11_500, 250, 750, 1_500, 1_000),
    'slow': Pace(12_500, 500, 1_500, 3_500, 2_000),
}


class MarketClock:
    def __init__(self, now, request_ms, quotes):
        self.now = now
        self.request_ms = request_ms
        self.quotes = quotes
        self.receipts = []

    def fetch(self):
        self.now += self.request_ms
        self.receipts.append(self.now)
        if self.now not in self.quotes:
            raise KeyError('requested quote outside the archive panel')
        return self.quotes[self.now]

    def sleep(self, seconds):
        self.now += round(seconds * 1000)


def account_entry(row, book, pace, publish_at, account_free_at, deadline_ms, spec, tick):
    clock = MarketClock(max(publish_at, account_free_at) + pace.account_prepare_ms, pace.request_ms, book)
    target1 = round_to_tick(row['entry'] + .5 * (row['entry'] - row['stop']), tick)
    sized = lambda price: calc_contracts(CAPITAL * FRACTION, LEVERAGE, price, spec)
    try:
        _, quote = execution_quote_with_retry(clock.fetch, row['direction'], None,
            clock_ms=lambda: clock.now, sleep=clock.sleep, deadline_ms=deadline_ms, size_for_price=sized)
        quote.validate_bracket(row['stop'], target1)
        result = dict(entry_status='eligible', reference_entry=quote.average)
    except SpreadLimitError:
        result = dict(entry_status='execution_spread')
    except (TypeError, ValueError, IndexError) as exc:
        result = dict(entry_status=str(exc))
    if any(book[stamp] is None for stamp in clock.receipts):
        raise ValueError('account quote was absent from the public archive')
    evidence = dict(entry_ms=clock.now, execution_attempts=clock.receipts, stock_reference_ready=False)
    return dict(row, **result, **evidence), clock.now + pace.account_finish_ms


def serial(rows, books, specs, ticks, pace, policy):
    publications = deque()
    account_free_at = 0
    for at, batch in groupby(sorted(rows, key=lambda row: row['epoch']), key=lambda row: row['epoch']):
        scan = at * 1000
        public_now = scan + pace.ready_ms
        deadline_ms = scan + VENUE_MAX_ENTRY_DELAY_SEC * 1000
        while publications and publications[0] < public_now - 3_600_000:
            publications.popleft()
        if policy == 'model_quality':
            queue = sorted(batch, key=lambda row: (-row['model_quality'], row['symbol']))
        else:
            queue = sorted(batch, key=lambda row: row['symbol'])
        for row in queue:
            if len(publications) >= MAX_PUBLICATIONS:
                yield dict(row, entry_status='publication_rate_limit', entry_ms=public_now)
                continue
            symbol = row['symbol']
            clock = MarketClock(public_now, pace.request_ms, books[symbol])
            wide, _, observed = publication_book(clock.fetch, clock=lambda: clock.now / 1000, sleep=clock.sleep)
            public_now = clock.now
            # A missing archival sample cannot establish an inconclusive live response.
            if any(stamp not in books[symbol] or books[symbol][stamp] is None for stamp in clock.receipts):
                raise ValueError('publication quote was absent from the public archive')
            if wide or public_now >= deadline_ms:
                yield dict(row, entry_status='spread_at_publish' if wide else 'publication_deadline',
                           entry_ms=public_now, publication_attempts=clock.receipts)
                continue
            public_now += pace.send_ms
            publications.append(public_now)
            planned, account_free_at = account_entry(row, books[symbol], pace, public_now, account_free_at,
                                                     deadline_ms, specs[symbol], ticks[symbol])
            yield dict(planned, published_ms=public_now, publication_attempts=clock.receipts,
                       publication_book_verified=observed is not None)


def entry_panels(rows, specs):
    windows = Counter(row['epoch'] for row in rows)
    archives = book_files(dict(rows=rows, specs=specs))
    points = defaultdict(set)
    for row in rows:
        at = row['epoch']
        needed = points[key_for(row['symbol'], at * 1000)]
        for pace in PACES.values():
            needed.update(at * 1000 + dt for dt in range(pace.ready_ms, pace.envelope(windows[at]) + STEP_MS, STEP_MS))
    jobs = []
    for archive_key, stamps in sorted(points.items()):
        day, symbol = archive_key.split(':', 1)
        record = archives[archive_key]
        if 'path' not in record:
            raise ValueError(f'archive book unavailable for {archive_key}')
        jobs.append((symbol, int(day), record, stamps))
    books, checks = pull_samples(jobs, stage='serial entry books')
    return books, checks, archives


def book_inputs(label, start, stop, folder):
    captured, raw, tape_checks, source = freeze(start, stop, folder)
    specs = captured['specs']
    ticks = {s: float(spec['tickSz']) for s, spec in specs.items()}
    print(label, len(raw), 'candidates', flush=True)
    books, entry_checks, archives = entry_panels(raw, specs)
    plans = {pace: {name: list(serial(raw, books, specs, ticks, value, name)) for name in RANKINGS}
             for pace, value in PACES.items()}
    paths_by_key = {key(row): row for pair in plans.values() for rows in pair.values()
                    for row in rows if row['entry_status'] == 'eligible'}
    qualified = list(paths_by_key.values())
    print(label, 'unique executable proposals', len(qualified), flush=True)
    trades, repairs = outcome_tape(captured, qualified, folder)
    checks = resolve_all(qualified, {'files': trades}, {'symbols': captured['references']}, ticks,
                         trade_loader=load_repaired_tape)
    if any(item.get('error') or item['mismatches'] for item in checks.values()):
        raise ValueError('public outcomes disagree with verified bars')
    for pair in plans.values():
        for rows in pair.values():
            for row in rows:
                if row['entry_status'] == 'eligible':
                    row['path'] = paths_by_key[key(row)]['path']

    stocks = []
    stock_evidence = None
    if label == 'nov21':
        stocks, stock_specs, stock_ticks, stock_books, stock_evidence = stock_overlap(start, stop)
        if specs.keys() & stock_specs.keys():
            raise ValueError('stock and crypto symbols overlap')
        specs.update(stock_specs)
        ticks.update(stock_ticks)
        merge_books(books, stock_books)
    more, exits = exit_books(qualified + stocks, {'books': archives}, specs, books)
    evidence = dict(model_capture_sha256=source, tape_checks=tape_checks, entry_quotes=entry_checks,
                    tape_repairs_sha256=repairs, outcome_reconciliation=checks, exit_quotes=exits, stock=stock_evidence,
                    extra_archive_sha256={label: archive['sha256'] for label, archive in more.items()})
    return plans, specs, ticks, books, archives | more, trades, stocks, evidence


def compare(label, start, stop, folder):
    plans, specs, ticks, books, archives, trades, stocks, evidence = book_inputs(label, start, stop, folder)
    filled = dict(repriced=[], extra_exit_checks={})
    cash = with_actual_fill_path(trades, archives, filled)
    scored = {}
    for pace, queue in plans.items():
        scored[pace] = {}
        for account, stock_rows in [('crypto', []), *([('shared', stocks)] if stocks else [])]:
            output = {}
            for lag in CLOSE_LAGS:
                cases = {}
                original = None
                for name, rows in queue.items():
                    result = cash(rows + stock_rows, specs, ticks, books, clock=60, lag=lag,
                                  setting='requested5', max_positions=12)
                    summary = cash_summary(result, {})
                    if name == 'alphabetical':
                        original = result
                    elif 'closed' in original and 'closed' in result:
                        summary['changes'] = trade_difference(original, result)
                    if 'closed' in result:
                        summary['positions'] = result['closed']
                    cases[name] = summary
                output[str(lag)] = cases
                print(label, pace, account, lag,
                      {name: (p.get('trades'), p.get('cash_pnl_usd'), p.get('reason')) for name, p in cases.items()},
                      flush=True)
            scored[pace][account] = output
    decisions = {pace: {name: [dict((k, v) for k, v in row.items() if k in (
        'symbol', 'epoch', 'direction', 'model_quality', 'entry_status', 'entry_ms', 'published_ms',
        'publication_attempts', 'publication_book_verified', 'execution_attempts')) for row in rows]
        for name, rows in pair.items()} for pace, pair in plans.items()}
    return dict(window_utc=[start, stop], market=evidence, compounding=filled, decisions=decisions, cash=scored)


def main():
    source = ROOT / 'crypto_quote_priority_nov23_shared.json'
    cases = {label: compare(label, start, stop, folder) for label, start, stop, folder in COHORTS}
    write_json(ROOT / f'{STEM}.json', dict(cases=cases, ranked_by=RANKINGS, pace_ms={k: asdict(v) for k, v in PACES.items()},
        reference_shared_report_sha256=file_hash(source), starting_capital_usd=CAPITAL,
        margin_fraction=FRACTION, leverage=LEVERAGE, max_publications_per_hour=MAX_PUBLICATIONS,
        live_activation_supported=False, limitations=[
            'All decisions before entry use only prior public order-book events and the frozen model.',
            'Production publication and account spread retry functions are reused with a synthetic clock.',
            'Publication consumes a slot even if its subsequent account entry is rejected.',
            'Scenario request, send and account durations are stresses, not measured historical latency.',
            'The simulated order uses the final execution-check quote. Private exchange acknowledgments and user losses are outside the model.',
            'The same account shares reservation and margin checks. Live signal-journal side and loss pauses are unavailable in this historical cohort.',
            'Two short calendar windows are not sufficient to establish monthly profitability.',
        ]))


if __name__ == '__main__':
    main()
