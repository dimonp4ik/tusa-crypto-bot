"""Test delayed market entry for observed stock signals that meet the original levels."""

from bisect import bisect_left, bisect_right
from collections import Counter, defaultdict
import json

from okx_selective_single_target_nov23 import cash_summary, key
from okx_xperp_cash_tick_nov19 import CAPITAL, CLOSE_LAGS, FRACTION, LEVERAGE, close_quote, simulate
from okx_xperp_multiday_capture_nov20 import book_files
from okx_xperp_multiday_replay_nov20 import ROOT, exit_books, file_hash, key_for, pull_samples
from okx_xperp_tick_path_nov19 import resolve_ticks
import stock_native_geometry_match_nov22 as native
from src.execution_guard import SpreadLimitError, execution_quote
from src.okx_trader import calc_contracts, round_to_tick


FAST = (0, 200, 800)
DELAYED = (10_000, 25_000, 55_000, 90_000, 130_000, 145_000)
POLICIES = {'current': (0, False), 'retry_60s': (60_000, False),
            'retry_before_deadline': (145_000, False),
            'retry_wide_or_stale': (145_000, True)}
MINIMUM_RETRY_RR = 1.5


def sample_points(row):
    return [row['entry_ms'] + delay for delay in FAST + DELAYED
            if row['entry_ms'] + delay < (row['epoch'] + 180) * 1000]


def prepare(plan):
    _, specs, ticks, _, tapes, notes, checks = native.prepare_cases(plan)
    by_key = {key(dict(r, bot='stocks')): r for r in notes}
    prepared = []
    for row in plan['cases']['contract_observer']:
        note = by_key[key(row)]
        if note.get('ticker_backing') and note.get('paper_path', {}).get('known'):
            prepared.append(dict(row, entry=note['ticker_reference'], stop=note['fixed_stop'],
                                 target=note['fixed_target'], observed_gate=note['status'],
                                 stock_reference_ready=True))
    admitted = list(native.limit_bursts(prepared, max_same_side=2))
    accepted = [row for row in admitted if row.get('entry_status') != 'publication_rate_limit']
    return accepted, specs, ticks, tapes, checks, Counter(x['status'] for x in notes), len(admitted)-len(accepted)


def collect(rows, specs):
    archives = book_files(dict(rows=rows, specs=specs))
    points = defaultdict(set)
    for row in rows:
        for at in sample_points(row):
            points[key_for(row['symbol'], at)].add(at)
    jobs = []
    for archive_key, at in sorted(points.items()):
        day, symbol = archive_key.split(':', 1)
        item = archives.get(archive_key, {})
        if 'path' in item:
            jobs.append((symbol, int(day), item, sorted(at)))
    evidence, checked = pull_samples(jobs, stage='archived native entry books')
    return evidence, checked, archives


def stop_waiting(row, times, tape, received_at, retry_at, tick):
    side = 1 if row['direction'] == 'LONG' else -1
    stop = side * round_to_tick(row['stop'], tick)
    target = side * round_to_tick(row['target'], tick)
    backing = side * float(row['native_candidate']['backing_level'])
    before = bisect_left(times, received_at)
    through = bisect_right(times, retry_at)
    for trade in tape[before:through]:
        price = side * trade.price
        if price <= stop:
            return 'stop_touched_while_waiting'
        if price >= target:
            return 'target_touched_while_waiting'
        if price < backing:
            return 'backing_broken_while_waiting'
    return None


def pick(row, spec, tick, quotes, tapes, policy, backing_allows):
    deadline, retry_stale = policy
    side = 1 if row['direction'] == 'LONG' else -1
    origin = row['entry_ms']
    times, tape = tapes[row['symbol']]
    attempts = []
    reason = 'no_eligible_snapshot'
    for at in sample_points(row):
        delta = at - origin
        if delta > deadline and delta not in FAST:
            break
        if delta not in FAST:
            touched = stop_waiting(row, times, tape, origin, at, tick)
            if touched:
                reason = touched
                break
        book = quotes.get(at)
        if not book:
            reason = 'missing_archived_entry_snapshot'
            break
        try:
            quote = execution_quote(book, row['direction'], None, at, size_for_price=lambda px:
                                    calc_contracts(CAPITAL * FRACTION, LEVERAGE, px, spec))
            quote.validate_bracket(row['stop'], row['target'])
        except SpreadLimitError as exc:
            reason = 'wide'
            attempts.append(dict(at=at, delay_ms=delta, status=reason, spread_bps=exc.measured*10_000))
            continue
        except (ValueError, TypeError, KeyError, IndexError) as exc:
            reason = str(exc)
            if retry_stale and reason == 'stale order book':
                attempts.append(dict(at=at, delay_ms=delta, status=reason, book_age_ms=at-float(book['ts'])))
                continue
            break
        rr = side * (row['target'] - quote.worst) / (side * (quote.worst - row['stop']))
        if not all(backing_allows(row['native_candidate'], price) for price in (quote.best, quote.worst)):
            reason = 'book_broke_backing'
            break
        if delta not in FAST and rr < MINIMUM_RETRY_RR:
            reason = 'room_to_target_too_small'
            break
        attempts.append(dict(at=at, delay_ms=delta, status='eligible', spread_bps=quote.spread_fraction*10_000,
                             executable_rr=rr, avg=quote.average, contracts=quote.contracts))
        updated = dict(row, entry_status='eligible', entry_ms=at, reference_entry=quote.average,
                       retry_ms=delta, entry_attempts=attempts)
        updated['path'] = {60: resolve_ticks(tape, times, signal=updated, entered_at=at,
                             entry_px=quote.average, tick_size=tick, amend_seconds=60)}
        return updated
    return dict(row, entry_status=reason, entry_attempts=attempts)


def cash_results(policies, specs, ticks, books, tapes):
    def stressed_exit(bps):
        def close(symbol, at, direction, size):
            book = books[symbol].get(at)
            if book is not None:
                return close_quote(book, at, direction, size)
            price = native.price_before(tapes, symbol, at, inclusive=True)
            if price is None:
                raise ValueError('no recent exchange print or archived book at exit')
            return price * (1 - (1 if direction == 'LONG' else -1) * bps/10_000)
        return close

    results = {}
    for label, rows in policies.items():
        traded_after_fast = {key(row) for row in rows if row.get('retry_ms', 0) not in FAST}
        results[label] = dict(
            statuses=Counter(r['entry_status'] for r in rows),
            modes={f'{lag}ms_{mode}': cash_summary(simulate(
                rows, specs, ticks, books, clock=60, lag=lag, setting='requested5', exit_price=price),
                traded_after_fast) for lag in CLOSE_LAGS for mode,price in
                (('exact', None), ('stress_5bps_if_no_book', stressed_exit(5)),
                 ('stress_30bps_if_no_book', stressed_exit(30)))},
        )
    return results


def main():
    plan = native.freeze()
    rows, specs, ticks, tapes, tape_checks, original_statuses, capped = prepare(plan)
    books, entries, archive = collect(rows, specs)
    # Apply every policy to the same entry-archive coverage. Report unavailable
    # instrument-days separately, so missing quotes do not become failed entries.
    covered, missing = [], []
    for row in rows:
        pool = covered if 'path' in archive.get(key_for(row['symbol'], row['entry_ms']), {}) else missing
        pool.append(row)
    backing_allows = native.native_module().backing_level_allows_entry
    variants = {label: [pick(r, specs[r['symbol']], ticks[r['symbol']], books[r['symbol']], tapes,
                             policy, backing_allows) for r in covered] for label,policy in POLICIES.items()}
    exit_rows = [row for rows in variants.values() for row in rows if row['entry_status'] == 'eligible']
    after, exits = exit_books(exit_rows, {'books': archive}, specs, books)
    results = cash_results(variants, specs, ticks, books, tapes)
    summaries = {label: {'entry_statuses': values['statuses'], 'cash': {mode:
                         (row.get('cash_pnl_usd'), row.get('trades'), row.get('reason'))
                         for mode,row in values['modes'].items()}} for label,values in results.items()}
    print('qualified signals with entry archive', len(covered), 'without', len(missing), flush=True)
    print(json.dumps(summaries, indent=2), flush=True)
    report = dict(initial_capital=CAPITAL, margin=FRACTION, leverage=LEVERAGE,
                  policies={k: dict(max_delay_ms=v[0], retries_stale=v[1]) for k,v in POLICIES.items()},
                  fast_offsets_ms=FAST, delayed_offsets_ms=DELAYED,
                  source_hashes={name: file_hash(native.DATA / name) for name in
                    ('plan.json','capture.json','trade_repair.json')},
                  production_observer_statuses=original_statuses, publication_cap_rejections=capped,
                  entries_by_phase=Counter(r['phase'] for r in covered),
                  missing_entry_archive=[key(r) for r in missing],
                  archive_sha256={k:v['sha256'] for k,v in (archive|after).items() if 'sha256' in v},
                  entry_book_checks=entries, tape_checks=tape_checks, exit_book_checks=exits,
                  attempts={policy: [{k:r[k] for k in ('entry_ms','symbol','epoch','phase','direction',
                      'entry_status','entry_attempts','observed_gate')} for r in rows] for policy,rows in variants.items()},
                  results=results, live_activation_supported=False, limitations=[
                      'Native observer signals are cloned before any variant compares exits.',
                      'The publication limit is applied once using original receipt times. It is an approximation of live publication.',
                      'Delayed variants keep the existing spread, depth, price and backing checks and cannot enter after the original deadline.',
                      'The stale-book retry was added after seeing stale initial books and is exploratory.',
                      'Market price and size are inferred from timestamped OKX public depth. Some instrument-days have no archive.',
                      'Stock-only account replay, with the common 12-position and 90% capacity rules; crypto fills are outside this cohort.',
                      'A mode using trade-print stress is labeled explicitly and fails when it has neither an exit book nor a recent print.',
                  ])
    path = ROOT / 'stock_native_spread_replay_nov23.json'
    path.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8', newline='\n')


if __name__ == '__main__':
    main()
