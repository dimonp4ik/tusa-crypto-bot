"""Rescore the frozen native stock cohort after the missing books appeared."""

from collections import Counter, defaultdict

from okx_selective_single_target_nov23 import key
from okx_unbankable_tp1_nov21 import read_json
from okx_xperp_multiday_capture_nov20 import book_files
from okx_xperp_multiday_replay_nov20 import ROOT, exit_books, file_hash, key_for, pull_samples
from okx_xperp_tick_path_nov19 import resolve_ticks
from src.execution_guard import MAX_ENTRY_SPREAD, SpreadLimitError, execution_quote
from src.okx_trader import calc_contracts
import stock_native_spread_replay_nov23 as original
from stock_liquid_quote_routes_nov21 import write_json


STEM = 'stock_spread_recovery_nov23'
GRID_MS = 5_000
STOCK_DEADLINE_MS = 180_000
DELAY_POLICIES = {
    'retry_every_five_seconds': (1, .0005, 1.5),
    'two_fresh_quotes_five_seconds_apart': (2, .0005, 1.5),
    'two_quotes_up_to_10bp_with_2r': (2, .001, 2.),
    'two_quotes_up_to_15bp_with_2r': (2, .0015, 2.),
}


def repeated_dates(row):
    origin = row['entry_ms']
    deadline = row['epoch'] * 1000 + STOCK_DEADLINE_MS
    return range(origin + GRID_MS, deadline, GRID_MS)


def collect(rows, specs):
    archives = book_files(dict(rows=rows, specs=specs))
    points = defaultdict(set)
    for row in rows:
        for at in set(original.sample_points(row)) | set(repeated_dates(row)):
            points[key_for(row['symbol'], at)].add(at)
    jobs = []
    for name, stamps in sorted(points.items()):
        archive = archives[name]
        if 'path' not in archive:
            continue
        day, symbol = name.split(':', 1)
        jobs.append((symbol, int(day), archive, stamps))
    snapshots, checks = pull_samples(jobs, stage='refreshed stock entry archives')
    return snapshots, checks, archives


def recheck(row, first, spec, tick, snapshots, tapes, backing_allows, consecutive, *,
            max_spread=MAX_ENTRY_SPREAD, minimum_reward_r=original.MINIMUM_RETRY_RR):
    if first['entry_status'] not in ('wide', 'stale order book'):
        return first
    symbol, direction, origin = row['symbol'], row['direction'], row['entry_ms']
    side = 1 if direction == 'LONG' else -1
    times, trades = tapes[symbol]
    attempts = list(first['entry_attempts'])
    size = lambda price: calc_contracts(original.CAPITAL * original.FRACTION, original.LEVERAGE, price, spec)
    previous_exchange_ms = None
    accepted_checks = 0
    reason = first['entry_status']
    for at in repeated_dates(row):
        stopped = original.stop_waiting(row, times, trades, origin, at - 1, tick)
        if stopped:
            reason = stopped
            break
        book = snapshots.get(at)
        if book is None:
            reason = 'missing_archived_entry_snapshot'
            break
        try:
            quote = execution_quote(book, direction, None, at, size_for_price=size, max_spread=max_spread)
            quote.validate_bracket(row['stop'], row['target'])
        except SpreadLimitError as exc:
            reason = 'wide'
            attempts.append(dict(at=at, status=reason, spread_bps=exc.measured * 10_000))
            accepted_checks = 0
            previous_exchange_ms = None
            continue
        except (TypeError, ValueError, KeyError, IndexError) as exc:
            reason = str(exc)
            attempts.append(dict(at=at, status=reason))
            if reason == 'stale order book':
                accepted_checks = 0
                previous_exchange_ms = None
                continue
            break
        if not all(backing_allows(row['native_candidate'], px) for px in (quote.best, quote.worst)):
            reason = 'book_broke_backing'
            break
        reward_r = side * (row['target'] - quote.worst) / (side * (quote.worst - row['stop']))
        if reward_r < minimum_reward_r:
            reason = 'room_to_target_too_small'
            break
        exchange_ms = int(book['ts'])
        if previous_exchange_ms is None or exchange_ms > previous_exchange_ms:
            accepted_checks += 1
        else:
            accepted_checks = 1
        previous_exchange_ms = exchange_ms
        reason = 'eligible' if accepted_checks >= consecutive else 'waiting_for_second_quote'
        attempts.append(dict(at=at, status=reason, spread_bps=quote.spread_fraction * 10_000,
                             executable_rr=reward_r, exchange_ms=exchange_ms))
        if reason == 'eligible':
            entered = dict(row, entry_status=reason, entry_ms=at, reference_entry=quote.average,
                           retry_ms=at - origin, entry_attempts=attempts, research_entry_spread=max_spread,
                           research_minimum_reward_r=minimum_reward_r,
                           research_backing_level=float(row['native_candidate']['backing_level']))
            entered['path'] = {60: resolve_ticks(trades, times, signal=entered, entered_at=at,
                                                 entry_px=quote.average, tick_size=tick, amend_seconds=60)}
            return entered
    return dict(row, entry_status=reason, entry_attempts=attempts)


def main():
    plan = original.native.freeze()
    rows, specs, ticks, tapes, tape_checks, _, capped = original.prepare(plan)
    print('Comparable native stock signals', len(rows), 'after publication caps', capped, flush=True)
    books, entry_checks, archives = collect(rows, specs)
    known = [row for row in rows if 'path' in archives.get(key_for(row['symbol'], row['entry_ms']), {})]
    missing = [key(row) for row in rows if row not in known]
    backing_allows = original.native.native_module().backing_level_allows_entry
    picked = {name: [original.pick(row, specs[row['symbol']], ticks[row['symbol']], books[row['symbol']], tapes,
                                   policy, backing_allows) for row in known] for name, policy in original.POLICIES.items()}
    for name, (count, max_spread, minimum_reward_r) in DELAY_POLICIES.items():
        picked[name] = [recheck(row, first, specs[row['symbol']], ticks[row['symbol']], books[row['symbol']],
                                tapes, backing_allows, count, max_spread=max_spread, minimum_reward_r=minimum_reward_r)
                        for row, first in zip(known, picked['current'], strict=True)]
    changed = {key(row): row for group in picked.values() for row in group if row['entry_status'] == 'eligible'}
    downloaded, exit_checks = exit_books(list(changed.values()), {'books': archives}, specs, books)
    scores = original.cash_results(picked, specs, ticks, books, tapes)
    previous_path = ROOT / 'stock_native_spread_replay_nov23.json'
    prior = read_json(previous_path)
    old_missing = {tuple(values) for values in prior['missing_entry_archive']}
    newly_available = old_missing.intersection(key(row) for row in known)
    for name, score in scores.items():
        print(name, score['statuses'], {test: (value.get('trades'), value.get('cash_pnl_usd'), value.get('reason'))
                    for test, value in score['modes'].items() if 'exact' in test}, flush=True)
    report = dict(older_results_sha256=file_hash(previous_path), previous_missing=sorted(old_missing),
                  newly_available=sorted(newly_available), still_missing=missing, considered=len(rows),
                  publication_cap_rejections=capped, retry_grid_ms=GRID_MS, original_retry_policy=original.POLICIES,
                  dense_retry_policy=DELAY_POLICIES, archive_sha256={k: r['sha256'] for k, r in
                      (archives | downloaded).items() if 'sha256' in r},
                  public_tape=tape_checks, entry_books=entry_checks, exit_books=exit_checks, results=scores,
                  attempts={name: [{k: row[k] for k in ('symbol', 'epoch', 'direction', 'entry_ms', 'entry_status',
                         'entry_attempts')} for row in group] for name, group in picked.items()},
                  live_activation_supported=False, limitations=[
                      'Native signal geometry and eligible publication slots are held fixed.',
                      'Every delayed quote preserves the setup’s backing level and the original slippage cap.',
                      'Research-only spread bands up to 10 or 15 basis points require two fresh quotes and 2R executable reward.',
                      'Exact outcomes fail when an archived exit is unavailable. Stressed print-based estimates are shown separately.',
                      'This is a short stock-only cohort; simultaneous crypto trades outside it do not enter the calculation.',
                  ])
    write_json(ROOT / f'{STEM}.json', report)


if __name__ == '__main__':
    main()
