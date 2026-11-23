"""Compare pre-entry priority rules at one common archived quote clock."""

import argparse
from collections import Counter, deque
from itertools import groupby
from unittest.mock import patch

import crypto_archive_forward_nov23 as previous
from crypto_scan_priority_nov22 import attribute, reconstruct_rank
import okx_cross_bot_overlap_nov21 as joint
from okx_crypto_risk_range_nov22 import attach_entry_quotes, merge_books
from okx_post_close_cash_nov23 import trade_difference
from okx_post_stop_reclaim_nov23 import FORWARD_DATA, FORWARD_START, FORWARD_STOP, with_actual_fill_path
from okx_recent_trade_repair_nov21 import load_repaired_tape, repair_recent_days
from okx_selective_single_target_nov23 import cash_summary, key
from okx_unbankable_tp1_nov21 import read_json
from okx_xperp_cash_tick_nov19 import CAPITAL, CLOSE_LAGS, FRACTION, LEVERAGE
from okx_xperp_multiday_capture_nov20 import local_trade_days
from okx_xperp_multiday_replay_nov20 import HOLD, ROOT, exit_books, file_hash, resolve_all
from okx_xperp_tick_path_nov19 import load_trade_tape
import stock_native_geometry_match_nov22 as native_stocks
from stock_liquid_quote_routes_nov21 import write_json


STEM = 'crypto_quote_priority_nov23'
SAMPLE_MS = 11_000
ROLLING_LIMIT = 2
POLICIES = ('alphabetical', 'model_quality', 'btc_aligned_then_quality',
            'narrow_spread_then_quality', 'execution_reward_r', 'different_side_if_available')
COHORTS = (
    ('nov20', previous.START, previous.ENTRY_STOP, previous.DATA),
    ('nov21', FORWARD_START, FORWARD_STOP, FORWARD_DATA),
)


def features(row):
    quote = row['tries'][-1]
    worst = quote['worst']
    return dict(spread_bps=quote['spread_bps'], execution_reward_r=abs(row['target'] - worst) / abs(worst - row['stop']))


def sorted_scan(rows, policy):
    if policy == 'alphabetical':
        return sorted(rows, key=lambda r: r['symbol'])
    quality = sorted(rows, key=lambda r: (-r['model_quality'], r['symbol']))
    if policy == 'model_quality':
        return quality
    if policy == 'btc_aligned_then_quality':
        return sorted(quality, key=lambda r: r['market_alignment'] != 'aligned')
    if policy == 'narrow_spread_then_quality':
        return sorted(quality, key=lambda r: r['spread_bps'])
    if policy == 'execution_reward_r':
        return sorted(quality, key=lambda r: -r['execution_reward_r'])
    if policy == 'different_side_if_available':
        if not quality:
            return quality
        first, *rest = quality
        return [first, *sorted(rest, key=lambda r: r['direction'] == first['direction'])]
    raise ValueError('unknown ranking policy')


def admitted(rows, policy):
    recent = deque()
    for at, scan in groupby(sorted(rows, key=lambda r: r['epoch']), key=lambda r: r['epoch']):
        while recent and recent[0] <= at - 3600:
            recent.popleft()
        eligible = []
        for row in scan:
            if row['entry_status'] == 'eligible':
                eligible.append(row)
            else:
                yield row
        for priority, row in enumerate(sorted_scan(eligible, policy)):
            if len(recent) >= ROLLING_LIMIT:
                yield dict(row, entry_status='publication_rate_limit')
            else:
                recent.append(at)
                yield dict(row, same_clock_priority=priority)


def freeze(start, stop, folder):
    manifest = folder / 'crypto_capture.json'
    captured = read_json(manifest)
    needed = set(local_trade_days(start, stop).values())
    if any(not needed.issubset(record['days']) for record in captured['trades'].values()):
        raise ValueError('exact freeze requires all public trades at decision time')

    def decision_tape(item):
        return load_trade_tape(dict(item, complete=True))

    with patch.multiple(joint, START=start, ENTRY_END=stop, HOLD={'crypto': 900}, load_trade_tape=decision_tape):
        rows, annotations, checks = reconstruct_rank(captured)
    return captured, [dict(row, **annotations[row['symbol'], row['epoch']]) for row in rows], checks, file_hash(manifest)


def outcome_tape(captured, relevant, folder):
    trades = captured['trades']
    repairs = repair_recent_days(trades, relevant, hold_seconds=HOLD['crypto'], folder=folder / 'repaired')
    evidence = {symbol: {day: item['sha256'] for day, item in values.items()} for symbol, values in repairs.items()}
    return {symbol: dict(item, repaired=repairs[symbol]) if symbol in repairs else dict(item, complete=True)
            for symbol, item in trades.items()}, evidence


def stock_overlap(start, stop):
    plan = native_stocks.freeze()
    cases, specs, ticks, books, _, _, checks = native_stocks.prepare_cases(plan)
    rows = [row for row in cases['production_1pct'] if start <= row['epoch'] < stop]
    if not rows or any(not row['path'][60]['known'] for row in rows):
        raise ValueError('native stock overlap needs known trade paths and observed entry books')
    symbols = {row['symbol'] for row in rows}
    return (rows, {symbol: specs[symbol] for symbol in symbols}, {symbol: ticks[symbol] for symbol in symbols},
            {symbol: books[symbol] for symbol in symbols}, dict(
                source_sha256=file_hash(native_stocks.DATA / 'plan.json'),
                case_source_sha256=file_hash(native_stocks.DATA / 'capture.json'),
                tape_checks={symbol: checks[symbol] for symbol in symbols},
                observations=[dict(symbol=row['symbol'], entry_ms=row['entry_ms'], entry=row['reference_entry'],
                                   exit_path=row['path'][60]) for row in rows]))


def replay(label, start, stop, folder, *, expected, shared=False):
    captured, raw, checks, source_sha = freeze(start, stop, folder)
    specs = captured['specs']
    ticks = {symbol: float(spec['tickSz']) for symbol, spec in specs.items()}
    available = {}
    plan = dict(specs=specs, parameters=dict(crypto_offset_ms=SAMPLE_MS, retry_offsets_ms=[0]))
    print(label, 'frozen proposals', len(raw), flush=True)
    proposals, books, entries = attach_entry_quotes(raw, plan, available, ticks)
    if any(row['entry_status'] == 'unknown_quote' for row in proposals):
        raise ValueError('the synchronized quote panel is incomplete')
    rows = [dict(row, **features(row)) if row['entry_status'] == 'eligible' else row for row in proposals]
    candidates = {name: {key(row) for row in admitted(rows, name) if row['entry_status'] == 'eligible'} for name in POLICIES}
    union = set().union(*candidates.values())
    relevant = [row for row in rows if key(row) in union]
    print(label, 'selected by at least one rule', len(relevant), flush=True)
    trades, repairs = outcome_tape(captured, relevant, folder)
    paths = resolve_all(relevant, {'files': trades}, {'symbols': captured['references']}, ticks,
                        trade_loader=load_repaired_tape)
    if any(check.get('error') or check['mismatches'] for check in paths.values()):
        raise ValueError('the publicly printed outcomes are incomplete')
    stock_rows = []
    stock_evidence = None
    if shared:
        stock_rows, stock_specs, stock_ticks, stock_books, stock_evidence = stock_overlap(start, stop)
        if specs.keys() & stock_specs.keys():
            raise ValueError('ambiguous instrument in the shared replay')
        specs.update(stock_specs)
        ticks.update(stock_ticks)
        merge_books(books, stock_books)
    extra, exits = exit_books(relevant + stock_rows, {'books': available}, specs, books)
    fills = dict(repriced=[], extra_exit_checks={})
    cash = with_actual_fill_path(trades, available | extra, fills)
    by_lag = {}
    for lag in CLOSE_LAGS:
        policies = {}
        original = None
        for name in POLICIES:
            portfolio = list(admitted(rows, name)) + stock_rows
            result = cash(portfolio, specs, ticks, books, clock=60, lag=lag, setting='requested5', max_positions=12)
            view = cash_summary(result, {})
            if name == 'alphabetical':
                original = result
                if expected:
                    for field in ('trades', 'return_pct', 'drawdown_pct', 'contribution_pct'):
                        if view[field] != expected[str(lag)][field]:
                            raise ValueError(f'previous synchronized cash control differs in {field}')
            elif 'closed' in original and 'closed' in result:
                view['changes'] = trade_difference(original, result)
            if 'closed' in result:
                view['positions'] = result['closed']
                view['by_market_alignment'] = attribute(result['closed'], portfolio)
            policies[name] = view
        by_lag[str(lag)] = policies
        if stock_rows:
            stock_only = cash(stock_rows, specs, ticks, books, clock=60, lag=lag, setting='requested5', max_positions=12)
            policies['observed_stock_only'] = cash_summary(stock_only, {})
        print(label, lag, {name: (p.get('trades'), p.get('cash_pnl_usd'), p.get('reason'))
                           for name, p in policies.items()}, flush=True)
    return dict(window_utc=[start, stop], source_sha256=source_sha, proposal_count=len(raw),
                quote_status=Counter(row['entry_status'] for row in rows),
                decisions=[dict(symbol=r['symbol'], epoch=r['epoch'], direction=r['direction'],
                                model_quality=r['model_quality'], btc_alignment=r['market_alignment'],
                                spread_bps=r.get('spread_bps'), reward_r=r.get('execution_reward_r'),
                                quote_status=r['entry_status']) for r in rows],
                admitted_by_policy={name: sorted(values) for name, values in candidates.items()},
                tape_checks=checks, outcome_repair_sha256=repairs, path_checks=paths, entry_checks=entries,
                exit_checks=exits, actual_fill_checks=fills, stock_overlap=stock_evidence,
                exit_archive_sha256={archive: item['sha256'] for archive, item in extra.items()}, results=by_lag)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--shared', action='store_true', help='use the observed native stock overlap on November 21')
    args = parser.parse_args()
    original_source = ROOT / 'crypto_archive_forward_nov23.json'
    controls = {lag: values['alphabetical'] for lag, values in read_json(original_source)['results']['same_time_11s'].items()}
    selected = COHORTS[-1:] if args.shared else COHORTS
    cases = {label: replay(label, start, stop, folder, expected=controls if label == 'nov20' else None, shared=args.shared)
             for label, start, stop, folder in selected}
    suffix = '_shared' if args.shared else ''
    account_scope = ('The shared sample starts flat and adds the independently observed native stock trade.' if args.shared
                     else 'This isolated crypto sample uses shared capacity limits without simultaneous stock orders.')
    write_json(ROOT / f'{STEM}{suffix}.json', dict(cases=cases, common_quote_offset_ms=SAMPLE_MS, policies=POLICIES,
        rolling_limit_per_hour=ROLLING_LIMIT, capital_usd=CAPITAL, margin_fraction=FRACTION, leverage=LEVERAGE,
        synchronized_control_sha256=file_hash(original_source), live_activation_supported=False, limitations=[
            'Every policy receives the same exact 11-second historical depth snapshots before ranking.',
            'The synchronized panel models a common decision time. It does not reproduce live serial requests and retries.',
            'Each policy retains the original signal geometry, exact exit quotes and compounded cash sizing.',
            account_scope,
            'Prior exits and quotes are public evidence; the comparisons do not describe monthly or private account outcomes.',
        ]))


if __name__ == '__main__':
    main()
