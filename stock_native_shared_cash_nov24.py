"""Replay verified native stock fills and crypto on the same historical cash ledger."""

from collections import Counter, defaultdict
from pathlib import Path
from unittest.mock import patch

import stock_contract_spread_holdout_nov23 as stock
import stock_native_geometry_match_nov22 as geometry
import stock_native_spread_replay_nov23 as spread
from okx_cross_bot_overlap_nov21 import ENTRY_END, START, crypto_cadence, joint_inputs, overlap_view
from okx_joint_exact_exit_nov21 import ENTRY_END as VERIFIED_END
from okx_selective_single_target_nov23 import cash_summary, key
from okx_unbankable_tp1_nov21 import read_json, recent_inputs
from okx_xperp_cash_tick_nov19 import CLOSE_LAGS, simulate
from okx_xperp_multiday_capture_nov20 import book_files
from okx_xperp_multiday_replay_nov20 import ROOT, exit_books, file_hash, key_for, pull_samples
from okx_xperp_tick_path_nov19 import resolve_ticks
from stock_liquid_quote_routes_nov21 import write_json
from stock_spread_recovery_nov23 import DELAY_POLICIES


POLICIES = ('current', 'two_fresh_quotes_five_seconds_apart',
            'two_quotes_up_to_10bp_with_2r', 'two_quotes_up_to_15bp_with_2r')
CAPS = (6, 12)
FIELDS = ('status', 'trades', 'return_pct', 'drawdown_pct', 'contribution_pct')


def identity(row):
    return row['symbol'], row['epoch'], row['direction']


def stock_rows():
    reference_path = ROOT / 'stock_contract_spread_holdout_nov23.json'
    reference = read_json(reference_path)
    plan = stock.plan()
    if reference['plan_sha256'] != file_hash(stock.DATA / 'plan.json'):
        raise ValueError('native observer entry plan changed')
    with patch.object(geometry, 'DATA', stock.DATA):
        proposals, specs, ticks, tapes, tape_checks, _, _ = spread.prepare(plan)
    previous = reference['result']
    if len(proposals) != previous['admitted_signals']:
        raise ValueError('native proposal admission changed')
    original = {identity(r): r for r in proposals}
    if len(original) != len(proposals):
        raise ValueError('duplicate native proposal')
    base_times = {identity(r): r['entry_ms'] for r in previous['fills']['current']}
    entry_points = defaultdict(set)
    material = {}
    for policy in POLICIES:
        rows = []
        for fill in previous['fills'][policy]:
            row = original[identity(fill)]
            symbol, stamp = fill['symbol'], fill['entry_ms']
            times, tape = tapes[symbol]
            if not row['entry_ms'] <= stamp < (row['epoch'] + 180) * 1000:
                raise ValueError('fill outside native publication horizon')
            path = resolve_ticks(tape, times, signal=row, entered_at=stamp, entry_px=fill['reference_entry'],
                                 tick_size=ticks[symbol], amend_seconds=60)
            if path != fill['path']['60']:
                raise ValueError('archived native exit disagrees with verified prints')
            updated = dict(row, entry_status='eligible', reference_entry=fill['reference_entry'],
                           entry_ms=stamp, path={60: path})
            if policy != 'current' and stamp != base_times.get(identity(row)):
                _, spread_limit, reward_floor = DELAY_POLICIES[policy]
                updated.update(research_entry_spread=spread_limit, research_minimum_reward_r=reward_floor,
                               research_backing_level=row['native_candidate']['backing_level'])
            rows.append(updated)
            entry_points[key_for(symbol, stamp)].add(stamp)
        material[policy] = rows

    fills = list({key(r): r for group in material.values() for r in group}.values())
    archives = book_files(dict(rows=fills, specs=specs))
    jobs = []
    for day_symbol, points in sorted(entry_points.items()):
        day, symbol = day_symbol.split(':', 1)
        archive = archives[day_symbol]
        if archive.get('sha256') != previous['archive_sha256'][day_symbol]:
            raise ValueError('native entry archive digest changed')
        jobs.append((symbol, int(day), archive, sorted(points)))
    books, entry_checks = pull_samples(jobs, stage='native executable quotes confirmed')
    extra, exit_checks = exit_books(fills, {'books': archives}, specs, books)
    for label, archive in extra.items():
        if archive.get('sha256') != previous['archive_sha256'][label]:
            raise ValueError('native exit archive digest changed')
    for policy, rows in material.items():
        for lag in CLOSE_LAGS:
            actual = simulate(rows, specs, ticks, books, clock=60, lag=lag, setting='requested5')
            expected = previous['outcomes'][policy]['modes'][f'{lag}ms_exact']
            if any(actual.get(field) != expected.get(field) for field in FIELDS):
                raise ValueError(f'native cash control mismatch for {policy} at {lag}ms')
    return plan['entries_utc'], material, specs, ticks, books, dict(
        entry_books=entry_checks, exit_books=exit_checks, tape=tape_checks,
        reference_sha256=file_hash(reference_path), plan_sha256=reference['plan_sha256'])


def closed_summary(result, rows):
    summary = cash_summary(result, {})
    if 'closed' not in result:
        return summary
    stock_candidates = {key(r): r for r in rows if r['bot'] == 'stocks'}
    phases = defaultdict(Counter)
    summary['stock_trades'] = []
    for trade in result['closed']:
        if trade['bot'] != 'stocks':
            continue
        candidate = stock_candidates[key(trade)]
        phase = candidate['phase']
        side = candidate['direction']
        aligned = (1 if side == 'LONG' else -1) * candidate['native_candidate']['basket_trend_16']
        pnl = trade['bank_pnl'] + trade['runner_pnl']
        phases[phase][trade['code']] += 1
        summary['stock_trades'].append(dict(trade, phase=phase, direction=side,
                                            signed_basket_trend_16=aligned, cash_pnl_usd=pnl))
    return dict(summary, overlap=overlap_view(result['closed']), stock_outcomes_by_phase=phases)


def score(crypto, stocks, specs, ticks, books, start, end):
    common_crypto = [r for r in crypto if start <= r['epoch'] < end]
    cases = {}
    for cap in CAPS:
        for lag in CLOSE_LAGS:
            name = f'{cap}_slots_{lag}ms'
            cases[name] = {}
            for policy, candidates in (('crypto_only', []), *stocks.items()):
                common_stocks = [r for r in candidates if start <= r['epoch'] < end]
                rows = common_crypto + common_stocks
                result = simulate(rows, specs, ticks, books, clock=60, lag=lag, setting='requested5', max_positions=cap)
                cases[name][policy] = closed_summary(result, rows)
            print(end, name, {k: (r.get('trades'), r.get('return_pct'), r.get('reason'))
                              for k, r in cases[name].items()}, flush=True)
    return cases


def main():
    (stock_start, stock_stop), stocks, stock_specs, stock_ticks, stock_books, evidence = stock_rows()
    old_rows, specs, ticks, books, archives, *_ = recent_inputs()
    crypto = [r for r in old_rows if r['bot'] == 'crypto']
    if set(stock_specs) & {r['symbol'] for r in crypto}:
        raise ValueError('colliding stock and crypto instrument symbols')
    books.update(stock_books)
    specs.update(stock_specs)
    ticks.update(stock_ticks)
    start = max(START, stock_start)
    results = {'verified_common_window': score(crypto, stocks, specs, ticks, books, start, VERIFIED_END)}

    extended, _, _, _, extra_books, hashes, *_ = joint_inputs()
    end = min(ENTRY_END, stock_stop)
    additional = [r for r in extended if VERIFIED_END <= r['epoch'] < end]
    extended_crypto = list(crypto_cadence(extended))
    eligible = [r for r in additional if r['entry_status'] == 'eligible']
    for symbol in {r['symbol'] for r in extended}:
        books[symbol].update(extra_books[symbol])
    _, extended_checks = exit_books(eligible, {'books': archives}, specs, books)
    results['extended_common_window'] = score(extended_crypto, stocks, specs, ticks, books, start, end)
    path = ROOT / 'stock_native_shared_cash_nov24.json'
    write_json(path, dict(results=results, windows=dict(start=start, verified_end=VERIFIED_END, extended_end=end),
        exact_stock_control_verified=True, native_sources=evidence, extended_crypto_books=extended_checks,
        crypto_source_hashes=hashes, script_sha256=file_hash(Path(__file__)), live_activation_supported=False,
        limitations=[
            'All native stock route entries already passed the observed standalone entry replay. A changed cash balance can affect sizing and rejected quotes are not re-admitted.',
            'Crypto and stock share simulated cash, total positions and margin. The replay does not model the deployed daily pause or task-queue latency.',
            'Experiment uses November public OKX order books and matched trades with a fixed initial $130 balance, not actual account fills or a complete calendar month.',
        ]))


if __name__ == '__main__':
    main()
