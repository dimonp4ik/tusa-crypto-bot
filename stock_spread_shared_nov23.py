"""Reprice stock spread variants on the previously frozen serial crypto paths."""

from collections import defaultdict

from crypto_quote_priority_nov23 import COHORTS, freeze, outcome_tape
from okx_crypto_risk_range_nov22 import merge_books
from okx_post_close_cash_nov23 import trade_difference
from okx_post_stop_reclaim_nov23 import with_actual_fill_path
from okx_recent_trade_repair_nov21 import load_repaired_tape
from okx_selective_single_target_nov23 import cash_summary, key
from okx_unbankable_tp1_nov21 import read_json
from okx_xperp_cash_tick_nov19 import CLOSE_LAGS
from okx_xperp_l2_entry_replay_nov19 import attempt
from okx_xperp_multiday_capture_nov20 import book_files
from okx_xperp_multiday_replay_nov20 import ROOT, exit_books, file_hash, key_for, pull_samples, resolve_all
from okx_xperp_tick_path_nov19 import resolve_ticks
from src.execution_guard import MAX_ENTRY_SPREAD, execution_quote
from src.okx_trader import calc_contracts
import stock_native_spread_replay_nov23 as native_spread
from stock_liquid_quote_routes_nov21 import write_json
from stock_spread_recovery_nov23 import DELAY_POLICIES


STEM = 'stock_spread_shared_nov23'
RULES = ('current', 'two_fresh_quotes_five_seconds_apart', 'two_quotes_up_to_10bp_with_2r',
         'two_quotes_up_to_15bp_with_2r')


def current_stock_inputs(report):
    plan = native_spread.native.freeze()
    frozen, specs, ticks, tapes, checks, _, _ = native_spread.prepare(plan)
    lookup = {(row['symbol'], row['epoch']): row for row in frozen}
    scenarios = {}
    for policy in RULES:
        selected = []
        for decision in report['attempts'][policy]:
            if decision['entry_status'] != 'eligible':
                continue
            row = lookup[decision['symbol'], decision['epoch']]
            item = dict(row, entry_status='eligible', entry_ms=decision['entry_ms'])
            if policy in DELAY_POLICIES:
                _, spread, reward = DELAY_POLICIES[policy]
                item.update(research_entry_spread=spread, research_minimum_reward_r=reward,
                            research_backing_level=float(row['native_candidate']['backing_level']))
            selected.append(item)
        scenarios[policy] = selected
    return scenarios, specs, ticks, tapes, checks


def entry_snapshots(rows, specs):
    coverage = book_files(dict(rows=rows, specs=specs))
    samples = defaultdict(set)
    for row in rows:
        samples[key_for(row['symbol'], row['entry_ms'])].add(row['entry_ms'])
    jobs = []
    for label, times in sorted(samples.items()):
        day, symbol = label.split(':', 1)
        item = coverage[label]
        if 'path' not in item:
            raise ValueError(f'entry book unavailable for {label}')
        jobs.append((symbol, int(day), item, times))
    books, checks = pull_samples(jobs, stage='frozen entries for the shared spread study')
    return books, checks, coverage


def compare(label, start, stop, folder, crypto_report, stock_inputs):
    view = crypto_report['cases'][label]
    capture, candidates, _, digest = freeze(start, stop, folder)
    if view['market']['model_capture_sha256'] != digest:
        raise ValueError('crypto source diverged from the serial replay')
    by_signal = {(row['symbol'], row['epoch']): row for row in candidates}
    crypto_plans = {}
    for pace, pair in view['decisions'].items():
        crypto_plans[pace] = [dict(by_signal[decision['symbol'], decision['epoch']], **{
            k: decision[k] for k in ('entry_status', 'entry_ms')}) for decision in pair['alphabetical']]
    unique_crypto = {key(row): row for plan in crypto_plans.values() for row in plan if row['entry_status'] == 'eligible'}
    stock_rules, stock_specs, stock_ticks, stock_tapes, _ = stock_inputs
    stock_plans = {rule: [row for row in stocks if start <= row['epoch'] < stop] for rule, stocks in stock_rules.items()}
    unique_stocks = {key(row): row for rows in stock_plans.values() for row in rows}
    all_crypto, all_stocks = list(unique_crypto.values()), list(unique_stocks.values())
    crypto_specs = capture['specs']
    if crypto_specs.keys() & stock_specs.keys():
        raise ValueError('asset identifier shared between both trading universes')
    specs = crypto_specs | {r['symbol']: stock_specs[r['symbol']] for r in all_stocks}
    ticks = {s: float(spec['tickSz']) for s, spec in crypto_specs.items()} | {
        r['symbol']: stock_ticks[r['symbol']] for r in all_stocks}
    books, entry_checks, archives = entry_snapshots(all_crypto + all_stocks, specs)
    for row in all_crypto:
        symbol, at = row['symbol'], row['entry_ms']
        quote = attempt(books[symbol].get(at), at, row['direction'], specs[symbol], row['stop'], row['target'])
        if quote['status'] != 'eligible':
            raise ValueError('a frozen crypto quote changed')
        row['reference_entry'] = quote['average']
    for row in all_stocks:
        symbol, at = row['symbol'], row['entry_ms']
        times, tape = stock_tapes[symbol]
        spec = specs[symbol]
        quote = execution_quote(books[symbol][at], row['direction'], None, at,
            size_for_price=lambda price: calc_contracts(native_spread.CAPITAL * native_spread.FRACTION,
                                                        native_spread.LEVERAGE, price, spec),
            max_spread=row.get('research_entry_spread', MAX_ENTRY_SPREAD))
        row['reference_entry'] = quote.average
        row['path'] = {60: resolve_ticks(tape, times, signal=row, entered_at=at, entry_px=quote.average,
                                        tick_size=ticks[symbol], amend_seconds=60)}
    trades, repairs = outcome_tape(capture, all_crypto, folder)
    trade_checks = resolve_all(all_crypto, {'files': trades}, {'symbols': capture['references']}, ticks,
                               trade_loader=load_repaired_tape)
    if any(v.get('error') or v['mismatches'] for v in trade_checks.values()):
        raise ValueError('crypto data fail the observed trade-volume reconciliation')
    for plan in crypto_plans.values():
        for row in plan:
            if row['entry_status'] == 'eligible':
                row.update(unique_crypto[key(row)])
    for rows in stock_plans.values():
        for row in rows:
            # Paths are shared at an identical timestamp. Market eligibility
            # parameters stay attached to the variant that actually used them.
            exact = unique_stocks[key(row)]
            row.update(path=exact['path'], reference_entry=exact['reference_entry'])

    added, exits = exit_books(all_crypto + all_stocks, {'books': archives}, specs, books)
    market_fill_checks = dict(repriced=[], extra_exit_checks={})
    run = with_actual_fill_path(trades, archives | added, market_fill_checks)
    results = {}
    for pace, crypto in crypto_plans.items():
        outcomes = {}
        for lag in CLOSE_LAGS:
            args = dict(clock=60, lag=lag, setting='requested5', max_positions=12)
            control = run(crypto, specs, ticks, books, **args)
            reference = view['cash'][pace]['crypto'][str(lag)]['alphabetical']
            if 'closed' not in control or control['closed'] != reference['positions']:
                raise ValueError(f'published complete crypto control differs on {label} {pace} {lag}')
            cases = {}
            for rule, stocks in stock_plans.items():
                trial = run(crypto + stocks, specs, ticks, books, **args)
                result = cash_summary(trial, {key(row) for row in stocks})
                if 'closed' in trial:
                    result['closed_stock_positions'] = [row for row in trial['closed'] if row['bot'] == 'stocks']
                    result['changes'] = trade_difference(control, trial)
                cases[rule] = result
            outcomes[str(lag)] = cases
            print(label, pace, lag, {name: (row.get('trades'), row.get('cash_pnl_usd'), row.get('reason'))
                                     for name, row in cases.items()}, flush=True)
        results[pace] = outcomes
    return dict(window_utc=[start, stop], crypto_source_sha256=digest, crypto_tape_checks=trade_checks,
                repaired_trade_source_sha256=repairs, entries=entry_checks, exits=exits,
                dynamic_fill_checks=market_fill_checks, results=results)


def main():
    crypto_path = ROOT / 'crypto_serial_priority_nov23.json'
    stock_path = ROOT / 'stock_spread_recovery_nov23.json'
    crypto_report = read_json(crypto_path)
    stock_inputs = current_stock_inputs(read_json(stock_path))
    result = {label: compare(label, start, stop, folder, crypto_report, stock_inputs)
              for label, start, stop, folder in COHORTS}
    write_json(ROOT / f'{STEM}.json', dict(cases=result, controls={p.name: file_hash(p) for p in (crypto_path, stock_path)},
        stock_tape_checks=stock_inputs[4], live_activation_supported=False, limitations=[
            'The shared model starts flat in both short calendar windows.',
            'The crypto side retains the published alphabetical sequences and three fixed timing stresses.',
            'Stock proposals use the frozen observation’s original publication admission and preserved trade levels.',
            'An independent observed NVDA position from an earlier methodology check is not an extra stock proposal here.',
            'The results are exact public-book simulations. They do not establish calendar-month or private account returns.',
        ]))


if __name__ == '__main__':
    main()
