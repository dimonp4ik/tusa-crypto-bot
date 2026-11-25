"""Compare prospective loss pauses on verified shared-account X-Perp books."""

import argparse
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import stock_native_geometry_match_nov22 as native
import stock_native_spread_replay_nov23 as stock
from crypto_quote_priority_nov23 import freeze, outcome_tape
from crypto_scan_priority_nov22 import book_cadence
from okx_crypto_risk_range_nov22 import attach_entry_quotes
from okx_post_close_cash_nov23 import trade_difference
from okx_recent_trade_repair_nov21 import load_repaired_tape
from okx_selective_single_target_nov23 import EARLIER, JOINT, cash_summary
from okx_sweep_context_cash_nov23 import holdout_inputs
from okx_unbankable_tp1_nov21 import old_inputs, read_json, recent_inputs
from okx_xperp_cash_tick_nov19 import CLOSE_LAGS, RELEASE_MS, simulate
from okx_xperp_multiday_capture_nov20 import local_trade_days
from okx_xperp_multiday_replay_nov20 import ROOT, exit_books, file_hash, resolve_all
from stock_contract_spread_holdout_nov23 import DATA as STOCK_DATA
from stock_liquid_quote_routes_nov21 import write_json
from stock_spread_shared_nov23 import entry_snapshots


STEM = 'cash_loss_pause_audit_nov24'
FRESH = ROOT / 'cash_loss_pause_holdout_nov24'
START = int(datetime(2026, 11, 22, tzinfo=timezone.utc).timestamp())
STOP = START+43_200
POLICIES = (None, 'daily', 'bot_hour', 'direction_hour', 'account_hour')


def compare(material, cohorts, *, controls=None, cap=12, clocks=(60,)):
    rows, specs, ticks, books, *_ = material
    output = {}
    for name, keep in cohorts.items():
        cohort = [row for row in rows if keep(row)]
        comparisons = {}
        for clock in clocks:
            for lag in CLOSE_LAGS:
                for release in (120_000, RELEASE_MS):
                    args = dict(clock=clock, lag=lag, setting='requested5', max_positions=cap,
                                release_delay_ms=release)
                    control = simulate(cohort, specs, ticks, books, **args)
                    if controls is not None and release == RELEASE_MS and clock == 60:
                        expected = controls[name][str(lag)]
                        for field in ('return_pct', 'drawdown_pct', 'trades', 'contribution_pct'):
                            if control[field] != expected[field]:
                                raise ValueError(f'Published baseline differs in {name}: {field}')
                    policy_cases = {}
                    for policy in POLICIES:
                        trial = control if policy is None else simulate(
                            cohort, specs, ticks, books, **args, loss_pause_policy=policy)
                        report = cash_summary(trial, {})
                        if policy is not None and 'closed' in trial and 'closed' in control:
                            report['attribution'] = trade_difference(control, trial)
                        policy_cases[policy or 'baseline'] = report
                    comparisons[f'{clock}s_{lag}ms_release_{release//1000}s'] = policy_cases
        output[name] = comparisons
        print(name, {k: (v.get('trades'), v.get('cash_pnl_usd'), v.get('reason'))
                     for k, v in comparisons['60s_200ms_release_900s'].items()}, flush=True)
    return output


def current_native_stock():
    previous_path = ROOT / 'stock_contract_spread_holdout_nov23.json'
    previous = read_json(previous_path)['result']
    plan = read_json(STOCK_DATA / 'plan.json')
    with patch.object(native, 'DATA', STOCK_DATA):
        rows, specs, ticks, _, _, _, _ = stock.prepare(plan)
    by_candidate = {(row['symbol'], row['epoch']): row for row in rows}
    eligible = []
    policies = ('current', 'two_quotes_up_to_15bp_with_2r')
    for policy in policies:
        for fill in previous['fills'][policy]:
            source = by_candidate[fill['symbol'], fill['epoch']]
            row = dict(source, entry_status='eligible', stock_reference_ready=True, **{
                k: fill[k] for k in ('entry_ms', 'reference_entry')})
            row.update(path={int(k): v for k, v in fill['path'].items()}, native_policy=policy)
            if policy != 'current':
                row.update(research_entry_spread=.0015, research_minimum_reward_r=2.,
                           research_backing_level=float(row['native_candidate']['backing_level']))
            eligible.append(row)
    books, book_checks, files = entry_snapshots(eligible, specs)
    extra, exits = exit_books(eligible, {'books': files}, specs, books)
    cohorts = {f'native_stock_{policy}': (lambda row, policy=policy: row['native_policy'] == policy)
               for policy in policies}
    material = eligible, specs, ticks, books, files | extra, {}, {}
    controls = {f'native_stock_{policy}': {str(lag): previous['outcomes'][policy]['modes'][f'{lag}ms_exact']
                                          for lag in CLOSE_LAGS} for policy in policies}
    scores = compare(material, cohorts, controls=controls)
    return scores, dict(archive_paths=file_hash(previous_path), entry_books=book_checks, exit_books=exits)


def fresh_crypto():
    captured, rows, checks, digest = freeze(START, STOP, FRESH)
    specs = captured['specs']
    ticks = {symbol: float(spec['tickSz']) for symbol, spec in specs.items()}
    books = {}
    book_plan = dict(specs=specs, parameters=dict(crypto_offset_ms=11_000, retry_offsets_ms=[0]))
    candidates, quotes, evidence = attach_entry_quotes(rows, book_plan, books, ticks)
    if any(row['entry_status'] == 'unknown_quote' for row in candidates):
        raise ValueError('New crypto entry books are incomplete')
    selected = list(book_cadence(candidates, 'alphabetical'))
    filled = [row for row in selected if row['entry_status'] == 'eligible']
    archived = captured['trades']
    trades = {symbol: dict(item, complete=True) for symbol, item in archived.items()}
    reference = {'symbols': captured['references']}
    trade_checks = resolve_all(filled, {'files':trades}, reference, ticks)
    session_end = {day: midnight+86_400_000
                   for midnight, day in local_trade_days(START, STOP+86_400).items()}
    trusted_until = {symbol: max(session_end[day] for day in item['days']) for symbol, item in archived.items()}
    unresolved = [row for row in filled if any(not path['known'] or path['end_ms'] >= trusted_until[row['symbol']]
                                                for path in row['path'].values())]
    repairs = {}
    if unresolved:
        trades, repairs = outcome_tape(captured, unresolved, FRESH)
        updates = resolve_all(unresolved, {'files':trades}, reference, ticks, trade_loader=load_repaired_tape)
        trade_checks.update(updates)
    extra, exit_checks = exit_books(filled, {'books':books}, specs, quotes)
    material = selected, specs, ticks, quotes, books | extra, trades, captured['references']
    scores = compare(material, {'nov22_crypto_12h': lambda _: True}, clocks=(60,900))
    return scores, dict(capture_sha256=digest, decision_checks=checks, entry_count=len(rows), entry_books=evidence,
                        repairs=repairs, trade_checks=trade_checks, exit_books=exit_checks,
                        archived_prints_through_ms=trusted_until)


def archived_cases():
    source = ROOT / 'okx_sweep_context_cash_nov23.json'
    published = read_json(source)['exact_cash_cases']
    controls = {name: {lag: variants['baseline'] for lag, variants in runs.items()} for name, runs in published.items()}
    early = compare(old_inputs(), EARLIER, controls=controls, clocks=(60,900))
    joint = compare(recent_inputs(), JOINT, controls=controls, cap=6, clocks=(60,900))
    latest = compare(holdout_inputs(), {'nov20_four_hours': lambda _: True}, controls=controls, clocks=(60,900))
    native_scores, native_checks = current_native_stock()
    return early | joint | latest | native_scores, native_checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--part', choices=('archived', 'fresh', 'merge'))
    mode = parser.parse_args().part
    handlers = dict(archived=archived_cases, fresh=fresh_crypto)
    files_by_part = {key: FRESH / f'{key}_result.json' for key in handlers}
    if mode in handlers:
        cases, evidence = handlers[mode]()
        write_json(files_by_part[mode], dict(cases=cases, evidence=evidence))
        return
    parts = ({key: read_json(path) for key, path in files_by_part.items()} if mode == 'merge' else
             {key: dict(zip(('cases','evidence'), func(), strict=True)) for key,func in handlers.items()})
    source = ROOT / 'okx_sweep_context_cash_nov23.json'
    files = (source, Path(__file__), Path('loss_pause_policy_nov24.py'), Path('okx_xperp_cash_tick_nov19.py'))
    write_json(ROOT / f'{STEM}.json', dict(cases=parts['archived']['cases'] | parts['fresh']['cases'],
        evidence={key: part['evidence'] for key,part in parts.items()}, source_sha256={p.name:file_hash(p) for p in files},
        live_activation_supported=False, limitations=[
            'Pauses learn about closes only on the modeled release clock, with two and fifteen minute cases.',
            'The daily control models the portfolio rule using verified executable close outcomes. Production applies it to its signal ledger.',
            'Rolling alternatives wait one hour after two negative cash exits within an hour in the stated scope.',
            'Historical upstream publication and the prequalified stock reference gate remain fixed.',
            'The Nov 17–19 shared stock sleeve is experimental. The independent native stock cohort keeps the original risk brackets.',
            'November 22 uses a single synchronized entry book. Its results do not represent personal fills or a full month.',
        ]))


if __name__ == '__main__':
    main()
