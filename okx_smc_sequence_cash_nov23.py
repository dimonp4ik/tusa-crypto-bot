"""Replay ordered Smart Money signals with the original account and quotes."""

from bisect import bisect_left
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from okx_post_close_cash_nov23 import trade_difference
from okx_selective_single_target_nov23 import EARLIER, JOINT, cash_summary, key
from okx_sweep_context_cash_nov23 import context_windows, holdout_inputs, stock_paper_rows, stock_sources, window
from okx_unbankable_tp1_nov21 import old_inputs, read_json, recent_inputs
from okx_xperp_cash_tick_nov19 import CLOSE_LAGS, simulate
from okx_xperp_multiday_replay_nov20 import ROOT, file_hash
from okx_spot_confirmation_nov22 import load_native
from smc_sequence_nov23 import state
from stock_liquid_quote_routes_nov21 import write_json


POLICIES = {
    'baseline': (None, None),
    'require_equal_level_sweep': ('equal_sweep', None),
    'require_structure_break': ('displacement', None),
    'require_equal_level_and_structure_break': ('equal_displacement', None),
    'require_first_order_block_retest': ('block_retest', None),
    'avoid_opposing_structure_break': (None, 'displacement'),
    'avoid_opposing_block_retest': (None, 'block_retest'),
}


def context(bars, direction, tick):
    opposite = 'SHORT' if direction == 'LONG' else 'LONG'
    return state(bars, direction, tick), state(bars, opposite, tick)


def accepts(policy, features):
    required, rejected = POLICIES[policy]
    aligned, opposed = features
    return (required is None or required in (aligned or ())) and rejected not in (opposed or ())


def exact_cases(material, cohorts, controls, *, cap, joint=False):
    rows, specs, ticks, books, _, trades, refs = material
    if joint:
        _, refs = stock_sources(trades, refs)
    windows, evidence = context_windows(rows, specs, refs)
    states = {key(row): context(windows[key(row)], row['direction'], ticks[row['symbol']])
              for row in rows if key(row) in windows}
    cases = {}
    for name, keep in cohorts.items():
        cohort = [row for row in rows if keep(row)]
        cases[name] = {}
        for lag in CLOSE_LAGS:
            policies = {}
            for policy in POLICIES:
                refused = {key(row) for row in cohort if key(row) in states and not accepts(policy, states[key(row)])}
                selected = [dict(row, entry_status='smc_sequence_filter') if key(row) in refused else row
                            for row in cohort]
                replay = simulate(selected, specs, ticks, books, clock=60, lag=lag,
                                  setting='requested5', max_positions=cap)
                summary = cash_summary(replay, {})
                if policy == 'baseline':
                    baseline = replay
                    for field in ('trades', 'return_pct', 'drawdown_pct', 'contribution_pct'):
                        if summary[field] != controls[name][str(lag)][field]:
                            raise ValueError(f'Baseline changed for {name}/{lag}/{field}')
                summary['eligible_rejected'] = dict(Counter(identity[0] for identity in refused))
                if refused and 'closed' in baseline and 'closed' in replay:
                    summary['trade_changes'] = trade_difference(baseline, replay)
                summary['outcomes_by_bot'] = {
                    bot: dict(Counter(trade['code'] for trade in replay.get('closed', []) if trade['bot'] == bot))
                    for bot in ('crypto', 'stocks')
                }
                policies[policy] = summary
            cases[name][str(lag)] = policies
            print(name, lag, {p: (v.get('trades'), round(v.get('cash_pnl_usd', 0), 3))
                              for p, v in policies.items()}, flush=True)
    coverage = defaultdict(Counter)
    for identity, ctx in states.items():
        for relation, flags in zip(('aligned', 'opposed'), ctx, strict=True):
            labels = {'unknown'} if flags is None else flags or {'no_pattern'}
            coverage[identity[0]].update(f'{relation}:{flag}' for flag in labels)
    return cases, dict(coverage=coverage, candle_downloads=evidence,
                       references={symbol: ref['sha256'] for symbol, ref in refs.items()})


def historical_crypto(baseline):
    from crypto_active_entry_runner_nov19_audit import activity_gate
    from crypto_minimum_bank_runner_nov19_audit import admitted_records, cash_result, checked_row
    from crypto_native_price_pool_nov19 import MANIFEST, load_pool
    from liquidity_trap_joint_nov19_audit import SNAPSHOT, with_exact_tick

    candles = load_native(read_json(MANIFEST)['symbols'])
    spec_path = ROOT / 'shared_xperp_specs_nov14.json'
    specs = read_json(spec_path)
    instruments = read_json(SNAPSHOT)['instruments']
    ticks = {symbol: float(with_exact_tick(specs[symbol], instruments)['tickSz']) for symbol in candles}
    pool, metadata = load_pool()
    stop = datetime(2026, 11, 1, tzinfo=timezone.utc).timestamp()
    candidates = []
    for row in pool[pool.is_frozen_candidate & pool.entry_time.lt(stop)].to_dict('records'):
        symbol, epoch = row['symbol'], int(row['entry_time'])
        frame = candles[symbol]
        i = bisect_left(frame['time'], epoch)
        if i >= len(frame['time']) or frame['time'][i] != epoch:
            raise ValueError('Native candle missing from the published baseline')
        if activity_gate(frame, i):
            flags = context(window(frame, epoch), row['direction'], ticks[symbol])
            candidates.append((flags, checked_row(row, frame, i, 'be_tp1', .0005)))

    results = {}
    for policy in POLICIES:
        selected = [row for flags, row in candidates if accepts(policy, flags)]
        admitted, unknown = admitted_records(selected)
        if unknown:
            results[policy] = dict(admitted_unknown=unknown, candidates=len(selected))
            continue
        result = cash_result(admitted, specs, 'requested5')
        if policy == 'baseline' and result != baseline:
            raise ValueError('Published historical candle baseline changed')
        results[policy] = dict(cash=result, candidates=len(selected))
        print('jul_oct', policy, result['summary']['trades'], result['summary']['return_pct'], flush=True)
    return dict(results=results, coverage=sum(flags[0] is not None for flags, _ in candidates),
                pool_sha256=metadata['sha256'], manifest_sha256=file_hash(MANIFEST),
                specs_sha256=file_hash(spec_path), tick_snapshot_sha256=file_hash(SNAPSHOT))


def observed_stocks():
    import stock_native_geometry_match_nov22 as observer

    rows, ticks, _, digest = stock_paper_rows()
    frozen = read_json(observer.DATA / 'plan.json')
    folder, _ = observer.ARCHIVES['contract_followup']
    frames = {}
    outcomes = {policy: Counter() for policy in POLICIES}
    for row, _ in rows:
        epoch, symbol = row['epoch'], row['symbol']
        if epoch not in frames:
            source = folder / f'bucket_{epoch}.json'
            if file_hash(source) != frozen['archive_sha256'][source.name]:
                raise ValueError('Stock observation changed')
            frames[epoch] = {item['symbol']: item['closed_frame'] for item in read_json(source)['observations']}
        features = context(window(frames[epoch][symbol], epoch), row['direction'], ticks[symbol])
        for policy in POLICIES:
            if accepts(policy, features):
                outcomes[policy][row['paper_path']['code']] += 1
    return dict(outcomes=outcomes, plan_sha256=digest)


def subsequent_opposition():
    import crypto_archive_forward_nov23 as archive
    captured, rows, checks, digest = archive.load_candidates()
    proposals = [dict(row, entry_ms=row['epoch']*1000, entry_status='eligible') for row in rows]
    specs, refs = captured['specs'], captured['references']
    windows, downloads = context_windows(proposals, specs, refs)
    events = []
    missing = []
    for row in proposals:
        symbol, at = row['symbol'], row['epoch']
        bars = windows[key(row)]
        if bars is None:
            missing.append((symbol, at))
        elif not accepts('avoid_opposing_structure_break', context(bars, row['direction'], float(specs[symbol]['tickSz']))):
            events.append((symbol, at))
    return dict(start=archive.START, end=archive.ENTRY_STOP, proposals=len(rows), opposing_breaks=events,
                missing=missing, identical_portfolio_if_no_veto=not missing and not events, capture_sha256=digest,
                candidate_checks=checks, candle_downloads=downloads,
                references={symbol: ref['sha256'] for symbol, ref in refs.items()})


def main():
    source = ROOT / 'okx_sweep_context_cash_nov23.json'
    previous = read_json(source)
    controls = {name: {lag: variants['baseline'] for lag, variants in value.items()}
                for name, value in previous['exact_cash_cases'].items()}
    early, early_check = exact_cases(old_inputs(), EARLIER, controls, cap=12)
    joint, joint_check = exact_cases(recent_inputs(), JOINT, controls, cap=6, joint=True)
    fresh, fresh_check = exact_cases(holdout_inputs(), {'nov20_four_hours': lambda _: True}, controls, cap=12)
    long_crypto = historical_crypto(previous['historical_crypto_native_candle_proxy']['cases']['baseline']['cash'])
    stock = observed_stocks()
    later = subsequent_opposition()
    write_json(ROOT / 'okx_smc_sequence_cash_nov23.json', dict(
        source_sha256={p.name: file_hash(p) for p in (source, Path(__file__), Path('smc_sequence_nov23.py'))},
        policies=POLICIES, exact_cash_cases=early | joint | fresh,
        exact_checks=dict(early=early_check, joint=joint_check, fresh=fresh_check), later_crypto_opposition=later,
        historical_crypto_candle_proxy=long_crypto, native_stock_paper_outcomes=stock,
        live_activation_supported=False, limitations=[
            'These tests keep the original entries, exits and client TP1 setup; they measure filters rather than new orders.',
            'Pattern stages use only candles closed by the original decision. A stage can apply for two closes unless invalidated.',
            'The pre-November crypto sample uses a candle proxy; native stock observer outcomes are paper results.',
            'Exact November windows were reused for this hypothesis and are exploratory.',
            'Candles describe price and activity. They do not reveal stop order inventory or trader identity.',
        ]))


if __name__ == '__main__':
    main()
