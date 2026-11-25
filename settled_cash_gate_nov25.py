"""Compare completed-outcome admission on exact public books and common cash."""

from bisect import bisect_left
from collections import Counter
from pathlib import Path
from unittest.mock import patch

import stock_crypto_release_reaction_nov24 as shared
from okx_post_close_cash_nov23 import trade_difference
from okx_selective_single_target_nov23 import EARLIER, JOINT, cash_summary, key
from okx_spot_confirmation_nov22 import load_native
from okx_sweep_context_cash_nov23 import holdout_inputs
from okx_unbankable_tp1_nov21 import old_inputs, read_json, recent_inputs
from okx_xperp_cash_tick_nov19 import CLOSE_LAGS, RELEASE_MS, simulate
from okx_xperp_multiday_replay_nov20 import ROOT, file_hash
from settled_outcome_gate_nov25 import POLICIES, filter_events
from src.okx_trader import round_to_tick
from stock_liquid_quote_routes_nov21 import write_json


FIELDS = ('status', 'trades', 'return_pct', 'drawdown_pct', 'contribution_pct')


def market_groups(rows, refs):
    bitcoin = load_native({'BTCUSDT': refs['BTCUSDT']})['BTCUSDT'] if 'BTCUSDT' in refs else None
    groups = {}
    for row in rows:
        side = 1 if row['direction'] == 'LONG' else -1
        if row['bot'] == 'stocks':
            extra = row.get('native_candidate') or {}
            change = extra.get('basket_trend_16')
        else:
            i = bisect_left(bitcoin['time'], row['epoch']) if bitcoin is not None else 0
            window = bitcoin['time'][i-17:i] if i >= 17 else []
            if list(window) != list(range(row['epoch']-17*900, row['epoch'], 900)):
                change = None
            else:
                change = bitcoin['close'][i-1]/bitcoin['close'][i-17]-1
        if change is not None:
            groups[key(row)] = 'opposed' if side*change < 0 else 'aligned'
    return groups


def shadow_outcomes(rows, specs, ticks, books, *, lag):
    results, unknown = {}, Counter()
    for row in rows:
        one = simulate([row], specs, ticks, books, clock=60, lag=lag, setting='requested5')
        if not one.get('closed'):
            unknown[one.get('reason', 'not_priceable')] += 1
            continue
        fill, = one['closed']
        stop = round_to_tick(row['stop'], ticks[row['symbol']])
        risk = abs(fill['entry_px']-stop) * fill['qty'] * float(specs[row['symbol']]['ctVal'])
        if not risk > 0:
            raise ValueError('no initial cash risk')
        pnl = fill['bank_pnl'] + fill['runner_pnl']
        results[key(row)] = fill['exit_ms'] + RELEASE_MS, pnl/risk
    return results, unknown


def compare(rows, specs, ticks, books, references, expected, *, cap=12):
    eligible = [row for row in rows if row['entry_status'] == 'eligible'
                and (row['bot'] != 'stocks' or row['stock_reference_ready'])]
    market = market_groups(eligible, references)

    def group(row, partition):
        side = row['bot'], row['direction']
        if partition == 'side':
            return side
        context = market.get(key(row))
        return side + (context,) if context is not None else None

    report = {}
    for lag in CLOSE_LAGS:
        runtime = dict(clock=60, lag=lag, setting='requested5', max_positions=cap)
        base = simulate(rows, specs, ticks, books, **runtime)
        if any(base.get(field) != expected[str(lag)].get(field) for field in FIELDS):
            raise ValueError('frozen comparison failed to reproduce the exact baseline')
        observations, incomplete = shadow_outcomes(eligible, specs, ticks, books, lag=lag)
        cases = dict(baseline=cash_summary(base, {}))
        for name, policy in POLICIES.items():
            rejected = filter_events(eligible, observations, policy, identify=key,
                    entrance=lambda r: r['entry_ms'], group=group)
            inputs = [dict(row, entry_status='settled_shadow_health') if key(row) in rejected else row for row in rows]
            outcome = simulate(inputs, specs, ticks, books, **runtime)
            affected = [dict(symbol=r['symbol'], entry_ms=r['entry_ms'], code=r['code'], **rejected[key(r)])
                        for r in base['closed'] if key(r) in rejected]
            cases[name] = dict(cash_summary(outcome, {}), rejected_candidates=len(rejected),
                               matched_rejected_trades=affected, difference=trade_difference(base, outcome))
        report[str(lag)] = dict(cases=cases, visible_outcomes=len(observations), incomplete=incomplete,
                                market_group_unknown=len(eligible)-len(market))
        print(lag, {k:(x.get('return_pct'),x.get('rejected_candidates'),x.get('reason'))
                    for k,x in cases.items()}, flush=True)
    return dict(results=report)


def chronological_cohort(data, windows, expected, cap):
    rows, specs, ticks, books, _, _, refs = data
    result = {}
    for name, keep in windows.items():
        source = {lag: group['baseline'] for lag,group in expected[name].items()}
        print(name, flush=True)
        result[name] = compare([r for r in rows if keep(r)], specs, ticks, books, refs, source, cap=cap)
    return result


def main():
    control_path = ROOT / 'okx_macro_reaction_cash_nov23.json'
    reference = read_json(control_path)['exact_cash']
    earlier = chronological_cohort(old_inputs(), EARLIER, reference, 12)
    joint = chronological_cohort(recent_inputs(), JOINT, reference, 6)
    test = chronological_cohort(holdout_inputs(), {'nov20_four_hours':lambda _:True}, reference, 12)

    def current_native(rows, specs, ticks, books, refs, _times, controls):
        print('native_joint', flush=True)
        return compare(rows, specs, ticks, books, refs, controls)

    with patch.object(shared, 'compare', current_native):
        native = shared.exact_shared([])
    write_json(ROOT/'settled_cash_gate_nov25.json', dict(cases=earlier|joint|test|{'native_joint':native},
        policies={k: vars(v) for k,v in POLICIES.items()},
        source_sha256={p.name:file_hash(p) for p in (control_path, Path(__file__), Path('settled_outcome_gate_nov25.py'))},
        live_activation_supported=False, limitations=[
            'Each shadow path is priced as an individual executable order at the fixed test starting cash. The gated account itself compounds and shares slots.',
            'Outcome availability includes the same modeled reservation release delay. Filtered market proposals keep producing shadow outcomes.',
            'Missing book or trade evidence cannot vote in either direction. One mixed sample is experimental; native_joint uses actual production stock proposals.',
            'Cash paths are historical public-book simulations. Short and overlapping windows cannot establish monthly performance.',
        ]))


if __name__ == '__main__':
    main()
