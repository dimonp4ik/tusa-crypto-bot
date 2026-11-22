"""Test whether a still-open price gap matters at a market entry."""

from collections import Counter
from statistics import median

from okx_post_close_cash_nov23 import trade_difference
from okx_selective_single_target_nov23 import EARLIER, JOINT, cash_summary, key
from okx_sweep_context_cash_nov23 import context_windows, holdout_inputs, stock_sources
from okx_unbankable_tp1_nov21 import old_inputs, read_json, recent_inputs
from okx_xperp_cash_tick_nov19 import CLOSE_LAGS, simulate
from okx_xperp_multiday_replay_nov20 import ROOT, file_hash
from stock_liquid_quote_routes_nov21 import write_json


MAX_DISTANCE_R = .75
MIN_GAP_RANGE = .10
MIN_IMPULSE_RANGE = .75
POLICIES = ('baseline', 'avoid_fresh_opposing_gap', 'require_fresh_support', 'avoid_only_filled_support')


def gap_context(bars, *, direction, entry, stop, tick):
    if bars is None:
        return None
    side = 1 if direction == 'LONG' else -1
    risk = abs(entry - stop)
    flags = set()
    for end in range(22, len(bars)):
        first, impulse, final = bars[end - 2:end + 1]
        typical = median(high - low for _, high, low, _, _ in bars[end - 22:end - 2])
        if typical <= 0:
            continue
        low, high = first[1], final[2]
        sign = 1
        if final[1] < first[2]:
            low, high = final[1], first[2]
            sign = -1
        if high - low < max(tick, MIN_GAP_RANGE * typical):
            continue
        if sign * (impulse[3] - impulse[0]) < MIN_IMPULSE_RANGE * typical:
            continue
        near, far = (high, low) if sign == 1 else (low, high)
        room = sign * (entry - near)
        if not 0 <= room <= MAX_DISTANCE_R * risk:
            continue
        later = bars[end + 1:]
        if sign == 1:
            fill_edge = min((bar[2] for bar in later), default=near)
            touched = bool(later) and fill_edge <= near
            filled = bool(later) and fill_edge <= far
        else:
            fill_edge = max((bar[1] for bar in later), default=near)
            touched = bool(later) and fill_edge >= near
            filled = bool(later) and fill_edge >= far
        relation = 'support' if sign == side else 'opposing'
        state = 'filled' if filled else 'touched' if touched else 'fresh'
        flags.add(f'{state}_{relation}')
    return sorted(flags)


def accepts(policy, flags):
    flags = set(flags or ())
    return {
        'baseline': True,
        'avoid_fresh_opposing_gap': 'fresh_opposing' not in flags,
        'require_fresh_support': 'fresh_support' in flags,
        'avoid_only_filled_support': 'filled_support' not in flags
        or bool(flags & {'fresh_support', 'touched_support'}),
    }[policy]


def evaluate(material, cohorts, controls, *, cap, joint=False):
    rows, specs, ticks, books, _, trades, refs = material
    if joint:
        _, refs = stock_sources(trades, refs)
    windows, extra = context_windows(rows, specs, refs)
    states = {
        key(row): gap_context(windows[key(row)], direction=row['direction'], entry=row['reference_entry'],
                              stop=row['stop'], tick=ticks[row['symbol']])
        for row in rows if key(row) in windows
    }
    cases = {}
    for name, keep in cohorts.items():
        original = [row for row in rows if keep(row)]
        by_lag = {}
        for lag in CLOSE_LAGS:
            scenarios = {}
            for policy in POLICIES:
                vetoed = {key(row) for row in original if key(row) in states and not accepts(policy, states[key(row)])}
                selected = [dict(row, entry_status='fvg_filter') if key(row) in vetoed else row for row in original]
                replay = simulate(selected, specs, ticks, books, clock=60, lag=lag, setting='requested5',
                                  max_positions=cap)
                summary = cash_summary(replay, {})
                if policy == 'baseline':
                    control = replay
                    expected = controls[name][str(lag)]
                    for field in ('trades', 'return_pct', 'drawdown_pct', 'contribution_pct'):
                        if summary[field] != expected[field]:
                            raise ValueError(f'previous baseline differs in {name}: {field}')
                else:
                    summary['vetoed'] = sorted(vetoed)
                    if 'closed' in replay:
                        summary['trade_changes'] = trade_difference(control, replay)
                scenarios[policy] = summary
            by_lag[str(lag)] = scenarios
            print(name, lag, {k: (v.get('trades'), v.get('cash_pnl_usd')) for k, v in scenarios.items()}, flush=True)
        cases[name] = by_lag
    coverage = Counter(f'{bot}:{state}' for (bot, _, _), flags in states.items() for state in
                       (['unknown'] if flags is None else flags or ['no_nearby_gap']))
    return cases, dict(coverage=coverage, extra_candle_sha256=extra,
                       reference_sha256={sym: item['sha256'] for sym, item in refs.items()})


def main():
    source = ROOT / 'okx_sweep_context_cash_nov23.json'
    previous = read_json(source)['exact_cash_cases']
    controls = {name: {lag: values['baseline'] for lag, values in runs.items()} for name, runs in previous.items()}
    early, early_checks = evaluate(old_inputs(), EARLIER, controls, cap=12)
    joint, joint_checks = evaluate(recent_inputs(), JOINT, controls, cap=6, joint=True)
    fresh, recent_checks = evaluate(holdout_inputs(), {'nov20_four_hours': lambda _: True}, controls, cap=12)
    report = dict(cases=early | joint | fresh, source_sha256=file_hash(source), policies=POLICIES,
                  minimum_gap_range=MIN_GAP_RANGE, minimum_impulse_range=MIN_IMPULSE_RANGE,
                  maximum_distance_r=MAX_DISTANCE_R, live_activation_supported=False,
                  checks=dict(early=early_checks, joint=joint_checks, recent=recent_checks), limitations=[
                      'The three candles must all close before a gap can be classified.',
                      'Later closed candles establish whether price touched the near edge or filled the far edge.',
                      'A nearby gap marks observed price geometry. It conveys no trader identity or flow.',
                      'Previously studied November windows are reused here to screen a new hypothesis.',
                      'The joint equity sample includes experimental candidates. It is not a statement of live revenue.',
                  ])
    write_json(ROOT / 'okx_fvg_freshness_cash_nov23.json', report)


if __name__ == '__main__':
    main()
