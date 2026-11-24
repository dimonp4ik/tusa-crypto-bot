"""Replay stop amendments for crypto positions below the requested TP1 lot."""

from bisect import bisect_right
from functools import partial
from pathlib import Path

from crypto_scan_priority_holdout_nov22 import DATA, load_cohort
from crypto_scan_priority_nov22 import book_cadence
from okx_delayed_trailing_cash_nov23 import paired_dollars
from okx_selective_single_target_nov23 import EARLIER, JOINT, cash_summary, key
from okx_spot_confirmation_holdout_nov22 import execution_samples
from okx_unbankable_tp1_nov21 import old_inputs, read_json, recent_inputs
from okx_xperp_cash_tick_nov19 import CLOSE_LAGS, simulate
from okx_xperp_multiday_replay_nov20 import HOLD, ROOT, exit_books, file_hash, resolve_all
from src.okx_trader import round_to_tick
from stock_liquid_quote_routes_nov21 import write_json


POLICIES = ('half_at_tp1_be_at_1r', 'be_on_confirmed_tp1', 'half_at_tp1_be_on_confirmed_tp1_or_1r')
CLOSE_FRESHNESS_MS = 10_000


def resolve_stop(tape, times, *, signal, entered_at, entry_px, tick_size, amend_seconds, policy):
    """Use only trades available at the amendment clock, preserving venue triggers."""
    if policy not in POLICIES or amend_seconds not in (60, 900):
        raise ValueError('unknown TP1 confirmation policy or amendment clock')
    sign = 1 if signal['direction'] == 'LONG' else -1
    risk = sign * (signal['entry'] - signal['stop'])
    stop = sign * round_to_tick(signal['stop'], tick_size)
    be = round_to_tick(entry_px, tick_size)
    be_level = sign * be
    target = sign * round_to_tick(signal['target'], tick_size)
    tp1 = sign * round_to_tick(signal['entry'] + sign * .5 * risk, tick_size)
    one_r = sign * round_to_tick(entry_px + sign * risk, tick_size)
    half = sign * round_to_tick(entry_px - sign * .5 * risk, tick_size)
    expiry = (int(signal['epoch']) + HOLD['crypto']) * 1000
    if not (risk > 0 and stop < sign * entry_px < tp1 < target and entered_at < expiry):
        return dict(known=False, reason='invalid_entry_relative_to_tp1_confirmation_bracket')

    interval = amend_seconds * 1000
    bucket = entered_at // interval
    reached_tp1 = reached_one_r = False
    last_trade = None
    desired_stop = stop
    first = bisect_right(times, entered_at)
    end = bisect_right(times, expiry - 1)
    for index in range(first, end):
        trade = tape[index]
        price = sign * trade.price
        # Orders already active at the venue take precedence on this print.
        code = None
        if price <= stop:
            code = 'BE' if stop == be_level else 'TP1_HALF_STOP' if stop == half else 'SL'
        elif price >= target:
            code = 'TARGET'
        if code:
            return dict(known=True, code=code, end_ms=trade.ms, trigger_price=trade.price,
                        bank=None, be_level=be, effective_stop=sign * stop)

        current_bucket = trade.ms // interval
        if current_bucket > bucket:
            boundary = (bucket + 1) * interval
            confirmed = (current_bucket == bucket + 1 and last_trade is not None
                         and boundary - last_trade.ms <= CLOSE_FRESHNESS_MS
                         and sign * last_trade.price >= tp1)
            if reached_one_r and policy != 'be_on_confirmed_tp1' or confirmed and policy != POLICIES[0]:
                desired_stop = max(desired_stop, be_level)
            elif reached_tp1 and policy != 'be_on_confirmed_tp1':
                desired_stop = max(desired_stop, half)
            # An amendment below the extant stop or already crossed cannot fill retroactively.
            if stop < desired_stop < price:
                stop = desired_stop
            bucket = current_bucket

        reached_tp1 |= price >= tp1
        reached_one_r |= price >= one_r
        last_trade = trade

    return dict(known=True, code='TIME', end_ms=expiry, trigger_price=None,
                bank=None, be_level=be, effective_stop=sign * stop)


def score(material, windows, reference, max_positions):
    rows, specs, ticks, books, archived_books, trades, candles = material
    rows = [dict(row, runner_paths={}) for row in rows]
    paths, tape_checks = [], {}
    for policy in POLICIES:
        candidates = [dict(row) for row in rows if row['bot'] == 'crypto' and row['entry_status'] == 'eligible']
        tape_checks[policy] = resolve_all(candidates, {'files': trades}, {'symbols': candles}, ticks,
                                          path_resolver=partial(resolve_stop, policy=policy))
        by_entry = {key(row): row['path'] for row in candidates}
        for row in rows:
            if key(row) in by_entry:
                row['runner_paths'][policy] = by_entry[key(row)]
        paths.extend(candidates)
        print(policy, len(candidates), 'paths', flush=True)
    new_archives, book_checks = exit_books(rows + paths, {'books': archived_books}, specs, books)

    cases = {}
    for name, include in windows.items():
        window = [row for row in rows if include(row)]
        cases[name] = {}
        for clock in (60, 900):
            for lag in CLOSE_LAGS:
                scenario = f'{clock}s_{lag}ms'
                run = dict(clock=clock, lag=lag, setting='requested5', max_positions=max_positions)
                baseline = simulate(window, specs, ticks, books, **run)
                control = reference[name][scenario]['baseline']
                for field in ('status', 'trades', 'return_pct', 'drawdown_pct', 'contribution_pct'):
                    if baseline.get(field) != control.get(field):
                        raise ValueError(f'frozen control mismatch for {name}/{scenario}/{field}')
                variants = {}
                for policy in POLICIES:
                    trial = simulate(window, specs, ticks, books, **run, unbankable_runner=policy)
                    variants[policy] = dict(cash_summary(trial, {}), paired=paired_dollars(baseline, trial))
                cases[name][scenario] = dict(baseline=cash_summary(baseline, {}), **variants)
                print(name, scenario, {rule: result.get('return_pct', result.get('reason'))
                                      for rule, result in cases[name][scenario].items()}, flush=True)
    return cases, dict(tape=tape_checks, exit_l2=book_checks,
                       downloaded_sha256={k: v['sha256'] for k, v in new_archives.items()})


def holdout_inputs():
    capture_path = DATA / 'crypto_capture.json'
    captured, proposals, _ = load_cohort(capture_path)
    rows, books, _ = execution_samples(captured, proposals, file_hash(capture_path))
    selected = list(book_cadence(rows, 'alphabetical'))
    if any(row['entry_status'] == 'unknown_quote' for row in selected):
        raise ValueError('unknown entry quote in the holdout')
    specs = captured['specs']
    ticks = {symbol: float(spec['tickSz']) for symbol, spec in specs.items()}
    return selected, specs, ticks, books, {}, captured['trades'], captured['references']


def main():
    source = ROOT / 'okx_delayed_trailing_cash_nov23.json'
    holdout = ROOT / 'okx_delayed_trailing_holdout_nov23.json'
    controls = read_json(source)['cases']
    saved = {}
    for name, inputs, windows, reference, cap in (
        ('earlier', old_inputs, EARLIER, controls, 12),
        ('joint', recent_inputs, JOINT, controls, 6),
        ('holdout', holdout_inputs, {'nov20_four_hours': lambda _: True}, read_json(holdout)['cases'], 12),
    ):
        saved[name] = dict(zip(('cases', 'verification'), score(inputs(), windows, reference, cap), strict=True))
    write_json(ROOT / 'tp1_close_confirmation_nov24.json', dict(
        samples=saved, policies=POLICIES, close_freshness_ms=CLOSE_FRESHNESS_MS,
        controls={p.name: file_hash(p) for p in (source, holdout, Path(__file__))},
        live_activation_supported=False, limitations=[
            'Existing November entries and contemporaneous public depth, not account execution records.',
            'Only the crypto runner changes when the simulated filled lot cannot place the requested 5% TP1.',
            'The joint cohort uses an experimental stock continuation signal, not the production stock model.',
            'The clocks approximate stop acknowledgments. The previous interval needs a print in its final 10 seconds.',
            'No month is complete and realized drawdown excludes changes in unrealized PnL.',
        ]))


if __name__ == '__main__':
    main()
