"""Price early failure exits on verified OKX trades and order books."""

from bisect import bisect_left, bisect_right
from collections import Counter, defaultdict
from pathlib import Path
from unittest.mock import patch

import cash_loss_pause_audit_nov24 as fresh_data
import first_print_confirmation_nov25 as entry_data
import okx_xperp_cash_tick_nov19 as cash
from okx_post_close_cash_nov23 import trade_difference
from okx_post_stop_reclaim_nov23 import with_actual_fill_path
from okx_recent_trade_repair_nov21 import load_repaired_tape
from okx_selective_single_target_nov23 import EARLIER, cash_summary
from okx_spot_confirmation_nov22 import load_native
from okx_unbankable_tp1_nov21 import old_inputs, read_json
from okx_xperp_multiday_replay_nov20 import ROOT, exit_books, file_hash
from okx_xperp_tick_path_nov19 import reconcile
from src.okx_trader import round_to_tick
from stock_liquid_quote_routes_nov21 import write_json


MINUTE = 60_000
MAX_PRINT_AGE = MINUTE
MIN_ADVERSE = .40
MAX_PROGRESS = .20
POLICIES = {'five_minutes': (5,), 'fifteen_minutes': (15,), 'five_or_fifteen_minutes': (5, 15)}


def check_stamp(row, minutes):
    """Observe at the first minute poll at or beyond the scheduled hold age."""
    return ((row['entry_ms'] + MINUTE - 1) // MINUTE + minutes) * MINUTE


def cut(row, fill, path, lag, *, times, trades, mismatches, tick, minutes):
    side = 1 if row['direction'] == 'LONG' else -1
    risk = side * (fill - round_to_tick(row['stop'], tick))
    if risk <= 0:
        raise ValueError('nonpositive original stop risk')
    first = bisect_right(times, row['entry_ms'])
    for age in minutes:
        decision = check_stamp(row, age)
        # Original exchange orders take precedence through the market-exit lag.
        if path['end_ms'] <= decision + lag or (path['bank'] and path['bank']['ms'] <= decision + lag):
            break
        start = row['entry_ms'] // 900_000 * 900
        end = (decision - 1) // 900_000 * 900
        if any(stamp in mismatches for stamp in range(start, end + 900, 900)):
            raise ValueError('missing verified tape for early-exit decision')
        last = bisect_left(times, decision)
        if last <= first or times[last-1] == times[first] or decision - times[last-1] > MAX_PRINT_AGE:
            continue
        last_move = side * (trades[last-1].price - fill) / risk
        if last_move > -MIN_ADVERSE:
            continue
        progress = max(side * (event.price - fill) for event in trades[first:last]) / risk
        if progress >= MAX_PROGRESS:
            continue
        return dict(path, code='EARLY_FAILURE', end_ms=decision, trigger_price=trades[last-1].price,
                    bank=None, simultaneous_targets=False)
    return path


def tape_inputs(rows, trades, sources):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row['symbol']].append(row)
    frames = load_native({symbol: sources[symbol] for symbol in grouped})
    result = {}
    for symbol, proposals in grouped.items():
        tape = load_repaired_tape(trades[symbol])
        first = min(row['entry_ms'] for row in proposals) // 900_000 * 900
        last_poll = max(check_stamp(row, 15) for row in proposals)
        end = (last_poll - 1) // 900_000 * 900 + 900
        errors = reconcile(tape, frames[symbol], start=first, end=end)
        result[symbol] = dict(times=[event.ms for event in tape], trades=tape, mismatches=errors)
    return result


def compare(rows, specs, ticks, books, archives, trades, references, expected):
    eligible = [r for r in rows if r['entry_status'] == 'eligible'
                and (r['bot'] != 'stocks' or r['stock_reference_ready'])]
    tapes = tape_inputs(eligible, trades, references)
    requests = [
        dict(row, path={60: dict(known=True, end_ms=check_stamp(row, age), bank=None)})
        for row in eligible for age in (5, 15)
        if row['path'][60]['known'] and row['path'][60]['end_ms'] > check_stamp(row, age)
        and (not row['path'][60]['bank'] or row['path'][60]['bank']['ms'] > check_stamp(row, age))
    ]
    additional, book_checks = exit_books(requests, {'books': archives}, specs, books)
    repricing = dict(repriced=[], extra_exit_checks={})
    replay = with_actual_fill_path(trades, archives | additional, repricing)
    output = {}
    for lag in cash.CLOSE_LAGS:
        options = dict(clock=60, lag=lag, setting='requested5')
        baseline = cash.simulate(rows, specs, ticks, books, **options)
        if any(baseline.get(k) != expected[str(lag)].get(k) for k in entry_data.FIELDS):
            raise ValueError('unchanged exit cash control differs from the frozen reference')
        outcomes = dict(baseline=cash_summary(baseline, {}))
        baseline_by_entry = {(r['bot'], r['symbol'], r['entry_ms']): r for r in baseline['closed']}
        for name, schedule in POLICIES.items():
            def manage(row, fill, path, latency):
                symbol = row['symbol']
                return cut(row, fill, path, latency, **tapes[symbol], tick=ticks[symbol], minutes=schedule)

            trial = replay(rows, specs, ticks, books, **options, manage_exit=manage)
            outcomes[name] = cash_summary(trial, {})
            if 'closed' not in trial:
                continue
            cuts = Counter()
            for new in trial['closed']:
                if new['code'] != 'EARLY_FAILURE':
                    continue
                key = new['bot'], new['symbol'], new['entry_ms']
                original = baseline_by_entry.get(key)
                if original is None:
                    cuts[new['bot'], 'not_in_baseline'] += 1
                else:
                    pnl = original['bank_pnl'] + original['runner_pnl']
                    label = 'profitable' if pnl > 0 else 'losing' if pnl < 0 else 'flat'
                    cuts[new['bot'], original['code'], label] += 1
            outcomes[name]['cuts_by_original_result'] = {':'.join(key): count for key, count in cuts.items()}
            outcomes[name]['trade_changes'] = trade_difference(baseline, trial)
        output[str(lag)] = outcomes
        print(lag, {k:(v.get('trades'), v.get('return_pct'), v.get('drawdown_pct'), v.get('reason'),
                       v.get('cuts_by_original_result')) for k, v in outcomes.items()}, flush=True)
    return dict(results=output, possible_exit_quotes=len(requests), book_checks=book_checks,
                mismatch_counts={s:len(t['mismatches']) for s,t in tapes.items()}, repricing=repricing)


def older():
    rows, *inputs = old_inputs()
    source = ROOT / 'first_print_confirmation_older_nov25.json'
    previous = read_json(source)['cases']
    return {name: compare([r for r in rows if keep(r)], *inputs,
                          {lag: results['baseline'] for lag, results in previous[name]['results'].items()})
            for name, keep in EARLIER.items()}


def native_joint():
    with patch.object(entry_data, 'compare', compare):
        return entry_data.native_joint()


def fresh():
    source = ROOT / 'first_print_delay_holdout_nov25.json'
    previous = read_json(source)['scores']['results']
    controls = {lag: results['baseline'] for lag, results in previous.items()}
    def evaluate(material, *_args, **_kwargs):
        return compare(*material, controls)
    with patch.object(fresh_data, 'compare', evaluate):
        cases, evidence = fresh_data.fresh_crypto()
    return {'nov22_crypto_12h': dict(cases, evidence=evidence, source_sha256=file_hash(source))}


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    runners = {'older': older, 'native_joint': native_joint, 'fresh': fresh}
    parser.add_argument('part', choices=runners)
    part = parser.parse_args().part
    cases = runners[part]()
    sources = (Path(__file__), Path(cash.__file__), Path(entry_data.__file__))
    write_json(ROOT / f'first_failed_move_exit_{part}_nov25.json', dict(
        cases=cases, policies=POLICIES, minimum_adverse_r=MIN_ADVERSE, maximum_progress_r=MAX_PROGRESS,
        script_sha256={p.name: file_hash(p) for p in sources}, live_activation_supported=False,
        limitations=[
            'New exits are possible only before the first bracket fill and after a confirmed recent print.',
            'A minute poll triggers a full market close at public order-book prices after a 200 or 1000ms lag.',
            'The shared account reruns position sizes, open slot reservations and trade sequencing.',
            'The original signal selection is fixed. The modeled software close does not verify exchange cancellation races.',
            'Public trades and depth price a historical simulation, not private realized returns or a full month.',
        ]))


if __name__ == '__main__':
    main()
