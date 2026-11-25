"""Price causal 15/45-second entry confirmation against real OKX market archives."""

from bisect import bisect_left
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median
from unittest.mock import patch

import stock_crypto_release_reaction_nov24 as shared
from okx_cross_bot_overlap_nov21 import DATA as CRYPTO_BOOK_DATA
from okx_post_stop_reclaim_nov23 import with_actual_fill_path
from okx_recent_trade_repair_nov21 import load_repaired_tape
from okx_selective_single_target_nov23 import EARLIER, cash_summary, key
from okx_spot_confirmation_nov22 import load_native
from okx_unbankable_tp1_nov21 import old_inputs, read_json
from okx_xperp_cash_tick_nov19 import CLOSE_LAGS, simulate
from okx_xperp_l2_entry_replay_nov19 import attempt
from okx_xperp_multiday_capture_nov20 import book_files
from okx_xperp_multiday_replay_nov20 import ROOT, exit_books, file_hash, key_for, pull_samples, resolve_all
from okx_xperp_tick_path_nov19 import reconcile
from src.okx_trader import round_to_tick
from stock_contract_spread_holdout_nov23 import DATA as STOCK_BOOK_DATA
from stock_liquid_quote_routes_nov21 import write_json


POLICIES = {'delay_15_seconds': (15_000, 'clock'), 'confirm_15_seconds': (15_000, 'progress'),
            'confirm_45_seconds': (45_000, 'progress'), 'reclaim_45_seconds': (45_000, 'reclaim')}
FIELDS = ('status', 'trades', 'return_pct', 'drawdown_pct', 'contribution_pct')


def confirmation(row, times, trades, *, at, rule, tick):
    start = row['entry_ms']
    side = 1 if row['direction'] == 'LONG' else -1
    entry = row['entry'] * side
    stop = round_to_tick(row['stop'], tick) * side
    target = row['target'] if row['bot'] == 'stocks' else row['entry']+.5*(row['entry']-row['stop'])
    near = round_to_tick(target, tick) * side
    risk = entry - stop
    if not risk > 0:
        return 'invalid_original_geometry'
    # A matched trade at the request time is excluded from the decision.
    first = bisect_left(times, start)
    midpoint = bisect_left(times, (start+at)//2)
    end = bisect_left(times, at)
    seen = [side * trade.price for trade in trades[first:end]]
    if any(price <= stop or price >= near for price in seen):
        return 'original_bracket_touched'
    if rule == 'clock':
        return 'confirmed'
    if len(set(times[first:end])) < 3 or midpoint == first or midpoint == end:
        return 'sparse_public_trades'
    head, tail = seen[:midpoint-first], seen[midpoint-first:]
    if rule == 'reclaim':
        passed = min(head) <= entry-.1*risk and tail[-1] >= entry
    else:
        passed = tail[-1] >= entry and median(tail) >= max(entry, median(head))+.05*risk
    return 'confirmed' if passed else 'confirmation_missing'


def load_evidence(rows, trades, references):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row['symbol']].append(row)
    frames = load_native({s: references[s] for s in grouped})
    tapes, verified = {}, {}
    for symbol, own in grouped.items():
        events = load_repaired_tape(trades[symbol])
        times = [t.ms for t in events]
        tapes[symbol] = times, events
        mismatches = reconcile(events, frames[symbol], start=min(r['epoch'] for r in own),
                               end=max(r['epoch'] for r in own)+900)
        verified[symbol] = all(r['epoch'] not in mismatches for r in own)
    return tapes, verified


def quote_points(rows, specs, archives):
    stamps = defaultdict(set)
    for row in rows:
        stamps[key_for(row['symbol'], row['entry_ms'])].add(row['entry_ms'])
    missing = [r for r in rows if 'path' not in archives.get(key_for(r['symbol'], r['entry_ms']), {})]
    if missing:
        archives.update(book_files(dict(rows=missing, specs=specs)))
    jobs = []
    for bucket, points in sorted(stamps.items()):
        day, symbol = bucket.split(':', 1)
        item = archives.get(bucket, {})
        if 'path' in item:
            jobs.append((symbol, int(day), item, sorted(points)))
    return pull_samples(jobs, stage='entry confirmation books')


def compare(rows, specs, ticks, books, archives, trades, references, expected, *, vary=('crypto', 'stocks')):
    checked = [r for r in rows if r['bot'] in vary and r['entry_status'] == 'eligible'
               and (r['bot']!='stocks' or r['stock_reference_ready'])]
    relevant_specs = {symbol:specs[symbol] for symbol in {r['symbol'] for r in checked}}
    tapes, verified = load_evidence(checked, trades, references)
    pending = defaultdict(dict)
    reasons = defaultdict(Counter)
    for policy, (wait, rule) in POLICIES.items():
        for original in checked:
            identity = key(original)
            symbol = original['symbol']
            at = original['entry_ms'] + wait
            if not verified[symbol]:
                status = 'unknown_confirmation_tape'
            elif at >= (original['epoch']+180)*1000:
                status = 'entry_deadline_passed'
            else:
                status = confirmation(original, *tapes[symbol], at=at, rule=rule, tick=ticks[symbol])
            if status == 'confirmed':
                pending[policy][identity] = dict(original, entry_ms=at)
            else:
                pending[policy][identity] = dict(original, entry_status=status)
            reasons[policy][status] += 1
    to_quote = [r for proposals in pending.values() for r in proposals.values()
                if r['entry_status'] == 'eligible']
    quotes, quote_checks = quote_points(to_quote, relevant_specs, archives)
    for symbol, sample in quotes.items():
        books[symbol].update(sample)
    active = {}
    for policy, proposals in pending.items():
        for original in checked:
            identity = key(original)
            row = proposals[identity]
            if row['entry_status'] != 'eligible':
                continue
            symbol, at = row['symbol'], row['entry_ms']
            spec, tick = specs[symbol], ticks[symbol]
            near = row['target'] if row['bot'] == 'stocks' else row['entry']+.5*(row['entry']-row['stop'])
            first = attempt(quotes[symbol].get(at), at, row['direction'], spec,
                            round_to_tick(row['stop'], tick), round_to_tick(near, tick))
            reason = first['status']
            if reason == 'eligible':
                worst = first['worst']
                reward_r = abs(row['target']-worst)/abs(worst-row['stop'])
                backing = (row.get('native_candidate') or {}).get('backing_level')
                side = 1 if row['direction'] == 'LONG' else -1
                if reward_r < 1.5 or (backing is not None and side*(worst-backing) < 0):
                    reason = 'confirmation_quote_lost_geometry'
                else:
                    row['reference_entry'] = first['average']
                    active.setdefault(key(row), row)
            row['entry_status'] = reason
            reasons[policy][f'quote:{reason}'] += 1

    rebuilt = list(active.values())
    instruments = {specs[s]['instId']:s for s in tapes}
    trade_checks = resolve_all(rebuilt, {'files':trades}, {'symbols':references}, ticks,
                  trade_loader=lambda item:tapes[instruments[item['instrument']]][1])
    for collection in pending.values():
        for row in collection.values():
            if row['entry_status'] == 'eligible':
                row['path'] = active[key(row)]['path']
    extra, close_checks = exit_books(rebuilt, {'books':archives}, specs, books)
    repricing = dict(repriced=[], extra_exit_checks={})
    replay = with_actual_fill_path(trades, archives|extra, repricing)
    results = {}
    for lag in CLOSE_LAGS:
        params = dict(clock=60, lag=lag, setting='requested5')
        baseline = simulate(rows, specs, ticks, books, **params)
        if any(baseline.get(f) != expected[str(lag)].get(f) for f in FIELDS):
            raise ValueError('cash control changed before delay')
        outcomes = dict(baseline=cash_summary(baseline, {}))
        for policy, proposals in pending.items():
            unresolved = [k for k,r in proposals.items() if r['entry_status'] in (
                'unknown_quote', 'unknown_confirmation_tape')]
            if unresolved:
                outcomes[policy] = dict(status='UNSCORABLE', no_entry_evidence=unresolved)
                continue
            changes = [proposals.get(key(original), original) for original in rows]
            outcomes[policy] = cash_summary(replay(changes, specs, ticks, books, **params), {})
        results[str(lag)] = outcomes
        print(lag, {k:(v.get('trades'), v.get('return_pct'), v['status']) for k,v in outcomes.items()}, flush=True)
    return dict(results=results, decisions=reasons, entry_books=quote_checks, exit_books=close_checks,
                tape_checks=trade_checks, repricing=repricing, original_executable_entries=len(checked))


def older():
    rows, specs, ticks, books, archives, trades, refs = old_inputs()
    expected = read_json(ROOT/'okx_macro_reaction_cash_nov23.json')['exact_cash']
    outputs = {}
    for name, keep in EARLIER.items():
        print(name, flush=True)
        control = {str(lag):expected[name][str(lag)]['baseline'] for lag in CLOSE_LAGS}
        outputs[name] = compare([r for r in rows if keep(r)], specs, ticks, books, archives, trades, refs, control)
    return outputs


def native_joint():
    sources = read_json(CRYPTO_BOOK_DATA/'crypto_capture.json')['trades']
    repair = read_json(STOCK_BOOK_DATA/'trade_repair.json')['intervals']
    equity = read_json(STOCK_BOOK_DATA/'capture.json')['trades']
    stocks = {symbol:dict(item, repaired=repair.get(symbol, {})) for symbol,item in equity.items()}
    archives = read_json(CRYPTO_BOOK_DATA/'crypto_books.json')['books']

    def verify(rows, specs, ticks, books, refs, _unused, control):
        return compare(rows, specs, ticks, books, archives, sources|stocks, refs, control)

    with patch.object(shared, 'compare', verify):
        return {'native_joint':shared.exact_shared([])}


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('part', choices=('older', 'native_joint'))
    part = parser.parse_args().part
    outputs = {'older':older, 'native_joint':native_joint}[part]()
    write_json(ROOT/f'first_print_confirmation_{part}_nov25.json', dict(cases=outputs,
        policies=POLICIES, script_sha256=file_hash(Path(__file__)), live_activation_supported=False,
        limitations=['Only original quote-qualified candidates can wait for confirmation. Earlier refused signals are not backfilled.',
        'No published stop or payoff is recalculated before the actual delayed entry. The shared cash ledger recomputes concurrency and closes.',
        'This uses historical observed prices, traded quantity and public order books. It does not measure private broker cash or monthly results.']))


if __name__ == '__main__':
    main()
