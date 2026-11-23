"""Replay one causal attempt after a confirmed crypto stop and recovery."""

import argparse
from bisect import bisect_left
from collections import Counter, defaultdict
from datetime import datetime, timezone
import pickle
from pathlib import Path
from unittest.mock import patch

import okx_cross_bot_overlap_nov21 as joint
from okx_crypto_risk_range_nov22 import attach_entry_quotes, merge_books
from okx_post_close_cash_nov23 import trade_difference
from okx_recent_trade_repair_nov21 import load_repaired_tape, repair_recent_days
from okx_selective_single_target_nov23 import EARLIER, JOINT, cash_summary, single_exit
from okx_unbankable_tp1_nov21 import old_inputs, read_json, recent_inputs
from okx_xperp_cash_tick_nov19 import CLOSE_LAGS, RELEASE_MS, simulate
from okx_xperp_multiday_capture_nov20 import local_trade_days
from okx_xperp_multiday_replay_nov20 import HOLD, ROOT, exit_books, file_hash, resolve_all
from okx_xperp_tick_path_nov19 import load_trade_tape, reconcile, resolve_ticks
from src.okx_trader import round_to_tick
from stock_liquid_quote_routes_nov21 import write_json


STEM = 'okx_post_stop_reclaim_nov23'
POLICIES = {'5m_reclaims_quarter_r': (300, .25), '15m_reclaims_original_entry': (900, 1.)}
WAIT_MS = 3_600_000
RISK_BOUNDS = (.005, .02)
MINIMUM_REWARD_R = 1.5
MINIMUM_TARGET_MOVE = .005
ENTRY_OFFSET_MS = 10_000
RETRY_MS = (0, 200, 800)
CRYPTO_HOURLY_LIMIT = 2
FORWARD_START = int(datetime(2026, 11, 21, tzinfo=timezone.utc).timestamp())
FORWARD_STOP = FORWARD_START + 12 * 3600
FORWARD_DATA = ROOT / 'crypto_reclaim_forward_nov21'


def proposal(parent, path, tape, times, tick, policy):
    bar_seconds, reclaim_r = policy
    side = 1 if parent['direction'] == 'LONG' else -1
    risk = abs(parent['entry'] - parent['stop'])
    reclaim = side * parent['stop'] + reclaim_r * risk
    triggered = path['end_ms']
    start = bisect_left(times, triggered)
    duration = bar_seconds * 1000
    first = triggered + max(CLOSE_LAGS) + RELEASE_MS
    maturity = (parent['epoch'] + HOLD['crypto']) * 1000
    end = min(triggered + WAIT_MS, maturity - ENTRY_OFFSET_MS - max(RETRY_MS))
    for confirmed in range((first // duration + 1) * duration, end + 1, duration):
        after = bisect_left(times, confirmed, lo=start)
        if after <= start or not 0 < confirmed - times[after - 1] <= 60_000:
            continue
        last = tape[after - 1]
        if times[after - 1] < confirmed - duration or side * last.price < reclaim:
            continue
        worst_print = min(side * trade.price for trade in tape[start:after])
        stop = round_to_tick(side * (worst_print - tick), tick)
        target = round_to_tick(parent['target'], tick)
        new_risk = side * (last.price - stop)
        reward = side * (target - last.price)
        if (not RISK_BOUNDS[0] <= new_risk / last.price <= RISK_BOUNDS[1]
                or reward < MINIMUM_TARGET_MOVE * last.price or reward < MINIMUM_REWARD_R * new_risk):
            return None, 'first_recovery_geometry_refused'
        return dict(symbol=parent['symbol'], bot='crypto', direction=parent['direction'], epoch=confirmed // 1000,
                    parent_epoch=parent['epoch'], reentry_parent_ms=parent['entry_ms'], previous_stop_ms=triggered,
                    entry=last.price, stop=stop, target=target, reclaim_price=side * reclaim,
                    paper_reference_ready=False), 'proposed'
    return None, 'no_confirmed_recovery'


def validate_tapes(parents, trades, references):
    by_symbol = defaultdict(list)
    for row in parents:
        by_symbol[row['symbol']].append(row)
    tapes, hashes = {}, {}
    for symbol, rows in sorted(by_symbol.items()):
        evidence = references[symbol]
        path = Path(evidence['path'])
        if file_hash(path) != evidence['sha256']:
            raise ValueError('native candle provenance changed')
        tape = load_repaired_tape(trades[symbol])
        frame = pickle.loads(path.read_bytes())
        first = min(row['epoch'] for row in rows)
        limit = HOLD['crypto']
        last = max(row['epoch'] + limit for row in rows)
        relevant = {at for row in rows for at in range(row['epoch'], row['epoch'] + limit, 900)}
        mismatch = reconcile(tape, frame, start=first, end=last)
        if set(mismatch) & relevant:
            raise ValueError(f'public trades disagree with reference bars for {symbol}')
        tapes[symbol] = [trade.ms for trade in tape], tape
        sources = trades[symbol]['days'] | trades[symbol].get('repaired', {})
        hashes[symbol] = dict(bars_checked=len(relevant), candle_sha256=evidence['sha256'],
                              trade_sha256={day: item['sha256'] for day, item in sources.items()})
    return tapes, hashes


def verify_entry(row, tick, tapes):
    if row['entry_status'] != 'eligible':
        return
    side = 1 if row['direction'] == 'LONG' else -1
    worst = row['tries'][-1]['worst']
    risk = side * (worst - row['stop'])
    reward = side * (row['target'] - worst)
    if (side * (worst - row['reclaim_price']) < 0 or not RISK_BOUNDS[0] <= risk / worst <= RISK_BOUNDS[1]
            or reward < max(MINIMUM_TARGET_MOVE * worst, MINIMUM_REWARD_R * risk)):
        row['entry_status'] = 'reentry_quote_geometry_refused'
        return
    times, tape = tapes[row['symbol']]
    row['path'] = {60: single_exit(tape, times, signal=dict(row, epoch=row['parent_epoch']),
                                  entered_at=row['entry_ms'], entry_px=row['reference_entry'],
                                  tick_size=tick, amend_seconds=60)}


def evaluate(material, cohorts, controls, *, cap, cash_replay=simulate):
    rows, specs, ticks, books, archives, trades, references = material
    parents = [row for row in rows if row['bot'] == 'crypto' and row['entry_status'] == 'eligible'
               and row['path'][60].get('known') and row['path'][60]['code'] == 'SL']
    tapes, tape_checks = validate_tapes(parents, trades, references)
    variants = {}
    candidate_checks = {}
    for name, policy in POLICIES.items():
        found = []
        reasons = Counter()
        for parent in parents:
            times, tape = tapes[parent['symbol']]
            row, reason = proposal(parent, parent['path'][60], tape, times, ticks[parent['symbol']], policy)
            reasons[reason] += 1
            if row:
                found.append(row)
        variants[name] = found
        candidate_checks[name] = reasons
    print('Potential original stop paths', len(parents), 'reentry proposals', candidate_checks, flush=True)
    flat = [row for variant in variants.values() for row in variant]
    plan = dict(specs=specs, parameters=dict(crypto_offset_ms=ENTRY_OFFSET_MS, retry_offsets_ms=RETRY_MS))
    reco, added, entry_checks = attach_entry_quotes(flat, plan, archives, ticks)
    merge_books(books, added)
    for row in reco:
        verify_entry(row, ticks[row['symbol']], tapes)
    copied = iter(reco)
    repaired = {name: [next(copied) for _ in proposed] for name, proposed in variants.items()}
    extra, exits = exit_books(reco, {'books': archives}, specs, books)
    outcomes = {}
    for name, keep in cohorts.items():
        control_rows = [row for row in rows if keep(row)]
        by_lag = {}
        for lag in CLOSE_LAGS:
            options = dict(clock=60, lag=lag, setting='requested5', max_positions=cap)
            control = cash_replay(control_rows, specs, ticks, books, **options)
            if name in controls:
                expected = controls[name]['baseline'][str(lag)]['requested5']
                for field in ('trades', 'return_pct', 'drawdown_pct', 'contribution_pct'):
                    if control[field] != expected[field]:
                        raise ValueError(f'published cash control changed in {name}: {field}')
            study_options = dict(options, crypto_publications_per_hour=CRYPTO_HOURLY_LIMIT)
            comparable = cash_replay(control_rows, specs, ticks, books, **study_options)
            if comparable.get('closed') != control.get('closed'):
                raise ValueError(f'rolling budget changes the original portfolio in {name}')
            scores = dict(baseline=cash_summary(control, {}))
            if name not in controls and 'closed' in control:
                planned = {(row['symbol'], row['entry_ms']): row['path'][60]['code'] for row in parents}
                changed_stops = [trade for trade in control['closed'] if trade['bot'] == 'crypto' and trade['code'] == 'SL'
                                 and (trade['symbol'], trade['entry_ms']) not in planned]
                scores['baseline'].update(closed_positions=control['closed'], initially_unplanned_stops=changed_stops)
            for policy, proposed in repaired.items():
                extras = [row for row in proposed if keep(dict(row, epoch=row['parent_epoch']))]
                if any(row['entry_status'] == 'unknown_quote' for row in extras):
                    trial = dict(status='UNSCORABLE_UNKNOWN_REENTRY_BOOK')
                else:
                    trial = cash_replay(control_rows + extras, specs, ticks, books, **study_options)
                summary = cash_summary(trial, {})
                if 'closed' in trial:
                    reentries = [trade for trade in trial['closed'] if 'reentry_parent_ms' in trade]
                    summary.update(routed=len(reentries), reentries=reentries)
                    summary['comparison'] = trade_difference(control, trial)
                scores[policy] = dict(summary, candidate_status=Counter(row['entry_status'] for row in extras))
            by_lag[str(lag)] = scores
            print(name, lag, {label: (result.get('trades'), result.get('routed'), result.get('cash_pnl_usd'))
                  for label, result in scores.items()}, flush=True)
        outcomes[name] = by_lag
    return dict(outcomes=outcomes, parents=len(parents), candidate_checks=candidate_checks,
                entry_book_checks=entry_checks, exit_book_checks=exits, tape_checks=tape_checks,
                extra_book_sha256={key: item['sha256'] for key, item in extra.items()})


def with_actual_fill_path(trades, archives, evidence):
    tapes = {}
    updated = {}

    def cash_replay(rows, specs, ticks, books, **options):
        needed = {}

        def reprice(row, price, clock):
            symbol, at = row['symbol'], row['entry_ms']
            parent = row.get('reentry_parent_ms')
            key = symbol, at, parent, price, clock
            if key not in updated:
                if symbol not in tapes:
                    tape = load_repaired_tape(trades[symbol])
                    tapes[symbol] = [event.ms for event in tape], tape
                times, tape = tapes[symbol]
                explicit = dict(row, epoch=row['parent_epoch']) if parent is not None else row
                resolve = single_exit if parent is not None else resolve_ticks
                path = resolve(tape, times, signal=explicit, entered_at=at, entry_px=price,
                               tick_size=ticks[symbol], amend_seconds=clock)
                updated[key] = dict(row, path={clock: path})
                evidence['repriced'].append(dict(symbol=symbol, entry_ms=at, initial_fill=row['reference_entry'],
                                                 actual_fill=price, reentry_parent_ms=parent, path=path))
            needed[key] = updated[key]
            return updated[key]['path'][clock]

        # A missing newly triggered market exit must be read before cash can
        # change subsequent position sizes. Repeat only when quotes were added.
        for _ in range(len(rows) + 1):
            result = simulate(rows, specs, ticks, books, **options, reprice_path=reprice)
            if result.get('reason') != 'missing exit L2 sample' or not needed:
                return result
            before = sum(map(len, books.values()))
            extra, checks = exit_books(list(needed.values()), {'books': archives}, specs, books)
            archives.update(extra)
            evidence['extra_exit_checks'].update(checks)
            if sum(map(len, books.values())) == before:
                return result
        raise ValueError('unable to reconcile changed market exits in order')

    return cash_replay


def forward():
    manifest = FORWARD_DATA / 'crypto_capture.json'
    with patch.multiple(joint, START=FORWARD_START, ENTRY_END=FORWARD_STOP, DATA=FORWARD_DATA):
        if not manifest.exists():
            joint.capture_crypto()
        captured = read_json(manifest)
        needed = set(local_trade_days(FORWARD_START, FORWARD_STOP).values())
        if any(not needed.issubset(item['days']) for item in captured['trades'].values()):
            raise ValueError('fresh market tape unavailable for the decision window')
        # First freeze proposals using only the available full decision tape.
        # A separate pass covers each eligible proposal through its actual hold.
        def decision_tape(item):
            return load_trade_tape(dict(item, complete=True))

        with patch.multiple(joint, HOLD={'crypto': 900}, load_trade_tape=decision_tape):
            rows, tape_checks = joint.reconstruct(captured)
    print('Fresh frozen proposals', len(rows), flush=True)
    specs = captured['specs']
    ticks = {s: float(row['tickSz']) for s, row in specs.items()}
    archives = {}
    plan = dict(specs=specs, parameters=dict(crypto_offset_ms=ENTRY_OFFSET_MS, retry_offsets_ms=RETRY_MS))
    proposals, books, entry_checks = attach_entry_quotes(rows, plan, archives, ticks)
    if any(row['entry_status'] == 'unknown_quote' for row in proposals):
        raise ValueError('fresh baseline entry quote unavailable')
    archive_trades = captured['trades']
    references = captured['references']
    admitted = list(joint.crypto_cadence(proposals))
    qualified = [row for row in admitted if row['entry_status'] == 'eligible']
    repaired = repair_recent_days(archive_trades, qualified, hold_seconds=HOLD['crypto'],
                                  folder=FORWARD_DATA / 'repaired')
    trades = {symbol: dict(item, repaired=repaired[symbol]) if symbol in repaired else dict(item, complete=True)
              for symbol, item in archive_trades.items()}
    paths = resolve_all(qualified, {'files': trades}, {'symbols': references}, ticks,
                        trade_loader=load_repaired_tape)
    if any(v.get('error') or v['mismatches'] for v in paths.values()):
        raise ValueError('fresh baseline trade paths unavailable')
    extra, exits = exit_books(qualified, {'books': archives}, specs, books)
    all_archives = archives | extra
    changed_fills = dict(repriced=[], extra_exit_checks={})
    cash_replay = with_actual_fill_path(trades, all_archives, changed_fills)
    results = evaluate((admitted, specs, ticks, books, all_archives, trades, references),
                       {'nov21_crypto_forward': lambda _: True}, {}, cap=12, cash_replay=cash_replay)
    return dict(entry_window_utc=[FORWARD_START, FORWARD_STOP], source_sha256=file_hash(manifest),
                baseline_candidate_count=len(proposals), baseline_entries=entry_checks, baseline_exits=exits,
                baseline_tape=tape_checks, baseline_paths=paths, actual_fill_path_checks=changed_fills,
                history_trade_repairs={s: {day: item['sha256'] for day, item in fixes.items()}
                                       for s, fixes in repaired.items()}, **results)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--forward-only', action='store_true', help='refresh the forward cohort of an existing report')
    args = parser.parse_args()
    source = ROOT / 'okx_tp1_cash_sensitivity_nov22.json'
    if args.forward_only:
        path = ROOT / f'{STEM}.json'
        prior = read_json(path)
        if prior['original_cash_control_sha256'] != file_hash(source):
            raise ValueError('historical control changed before forward refresh')
        prior['forward'] = forward()
        write_json(path, prior)
        return
    controls = read_json(source)['cases']
    prior = evaluate(old_inputs(), EARLIER, controls, cap=12)
    recent = evaluate(recent_inputs(), JOINT, controls, cap=6)
    holdout = forward()
    write_json(ROOT / f'{STEM}.json', dict(prior=prior, joint=recent, forward=holdout, policies=POLICIES,
        max_wait_ms=WAIT_MS, risk_bounds_fraction=RISK_BOUNDS, minimum_reward_r=MINIMUM_REWARD_R,
        original_cash_control_sha256=file_hash(source), live_activation_supported=False, limitations=[
            'The parent must have opened, lost at the original stop and released its slot in the very same replay.',
            'A fully closed confirmation candle and a recent real print are required. The original target and expiry persist.',
            'A reentry has one take-profit and participates in the same rolling crypto publication and account limits.',
            'Market geometry and execution are based on public exchange prints and books. Private fills are outside this report.',
            'The joint portfolio includes experimental equity entries and is not independently representative of monthly returns.',
        ]))


if __name__ == '__main__':
    main()
