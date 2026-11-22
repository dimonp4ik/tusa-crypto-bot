"""Replay causal stop changes without altering the original market-entry brackets."""

from bisect import bisect_right
from collections import Counter
from dataclasses import dataclass
import json
import sys

from okx_cross_bot_overlap_nov21 import STOCK_DATA
from okx_recent_trade_repair_nov21 import load_repaired_tape
from okx_selective_single_target_nov23 import EARLIER, JOINT, cash_summary, key
from okx_unbankable_tp1_nov21 import old_inputs, recent_inputs
from okx_xperp_cash_tick_nov19 import CAPITAL, CLOSE_LAGS, FRACTION, LEVERAGE, simulate
from okx_xperp_multiday_replay_nov20 import HOLD, ROOT, exit_books, file_hash, resolve_all
from src.okx_trader import round_to_tick


@dataclass(frozen=True)
class StopRule:
    arm_r: float
    distance_r: float | None = None


# Freeze a small grid before reading the differentiated exits. A missing distance means BE.
RULES = {
    'be_after_075r': StopRule(.75),
    'be_after_1r': StopRule(1.),
    'trail_075r_distance_050r': StopRule(.75, .50),
    'trail_1r_distance_050r': StopRule(1., .50),
    'trail_1r_distance_075r': StopRule(1., .75),
    'trail_125r_distance_075r': StopRule(1.25, .75),
}


def resolve_trailing(tape, times, *, signal, entered_at, entry_px, tick_size, amend_seconds):
    rule = RULES[signal['trail_rule']]
    side = 1 if signal['direction'] == 'LONG' else -1
    entry = side * entry_px
    risk = abs(signal['entry'] - signal['stop'])
    stop = side * round_to_tick(signal['stop'], tick_size)
    breakeven = side * round_to_tick(entry_px, tick_size)
    target = side * round_to_tick(signal['target'], tick_size)
    expiry = (int(signal['epoch']) + HOLD[signal['bot']]) * 1000
    if not risk > 0 or not stop < entry < target or entered_at >= expiry:
        return dict(known=False, reason='invalid_entry_relative_to_trailing_bracket')
    interval = amend_seconds * 1000
    arm = entry + risk * rule.arm_r
    activation = None
    desired = stop
    for i in range(bisect_right(times, entered_at), bisect_right(times, expiry - 1)):
        tick = tape[i]
        price = side * tick.price
        # A pre-existing venue trigger always takes priority over a proposed amendment.
        if price <= stop:
            code = 'TRAIL_LOCK' if stop > breakeven else 'BE' if stop == breakeven else 'SL'
        elif price >= target:
            code = 'TARGET'
        else:
            code = None
        if code:
            return dict(known=True, code=code, end_ms=tick.ms, trigger_price=tick.price,
                        bank=None, be_level=side * breakeven, effective_stop=side * stop)
        due = (tick.ms // interval + 1) * interval
        if activation is not None and tick.ms >= activation:
            # Use the first observed print to check that the new trigger is still valid.
            # Never credit a retroactive stop at a price the market has already crossed.
            if desired < price:
                stop = desired
                activation = None
            else:
                activation = due
        if price >= arm:
            proposed = (breakeven if rule.distance_r is None else
                        side * round_to_tick(side * (price - rule.distance_r * risk), tick_size))
            desired = max(desired, proposed)
            if desired > stop and activation is None:
                activation = due
    return dict(known=True, code='TIME', end_ms=expiry, trigger_price=None,
                bank=None, be_level=side * breakeven, effective_stop=side * stop)


def stock_sources(trades, reference):
    capture_path, repair_path = (STOCK_DATA / name for name in ('capture.json', 'trade_repair.json'))
    capture, repair = (json.loads(p.read_bytes()) for p in (capture_path, repair_path))
    if repair['capture_sha256'] != file_hash(capture_path):
        raise ValueError('stock tape repair disagrees with the frozen capture')
    for symbol, intervals in repair['intervals'].items():
        capture['trades'][symbol]['repaired'] = intervals
    return trades | capture['trades'], reference | capture['reference']


def paths_for(rows, ticks, trades, reference):
    variants = {key(r): {} for r in rows}
    rebuilt = []
    for row in rows:
        if row['entry_status'] != 'eligible' or row['bot'] == 'stocks' and not row['stock_reference_ready']:
            continue
        for name in RULES:
            trial = dict(row, trail_rule=name)
            variants[key(row)][name] = trial
            rebuilt.append(trial)
    checks = resolve_all(rebuilt, {'files': trades}, {'symbols': reference}, ticks,
                         trade_loader=load_repaired_tape, path_resolver=resolve_trailing)
    enriched = [dict(r, runner_paths={name: trial['path'] for name, trial in variants[key(r)].items()})
                for r in rows]
    return enriched, rebuilt, checks


def paired_dollars(control, trial):
    base = {key(trade): trade for trade in control.get('closed', [])}
    shifted = {key(trade): trade for trade in trial.get('closed', [])}
    notes = {}
    for identity in sorted(base.keys() & shifted.keys()):
        a, b = base[identity], shifted[identity]
        pnl_a, pnl_b = (t['bank_pnl'] + t['runner_pnl'] for t in (a, b))
        if abs(pnl_b - pnl_a) < 1e-9 and a['code'] == b['code']:
            continue
        transition = f"{a['bot']} {a['code']} -> {b['code']}"
        note = notes.setdefault(transition, dict(positions=0, cash_delta_usd=0.))
        note['positions'] += 1
        note['cash_delta_usd'] += pnl_b - pnl_a
    return dict(transition=notes, removed=len(base.keys() - shifted.keys()),
                added=len(shifted.keys() - base.keys()))


def evaluate(material, cohorts, cap, control, *, joint=False):
    rows, specs, ticks, books, archive, trades, reference = material
    if joint:
        trades, reference = stock_sources(trades, reference)
    proposals, paths, tape_checks = paths_for(rows, ticks, trades, reference)
    print(f'{len(paths)} market paths reconciled', flush=True)
    needed = paths + [r for r in rows if r['entry_status'] == 'eligible']
    extra_books, book_checks = exit_books(needed, {'books': archive}, specs, books)
    outcomes = {}
    assignments = {'baseline': (None, None)}
    assignments.update({f'crypto_{name}': (name, None) for name in RULES})
    if joint:
        assignments.update({f'stocks_{name}': (None, name) for name in RULES})
        both = 'trail_1r_distance_075r'
        assignments['both_trail_1r_distance_075r'] = (both, both)
    for window, keep in cohorts.items():
        chosen = [row for row in proposals if keep(row)]
        outcomes[window] = {}
        for lag in CLOSE_LAGS:
            for clock in (60, 900):
                run = dict(clock=clock, lag=lag, setting='requested5', max_positions=cap)
                baseline = simulate(chosen, specs, ticks, books, **run)
                if clock == 60:
                    expected = control['cases'][window]['baseline'][str(lag)]['requested5']
                    for field in ('trades', 'return_pct', 'drawdown_pct', 'contribution_pct'):
                        if baseline[field] != expected[field]:
                            raise ValueError(f'published baseline disagrees in {window}: {field}')
                if 'closed' not in baseline:
                    outcomes[window][f'{clock}s_{lag}ms'] = {'baseline': baseline}
                    continue
                cases = {}
                for label, (crypto, stocks) in assignments.items():
                    candidates = [dict(r, path=r['runner_paths'][stocks])
                                  if stocks and r['bot'] == 'stocks' and r['runner_paths'] else r for r in chosen]
                    replay = (baseline if label == 'baseline' else
                              simulate(candidates, specs, ticks, books, **run, unbankable_runner=crypto))
                    summary = cash_summary(replay, {})
                    summary.pop('routed', None)
                    cases[label] = (dict(summary, comparison=paired_dollars(baseline, replay))
                                    if 'closed' in replay else summary)
                outcomes[window][f'{clock}s_{lag}ms'] = cases
                print(window, clock, lag, {k: round(v['cash_pnl_usd'], 3) if 'cash_pnl_usd' in v else v['reason']
                                           for k,v in cases.items()}, flush=True)
    return outcomes, dict(tape=tape_checks, exits=book_checks,
        additional_zip_sha256={k: v['sha256'] for k,v in extra_books.items()},
        eligible=Counter(r['bot'] for r in paths if r['trail_rule'] == next(iter(RULES))))


def main():
    reference = ROOT / 'okx_tp1_cash_sensitivity_nov22.json'
    control = json.loads(reference.read_bytes())
    earlier, earlier_sources = evaluate(old_inputs(), EARLIER, 12, control)
    joint, joint_sources = evaluate(recent_inputs(), JOINT, 6, control, joint=True)
    report = dict(capital_usd=CAPITAL, margin_fraction=FRACTION, leverage=LEVERAGE,
                  baseline_sha256=file_hash(reference), rules={k: vars(v) for k,v in RULES.items()},
                  cases=earlier | joint, source_checks=dict(earlier=earlier_sources, joint=joint_sources),
                  live_activation_supported=False, limitations=[
                      'The source cohorts start fresh and cover previously studied dates, not independent months.',
                      'Crypto stop changes are conditional on the actual filled-lot model having no valid requested TP1 partial.',
                      'The stock rows are an experimental continuation route, separate from the live native strategy.',
                      'Clock choices approximate exchange amendment acknowledgements using the first observed print at or after the boundary.',
                      'Quotes are verified public books, not account fills; drawdown is based on realized cash.',
                  ])
    target = ROOT / 'okx_delayed_trailing_cash_nov23.json'
    target.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8', newline='\n')


def holdout():
    from crypto_scan_priority_holdout_nov22 import DATA, load_cohort
    from crypto_scan_priority_nov22 import book_cadence
    from okx_spot_confirmation_holdout_nov22 import execution_samples

    capture_path = DATA / 'crypto_capture.json'
    control_path = ROOT / 'crypto_scan_priority_holdout_nov22.json'
    captured, proposals, _ = load_cohort(capture_path)
    rows, books, entry_evidence = execution_samples(captured, proposals, file_hash(capture_path))
    selected = list(book_cadence(rows, 'alphabetical'))
    if any(r['entry_status'] == 'unknown_quote' for r in selected):
        raise ValueError('the frozen holdout has unknown entry quotes')
    specs = captured['specs']
    ticks = {symbol: float(item['tickSz']) for symbol, item in specs.items()}
    prior = json.loads(control_path.read_bytes())['results']['alphabetical']
    window = 'nov20_four_hours'
    control = {'cases': {window: {'baseline': {lag: {'requested5': value} for lag,value in prior.items()}}}}
    material = (selected, specs, ticks, books, {}, captured['trades'], captured['references'])
    outcomes, sources = evaluate(material, {window: lambda _: True}, 12, control)
    report = dict(cases=outcomes, source_checks=sources, entry_evidence=entry_evidence,
                  source_hashes={p.name: file_hash(p) for p in (capture_path, control_path)},
                  rules={k: vars(v) for k,v in RULES.items()}, live_activation_supported=False,
                  limitations=['Disjoint four-hour entry window; all 12-hour outcome horizons fit in the archives.',
                               'Crypto only. The baseline and trial start at $130; outcomes are neither account nor monthly returns.'])
    target = ROOT / 'okx_delayed_trailing_holdout_nov23.json'
    target.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8', newline='\n')


def native_stocks():
    import stock_native_geometry_match_nov22 as native

    cases, specs, ticks, books, tapes, _, _ = native.prepare_cases(native.freeze())
    rows = cases['production_1pct']
    reference = ROOT / f'{native.STEM}.json'
    expected = json.loads(reference.read_bytes())
    routed = {'baseline': rows}
    for name in RULES:
        candidates = []
        for row in rows:
            symbol = row['symbol']
            times, tape = tapes[symbol]
            path = resolve_trailing(tape, times, signal=dict(row, trail_rule=name),
                                    entered_at=row['entry_ms'], entry_px=row['reference_entry'],
                                    tick_size=ticks[symbol], amend_seconds=60)
            candidates.append(dict(row, path={60: path}))
        routed[name] = candidates
    outcomes = {}
    for bps in (5, 15, 30):
        variants = {name: native.cash_view(candidates, specs, ticks, books, tapes, bps)
                    for name,candidates in routed.items()}
        if variants['baseline'] != expected['results']['production_1pct'][str(bps)]:
            raise ValueError('published native stock baseline did not reproduce')
        outcomes[str(bps)] = variants
        print('native_stocks', bps, {k: (v.get('trades'), v.get('cash_pnl_usd'), v['status'])
                                     for k,v in variants.items()}, flush=True)
    paths = [row for candidates in routed.values() for row in candidates]
    archived, checks = exit_books(paths, {'books': {}}, specs, books)
    exact = {}
    for lag in CLOSE_LAGS:
        exact[str(lag)] = {
            name: cash_summary(simulate(list(native.limit_bursts(candidates, max_same_side=2)),
                                   specs, ticks, books, clock=60, lag=lag, setting='requested5'), {})
            for name,candidates in routed.items()
        }
        print('native_stocks_l2', lag, {k: (v.get('cash_pnl_usd'), v.get('reason'))
                                       for k,v in exact[str(lag)].items()}, flush=True)
    report = dict(cases=outcomes, source_sha256=file_hash(reference),
                  source_input_hashes=expected['source_hashes'], candidates=len(rows),
                  exact_book_cases=exact, exit_book_checks=checks,
                  downloaded_book_sha256={k: v['sha256'] for k,v in archived.items()},
                  rules={k: vars(v) for k,v in RULES.items()}, live_activation_supported=False,
                  limitations=['Native stock admission and observed entry books with verified public contract prints.',
                               'The print-based stress results have separate assumptions from the exact public book replay.',
                               'Stock-only stress test using the earlier observer cohort. Joint exact stock results use the continuation route.'])
    target = ROOT / 'okx_delayed_trailing_native_stocks_nov23.json'
    target.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8', newline='\n')


if __name__ == '__main__':
    if sys.argv[1:] == ['--holdout']:
        holdout()
    elif sys.argv[1:] == ['--stocks']:
        native_stocks()
    else:
        main()
