"""Compare conditional single exits on verified OKX prints and order books."""

from bisect import bisect_right
from collections import Counter
from decimal import Decimal
import json
import sys

from okx_cross_bot_overlap_nov21 import STOCK_DATA
from okx_entry_screen_cash_nov21 import entry_features
from okx_entry_screen_holdout_nov21 import ENTRY_END as SPLIT
from okx_recent_trade_repair_nov21 import load_repaired_tape
from okx_unbankable_tp1_nov21 import old_inputs, recent_inputs
from okx_xperp_cash_tick_nov19 import CAPITAL, CLOSE_LAGS, FRACTION, LEVERAGE, simulate
from okx_xperp_multiday_replay_nov20 import HOLD, ROOT, exit_books, file_hash, resolve_all
from okx_xperp_tick_path_nov19 import resolve_ticks
from src.execution_guard import execution_quote
from src.okx_trader import calc_contracts, round_to_tick
from src.tp1_plan import plan_tp1_close


POLICIES = {
    "indivisible_tp1": ("crypto", "indivisible", .5, False),
    "indivisible_1r": ("crypto", "indivisible", 1., False),
    "small_tp1": ("crypto", "small", .5, False),
    "small_1r": ("crypto", "small", 1., False),
    "small_1_5r": ("crypto", "small", 1.5, False),
    "adverse_book_1r": ("crypto", "adverse", 1., False),
    "adverse_book_1_5r": ("crypto", "adverse", 1.5, False),
    "small_1r_with_breakeven": ("crypto", "small", 1., True),
    "small_1_5r_with_breakeven": ("crypto", "small", 1.5, True),
    "adverse_book_1_5r_with_breakeven": ("crypto", "adverse", 1.5, True),
    "stocks_adverse_book_1_5r": ("stocks", "adverse", 1.5, False),
}


def single_exit(tape, times, *, signal, entered_at, entry_px, tick_size, amend_seconds):
    if signal.get('breakeven_after_half_r'):
        path = resolve_ticks(tape, times, signal=signal, entered_at=entered_at, entry_px=entry_px,
                             tick_size=tick_size, amend_seconds=amend_seconds)
        if not path['known']:
            return path
        code = {'TARGET': 'SINGLE_TARGET', 'SL_AFTER_TP1': 'SL_AFTER_FAVORABLE_MOVE'}.get(
            path['code'], path['code'])
        return dict(path, code=code, bank=None)
    sign = 1 if signal['direction'] == 'LONG' else -1
    stop = round_to_tick(signal['stop'], tick_size)
    target = round_to_tick(signal['target'], tick_size)
    expiry = (int(signal['epoch']) + HOLD[signal['bot']]) * 1000
    if not sign * stop < sign * entry_px < sign * target or not entered_at < expiry:
        return dict(known=False, reason='invalid_entry_relative_to_single_target')
    entry_tick = round_to_tick(entry_px, tick_size)
    for index in range(bisect_right(times, entered_at), bisect_right(times, expiry - 1)):
        trade = tape[index]
        if sign * (trade.price - stop) <= 0:
            code = 'SL'
        elif sign * (trade.price - target) >= 0:
            code = 'SINGLE_TARGET'
        else:
            continue
        return dict(known=True, code=code, end_ms=trade.ms, trigger_price=trade.price,
                    bank=None, be_level=entry_tick)
    return dict(known=True, code='TIME', end_ms=expiry, trigger_price=None,
                bank=None, be_level=entry_tick)


def policy_flags(row, spec, book):
    quote = execution_quote(book, row['direction'], None, row['entry_ms'],
                            size_for_price=lambda px: calc_contracts(CAPITAL * FRACTION, LEVERAGE, px, spec))
    quantity = Decimal(str(quote.contracts))
    bank = plan_tp1_close(quote.contracts, 5., spec, ensure_minimum=True)
    minimum = Decimal(str(bank['minimum_size']))
    if minimum <= 0:
        raise ValueError('invalid minimum lot for a scored entry')
    return dict(indivisible=bank['full_close'], small=quantity <= 4 * minimum,
                adverse=entry_features(row, book)['aligned_imbalance_10bps'] < -.30)


def key(row):
    return row['bot'], row['symbol'], row['entry_ms']


def variants_for(rows, specs, ticks, books):
    rows_by_key = {key(row): row for row in rows}
    if len(rows_by_key) != len(rows):
        raise ValueError('duplicate source entries')
    variants = {name: {} for name in POLICIES}
    paths = {}
    for identity, row in rows_by_key.items():
        if row['entry_status'] != 'eligible' or row['bot'] == 'stocks' and not row['stock_reference_ready']:
            continue
        symbol = row['symbol']
        flags = policy_flags(row, specs[symbol], books[symbol][row['entry_ms']])
        for name, (bot, condition, multiple, guard) in POLICIES.items():
            if bot != row['bot'] or not flags[condition]:
                continue
            target = round_to_tick(row['entry'] + multiple * (row['entry'] - row['stop']), ticks[symbol])
            sign = 1 if row['direction'] == 'LONG' else -1
            if sign * (target - row['target']) >= 0 or sign * (target - row['reference_entry']) <= 0:
                continue
            variant = paths.setdefault((identity, target, guard), dict(
                row, target=target, breakeven_after_half_r=guard))
            variants[name][identity] = variant
    return variants, list(paths.values())


def cash_summary(result, selected):
    summary = {key: value for key, value in result.items() if key != 'closed'}
    if 'closed' not in result:
        return summary
    pnl = [row['bank_pnl'] + row['runner_pnl'] for row in result['closed']]
    return dict(summary, cash_pnl_usd=sum(pnl), profitable=sum(value > 0 for value in pnl),
                routed=sum(key(row) in selected for row in result['closed']),
                exit_counts=Counter(row['code'] for row in result['closed']))


def reconciled_variants(material, *, joint):
    rows, specs, ticks, books, archive, trades, reference = material
    stock_tape_hashes = {}
    if joint:
        capture_path = STOCK_DATA / 'capture.json'
        repair_path = STOCK_DATA / 'trade_repair.json'
        capture = json.loads(capture_path.read_bytes())
        repair = json.loads(repair_path.read_bytes())
        if repair['capture_sha256'] != file_hash(capture_path):
            raise ValueError('stock tape repair disagrees with the frozen capture')
        for symbol, intervals in repair['intervals'].items():
            capture['trades'][symbol]['repaired'] = intervals
        trades = trades | capture['trades']
        reference = reference | capture['reference']
        stock_tape_hashes = {p.name: file_hash(p) for p in (capture_path, repair_path)}
    variants, paths = variants_for(rows, specs, ticks, books)
    tape_checks = resolve_all(paths, {'files': trades}, {'symbols': reference}, ticks,
                              trade_loader=load_repaired_tape, path_resolver=single_exit)
    print(f'{len(paths)} trial paths reconciled against public trade prints', flush=True)
    return variants, paths, tape_checks, stock_tape_hashes


def evaluate(material, cohorts, cap, control, *, joint=False):
    rows, specs, ticks, books, archive, *_ = material
    variants, paths, tape_checks, stock_tape_hashes = reconciled_variants(material, joint=joint)
    new_books, book_checks = exit_books(paths, {'books': archive}, specs, books)
    report = {}
    for name, choose in cohorts.items():
        selected_rows = [row for row in rows if choose(row)]
        report[name] = {}
        for lag in CLOSE_LAGS:
            params = dict(clock=60, lag=lag, setting='requested5', max_positions=cap)
            baseline = cash_summary(simulate(selected_rows, specs, ticks, books, **params), {})
            expected = control['cases'][name]['baseline'][str(lag)]['requested5']
            for field in ('trades', 'return_pct', 'drawdown_pct', 'contribution_pct'):
                if baseline[field] != expected[field]:
                    raise ValueError(f'published baseline disagrees in {name}: {field}')
            results = {'baseline': baseline}
            for label, variant in variants.items():
                candidates = [variant.get(key(row), row) for row in selected_rows]
                results[label] = cash_summary(simulate(candidates, specs, ticks, books, **params), variant)
            report[name][str(lag)] = results
            print(name, lag, {label: (round(result['cash_pnl_usd'], 3), result['routed'])
                              if 'cash_pnl_usd' in result else result['reason']
                              for label, result in results.items()}, flush=True)
    return report, dict(tape=tape_checks, books=book_checks, stock_tape_sha256=stock_tape_hashes,
                        added_book_archive_sha256={k: v['sha256'] for k, v in new_books.items()})


def explain_changes(material, cohorts, cap, *, joint=False):
    rows, specs, ticks, books, *_ = material
    variants, *_ = reconciled_variants(material, joint=joint)
    source_by_key = {key(row): row for row in rows}
    report = {}
    for name, choose in cohorts.items():
        selected = [row for row in rows if choose(row)]
        baseline = simulate(selected, specs, ticks, books, clock=60, lag=200,
                            setting='requested5', max_positions=cap)
        report[name] = {}
        for label, variant in variants.items():
            outcomes = Counter()
            for trade in baseline['closed']:
                identity = key(trade)
                if identity not in variant:
                    continue
                original = source_by_key[identity]['path'][60]
                candidate = variant[identity]['path'][60]
                if not candidate['known']:
                    outcomes['unknown_alternate_path'] += 1
                    continue
                outcomes[f"{original['code']} -> {candidate['code']}"] += 1
            report[name][label] = outcomes
        print(name, json.dumps(report[name]), flush=True)
    return report


EARLIER = {'nov05_08': lambda row: row['epoch'] < SPLIT,
           'nov09_14': lambda row: row['epoch'] >= SPLIT}
JOINT = {'nov17_19_joint': lambda _: True}
OUTPUT = ROOT / 'okx_selective_single_target_nov23.json'


def main():
    baseline_path = ROOT / 'okx_tp1_cash_sensitivity_nov22.json'
    control = json.loads(baseline_path.read_bytes())
    earlier, earlier_sources = evaluate(old_inputs(), EARLIER, 12, control)
    joint, joint_sources = evaluate(recent_inputs(), JOINT, 6, control, joint=True)
    report = dict(capital_usd=CAPITAL, margin_fraction=FRACTION, leverage=LEVERAGE,
                  baseline_sha256=file_hash(baseline_path), policies=POLICIES, cases=earlier | joint,
                  source_checks=dict(earlier=earlier_sources, joint=joint_sources),
                  live_activation_supported=False, limitations=[
                      'Retrospective cohorts with separate initial balances. The equity drawdown of open positions is unavailable.',
                      'Route classification uses the displayed executable size for the initial account balance at each entry.',
                      'A routed position has one full-size market take. Guarded variants can move the stop to entry after a 0.5R favorable move.',
                      'Exchange market exits are estimated from public depth at 200 ms and at 1000 ms after the trigger.',
                      'Entry and pre-admission rules are held to the existing cohort. Account limits match the control.',
                  ])
    OUTPUT.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8', newline='\n')


def append_diagnostics():
    report = json.loads(OUTPUT.read_bytes())
    expected_policies = json.loads(json.dumps(POLICIES))
    if report['policies'] != expected_policies:
        raise ValueError('cash report must be rerun before adding diagnostics')
    earlier = explain_changes(old_inputs(), EARLIER, 12)
    joint = explain_changes(recent_inputs(), JOINT, 6, joint=True)
    report['paired_trigger_diagnostics'] = earlier | joint
    note = 'Trigger comparisons hold baseline admissions fixed; the cash runs can admit different trades.'
    if note not in report['limitations']:
        report['limitations'].append(note)
    OUTPUT.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8', newline='\n')


if __name__ == '__main__':
    if sys.argv[1:] == ['--diagnose']:
        append_diagnostics()
    else:
        main()
