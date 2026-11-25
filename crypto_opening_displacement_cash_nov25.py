"""Evaluate whether a fast adverse move from the model open predicts failed entries."""

from bisect import bisect_left
from collections import Counter, defaultdict
from pathlib import Path

from okx_delayed_trailing_cash_nov23 import paired_dollars
from okx_recent_trade_repair_nov21 import load_repaired_tape
from okx_selective_single_target_nov23 import EARLIER, JOINT, cash_summary, key
from okx_spot_confirmation_nov22 import load_native
from okx_sweep_context_cash_nov23 import holdout_inputs
from okx_unbankable_tp1_nov21 import old_inputs, read_json, recent_inputs
from okx_xperp_cash_tick_nov19 import CLOSE_LAGS, simulate
from okx_xperp_multiday_replay_nov20 import ROOT, file_hash
from okx_xperp_tick_path_nov19 import reconcile
from stock_liquid_quote_routes_nov21 import write_json


LIMITS = (10, 20, 30, 40)
POLICIES = ('baseline', *(f'below_open_{limit}bp' for limit in LIMITS), 'one_btc_eth_same_scan')
MAJORS = frozenset(('BTCUSDT', 'ETHUSDT'))


def reconstructed_opening_anchors(rows, frames, trades):
    grouped = defaultdict(list)
    for row in rows:
        if row.get('opening_anchor') is None:
            grouped[row['symbol']].append(row)
    anchors = {}
    for symbol, own in grouped.items():
        frame = frames.get(symbol)
        if frame is None or symbol not in trades:
            continue
        try:
            tape = load_repaired_tape(trades[symbol])
            times = [trade.ms for trade in tape]
            disagreements = reconcile(tape, frame, start=min(r['epoch'] for r in own),
                                      end=max(r['epoch'] for r in own)+900)
        except (OSError, ValueError, KeyError):
            continue
        for row in own:
            stamp = row['epoch']
            if stamp in disagreements:
                continue
            previous_bar = bisect_left(frame['time'], stamp)-1
            if previous_bar<0 or frame['time'][previous_bar] != stamp-900:
                continue
            scan_ms = min([row['entry_ms']] + [t['ms'] for t in row.get('tries', []) if 'ms' in t])
            first = bisect_left(times, stamp*1000)
            visible = first < len(times) and times[first] < scan_ms
            anchors[key(row)] = tape[first].price if visible else frame['close'][previous_bar]
    return anchors


def displacement(row, anchors):
    ref = row.get('opening_anchor')
    origin = 'observed_model_anchor'
    if ref is None:
        ref = anchors.get(key(row))
        if ref is None:
            return None, 'reconciliation_unavailable'
        origin = 'verified_public_open_at_replayed_scan'
    side = 1 if row['direction'] == 'LONG' else -1
    return 10_000 * side * (row['reference_entry'] / ref - 1), origin


def apply_policy(rows, name, features):
    first = set()
    rejected = set()
    adjusted = []
    for row in sorted(rows, key=lambda r: (r['entry_ms'], r['bot'], r['symbol'])):
        identity = key(row)
        if identity not in features or name == 'baseline':
            adjusted.append(row)
            continue
        reject = False
        if name == 'one_btc_eth_same_scan' and row['symbol'] in MAJORS:
            grouped = row['epoch'], row['direction']
            reject = grouped in first
            first.add(grouped)
        elif name.startswith('below_open_'):
            limit = int(name.removeprefix('below_open_').removesuffix('bp'))
            change, _ = features[identity]
            reject = change is not None and change <= -limit
        if reject:
            rejected.add(identity)
            row = dict(row, entry_status='opening_displacement_filter')
        adjusted.append(row)
    return adjusted, rejected


def evaluate(material, windows, controls, *, cap):
    rows, specs, ticks, books, _, trades, references = material
    frames = load_native(references)
    eligible = [r for r in rows if r['bot'] == 'crypto' and r['entry_status'] == 'eligible']
    anchors = reconstructed_opening_anchors(eligible, frames, trades)
    features = {key(row): displacement(row, anchors) for row in eligible}
    report = {}
    for name, keep in windows.items():
        sample = [r for r in rows if keep(r)]
        report[name] = {}
        for lag in CLOSE_LAGS:
            params = dict(clock=60, lag=lag, setting='requested5', max_positions=cap)
            base = simulate(sample, specs, ticks, books, **params)
            expected = controls[name][str(lag)]['baseline']
            fields = ('status', 'trades', 'return_pct', 'drawdown_pct', 'contribution_pct')
            if any(base.get(field) != expected.get(field) for field in fields):
                raise ValueError(f'frozen control changed for {name}/{lag}')
            outputs = {}
            for policy in POLICIES:
                marked, refused = apply_policy(sample, policy, features)
                result = base if policy == 'baseline' else simulate(marked, specs, ticks, books, **params)
                removed = [r for r in base['closed'] if key(r) in refused]
                outputs[policy] = dict(cash_summary(result, {}), candidates_rejected=len(refused),
                    direct_removals=[dict(symbol=r['symbol'], entry_ms=r['entry_ms'], code=r['code'],
                        cash_pnl_usd=r['bank_pnl']+r['runner_pnl'], price_move_bp=features[key(r)][0])
                        for r in removed], paired=paired_dollars(base, result))
            report[name][str(lag)] = outputs
            print(name, lag, {k: (v['candidates_rejected'], v.get('return_pct'), v.get('reason'))
                              for k, v in outputs.items()}, flush=True)
    return dict(cases=report, anchor_coverage=Counter(source for _, source in features.values()))


def main():
    control = ROOT / 'okx_macro_reaction_cash_nov23.json'
    expected = read_json(control)['exact_cash']
    results = {}
    for name, material, windows, cap in (
        ('earlier', old_inputs, EARLIER, 12),
        ('joint', recent_inputs, JOINT, 6),
        ('later', holdout_inputs, {'nov20_four_hours': lambda _: True}, 12),
    ):
        results[name] = evaluate(material(), windows, expected, cap=cap)
    write_json(ROOT / 'crypto_opening_displacement_cash_nov25.json', dict(
        results=results, policies=POLICIES, control_sha256=file_hash(control), script_sha256=file_hash(Path(__file__)),
        live_activation_supported=False, limitations=[
            'Entry quotes are public archived executable depth, not private order fills.',
            'The pre-trade gate uses the immediately executable price for the modeled account at the original time.',
            'Earlier opening anchors are reconstructed from reconciled public prints available at the first archived quote. Original in-process market snapshots are unavailable.',
            'The mixed stock route is experimental. The short windows cannot establish calendar-month returns or support live activation.',
        ]))


if __name__ == '__main__':
    main()
