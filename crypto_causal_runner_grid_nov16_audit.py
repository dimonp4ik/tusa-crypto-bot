"""Frozen classifier, causal entry reference, 32 exit policies and joint cash scenarios."""
import hashlib
import itertools
import json
import math
import pickle
from pathlib import Path

import pandas as pd

import crypto_close_anchor_nov16_audit as anchor
from crypto_runner_exit_path_nov16 import replay
from lot_aware_tp1_audit import simulate
from shared_factor_account_nov14_audit import stock_ready_rows

ROOT = Path('reports/audit_2026_09_08')
STOCK = Path('../tusa stocks/reports/audit_2026_09_08')
RUNTIME = anchor.runtime
MONTHS = [(f'2026-{m:02}', anchor.timestamp(f'2026-{m:02}-01'),
           anchor.timestamp(f'2026-{m+1:02}-01')) for m in (7, 8, 9, 10, 11)]


def policies():
    for target, activation in itertools.product((1.5, 2., 3., 4.), (.5, 1., 1.25, 99.)):
        for trail in ((None, .5, 1.) if target > 2 else (None,)):
            yield dict(target_r=target, activation_r=activation, trail_distance_r=trail)


def portfolio(frame, *, operational_pause):
    rows = RUNTIME.base.standardized(frame, 'causal_runner_diagnostic')
    if operational_pause:
        # Current main.py pauses for SL_HIT events only, and can resume after
        # another outstanding signal closes without SL_HIT. This is a rules
        # proxy, not exact publication-time/order/Telegram replay.
        for row in rows:
            row['outcome'] = 'SL' if row['sequential_outcome'] == 'SL' else 'TP'
    return RUNTIME.runtime_gate(rows, latch_loss_pause=not operational_pause)


def cash_months(crypto, stock, specs):
    frame = pd.concat([pd.DataFrame(crypto).assign(bot='crypto'), stock], ignore_index=True)
    output = {}
    for label, start, end in MONTHS:
        cohort = frame[frame.entry_time.ge(start) & frame.exit_time.lt(end)]
        results = {}
        for priority in ('crypto', 'stocks'):
            result = simulate(cohort.to_dict('records'), specs, capital=130., slots=12,
                bot_slots=dict(crypto=4, stocks=8), simultaneous_writers=2, release_delay=900,
                tie_key=lambda row, p=priority: (row['bot'] != p, row['symbol']))
            assert result['peak_positions'] <= 12
            assert result['peak_by_bot'].get('crypto', 0) <= 4
            assert result['peak_by_bot'].get('stocks', 0) <= 8
            assert result['peak_reserved_margin_pct'] <= 90 + 1e-9
            result.pop('ledger')
            results[priority + '_first'] = result
        output[label] = results
    return output


def main():
    manifest = ROOT / 'xperp_15m_nov14_merged/manifest.json'
    meta = json.loads(manifest.read_text(encoding='utf-8'))
    candles = {}
    sources = {}
    for symbol, item in meta['symbols'].items():
        if not item.get('path') or symbol not in RUNTIME.base.ROBUST_SYMBOLS:
            continue
        path = Path(item['path'])
        raw = path.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        if digest != item['sha256']:
            raise ValueError('native candle hash mismatch')
        candles[symbol] = pickle.loads(raw)
        sources[str(path)] = digest
    far = ROOT / 'crypto_xperp_raw_router_audit_2_0_nov14_merged_raw_families.csv'
    near = ROOT / 'crypto_xperp_raw_router_audit_0_5_nov14_merged_raw_families.csv'
    raw = RUNTIME.joiner.join_direct_outcomes(far, near)
    raw = raw[raw.symbol.isin(candles)].assign(period='direct')
    candidates = anchor.scored_candidates(anchor.reanchor(raw, candles))
    paths = RUNTIME.exact.paths(ROOT / 'xperp_15m_nov14_merged')
    baseline_literal = RUNTIME.sequential.attach_current(candidates, paths)
    expected = pd.read_pickle(ROOT / 'crypto_close_anchor_nov16_rescored_known_close.pkl')
    lookups = {s: {int(t): i for i, t in enumerate(c['time'])} for s, c in candles.items()}
    spec_path = ROOT / 'shared_xperp_specs_nov14.json'
    specs = json.loads(spec_path.read_text(encoding='utf-8'))
    stock_path = STOCK / 'stock_close_anchor_nov16_causal_candidates_known_close.pkl'
    stock = stock_ready_rows(pd.read_pickle(stock_path)).assign(bot='stocks',
        sequential_banked=False, near_exit_time=math.nan)
    stock['sequential_runner_r'] = stock.net_r
    report = dict(status='EXPLORATORY_EXIT_AND_ACCOUNT_SENSITIVITY', objective_verified=False,
        live_change_approved=False, candidates=len(candidates), policy_count=32,
        limitations=[
            'Previously inspected native history; no independent unseen period.',
            'Known close is observable, not an executable market fill at publication.',
            '15m stop-first ambiguity; nominal stops update next bar; stress is not a guaranteed portfolio lower bound.',
            'Operational pause proxy excludes publication delays and account/quote rejections.',
            'Joint monthly scenarios start flat at $130; 3% margin, 10x, 5% TP1 with contract rounding.',
            'Current contract specs assumed across history; no historical bid/ask, funding or intrabar equity.',
            'Stock eligibility uses 20 completed PAPER outcomes strictly before entry; no actual LIVE authorization.',
            'Closed entry cohorts exclude inherited positions and exits beyond month end; July/November are partial.'],
        results={})
    literal_keys = ['symbol', 'entry_time', 'direction', 'sequential_outcome',
                    'sequential_runner_r', 'sequential_banked', 'sequential_exit_time']
    for rule in policies():
        for stress in (False, True):
            records = []
            for row in candidates.to_dict('records'):
                out = replay(candles[row['symbol']], lookups[row['symbol']][int(row['entry_time'])],
                    direction=row['direction'], entry=float(row['entry']), stop=float(row['sl']),
                    stress=stress, **rule)
                banked = out['banked']
                records.append(dict(row, sequential_outcome=out['outcome'],
                    sequential_runner_r=out['runner_r'], sequential_banked=banked,
                    sequential_exit_time=out['exit_time'], exit_time=out['exit_time'],
                    near_exit_time=out['near_exit_time'] if out['near_exit_time'] is not None else math.nan,
                    net_r=.025 + .95 * out['runner_r'] if banked else out['runner_r']))
            frame = pd.DataFrame(records)
            legacy = portfolio(frame, operational_pause=False)
            operational = portfolio(frame, operational_pause=True)
            is_baseline = rule == dict(target_r=2., activation_r=.5, trail_distance_r=None) and not stress
            if is_baseline:
                order = literal_keys[:3]
                pd.testing.assert_frame_equal(frame[literal_keys].sort_values(order).reset_index(drop=True),
                    baseline_literal[literal_keys].sort_values(order).reset_index(drop=True),
                    check_dtype=False, atol=1e-10, rtol=0)
                keys = order + ['net_r', 'exit_time']
                pd.testing.assert_frame_equal(pd.DataFrame(legacy)[keys].sort_values(order).reset_index(drop=True),
                    expected[keys].sort_values(order).reset_index(drop=True), check_dtype=False, atol=1e-10, rtol=0)
                report['baseline_parity_rows'] = len(legacy)
            key = json.dumps(dict(rule, stress=stress), sort_keys=True)
            report['results'][key] = dict(rule=rule, stress=stress,
                legacy_model={label: anchor.metrics(legacy, start, end) for label, start, end in MONTHS},
                operational_pause_proxy={label: anchor.metrics(operational, start, end) for label, start, end in MONTHS},
                joint_cash=cash_months(operational, stock, specs))
            print(key, {label: value['return_pct'] for label, value in
                  report['results'][key]['operational_pause_proxy'].items()}, flush=True)
    for path in (manifest, far, near, spec_path, stock_path, Path(__file__),
                 Path('crypto_runner_exit_path_nov16.py'), Path(anchor.__file__),
                 Path('crypto_runtime_shadow_portfolio_oct3_audit.py'), Path('lot_aware_tp1_audit.py'),
                 Path('shared_factor_account_nov14_audit.py'), RUNTIME.router.ARTIFACT_PATH):
        sources[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    report['sources'] = sources
    (ROOT / 'crypto_causal_runner_grid_nov16.json').write_text(json.dumps(report, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
