"""Replay mature observer candidates after each actual public book receipt."""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from statistics import median
import time

from public_candle_window import fetch_window
from src.execution_guard import MAX_ENTRY_SPREAD, execution_quote
from src.okx_trader import calc_contracts


ROOT = Path('reports/audit_2026_09_08')
ARCHIVE = Path('../tusa stocks/reports/structure_observer_prospective_nov18')
OUTPUT = ROOT / 'stock_quote_exit_attribution_nov19'
PROFILE = 'stock_swing2_original_target_quote_observer_v1'
DECLARED_AT = 1795123704
HOLD_SECONDS = 57600
CAPITAL = 130.
MARGIN_FRACTION = .03
LEVERAGE = 10


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def collect_plan():
    cutoff = (DECLARED_AT - HOLD_SECONDS) // 900 * 900
    items = []
    scans = {}
    for path in sorted(ARCHIVE.glob('bucket_*.json')):
        raw = path.read_bytes()
        scan = json.loads(raw)
        bucket = scan['bucket']
        if bucket > cutoff or scan['profile'] != PROFILE or not scan['completed_at']:
            continue
        if not (scan['summary']['protocol_ok'] and scan['summary']['history_complete']):
            continue
        if path.stem != f'bucket_{bucket}':
            raise ValueError('archive timestamp mismatch')
        scans[path.name] = hashlib.sha256(raw).hexdigest()
        for row in scan['observations']:
            if not row['candidate']:
                continue
            payload = row['payload']
            geometry = payload['geometry']
            if geometry['hold_seconds'] != HOLD_SECONDS or payload['received_at'] != row['received_at']:
                raise ValueError('archive candidate contract mismatch')
            receipt = float(row['received_at'])
            if not math.isfinite(receipt):
                raise ValueError('invalid quote timestamp')
            start = int(receipt // 60) * 60
            items.append(dict(bucket=bucket, symbol=row['symbol'], instrument=row['instrument'],
                              start=start, end=bucket + HOLD_SECONDS, receipt=receipt,
                              valid_book=bool(row['quote_ok'] and payload['book']['ok']),
                              geometry=geometry, book=payload['book'], spec=payload.get('spec')))
    if not scans or not items or len({(r['bucket'], r['symbol']) for r in items}) != len(items):
        raise ValueError('missing or duplicate observer candidates')
    source = [Path(__file__), Path('public_candle_window.py'),
              Path('src/execution_guard.py'), Path('src/okx_trader.py')]
    return dict(profile=PROFILE, outcome_maturity_cutoff=cutoff, archive_scans=scans,
                candidates=items, sources={str(path): sha(path) for path in source})


def book_variant(row):
    if not row['valid_book'] or not row['spec']:
        return dict(status='missing_public_book_or_spec')
    if not 0 <= row['receipt'] - row['bucket'] <= 180:
        return dict(status='late_public_snapshot')
    book = row['book']
    g = row['geometry']
    direction = g['side']
    raw = dict(bids=book['bids'], asks=book['asks'], ts=book['exchange_at'] * 1000)
    spread = float(book['spread_bps'])
    # Deliberately diagnose wide books too. The executable subset below still
    # uses the actual production constraint; orders are never submitted.
    try:
        quote = execution_quote(raw, direction, None, row['receipt'] * 1000, max_spread=1.,
            size_for_price=lambda price: calc_contracts(CAPITAL*MARGIN_FRACTION, LEVERAGE,
                                                         price, row['spec']))
        quote.validate_bracket(g['stop'], g['target'])
    except ValueError as exc:
        return dict(status=str(exc), spread_bps=spread)
    return dict(status='qualified_5bp' if quote.spread_fraction <= MAX_ENTRY_SPREAD else 'wide',
                average=quote.average, worst=quote.worst, contracts=quote.contracts,
                spread_bps=quote.spread_fraction * 10_000)


def minute_exit(frame, row):
    """Use only known ordering after receipt; ambiguous boundary stays unknown."""
    g = row['geometry']
    sign = 1 if g['side'] == 'LONG' else -1
    stop = sign * g['stop']
    target = sign * g['target']
    for i, opening_minute in enumerate(frame['time']):
        if not frame['volume'][i] > 0:
            continue
        o, h, l, close = (frame[key][i] for key in ('open', 'high', 'low', 'close'))
        adverse, favorable = (l, h) if sign > 0 else (-h, -l)
        if opening_minute < row['receipt']:
            if adverse <= stop or favorable >= target:
                return dict(known=False, reason='receipt_minute_touched_a_level')
            continue
        if sign * o <= stop:
            code, price = 'SL_GAP', o
        elif sign * o >= target:
            code, price = 'TARGET', g['target']
        elif adverse <= stop and favorable >= target:
            return dict(known=False, reason='stop_and_target_in_same_minute')
        elif adverse <= stop:
            code, price = 'SL', g['stop']
        elif favorable >= target:
            code, price = 'TARGET', g['target']
        elif i == len(frame['time']) - 1:
            code, price = 'TIME', close
        else:
            continue
        return dict(known=True, outcome=code, fill_reference=price, exit_time=opening_minute + 60)
    return dict(known=False, reason='missing_terminal_trade')


def replay(row):
    book = book_variant(row)
    result = dict(bucket=row['bucket'], symbol=row['symbol'], direction=row['geometry']['side'],
                  receipt=row['receipt'], book=book)
    if not row['bucket'] <= row['receipt'] <= row['bucket'] + 180:
        return dict(result, path=dict(known=False, reason='outside_observation_deadline'))
    for attempt in range(3):
        try:
            candles, digest = fetch_window(row['instrument'], row['start'], row['end'], 60, OUTPUT/'minutes')
            break
        except Exception as exc:
            if attempt == 2:
                return dict(result, path=dict(known=False, reason='minute_history_unavailable',
                                             error_type=type(exc).__name__))
            time.sleep(attempt + 1)
    path = minute_exit(candles, row)
    result.update(path=path, minute_sha256=digest)
    if path['known']:
        sign = 1 if result['direction'] == 'LONG' else -1
        signal = row['geometry']['signal_entry']
        result['signal_return'] = sign*(path['fill_reference']/signal - 1)
        if 'average' in book:
            result['book_return'] = sign*(path['fill_reference']/book['average'] - 1)
            result['entry_cost_bps'] = 10_000*sign*(book['average']/signal - 1)
    return result


def group(rows):
    known = [r for r in rows if r['path']['known']]
    matched = [r for r in known if 'book_return' in r]
    return dict(rows=len(rows), known=len(known), unknown=Counter(r['path']['reason'] for r in rows
        if not r['path']['known']), outcomes=Counter(r['path']['outcome'] for r in known),
        signal_wins=sum(r['signal_return']>0 for r in known), comparable=len(matched),
        matched_signal_wins=sum(r['signal_return']>0 for r in matched),
        matched_book_wins=sum(r['book_return']>0 for r in matched),
        signal_return_pct_sum=100*math.fsum(r['signal_return'] for r in known),
        matched_signal_return_pct_sum=100*math.fsum(r['signal_return'] for r in matched),
        matched_book_return_pct_sum=100*math.fsum(r['book_return'] for r in matched),
        median_entry_cost_bps=median(r['entry_cost_bps'] for r in matched) if matched else None)


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    plan_path = OUTPUT / 'plan.json'
    proposed = collect_plan()
    if plan_path.exists():
        if json.loads(plan_path.read_bytes()) != proposed:
            raise ValueError('frozen cohort or source changed')
    else:
        plan_path.write_text(json.dumps(proposed, indent=2, allow_nan=False), encoding='utf-8')
    print(len(proposed['archive_scans']), 'mature windows', len(proposed['candidates']), 'proposals', flush=True)
    results = []
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(replay, item) for item in proposed['candidates']]
        for job in as_completed(futures):
            results.append(job.result())
            if len(results) % 20 == 0:
                print('matched windows', len(results), '/', len(futures), flush=True)
    results.sort(key=lambda row: (row['bucket'], row['symbol']))
    groups = {'all': group(results), 'qualified_5bp': group([r for r in results if r['book']['status']=='qualified_5bp']),
              'wide': group([r for r in results if r['book']['status']=='wide'])}
    report = dict(status='PROSPECTIVE_PUBLIC_QUOTE_AND_TRADE_PATH', plan_sha256=sha(plan_path),
                  created_at=datetime.now(timezone.utc).isoformat(), book_results=Counter(r['book']['status'] for r in results),
                  groups=groups, results=results, objective_verified=False, live_activation_supported=False,
                  limitations=[
                      'Archived structural observer candidates, not the live production signal journal.',
                      'Same post-receipt stop and target levels, no early-wick guessing. Missing exits are unresolved.',
                      'The first public snapshot at a hypothetical order size does not represent private fills or retries.',
                      'Uncompounded signed price moves, no account PnL, fee, readiness, or overlapping-signal admission claim.',
                      'V1 observer uses legacy base-volume liquidity detection, and capture spans a few days.',
                  ])
    target = ROOT / 'stock_quote_exit_attribution_nov19.json'
    target.write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps(dict(book_results=report['book_results'], groups=groups)), flush=True)


if __name__ == '__main__':
    main()
