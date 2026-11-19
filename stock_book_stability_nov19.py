"""Measure broader order-book opportunity and shortlist stability by phase."""
from collections import Counter, defaultdict
from datetime import datetime
import hashlib
import json
from pathlib import Path

from okx_xperp_l2_entry_replay_nov19 import attempt, cash_phase
from stock_quote_exit_attribution_nov19 import ARCHIVE, PROFILE, ROOT, book_variant
from us_cash_session_nov19 import ET


INITIAL_DATE = '2026-11-18'
LATER_DATE = '2026-11-19'
MIN_REGULAR_WINDOWS = 20
MIN_BOTH_SIDES = .50


def score_row(scan, row):
    payload = row['payload']
    book, spec = payload['book'], payload.get('spec')
    observed = row['received_at']
    checks = ['missing_public_book_or_spec'] * 2
    if row['quote_ok'] and book.get('ok') and spec and 0 <= observed-scan['bucket'] <= 180:
        raw = dict(bids=book['bids'], asks=book['asks'], ts=1000*book['exchange_at'])
        checks = [attempt(raw, observed*1000, side, spec)['status'] for side in ('LONG', 'SHORT')]
    candidate_status = None
    if row['candidate']:
        candidate_status = book_variant(dict(bucket=scan['bucket'], receipt=observed,
            geometry=payload['geometry'], book=book, spec=spec, valid_book=bool(row['quote_ok'])))['status']
    new_york_date = datetime.fromtimestamp(observed, ET).date().isoformat()
    return dict(bucket=scan['bucket'], date=new_york_date, phase=cash_phase(observed), symbol=row['symbol'],
                eligible_directions=checks.count('eligible'), directional_status=checks,
                candidate_status=candidate_status)


def aggregate(rows):
    result = Counter(observations=len(rows), directional_tests=2*len(rows),
                     any_side=0, both_sides=0, structural_candidates=0, passed_structural_candidates=0)
    for row in rows:
        result['any_side'] += row['eligible_directions'] > 0
        result['both_sides'] += row['eligible_directions'] == 2
        if row['candidate_status'] is not None:
            result['structural_candidates'] += 1
            result['passed_structural_candidates'] += row['candidate_status'] == 'qualified_5bp'
    return dict(result, directional_status=Counter(s for row in rows for s in row['directional_status']))


def shortlist(rows):
    sample = defaultdict(lambda: defaultdict(list))
    for row in rows:
        if row['phase'] == 'regular' and row['date'] in (INITIAL_DATE, LATER_DATE):
            sample[row['symbol']][row['date']].append(row)
    chosen = {}
    for symbol, periods in sorted(sample.items()):
        initial = aggregate(periods[INITIAL_DATE])
        if (initial['observations'] >= MIN_REGULAR_WINDOWS
                and initial['both_sides'] / initial['observations'] >= MIN_BOTH_SIDES):
            chosen[symbol] = {INITIAL_DATE: initial, LATER_DATE: aggregate(periods[LATER_DATE])}
    return chosen


def main():
    rows = []
    sources = {}
    for path in sorted(ARCHIVE.glob('bucket_*.json')):
        raw = path.read_bytes()
        scan = json.loads(raw)
        summary = scan.get('summary') or {}
        if (scan['profile'] != PROFILE or scan['completed_at'] is None or not summary.get('protocol_ok')
                or not summary.get('history_complete')):
            continue
        observations = scan['observations']
        if path.stem != f"bucket_{scan['bucket']}" or len({r['symbol'] for r in observations}) != summary['expected']:
            raise ValueError('archive time or universe mismatch')
        sources[path.name] = hashlib.sha256(raw).hexdigest()
        rows.extend(score_row(scan, row) for row in observations)
    phase = defaultdict(list)
    daily = defaultdict(lambda: defaultdict(list))
    for row in rows:
        phase[row['phase']].append(row)
        daily[row['date']][row['phase']].append(row)
    results = {key: aggregate(group) for key, group in phase.items()}
    selected = shortlist(rows)
    report = dict(status='OBSERVED_STOCK_QUOTE_STABILITY', sources=sources, windows=len(sources),
                  phase=results, daily={day: {key: aggregate(v) for key, v in phases.items()}
                                        for day, phases in daily.items()},
                  shortlist=selected, selection=dict(initial=INITIAL_DATE, later=LATER_DATE,
                  regular_min_observations=MIN_REGULAR_WINDOWS, minimum_both_sides=MIN_BOTH_SIDES),
                  objective_verified=False, live_activation_supported=False,
                  limitations=['One archived public book per instrument and 15m window. No retry/fill evidence.',
                      'NY cash phases are clock slices for all stock symbols, including international listings.',
                      'Shortlist is selected by observable order-book access only, without PnL or strategy refit.',
                      'Later day is a chronological cross-check on the same previously viewed short quote archive.'])
    path = ROOT/'stock_book_stability_nov19.json'
    path.write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    print('phase', json.dumps(results), flush=True)
    print('shortlist', {s: {d: (v['both_sides'], v['observations']) for d, v in x.items()}
                        for s, x in selected.items()}, flush=True)


if __name__ == '__main__':
    main()
