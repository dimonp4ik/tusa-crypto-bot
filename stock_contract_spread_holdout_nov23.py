"""Reconstruct the pre-November-20 native stock cohort using contract volumes."""

from bisect import bisect_left
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import math
from unittest.mock import patch

from okx_selective_single_target_nov23 import key
from okx_unbankable_tp1_nov21 import read_json
from okx_xperp_l2_entry_replay_nov19 import cash_phase
from okx_xperp_multiday_replay_nov20 import ROOT, exit_books, file_hash, key_for
from public_candle_window import fetch_window
from stock_liquid_quote_routes_nov21 import ARCHIVES, NATIVE_PATH, capture, native_module, repair_trades, write_json
import stock_native_geometry_match_nov22 as native_replay
import stock_native_spread_replay_nov23 as spread
from stock_spread_recovery_nov23 import DELAY_POLICIES, collect, recheck


STEM = 'stock_contract_spread_holdout_nov23'
DATA = ROOT / STEM
BAR = 900
FIELDS = ('open', 'high', 'low', 'close')


def confirmed_frames(universe, begin, stop):
    def load(item):
        symbol, instrument = item
        frame = defaultdict(list)
        source = []
        for left in range(begin, stop, 86400):
            data, sha = fetch_window(instrument, left, min(left + 86400, stop), BAR, DATA / 'decision_candles')
            source.append(sha)
            for field, values in data.items():
                frame[field].extend(values)
        frame['volume_contracts'] = frame['volume']
        return symbol, (dict(frame), source)

    with ThreadPoolExecutor(max_workers=4) as pool:
        return dict(pool.map(load, sorted(universe.items())))


def same_prices(source, observed):
    return source['time'] == observed['time'] and all(
        len(source[field]) == len(observed[field]) and all(math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-10)
        for a, b in zip(source[field], observed[field], strict=True)) for field in FIELDS)


def plan():
    output = DATA / 'plan.json'
    if output.exists():
        saved = read_json(output)
        if saved['native_sha256'] != file_hash(NATIVE_PATH):
            raise ValueError('native stock model changed after freeze')
        return saved
    native = native_module()
    folder, profile = ARCHIVES['legacy_discovery']
    scans = []
    inventory = {}
    legacy_candidates = set()
    for path in sorted(folder.glob('bucket_*.json')):
        scan = read_json(path)
        bucket = int(path.stem.removeprefix('bucket_'))
        if (scan['profile'], scan['bucket']) != (profile, bucket):
            raise ValueError('archive identity mismatch')
        summary = scan['summary']
        if not scan['completed_at'] or not summary['history_complete'] or not summary['protocol_ok']:
            continue
        items = {row['symbol']: row for row in scan['observations']}
        if set(items) != set(native.AUDITED_SYMBOLS) or len(items) != len(scan['observations']):
            raise ValueError('observed universe differs from frozen stock universe')
        for symbol, row in items.items():
            if inventory.setdefault(symbol, row['instrument']) != row['instrument']:
                raise ValueError('instrument rollover inside the cohort')
            candidate = row['payload']['native_candidate']
            if candidate:
                legacy_candidates.add((bucket, symbol, candidate['direction']))
        scans.append((bucket, path))
    if not scans:
        raise ValueError('no complete observed stock scans')
    begin = scans[0][0] - native.REQUIRED_HISTORY_BARS * BAR
    end = scans[-1][0]
    sources = confirmed_frames(inventory, begin, end)
    rows, unavailable, digests = [], {}, {}
    for bucket, path in scans:
        snapshot = read_json(path)
        observed = {row['symbol']: row for row in snapshot['observations']}
        frames = {}
        for symbol, (frame, _) in sources.items():
            i = bisect_left(frame['time'], bucket)
            frames[symbol] = {field: values[i-native.REQUIRED_HISTORY_BARS:i] for field, values in frame.items()}
        changed = [symbol for symbol, frame in frames.items() if not same_prices(frame, observed[symbol]['closed_frame'])]
        if changed:
            unavailable[str(bucket)] = changed
            continue
        decision_at = float(snapshot['summary']['decision_at'])
        proposals = native.native_shadow_setups(frames, now=decision_at)
        digests[path.name] = file_hash(path)
        for proposal in proposals:
            symbol = proposal['symbol']
            item = observed[symbol]
            payload = item['payload']
            receipt = float(item['received_at'])
            if not decision_at <= receipt < bucket + native.MAX_SIGNAL_AGE_SECONDS:
                continue
            rows.append(dict(symbol=symbol, instrument=item['instrument'], epoch=bucket, bot='stocks',
                route='native_contract_volume', direction=proposal['direction'], native_candidate=proposal,
                structural=None, entry_ms=round(receipt * 1000), receipt=receipt, phase=cash_phase(receipt),
                spec=payload['spec'], public_book=payload['book'], valid_book=bool(item['quote_ok']),
                preceding_close=float(frames[symbol]['close'][-1])))
    rederived = {(r['epoch'], r['symbol'], r['direction']) for r in rows}
    saved = dict(cases=dict(contract_observer=rows), counts=Counter(r['phase'] for r in rows), indexes={},
                 native_sha256=file_hash(NATIVE_PATH), entries_utc=[scans[0][0], scans[-1][0] + BAR],
                 observed_scans=len(scans), rechecked_scans=len(digests), observed_sha256=digests,
                 contract_candle_sha256={symbol: source for symbol, (_, source) in sources.items()},
                 changed_price_scans=unavailable, legacy_candidates=len(legacy_candidates),
                 new_vs_legacy=sorted(rederived - legacy_candidates), removed_vs_legacy=sorted(legacy_candidates - rederived))
    write_json(output, saved)
    print('Older native frozen candidates', len(rows), 'reconciled scans', len(digests), '/', len(scans), flush=True)
    return saved


def evaluate(frozen):
    if not frozen['cases']['contract_observer']:
        return dict(status='NO_RECONSTRUCTED_CANDIDATES')
    if not (DATA / 'capture.json').exists():
        capture(data=DATA)
    if not (DATA / 'trade_repair.json').exists():
        repair_trades(data=DATA)
    with patch.object(native_replay, 'DATA', DATA):
        rows, specs, ticks, tapes, tape_checks, _, capped = spread.prepare(frozen)
    if not rows:
        return dict(status='NO_SIGNALS_AFTER_PUBLICATION_AND_GEOMETRY', publication_cap_rejections=capped,
                    tape_checks=tape_checks)
    books, entry_checks, archives = collect(rows, specs)
    available = [row for row in rows if 'path' in archives[key_for(row['symbol'], row['entry_ms'])]]
    missing = [key(row) for row in rows if row not in available]
    model = native_module()
    base = [spread.pick(row, specs[row['symbol']], ticks[row['symbol']], books[row['symbol']], tapes,
                        spread.POLICIES['current'], model.backing_level_allows_entry) for row in available]
    groups = dict(current=base)
    for name, (count, max_spread, reward) in DELAY_POLICIES.items():
        groups[name] = [recheck(row, first, specs[row['symbol']], ticks[row['symbol']], books[row['symbol']],
                               tapes, model.backing_level_allows_entry, count, max_spread=max_spread,
                               minimum_reward_r=reward) for row, first in zip(available, base, strict=True)]
    fills = {key(row): row for group in groups.values() for row in group if row['entry_status'] == 'eligible'}
    downloaded, exits = exit_books(list(fills.values()), {'books': archives}, specs, books)
    outcomes = spread.cash_results(groups, specs, ticks, books, tapes)
    for name, result in outcomes.items():
        print(name, result['statuses'], [(label, value.get('trades'), value.get('cash_pnl_usd'), value.get('reason'))
              for label, value in result['modes'].items() if label.endswith('_exact')], flush=True)
    return dict(status='EVALUATED', admitted_signals=len(rows), missing_entry_archive=missing, outcomes=outcomes,
                entry_checks=entry_checks, tape_checks=tape_checks, exact_exit_checks=exits, publication_cap_rejections=capped,
                archive_sha256={k: x['sha256'] for k, x in (archives | downloaded).items() if 'sha256' in x},
                fills={name: [dict(symbol=row['symbol'], epoch=row['epoch'], entry_ms=row['entry_ms'],
                                   direction=row['direction'], reference_entry=row['reference_entry'], path=row['path'])
                              for row in items if row['entry_status'] == 'eligible'] for name, items in groups.items()})


def main():
    DATA.mkdir(parents=True, exist_ok=True)
    frozen = plan()
    evidence = {k: frozen[k] for k in ('entries_utc', 'observed_scans', 'rechecked_scans', 'changed_price_scans',
                                     'legacy_candidates', 'new_vs_legacy', 'removed_vs_legacy', 'counts')}
    write_json(ROOT / f'{STEM}.json', dict(evidence=evidence, plan_sha256=file_hash(DATA / 'plan.json'),
        parameters=DELAY_POLICIES, result=evaluate(frozen), live_activation_supported=False, limitations=[
            'The exact stock strategy is replayed on confirmed contract candles using the original observation times.',
            'Older source volume is replaced only after open, high, low and close agree with the original frozen observations.',
            'Candidates from revised or missing windows do not enter the portfolio.',
            'The portfolio is a public-book stock-only replay. The entry dates are disjoint from the November 20–21 stock sample.',
        ]))


if __name__ == '__main__':
    main()
