"""Compare scheduled release windows and the first observed 15-minute reaction."""

from bisect import bisect_right
from collections import Counter
from datetime import datetime
from pathlib import Path
import pickle

from crypto_minimum_bank_runner_nov19_audit import admitted_records, cash_result
from crypto_native_price_pool_nov19 import MANIFEST
from okx_cross_bot_overlap_nov21 import DATA, ENTRY_END, START, crypto_cadence, joint_inputs
from okx_joint_exact_exit_nov21 import ENTRY_END as VERIFIED_END
from okx_macro_reaction_cash_nov23 import ET, REACTION, events, volatility
from okx_post_close_cash_nov23 import trade_difference
from okx_selective_single_target_nov23 import EARLIER, cash_summary, key
from okx_spot_confirmation_nov22 import load_native
from okx_sweep_context_cash_nov23 import HISTORY, window
from okx_swing_target_cash_nov23 import historical_brackets
from okx_unbankable_tp1_nov21 import old_inputs, read_json
from okx_xperp_cash_tick_nov19 import CLOSE_LAGS, simulate
from okx_xperp_multiday_replay_nov20 import ROOT, exit_books, file_hash
from public_candle_window import fetch_window
from stock_liquid_quote_routes_nov21 import write_json
from stock_native_shared_cash_nov24 import FIELDS, stock_rows
from stock_contract_spread_holdout_nov23 import DATA as NATIVE_STOCK_DATA


POLICIES = ('baseline', 'clock_15m_before_45m_after', 'clock_15m_before_90m_after',
            'wait_for_first_close', 'veto_adverse_first_candle', 'wait_then_veto_adverse')
FRED_CLAIMS = 'https://fred.stlouisfed.org/releases/calendar?rid=180&ve=2026-12-31&view=year&vs=2026-10-01'
FRED_NOV19 = 'https://fred.stlouisfed.org/releases/calendar?ve=2026-11-19&view=week&vs=2026-11-19'
PHILADELPHIA_FED = 'https://www.philadelphiafed.org/surveys-and-data/regional-economic-analysis/mbos-2026-11'
FED_MINUTES = 'https://www.federalreserve.gov/newsevents/pressreleases/monetary20261118a.htm'
BAR = 900
MAX_REACTION_AGE = 90 * 60


def release_calendar():
    existing, digest = events()
    reports = [dict(item, source=FED_MINUTES) if item['published_at'] == 1795028400 else item
               for item in existing]
    for day in (5, 12, 19):
        released = int(datetime(2026, 11, day, 8, 30, tzinfo=ET).timestamp())
        release = dict(kind='INITIAL_JOBLESS_CLAIMS', published_at=released, source=FRED_CLAIMS)
        if day == 19:
            release.update(kind='JOBLESS_CLAIMS_AND_PHILADELPHIA_FED_MFG', source=FRED_NOV19,
                           secondary_source=PHILADELPHIA_FED)
        reports.append(release)
    reports.sort(key=lambda row: row['published_at'])
    times = [row['published_at'] for row in reports]
    if len(times) != len(set(times)) or any(t % BAR for t in times):
        raise ValueError('ambiguous release timestamp')
    return reports, times, digest


def context(at, direction, times, reaction):
    i = bisect_right(times, at)
    last = times[i - 1] if i else None
    age = at - last if last is not None else None
    before = times[i] - at if i < len(times) else None
    signed = None
    if age is not None and BAR <= age <= MAX_REACTION_AGE:
        size = reaction(last)
        if size is not None:
            signed = size if direction == 'LONG' else -size
    return dict(before=before, age=age, release=last, signed_first_candle=signed)


def filtered(policy, observation):
    if policy == 'baseline':
        return False
    before, age = observation['before'], observation['age']
    if policy in ('clock_15m_before_45m_after', 'clock_15m_before_90m_after'):
        until = 2700 if policy == 'clock_15m_before_45m_after' else 5400
        return before is not None and before <= BAR or age is not None and age <= until
    waiting = age is not None and age < BAR
    signed = observation['signed_first_candle']
    adverse = signed is not None and signed <= -REACTION
    return {'wait_for_first_close': waiting, 'veto_adverse_first_candle': adverse,
            'wait_then_veto_adverse': waiting or adverse}[policy]


def make_measure(frames, specs=None):
    seen, downloads = {}, {}

    def reaction(symbol, at):
        if (symbol, at) in seen:
            return seen[symbol, at]
        close = at + BAR
        frame = frames.get(symbol)
        prior = window(frame, close) if frame is not None else None
        if prior is None and specs is not None:
            instrument = specs[symbol]['instId']
            location = ROOT / 'stock_crypto_release_reaction_nov24' / 'candles'
            try:
                data, digest = fetch_window(instrument, close - HISTORY * BAR, close, BAR, location)
                prior = window(data, close)
            except (OSError, ValueError) as exc:
                digest = type(exc).__name__
            downloads[f'{instrument}:{at}'] = digest
        seen[symbol, at] = volatility(prior, 'LONG')
        return seen[symbol, at]

    return reaction, downloads


def compare(rows, specs, ticks, books, references, times, expected):
    reaction, downloads = make_measure(load_native(references), specs)
    contexts = {key(r): context(r['entry_ms'] / 1000, r['direction'], times,
                              lambda at: reaction(r['symbol'], at))
                for r in rows if r['entry_status'] == 'eligible'}
    results = {}
    for lag in CLOSE_LAGS:
        params = dict(clock=60, lag=lag, setting='requested5')
        base = simulate(rows, specs, ticks, books, **params)
        if any(base.get(field) != expected[str(lag)].get(field) for field in FIELDS):
            raise ValueError('calendar experiment did not reproduce the cash control')
        cases = {}
        for name in POLICIES:
            rejected = {identity for identity, obs in contexts.items() if filtered(name, obs)}
            admitted = [dict(row, entry_status='scheduled_release_filter') if key(row) in rejected else row
                        for row in rows]
            trial = base if name == 'baseline' else simulate(admitted, specs, ticks, books, **params)
            removed = [r for r in base['closed'] if key(r) in rejected]
            cases[name] = dict(cash_summary(trial, {}), candidates_rejected=Counter(k[0] for k in rejected),
                               matched_original_exits=[dict(symbol=r['symbol'], bot=r['bot'], code=r['code'],
                                   pnl=r['bank_pnl'] + r['runner_pnl'], entry_ms=r['entry_ms'],
                                   **contexts[key(r)]) for r in removed], differences=trade_difference(base, trial))
        results[str(lag)] = cases
    return dict(results=results, candle_downloads=downloads,
        reaction_missing=sum(v['age'] is not None and BAR <= v['age'] <= MAX_REACTION_AGE
                             and v['signed_first_candle'] is None for v in contexts.values()))


def exact_shared(times):
    (start, end), routes, stock_specs, stock_ticks, stock_books, stock_evidence = stock_rows()
    crypto, _, specs, ticks, books, hashes, *_ = joint_inputs()
    crypto = list(crypto_cadence(crypto))
    cached = pickle.loads((DATA / 'joint_exact_exit_samples.pkl').read_bytes())
    frozen_stock = ROOT / 'stock_liquid_quote_routes_nov21.json'
    if cached['sources'] != dict(hashes, **{str(frozen_stock): file_hash(frozen_stock)}):
        raise ValueError('cash exit sources changed')
    snapshots, extra_archives, _ = cached['result']
    for symbol in {row['symbol'] for row in crypto}:
        books[symbol].update(snapshots.get(symbol, {}))
    books.update(stock_books)
    specs.update(stock_specs)
    ticks.update(stock_ticks)
    end = min(end, ENTRY_END)
    left = max(start, START)
    late = [r for r in crypto if VERIFIED_END <= r['epoch'] < end and r['entry_status'] == 'eligible']
    archive = read_json(DATA / 'crypto_books.json')['books'] | extra_archives
    _, exits = exit_books(late, {'books': archive}, specs, books)
    source = ROOT / 'stock_native_shared_cash_nov24.json'
    baseline = read_json(source)['results']['extended_common_window']
    controls = {str(lag): baseline[f'12_slots_{lag}ms']['current'] for lag in CLOSE_LAGS}
    refs = read_json(DATA / 'crypto_capture.json')['references']
    refs.update(read_json(NATIVE_STOCK_DATA / 'capture.json')['reference'])
    rows = [r for r in crypto + routes['current'] if left <= r['epoch'] < end]
    result = compare(rows, specs, ticks, books, refs, times, controls)
    return dict(result, start=left, end=end, baseline_sha256=file_hash(source),
                stock_evidence=stock_evidence, crypto_exit_books=exits)


def history(times, expected):
    material, specs, _, _, metadata, *_ = historical_brackets()
    candidates = material['baseline']
    reaction, _ = make_measure(load_native(read_json(MANIFEST)['symbols']))
    contexts = [context(r['entry_time'], r['direction'], times, lambda at: reaction(r['symbol'], at))
                for r in candidates]
    result = {}
    for label in POLICIES:
        accepted = [r for r, obs in zip(candidates, contexts, strict=True) if not filtered(label, obs)]
        rows, unknown = admitted_records(accepted)
        if unknown:
            result[label] = dict(status='UNSCORABLE', unknown=unknown)
            continue
        cash = cash_result(rows, specs, 'requested5')
        if label == 'baseline' and cash != expected:
            raise ValueError('candle proxy did not reproduce the published control')
        result[label] = dict(cash=cash, candidates_rejected=len(candidates) - len(accepted))
    return dict(results=result, candidate_pool_sha256=metadata['sha256'], candles_sha256=file_hash(MANIFEST))


def main():
    releases, times, digest = release_calendar()
    control_path = ROOT / 'okx_macro_reaction_cash_nov23.json'
    control = read_json(control_path)
    rows, specs, ticks, books, _, _, refs = old_inputs()
    earlier = {}
    for name, keep in EARLIER.items():
        expected = {str(lag): control['exact_cash'][name][str(lag)]['baseline'] for lag in CLOSE_LAGS}
        earlier[name] = compare([r for r in rows if keep(r)], specs, ticks, books, refs, times, expected)
    historical = history(times, control['crypto_older_candle_proxy']['cases']['baseline']['cash'])
    for name, sample in earlier.items():
        print(name, {k: round(v['return_pct'], 4) for k, v in sample['results']['1000'].items()}, flush=True)
    joint = exact_shared(times)
    print('native_joint', {k: (v.get('trades'), v.get('return_pct'), v.get('reason'))
                          for k,v in joint['results']['1000'].items()}, flush=True)
    write_json(ROOT / 'stock_crypto_release_reaction_nov24.json', dict(
        official_releases=releases, inherited_calendar_sha256=digest, policies=POLICIES,
        frozen_threshold_in_median_ranges=REACTION, earlier_crypto=earlier, native_joint=joint,
        historical_crypto_candles=historical, controls_sha256=file_hash(control_path),
        script_sha256=file_hash(Path(__file__)), live_activation_supported=False, limitations=[
            'Historical candles are a separate crypto proxy. Exact November cases use public OKX books and trades.',
            'The first release candle cannot influence entries before its close. Missing later candle evidence does not veto an entry.',
            'Release-time overlaps do not establish which news item moved an instrument. Initial claims are supplemented only in the reviewed November windows.',
            'Mixed execution starts with common cash and existing slot and margin limits. No unscheduled news, scheduled earnings or complete-month live validation.',
        ]))


if __name__ == '__main__':
    main()
