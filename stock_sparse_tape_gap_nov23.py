"""Audit recent oversized stock PAPER stops and screen causal liquidity filters."""

from bisect import bisect_left, bisect_right
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import math
from pathlib import Path

from okx_post_close_cash_nov23 import trade_difference
from okx_recent_trade_repair_nov21 import fetch_interval, validated_print
from okx_selective_single_target_nov23 import cash_summary
from okx_unbankable_tp1_nov21 import read_json, recent_inputs
from okx_xperp_cash_tick_nov19 import CLOSE_LAGS, simulate
from okx_xperp_multiday_replay_nov20 import file_hash
from public_candle_window import fetch_window
import stock_native_geometry_match_nov22 as native
from stock_liquid_quote_routes_nov21 import ROOT, write_json


STEM = 'stock_sparse_tape_gap_nov23'
DATA = ROOT / STEM
CASES = (
    (434, 'SOFTBANK-USD_UM_XPERP-310704', 'LONG', '06:47:01.069', -1.67),
    (439, 'SOXL-USD_UM_XPERP-310627', 'SHORT', '08:32:02.366', -1.00),
    (438, 'BMNR-USD_UM_XPERP-310822', 'SHORT', '08:36:02.192', -1.46),
    (435, 'SKHY-USD_UM_XPERP-310822', 'SHORT', '09:55:01.721', -1.80),
)
POLICIES = {
    'baseline': (0, 15),
    'min_3_active_last_15_minutes': (3, 15),
    'min_5_active_last_15_minutes': (5, 15),
    'min_3_active_and_trade_in_last_5_minutes': (3, 5),
}


def reconcile_minutes(frame, tape):
    groups = defaultdict(list)
    for trade in tape:
        groups[trade.ms // 60_000 * 60].append(trade)
    differences = {}
    for i, stamp in enumerate(frame['time']):
        trades = groups.pop(stamp, [])
        amount = math.fsum(t.contracts for t in trades)
        if not math.isclose(amount, frame['volume'][i], rel_tol=1e-8, abs_tol=1e-8):
            differences[stamp] = 'volume'
        elif trades:
            prices = [t.price for t in trades]
            values = (prices[0], max(prices), min(prices), prices[-1])
            if any(not math.isclose(v, frame[k][i], rel_tol=1e-10, abs_tol=1e-10)
                   for k, v in zip(('open', 'high', 'low', 'close'), values, strict=True)):
                differences[stamp] = 'ohlc'
    if groups:
        raise ValueError('trade tape outside the validated minutes')
    return differences


def incident(case):
    signal_id, instrument, side, clock, shown_r = case
    alert_ms = round(datetime.fromisoformat(f'2026-11-22T{clock}+00:00').timestamp() * 1000)
    floor = alert_ms // 60_000 * 60
    start, end = floor - 4 * 3600, floor + 60
    minutes, digest = fetch_window(instrument, start, end, 60, DATA / 'candles')
    tape_source = fetch_interval(instrument, start * 1000, end * 1000, DATA)
    stored = read_json(Path(tape_source['path']))
    if (stored['instrument'], stored['start_ms'], stored['end_ms']) != (instrument, start * 1000, end * 1000):
        raise ValueError('trade source changed')
    tape = sorted((validated_print(row, instrument) for row in stored['rows']), key=lambda r: (r.ms, r.identity))
    errors = reconcile_minutes(minutes, tape)
    if errors:
        raise ValueError(f'incomplete incident tape for {instrument}: {errors}')
    timestamps = [event.ms for event in tape]
    known = tape[:bisect_right(timestamps, alert_ms)]
    boundary = alert_ms - 4 * 60_000
    side_sign = 1 if side == 'LONG' else -1
    near_alert = []
    for before, current in zip(known, known[1:]):
        if current.ms < boundary:
            continue
        near_alert.append(dict(at_ms=current.ms, price=current.price, previous_price=before.price,
                               previous_trade_seconds=(current.ms - before.ms) / 1000,
                               adverse_jump_bps=side_sign * (1 - current.price / before.price) * 10_000,
                               contracts=current.contracts))
    return dict(signal_id=signal_id, instrument=instrument, direction=side, alert_ms=alert_ms,
                reported_signal_r=shown_r, checked_minutes=len(minutes['time']), trades=len(known),
                active_minutes_last_hour=sum(v > 0 for t, v in zip(minutes['time'], minutes['volume'], strict=True)
                                             if floor - 3600 <= t < floor),
                prices_near_stop_alert=near_alert, candle_sha256=digest, trade_sha256=tape_source['sha256'])


def market_activity(row):
    symbol, instrument, at = row['symbol'], row['instrument'], row['epoch']
    frame, digest = fetch_window(instrument, at - 15 * 60, at, 60, DATA / 'candles')
    active = [i for i, volume in enumerate(frame['volume']) if volume > 0]
    return (symbol, row['entry_ms']), dict(active_minutes=len(active),
                newest_traded_minute=15-active[-1] if active else None, candle_sha256=digest)


def admits_activity(point, policy):
    if policy == 'baseline':
        return True
    minimum, recency = POLICIES[policy]
    newest = point['newest_traded_minute']
    return point['active_minutes'] >= minimum and newest is not None and newest <= recency


def paper_results(rows):
    outcomes = Counter(row['paper_path']['code'] for row in rows)
    gap_stops = 0
    for row in rows:
        path = row['paper_path']
        if path['code'] != 'SL':
            continue
        entry = row['ticker_reference']
        loss = abs(path['trigger_price'] - entry) / abs(row['fixed_stop'] - entry)
        gap_stops += loss >= 1.25
    return dict(trades=len(rows), outcomes=outcomes, stops_at_least_1_25r_by_trigger=gap_stops)


def research():
    plan = native.freeze()
    cases, specs, ticks, books, tapes, diagnostics, checks = native.prepare_cases(plan)
    original = [row for row in diagnostics if (row.get('paper_path') or {}).get('known')]
    keys = {(row['symbol'], row['entry_ms']) for row in original}
    with ThreadPoolExecutor(max_workers=4) as pool:
        activities = dict(pool.map(market_activity, (row for row in plan['cases']['contract_observer']
                                                     if (row['symbol'], row['entry_ms']) in keys)))
    if activities.keys() != keys:
        raise ValueError('incomplete preceding activity coverage')
    evidence = []
    for row in original:
        key = row['symbol'], row['entry_ms']
        evidence.append(dict(symbol=row['symbol'], entry_ms=row['entry_ms'], direction=row['direction'],
                             phase=row['phase'], result=row['paper_path']['code'],
                             execution_gate=row['status'], **activities[key]))
    report = {}
    for name in POLICIES:
        def accepts(row):
            return admits_activity(activities[row['symbol'], row['entry_ms']], name)

        kept = [row for row in original if accepts(row)]
        cash_rows = [row for row in cases['production_1pct'] if accepts(row)]
        summarized = dict(aggregate=paper_results(kept), cash_entries=len(cash_rows),
                          cash_stress={str(bps): native.cash_view(cash_rows, specs, ticks, books, tapes, bps)
                                       for bps in (5, 15, 30)})
        summarized['by_phase'] = {phase: paper_results([r for r in kept if r['phase'] == phase])
                                  for phase in sorted({r['phase'] for r in original})}
        report[name] = summarized
        print(name, summarized['aggregate'], 'observed qualified markets', len(cash_rows), flush=True)
    return dict(cases=report, activity=evidence, tape_checks=checks)


def shared_stress():
    rows, specs, ticks, books, *_ = recent_inputs()
    eligible = [row for row in rows if row['bot'] == 'stocks' and row['entry_status'] == 'eligible'
                and row['stock_reference_ready']]
    with ThreadPoolExecutor(max_workers=4) as pool:
        activity = dict(pool.map(market_activity, eligible))
    expected_path = ROOT / 'okx_tp1_cash_sensitivity_nov22.json'
    controls = read_json(expected_path)['cases']['nov17_19_joint']['baseline']
    scenarios = {}
    for cap in (6, 12):
        for lag in CLOSE_LAGS:
            outcomes = {}
            args = dict(clock=60, lag=lag, setting='requested5', max_positions=cap)
            control = simulate(rows, specs, ticks, books, **args)
            if cap == 6:
                expected = controls[str(lag)]['requested5']
                for field in ('trades', 'return_pct', 'drawdown_pct', 'contribution_pct'):
                    if control[field] != expected[field]:
                        raise ValueError(f'published shared book control differs in {field}')
            for name in POLICIES:
                rejected = {key for key, point in activity.items() if not admits_activity(point, name)}
                selected = [dict(row, entry_status='traded_minute_filter')
                            if row['bot'] == 'stocks' and (row['symbol'], row['entry_ms']) in rejected else row
                            for row in rows]
                replay = control if name == 'baseline' else simulate(selected, specs, ticks, books, **args)
                result = cash_summary(replay, {})
                if 'closed' in control and 'closed' in replay and name != 'baseline':
                    result['positions_changed'] = trade_difference(control, replay)
                outcomes[name] = dict(result, rejected=len(rejected))
            scenarios[f'{cap}_positions_{lag}ms'] = outcomes
            print(f'shared {cap} positions {lag} ms', {name: (row.get('trades'), row.get('cash_pnl_usd'))
                  for name, row in outcomes.items()}, flush=True)
    return dict(source_sha256=file_hash(expected_path), results=scenarios,
                entries=[dict(symbol=symbol, entry_ms=at, **point) for (symbol, at), point in sorted(activity.items())])


def main():
    DATA.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=4) as pool:
        incidents = list(pool.map(incident, CASES))
    for row in incidents:
        largest = max(row['prices_near_stop_alert'], key=lambda event: event['adverse_jump_bps'], default=None)
        print(row['instrument'], row['active_minutes_last_hour'], 'largest adverse jump', largest, flush=True)
    write_json(ROOT / f'{STEM}.json', dict(incidents=incidents, policies=POLICIES, historical=research(),
        shared_cash=shared_stress(),
        live_activation_supported=False, limitations=[
            'The stop timestamps are the monitor log times. The R values are stock signal journal estimates.',
            'Public trades spanning the simulated close reconcile against all one-minute candles. They do not identify a private fill.',
            'Activity uses only the 15 completed minutes preceding the original scan.',
            'A historical stock observer sample uses public entry quotes and stressed exits when quoting cash.',
            'The November 17–19 shared sample uses exact public quotes and includes experimental stock continuation proposals.',
        ]))


if __name__ == '__main__':
    main()
