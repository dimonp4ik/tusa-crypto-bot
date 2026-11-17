"""Paired public-price paths through recorded model closure, no private PnL claims."""
import hashlib
import json
import math
from pathlib import Path
from statistics import median
import time

from observed_be_path_nov16 import PricePath
from public_candle_window import fetch_window
from public_trade_second import fetch_second_trades, validate_minute_seconds

ROOT = Path('reports/audit_2026_09_08/live_filter_review_nov15')
MODES = ('next_strategy_bar', 'next_minute', 'observed_tp1', 'observed_tp1_plus_60')


def bar(frame, index, seconds):
    return {key: values[index] for key, values in frame.items()} | dict(seconds=seconds)


def reconstruct(row, instrument, folder):
    opened, closed = float(row['opened_at']), float(row['closed_at'])
    if (row['strategy_runner_mode'] != 'fixed_be' or row['strategy_bar_seconds'] != 900
            or row['strategy_stop_on_close'] or not opened < closed):
        raise ValueError('unsupported published execution contract')
    start, end = int(opened // 60) * 60, int(math.ceil(closed / 60)) * 60
    time.sleep(.12)
    minutes, digest = fetch_window(instrument, start, end, 60, folder)
    hashes = {f'{instrument}:1m:{start}:{end}': digest}
    paths = {mode: PricePath(row, mode) for mode in MODES}
    refined_minutes = refined_seconds = zero_minutes = 0

    def consume(stamp, price, *, known_by=None):
        if opened <= stamp < closed:
            for path in paths.values():
                path.consume(stamp, price, known_by=known_by)

    for i, stamp in enumerate(minutes['time']):
        if all(path.event is not None for path in paths.values()):
            break
        parent = bar(minutes, i, 60)
        if not parent['volume']:
            zero_minutes += 1
            continue
        boundary = stamp < opened or stamp + 60 > closed
        if not boundary and not any(path.needs_detail(parent) for path in paths.values()):
            for path in paths.values():
                path.consume_untouched_bar(parent)
            continue
        time.sleep(.12)
        seconds, shash = fetch_window(instrument, stamp, stamp + 60, 1, folder)
        hashes[f'{instrument}:1s:{stamp}'] = shash
        validate_minute_seconds(parent, seconds, stamp)
        refined_minutes += 1
        for j, second in enumerate(seconds['time']):
            if second + 1 <= opened or second >= closed or all(p.event is not None for p in paths.values()):
                continue
            unit = bar(seconds, j, 1)
            if not unit['volume']:
                continue
            part = second < opened or second + 1 > closed
            if part or any(path.needs_detail(unit) for path in paths.values()):
                trades, thash = fetch_second_trades(instrument, second, unit, folder)
                hashes[f'{instrument}:trades:{second}'] = thash
                refined_seconds += 1
                for millis, _, price, _ in trades:
                    consume(millis / 1000, price)
            else:
                for path in paths.values():
                    path.consume_untouched_bar(unit)
    return dict(id=row['id'], symbol=row['symbol'], direction=row['direction'],
        model_status=row['status'], model_r=row['realized_r'],
        opened_at=opened, model_closed_at=closed, model_tp1_observed_at=row.get('tp1_hit_at'),
        scenarios={name: path.finish() for name, path in paths.items()},
        refined_minutes=refined_minutes, refined_trade_seconds=refined_seconds,
        zero_volume_minutes_ignored=zero_minutes, source_hashes=hashes)


def main():
    source = ROOT / 'crypto_observed_be_signals_nov16.json'
    manifest = ROOT / 'crypto_15m/manifest.json'
    exported = json.loads(source.read_text(encoding='utf-8'))
    meta = json.loads(manifest.read_text(encoding='utf-8'))
    folder = ROOT / 'observed_be_paths_nov16'
    folder.mkdir(exist_ok=True)
    output = []
    for row in sorted(exported['signals'], key=lambda r: r['id']):
        print('begin', row['id'], row['symbol'], flush=True)
        try:
            item = reconstruct(row, meta['symbols'][row['symbol']]['instrument'], folder)
            print('done', row['id'], {mode: item['scenarios'][mode]['kind'] for mode in MODES}, flush=True)
        except Exception as exc:
            item = dict(id=row['id'], symbol=row['symbol'], model_status=row['status'],
                        model_r=row['realized_r'], error_type=type(exc).__name__, error=str(exc))
            print('unresolved', row['id'], type(exc).__name__, str(exc), flush=True)
        output.append(item)
    complete = [r for r in output if 'scenarios' in r]
    summary = dict(signals=len(output), complete=len(complete),
        unresolved=[r['id'] for r in output if 'scenarios' not in r],
        all_saved_model_r=sum(float(r['model_r']) for r in output), scenarios={})
    for mode in MODES:
        known = [r for r in complete if r['scenarios'][mode]['whole_position_r'] is not None]
        summary['scenarios'][mode] = dict(
            known=len(known), matched_saved_model_r=sum(float(r['model_r']) for r in known),
            public_price_whole_position_r=sum(r['scenarios'][mode]['whole_position_r'] for r in known),
            public_price_five_percent_r=sum(r['scenarios'][mode]['nominal_five_percent_r'] for r in known),
            bracket_events=sum(r['scenarios'][mode]['kind'] not in ('MODEL_CLOSE_MARK', 'NO_POST_PUBLICATION_TRADE') for r in known),
            model_close_marks=sum(r['scenarios'][mode]['kind']=='MODEL_CLOSE_MARK' for r in known))
    stopped = [r['scenarios']['observed_tp1']['max_favorable_r_before_exit']
               for r in complete if r['model_status'] == 'SL_HIT']
    summary['saved_sl_excursions'] = dict(count=len(stopped),
        median_favorable_r=median(stopped) if stopped else None,
        below_quarter_r=sum(v < .25 for v in stopped),
        below_half_r=sum(v < .5 for v in stopped),
        diagnostic_only_not_an_entry_feature=True)
    report = dict(status='OBSERVED_TP1_PUBLIC_PATH_DIAGNOSTIC_NOT_ACCOUNT_PNL',
        objective_verified=False, live_change_approved=False, summary=summary, rows=output,
        limitations=[
            'Published signal prices are not individual account entry fills or breakeven levels.',
            'Observed TP1 is a model DB timestamp before a private amendment request, not its acceptance time.',
            'Observed scenarios condition on a successful amendment after zero/60 seconds; acceptance and private fills are unverified.',
            'Public last-trade crossings are trigger-price proxies, not executable bid/ask market fills.',
            'All paths stop at recorded model closure, before the subsequent account market-close call.',
            'No post-publication trade means unknown, not invented flat PnL; terminal marks are estimates.',
            'First/last seconds are clipped at millisecond trade timestamps; submillisecond ordering remains unknown.',
            'Zero and 5% partial exits are two price-path scenarios, not client-size recommendations.',
            'Fixed published sample; alternate exits can change future admission, which is not replayed here.'],
        sources={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (
            source, manifest, Path(__file__), Path('observed_be_path_nov16.py'),
            Path('public_trade_second.py'), Path('public_candle_window.py'), Path('src/autotrader.py'))})
    (ROOT / 'crypto_observed_be_nov16.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
