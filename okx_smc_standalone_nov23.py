"""Compare the order-block sequence as an independent entry on both markets."""

from bisect import bisect_left
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from statistics import median

from account_cash_guard_nov17 import stamp
from liquidity_trap_joint_nov19_audit import attach_outcome, load_frame, verified_cash, with_exact_tick, SNAPSHOT
from okx_unbankable_tp1_nov21 import read_json
from okx_xperp_multiday_replay_nov20 import file_hash
from phase_entry_scenarios_nov19 import HORIZON, stock_clock
from phase_scenarios_nov19_audit import (CRYPTO_MANIFEST, ROOT, SPEC_PATH, STOCK_MANIFEST, STOCK_ROOT,
                                        frame_of, portfolio)
from price_grid_bracket_nov19 import rounded_bracket
from smc_sequence_nov23 import sequences
from stock_liquid_quote_routes_nov21 import write_json


BAR = 900
BEGIN = int(stamp('2026-08-01'))
HOLDOUT = int(stamp('2026-10-01'))
FAMILIES = ('sweep', 'displacement', 'block_retest')


def regime(closes, sign):
    move = closes[-1] - closes[0]
    travelled = sum(abs(b-a) for a, b in zip(closes, closes[1:]))
    if abs(move) <= .30 * travelled:
        return 'range'
    return 'aligned' if sign * move > 0 else 'opposing'


def discover(frame, symbol, spec, *, stocks, begin, end):
    times = list(frame['time'])
    if any(t % BAR for t in times) or any(b <= a for a, b in zip(times, times[1:])):
        raise ValueError('Unordered native candles')
    bars = list(zip(*(frame[field] for field in ('open', 'high', 'low', 'close', 'volume')), strict=True))
    tick = float(spec['tickSz'])
    open_rows = {}
    counts = Counter()
    for s in range(max(27, bisect_left(times, begin)-8), bisect_left(times, end)):
        first = s-27
        if times[s] - times[first] != 27*BAR:
            continue
        newest = min(s+7, len(times)-1)
        while times[newest]-times[s] != (newest-s)*BAR:
            newest -= 1
        for sign, side in ((1, 'LONG'), (-1, 'SHORT')):
            edge = 2 if sign > 0 else 1
            if sign*bars[s][edge] > min(sign*b[edge] for b in bars[s-3:s]):
                continue
            events = sequences(bars[first:newest+1], side, tick)
            for event in events:
                if event.sweep != s-first:
                    continue
                stages = [*zip(FAMILIES, (event.sweep, event.displacement, event.retest), strict=True),
                          ('displacement_local_stop', event.displacement)]
                for stage, at in stages:
                    if at is None:
                        continue
                    j = first + at
                    epoch = int(times[j]) + BAR
                    expire = stock_clock(epoch) if stocks else None
                    if not begin <= epoch < end or stocks and expire is None:
                        continue
                    if sum(bar[4] > 0 for bar in bars[j-15:j+1]) < 12:
                        continue
                    counts[stage] += 1
                    entry = bars[j][3] * (1 + sign * .0005)
                    typical = median(row[1]-row[2] for row in bars[s-20:s])
                    stop_anchor = j if stage == 'displacement_local_stop' else s
                    stop = bars[stop_anchor][edge] - sign * .15 * typical
                    risk = max(.003, sign*(entry-stop)/entry)
                    if risk > .01:
                        counts[f'{stage}_wide_stop'] += 1
                        continue
                    label = regime([row[3] for row in bars[j-23:j+1]], sign)
                    bracket = rounded_bracket(dict(symbol=symbol, direction=side, entry=entry,
                        sl=entry*(1-sign*risk), tp1=entry*(1+sign*.005),
                        tp2=entry*(1+sign*max(.01, 2*risk)), risk_cap=.01, entry_time=epoch,
                        entry_index=j+1, expire_at=expire, family=stage, equal_pivots=event.equal_pivots,
                        signal_strength=(bars[s][4]/median(row[4] for row in bars[s-20:s])),
                        phase=label, sweep_close=int(times[s])+BAR), spec)
                    if bracket is None:
                        counts[f'{stage}_price_grid'] += 1
                        continue
                    key = stage, side, epoch
                    existing = open_rows.get(key)
                    if existing is None or existing['signal_strength'] < bracket['signal_strength']:
                        open_rows[key] = bracket
    return list(open_rows.values()), dict(counts)


def outcome(row, path, *, traded):
    at = bisect_left(path['time'], row['entry_time'])
    if at >= len(path['time']) or path['time'][at] != row['entry_time']:
        return dict(row, known=False, unknown_reason='missing_entry_candle',
                    exit_time=row['entry_time']+HORIZON*BAR)
    return attach_outcome(dict(row, entry_index=at), path, traded)


def main():
    instruments = read_json(SNAPSHOT)['instruments']
    specs = read_json(SPEC_PATH)
    series = {}
    sources = {}
    for symbol, source in read_json(CRYPTO_MANIFEST)['symbols'].items():
        path = Path(source['path'])
        specs[symbol] = with_exact_tick(specs[symbol], instruments)
        series[symbol] = (load_frame(path, source['sha256']), None)
        sources[str(path)] = source['sha256']
    for symbol, source in read_json(STOCK_MANIFEST)['symbols'].items():
        candles = []
        specs[symbol] = with_exact_tick(source['spec'], instruments)
        for field, digest in (('trade_path', 'trade_sha256'), ('path', 'sha256')):
            path = STOCK_ROOT / source[field]
            candles.append(load_frame(path, source[digest]))
            sources[str(path)] = source[digest]
        series[symbol] = tuple(candles)
    end = min(int(frame['time'][-1]) for frames in series.values() for frame in frames if frame is not None)
    end -= HORIZON*BAR
    rows = {'crypto': [], 'stock_trade': [], 'stock_mark': []}
    diagnostics = {}
    for symbol, (trade, mark) in sorted(series.items()):
        found, diagnostics[symbol] = discover(trade, symbol, specs[symbol], stocks=mark is not None,
                                               begin=BEGIN, end=end)
        name = 'crypto' if mark is None else 'stock_trade'
        rows[name].extend(outcome(row, trade, traded=True) for row in found)
        if mark is not None:
            rows['stock_mark'].extend(outcome(row, mark, traded=False) for row in found)
        print(symbol, name, Counter(row['family'] for row in found), flush=True)

    outcomes = {}
    for cohort, start, finish in (('aug_sep', BEGIN, HOLDOUT), ('oct_nov', HOLDOUT, end)):
        outcomes[cohort] = {}
        for route, family, equal, phases in (
            ('reclaim', 'sweep', False, {'aligned', 'opposing', 'range'}),
            ('equal_liquidity', 'sweep', True, {'aligned', 'opposing', 'range'}),
            ('structure_break', 'displacement', False, {'aligned', 'opposing', 'range'}),
            ('first_block_retest', 'block_retest', False, {'aligned', 'opposing', 'range'}),
            ('structure_range', 'displacement', False, {'range'}),
            ('structure_aligned', 'displacement', False, {'aligned'}),
            ('structure_local_stop', 'displacement_local_stop', False, {'aligned', 'opposing', 'range'}),
            ('structure_local_stop_range', 'displacement_local_stop', False, {'range'}),
            ('structure_local_stop_aligned', 'displacement_local_stop', False, {'aligned'}),
        ):
            results = {}
            phases_seen = {}
            accepted = {}
            for name, all_rows in rows.items():
                selected = [row for row in all_rows if row['family'] == family and row['phase'] in phases
                            and (not equal or row['equal_pivots']) and start <= row['entry_time']
                            and row['entry_time']+HORIZON*BAR < finish]
                phases_seen[name] = dict(Counter(row['phase'] for row in selected))
                bot = 'crypto' if name == 'crypto' else 'stocks'
                accepted[name] = portfolio(frame_of(selected), bot)
            for name in ('stock_trade', 'stock_mark'):
                crypto, stocks = accepted['crypto'], accepted[name]
                unknown = int((~crypto.known.astype(bool)).sum()+(~stocks.known.astype(bool)).sum())
                if unknown:
                    results[name] = dict(unknown_paths=unknown)
                    continue
                results[name] = verified_cash(crypto, stocks, specs, policy='requested5',
                                              priority='crypto', start=start)
            outcomes[cohort][route] = dict(cash=results, pre_gate_regimes=phases_seen)
            print(cohort, route, {k: v.get('summary', {}).get('return_contribution_pct', v)
                                  for k, v in results.items()}, flush=True)
    source_files = (Path(__file__), Path('smc_sequence_nov23.py'), STOCK_MANIFEST, CRYPTO_MANIFEST, SNAPSHOT, SPEC_PATH)
    write_json(ROOT / 'okx_smc_standalone_nov23.json', dict(
        entry_window_utc=[datetime.fromtimestamp(v, timezone.utc).isoformat() for v in (BEGIN, HOLDOUT, end)],
        outcomes=outcomes, diagnostics=diagnostics, candle_sha256=sources,
        source_sha256={str(p): file_hash(p) for p in source_files}, live_activation_supported=False, limitations=[
            'Standalone hypotheses, with stock entries restricted to weekdays 10:00 to 15:30 New York time.',
            'Historical candle simulation with market price stress and actual contract increments, without historic books.',
            'Both stock exit triggers are measured. Stock signal readiness to enter real money is not assumed by this comparison.',
            'The local-stop sensitivity puts the stop beyond the displacement candle instead of beyond the sweep wick.',
            'No route parameters fitted on October or November; the same route definitions run in both time windows.',
        ]))


if __name__ == '__main__':
    main()
