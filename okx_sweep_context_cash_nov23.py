"""Test confirmed liquidity sweeps and nearby swing levels as entry context."""

from bisect import bisect_left
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import pickle
from statistics import median

from okx_delayed_trailing_cash_nov23 import stock_sources
from okx_selective_single_target_nov23 import EARLIER, JOINT, cash_summary, key
from okx_spot_confirmation_nov22 import load_native
from okx_unbankable_tp1_nov21 import old_inputs, recent_inputs
from okx_xperp_cash_tick_nov19 import CAPITAL, CLOSE_LAGS, FRACTION, LEVERAGE, simulate
from okx_xperp_multiday_replay_nov20 import ROOT, file_hash
from public_candle_window import fetch_window


HISTORY = 36
BAR = 900
POLICIES = ('baseline', 'veto_opposite_sweep', 'veto_close_swing', 'veto_either',
            'veto_swing_without_progress', 'veto_sweep_at_near_swing', 'require_aligned_sweep')
PREFIX = 'okx_sweep_context_cash_nov23'


def window(frame, epoch):
    first = bisect_left(frame['time'], epoch-HISTORY*BAR)
    end = first+HISTORY
    if list(frame['time'][first:end]) != list(range(epoch-HISTORY*BAR, epoch, BAR)):
        return None
    volume_field = 'volume_contracts' if 'volume_contracts' in frame else 'volume'
    return list(zip(*(frame[k][first:end] for k in ('open','high','low','close',volume_field)), strict=True))


def market_context(bars, *, direction, entry, stop, tick):
    if bars is None:
        return None
    sign = 1 if direction == 'LONG' else -1
    highs = [row[1] for row in bars]
    lows = [row[2] for row in bars]
    vols = [row[4] for row in bars]
    sweeps = []
    for i in range(len(bars)-3, len(bars)):
        width = highs[i]-lows[i]
        typical = median(highs[j]-lows[j] for j in range(i-20,i))
        volume = median(vols[i-20:i])
        if typical <= 0 or width <= 0 or volume <= 0 or vols[i] < 1.25*volume:
            continue
        close = bars[i][3]
        lower = min(lows[i-12:i])
        upper = max(highs[i-12:i])
        pierce = max(tick, .1*typical)
        reclaim = max(tick, .05*typical)
        bull = lower-lows[i] >= pierce and close-lower >= reclaim and close-lows[i] >= .6*width
        bear = highs[i]-upper >= pierce and upper-close >= reclaim and highs[i]-close >= .6*width
        if bull or bear:
            sweeps.append(1 if bull else -1)
    # A two-right-bar swing becomes observable only after both bars close.
    points = highs if sign > 0 else [-price for price in lows]
    edge = None
    for i in range(2, len(points)-2):
        candidate = points[i]
        if vols[i] <= 0 or candidate <= sign*entry:
            continue
        if candidate <= max(points[i-2:i]) or candidate <= max(points[i+1:]):
            continue
        edge = candidate if edge is None else min(edge, candidate)
    risk = abs(entry-stop)
    distance = (edge-sign*entry)/risk if edge is not None and risk > 0 else None
    return dict(sweep='aligned' if sweeps and sweeps[-1] == sign else 'opposed' if sweeps else 'none',
                untested_swing_room_r=distance, last_three_bars_progress=sign*(bars[-1][3]-bars[-4][3]))


def refuses(name, context):
    if context is None:
        return name == 'require_aligned_sweep'
    opposite = context['sweep'] == 'opposed'
    room = context['untested_swing_room_r']
    blocked = room is not None and room < .75
    return {'baseline': False, 'veto_opposite_sweep': opposite, 'veto_close_swing': blocked,
            'veto_either': opposite or blocked, 'veto_sweep_at_near_swing': opposite and blocked,
            'veto_swing_without_progress': blocked and context['last_three_bars_progress'] <= 0,
            'require_aligned_sweep': context['sweep'] != 'aligned'}[name]


def contexts(rows, specs, references):
    frames = load_native(references)
    contexts = {}
    downloads = {}
    for row in rows:
        if row['entry_status'] != 'eligible' or row['bot'] == 'stocks' and not row['stock_reference_ready']:
            continue
        symbol, epoch = row['symbol'], row['epoch']
        frame = frames.get(symbol)
        bars = window(frame, epoch) if frame is not None else None
        if bars is None:
            instrument = specs[symbol]['instId']
            identity = instrument,epoch
            if identity not in downloads:
                try:
                    data,digest = fetch_window(instrument, epoch-HISTORY*BAR, epoch, BAR, ROOT/PREFIX/'candles')
                    downloads[identity] = (window(data, epoch), digest)
                except (ValueError, OSError) as exc:
                    downloads[identity] = (None, type(exc).__name__)
            bars = downloads[identity][0]
        contexts[key(row)] = market_context(bars, direction=row['direction'], entry=row['reference_entry'],
                                            stop=row['stop'], tick=float(specs[symbol].get('tickSz') or 0))
    return contexts, {f'{s}:{at}': digest for (s,at),(_,digest) in downloads.items()}


def evaluate(material, cohorts, cap, controls, *, joint=False):
    rows, specs, ticks, books, _, trades, references = material
    if joint:
        _,references = stock_sources(trades, references)
    features, extra = contexts(rows, specs, references)
    cases = {}
    for name,keep in cohorts.items():
        cohort = [r for r in rows if keep(r)]
        cases[name] = {}
        for lag in CLOSE_LAGS:
            variants = {}
            for label in POLICIES:
                marked = []
                rejected = Counter()
                rejected_keys = set()
                for row in cohort:
                    identity = key(row)
                    if identity not in features or not refuses(label, features[identity]):
                        marked.append(row)
                    else:
                        rejected[row['bot']] += 1
                        rejected_keys.add(identity)
                        marked.append(dict(row, entry_status='sweep_context_filter'))
                result = simulate(marked, specs, ticks, books, clock=60, lag=lag,
                                  setting='requested5', max_positions=cap)
                summary = cash_summary(result, {})
                if label == 'baseline':
                    baseline_trades = result['closed']
                    expected = controls[name][str(lag)]
                    for field in ('trades','return_pct','drawdown_pct','contribution_pct'):
                        if summary[field] != expected[field]:
                            raise ValueError(f'published baseline disagrees in {name}: {field}')
                removed = [trade for trade in baseline_trades if key(trade) in rejected_keys]
                pnl = [r['bank_pnl']+r['runner_pnl'] for r in removed]
                variants[label] = dict(summary, filtered=rejected, filtered_baseline_trades=dict(
                    pnl_usd=sum(pnl), profitable=sum(v>0 for v in pnl), exits=Counter(r['code'] for r in removed)))
            cases[name][str(lag)] = variants
            print(name,lag,{k:(v.get('trades'),round(v['cash_pnl_usd'],3)) if 'cash_pnl_usd' in v
                           else v.get('reason') for k,v in variants.items()},flush=True)
    coverage = Counter(f'{identity[0]}:{ctx["sweep"] if ctx else "unknown"}' for identity,ctx in features.items())
    return cases, dict(coverage=coverage, candle_sha256=extra,
                       reference_sha256={symbol: item['sha256'] for symbol,item in references.items()})


def long_crypto():
    from crypto_active_entry_runner_nov19_audit import activity_gate
    from crypto_minimum_bank_runner_nov19_audit import admitted_records, cash_result, checked_row
    from crypto_native_price_pool_nov19 import MANIFEST, load_pool

    candles = load_native(json.loads(MANIFEST.read_bytes())['symbols'])
    spec_path = ROOT/'shared_xperp_specs_nov14.json'
    specs = json.loads(spec_path.read_bytes())
    pool, metadata = load_pool()
    rows = []
    cutoff = datetime(2026,11,1,tzinfo=timezone.utc).timestamp()
    for row in pool[pool.is_frozen_candidate & (pool.entry_time < cutoff)].to_dict('records'):
        c = candles[row['symbol']]
        i = bisect_left(c['time'], int(row['entry_time']))
        if i >= len(c['time']) or c['time'][i] != int(row['entry_time']):
            raise ValueError('native candle missing for the established baseline')
        if not activity_gate(c,i):
            continue
        ctx = market_context(window(c,int(row['entry_time'])), direction=row['direction'], entry=row['entry'],
                             stop=row['sl'], tick=float(specs[row['symbol']].get('tickSz') or 0))
        rows.append((ctx, checked_row(row,c,i,'be_tp1',.0005)))
    source = ROOT/'crypto_nearer_target_long_history_nov22.json'
    expected = json.loads(source.read_bytes())['cases']['control_2r']['clients']['requested5']
    outcomes = {}
    for policy in POLICIES:
        selected = [row for ctx,row in rows if not refuses(policy,ctx)]
        admitted, unknown = admitted_records(selected)
        if unknown:
            outcomes[policy] = dict(status='UNSCORABLE', admitted_unknown=unknown, candidate_count=len(selected))
            continue
        score = cash_result(admitted,specs,'requested5')
        if policy == 'baseline' and score != expected:
            raise ValueError('published pre-November candle baseline changed')
        outcomes[policy] = dict(status='SCORED_CANDLE_PROXY', cash=score, candidate_count=len(selected))
        print('jul_oct', policy, score['summary']['trades'], score['summary']['return_pct'],
              score['summary']['cash_drawdown_pct'], flush=True)
    return dict(cases=outcomes, source_sha256=file_hash(source), specs_sha256=file_hash(spec_path),
                candle_pool_sha256=metadata['sha256'], manifest_sha256=file_hash(MANIFEST),
                context_count=sum(ctx is not None for ctx,_ in rows))


def holdout_inputs():
    from crypto_scan_priority_holdout_nov22 import DATA, load_cohort
    from crypto_scan_priority_nov22 import book_cadence
    from okx_spot_confirmation_holdout_nov22 import execution_samples

    capture = DATA/'crypto_capture.json'
    captured,proposals,_ = load_cohort(capture)
    rows,books,_ = execution_samples(captured,proposals,file_hash(capture))
    selection = list(book_cadence(rows,'alphabetical'))
    if any(row['entry_status'] == 'unknown_quote' for row in selection):
        raise ValueError('holdout contains unknown entry quotes')
    specs = captured['specs']
    ticks = {symbol: float(spec['tickSz']) for symbol,spec in specs.items()}
    return selection,specs,ticks,books,{},captured['trades'],captured['references']


def stock_paper_rows():
    import stock_native_geometry_match_nov22 as observer

    plan = observer.freeze()
    _, _, ticks, _, tapes, notes, _ = observer.prepare_cases(plan)
    folder,_ = observer.ARCHIVES['contract_followup']
    observations = {}
    rows = []
    for row in notes:
        path = row.get('paper_path')
        if not path or not path['known']:
            continue
        epoch, symbol = row['epoch'],row['symbol']
        if epoch not in observations:
            source = folder/f'bucket_{epoch}.json'
            if file_hash(source) != plan['archive_sha256'][source.name]:
                raise ValueError('native observer archive changed')
            observations[epoch] = {r['symbol']: r['closed_frame'] for r in json.loads(source.read_bytes())['observations']}
        ctx = market_context(window(observations[epoch][symbol],epoch),direction=row['direction'],
                             entry=row['ticker_reference'],stop=row['fixed_stop'],tick=ticks[symbol])
        rows.append((row,ctx))
    return rows,ticks,tapes,file_hash(observer.DATA/'plan.json')


def stock_paper():
    rows,_,_,digest = stock_paper_rows()
    outcomes = {label: Counter() for label in POLICIES}
    features = Counter()
    for row,ctx in rows:
        features[ctx['sweep'] if ctx else 'unknown'] += 1
        for label in POLICIES:
            if not refuses(label,ctx):
                outcomes[label][row['paper_path']['code']] += 1
    print('native stock at ticker', {k:dict(v) for k,v in outcomes.items()},flush=True)
    return dict(paper_counts=outcomes, context=features, plan_sha256=digest,
                limitation='Native paper outcomes at the observed ticker; executable cash is in the separate book study.')


def main():
    native = ROOT/'okx_tp1_cash_sensitivity_nov22.json'
    latest = ROOT/'crypto_scan_priority_holdout_nov22.json'
    controls = {window:{lag: values['requested5'] for lag,values in item['baseline'].items()}
                for window,item in json.loads(native.read_bytes())['cases'].items()}
    holdout = 'nov20_four_hours'
    controls[holdout] = json.loads(latest.read_bytes())['results']['alphabetical']
    historical = long_crypto()
    early, early_sources = evaluate(old_inputs(), EARLIER,12,controls)
    joint, joint_evidence = evaluate(recent_inputs(), JOINT,6,controls,joint=True)
    fresh, fresh_evidence = evaluate(holdout_inputs(), {holdout: lambda _: True},12,controls)
    paper = stock_paper()
    report = dict(capital_usd=CAPITAL, margin_fraction=FRACTION, leverage=LEVERAGE, policies=POLICIES,
                  historical_crypto_native_candle_proxy=historical, exact_cash_cases=early|joint|fresh,
                  native_stock_observer_paper=paper,
                  source_sha256={p.name:file_hash(p) for p in (native,latest)},
                  source_checks=dict(early=early_sources,joint=joint_evidence,holdout=fresh_evidence),
                  live_activation_supported=False, limitations=[
                      'A sweep is a measured high-volume breach and reclaim of prior range, with no inference about trader identity.',
                      'An untested swing must already be confirmed by two later closed bars and remain untouched through the decision.',
                      'Portfolio replay keeps the original fixed brackets and entry order. A filtered proposal does not retroactively replace the original publication rank.',
                      'The long crypto sample uses native candles plus stressed market exits. Cash from the exact November samples uses public depth.',
                      'The two conditional swing filters were added after the first standalone level check, so comparisons are exploratory.',
                      'Stock continuation in the joint cohort is experimental. The separate November 20 holdout has only a four-hour crypto entry window.',
                      'Cash drawdown uses closed and settled positions in these samples, not floating account equity.',
                  ])
    (ROOT/f'{PREFIX}.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8',newline='\n')


if __name__ == '__main__':
    main()
