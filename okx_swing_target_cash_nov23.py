"""Price market exits near confirmed range levels on the original entries."""

from bisect import bisect_left
from collections import Counter
from datetime import datetime, timezone
import json

from crypto_active_entry_runner_nov19_audit import activity_gate
from crypto_minimum_bank_runner_nov19_audit import admitted_records, cash_result, checked_row
from crypto_native_price_pool_nov19 import MANIFEST, load_pool
from crypto_nearer_target_long_history_nov22 import EXIT_STRESS, shorten
from okx_delayed_trailing_cash_nov23 import paired_dollars, stock_sources
from okx_recent_trade_repair_nov21 import load_repaired_tape
from okx_selective_single_target_nov23 import EARLIER, JOINT, cash_summary, key
from okx_spot_confirmation_nov22 import load_native
from okx_sweep_context_cash_nov23 import contexts, holdout_inputs, market_context, stock_paper_rows, window
from okx_unbankable_tp1_nov21 import old_inputs, recent_inputs
from okx_xperp_cash_tick_nov19 import CAPITAL, CLOSE_LAGS, FRACTION, LEVERAGE, simulate
from okx_xperp_multiday_replay_nov20 import ROOT, exit_books, file_hash, resolve_all
from okx_xperp_tick_path_nov19 import resolve_ticks
from src.okx_trader import round_to_tick


MIN_MOVE = .005
MAX_ROOM_R = 1.5
CONTROL_MULTIPLE = 1.25
POLICIES = ('baseline', 'swing_target', 'fixed_target_same_signals',
            'swing_when_price_not_advancing', 'fixed_target_same_slow_signals')
PREFIX = 'okx_swing_target_cash_nov23'


def new_targets(row, context, tick):
    if context is None or tick <= 0:
        return {}
    room = context['untested_swing_room_r']
    if room is None or not 0 < room <= MAX_ROOM_R:
        return {}
    side = 1 if row['direction'] == 'LONG' else -1
    entry, quote, stop, target = (row[k] for k in ('entry','reference_entry','stop','target'))
    level = quote + side * room * abs(quote-stop)
    swing = round_to_tick(round_to_tick(level,tick) - side*tick,tick)
    fixed = round_to_tick(entry + CONTROL_MULTIPLE * (entry-stop),tick)
    # Preserve the desk's original TP1 and require actual room from the market quote.
    tp1 = round_to_tick(entry + .5*(entry-stop),tick) if row['bot'] == 'crypto' else quote
    for price in (swing,fixed):
        if side*(price-quote) < MIN_MOVE*quote or side*(price-tp1) < .5*tick or side*(target-price) < .5*tick:
            return {}
    results = dict(swing_target=swing, fixed_target_same_signals=fixed)
    if context['last_three_bars_progress'] <= 0:
        results.update(swing_when_price_not_advancing=swing, fixed_target_same_slow_signals=fixed)
    return results


def prepare_variants(rows, specs, ticks, trades, references):
    features, extra = contexts(rows, specs, references)
    variants = {name: {} for name in POLICIES}
    unique = {}
    original = []
    for row in rows:
        identity = key(row)
        if identity not in features:
            continue
        alternatives = new_targets(row,features[identity],ticks[row['symbol']])
        if not alternatives:
            continue
        original.append(row)
        for name, price in alternatives.items():
            variants[name][identity] = unique.setdefault((identity,price),dict(row,target=price))
    trials = list(unique.values())
    parity = [dict(row) for row in original]
    checks = resolve_all(trials+parity, {'files': trades}, {'symbols': references}, ticks,
                         trade_loader=load_repaired_tape)
    for before,after in zip(original,parity,strict=True):
        if before['path'][60] != after['path'][60]:
            raise ValueError(f'control tape replay does not match {key(before)}')
    return variants,trials,dict(tape=checks, downloaded_candles=extra,
                                 reference_sha256={k:v['sha256'] for k,v in references.items()})


def admission_cash_delta(baseline_usd, trial_usd, paired):
    shared = sum(row['cash_delta_usd'] for row in paired['transition'].values())
    return round(trial_usd-baseline_usd-shared, 9)


def evaluate(material, cohorts, cap, controls, *, joint=False, capacity_overrides=None):
    rows, specs, ticks, books, archive, trades, references = material
    if joint:
        trades,references=stock_sources(trades,references)
    variants,trials,evidence = prepare_variants(rows,specs,ticks,trades,references)
    print(f'{len(trials)} alternative market paths checked',flush=True)
    added,book_checks = exit_books(trials, {'books':archive}, specs, books)
    evidence.update(exit_books=book_checks, archive_sha256={k:v['sha256'] for k,v in added.items()})
    values = {}
    for name,keep in cohorts.items():
        original = [r for r in rows if keep(r)]
        values[name] = {}
        for lag in CLOSE_LAGS:
            parameters = dict(clock=60,lag=lag,setting='requested5',
                              max_positions=(capacity_overrides or {}).get(name,cap))
            baseline = simulate(original,specs,ticks,books,**parameters)
            for field in ('trades','return_pct','drawdown_pct','contribution_pct'):
                if baseline[field] != controls[name][str(lag)][field]:
                    raise ValueError(f'published L2 cash baseline changed: {name}, {field}')
            reports = {'baseline':cash_summary(baseline,{})}
            for policy, candidate in variants.items():
                if policy == 'baseline':
                    continue
                changed = [candidate.get(key(row),row) for row in original]
                replay = simulate(changed,specs,ticks,books,**parameters)
                summary = cash_summary(replay,candidate)
                if 'closed' in replay:
                    paired = paired_dollars(baseline,replay)
                    paired['cash_from_added_minus_removed_usd'] = admission_cash_delta(
                        reports['baseline']['cash_pnl_usd'],summary['cash_pnl_usd'],paired)
                    summary['comparison'] = paired
                reports[policy]=summary
            values[name][str(lag)] = reports
            print(name,lag,{k:(round(v['cash_pnl_usd'],3),v['routed'],v['trades']) if 'cash_pnl_usd' in v else v.get('reason')
                           for k,v in reports.items()},flush=True)
    return values,evidence


def historical_brackets():
    candles = load_native(json.loads(MANIFEST.read_bytes())['symbols'])
    spec_path = ROOT/'shared_xperp_specs_nov14.json'
    specs = json.loads(spec_path.read_bytes())
    universe = ROOT/'native_universe_nov19_snapshot.json'
    ticks_by_id = {item['instId']: float(item['tickSz']) for item in
                   json.loads(universe.read_bytes())['instruments'] if item.get('tickSz')}
    pool, metadata = load_pool()
    cutoff = datetime(2026,11,1,tzinfo=timezone.utc).timestamp()
    cases = {name: [] for name in POLICIES}
    assigned, changed = Counter(),Counter()
    for row in pool[pool.is_frozen_candidate & (pool.entry_time < cutoff)].to_dict('records'):
        symbol,stamp = row['symbol'],int(row['entry_time'])
        c = candles[symbol]
        i = bisect_left(c['time'],stamp)
        if i >= len(c['time']) or c['time'][i] != stamp:
            raise ValueError('native candle missing for the historical control')
        if not activity_gate(c,i):
            continue
        original = checked_row(row,c,i,'be_tp1',EXIT_STRESS)
        entry, stop = row['entry'],row['sl']
        tick = ticks_by_id[specs[symbol]['instId']]
        ctx = market_context(window(c,stamp),direction=row['direction'],entry=entry,stop=stop,tick=tick)
        targets = new_targets(dict(row,bot='crypto',reference_entry=entry,stop=stop,
                                   target=entry+2*(entry-stop)),ctx,tick)
        for policy in POLICIES:
            candidate = original
            if policy in targets:
                assigned[policy] += 1
                candidate = shorten(original,c,i,target_px=targets[policy])
                changed[policy] += candidate is not original
            cases[policy].append(candidate)
    return cases,specs,spec_path,universe,metadata,assigned,changed


def historical_crypto():
    cases,specs,spec_path,universe,metadata,assigned,changed = historical_brackets()
    reference = ROOT/'crypto_nearer_target_long_history_nov22.json'
    baseline = json.loads(reference.read_bytes())['cases']['control_2r']['clients']['requested5']
    results = {}
    for policy,rows in cases.items():
        admitted,unknown=admitted_records(rows)
        if unknown:
            results[policy]=dict(status='UNSCORABLE', admitted_unknown=unknown)
            continue
        result = cash_result(admitted,specs,'requested5')
        if policy == 'baseline' and result != baseline:
            raise ValueError('published candle cash baseline changed')
        results[policy]=dict(status='SCORED_CANDLE_PROXY', cash=result,
                             assigned=assigned[policy], changed_paths=changed[policy])
        print('jul_oct',policy,round(result['summary']['return_pct'],3),
              round(result['summary']['cash_drawdown_pct'],3),assigned[policy],changed[policy],flush=True)
    return dict(cases=results, source_sha256={p.name:file_hash(p) for p in (reference,spec_path,MANIFEST,universe)},
                verified_cache_sha256=metadata['sha256'])


def native_stock_paper():
    stock,ticks,tapes,plan_digest = stock_paper_rows()
    cases = {name: dict(assigned=0,outcomes=Counter(),closed_before_baseline=0,modified=[]) for name in POLICIES}
    for row,ctx in stock:
        symbol, entry = row['symbol'],row['ticker_reference']
        tick = ticks[symbol]
        original = row['paper_path']
        modeled = dict(row,bot='stocks',entry=entry,reference_entry=entry,stop=row['fixed_stop'],target=row['fixed_target'])
        alternatives = new_targets(modeled,ctx,tick)
        times,tape=tapes[symbol]
        for policy,case in cases.items():
            path = original
            if policy in alternatives:
                case['assigned'] += 1
                moved = dict(modeled,target=alternatives[policy])
                path = resolve_ticks(tape,times,signal=moved,entered_at=row['entry_ms'],entry_px=entry,
                                     tick_size=tick,amend_seconds=60)
            case['outcomes'][path.get('code','UNSCORABLE')] += 1
            if path['known'] and path['end_ms'] < original['end_ms']:
                case['closed_before_baseline'] += 1
                case['modified'].append(dict(symbol=symbol, epoch=row['epoch'],
                    observed_entry_gate=row['status'], baseline=original['code'], trial=path['code']))
    print('native_stock_paper',cases,flush=True)
    return dict(plan_sha256=plan_digest,cases=cases,
                limitation='Uses verified trades for ordered triggers but only paper observations at the ticker.')


def configured_capacity_control():
    source = ROOT/'okx_joint_exact_exit_nov21.json'
    archive = json.loads(source.read_bytes())['scenarios']
    window = 'nov17_19_joint_cap12'
    controls = {window:{str(lag):archive[f'both_base_cadence|cap12|requested5|{lag}ms']['book'] for lag in CLOSE_LAGS}}
    return window,controls,source


def main():
    source = ROOT/'okx_sweep_context_cash_nov23.json'
    baseline = json.loads(source.read_bytes())['exact_cash_cases']
    controls = {window:{lag:values['baseline'] for lag,values in lags.items()} for window,lags in baseline.items()}
    holdout = 'nov20_four_hours'
    cap12,extra_controls,cap12_source = configured_capacity_control()
    controls.update(extra_controls)
    earlier,early_sources = evaluate(old_inputs(),EARLIER,12,controls)
    joint,joint_evidence = evaluate(recent_inputs(),JOINT|{cap12:lambda _:True},6,controls,
                                   joint=True,capacity_overrides={cap12:12})
    fresh,fresh_evidence = evaluate(holdout_inputs(),{holdout:lambda _:True},12,controls)
    history = historical_crypto()
    report = dict(starting_capital_usd=CAPITAL, margin=FRACTION, leverage=LEVERAGE,
                  rule=dict(minimum_move=MIN_MOVE, maximum_swing_room_r=MAX_ROOM_R,
                            fixed_target_r=CONTROL_MULTIPLE, offset_ticks=1),
                  monthly_candle_proxy=history, cash_cases=earlier|joint|fresh,
                  source_sha256={p.name:file_hash(p) for p in (source,cap12_source)},
                  native_stock_observer_paper=native_stock_paper(),
                  archive_checks=dict(early=early_sources,joint=joint_evidence,holdout=fresh_evidence),
                  live_activation_supported=False, limitations=[
                      'The market exit is recalculated in the actual book after the trade trigger; a level touch is not treated as a fill.',
                      'Only the final target changes. Entry, planned crypto TP1, original stops and timing rules are shared.',
                      'The fixed 1.25R target uses the exact same proposals as the swing policy.',
                      'The shared stock route is an experimental continuation portfolio, separate from native production.',
                      'The long crypto sample uses conservative candle collision handling and proxy market exit stress.',
                      'Previously examined periods and short independent quote windows do not establish monthly performance.',
                  ])
    (ROOT/f'{PREFIX}.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8',newline='\n')


if __name__=='__main__':
    main()
